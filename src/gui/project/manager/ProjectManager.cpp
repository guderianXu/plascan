#include "ProjectManager.h"
#include "ProjectReconstructionManager.h"
#include "ProjectTerrainProductsManager.h"
#include "ProjectCameraSetupManager.h"
#include "ProjectTaskDispatcher.h"
#include "ProjectUiCommands.h"
#include "ProjectData.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "ProjectCameraImportService.h"
#include "ProjectBundleAdjustExecution.h"
#include "ProjectBundleAdjustWorkflow.h"
#include "ProjectCameraInitialization.h"
#include "ProjectDenseWorkflowConfig.h"
#include "ProjectResourceCleanupService.h"
#include "ProjectTiePointResultService.h"

#include "DenseMatchRunner.h"
#include "ProjectMetadataOperations.h"
#include "ProjectSfmWorkflow.h"
#include "ProjectSparseWorkflow.h"
#include "ProjectResultRecords.h"
#include "ProjectReferenceDatasets.h"
#include "ProjectSurveyControl.h"
#include "ProjectWorkflowUtils.h"
#include "ProjectWorkflowReports.h"
#include "GenerateMaskDialog.h"
#include "SurveyControlDialog.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "MaskGenerator.h"
#include "Sam21MaskGenerator.h"
#include "u2net/U2NetMaskGenerator.h"
#include "io/PathIO.h"
#include "model/TorchScriptModelResolver.h"
#include "model/U2NetModelCatalog.h"
#include "filtering/SparsePointCloudProcessor.h"
#include "FileDialogStateManager.h"
#include "Camera.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "SparseCloudPreprocessor.h"
#include "DenseCloudBuilder.h"
#include "Intersection.h"
#include "BundleAdjust.h"
#include "SparseCloudValidator.h"
#include "SurfaceReconstructor.h"

#include "AerialTriangulationService.h"

#include <QMessageBox>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressDialog>
#include <QThread>
#include <cmath>
#include <QFile>
#include <QSet>
#include <QFileInfo>
#include <QTextStream>
#include <QDateTime>
#include <QImage>
#include <QImageReader>
#include <QPainter>
#include <QPen>
#include <QRegularExpression>
#include <QColor>

#include <algorithm>
#include <atomic>
#include <array>
#include <limits>
#include <memory>
#include <optional>

#include "OpenCvCompat.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using xjw::gui::project::cameraFromJson;
using xjw::gui::project::cameraToJson;
using xjw::gui::project::BundleAdjustExecutionResult;
using xjw::gui::project::buildDepthGenConfig;
using xjw::gui::project::buildSparsePointWorkflowSuccessMessage;
using xjw::gui::project::commitBundleAdjustPreview;
using xjw::gui::project::existingCameraImages;
using xjw::gui::project::denseGenerationSettingsFromJson;
using xjw::gui::project::denseRefineSettingsFromJson;
using xjw::gui::project::finalizeBundleAdjustArtifacts;
using xjw::gui::project::finalizeInitializedCameraPoses;
using xjw::gui::project::focalPixelsFromExif;
using xjw::gui::project::InitPoseFinalizeResult;
using xjw::gui::project::makeAtResultRecord;
using xjw::gui::project::makeDenseResultRecord;
using xjw::gui::project::makeDepthResultRecord;
using xjw::gui::project::makeInitializedCameraMeta;
using xjw::gui::project::makeModelResultRecord;
using xjw::gui::project::normalizePath;
using xjw::gui::project::pathTokenMatchesImage;
using xjw::gui::project::persistProjectMeta;
using xjw::gui::project::projectFilesMeta;
using xjw::gui::project::resolveInitTargets;
using xjw::gui::project::resolveLatestDenseCloudPath;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::gui::project::resolveSparsePointContext;
using xjw::gui::project::runBundleAdjustExecution;
using xjw::gui::project::runSparsePointWorkflow;
using xjw::gui::project::replaceMetaArrayWithLatest;
using xjw::gui::project::replaceProjectRecordWithLatest;
using xjw::gui::project::SparsePointContext;
using xjw::gui::project::SparsePointOperationResult;
using xjw::gui::project::SparsePointWorkflowKind;
using xjw::gui::project::SparsePointWorkflowSpec;
using xjw::gui::project::sparseOperationDisplayName;
using xjw::gui::project::sparsePointWorkflowSpec;
using xjw::gui::project::summarizeAtResults;
using xjw::gui::project::findLatestAtResultIndex;
using xjw::gui::project::upsertProjectRecordByPath;
using xjw::gui::project::upsertMetaArrayRecordByIndex;
using xjw::gui::project::upsertMetaArrayRecordByPath;
using xjw::gui::project::withPreparedCameras;
using xjw::gui::project::writeJsonObjectFile;
using xjw::gui::project::writeBundleAdjustReport;

namespace
{

struct ImageFolderScan
{
    bool success = false;
    QString folder;
    QString errorMessage;
    QStringList imagePaths;
};

QStringList imageFolderNameFilters()
{
    return QStringList{
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff"),
        QStringLiteral("*.TIF"),
        QStringLiteral("*.TIFF"),
        QStringLiteral("*.png"),
        QStringLiteral("*.PNG"),
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.JPG"),
        QStringLiteral("*.JPEG")
    };
}

ImageFolderScan scanImageFolder(const QString &folder)
{
    ImageFolderScan scan;
    scan.folder = folder;

    QDir dir(folder);
    if (!dir.exists())
    {
        scan.errorMessage = QStringLiteral("文件夹不存在: %1").arg(folder);
        return scan;
    }

    const QFileInfoList files = dir.entryInfoList(imageFolderNameFilters(),
                                                  QDir::Files | QDir::Readable,
                                                  QDir::Name | QDir::IgnoreCase);
    scan.imagePaths.reserve(files.size());
    for (const QFileInfo &file : files)
    {
        scan.imagePaths.append(file.absoluteFilePath());
    }

    scan.success = true;
    return scan;
}

QString imageLabel(const QString &path)
{
    const QFileInfo fi(path);
    return fi.fileName().isEmpty() ? path : fi.fileName();
}

bool isBaPriorRole(const QString &role)
{
    const QString normalized = role.toLower();
    return normalized == QLatin1String("ba_prior")
        || normalized == QLatin1String("bundle_adjustment")
        || normalized == QLatin1String("reference_prior");
}

QString firstReferenceDemPriorPath(const QJsonObject &meta)
{
    const QJsonArray references = meta.value(QStringLiteral("reference_datasets")).toArray();
    for (const QJsonValue &value : references)
    {
        const QJsonObject reference = value.toObject();
        if (!isBaPriorRole(reference.value(QStringLiteral("role")).toString()))
        {
            continue;
        }

        const QString type = reference.value(QStringLiteral("type")).toString().toLower();
        const QString path = reference.value(QStringLiteral("path")).toString().trimmed();
        if ((type == QLatin1String("dem") || type == QLatin1String("reference_dem")) && !path.isEmpty())
        {
            return path;
        }
    }
    return QString();
}

QString firstReferenceLaserPriorPath(const QJsonObject &meta)
{
    const QJsonArray references = meta.value(QStringLiteral("reference_datasets")).toArray();
    for (const QJsonValue &value : references)
    {
        const QJsonObject reference = value.toObject();
        if (!isBaPriorRole(reference.value(QStringLiteral("role")).toString()))
        {
            continue;
        }

        const QString type = reference.value(QStringLiteral("type")).toString().toLower();
        const QString path = reference.value(QStringLiteral("path")).toString().trimmed();
        if (path.isEmpty())
        {
            continue;
        }

        const QString suffix = QFileInfo(path).suffix().toLower();
        const bool laserType = type == QLatin1String("lidar")
            || type == QLatin1String("reference_lidar")
            || type == QLatin1String("point_cloud");
        if (laserType && suffix == QLatin1String("ply"))
        {
            return path;
        }
    }
    return QString();
}

void appendMetaArrayRecord(QJsonObject *meta,
                           const QString &arrayKey,
                           const QJsonObject &record)
{
    // 通用追加：读取数组 -> append -> 写回，避免各处重复样板。
    if (!meta) return;
    QJsonArray arr = meta->value(arrayKey).toArray();
    arr.append(record);
    (*meta)[arrayKey] = arr;
}

xjw::mask::MaskGenerationOptions maskOptionsFromSettings(const QJsonObject &settings)
{
    xjw::mask::MaskGenerationOptions options;
    const QString method = settings.value(QStringLiteral("method")).toString(QStringLiteral("black_background"));
    options.method = method == QLatin1String("threshold")
        ? xjw::mask::MaskGenerationMethod::Threshold
        : xjw::mask::MaskGenerationMethod::BlackBackground;
    options.threshold = settings.value(QStringLiteral("auto_threshold")).toBool(true)
        ? -1.0
        : settings.value(QStringLiteral("threshold")).toDouble(3.0);
    options.morphologyRadius = settings.value(QStringLiteral("morphology_radius")).toInt(2);
    options.minComponentArea = settings.value(QStringLiteral("min_component_area")).toInt(64);
    options.keepLargestComponent = true;
    return options;
}

xjw::mask::MaskOperation maskOperationFromSettings(const QJsonObject &settings)
{
    const QString operation = settings.value(QStringLiteral("operation")).toString(QStringLiteral("replace"));
    if (operation == QLatin1String("union"))
    {
        return xjw::mask::MaskOperation::Union;
    }
    if (operation == QLatin1String("intersection"))
    {
        return xjw::mask::MaskOperation::Intersection;
    }
    if (operation == QLatin1String("difference"))
    {
        return xjw::mask::MaskOperation::Difference;
    }
    return xjw::mask::MaskOperation::Replace;
}

QStringList maskTargetsFromSettings(const QJsonObject &settings, const QStringList &allImages)
{
    const QString scope = settings.value(QStringLiteral("scope")).toString(QStringLiteral("selected_images"));
    if (scope == QLatin1String("all_images"))
    {
        return allImages;
    }

    QStringList selectedImages;
    const QJsonArray selected = settings.value(QStringLiteral("selected_images")).toArray();
    for (const QJsonValue &value : selected)
    {
        const QString path = value.toString();
        if (!path.trimmed().isEmpty())
        {
            selectedImages.push_back(path);
        }
    }

    if (scope == QLatin1String("current_image"))
    {
        const QString currentImage = settings.value(QStringLiteral("current_image")).toString();
        return currentImage.trimmed().isEmpty() ? QStringList{} : QStringList{currentImage};
    }
    return selectedImages.isEmpty() ? allImages : selectedImages;
}

std::string utf8StdString(const QString &value)
{
    const QByteArray bytes = QDir::toNativeSeparators(value).toUtf8();
    return std::string(bytes.constData(), static_cast<std::size_t>(bytes.size()));
}

QString plascanSourceRoot()
{
#ifdef PLASCAN_SOURCE_DIR
    return QDir::cleanPath(QStringLiteral(PLASCAN_SOURCE_DIR));
#else
    return QDir::cleanPath(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("../..")));
#endif
}

QString resolveSam21InstallerScript()
{
    const QString sourceScript = QDir(plascanSourceRoot()).filePath(QStringLiteral("scripts/install_sam21_model.py"));
    if (QFileInfo::exists(sourceScript))
    {
        return QDir::cleanPath(sourceScript);
    }

    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList candidates{
        QDir(appDir).filePath(QStringLiteral("scripts/install_sam21_model.py")),
        QDir(appDir).filePath(QStringLiteral("../scripts/install_sam21_model.py")),
        QDir(appDir).filePath(QStringLiteral("../../scripts/install_sam21_model.py")),
    };
    for (const QString &candidate : candidates)
    {
        if (QFileInfo::exists(candidate))
        {
            return QDir::cleanPath(candidate);
        }
    }
    return QDir::cleanPath(sourceScript);
}

QString resolvePythonExecutable()
{
    QString pythonExecutable = qEnvironmentVariable("PLASCAN_PYTHON_EXECUTABLE").trimmed();
    if (pythonExecutable.isEmpty())
    {
        pythonExecutable = qEnvironmentVariable("PLASCAN_PYTHON").trimmed();
    }
    if (!pythonExecutable.isEmpty())
    {
        return QDir::cleanPath(pythonExecutable);
    }

    const QString sourceRoot = plascanSourceRoot();
#ifdef Q_OS_WIN
    const QString projectVenv = QDir(sourceRoot).filePath(QStringLiteral(".venv/Scripts/python.exe"));
    const QString bundled = QDir(sourceRoot).filePath(QStringLiteral("build/env/python-runtime/Scripts/python.exe"));
#else
    const QString projectVenv = QDir(sourceRoot).filePath(QStringLiteral(".venv/bin/python"));
    const QString bundled = QDir(sourceRoot).filePath(QStringLiteral("build/env/python-runtime/bin/python"));
#endif
    if (QFileInfo::exists(projectVenv))
    {
        return QDir::cleanPath(projectVenv);
    }
    if (QFileInfo::exists(bundled))
    {
        return QDir::cleanPath(bundled);
    }
    return QStringLiteral("python");
}

std::optional<xjw::mask::Sam21MaskGeneratorConfig> sam21MaskConfigFromSettings(const QJsonObject &settings,
                                                                               QString *error)
{
    const QString variantToken = settings.value(QStringLiteral("sam21_variant")).toString(QStringLiteral("tiny"));
    const xjw::mask::Sam21ModelVariant variant =
        xjw::mask::sam21VariantFromToken(variantToken.toStdString());
    const bool requestCuda =
        settings.value(QStringLiteral("sam21_device")).toString(QStringLiteral("cuda")) == QLatin1String("cuda");
    const bool allowFallback = settings.value(QStringLiteral("sam21_allow_fallback")).toBool(true);

    const auto requestedNames = xjw::mask::sam21TorchScriptModelNames(variant, requestCuda);
    const auto cpuNames = xjw::mask::sam21TorchScriptModelNames(variant, false);
    const xjw::common::model::TorchScriptModelResolver resolver;
    const QString requestedEncoder = resolver.findModel(QString::fromStdString(requestedNames.encoder));
    const QString requestedDecoder = resolver.findModel(QString::fromStdString(requestedNames.decoder));
    const QString cpuEncoder = resolver.findModel(QString::fromStdString(cpuNames.encoder));
    const QString cpuDecoder = resolver.findModel(QString::fromStdString(cpuNames.decoder));

    bool useCuda = requestCuda;
    QString encoder = requestedEncoder;
    QString decoder = requestedDecoder;
    if (encoder.isEmpty() || decoder.isEmpty())
    {
        if (requestCuda && allowFallback && !cpuEncoder.isEmpty() && !cpuDecoder.isEmpty())
        {
            useCuda = false;
            encoder = cpuEncoder;
            decoder = cpuDecoder;
        }
        else
        {
            if (error)
            {
                const QString device = requestCuda ? QStringLiteral("CUDA") : QStringLiteral("CPU");
                *error = QStringLiteral("未找到 SAM2.1 %1 TorchScript 模型：%2, %3。请放到 PLASCAN_MODEL_DIR 或 resources/models。")
                    .arg(device,
                         QString::fromStdString(requestedNames.encoder),
                         QString::fromStdString(requestedNames.decoder));
            }
            return std::nullopt;
        }
    }

    xjw::mask::Sam21MaskGeneratorConfig config;
    config.encoderModelPath = utf8StdString(encoder);
    config.decoderModelPath = utf8StdString(decoder);
    config.cpuEncoderModelPath = utf8StdString(cpuEncoder);
    config.cpuDecoderModelPath = utf8StdString(cpuDecoder);
    config.useCuda = useCuda;
    config.cudaDevice = std::max(0, settings.value(QStringLiteral("sam21_cuda_device")).toInt(0));
    config.allowDeviceFallback = allowFallback;
    config.inputSize = std::clamp(settings.value(QStringLiteral("sam21_input_size")).toInt(1024), 256, 2048);
    config.maskThreshold = settings.value(QStringLiteral("sam21_mask_threshold")).toDouble(0.0);
    config.multimaskOutput = true;
    return config;
}

std::optional<xjw::mask::U2NetMaskGeneratorConfig> u2netMaskConfigFromSettings(const QJsonObject &settings,
                                                                               QString *error)
{
    const xjw::common::model::TorchScriptModelResolver resolver;
    const auto status = xjw::common::model::u2netModelStatus(resolver);
    if (!status.isInstalled)
    {
        if (error)
        {
            *error = QStringLiteral("未找到 U2Net ONNX 模型：U2Net_v1.onnx。请放到 PLASCAN_MODEL_DIR 或 resources/models。");
        }
        return std::nullopt;
    }

    xjw::mask::U2NetMaskGeneratorConfig config;
    config.modelPath = utf8StdString(status.modelPath);
    config.useCuda =
        settings.value(QStringLiteral("u2net_device")).toString(QStringLiteral("cuda")) == QLatin1String("cuda");
    config.allowDeviceFallback = settings.value(QStringLiteral("u2net_allow_fallback")).toBool(true);
    config.cudaDevice = std::max(0, settings.value(QStringLiteral("sam21_cuda_device")).toInt(0));
    config.inputSize = std::clamp(settings.value(QStringLiteral("u2net_input_size")).toInt(320), 128, 1024);
    config.foregroundThreshold =
        static_cast<float>(std::clamp(settings.value(QStringLiteral("u2net_mask_threshold")).toDouble(0.5),
                                      0.01,
                                      0.99));
    config.morphologyRadius = 1;
    config.minComponentArea = 64;
    config.keepLargestComponent = true;
    return config;
}

} // namespace

ProjectManager::ProjectManager(ProjectData *projectData, QWidget *parent)
    : QObject(parent)
    , _parent(parent)
    , _projectData(projectData)
    , _reconstructionManager(new ProjectReconstructionManager(this, projectData, parent, this))
    , _terrainProductsManager(new ProjectTerrainProductsManager(this, projectData, parent, this))
    , _cameraSetupManager(new ProjectCameraSetupManager(this, projectData, parent, this))
    , _taskDispatcher(new ProjectTaskDispatcher(_cameraSetupManager,
                                                 _terrainProductsManager,
                                                 _reconstructionManager,
                                                 this))
    , _uiCommands(new ProjectUiCommands(projectData, parent))
{
    _uiCommands->setDirectoryAccessors(
        [this](const QString &key) { return getLastUsedDir(key); },
        [this](const QString &key, const QString &dir) { saveLastUsedDir(key, dir); });

    // 连接ProjectData信号
    if (_projectData)
    {
        connect(_projectData, &ProjectData::projectOpened,
                this, &ProjectManager::projectOpened);
        connect(_projectData, &ProjectData::projectSaved,
                this, &ProjectManager::projectSaved);
        connect(_projectData, &ProjectData::projectClosed,
                this, &ProjectManager::projectClosed);
        connect(_projectData, &ProjectData::metadataChanged,
                this,
                [this](const QJsonObject &meta)
                {
                    emit projectMetadataChanged(
                        xjw::gui::project::ProjectTiePointResultService::metadataWithCurrentOnly(
                            meta,
                            currentProjectPath()));
                });
        connect(_projectData, &ProjectData::dirtyStateChanged,
                this, &ProjectManager::metadataDirtyChanged);
    }

        connect(_reconstructionManager, &ProjectReconstructionManager::mvsProgressChanged,
            this, &ProjectManager::mvsProgressChanged);
        connect(_reconstructionManager, &ProjectReconstructionManager::mvsProgressFinished,
            this, &ProjectManager::mvsProgressFinished);
        connect(_reconstructionManager, &ProjectReconstructionManager::denseCloudResultReady,
            this, &ProjectManager::denseCloudResultReady);
        connect(_reconstructionManager, &ProjectReconstructionManager::meshProgressChanged,
            this, &ProjectManager::meshProgressChanged);
        connect(_reconstructionManager, &ProjectReconstructionManager::meshProgressFinished,
            this, &ProjectManager::meshProgressFinished);
        connect(_reconstructionManager, &ProjectReconstructionManager::atProgressChanged,
            this, &ProjectManager::atProgressChanged);
        connect(_reconstructionManager, &ProjectReconstructionManager::atProgressFinished,
            this, &ProjectManager::atProgressFinished);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::demPipelineProgressChanged,
            this, &ProjectManager::demPipelineProgressChanged);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::demPipelineFinished,
            this, &ProjectManager::demPipelineFinished);

    LOG_INFO(QStringLiteral("ProjectManager 已初始化(精简版)"));
}

ProjectManager::~ProjectManager()
{
}

//==============================================================================
// 项目操作
//==============================================================================

void ProjectManager::createNewProject()
{
    QString plascanPath;
    if (_uiCommands && _uiCommands->createNewProject(&plascanPath))
    {
        emit projectCreated(plascanPath);
        LOG_INFO(QStringLiteral("项目已创建: %1").arg(plascanPath));
    }
}

void ProjectManager::openProject()
{
    QString plascanPath;
    if (_uiCommands && _uiCommands->selectProjectByDialog(&plascanPath))
    {
        openProjectFromPath(plascanPath);
    }
}

void ProjectManager::openProjectFromPath(const QString &plascanPath)
{
    if (plascanPath.trimmed().isEmpty())
    {
        return;
    }
    const QString projectPath = QDir::cleanPath(QFileInfo(plascanPath).absoluteFilePath());
    if (_projectOpenInProgress)
    {
        showWarning(QStringLiteral("正在打开项目，请稍候。"));
        return;
    }

    _projectOpenInProgress = true;
    emit projectOpenStarted(projectPath);
    emit projectOpenProgressChanged(QStringLiteral("正在读取项目文件..."), 10);

    xjw::gui::tasks::runGuarded(
        this,
        [projectPath]() -> ProjectOpenSnapshot
        {
            return ProjectData::loadProjectOpenSnapshot(projectPath);
        },
        [projectPath](ProjectManager *self, ProjectOpenSnapshot snapshot)
        {
            emit self->projectOpenProgressChanged(QStringLiteral("正在初始化项目界面..."), 75);

            auto finishWithError = [self](const QString &message)
            {
                self->_projectOpenInProgress = false;
                emit self->projectOpenFinished(false, message);
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("打开项目失败: %1").arg(message));
            };

            if (!snapshot.success)
            {
                finishWithError(snapshot.errorMessage.isEmpty()
                                    ? QStringLiteral("读取项目文件失败")
                                    : snapshot.errorMessage);
                return;
            }

            QString error;
            if (!self->_projectData || !self->_projectData->openProjectFromSnapshot(snapshot, &error))
            {
                finishWithError(error.isEmpty() ? QStringLiteral("应用项目数据失败") : error);
                return;
            }

            emit self->projectOpenProgressChanged(QStringLiteral("正在启动结果数据后台加载..."), 95);
            if (!snapshot.resultsLoaded)
            {
                self->loadProjectResultsAsync(projectPath);
            }

            self->_projectOpenInProgress = false;
            emit self->projectOpenFinished(true, QStringLiteral("项目已打开"));
            LOG_INFO(QStringLiteral("项目已打开: %1").arg(projectPath));
        });
}

void ProjectManager::loadProjectResultsAsync(const QString &plascanPath)
{
    if (plascanPath.trimmed().isEmpty())
    {
        return;
    }

    xjw::gui::tasks::runGuarded(
        this,
        [plascanPath]() -> ProjectResultsSnapshot
        {
            return ProjectData::loadProjectResultsSnapshot(plascanPath);
        },
        [plascanPath](ProjectManager *self, ProjectResultsSnapshot snapshot)
        {
            const QString currentProjectPath = self->_projectData
                ? QDir::cleanPath(self->_projectData->currentProjectPath())
                : QString();
            if (!self->_projectData || currentProjectPath != QDir::cleanPath(plascanPath))
            {
                return;
            }

            QString error;
            if (!self->_projectData->applyResultsSnapshot(snapshot, &error))
            {
                LOG_WARN(QStringLiteral("项目结果数据后台加载失败: %1").arg(error));
                return;
            }

            if (snapshot.hasResults)
            {
                LOG_INFO(QStringLiteral("项目结果数据已后台加载: %1").arg(plascanPath));
            }
        });
}

void ProjectManager::saveProject()
{
    if (!_projectData)
    {
        return;
    }

    emit saveStarted();
    const bool success = _uiCommands && _uiCommands->saveProject();

    emit saveFinished(success);
}

void ProjectManager::closeProject()
{
    if (_uiCommands)
    {
        _uiCommands->closeProject();
    }
}

//==============================================================================
// 资源管理
//==============================================================================

void ProjectManager::addPhoto()
{
    if (_uiCommands)
    {
        (void)_uiCommands->addPhoto();
    }
}

void ProjectManager::addFolder()
{
    if (!_uiCommands || !_projectData)
    {
        return;
    }

    QString folder;
    if (!_uiCommands->selectImageFolder(&folder))
    {
        return;
    }

    const QString projectPath = currentProjectPath();
    xjw::gui::tasks::runGuarded(
        this,
        [folder]() -> ImageFolderScan
        {
            return scanImageFolder(folder);
        },
        [folder, projectPath](ProjectManager *self, ImageFolderScan scan)
        {
            if (!self || !self->_projectData || self->currentProjectPath() != projectPath)
            {
                return;
            }

            if (!scan.success)
            {
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("添加文件夹失败: %1").arg(scan.errorMessage));
                return;
            }

            if (scan.imagePaths.isEmpty())
            {
                QMessageBox::information(self->_parent,
                                         QStringLiteral("提示"),
                                         QStringLiteral("文件夹中没有找到可导入的影像: %1").arg(folder));
                return;
            }

            QString error;
            if (!self->_projectData->addImages(scan.imagePaths, &error))
            {
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("添加文件夹失败: %1").arg(error));
                return;
            }

            LOG_INFO(QStringLiteral("已从文件夹添加 %1 张影像: %2")
                         .arg(scan.imagePaths.size())
                         .arg(folder));
            if (!error.isEmpty())
            {
                QMessageBox::information(self->_parent, QStringLiteral("提示"), error);
            }
        });
}

bool ProjectManager::importCameraForImage(const QString &imagePath)
{
    return _taskDispatcher->importCameraForImage(imagePath);
}

void ProjectManager::startTriangulationAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::Triangulation, settings);
}

void ProjectManager::startDenseMatchAsync(const QJsonObject &settings)
{
    startDenseMatchAsyncWithProgress(settings, nullptr);
}

void ProjectManager::startDenseMatchAsyncWithProgress(
    const QJsonObject &settings,
    std::shared_ptr<std::atomic<int>> progress,
    std::shared_ptr<std::atomic<bool>> cancelFlag)
{
    DenseMatchRunner::run(settings, progress, cancelFlag);
}

void ProjectManager::startSparseCloudOutlierRemovalAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::SparseOutlierRemoval, settings);
}

void ProjectManager::startSparseCloudLocalOptimAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::SparseLocalOptimization, settings);
}

void ProjectManager::startSparseCloudRefineAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::SparseRefine, settings);
}

bool ProjectManager::importCamerasByFilenameBatch()
{
    return _taskDispatcher->importCamerasByFilenameBatch();
}

bool ProjectManager::initializeCamerasFromExifOrDefault(const QJsonObject &settings)
{
    return _taskDispatcher->initializeCamerasFromExifOrDefault(settings);
}

bool ProjectManager::initializeCamerasFromIntrinsics(const QJsonObject &settings)
{
    return _taskDispatcher->initializeCamerasFromIntrinsics(settings);
}

bool ProjectManager::initializeCameraPosesWithSFM(const QJsonObject &settings)
{
    return _taskDispatcher->initializeCameraPosesWithSFM(settings);
}

void ProjectManager::removeResource(const QString &resourcePath)
{
    if (_projectData)
    {
        _projectData->removeResource(resourcePath);
    }
}

void ProjectManager::removeResources(const QStringList &resourcePaths)
{
    if (_projectData)
    {
        _projectData->removeResources(resourcePaths);
    }
}

void ProjectManager::importReferenceDataset()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再导入参考 DEM/LiDAR。")))
    {
        return;
    }

    const QString selected = QFileDialog::getOpenFileName(
        _parent,
        QStringLiteral("导入参考 DEM/LiDAR"),
        getLastUsedDir(QStringLiteral("reference_dataset")),
        QStringLiteral("参考地形/点云 (*.tif *.tiff *.vrt *.las *.laz *.copc *.ply *.xyz *.csv);;"
                       "DEM (*.tif *.tiff *.vrt);;"
                       "LiDAR (*.las *.laz *.copc);;"
                       "点云 (*.ply *.xyz *.csv);;"
                       "所有文件 (*)"));
    if (selected.isEmpty())
    {
        return;
    }

    saveLastUsedDir(QStringLiteral("reference_dataset"), QFileInfo(selected).absolutePath());

    QString error;
    if (!registerReferenceDataset(selected, QString(), QStringLiteral("validation"), &error))
    {
        showWarning(error.isEmpty() ? QStringLiteral("导入参考数据失败。") : error,
                    QStringLiteral("导入参考 DEM/LiDAR"));
        return;
    }

    LOG_INFO(QStringLiteral("参考数据已导入: %1").arg(QFileInfo(selected).fileName()));
}

bool ProjectManager::registerReferenceDataset(const QString &path,
                                              const QString &type,
                                              const QString &role,
                                              QString *errorMsg)
{
    return xjw::gui::project::registerReferenceDataset(_projectData, path, type, role, errorMsg);
}

void ProjectManager::openSurveyControlDialog()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再管理测绘控制点。")))
    {
        return;
    }

    SurveyControlDialog dialog(_parent);
    dialog.setSurveyControlMetadata(_projectData->metadata().value(QStringLiteral("survey_control")).toObject());

    connect(&dialog, &SurveyControlDialog::importCsvRequested, this, [this, &dialog]()
    {
        const QString selected = QFileDialog::getOpenFileName(
            &dialog,
            QStringLiteral("导入测绘控制 CSV"),
            getLastUsedDir(QStringLiteral("survey_control")),
            QStringLiteral("控制点 CSV (*.csv);;所有文件 (*)"));
        if (selected.isEmpty())
        {
            return;
        }

        saveLastUsedDir(QStringLiteral("survey_control"), QFileInfo(selected).absolutePath());
        const auto result = xjw::gui::project::importSurveyControlCsv(_projectData,
                                                                      selected,
                                                                      QString());
        if (!result.imported)
        {
            showWarning(result.errorMessage.isEmpty()
                            ? QStringLiteral("导入测绘控制 CSV 失败。")
                            : result.errorMessage,
                        QStringLiteral("测绘控制"));
            return;
        }

        dialog.setSurveyControlMetadata(_projectData->metadata().value(QStringLiteral("survey_control")).toObject());
        dialog.setStatusMessage(QStringLiteral("已导入：控制点 %1，检查点 %2，比例尺 %3")
                                    .arg(result.controlPointCount)
                                    .arg(result.checkPointCount)
                                    .arg(result.scaleBarCount));
        LOG_INFO(QStringLiteral("测绘控制 CSV 已导入: %1 controls=%2 checks=%3 scale_bars=%4")
                 .arg(QFileInfo(selected).fileName())
                 .arg(result.controlPointCount)
                 .arg(result.checkPointCount)
                 .arg(result.scaleBarCount));
    });

    dialog.exec();
}

void ProjectManager::setActiveImagePath(const QString &imagePath)
{
    _activeImagePath = imagePath;
}

void ProjectManager::installSam21Model(const QString &variantToken, GenerateMaskDialog *dialog)
{
    const QString cleanVariant = variantToken.trimmed().isEmpty() ? QStringLiteral("tiny") : variantToken.trimmed();
    const QString pythonExecutable = resolvePythonExecutable();
    const QString installerScript = resolveSam21InstallerScript();
    const QString sourceRoot = plascanSourceRoot();

    xjw::common::model::TorchScriptModelResolver resolver;
    const QString modelDir = resolver.defaultModelDir();
    if (modelDir.isEmpty() || !QDir().mkpath(modelDir))
    {
        showWarning(QStringLiteral("无法创建 SAM2.1 模型目录：%1").arg(modelDir), QStringLiteral("安装 SAM2.1 模型"));
        return;
    }

    if (!QFileInfo::exists(installerScript))
    {
        showWarning(QStringLiteral("未找到 SAM2.1 安装脚本：%1").arg(installerScript),
                    QStringLiteral("安装 SAM2.1 模型"));
        return;
    }

    auto *progressDialog = new QProgressDialog(QStringLiteral("正在安装 SAM2.1 模型..."),
                                               QStringLiteral("取消"),
                                               0,
                                               100,
                                               _parent);
    progressDialog->setWindowTitle(QStringLiteral("安装 SAM2.1 模型"));
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setMinimumDuration(0);
    progressDialog->setAutoClose(false);
    progressDialog->setAutoReset(false);
    progressDialog->setValue(0);
    progressDialog->show();

    auto *process = new QProcess(progressDialog);
    process->setProcessChannelMode(QProcess::MergedChannels);

    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("PLASCAN_MODEL_DIR"), modelDir);
    environment.insert(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE"), pythonExecutable);
    environment.insert(QStringLiteral("PLASCAN_PYTHON"), pythonExecutable);
    environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    process->setProcessEnvironment(environment);

    QStringList arguments;
    arguments << installerScript
              << QStringLiteral("--variant") << cleanVariant
              << QStringLiteral("--source-dir") << sourceRoot
              << QStringLiteral("--model-dir") << modelDir
              << QStringLiteral("--python-executable") << pythonExecutable
              << QStringLiteral("--devices") << QStringLiteral("auto");

    auto outputBuffer = std::make_shared<QString>();
    QPointer<QProgressDialog> progressGuard(progressDialog);
    QPointer<GenerateMaskDialog> dialogGuard(dialog);

    connect(progressDialog, &QProgressDialog::canceled,
            process,
            [process, progressGuard]()
            {
                if (progressGuard)
                {
                    progressGuard->setLabelText(QStringLiteral("正在取消 SAM2.1 模型安装..."));
                }
                process->kill();
            });

    connect(process, &QProcess::readyReadStandardOutput,
            process,
            [process, outputBuffer, progressGuard]()
            {
                const QString text = QString::fromUtf8(process->readAllStandardOutput());
                outputBuffer->append(text);
                const QStringList lines = text.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
                static const QRegularExpression downloadPercentPattern(QStringLiteral("PROGRESS download (\\d+)%"));
                for (const QString &rawLine : lines)
                {
                    const QString line = rawLine.trimmed();
                    if (!progressGuard || line.isEmpty())
                    {
                        continue;
                    }

                    const QRegularExpressionMatch match = downloadPercentPattern.match(line);
                    if (match.hasMatch())
                    {
                        progressGuard->setRange(0, 100);
                        progressGuard->setValue(std::clamp(match.captured(1).toInt(), 0, 100));
                        progressGuard->setLabelText(QStringLiteral("正在下载 SAM2.1 checkpoint..."));
                    }
                    else if (line.contains(QStringLiteral("PROGRESS export started")))
                    {
                        progressGuard->setRange(0, 0);
                        progressGuard->setLabelText(QStringLiteral("正在导出 TorchScript 模型..."));
                    }
                    else if (line.startsWith(QStringLiteral("PROGRESS")))
                    {
                        progressGuard->setLabelText(line.mid(QStringLiteral("PROGRESS").size()).trimmed());
                    }
                    else if (line.startsWith(QStringLiteral("EXPORT")))
                    {
                        progressGuard->setLabelText(line.mid(QStringLiteral("EXPORT").size()).trimmed());
                    }
                }
            });

    connect(process, &QProcess::errorOccurred,
            process,
            [this, progressGuard](QProcess::ProcessError error)
            {
                if (error == QProcess::FailedToStart)
                {
                    if (progressGuard)
                    {
                        progressGuard->close();
                        progressGuard->deleteLater();
                    }
                    showWarning(QStringLiteral("无法启动 Python。请检查 PLASCAN_PYTHON_EXECUTABLE、PLASCAN_PYTHON 或项目 .venv。"),
                                QStringLiteral("安装 SAM2.1 模型"));
                }
            });

    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            process,
            [this, process, outputBuffer, progressGuard, dialogGuard](int exitCode, QProcess::ExitStatus exitStatus)
            {
                const bool success = exitStatus == QProcess::NormalExit && exitCode == 0;
                if (progressGuard)
                {
                    progressGuard->setRange(0, 100);
                    progressGuard->setValue(success ? 100 : 0);
                    progressGuard->close();
                    progressGuard->deleteLater();
                }

                if (success)
                {
                    if (dialogGuard)
                    {
                        dialogGuard->refreshSam21ModelStatus();
                    }
                    QMessageBox::information(_parent,
                                             QStringLiteral("安装 SAM2.1 模型"),
                                             QStringLiteral("SAM2.1 模型安装完成。"));
                }
                else
                {
                    QString output = outputBuffer ? outputBuffer->trimmed() : QString();
                    if (output.size() > 2000)
                    {
                        output = output.right(2000);
                    }
                    showWarning(QStringLiteral("SAM2.1 模型安装失败。退出码：%1\n%2").arg(exitCode).arg(output),
                                QStringLiteral("安装 SAM2.1 模型"));
                }
                process->deleteLater();
            });

    process->start(pythonExecutable, arguments);
}

void ProjectManager::openGenerateMaskDialog()
{
    const QStringList allImages = _projectData ? _projectData->getAllImages() : QStringList();
    openGenerateMaskDialogForImages(allImages);
}

void ProjectManager::openGenerateMaskDialogForImages(const QStringList &requestedImages)
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再生成照片蒙版。")))
    {
        return;
    }
    if (_maskGenerationCancelFlag)
    {
        showWarning(QStringLiteral("已有照片蒙版生成任务正在运行，请等待完成或取消后再试。"),
                    QStringLiteral("生成蒙版"));
        return;
    }

    const QStringList allImages = _projectData ? _projectData->getAllImages() : QStringList();
    if (allImages.isEmpty())
    {
        showWarning(QStringLiteral("当前项目没有可生成蒙版的照片。"), QStringLiteral("生成蒙版"));
        return;
    }

    const QString projectPath = currentProjectPath();
    QHash<QString, QString> projectImages;
    QStringList resolvedAllImages;
    QSet<QString> projectImageKeys;
    for (const QString &imagePath : allImages)
    {
        const QString resolvedPath = ProjectIO::resolveProjectResourcePath(projectPath, imagePath);
        const QString key = normalizePath(resolvedPath);
        if (key.isEmpty() || projectImageKeys.contains(key))
        {
            continue;
        }
        projectImageKeys.insert(key);
        projectImages.insert(key, resolvedPath);
        resolvedAllImages.push_back(resolvedPath);
    }

    QStringList selectedImages;
    QSet<QString> seen;
    for (const QString &requestedPath : requestedImages)
    {
        const QString resolvedPath = ProjectIO::resolveProjectResourcePath(projectPath, requestedPath);
        const QString key = normalizePath(resolvedPath);
        if (!key.isEmpty() && projectImages.contains(key) && !seen.contains(key))
        {
            seen.insert(key);
            selectedImages.push_back(projectImages.value(key));
        }
    }
    if (selectedImages.isEmpty())
    {
        showWarning(QStringLiteral("没有选中可生成蒙版的照片。"), QStringLiteral("生成蒙版"));
        return;
    }

    const QString activeImage = ProjectIO::resolveProjectResourcePath(projectPath, _activeImagePath);
    const QString currentImage = projectImages.value(normalizePath(activeImage));
    GenerateMaskDialog dialog(selectedImages, currentImage, _parent);
    connect(&dialog, &GenerateMaskDialog::sam21InstallRequested,
            this,
            [this, &dialog](const QString &variantToken)
            {
                installSam21Model(variantToken, &dialog);
            });
    if (dialog.exec() != QDialog::Accepted)
    {
        return;
    }

    const QJsonObject settings = dialog.collectSettings();
    const QStringList targetImages = maskTargetsFromSettings(settings, resolvedAllImages);
    if (targetImages.isEmpty())
    {
        showWarning(QStringLiteral("没有选中可生成蒙版的照片。"), QStringLiteral("生成蒙版"));
        return;
    }

    const QString masksDir = ProjectIO::maskOutputDir(projectPath);
    if (masksDir.isEmpty() || !QDir().mkpath(masksDir))
    {
        showWarning(QStringLiteral("无法创建蒙版输出目录：%1").arg(masksDir), QStringLiteral("生成蒙版"));
        return;
    }

    const auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    const int totalImages = targetImages.size();
    _maskGenerationCancelFlag = cancelFlag;
    emit maskGenerationProgressChanged(QStringLiteral("生成蒙版"), 0, totalImages);

    struct GenerateMaskResult
    {
        QMap<QString, QJsonObject> maskRecordsByImage;
        QStringList generatedImages;
        QStringList errors;
        bool cancelled = false;
    };

    QPointer<ProjectManager> managerGuard(this);
    xjw::gui::tasks::runGuarded(
        this,
        [settings, targetImages, projectPath, cancelFlag, managerGuard, totalImages]() -> GenerateMaskResult
        {
            GenerateMaskResult result;
            const auto options = maskOptionsFromSettings(settings);
            const auto operation = maskOperationFromSettings(settings);
            const QString methodToken =
                settings.value(QStringLiteral("method")).toString(QStringLiteral("black_background"));
            const bool useSam21 = methodToken == QLatin1String("sam21");
            const bool useU2Net = methodToken == QLatin1String("u2net");

            std::unique_ptr<xjw::mask::Sam21MaskGenerator> sam21Generator;
            if (useSam21)
            {
                const auto variant = xjw::mask::sam21VariantFromToken(
                    settings.value(QStringLiteral("sam21_variant")).toString(QStringLiteral("tiny")).toStdString());
                const bool requestCuda =
                    settings.value(QStringLiteral("sam21_device")).toString(QStringLiteral("cuda"))
                    == QLatin1String("cuda");
                const auto modelNamesForLog = xjw::mask::sam21TorchScriptModelNames(variant, requestCuda);

                QString configError;
                const auto sam21Config = sam21MaskConfigFromSettings(settings, &configError);
                if (!sam21Config.has_value())
                {
                    result.errors << configError;
                    return result;
                }

                try
                {
                    sam21Generator = std::make_unique<xjw::mask::Sam21MaskGenerator>(sam21Config.value());
                    LOG_INFO(QStringLiteral("SAM2.1 蒙版模型已加载: encoder=%1 decoder=%2 device=%3")
                                 .arg(QString::fromStdString(modelNamesForLog.encoder),
                                      QString::fromStdString(modelNamesForLog.decoder),
                                      QString::fromStdString(sam21Generator->deviceLabel())));
                }
                catch (const std::exception &error)
                {
                    result.errors << QStringLiteral("SAM2.1 模型加载失败：%1").arg(QString::fromUtf8(error.what()));
                    return result;
                }
            }

            std::unique_ptr<xjw::mask::U2NetMaskGenerator> u2netGenerator;
            if (useU2Net)
            {
                QString configError;
                const auto u2netConfig = u2netMaskConfigFromSettings(settings, &configError);
                if (!u2netConfig.has_value())
                {
                    result.errors << configError;
                    return result;
                }

                try
                {
                    u2netGenerator = std::make_unique<xjw::mask::U2NetMaskGenerator>(u2netConfig.value());
                    LOG_INFO(QStringLiteral("U2Net ONNX 蒙版模型已加载: model=U2Net_v1.onnx device=%1")
                                 .arg(QString::fromStdString(u2netGenerator->deviceLabel())));
                }
                catch (const std::exception &error)
                {
                    result.errors << QStringLiteral("U2Net ONNX 模型加载失败：%1").arg(QString::fromUtf8(error.what()));
                    return result;
                }
            }

            auto reportProgress = [managerGuard, totalImages](int done)
            {
                if (!managerGuard)
                {
                    return;
                }
                xjw::gui::tasks::postGuarded(managerGuard.data(),
                                             [done, totalImages](ProjectManager *self)
                {
                    emit self->maskGenerationProgressChanged(QStringLiteral("生成蒙版"), done, totalImages);
                });
            };

            int completed = 0;
            for (const QString &imagePath : targetImages)
            {
                if (cancelFlag->load(std::memory_order_relaxed))
                {
                    result.cancelled = true;
                    break;
                }

                const cv::Mat source = xjw::common::io::readImage(imagePath, cv::IMREAD_UNCHANGED);
                if (source.empty())
                {
                    result.errors << QStringLiteral("%1: 读取失败").arg(QFileInfo(imagePath).fileName());
                    reportProgress(++completed);
                    continue;
                }

                cv::Mat generated;
                try
                {
                    if (useSam21)
                    {
                        const auto prompt = xjw::mask::Sam21Prompt::autoBox(source);
                        const auto maskResult = sam21Generator->generate(source, prompt);
                        generated = maskResult.mask;
                    }
                    else if (useU2Net)
                    {
                        const auto maskResult = u2netGenerator->generate(source);
                        generated = maskResult.mask;
                    }
                    else
                    {
                        generated = xjw::mask::generateMask(source, options);
                    }
                }
                catch (const std::exception &error)
                {
                    result.errors << QStringLiteral("%1: %2")
                                         .arg(QFileInfo(imagePath).fileName(),
                                              QString::fromUtf8(error.what()));
                    reportProgress(++completed);
                    continue;
                }

                if (generated.empty())
                {
                    result.errors << QStringLiteral("%1: 生成结果为空").arg(QFileInfo(imagePath).fileName());
                    reportProgress(++completed);
                    continue;
                }

                const QString maskPath = ProjectIO::maskOutputPathForImage(projectPath, imagePath);
                if (QFileInfo::exists(maskPath) && operation != xjw::mask::MaskOperation::Replace)
                {
                    const cv::Mat existing = xjw::common::io::readImage(maskPath, cv::IMREAD_GRAYSCALE);
                    if (!existing.empty())
                    {
                        generated = xjw::mask::composeMasks(existing, generated, operation);
                    }
                }

                if (!xjw::common::io::writeImage(maskPath, generated))
                {
                    result.errors << QStringLiteral("%1: 写入失败").arg(QFileInfo(maskPath).fileName());
                    reportProgress(++completed);
                    continue;
                }

                QJsonObject record;
                record.insert(QStringLiteral("mask_path"), QDir::cleanPath(maskPath));
                record.insert(QStringLiteral("mask_method"), methodToken);
                record.insert(QStringLiteral("mask_updated_at"),
                              QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
                result.maskRecordsByImage.insert(normalizePath(imagePath), record);
                result.generatedImages << imagePath;
                reportProgress(++completed);
            }
            return result;
        },
        [projectPath, masksDir, cancelFlag](ProjectManager *self, GenerateMaskResult result)
        {
            if (self->_maskGenerationCancelFlag == cancelFlag)
            {
                self->_maskGenerationCancelFlag.reset();
            }

            const bool finishedSuccessfully =
                !result.cancelled && result.errors.isEmpty() && !result.generatedImages.isEmpty();
            emit self->maskGenerationFinished(finishedSuccessfully);

            if (self->currentProjectPath() != projectPath)
            {
                return;
            }

            if (result.generatedImages.isEmpty())
            {
                const QString detail = result.errors.isEmpty()
                    ? QStringLiteral("未生成任何蒙版。")
                    : result.errors.join(QStringLiteral("; "));
                const QString message = result.cancelled
                    ? QStringLiteral("蒙版生成已取消。%1").arg(detail)
                    : QStringLiteral("蒙版生成失败：%1").arg(detail);
                self->showWarning(message, QStringLiteral("生成蒙版"));
                return;
            }

            QJsonObject meta = self->_projectData->coreFilesMeta();
            QJsonArray images = meta.value(QStringLiteral("images")).toArray();
            for (int i = 0; i < images.size(); ++i)
            {
                QJsonObject image = images.at(i).toObject();
                const QString imagePath = ProjectIO::resolveProjectResourcePath(
                    projectPath,
                    image.value(QStringLiteral("path")).toString());
                const QString normalized = normalizePath(imagePath);
                if (!result.maskRecordsByImage.contains(normalized))
                {
                    continue;
                }

                const QJsonObject record = result.maskRecordsByImage.value(normalized);
                image.insert(QStringLiteral("mask_path"), record.value(QStringLiteral("mask_path")));
                image.insert(QStringLiteral("mask_method"), record.value(QStringLiteral("mask_method")));
                image.insert(QStringLiteral("mask_updated_at"), record.value(QStringLiteral("mask_updated_at")));
                images.replace(i, image);
            }
            meta.insert(QStringLiteral("images"), images);

            persistProjectMeta(self->_projectData, meta, true);
            emit self->projectMetadataUpdated(projectPath);
            emit self->masksGenerated(result.generatedImages);

            QString message = result.cancelled
                ? QStringLiteral("已取消，已保留 %1 张照片的蒙版。").arg(result.generatedImages.size())
                : QStringLiteral("已生成 %1 张照片的蒙版。").arg(result.generatedImages.size());
            if (!result.errors.isEmpty())
            {
                message += QStringLiteral("\n部分失败：%1").arg(result.errors.join(QStringLiteral("; ")));
            }
            QMessageBox::information(self->_parent, QStringLiteral("生成蒙版"), message);
            LOG_INFO(QStringLiteral("蒙版生成完成: count=%1 dir=%2")
                         .arg(result.generatedImages.size())
                         .arg(masksDir));
        });
}

void ProjectManager::runReferenceQualityCheck()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再执行点云/DEM 精度检查。")))
    {
        return;
    }

    const auto result = xjw::gui::project::writeReferenceDatasetQualityReport(_projectData);
    if (!result.saved)
    {
        showWarning(result.errorMessage.isEmpty()
                        ? QStringLiteral("生成点云/DEM 精度检查报告失败。")
                        : result.errorMessage,
                    QStringLiteral("点云/DEM 精度检查"));
        return;
    }

    const QString status = result.record.value(QStringLiteral("comparison_available")).toBool()
        ? QStringLiteral("已找到可检查的参考数据与项目成果。")
        : QStringLiteral("报告已生成，但当前缺少参考数据或可比较的项目成果。");
    LOG_INFO(QStringLiteral("参考数据精度检查报告已生成: %1").arg(result.jsonPath));
    QMessageBox::information(_parent,
                             QStringLiteral("点云/DEM 精度检查"),
                             QStringLiteral("%1\nJSON: %2\nCSV: %3")
                                 .arg(status, result.jsonPath, result.csvPath));
}

void ProjectManager::prepareReferenceTerrainBundleAdjust()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再使用参考地形约束重新平差。")))
    {
        return;
    }

    const auto result = xjw::gui::project::writeReferenceTerrainPriorPreflightReport(_projectData);
    if (!result.saved)
    {
        showWarning(result.errorMessage.isEmpty()
                        ? QStringLiteral("生成参考地形平差前置检查报告失败。")
                        : result.errorMessage,
                    QStringLiteral("参考地形约束重新平差"));
        return;
    }

    const bool ready = result.record.value(QStringLiteral("ready")).toBool();
    const QString status = result.record.value(QStringLiteral("status")).toString();
    LOG_INFO(QStringLiteral("参考地形平差前置检查报告已生成: %1 ready=%2")
             .arg(result.jsonPath)
             .arg(ready));
    if (!ready)
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("参考地形约束重新平差"),
                                 QStringLiteral("前置检查未通过：%1。\n请先导入 role=ba_prior 的参考 DEM/LiDAR，并完成正式空三。\nJSON: %2\nCSV: %3")
                                     .arg(status, result.jsonPath, result.csvPath));
        return;
    }

    const QJsonObject meta = _projectData->metadata();
    const QString demPriorPath = firstReferenceDemPriorPath(meta);
    const QString laserPriorPath = firstReferenceLaserPriorPath(meta);
    if (demPriorPath.isEmpty() && laserPriorPath.isEmpty())
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("参考地形约束重新平差"),
                                 QStringLiteral("前置检查通过，但未找到可直接用于 BA 的参考数据。\n"
                                                "请导入 role=ba_prior 的 DEM（GeoTIFF/VRT），或带 LiDAR/点云类型的 PLY 文件。\n"
                                                "JSON: %1\nCSV: %2")
                                     .arg(result.jsonPath, result.csvPath));
        return;
    }

    const QStringList images = getAllImages();
    if (images.size() < 2)
    {
        QMessageBox::warning(_parent,
                             QStringLiteral("参考地形约束重新平差"),
                             QStringLiteral("项目影像少于 2 张，无法执行 BA。"));
        return;
    }

    const QString assetsDir = ProjectIO::projectAssetsDir(currentProjectPath());
    const QString outputDir = QDir(assetsDir).filePath(
        QStringLiteral("bundle_adjust/%1_%2")
            .arg(demPriorPath.isEmpty()
                     ? QStringLiteral("reference_laser")
                     : QStringLiteral("reference_terrain"))
            .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_hhmmss"))));

    const QString prompt = demPriorPath.isEmpty()
        ? QStringLiteral("将使用 LiDAR/点云 PLY 作为 BA 点到面 soft prior 重新平差。\n\n"
                         "参考点云: %1\n"
                         "无 normal 字段时: 按水平地形面约束使用\n"
                         "影像数量: %2\n"
                         "输出目录: %3\n\n"
                         "继续执行？")
              .arg(laserPriorPath)
              .arg(images.size())
              .arg(outputDir)
        : QStringLiteral("将使用参考 DEM 作为 BA 高程 soft prior 重新平差。\n\n"
                         "参考 DEM: %1\n"
                         "影像数量: %2\n"
                         "输出目录: %3\n\n"
                         "继续执行？")
              .arg(demPriorPath)
              .arg(images.size())
              .arg(outputDir);

    const QMessageBox::StandardButton choice = QMessageBox::question(
        _parent,
        QStringLiteral("参考地形约束重新平差"),
        prompt,
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::Yes);
    if (choice != QMessageBox::Yes)
    {
        return;
    }

    QJsonObject extra;
    if (demPriorPath.isEmpty())
    {
        extra[QStringLiteral("enable_laser_constraints")] = true;
        extra[QStringLiteral("laser_constraint_cloud_path")] = laserPriorPath;
        extra[QStringLiteral("laser_association_max_distance_m")] = 1.0;
        extra[QStringLiteral("laser_voxel_size_m")] = 0.0;
        extra[QStringLiteral("laser_max_curvature")] = 0.2;
        extra[QStringLiteral("laser_max_samples")] = 500000;
        extra[QStringLiteral("laser_missing_normals_as_height_planes")] = true;
        extra[QStringLiteral("laser_weight")] = 1.0;
        extra[QStringLiteral("laser_huber_delta_m")] =
            result.record.value(QStringLiteral("recommended_huber_delta_m")).toDouble(0.5);
    }
    else
    {
        extra[QStringLiteral("enable_reference_terrain_prior")] = true;
        extra[QStringLiteral("reference_terrain_dem_path")] = demPriorPath;
        extra[QStringLiteral("reference_terrain_sigma_m")] =
            result.record.value(QStringLiteral("recommended_sigma_m")).toDouble(1.0);
        extra[QStringLiteral("reference_terrain_huber_delta_m")] =
            result.record.value(QStringLiteral("recommended_huber_delta_m")).toDouble(0.5);
        extra[QStringLiteral("reference_terrain_max_association_distance_m")] = 2.0;
    }
    extra[QStringLiteral("refine_camera_pose")] = true;
    extra[QStringLiteral("export_run_json")] = true;
    extra[QStringLiteral("export_summary_txt")] = true;
    extra[QStringLiteral("export_camera_csv")] = true;
    extra[QStringLiteral("export_points_csv")] = true;

    if (demPriorPath.isEmpty())
    {
        LOG_INFO(QStringLiteral("LiDAR 点到面 BA soft prior 启动: cloud=%1 images=%2 output=%3")
                 .arg(laserPriorPath)
                 .arg(images.size())
                 .arg(outputDir));
    }
    else
    {
        LOG_INFO(QStringLiteral("参考地形 BA soft prior 启动: dem=%1 images=%2 output=%3")
                 .arg(demPriorPath)
                 .arg(images.size())
                 .arg(outputDir));
    }
    startBundleAdjustAsync(images, outputDir, qMax(1, QThread::idealThreadCount()), false, extra);
}

void ProjectManager::deleteGeneratedData(const QString &section, const QStringList &resourcePaths)
{
    if (!_projectData || resourcePaths.isEmpty())
    {
        return;
    }
    if (section == QStringLiteral("照片"))
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("照片分组不支持删除数据，请使用移除引用。"));
        return;
    }

    const int selectedCount = resourcePaths.size();
    const QMessageBox::StandardButton confirm = QMessageBox::question(
        _parent,
        QStringLiteral("删除数据"),
        selectedCount == 1
            ? QStringLiteral("确定删除所选%1数据及其关联生成文件吗？此操作不可撤销。").arg(section)
            : QStringLiteral("确定删除所选 %1 项%2数据及其关联生成文件吗？此操作不可撤销。").arg(selectedCount).arg(section),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (confirm != QMessageBox::Yes)
    {
        return;
    }

    if (section == QStringLiteral("连接点"))
    {
        const auto result = xjw::gui::project::ProjectTiePointResultService::deleteAll(_projectData);
        if (!result.success)
        {
            QMessageBox::warning(_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("删除失败：%1").arg(result.errorMessage));
            return;
        }

        refreshReconstructionQualityReport();
        QMessageBox::information(_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("已删除连接点数据及其关联生成文件。"));
        return;
    }

    const auto cleanupResult = xjw::gui::project::ProjectResourceCleanupService::cleanupGeneratedData(_projectData,
                                                                                                       section,
                                                                                                       resourcePaths);
    if (cleanupResult.unsupportedSection)
    {
        QMessageBox::warning(_parent,
                             QStringLiteral("删除数据"),
                             QStringLiteral("当前分组暂不支持删除数据：%1").arg(section));
        return;
    }

    if (!cleanupResult.success && !cleanupResult.errorMessage.isEmpty())
    {
        QMessageBox::warning(_parent,
                             QStringLiteral("删除数据"),
                             QStringLiteral("删除失败：%1").arg(cleanupResult.errorMessage));
        return;
    }

    if (cleanupResult.noMatchedRecords)
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("未找到可删除的%1数据记录。").arg(section));
        return;
    }

    if (cleanupResult.failedPaths.isEmpty())
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("已删除 %1 项%2数据。").arg(cleanupResult.removedCount).arg(section));
    }
    else
    {
        QMessageBox::warning(_parent,
                             QStringLiteral("删除数据"),
                             QStringLiteral("已移除 %1 项%2数据记录，但以下文件/目录删除失败：\n%3")
                                 .arg(cleanupResult.removedCount)
                                 .arg(section)
                                 .arg(cleanupResult.failedPaths.join(QStringLiteral("\n"))));
    }
}

void ProjectManager::packResource(const QString &resourcePath)
{
    QString err;
    if (_projectData && !_projectData->packResource(resourcePath, &err))
    {
        QMessageBox::warning(_parent, QStringLiteral("提示"), err);
    }
}

// Bundle adjust related APIs removed (core removed).

//==============================================================================
// 设置管理 (委托给ProjectData)
//==============================================================================

QJsonObject ProjectManager::loadUiSettings() const
{
    return _projectData ? _projectData->loadUiSettings() : QJsonObject();
}

//==============================================================================
// 查询接口 (委托给ProjectData)
//==============================================================================

bool ProjectManager::isDirty() const
{
    return _projectData ? _projectData->isDirty() : false;
}

QString ProjectManager::currentProjectPath() const
{
    return _projectData ? _projectData->currentProjectPath() : QString();
}

QJsonObject ProjectManager::currentMeta() const
{
    if (!_projectData)
    {
        return {};
    }

    return xjw::gui::project::ProjectTiePointResultService::metadataWithCurrentOnly(
        _projectData->metadata(),
        _projectData->currentProjectPath());
}

QJsonObject ProjectManager::coreProjectMeta() const
{
    return _projectData ? _projectData->coreFilesMeta() : QJsonObject();
}

QStringList ProjectManager::getImagesByCategory(const QString &category) const
{
    return _projectData ? _projectData->getImagesByCategory(category) : QStringList();
}

QStringList ProjectManager::getAllImages() const
{
    return _projectData ? _projectData->getAllImages() : QStringList();
}

QString ProjectManager::findMatchFileForPair(const QString &imgA, const QString &imgB) const
{
    return _projectData ? _projectData->findMatchFile(imgA, imgB) : QString();
}

bool ProjectManager::hasTemporaryMeta() const
{
    return _projectData ? _projectData->hasTemporaryMetadata() : false;
}

void ProjectManager::discardTemporaryMeta()
{
    if (_projectData) {
        _projectData->clearTemporaryMetadata();
    }
}

void ProjectManager::writeMetadataToTempAsync(const QJsonObject &meta, bool markDirty)
{
    if (_projectData) {
        _projectData->updateMetadata(meta, markDirty);
        _projectData->saveTemporaryMetadata();
    }
}

void ProjectManager::refreshReconstructionQualityReport()
{
    if (!_projectData || !_projectData->hasProject())
    {
        return;
    }

    const auto reportResult =
        xjw::gui::project::writeReconstructionQualityProjectReport(_projectData);
    if (!reportResult.saved && !reportResult.errorMessage.isEmpty())
    {
        LOG_WARN(QStringLiteral("重建质量报告刷新失败: %1").arg(reportResult.errorMessage));
    }
}

void ProjectManager::appendIpfindResult(const QString &input, const QString &output, const QJsonObject &settings)
{
    if (_projectData) {
        _projectData->appendIpfindResult(input, output, settings);
        // 通知界面刷新该影像的 interest points 缓存
        // 从输出路径推导后缀 (.sp/.dsk/.alk/.sift/...)
        QString suffix;
        for (const char *suf : {".sp", ".dsk", ".alk", ".sift", ".orb", ".akz", ".dedode"})
        {
            if (output.endsWith(QLatin1String(suf))) { suffix = QLatin1String(suf); break; }
        }
        emit ipfindResultAppended(input, suffix.isEmpty() ? QStringLiteral(".sp") : suffix);
    }
}

void ProjectManager::appendIpfindResults(const QVector<ProjectIpfindResultRecord> &records)
{
    if (!_projectData || records.isEmpty())
    {
        return;
    }

    _projectData->appendIpfindResults(records);
    for (const ProjectIpfindResultRecord &record : records)
    {
        QString suffix;
        for (const char *suf : {".sp", ".dsk", ".alk", ".sift", ".orb", ".akz", ".dedode"})
        {
            if (record.output.endsWith(QLatin1String(suf)))
            {
                suffix = QLatin1String(suf);
                break;
            }
        }
        emit ipfindResultAppended(record.input, suffix.isEmpty() ? QStringLiteral(".sp") : suffix);
    }
}

void ProjectManager::appendIpmatchResult(const QStringList &outputs, const QJsonObject &settings)
{
    if (_projectData) {
        _projectData->appendIpmatchResult(outputs, settings);
    }
}

void ProjectManager::appendIpmatchResults(const QVector<ProjectIpmatchResultRecord> &records)
{
    if (_projectData && !records.isEmpty())
    {
        _projectData->appendIpmatchResults(records);
    }
}

void ProjectManager::startBundleAdjustAsync(const QStringList &images,
                                            const QString &outputDir,
                                            int threads,
                                            bool dryRun,
                                            const QJsonObject &extraSettings)
{
    // ── 前置检查（仅快速校验，不做任何 IO）───────────────────────────────
    if (!ensureProjectOpen()) return;
    if (images.size() < 2) {
        QMessageBox::warning(_parent, QStringLiteral("提示"), QStringLiteral("至少需要选择两张影像"));
        return;
    }
    if (outputDir.trimmed().isEmpty()) {
        QMessageBox::warning(_parent, QStringLiteral("提示"), QStringLiteral("请指定输出目录"));
        return;
    }

    const QString outDir = QDir::cleanPath(outputDir);

    // ── 只获取核心数据（影像列表/相机，无结果数组，速度极快）──────────────
    // 结果数据（ipmatch_results）在后台线程直接读取，避免 UI 阻塞
    const QJsonObject coreData   = _projectData->coreFilesMeta();
    const QString     plascanPath = _projectData->currentProjectPath();

    const int minMatches = qMax(0, extraSettings.value(QStringLiteral("min_matches")).toInt(0));

    // ── 组装 BaServiceOptions（纯参数，不含相机/轨迹——后台填充）──────────
    xjw::gui::BaServiceOptions opts;
    opts.selectedImages   = images;
    opts.outputDir        = outDir;
    opts.dryRun           = dryRun;
    opts.threads          = threads;
    opts.baOpt.maxIterations       = qBound(3,  extraSettings.value(QStringLiteral("max_iterations")).toInt(20),  200);
    opts.baOpt.maxPointIterations  = qBound(1,  extraSettings.value(QStringLiteral("max_point_iterations")).toInt(12), 100);
    opts.baOpt.maxCameraIterations = qBound(1,  extraSettings.value(QStringLiteral("max_camera_iterations")).toInt(10), 100);
    opts.baOpt.refineCameraPose    = extraSettings.value(QStringLiteral("refine_camera_pose")).toBool(true);
    opts.baOpt.huberDelta          = extraSettings.value(QStringLiteral("huber_delta")).toDouble(3.0);
    opts.baOpt.finiteDiffEps       = extraSettings.value(QStringLiteral("finite_diff_eps")).toDouble(1e-6);
    opts.baOpt.damping             = extraSettings.value(QStringLiteral("damping")).toDouble(1e-3);
    opts.baOpt.stepTolerance       = extraSettings.value(QStringLiteral("step_tolerance")).toDouble(1e-8);
    opts.baOpt.numThreads          = threads;
    const QString baBackendName =
        extraSettings.value(QStringLiteral("ba_backend")).toString(QStringLiteral("auto")).trimmed().toLower();
    if (baBackendName == QLatin1String("auto"))
    {
        opts.baOpt.backend = xjw::BABackend::Auto;
    }
    else if (baBackendName == QLatin1String("legacy_cpu"))
    {
        opts.baOpt.backend = xjw::BABackend::LegacyCpu;
    }
    else if (baBackendName == QLatin1String("ceres_cpu"))
    {
        opts.baOpt.backend = xjw::BABackend::CeresCpu;
    }
    else if (baBackendName == QLatin1String("ceres_cuda"))
    {
        opts.baOpt.backend = xjw::BABackend::CeresCuda;
    }
    else if (baBackendName == QLatin1String("native_cuda"))
    {
        opts.baOpt.backend = xjw::BABackend::NativeCuda;
    }
    else
    {
        opts.baOpt.backend = xjw::BABackend::Auto;
    }
    opts.baOpt.ceresCudaDevice = qMax(
        0,
        extraSettings.value(QStringLiteral("ba_cuda_device")).toInt(0));
    opts.baOpt.minCeresCudaCameras = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_cuda_cameras")).toInt(opts.baOpt.minCeresCudaCameras));
    opts.baOpt.minCeresCudaObservations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_cuda_observations")).toInt(
            opts.baOpt.minCeresCudaObservations));
    opts.baOpt.nativeCudaDevice = qMax(
        0,
        extraSettings.value(QStringLiteral("ba_native_cuda_device")).toInt(opts.baOpt.nativeCudaDevice));
    opts.baOpt.minNativeCudaCameras = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_native_cuda_cameras")).toInt(
            opts.baOpt.minNativeCudaCameras));
    opts.baOpt.minNativeCudaObservations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_native_cuda_observations")).toInt(
            opts.baOpt.minNativeCudaObservations));
    opts.baOpt.nativeCudaMaxPcgIterations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_native_cuda_max_pcg_iterations")).toInt(
            opts.baOpt.nativeCudaMaxPcgIterations));
    opts.baOpt.nativeCudaPcgTolerance = qMax(
        1e-12,
        extraSettings.value(QStringLiteral("ba_native_cuda_pcg_tolerance")).toDouble(
            opts.baOpt.nativeCudaPcgTolerance));
    opts.baOpt.minCeresCpuObservations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_cpu_observations")).toInt(
            opts.baOpt.minCeresCpuObservations));
    opts.baOpt.maxCeresPointOnlyObservations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_max_ceres_point_only_observations")).toInt(
            opts.baOpt.maxCeresPointOnlyObservations));
    opts.baOpt.allowBackendFallback =
        extraSettings.value(QStringLiteral("ba_allow_backend_fallback")).toBool(true);
    opts.baOpt.enableBackendQualityGate =
        extraSettings.value(QStringLiteral("ba_enable_backend_quality_gate")).toBool(true);
    opts.baOpt.maxAcceptedRmsGrowth = qMax(
        0.0,
        extraSettings.value(QStringLiteral("ba_max_accepted_rms_growth")).toDouble(
            opts.baOpt.maxAcceptedRmsGrowth));
    opts.baOpt.minAcceptedValidTrackRatio = qMax(
        0.0,
        extraSettings.value(QStringLiteral("ba_min_accepted_valid_track_ratio")).toDouble(
            opts.baOpt.minAcceptedValidTrackRatio));
    opts.baOpt.compareAutoBackendWithLegacy =
        extraSettings.value(QStringLiteral("ba_compare_auto_backend_with_legacy")).toBool(true);
    opts.baOpt.enablePointFilter    = true;
    opts.baOpt.filterMaxReprojError = extraSettings.value(QStringLiteral("filter_max_reproj_error")).toDouble(2.5);
    opts.baOpt.filterSigmaFactor   = extraSettings.value(QStringLiteral("filter_sigma_factor")).toDouble(3.0);
    opts.exportTsai        = extraSettings.value(QStringLiteral("export_tsai")).toBool(true);
    opts.exportSummaryTxt  = extraSettings.value(QStringLiteral("export_summary_txt")).toBool(true);
    opts.exportPointsCsv   = extraSettings.value(QStringLiteral("export_points_csv")).toBool(true);
    opts.exportCameraCsv   = extraSettings.value(QStringLiteral("export_camera_csv")).toBool(true);
    opts.exportRunJson     = extraSettings.value(QStringLiteral("export_run_json")).toBool(true);
    opts.exportEvalPlot    = extraSettings.value(QStringLiteral("export_eval_plot")).toBool(true);
    opts.enableLaserConstraints = extraSettings.value(QStringLiteral("enable_laser_constraints")).toBool(false);
    opts.laserConstraintCloudPath = extraSettings.value(QStringLiteral("laser_constraint_cloud_path")).toString().trimmed();
    opts.laserAssociationMaxDistanceMeters = qMax(
        0.0,
        extraSettings.value(QStringLiteral("laser_association_max_distance_m")).toDouble(1.0));
    opts.laserVoxelSizeMeters = qMax(
        0.0,
        extraSettings.value(QStringLiteral("laser_voxel_size_m")).toDouble(0.0));
    opts.laserMaxCurvature = qBound(
        0.0,
        extraSettings.value(QStringLiteral("laser_max_curvature")).toDouble(0.2),
        1.0);
    opts.laserMaxSamples = qBound(
        1,
        extraSettings.value(QStringLiteral("laser_max_samples")).toInt(500000),
        10000000);
    opts.laserUseMissingNormalsAsHeightPlanes =
        extraSettings.value(QStringLiteral("laser_missing_normals_as_height_planes")).toBool(false);
    opts.laserWeight = qMax(
        0.0,
        extraSettings.value(QStringLiteral("laser_weight")).toDouble(1.0));
    opts.laserHuberDeltaMeters = qMax(
        1e-9,
        extraSettings.value(QStringLiteral("laser_huber_delta_m")).toDouble(0.2));
    opts.enableReferenceTerrainPrior =
        extraSettings.value(QStringLiteral("enable_reference_terrain_prior")).toBool(false);
    opts.referenceTerrainDemPath =
        extraSettings.value(QStringLiteral("reference_terrain_dem_path")).toString().trimmed();
    opts.referenceTerrainSigmaMeters = qMax(
        1e-9,
        extraSettings.value(QStringLiteral("reference_terrain_sigma_m")).toDouble(1.0));
    opts.referenceTerrainMaxAssociationDistanceMeters = qMax(
        0.0,
        extraSettings.value(QStringLiteral("reference_terrain_max_association_distance_m")).toDouble(2.0));
    opts.referenceTerrainHuberDeltaMeters = qMax(
        1e-9,
        extraSettings.value(QStringLiteral("reference_terrain_huber_delta_m")).toDouble(0.5));

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    setAtCancelFlag(cancelFlag);
    opts.baOpt.cancelFlag = cancelFlag;
    QPointer<ProjectManager> baProgressSelf(this);
    opts.baOpt.progressCallback =
        [baProgressSelf, cancelFlag](int currentIteration, int maxIterations, double avgRms, int validPoints) -> bool
        {
            if (!baProgressSelf ||
                cancelFlag->load(std::memory_order_relaxed))
            {
                return false;
            }

            const int safeMaxIterations = std::max(1, maxIterations);
            const int percent = qBound(
                10,
                10 + static_cast<int>(std::lround(80.0 * currentIteration / safeMaxIterations)),
                90);
            const QString stage = QStringLiteral("光束法平差优化中... %1/%2 RMS=%3 有效点=%4")
                .arg(currentIteration)
                .arg(safeMaxIterations)
                .arg(avgRms, 0, 'f', 4)
                .arg(validPoints);
            QMetaObject::invokeMethod(
                baProgressSelf.data(),
                [baProgressSelf, cancelFlag, stage, percent]()
                {
                    if (!baProgressSelf)
                    {
                        return;
                    }
                    if (baProgressSelf->_atCancelFlag != cancelFlag ||
                        cancelFlag->load(std::memory_order_relaxed))
                    {
                        return;
                    }
                    emit baProgressSelf->atProgressChanged(stage, percent);
                },
                Qt::QueuedConnection);
            return true;
        };

    emit atProgressChanged(QStringLiteral("光束法平差准备中..."), 1);

    LOG_INFO(QStringLiteral("BA: 特征/匹配准备完毕，启动光束法平差"));

    QPointer<ProjectManager> self(this);

xjw::gui::tasks::runGuarded(
    this,
    [self, coreData, plascanPath, images, minMatches,
     cancelFlag, opts = std::move(opts), isDryRun = dryRun]() mutable
    {
        auto finishTask = [self, cancelFlag](bool success)
        {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(
                self.data(),
                [self, cancelFlag, success]()
                {
                    if (!self)
                    {
                        return;
                    }
                    if (self->_atCancelFlag != cancelFlag)
                    {
                        return;
                    }
                    self->_atCancelFlag.reset();
                    emit self->atProgressFinished(success);
                },
                Qt::QueuedConnection);
        };

        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(
            self.data(),
            [self, cancelFlag]()
            {
                if (!self)
                {
                    return;
                }
                if (self->_atCancelFlag != cancelFlag ||
                    cancelFlag->load(std::memory_order_relaxed))
                {
                    return;
                }
                emit self->atProgressChanged(QStringLiteral("光束法平差构建输入..."), 5);
            },
            Qt::QueuedConnection);

        const BundleAdjustExecutionResult executionResult = runBundleAdjustExecution(coreData,
                                                                                     plascanPath,
                                                                                     images,
                                                                                     minMatches,
                                                                                     std::move(opts));

        if (cancelFlag->load(std::memory_order_relaxed))
        {
            LOG_INFO(QStringLiteral("BA: 用户取消了光束法平差"));
            if (self)
            {
                QMetaObject::invokeMethod(self.data(),
                [self, cancelFlag]()
                {
                    if (!self)
                    {
                        return;
                    }
                    if (self->_atCancelFlag == cancelFlag)
                    {
                        emit self->bundleAdjustPreviewReady(QJsonObject());
                    }
                },
                Qt::QueuedConnection);
            }
            finishTask(false);
            return;
        }

        if (executionResult.buildStatus != xjw::core::project::BaInputBuildStatus::Ok) {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(),
                [self, cancelFlag, buildStatus = executionResult.buildStatus]() {
                    if (!self)
                    {
                        return;
                    }
                    if (self->_atCancelFlag != cancelFlag)
                    {
                        return;
                    }
                    const QString msg =
                        buildStatus == xjw::core::project::BaInputBuildStatus::NotEnoughCameras
                        ? QStringLiteral("所选影像中可用相机参数不足（至少需要两台相机）")
                        : QStringLiteral("未找到可用于光束法平差的匹配点（请检查选中影像是否已有匹配结果）");
                    QMessageBox::warning(self->_parent, QStringLiteral("提示"), msg);
                    emit self->bundleAdjustPreviewReady(QJsonObject());
                    self->_atCancelFlag.reset();
                    emit self->atProgressFinished(false);
                },
                Qt::QueuedConnection);
            return;
        }

        LOG_INFO(
            QStringLiteral("BA: 启动后台平差计算 (%1 台相机, %2 条轨迹)...")
                .arg(executionResult.serviceResult.pendingCamUpdates.size())
                .arg(executionResult.serviceResult.resultJson.value(QStringLiteral("track_count")).toInt()));

        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(self.data(),
            [self,
             cancelFlag,
             baResult = executionResult.serviceResult,
             beforeCamMeta = executionResult.beforeCamMeta,
             isDryRun]()
            {
                if (!self)
                {
                    return;
                }
                if (self->_atCancelFlag != cancelFlag)
                {
                    return;
                }
                if (cancelFlag->load(std::memory_order_relaxed))
                {
                    emit self->bundleAdjustPreviewReady(QJsonObject());
                    self->_atCancelFlag.reset();
                    emit self->atProgressFinished(false);
                    return;
                }

                emit self->atProgressChanged(QStringLiteral("光束法平差整理结果..."), 95);

                if (!baResult.success && !isDryRun) {
                    QMessageBox::warning(self->_parent,
                        QStringLiteral("平差提示"),
                        QStringLiteral("光束法平差执行出现问题：%1").arg(baResult.errorMessage));
                }
                self->_pendingBaBeforeCameraMeta = beforeCamMeta;
                self->_pendingBaCameraMeta  = baResult.pendingCamUpdates;
                self->_pendingBaResult      = baResult.resultJson;
                self->_hasPendingBaPreview  = !self->_pendingBaCameraMeta.isEmpty();
                emit self->bundleAdjustPreviewReady(baResult.resultJson);
                self->_atCancelFlag.reset();
                emit self->atProgressFinished(baResult.success);
            },
            Qt::QueuedConnection);
    },
    [](ProjectManager *) {});
}

// ==============================================================================
// 空中三角测量相关
// ==============================================================================
bool ProjectManager::setImageCameras(const QMap<QString, QJsonObject> &cameras,
                                     int    *updatedCount,
                                     QString *errorMsg)
{
    // 直接委托给数据层——ProjectManager 不持有算法逻辑，仅转发
    if (!_projectData)
    {
        if (errorMsg) *errorMsg = QStringLiteral("ProjectData 未初始化");
        if (updatedCount) *updatedCount = 0;
        return false;
    }
    return _projectData->setImageCameras(cameras, updatedCount, errorMsg);
}

bool ProjectManager::replaceImageCameras(const QStringList &targetImagePaths,
                                         const QMap<QString, QJsonObject> &cameras,
                                         int *updatedCount,
                                         int *clearedCount,
                                         QString *errorMsg)
{
    if (!_projectData)
    {
        if (errorMsg) *errorMsg = QStringLiteral("ProjectData 未初始化");
        if (updatedCount) *updatedCount = 0;
        if (clearedCount) *clearedCount = 0;
        return false;
    }
    return _projectData->replaceImageCameras(targetImagePaths,
                                             cameras,
                                             updatedCount,
                                             clearedCount,
                                             errorMsg);
}

bool ProjectManager::clearImageCameras(const QStringList &imagePaths,
                                       int    *updatedCount,
                                       QString *errorMsg)
{
    if (!_projectData) {
        if (errorMsg) *errorMsg = QStringLiteral("ProjectData 未初始化");
        if (updatedCount) *updatedCount = 0;
        return false;
    }
    return _projectData->clearImageCameras(imagePaths, updatedCount, errorMsg);
}

QMap<QString, xjw::Camera> ProjectManager::getCamerasForImages(
        const QStringList &images,
        bool *hasCamerasForAll) const
{
    if (hasCamerasForAll) *hasCamerasForAll = true;

    QMap<QString, xjw::Camera> result;
    if (!_projectData)
    {
        if (hasCamerasForAll) *hasCamerasForAll = false;
        return result;
    }

    // 从运行时元数据中建立路径 → 影像元数据索引，再按需解析相机。
    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::gui::project::projectImageMetaByPath(projectFilesMeta(_projectData), true);

    for (const QString &imgPath : images)
    {
        const QString norm = normalizePath(imgPath);
        const QJsonObject imageMeta = imageMetaByPath.value(norm);
        if (imageMeta.isEmpty())
        {
            if (hasCamerasForAll) *hasCamerasForAll = false;
            continue;
        }

        xjw::Camera cam;
        if (!xjw::gui::project::imageCameraFromEntry(imageMeta, &cam))
        {
            if (hasCamerasForAll) *hasCamerasForAll = false;
            continue;
        }

        result.insert(norm, cam);
    }

    // 若有任意请求影像缺少相机，则 hasCamerasForAll 已在循环内置 false
    if (hasCamerasForAll && result.size() != images.size())
        *hasCamerasForAll = false;

    return result;
}

void ProjectManager::startGenerateModelAsync()
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::GenerateModel);
}

void ProjectManager::startGenerateModelAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::GenerateModel, settings);
}

void ProjectManager::startMeshReconstructionAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::MeshReconstruction, settings);
}

void ProjectManager::startTextureMappingAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::TextureMapping, settings);
}

void ProjectManager::startStereoAndPoint2DemAsync(const QStringList &images,
                                                   const QString &outputDir,
                                                   int threads,
                                                   bool genPointCloud,
                                                   double demResolution,
                                                   const QString &demType,
                                                   const QString &t_srs)
{
    _taskDispatcher->startStereoAndPoint2DemAsync(images,
                                                   outputDir,
                                                   threads,
                                                   genPointCloud,
                                                   demResolution,
                                                   demType,
                                                   t_srs);
}

void ProjectManager::startFullDemPipelineAsync(const QStringList &images,
                                               const QString &outputDir,
                                               const QJsonObject &pipelineSettings)
{
    _taskDispatcher->startFullDemPipelineAsync(images, outputDir, pipelineSettings);
}

void ProjectManager::startDemFromDenseCloudAsync(const QString &denseCloudPath,
                                                 const QString &outputDir,
                                                 double demResolution,
                                                 const QString &demType)
{
    _taskDispatcher->startDemFromDenseCloudAsync(denseCloudPath, outputDir, demResolution, demType);
}

void ProjectManager::startMapProjectAsync(const QStringList &images,
                                          const QString &demPath,
                                          const QString &outputPath,
                                          double resolution)
{
    _taskDispatcher->startMapProjectAsync(images,
                                           demPath,
                                           outputPath,
                                           resolution);
}

bool ProjectManager::acceptBundleAdjustPreview(QString *errorMsg)
{
    if (!_hasPendingBaPreview || _pendingBaCameraMeta.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("当前没有可应用的平差预览结果");
        return false;
    }

    const QStringList allImages = _projectData ? _projectData->getAllImages() : QStringList();
    const auto commitResult = commitBundleAdjustPreview(_projectData,
                                                        _pendingBaCameraMeta,
                                                        _pendingBaResult);
    if (!commitResult.success) {
        if (errorMsg) *errorMsg = commitResult.errorMessage;
        return false;
    }
    if (!commitResult.warningMessage.isEmpty()) {
        LOG_WARN(commitResult.warningMessage);
    }

    const QString assetsDir = ProjectIO::projectAssetsDir(currentProjectPath());
    const QString baOutputDir = assetsDir.isEmpty()
        ? QString()
        : QDir(assetsDir).filePath(
              QStringLiteral("aerial_triangulation/ba_refined_%1")
                  .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss"))));
    const auto artifactsResult = finalizeBundleAdjustArtifacts(
        assetsDir,
        _pendingBaResult,
        allImages,
        _pendingBaResult.value(QStringLiteral("output_dir")).toString(),
        QStringLiteral("reconstruction_bundle_adjust"),
        _pendingBaBeforeCameraMeta,
        _pendingBaCameraMeta,
        baOutputDir,
        true);
    if (!artifactsResult.reportWarning.isEmpty())
    {
        LOG_WARN(QStringLiteral("BA: %1").arg(artifactsResult.reportWarning));
    }
    if (artifactsResult.sparseCloudExport.exported)
    {
        replaceTiePointResult(artifactsResult.sparseCloudExport.sparseCloudPath,
                              artifactsResult.sparseCloudExport.pointCount,
                              allImages,
                              artifactsResult.sparseCloudExport.outputDir,
                              artifactsResult.sparseCloudExport.extraRecord);
        LOG_INFO(QStringLiteral("BA 精化点云已写入 AT 结果：%1 个点，路径=%2")
                     .arg(artifactsResult.sparseCloudExport.pointCount)
                     .arg(artifactsResult.sparseCloudExport.sparseCloudPath));
    }
    else if (!artifactsResult.sparseCloudExport.errorMessage.isEmpty())
    {
        LOG_WARN(QStringLiteral("写入 BA 点云 sidecar 失败: %1")
                     .arg(artifactsResult.sparseCloudExport.errorMessage));
    }

    _pendingBaCameraMeta.clear();
    _pendingBaBeforeCameraMeta.clear();
    _pendingBaResult = QJsonObject();
    _hasPendingBaPreview = false;

    QMessageBox::information(_parent,
                             QStringLiteral("光束法平差"),
                             QStringLiteral("已保留本次平差结果，并更新 %1 台相机参数。")
                                 .arg(commitResult.updatedCameraCount));
    return true;
}

void ProjectManager::discardBundleAdjustPreview()
{
    _pendingBaCameraMeta.clear();
    _pendingBaBeforeCameraMeta.clear();
    _pendingBaResult = QJsonObject();
    _hasPendingBaPreview = false;
}

void ProjectManager::applyBundleAdjustForAt(const QString     &assetsDir,
                                            const QStringList &images,
                                            const QString     &outputDir,
                                            const QMap<QString, QJsonObject> &beforeCameras)
{
    bool success = false;
    if (_hasPendingBaPreview && !_pendingBaCameraMeta.isEmpty())
    {
        const auto commitResult = commitBundleAdjustPreview(_projectData,
                                                            _pendingBaCameraMeta,
                                                            _pendingBaResult);
        if (commitResult.success)
        {
            LOG_INFO(
                QStringLiteral("BA(AT): 已更新 %1 台相机参数").arg(commitResult.updatedCameraCount));
            success = true;
        } 
        else 
        {
            LOG_WARN(
                commitResult.errorMessage);
        }
        if (!commitResult.warningMessage.isEmpty())
        {
            LOG_WARN(QStringLiteral("BA(AT): %1").arg(commitResult.warningMessage));
        }
    } 
    else 
    {
        LOG_WARN(
            QStringLiteral("BA(AT): 没有待应用的平差预览结果，可能是相机或匹配点不足"));
    }

    const auto artifactsResult = finalizeBundleAdjustArtifacts(assetsDir,
                                                               _pendingBaResult,
                                                               images,
                                                               outputDir,
                                                               QStringLiteral("workflow_aerial_triangulation"),
                                                               beforeCameras,
                                                               _pendingBaCameraMeta,
                                                               outputDir,
                                                               false);
    if (!artifactsResult.reportWarning.isEmpty())
    {
        LOG_WARN(QStringLiteral("BA(AT): %1").arg(artifactsResult.reportWarning));
    }

    if (artifactsResult.sparseCloudExport.exported)
    {
        replaceTiePointResult(artifactsResult.sparseCloudExport.sparseCloudPath,
                              artifactsResult.sparseCloudExport.pointCount,
                              images,
                              artifactsResult.sparseCloudExport.outputDir,
                              artifactsResult.sparseCloudExport.extraRecord);
    }
    else if (!artifactsResult.sparseCloudExport.errorMessage.isEmpty())
    {
        LOG_WARN(QStringLiteral("BA(AT): 无法导出稀疏点云文件: %1")
                     .arg(artifactsResult.sparseCloudExport.errorMessage));
    }

    // ── 4. 清理预览缓存 ────────────────────────────────────────────────────
    _pendingBaCameraMeta.clear();
    _pendingBaBeforeCameraMeta.clear();
    _pendingBaResult    = QJsonObject();
    _hasPendingBaPreview = false;

    // ── 5. 发出空三完成信号 ────────────────────────────────────────────────
    emit atProgressFinished(success);
}

bool ProjectManager::appendIntersectionResult(const QJsonObject &result, QString *errorMsg)
{
    return _projectData ? _projectData->appendIntersectionResult(result, errorMsg) : false;
}

QJsonArray ProjectManager::intersectionResults() const
{
    return _projectData ? _projectData->getIntersectionResults() : QJsonArray();
}

//==============================================================================
// FileDialogStateManager
//==============================================================================

void ProjectManager::setFileDialogStateManager(FileDialogStateManager *manager)
{
    _fileDialogState = manager;
}

QString ProjectManager::getLastUsedDir(const QString &key) const
{
    if (!_fileDialogState) return QDir::homePath();
    return _fileDialogState->lastDir(key);
}

void ProjectManager::saveLastUsedDir(const QString &key, const QString &dir)
{
    if (_fileDialogState) {
        _fileDialogState->setLastDir(key, dir);
    }
}

void ProjectManager::showWarning(const QString &message, const QString &title) const
{
    // 统一 warning 出口，后续若切换提示组件只需改这里。
    QMessageBox::warning(_parent, title, message);
}

bool ProjectManager::ensureProjectOpen(const QString &message, const QString &title) const
{
    // 将“项目是否打开”校验统一收口，避免重复 if 与文案分散。
    if (_projectData && _projectData->hasProject()) return true;
    showWarning(message, title);
    return false;
}

bool ProjectManager::replaceTiePointResult(const QString &sparseCloudPath,
                                           int sparsePointCount,
                                           const QStringList &selectedImages,
                                           const QString &outputDir,
                                           const QJsonObject &extraRecord)
{
    const auto result = xjw::gui::project::replaceTiePointResult(_projectData,
                                                                 sparseCloudPath,
                                                                 sparsePointCount,
                                                                 selectedImages,
                                                                 outputDir,
                                                                 extraRecord);
    if (!result.success)
    {
        LOG_ERROR(QStringLiteral("替换当前连接点失败: %1").arg(result.errorMessage));
        showWarning(result.errorMessage, QStringLiteral("连接点写入失败"));
        return false;
    }
    if (!result.cleanupWarnings.isEmpty())
    {
        LOG_WARN(QStringLiteral("当前连接点已更新，但旧文件清理失败: %1")
                     .arg(result.cleanupWarnings.join(QStringLiteral("；"))));
    }
    LOG_INFO(QStringLiteral("空三代次已更新为 %1；旧深度图、稠密点云、模型、DEM 和正射结果已失效")
                 .arg(result.reconstructionGenerationId));
    refreshReconstructionQualityReport();
    return true;
}

void ProjectManager::appendObsNetResult(int nodeCount,
                                        int edgeCount,
                                        const QString &algorithmName,
                                        const QJsonObject &extraInfo)
{
    xjw::gui::project::appendObsNetResult(_projectData,
                                          nodeCount,
                                          edgeCount,
                                          algorithmName,
                                          extraInfo);
}

// ── 追加空三（SFM）结果到 aerial_triangulation_results ─────────────────
QJsonArray ProjectManager::getAvailableAtResults() const
{
    return _taskDispatcher->getAvailableAtResults();
}

void ProjectManager::startEstimateDepthMapsAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::EstimateDepthMaps, settings);
}

void ProjectManager::startFuseDepthMapsAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::FuseDepthMaps, settings);
}

// ── 带配置参数的MVS稠密点云生成（PatchMatch 多视图） ─────────────────────
void ProjectManager::startGenerateDenseCloudAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::GenerateDenseCloud, settings);
}



// ── 密集点云后处理（SOR + 体素下采样 + 法向量估计） ─────────────────────────────
void ProjectManager::startDenseCloudRefineAsync(const QJsonObject &settings)
{
    _taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::RefineDenseCloud, settings);
}



// ── 取消正在运行的 MVS 任务 ──────────────────────────────────────────────────
void ProjectManager::cancelMvs()
{
    _taskDispatcher->cancelMvs();
}

// ── 取消正在运行的 AT/SFM 任务 ──────────────────────────────────────────────
void ProjectManager::cancelAt()
{
    if (_atCancelFlag)
    {
        _atCancelFlag->store(true);
        qDebug() << "[AT/BA] 已请求取消";
    }
}

// ── 取消正在运行的照片蒙版生成任务 ───────────────────────────────────────────
void ProjectManager::cancelMaskGeneration()
{
    if (_maskGenerationCancelFlag)
    {
        _maskGenerationCancelFlag->store(true, std::memory_order_relaxed);
        qDebug() << "[Mask] 已请求取消照片蒙版生成";
    }
}
