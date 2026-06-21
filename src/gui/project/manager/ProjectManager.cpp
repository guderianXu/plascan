// 必须在任何Qt头文件之前引入LibTorch相关头，避免 Qt 宏（slots/signals/emit）与 LibTorch 冲突
#include "compat/QtTorchMacroGuard.h"

#include "SuperPoint.h"
#include "FeatureFileIO.h"
#include "SuperGlueMatcher.h"
#include "SuperGlueMatchIO.h"

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

#include "DenseMatchService.h"
#include "DenseMatchConfig.h"
#include "ProjectMetadataOperations.h"
#include "ProjectSfmWorkflow.h"
#include "ProjectSparseWorkflow.h"
#include "ProjectTriangulationService.h"
#include "ProjectResultRecords.h"
#include "ProjectReferenceDatasets.h"
#include "ProjectWorkflowUtils.h"
#include "ProjectWorkflowReports.h"
#include "Logger.h"
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

#include "SFMService.h"

#include <QMessageBox>
#include <QDir>
#include <QFileDialog>
#include <QJsonArray>
#include <QJsonDocument>
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
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <array>
#include <limits>
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

} // namespace

ProjectManager::ProjectManager(ProjectData *projectData, QWidget *parent)
    : QObject(parent)
    , m_parent(parent)
    , m_projectData(projectData)
    , m_reconstructionManager(new ProjectReconstructionManager(this, projectData, parent, this))
    , m_terrainProductsManager(new ProjectTerrainProductsManager(this, projectData, parent, this))
    , m_cameraSetupManager(new ProjectCameraSetupManager(this, projectData, parent, this))
    , m_taskDispatcher(new ProjectTaskDispatcher(m_cameraSetupManager,
                                                 m_terrainProductsManager,
                                                 m_reconstructionManager,
                                                 this))
    , m_uiCommands(new ProjectUiCommands(projectData, parent))
{
    m_uiCommands->setDirectoryAccessors(
        [this](const QString &key) { return getLastUsedDir(key); },
        [this](const QString &key, const QString &dir) { saveLastUsedDir(key, dir); });

    // 连接ProjectData信号
    if (m_projectData)
    {
        connect(m_projectData, &ProjectData::projectOpened,
                this, &ProjectManager::projectOpened);
        connect(m_projectData, &ProjectData::projectSaved,
                this, &ProjectManager::projectSaved);
        connect(m_projectData, &ProjectData::projectClosed,
                this, &ProjectManager::projectClosed);
        connect(m_projectData, &ProjectData::metadataChanged,
                this, &ProjectManager::projectMetadataChanged);
        connect(m_projectData, &ProjectData::dirtyStateChanged,
                this, &ProjectManager::metadataDirtyChanged);
    }

        connect(m_reconstructionManager, &ProjectReconstructionManager::mvsProgressChanged,
            this, &ProjectManager::mvsProgressChanged);
        connect(m_reconstructionManager, &ProjectReconstructionManager::mvsProgressFinished,
            this, &ProjectManager::mvsProgressFinished);
        connect(m_reconstructionManager, &ProjectReconstructionManager::meshProgressChanged,
            this, &ProjectManager::meshProgressChanged);
        connect(m_reconstructionManager, &ProjectReconstructionManager::meshProgressFinished,
            this, &ProjectManager::meshProgressFinished);
        connect(m_reconstructionManager, &ProjectReconstructionManager::atProgressChanged,
            this, &ProjectManager::atProgressChanged);
        connect(m_reconstructionManager, &ProjectReconstructionManager::atProgressFinished,
            this, &ProjectManager::atProgressFinished);
        connect(m_terrainProductsManager, &ProjectTerrainProductsManager::demPipelineProgressChanged,
            this, &ProjectManager::demPipelineProgressChanged);
        connect(m_terrainProductsManager, &ProjectTerrainProductsManager::demPipelineFinished,
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
    if (m_uiCommands && m_uiCommands->createNewProject(&plascanPath))
    {
        emit projectCreated(plascanPath);
        LOG_INFO(QStringLiteral("项目已创建: %1").arg(plascanPath));
    }
}

void ProjectManager::openProject()
{
    QString plascanPath;
    if (m_uiCommands && m_uiCommands->openProjectByDialog(&plascanPath))
    {
        LOG_INFO(QStringLiteral("项目已打开: %1").arg(plascanPath));
    }
}

void ProjectManager::openProjectFromPath(const QString &plascanPath)
{
    if (m_uiCommands && m_uiCommands->openProjectFromPath(plascanPath))
    {
        LOG_INFO(QStringLiteral("项目已打开: %1").arg(plascanPath));
    }
}

void ProjectManager::saveProject()
{
    if (!m_projectData)
    {
        return;
    }

    emit saveStarted();
    const bool success = m_uiCommands && m_uiCommands->saveProject();

    emit saveFinished(success);
}

void ProjectManager::closeProject()
{
    if (m_uiCommands)
    {
        m_uiCommands->closeProject();
    }
}

//==============================================================================
// 资源管理
//==============================================================================

void ProjectManager::addPhoto()
{
    if (m_uiCommands)
    {
        (void)m_uiCommands->addPhoto();
    }
}

void ProjectManager::addFolder()
{
    if (m_uiCommands)
    {
        (void)m_uiCommands->addFolder();
    }
}

bool ProjectManager::importCameraForImage(const QString &imagePath)
{
    return m_taskDispatcher->importCameraForImage(imagePath);
}

void ProjectManager::startTriangulationAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::Triangulation, settings);
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
    const QJsonArray pairs = settings.value(QStringLiteral("match_pairs")).toArray();
    const QString outputDir  = settings.value(QStringLiteral("output_dir")).toString();
    const int algo           = settings.value(QStringLiteral("algorithm")).toInt(2);
    const int costFunc       = settings.value(QStringLiteral("cost_func")).toInt(3);
    const int minDisp        = settings.value(QStringLiteral("min_disparity")).toInt(0);
    const int maxDisp        = settings.value(QStringLiteral("max_disparity")).toInt(256);
    const int kernelW        = settings.value(QStringLiteral("kernel_w")).toInt(15);
    const int kernelH        = settings.value(QStringLiteral("kernel_h")).toInt(15);
    const bool useCuda       = settings.value(QStringLiteral("use_cuda")).toBool(true);
    const int cudaDevice     = settings.value(QStringLiteral("cuda_device")).toInt(0);
    const int subpixel       = settings.value(QStringLiteral("subpixel_mode")).toInt(1);
    const int p1             = settings.value(QStringLiteral("p1")).toInt(8);
    const int p2             = settings.value(QStringLiteral("p2")).toInt(32);
    const int directions     = settings.value(QStringLiteral("directions")).toInt(8);
    const int pyramid        = settings.value(QStringLiteral("pyramid")).toInt(2);
    const double lrThreshold = settings.value(QStringLiteral("lr_threshold")).toDouble(1.0);
    const int medianFilter   = settings.value(QStringLiteral("median_filter")).toInt(3);
    const int numThreads     = settings.value(QStringLiteral("threads")).toInt(4);

    LOG_INFO(QStringLiteral("密集匹配: %1 个匹配对, 算法=%2 代价=%3 CUDA=%4 视差=[%5,%6] 线程=%7")
        .arg(pairs.size()).arg(algo).arg(costFunc).arg(useCuda).arg(minDisp).arg(maxDisp)
        .arg(numThreads));

    QDir().mkpath(outputDir);

    int completed = 0;
    for (const QJsonValue &val : pairs)
    {
        if (cancelFlag && cancelFlag->load())
        {
            LOG_INFO(QStringLiteral("密集匹配已请求取消，停止处理剩余匹配对"));
            break;
        }

        const QJsonObject pair = val.toObject();
        const QString imgA = pair.value(QStringLiteral("imgA")).toString();
        const QString imgB = pair.value(QStringLiteral("imgB")).toString();

        QFileInfo fiA(imgA), fiB(imgB);
        LOG_INFO(QStringLiteral("[密集匹配 %1/%2] %3 <-> %4")
            .arg(completed + 1).arg(pairs.size())
            .arg(fiA.fileName()).arg(fiB.fileName()));

        xjw::dense_match::DenseMatchConfig cfg;
        cfg.algorithm       = static_cast<xjw::dense_match::StereoAlgorithm>(algo);
        cfg.costFunc        = static_cast<xjw::dense_match::CostFunction>(costFunc);
        cfg.subpixel        = static_cast<xjw::dense_match::SubpixelMode>(subpixel);
        cfg.minDisparity    = minDisp;
        cfg.maxDisparity    = maxDisp;
        cfg.corrKernelW     = kernelW;
        cfg.corrKernelH     = kernelH;
        cfg.useCuda         = useCuda;
        cfg.cudaDevice      = cudaDevice;
        cfg.p1              = p1;
        cfg.p2              = p2;
        cfg.sgmDirections   = directions;
        cfg.pyramidLevels   = pyramid;
        cfg.lrCheckThreshold = static_cast<float>(lrThreshold);
        cfg.medianFilterSize = medianFilter;
        cfg.enableLRCheck   = lrThreshold > 0.0;
        cfg.numThreads      = numThreads;
        cfg.leftImagePath   = imgA.toStdString();
        cfg.rightImagePath  = imgB.toStdString();

        cfg.outputDisparityPath = QStringLiteral("%1/%2__%3_disp.tif")
            .arg(outputDir)
            .arg(fiA.completeBaseName())
            .arg(fiB.completeBaseName()).toStdString();

        // Step 1: Load
        if (progress) progress->store(completed * 5 + 1);

        xjw::dense_match::DenseMatchService service(cfg);
        auto result = service.process();
        if (cancelFlag && cancelFlag->load())
        {
            LOG_INFO(QStringLiteral("密集匹配已请求取消，跳过当前结果保存"));
            break;
        }

        if (!result.disparity.empty())
        {
            // Step 5: Save
            if (progress) progress->store(completed * 5 + 5);

            xjw::dense_match::DenseMatchService::saveDisparity(
                result, cfg.outputDisparityPath);
            LOG_INFO(QStringLiteral("[密集匹配 %1/%2] 完成 -> %3")
                .arg(completed + 1).arg(pairs.size())
                .arg(QString::fromStdString(cfg.outputDisparityPath)));
        }
        else
        {
            LOG_INFO(QStringLiteral("[密集匹配 %1/%2] 失败: 视差图为空")
                .arg(completed + 1).arg(pairs.size()));
        }

        ++completed;
        if (progress)
            progress->store(completed * 5);
    }
}

void ProjectManager::startSparseCloudOutlierRemovalAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::SparseOutlierRemoval, settings);
}

void ProjectManager::startSparseCloudLocalOptimAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::SparseLocalOptimization, settings);
}

void ProjectManager::startSparseCloudRefineAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::SparseRefine, settings);
}

bool ProjectManager::importCamerasByFilenameBatch()
{
    return m_taskDispatcher->importCamerasByFilenameBatch();
}

bool ProjectManager::initializeCamerasFromExifOrDefault(const QJsonObject &settings)
{
    return m_taskDispatcher->initializeCamerasFromExifOrDefault(settings);
}

bool ProjectManager::initializeCamerasFromIntrinsics(const QJsonObject &settings)
{
    return m_taskDispatcher->initializeCamerasFromIntrinsics(settings);
}

bool ProjectManager::initializeCameraPosesWithSFM(const QJsonObject &settings)
{
    return m_taskDispatcher->initializeCameraPosesWithSFM(settings);
}

void ProjectManager::removeResource(const QString &resourcePath)
{
    if (m_projectData)
    {
        m_projectData->removeResource(resourcePath);
    }
}

void ProjectManager::removeResources(const QStringList &resourcePaths)
{
    if (m_projectData)
    {
        m_projectData->removeResources(resourcePaths);
    }
}

void ProjectManager::importReferenceDataset()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再导入参考 DEM/LiDAR。")))
    {
        return;
    }

    const QString selected = QFileDialog::getOpenFileName(
        m_parent,
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
    return xjw::gui::project::registerReferenceDataset(m_projectData, path, type, role, errorMsg);
}

void ProjectManager::runReferenceQualityCheck()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再执行点云/DEM 精度检查。")))
    {
        return;
    }

    const auto result = xjw::gui::project::writeReferenceDatasetQualityReport(m_projectData);
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
    QMessageBox::information(m_parent,
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

    const auto result = xjw::gui::project::writeReferenceTerrainPriorPreflightReport(m_projectData);
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
        QMessageBox::information(m_parent,
                                 QStringLiteral("参考地形约束重新平差"),
                                 QStringLiteral("前置检查未通过：%1。\n请先导入 role=ba_prior 的参考 DEM/LiDAR，并完成正式空三。\nJSON: %2\nCSV: %3")
                                     .arg(status, result.jsonPath, result.csvPath));
        return;
    }

    const QJsonObject meta = m_projectData->metadata();
    const QString demPriorPath = firstReferenceDemPriorPath(meta);
    const QString laserPriorPath = firstReferenceLaserPriorPath(meta);
    if (demPriorPath.isEmpty() && laserPriorPath.isEmpty())
    {
        QMessageBox::information(m_parent,
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
        QMessageBox::warning(m_parent,
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
        m_parent,
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
    if (!m_projectData || resourcePaths.isEmpty())
    {
        return;
    }
    if (section == QStringLiteral("照片"))
    {
        QMessageBox::information(m_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("照片分组不支持删除数据，请使用移除引用。"));
        return;
    }

    const int selectedCount = resourcePaths.size();
    const QMessageBox::StandardButton confirm = QMessageBox::question(
        m_parent,
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

    const auto cleanupResult = xjw::gui::project::ProjectResourceCleanupService::cleanupGeneratedData(m_projectData,
                                                                                                       section,
                                                                                                       resourcePaths);
    if (cleanupResult.unsupportedSection)
    {
        QMessageBox::warning(m_parent,
                             QStringLiteral("删除数据"),
                             QStringLiteral("当前分组暂不支持删除数据：%1").arg(section));
        return;
    }

    if (!cleanupResult.success && !cleanupResult.errorMessage.isEmpty())
    {
        QMessageBox::warning(m_parent,
                             QStringLiteral("删除数据"),
                             QStringLiteral("删除失败：%1").arg(cleanupResult.errorMessage));
        return;
    }

    if (cleanupResult.noMatchedRecords)
    {
        QMessageBox::information(m_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("未找到可删除的%1数据记录。").arg(section));
        return;
    }

    if (cleanupResult.failedPaths.isEmpty())
    {
        QMessageBox::information(m_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("已删除 %1 项%2数据。").arg(cleanupResult.removedCount).arg(section));
    }
    else
    {
        QMessageBox::warning(m_parent,
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
    if (m_projectData && !m_projectData->packResource(resourcePath, &err))
    {
        QMessageBox::warning(m_parent, QStringLiteral("提示"), err);
    }
}

// Bundle adjust related APIs removed (core removed).

//==============================================================================
// 设置管理 (委托给ProjectData)
//==============================================================================

QJsonObject ProjectManager::loadUiSettings() const
{
    return m_projectData ? m_projectData->loadUiSettings() : QJsonObject();
}

//==============================================================================
// 查询接口 (委托给ProjectData)
//==============================================================================

bool ProjectManager::isDirty() const
{
    return m_projectData ? m_projectData->isDirty() : false;
}

QString ProjectManager::currentProjectPath() const
{
    return m_projectData ? m_projectData->currentProjectPath() : QString();
}

QJsonObject ProjectManager::currentMeta() const
{
    return m_projectData ? m_projectData->metadata() : QJsonObject();
}

QJsonObject ProjectManager::coreProjectMeta() const
{
    return m_projectData ? m_projectData->coreFilesMeta() : QJsonObject();
}

QStringList ProjectManager::getImagesByCategory(const QString &category) const
{
    return m_projectData ? m_projectData->getImagesByCategory(category) : QStringList();
}

QStringList ProjectManager::getAllImages() const
{
    return m_projectData ? m_projectData->getAllImages() : QStringList();
}

QString ProjectManager::findMatchFileForPair(const QString &imgA, const QString &imgB) const
{
    return m_projectData ? m_projectData->findMatchFile(imgA, imgB) : QString();
}

bool ProjectManager::hasTemporaryMeta() const
{
    return m_projectData ? m_projectData->hasTemporaryMetadata() : false;
}

void ProjectManager::discardTemporaryMeta()
{
    if (m_projectData) {
        m_projectData->clearTemporaryMetadata();
    }
}

void ProjectManager::writeMetadataToTempAsync(const QJsonObject &meta, bool markDirty)
{
    if (m_projectData) {
        m_projectData->updateMetadata(meta, markDirty);
        m_projectData->saveTemporaryMetadata();
    }
}

void ProjectManager::refreshReconstructionQualityReport()
{
    if (!m_projectData || !m_projectData->hasProject())
    {
        return;
    }

    const auto reportResult =
        xjw::gui::project::writeReconstructionQualityProjectReport(m_projectData);
    if (!reportResult.saved && !reportResult.errorMessage.isEmpty())
    {
        LOG_WARN(QStringLiteral("重建质量报告刷新失败: %1").arg(reportResult.errorMessage));
    }
}

void ProjectManager::appendIpfindResult(const QString &input, const QString &output, const QJsonObject &settings)
{
    if (m_projectData) {
        m_projectData->appendIpfindResult(input, output, settings);
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

void ProjectManager::appendIpmatchResult(const QStringList &outputs, const QJsonObject &settings)
{
    if (m_projectData) {
        m_projectData->appendIpmatchResult(outputs, settings);
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
        QMessageBox::warning(m_parent, QStringLiteral("提示"), QStringLiteral("至少需要选择两张影像"));
        return;
    }
    if (outputDir.trimmed().isEmpty()) {
        QMessageBox::warning(m_parent, QStringLiteral("提示"), QStringLiteral("请指定输出目录"));
        return;
    }

    const QString outDir = QDir::cleanPath(outputDir);

    // ── 只获取核心数据（影像列表/相机，无结果数组，速度极快）──────────────
    // 结果数据（ipmatch_results）在后台线程直接读取，避免 UI 阻塞
    const QJsonObject coreData   = m_projectData->coreFilesMeta();
    const QString     plascanPath = m_projectData->currentProjectPath();

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
    opts.baOpt.progressCallback =
        [self = this, cancelFlag](int currentIteration, int maxIterations, double avgRms, int validPoints) -> bool
        {
            if (cancelFlag->load(std::memory_order_relaxed))
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
                self,
                [self, cancelFlag, stage, percent]()
                {
                    if (self->m_atCancelFlag != cancelFlag ||
                        cancelFlag->load(std::memory_order_relaxed))
                    {
                        return;
                    }
                    emit self->atProgressChanged(stage, percent);
                },
                Qt::QueuedConnection);
            return true;
        };

    emit atProgressChanged(QStringLiteral("光束法平差准备中..."), 1);

    LOG_INFO(QStringLiteral("BA: 特征/匹配准备完毕，启动光束法平差"));

    auto *self = this;

(void)QtConcurrent::run(
        [self, coreData, plascanPath, images, minMatches,
         cancelFlag, opts = std::move(opts), isDryRun = dryRun]() mutable
    {
        auto finishTask = [self, cancelFlag](bool success)
        {
            QMetaObject::invokeMethod(
                self,
                [self, cancelFlag, success]()
                {
                    if (self->m_atCancelFlag != cancelFlag)
                    {
                        return;
                    }
                    self->m_atCancelFlag.reset();
                    emit self->atProgressFinished(success);
                },
                Qt::QueuedConnection);
        };

        QMetaObject::invokeMethod(
            self,
            [self, cancelFlag]()
            {
                if (self->m_atCancelFlag != cancelFlag ||
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
            QMetaObject::invokeMethod(self,
                [self, cancelFlag]()
                {
                    if (self->m_atCancelFlag == cancelFlag)
                    {
                        emit self->bundleAdjustPreviewReady(QJsonObject());
                    }
                },
                Qt::QueuedConnection);
            finishTask(false);
            return;
        }

        if (executionResult.buildStatus != xjw::gui::project::BaInputBuildStatus::Ok) {
            QMetaObject::invokeMethod(self,
                [self, cancelFlag, buildStatus = executionResult.buildStatus]() {
                    if (self->m_atCancelFlag != cancelFlag)
                    {
                        return;
                    }
                    const QString msg =
                        buildStatus == xjw::gui::project::BaInputBuildStatus::NotEnoughCameras
                        ? QStringLiteral("所选影像中可用相机参数不足（至少需要两台相机）")
                        : QStringLiteral("未找到可用于光束法平差的匹配点（请检查选中影像是否已有匹配结果）");
                    QMessageBox::warning(self->m_parent, QStringLiteral("提示"), msg);
                    emit self->bundleAdjustPreviewReady(QJsonObject());
                    self->m_atCancelFlag.reset();
                    emit self->atProgressFinished(false);
                },
                Qt::QueuedConnection);
            return;
        }

        LOG_INFO(
            QStringLiteral("BA: 启动后台平差计算 (%1 台相机, %2 条轨迹)...")
                .arg(executionResult.serviceResult.pendingCamUpdates.size())
                .arg(executionResult.serviceResult.resultJson.value(QStringLiteral("track_count")).toInt()));

        QMetaObject::invokeMethod(self,
            [self,
             cancelFlag,
             baResult = executionResult.serviceResult,
             beforeCamMeta = executionResult.beforeCamMeta,
             isDryRun]()
            {
                if (self->m_atCancelFlag != cancelFlag)
                {
                    return;
                }
                if (cancelFlag->load(std::memory_order_relaxed))
                {
                    emit self->bundleAdjustPreviewReady(QJsonObject());
                    self->m_atCancelFlag.reset();
                    emit self->atProgressFinished(false);
                    return;
                }

                emit self->atProgressChanged(QStringLiteral("光束法平差整理结果..."), 95);

                if (!baResult.success && !isDryRun) {
                    QMessageBox::warning(self->m_parent,
                        QStringLiteral("平差提示"),
                        QStringLiteral("光束法平差执行出现问题：%1").arg(baResult.errorMessage));
                }
                self->m_pendingBaBeforeCameraMeta = beforeCamMeta;
                self->m_pendingBaCameraMeta  = baResult.pendingCamUpdates;
                self->m_pendingBaResult      = baResult.resultJson;
                self->m_hasPendingBaPreview  = !self->m_pendingBaCameraMeta.isEmpty();
                emit self->bundleAdjustPreviewReady(baResult.resultJson);
                self->m_atCancelFlag.reset();
                emit self->atProgressFinished(baResult.success);
            },
            Qt::QueuedConnection);
    });
}

// ==============================================================================
// 空中三角测量相关
// ==============================================================================
bool ProjectManager::setImageCameras(const QMap<QString, QJsonObject> &cameras,
                                     int    *updatedCount,
                                     QString *errorMsg)
{
    // 直接委托给数据层——ProjectManager 不持有算法逻辑，仅转发
    if (!m_projectData)
    {
        if (errorMsg) *errorMsg = QStringLiteral("ProjectData 未初始化");
        if (updatedCount) *updatedCount = 0;
        return false;
    }
    return m_projectData->setImageCameras(cameras, updatedCount, errorMsg);
}

bool ProjectManager::clearImageCameras(const QStringList &imagePaths,
                                       int    *updatedCount,
                                       QString *errorMsg)
{
    if (!m_projectData) {
        if (errorMsg) *errorMsg = QStringLiteral("ProjectData 未初始化");
        if (updatedCount) *updatedCount = 0;
        return false;
    }
    return m_projectData->clearImageCameras(imagePaths, updatedCount, errorMsg);
}

QMap<QString, xjw::Camera> ProjectManager::getCamerasForImages(
        const QStringList &images,
        bool *hasCamerasForAll) const
{
    if (hasCamerasForAll) *hasCamerasForAll = true;

    QMap<QString, xjw::Camera> result;
    if (!m_projectData)
    {
        if (hasCamerasForAll) *hasCamerasForAll = false;
        return result;
    }

    // 从运行时元数据中建立路径 → 影像元数据索引，再按需解析相机。
    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::gui::project::projectImageMetaByPath(projectFilesMeta(m_projectData), true);

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
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::GenerateModel);
}

void ProjectManager::startMeshReconstructionAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::MeshReconstruction, settings);
}

void ProjectManager::startTextureMappingAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::TextureMapping, settings);
}

void ProjectManager::startStereoAndPoint2DemAsync(const QStringList &images,
                                                   const QString &outputDir,
                                                   int threads,
                                                   bool genPointCloud,
                                                   double demResolution,
                                                   const QString &demType,
                                                   const QString &t_srs)
{
    m_taskDispatcher->startStereoAndPoint2DemAsync(images,
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
    m_taskDispatcher->startFullDemPipelineAsync(images, outputDir, pipelineSettings);
}

void ProjectManager::startDemFromDenseCloudAsync(const QString &denseCloudPath,
                                                 const QString &outputDir,
                                                 double demResolution,
                                                 const QString &demType)
{
    m_taskDispatcher->startDemFromDenseCloudAsync(denseCloudPath, outputDir, demResolution, demType);
}

void ProjectManager::startMapProjectAsync(const QStringList &images,
                                          const QString &demPath,
                                          const QString &outputPath,
                                          double resolution)
{
    m_taskDispatcher->startMapProjectAsync(images,
                                           demPath,
                                           outputPath,
                                           resolution);
}

bool ProjectManager::acceptBundleAdjustPreview(QString *errorMsg)
{
    if (!m_hasPendingBaPreview || m_pendingBaCameraMeta.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("当前没有可应用的平差预览结果");
        return false;
    }

    const QStringList allImages = m_projectData ? m_projectData->getAllImages() : QStringList();
    const auto commitResult = commitBundleAdjustPreview(m_projectData,
                                                        m_pendingBaCameraMeta,
                                                        m_pendingBaResult);
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
        m_pendingBaResult,
        allImages,
        m_pendingBaResult.value(QStringLiteral("output_dir")).toString(),
        QStringLiteral("reconstruction_bundle_adjust"),
        m_pendingBaBeforeCameraMeta,
        m_pendingBaCameraMeta,
        baOutputDir,
        true);
    if (!artifactsResult.reportWarning.isEmpty())
    {
        LOG_WARN(QStringLiteral("BA: %1").arg(artifactsResult.reportWarning));
    }
    if (artifactsResult.sparseCloudExport.exported)
    {
        appendAtResult(artifactsResult.sparseCloudExport.sparseCloudPath,
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

    m_pendingBaCameraMeta.clear();
    m_pendingBaBeforeCameraMeta.clear();
    m_pendingBaResult = QJsonObject();
    m_hasPendingBaPreview = false;

    QMessageBox::information(m_parent,
                             QStringLiteral("光束法平差"),
                             QStringLiteral("已保留本次平差结果，并更新 %1 台相机参数。")
                                 .arg(commitResult.updatedCameraCount));
    return true;
}

void ProjectManager::discardBundleAdjustPreview()
{
    m_pendingBaCameraMeta.clear();
    m_pendingBaBeforeCameraMeta.clear();
    m_pendingBaResult = QJsonObject();
    m_hasPendingBaPreview = false;
}

void ProjectManager::applyBundleAdjustForAt(const QString     &assetsDir,
                                            const QStringList &images,
                                            const QString     &outputDir,
                                            const QMap<QString, QJsonObject> &beforeCameras)
{
    bool success = false;
    if (m_hasPendingBaPreview && !m_pendingBaCameraMeta.isEmpty()) 
    {
        const auto commitResult = commitBundleAdjustPreview(m_projectData,
                                                            m_pendingBaCameraMeta,
                                                            m_pendingBaResult);
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
                                                               m_pendingBaResult,
                                                               images,
                                                               outputDir,
                                                               QStringLiteral("workflow_aerial_triangulation"),
                                                               beforeCameras,
                                                               m_pendingBaCameraMeta,
                                                               outputDir,
                                                               false);
    if (!artifactsResult.reportWarning.isEmpty())
    {
        LOG_WARN(QStringLiteral("BA(AT): %1").arg(artifactsResult.reportWarning));
    }

    if (artifactsResult.sparseCloudExport.exported)
    {
        appendAtResult(artifactsResult.sparseCloudExport.sparseCloudPath,
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
    m_pendingBaCameraMeta.clear();
    m_pendingBaBeforeCameraMeta.clear();
    m_pendingBaResult    = QJsonObject();
    m_hasPendingBaPreview = false;

    // ── 5. 发出空三完成信号 ────────────────────────────────────────────────
    emit atProgressFinished(success);
}

bool ProjectManager::appendIntersectionResult(const QJsonObject &result, QString *errorMsg)
{
    return m_projectData ? m_projectData->appendIntersectionResult(result, errorMsg) : false;
}

QJsonArray ProjectManager::intersectionResults() const
{
    return m_projectData ? m_projectData->getIntersectionResults() : QJsonArray();
}

//==============================================================================
// FileDialogStateManager
//==============================================================================

void ProjectManager::setFileDialogStateManager(FileDialogStateManager *manager)
{
    m_fileDialogState = manager;
}

QString ProjectManager::getLastUsedDir(const QString &key) const
{
    if (!m_fileDialogState) return QDir::homePath();
    return m_fileDialogState->lastDir(key);
}

void ProjectManager::saveLastUsedDir(const QString &key, const QString &dir)
{
    if (m_fileDialogState) {
        m_fileDialogState->setLastDir(key, dir);
    }
}

void ProjectManager::showWarning(const QString &message, const QString &title) const
{
    // 统一 warning 出口，后续若切换提示组件只需改这里。
    QMessageBox::warning(m_parent, title, message);
}

bool ProjectManager::ensureProjectOpen(const QString &message, const QString &title) const
{
    // 将“项目是否打开”校验统一收口，避免重复 if 与文案分散。
    if (m_projectData && m_projectData->hasProject()) return true;
    showWarning(message, title);
    return false;
}

void ProjectManager::appendAtResult(const QString &sparseCloudPath,
                                    int sparsePointCount,
                                    const QStringList &selectedImages,
                                    const QString &outputDir,
                                    const QJsonObject &extraRecord,
                                    int replaceIndex)
{
    xjw::gui::project::appendAtResult(m_projectData,
                                      sparseCloudPath,
                                      sparsePointCount,
                                      selectedImages,
                                      outputDir,
                                      extraRecord,
                                      replaceIndex);
    refreshReconstructionQualityReport();
}

void ProjectManager::appendObsNetResult(int nodeCount,
                                        int edgeCount,
                                        const QString &algorithmName,
                                        const QJsonObject &extraInfo)
{
    xjw::gui::project::appendObsNetResult(m_projectData,
                                          nodeCount,
                                          edgeCount,
                                          algorithmName,
                                          extraInfo);
}

// ── 追加空三（SFM）结果到 aerial_triangulation_results ─────────────────
QJsonArray ProjectManager::getAvailableAtResults() const
{
    return m_taskDispatcher->getAvailableAtResults();
}

void ProjectManager::startEstimateDepthMapsAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::EstimateDepthMaps, settings);
}

void ProjectManager::startFuseDepthMapsAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::FuseDepthMaps, settings);
}

// ── 带配置参数的MVS稠密点云生成（PatchMatch 多视图） ─────────────────────
void ProjectManager::startGenerateDenseCloudAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::GenerateDenseCloud, settings);
}



// ── 密集点云后处理（SOR + 体素下采样 + 法向量估计） ─────────────────────────────
void ProjectManager::startDenseCloudRefineAsync(const QJsonObject &settings)
{
    m_taskDispatcher->startReconstructionTask(ProjectReconstructionManager::Task::RefineDenseCloud, settings);
}



// ── 取消正在运行的 MVS 任务 ──────────────────────────────────────────────────
void ProjectManager::cancelMvs()
{
    m_taskDispatcher->cancelMvs();
}

// ── 取消正在运行的 AT/SFM 任务 ──────────────────────────────────────────────
void ProjectManager::cancelAt()
{
    if (m_atCancelFlag)
    {
        m_atCancelFlag->store(true);
        qDebug() << "[AT/BA] 已请求取消";
    }
}
