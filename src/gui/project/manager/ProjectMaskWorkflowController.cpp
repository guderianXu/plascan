#include "ProjectMaskWorkflowController.h"

#include "image/GenerateMaskDialog.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "MaskGenerator.h"
#include "ProjectData.h"
#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"
#include "ProjectMetadataOperations.h"
#include "ProjectOpenGuard.h"
#include "u2net/U2NetMaskGenerator.h"
#include "io/PathIO.h"
#include "model/ModelFileResolver.h"
#include "model/U2NetModelCatalog.h"

#include "OpenCvCompat.h"
#include <opencv2/imgcodecs.hpp>

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QMessageBox>
#include <QPointer>
#include <QSet>
#include <QStandardPaths>

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>

namespace
{

    struct GenerateMaskResult
    {
        QMap<QString, QJsonObject> recordsByImage;
        QStringList generatedImages;
        QStringList errors;
        QString inferenceBackend;
        QString inferenceDevice;
        QString inferencePrecision;
        QString inferenceEnvironment;
        QString modelSha256;
        QString deviceFallbackReason;
        bool engineReused = false;
        bool cancelled = false;
    };

    xjw::mask::MaskGenerationOptions generationOptions(const QJsonObject& settings)
    {
        xjw::mask::MaskGenerationOptions options;
        const QString method = settings.value(QStringLiteral("method")).toString(QStringLiteral("black_background"));
        options.method = method == QLatin1String("threshold") ? xjw::mask::MaskGenerationMethod::Threshold
                                                              : xjw::mask::MaskGenerationMethod::BlackBackground;
        options.threshold = settings.value(QStringLiteral("auto_threshold")).toBool(true)
                                ? -1.0
                                : settings.value(QStringLiteral("threshold")).toDouble(3.0);
        options.morphologyRadius = settings.value(QStringLiteral("morphology_radius")).toInt(2);
        options.minComponentArea = settings.value(QStringLiteral("min_component_area")).toInt(64);
        options.keepLargestComponent = true;
        return options;
    }

    xjw::mask::MaskOperation maskOperation(const QJsonObject& settings)
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

    QStringList maskTargets(const QJsonObject& settings, const QStringList& allImages)
    {
        const QString scope = settings.value(QStringLiteral("scope")).toString(QStringLiteral("selected_images"));
        if (scope == QLatin1String("all_images"))
        {
            return allImages;
        }
        if (scope == QLatin1String("current_image"))
        {
            const QString current = settings.value(QStringLiteral("current_image")).toString().trimmed();
            return current.isEmpty() ? QStringList{} : QStringList{current};
        }

        QStringList selected;
        for (const QJsonValue& value : settings.value(QStringLiteral("selected_images")).toArray())
        {
            const QString path = value.toString().trimmed();
            if (!path.isEmpty())
            {
                selected.push_back(path);
            }
        }
        return selected.isEmpty() ? allImages : selected;
    }

    std::optional<xjw::mask::U2NetMaskGeneratorConfig> u2netConfig(const QJsonObject& settings, QString* error)
    {
        const xjw::common::model::ModelFileResolver resolver;
        const auto status = xjw::common::model::u2netModelStatus(resolver);
        if (!status.isInstalled)
        {
            if (error)
            {
                *error = QStringLiteral("未找到 U2Net ONNX 模型：U2Net_v1.onnx。"
                                        "请放到 PLASCAN_MODEL_DIR 或 resources/models。");
            }
            return std::nullopt;
        }

        xjw::mask::U2NetMaskGeneratorConfig config;
        const QByteArray modelPath = QDir::toNativeSeparators(status.modelPath).toUtf8();
        config.modelPath = std::string(modelPath.constData(), static_cast<std::size_t>(modelPath.size()));
        QString backend_token = settings.value(QStringLiteral("u2net_backend")).toString().trimmed();
        if (backend_token.isEmpty())
        {
            backend_token = settings.value(QStringLiteral("u2net_device")).toString(QStringLiteral("auto")).trimmed();
        }
        if (backend_token == QLatin1String("cuda"))
        {
            backend_token = QStringLiteral("tensorrt");
        }
        const auto backend = xjw::mask::parseU2NetBackendType(backend_token.toStdString());
        config.backend = backend.value_or(xjw::mask::U2NetBackendType::Auto);
        config.allowDeviceFallback = settings.value(QStringLiteral("u2net_allow_fallback")).toBool(false);
        config.inputSize = xjw::mask::kU2NetModelInputSize;
        config.foregroundThreshold = static_cast<float>(
            std::clamp(settings.value(QStringLiteral("u2net_mask_threshold")).toDouble(0.5), 0.01, 0.99));
        config.morphologyRadius = 1;
        config.minComponentArea = 64;
        config.keepLargestComponent = true;
        config.preferFp16 = true;
        config.engineCacheDirectory = QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
                                          .filePath(QStringLiteral("models/u2net/engines"))
                                          .toStdString();
        return config;
    }

} // namespace

ProjectMaskWorkflowController::ProjectMaskWorkflowController(ProjectData* projectData,
                                                             QWidget* parentWidget,
                                                             QObject* parent)
    : QObject(parent), _projectData(projectData), _parentWidget(parentWidget)
{
    if (_projectData)
    {
        connect(_projectData, &ProjectData::projectOpened, this, [this](const QString&) { cancelActiveTask(); });
        connect(_projectData, &ProjectData::projectClosed, this, &ProjectMaskWorkflowController::cancelActiveTask);
        connect(_projectData,
                &ProjectData::activeChunkChanged,
                this,
                [this](const QString&, const QString&, int)
                {
                    if (_running)
                    {
                        cancelActiveTask();
                    }
                });
    }
}

void ProjectMaskWorkflowController::setActiveImagePath(const QString& imagePath)
{
    _activeImagePath = imagePath;
}

void ProjectMaskWorkflowController::openDialog()
{
    openDialogForImages(_projectData ? _projectData->getAllImages() : QStringList{});
}

void ProjectMaskWorkflowController::openDialogForImages(const QStringList& requestedImages)
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, QStringLiteral("请先打开项目，再生成照片蒙版。")))
    {
        return;
    }
    if (_running)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("生成蒙版"),
                             QStringLiteral("已有照片蒙版生成任务正在运行，请等待完成或取消后再试。"));
        return;
    }

    const QStringList allImages = _projectData->getAllImages();
    if (allImages.isEmpty())
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("生成蒙版"), QStringLiteral("当前项目没有照片。"));
        return;
    }

    const QString projectPath = _projectData->currentProjectPath();
    const QString chunkId = _projectData->activeChunkId();
    QHash<QString, QString> projectImages;
    QStringList resolvedImages;
    for (const QString& path : allImages)
    {
        const QString resolved = xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, path);
        const QString key = xjw::common::project::normalizePath(resolved);
        if (!key.isEmpty() && !projectImages.contains(key))
        {
            projectImages.insert(key, resolved);
            resolvedImages.push_back(resolved);
        }
    }

    QStringList selectedImages;
    QSet<QString> seen;
    for (const QString& path : requestedImages)
    {
        const QString resolved = xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, path);
        const QString key = xjw::common::project::normalizePath(resolved);
        if (projectImages.contains(key) && !seen.contains(key))
        {
            seen.insert(key);
            selectedImages.push_back(projectImages.value(key));
        }
    }
    if (selectedImages.isEmpty())
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("生成蒙版"), QStringLiteral("没有选中可处理的照片。"));
        return;
    }

    const QString active = xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, _activeImagePath);
    GenerateMaskDialog dialog(
        selectedImages, projectImages.value(xjw::common::project::normalizePath(active)), _parentWidget);
    if (dialog.exec() != QDialog::Accepted)
    {
        return;
    }

    const QJsonObject settings = dialog.collectSettings();
    const QStringList targets = maskTargets(settings, resolvedImages);
    const QString masksDir = xjw::common::project::ProjectIO::maskOutputDir(projectPath);
    if (targets.isEmpty() || masksDir.isEmpty() || !QDir().mkpath(masksDir))
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("生成蒙版"),
                             targets.isEmpty() ? QStringLiteral("没有选中可处理的照片。")
                                               : QStringLiteral("无法创建输出目录：%1").arg(masksDir));
        return;
    }

    const auto cancellation = _cancellation.reset();
    _running = true;
    emit progressChanged(QStringLiteral("生成蒙版"), 0, targets.size());
    QPointer<ProjectMaskWorkflowController> guard(this);
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [settings, targets, projectPath, chunkId, cancellation, guard]()
        {
            GenerateMaskResult result;
            const auto options = generationOptions(settings);
            const auto operation = maskOperation(settings);
            const QString method =
                settings.value(QStringLiteral("method")).toString(QStringLiteral("black_background"));
            const bool useU2Net = method == QLatin1String("u2net");
            std::unique_ptr<xjw::mask::U2NetMaskGenerator> u2net;
            if (useU2Net)
            {
                QString error;
                auto config = u2netConfig(settings, &error);
                if (!config)
                {
                    result.errors << error;
                    return result;
                }
                config->statusCallback = [guard, projectPath, chunkId, cancellation, total = targets.size()](
                                             const std::string& status_message)
                {
                    const QString message = QString::fromStdString(status_message);
                    xjw::gui::tasks::postGuarded(
                        guard,
                        [message, projectPath, chunkId, cancellation, total](auto* self)
                        {
                            if (!cancellation.isCancellationRequested() && self->matchesSession(projectPath, chunkId))
                            {
                                emit self->progressChanged(QStringLiteral("准备 U2Net：%1").arg(message), 0, total);
                            }
                        });
                };
                try
                {
                    u2net = std::make_unique<xjw::mask::U2NetMaskGenerator>(*config);
                    result.modelSha256 = QString::fromStdString(u2net->modelSha256());
                    LOG_INFO(
                        QStringLiteral("U2Net ONNX 蒙版模型已加载: model=U2Net_v1.onnx "
                                       "backend=%1 device=%2 precision=%3 reused=%4 env=%5")
                            .arg(QString::fromStdString(xjw::mask::u2netBackendTypeToken(u2net->actualBackend())),
                                 QString::fromStdString(u2net->deviceLabel()),
                                 QString::fromStdString(xjw::mask::u2netInferencePrecisionToken(u2net->precision())),
                                 u2net->engineReused() ? QStringLiteral("true") : QStringLiteral("false"),
                                 QString::fromStdString(u2net->environmentSummary())));
                }
                catch (const std::exception& e)
                {
                    result.errors << QStringLiteral("U2Net ONNX 模型加载失败：%1").arg(QString::fromUtf8(e.what()));
                    return result;
                }
            }

            int completed = 0;
            for (const QString& imagePath : targets)
            {
                if (cancellation.isCancellationRequested())
                {
                    result.cancelled = true;
                    break;
                }
                const auto report = [&]()
                {
                    ++completed;
                    xjw::gui::tasks::postGuarded(
                        guard,
                        [completed, total = targets.size(), projectPath, chunkId, cancellation](auto* self)
                        {
                            if (!cancellation.isCancellationRequested() && self->matchesSession(projectPath, chunkId))
                            {
                                emit self->progressChanged(QStringLiteral("生成蒙版"), completed, total);
                            }
                        });
                };
                const cv::Mat source = xjw::common::io::readImage(imagePath, cv::IMREAD_UNCHANGED);
                if (source.empty())
                {
                    result.errors << QFileInfo(imagePath).fileName() + QStringLiteral(": 读取失败");
                    report();
                    continue;
                }

                cv::Mat generated;
                try
                {
                    if (useU2Net)
                    {
                        const xjw::mask::U2NetMaskResult u2net_result = u2net->generate(source);
                        generated = u2net_result.mask;
                        result.inferenceBackend =
                            QString::fromStdString(xjw::mask::u2netBackendTypeToken(u2net_result.actualBackend));
                        result.inferenceDevice = QString::fromStdString(u2net_result.deviceLabel);
                        result.inferencePrecision =
                            QString::fromStdString(xjw::mask::u2netInferencePrecisionToken(u2net_result.precision));
                        result.inferenceEnvironment = QString::fromStdString(u2net_result.environmentSummary);
                        result.deviceFallbackReason = QString::fromStdString(u2net_result.fallbackReason);
                        result.engineReused = u2net_result.engineReused;
                    }
                    else
                    {
                        generated = xjw::mask::generateMask(source, options);
                    }
                }
                catch (const std::exception& e)
                {
                    result.errors << QStringLiteral("%1: %2").arg(QFileInfo(imagePath).fileName(),
                                                                  QString::fromUtf8(e.what()));
                    report();
                    continue;
                }
                const QString maskPath =
                    xjw::common::project::ProjectIO::maskOutputPathForImage(projectPath, imagePath);
                if (QFileInfo::exists(maskPath) && operation != xjw::mask::MaskOperation::Replace)
                {
                    const cv::Mat existing = xjw::common::io::readImage(maskPath, cv::IMREAD_GRAYSCALE);
                    if (!existing.empty())
                    {
                        generated = xjw::mask::composeMasks(existing, generated, operation);
                    }
                }
                if (generated.empty() || !xjw::common::io::writeImage(maskPath, generated))
                {
                    result.errors << QFileInfo(imagePath).fileName() + QStringLiteral(": 写入失败");
                    report();
                    continue;
                }

                QJsonObject record;
                record.insert(QStringLiteral("mask_path"), QDir::cleanPath(maskPath));
                record.insert(QStringLiteral("mask_method"), method);
                if (useU2Net)
                {
                    record.insert(QStringLiteral("mask_inference_backend"), result.inferenceBackend);
                    record.insert(QStringLiteral("mask_inference_device"), result.inferenceDevice);
                    record.insert(QStringLiteral("mask_inference_precision"), result.inferencePrecision);
                    record.insert(QStringLiteral("mask_model_sha256"), result.modelSha256);
                }
                record.insert(QStringLiteral("mask_updated_at"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
                result.recordsByImage.insert(xjw::common::project::normalizePath(imagePath), record);
                result.generatedImages << imagePath;
                report();
            }
            return result;
        },
        [projectPath, chunkId, masksDir](ProjectMaskWorkflowController* self,
                                         xjw::gui::tasks::TaskOutcome<GenerateMaskResult> outcome)
        {
            self->_running = false;
            if (!outcome.succeeded())
            {
                emit self->finished(false);
                if (self->matchesSession(projectPath, chunkId))
                    QMessageBox::warning(self->_parentWidget, QStringLiteral("生成蒙版"), outcome.errorMessage);
                return;
            }

            GenerateMaskResult result = std::move(*outcome.value);
            emit self->finished(!result.cancelled && result.errors.isEmpty() && !result.generatedImages.isEmpty());
            if (!self->matchesSession(projectPath, chunkId))
            {
                return;
            }
            if (result.generatedImages.isEmpty())
            {
                QMessageBox::warning(
                    self->_parentWidget,
                    QStringLiteral("生成蒙版"),
                    result.cancelled
                        ? QStringLiteral("蒙版生成已取消。")
                        : QStringLiteral("蒙版生成失败：%1").arg(result.errors.join(QStringLiteral("; "))));
                return;
            }

            QJsonObject meta = self->_projectData->coreFilesMeta();
            QJsonArray images = meta.value(QStringLiteral("images")).toArray();
            for (int i = 0; i < images.size(); ++i)
            {
                QJsonObject image = images.at(i).toObject();
                const QString path = xjw::common::project::ProjectIO::resolveProjectResourcePath(
                    projectPath, image.value(QStringLiteral("path")).toString());
                const auto record = result.recordsByImage.value(xjw::common::project::normalizePath(path));
                if (record.isEmpty())
                {
                    continue;
                }
                for (auto it = record.begin(); it != record.end(); ++it)
                {
                    image.insert(it.key(), it.value());
                }
                images.replace(i, image);
            }
            meta.insert(QStringLiteral("images"), images);
            xjw::gui::project::persistProjectMeta(self->_projectData, meta, true);
            emit self->projectMetadataUpdated(projectPath);
            emit self->masksGenerated(result.generatedImages);

            QString message =
                result.cancelled ? QStringLiteral("已取消，已保留 %1 张照片的蒙版。").arg(result.generatedImages.size())
                                 : QStringLiteral("已生成 %1 张照片的蒙版。").arg(result.generatedImages.size());
            if (!result.errors.isEmpty())
            {
                message += QStringLiteral("\n部分失败：%1").arg(result.errors.join(QStringLiteral("; ")));
            }
            if (!result.inferenceDevice.isEmpty())
            {
                message += QStringLiteral("\nU2Net 实际推理：%1 / %2 / %3")
                               .arg(result.inferenceBackend, result.inferenceDevice, result.inferencePrecision);
                if (result.inferenceBackend == QLatin1String("tensorrt"))
                {
                    message += result.engineReused ? QStringLiteral("（已复用本机 engine）")
                                                   : QStringLiteral("（已为本机新建 engine）");
                }
            }
            if (!result.deviceFallbackReason.isEmpty())
            {
                message += QStringLiteral("\n后端回退原因：%1").arg(result.deviceFallbackReason);
            }
            QMessageBox::information(self->_parentWidget, QStringLiteral("生成蒙版"), message);
            LOG_INFO(QStringLiteral("蒙版生成完成: count=%1 dir=%2 backend=%3 device=%4 "
                                    "precision=%5 reused=%6 env=%7 fallback=%8")
                         .arg(result.generatedImages.size())
                         .arg(masksDir,
                              result.inferenceBackend,
                              result.inferenceDevice,
                              result.inferencePrecision,
                              result.engineReused ? QStringLiteral("true") : QStringLiteral("false"),
                              result.inferenceEnvironment,
                              result.deviceFallbackReason));
        });
}

void ProjectMaskWorkflowController::cancelActiveTask()
{
    _cancellation.requestCancellation();
}

bool ProjectMaskWorkflowController::matchesSession(const QString& projectPath, const QString& chunkId) const
{
    return _projectData && _projectData->currentProjectPath() == projectPath &&
           _projectData->activeChunkId() == chunkId;
}
