#include "ProjectMaskWorkflowController.h"

#include "image/GenerateMaskDialog.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "MaskGenerator.h"
#include "ProjectMaskInferenceAdapter.h"
#include "ProjectData.h"
#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"
#include "ProjectMetadataOperations.h"
#include "ProjectOpenGuard.h"
#include "io/PathIO.h"

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

#include <atomic>
#include <exception>
#include <memory>

namespace
{

    struct GenerateMaskResult
    {
        QMap<QString, QJsonObject> recordsByImage;
        QStringList generatedImages;
        QStringList errors;
        QString inferenceModelId;
        QString inferenceModelFileName;
        QString inferenceBackend;
        QString inferenceDevice;
        QString inferencePrecision;
        QString inferenceEnvironment;
        QString modelSha256;
        QString deviceFallbackReason;
        QString enginePath;
        int inferenceInputSize = 0;
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
            const bool useAiMask = method == QLatin1String("u2net") ||
                                   method == QLatin1String("birefnet_dynamic");
            std::unique_ptr<xjw::gui::project::ProjectMaskInferenceAdapter> inference;
            if (useAiMask)
            {
                QString error;
                inference = xjw::gui::project::ProjectMaskInferenceAdapter::create(
                    method,
                    settings,
                    [guard, projectPath, chunkId, cancellation, total = targets.size()](
                        const std::string& status_message)
                    {
                        const QString message = QString::fromStdString(status_message);
                        xjw::gui::tasks::postGuarded(
                            guard,
                            [message, projectPath, chunkId, cancellation, total](auto* self)
                            {
                                if (!cancellation.isCancellationRequested() &&
                                    self->matchesSession(projectPath, chunkId))
                                {
                                    emit self->progressChanged(
                                        QStringLiteral("准备 AI 蒙版：%1").arg(message), 0, total);
                                }
                            });
                    },
                    &error);
                if (!inference)
                {
                    result.errors << error;
                    return result;
                }

                const xjw::gui::project::ProjectMaskInferenceResult metadata = inference->metadata();
                result.inferenceModelId = metadata.modelId;
                result.inferenceModelFileName = metadata.modelFileName;
                result.modelSha256 = metadata.modelSha256;
                result.inferenceBackend = metadata.backend;
                result.inferenceDevice = metadata.device;
                result.inferencePrecision = metadata.precision;
                result.inferenceEnvironment = metadata.environment;
                result.deviceFallbackReason = metadata.fallbackReason;
                result.enginePath = metadata.enginePath;
                result.inferenceInputSize = metadata.inputSize;
                result.engineReused = metadata.engineReused;
                LOG_INFO(
                    QStringLiteral("AI ONNX 蒙版模型已加载: model_id=%1 file=%2 backend=%3 "
                                   "device=%4 precision=%5 input=%6 reused=%7 engine=%8 env=%9")
                        .arg(result.inferenceModelId,
                             result.inferenceModelFileName,
                             result.inferenceBackend,
                             result.inferenceDevice,
                             result.inferencePrecision)
                        .arg(result.inferenceInputSize)
                        .arg(result.engineReused ? QStringLiteral("true") : QStringLiteral("false"),
                             result.enginePath,
                             result.inferenceEnvironment));
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
                xjw::gui::project::ProjectMaskInferenceResult inference_result;
                try
                {
                    if (useAiMask)
                    {
                        inference_result = inference->generate(source);
                        generated = inference_result.mask;
                        result.inferenceModelId = inference_result.modelId;
                        result.inferenceModelFileName = inference_result.modelFileName;
                        result.modelSha256 = inference_result.modelSha256;
                        result.inferenceBackend = inference_result.backend;
                        result.inferenceDevice = inference_result.device;
                        result.inferencePrecision = inference_result.precision;
                        result.inferenceEnvironment = inference_result.environment;
                        result.deviceFallbackReason = inference_result.fallbackReason;
                        result.enginePath = inference_result.enginePath;
                        result.inferenceInputSize = inference_result.inputSize;
                        result.engineReused = inference_result.engineReused;
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
                if (useAiMask)
                {
                    record.insert(QStringLiteral("mask_model_id"), inference_result.modelId);
                    record.insert(QStringLiteral("mask_model_file_name"), inference_result.modelFileName);
                    record.insert(QStringLiteral("mask_model_sha256"), inference_result.modelSha256);
                    record.insert(QStringLiteral("mask_model_input_size"), inference_result.inputSize);
                    record.insert(QStringLiteral("mask_inference_backend"), inference_result.backend);
                    record.insert(QStringLiteral("mask_inference_device"), inference_result.device);
                    record.insert(QStringLiteral("mask_inference_precision"), inference_result.precision);
                    record.insert(QStringLiteral("mask_inference_environment"), inference_result.environment);
                    record.insert(QStringLiteral("mask_inference_fallback_reason"), inference_result.fallbackReason);
                    record.insert(QStringLiteral("mask_engine_cache_path"), inference_result.enginePath);
                    record.insert(QStringLiteral("mask_engine_cache_reused"), inference_result.engineReused);
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
                message += QStringLiteral("\nAI 实际推理：%1 / %2 / %3 / %4")
                               .arg(result.inferenceModelId,
                                    result.inferenceBackend,
                                    result.inferenceDevice,
                                    result.inferencePrecision);
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
            LOG_INFO(QStringLiteral("蒙版生成完成: count=%1 dir=%2 model=%3 backend=%4 device=%5 "
                                    "precision=%6 reused=%7 engine=%8 env=%9 fallback=%10")
                         .arg(result.generatedImages.size())
                         .arg(masksDir,
                              result.inferenceModelId,
                              result.inferenceBackend,
                              result.inferenceDevice,
                              result.inferencePrecision)
                         .arg(result.engineReused ? QStringLiteral("true") : QStringLiteral("false"),
                              result.enginePath,
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
