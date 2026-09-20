#include "ProjectMaskWorkflowController.h"

#include "GuiTaskRunner.h"
#include "Logger.h"
#include "MaskGenerator.h"
#include "ProjectMaskInferenceAdapter.h"
#include "io/PathIO.h"
#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "project/tasks/ProjectTaskOrchestrator.h"

#include <opencv2/imgcodecs.hpp>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QPointer>
#include <QSaveFile>
#include <QSet>
#include <QUuid>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <memory>
#include <utility>

namespace
{

    struct FrozenMaskTarget
    {
        QString imagePath;
        QString finalPath;
        QString stagingPath;
    };

    struct GenerateMaskResult
    {
        QVector<StagedMaskArtifact> artifacts;
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

    struct InteractiveMaskSaveResult
    {
        std::optional<StagedMaskArtifact> artifact;
        QString errorMessage;
        bool cancelled = false;
    };

    struct ManagedMaskPath
    {
        QString sourcePath;
        QString physicalPath;
        QString mutationKey;
        QString referenceTargetKey;
        bool leafLink = false;
    };

    struct ManagedMaskRoot
    {
        QString sourcePath;
        QString physicalPath;
        QString key;
    };

    struct ProjectMaskSnapshot
    {
        QString imagePath;
        QString imageKey;
        bool selected = false;
        bool hasMaskState = false;
        QJsonObject maskRecord;
        std::optional<ManagedMaskPath> explicitMaskPath;
        ManagedMaskPath standardMaskPath;
    };

    struct ClearMaskTarget
    {
        QString imagePath;
        QVector<ManagedMaskPath> managedMaskPaths;
        QSet<QString> identityKeys;
    };

    struct IsolatedMaskFile
    {
        QString finalPath;
        QString tombstonePath;
        QString recoveryPath;
    };

    struct IsolatedMaskComponent
    {
        QStringList imagePaths;
        QVector<ManagedMaskPath> managedMaskPaths;
        QSet<QString> mutationKeys;
        QVector<IsolatedMaskFile> files;
    };

    struct RecoveryCopyMapping
    {
        QString finalPath;
        QString copyPath;
    };

    QString formatRecoveryCopyMapping(const QString& finalPath, const QString& copyPath)
    {
        return QStringLiteral("目标=%1，恢复副本=%2").arg(finalPath, copyPath);
    }

    QString logRetainedRecoveryFailures(const QStringList& failures)
    {
        if (failures.isEmpty())
        {
            return {};
        }
        const QString message =
            QStringLiteral("蒙版事务回滚失败，已保留可恢复文件：%1").arg(failures.join(QStringLiteral("；")));
        LOG_ERROR(message);
        return message;
    }

    const QStringList& maskMetadataKeys()
    {
        static const QStringList keys{QStringLiteral("mask_path"),
                                      QStringLiteral("mask_method"),
                                      QStringLiteral("mask_model_id"),
                                      QStringLiteral("mask_model_file_name"),
                                      QStringLiteral("mask_model_sha256"),
                                      QStringLiteral("mask_model_input_size"),
                                      QStringLiteral("mask_inference_backend"),
                                      QStringLiteral("mask_inference_device"),
                                      QStringLiteral("mask_inference_precision"),
                                      QStringLiteral("mask_inference_environment"),
                                      QStringLiteral("mask_inference_fallback_reason"),
                                      QStringLiteral("mask_engine_cache_path"),
                                      QStringLiteral("mask_engine_cache_reused"),
                                      QStringLiteral("mask_updated_at")};
        return keys;
    }

    bool hasMaskMetadata(const QJsonObject& image)
    {
        return std::any_of(maskMetadataKeys().cbegin(),
                           maskMetadataKeys().cend(),
                           [&image](const QString& key) { return image.contains(key); });
    }

    QJsonObject maskRecordFromImage(const QJsonObject& image)
    {
        QJsonObject record;
        for (const QString& key : maskMetadataKeys())
        {
            if (image.contains(key))
            {
                record.insert(key, image.value(key));
            }
        }
        return record;
    }

    QString normalizedAbsolutePath(const QString& path)
    {
        if (path.trimmed().isEmpty())
        {
            return {};
        }
        return QDir::fromNativeSeparators(QDir::cleanPath(QFileInfo(path).absoluteFilePath()));
    }

    QString managedPathKey(const QString& path)
    {
        const QString normalized = normalizedAbsolutePath(path);
#ifdef Q_OS_WIN
        return normalized.toCaseFolded();
#else
        return normalized;
#endif
    }

    bool isLeafLink(const QFileInfo& info)
    {
        return info.isSymLink() || info.isJunction();
    }

    bool hasUnresolvablePathLink(const QString& path)
    {
        const std::filesystem::path filesystem_path = xjw::common::io::toFilesystemPath(normalizedAbsolutePath(path));
        if (filesystem_path.empty())
        {
            return true;
        }

        std::filesystem::path current_path = filesystem_path.root_path();
        for (const std::filesystem::path& component : filesystem_path.relative_path())
        {
            current_path /= component;
            const QFileInfo info(xjw::common::io::fromFilesystemPath(current_path));
            if (isLeafLink(info) && info.canonicalFilePath().isEmpty())
            {
                return true;
            }
        }
        return false;
    }

    std::optional<ManagedMaskRoot> resolveManagedMaskRoot(const QString& path)
    {
        const QString source_path = normalizedAbsolutePath(path);
        if (source_path.isEmpty() || hasUnresolvablePathLink(source_path))
        {
            return std::nullopt;
        }

        const QFileInfo root_info(source_path);
        if (isLeafLink(root_info) && root_info.canonicalFilePath().isEmpty())
        {
            return std::nullopt;
        }
        if (root_info.exists() && !root_info.isDir())
        {
            return std::nullopt;
        }

        const auto comparison = xjw::common::io::comparePathsSafely(xjw::common::io::toFilesystemPath(source_path),
                                                                    xjw::common::io::toFilesystemPath(source_path));
        if (!comparison.valid)
        {
            return std::nullopt;
        }
        const QString physical_path =
            normalizedAbsolutePath(xjw::common::io::fromFilesystemPath(comparison.normalizedFirst));
        const QString key = managedPathKey(physical_path);
        if (physical_path.isEmpty() || key.isEmpty())
        {
            return std::nullopt;
        }
        return ManagedMaskRoot{source_path, physical_path, key};
    }

    std::optional<ManagedMaskPath> resolveManagedMaskPath(const QString& path, const ManagedMaskRoot& root)
    {
        const QString source_path = normalizedAbsolutePath(path);
        const QFileInfo source_info(source_path);
        const QString leaf_name = source_info.fileName();
        if (source_path.isEmpty() || leaf_name.isEmpty() || hasUnresolvablePathLink(source_info.absolutePath()))
        {
            return std::nullopt;
        }

        const auto parent_comparison =
            xjw::common::io::comparePathsSafely(xjw::common::io::toFilesystemPath(root.physicalPath),
                                                xjw::common::io::toFilesystemPath(source_info.absolutePath()));
        if (!parent_comparison.valid || (!parent_comparison.equivalent && !parent_comparison.firstIsAncestorOfSecond))
        {
            return std::nullopt;
        }

        const std::filesystem::path physical_entry =
            (parent_comparison.normalizedSecond / xjw::common::io::toFilesystemPath(leaf_name)).lexically_normal();
        const QString physical_path = normalizedAbsolutePath(xjw::common::io::fromFilesystemPath(physical_entry));
        const QString mutation_key = managedPathKey(physical_path);
        if (physical_path.isEmpty() || mutation_key.isEmpty())
        {
            return std::nullopt;
        }

        ManagedMaskPath resolved;
        resolved.sourcePath = source_path;
        resolved.physicalPath = physical_path;
        resolved.mutationKey = mutation_key;
        resolved.leafLink = isLeafLink(source_info);
        if (!resolved.leafLink)
        {
            return resolved;
        }

        const QString target_path = source_info.canonicalFilePath();
        if (target_path.isEmpty())
        {
            return resolved;
        }
        const auto target_comparison = xjw::common::io::comparePathsSafely(
            xjw::common::io::toFilesystemPath(root.physicalPath), xjw::common::io::toFilesystemPath(target_path));
        if (target_comparison.valid && target_comparison.firstIsAncestorOfSecond)
        {
            resolved.referenceTargetKey =
                managedPathKey(xjw::common::io::fromFilesystemPath(target_comparison.normalizedSecond));
        }
        return resolved;
    }

    bool managedMaskPathExists(const ManagedMaskPath& path)
    {
        return path.leafLink || QFileInfo::exists(path.physicalPath);
    }

    void addManagedMaskPathIdentities(const ManagedMaskPath& path, QSet<QString>* identities)
    {
        if (!identities)
        {
            return;
        }
        identities->insert(path.mutationKey);
        if (!path.referenceTargetKey.isEmpty())
        {
            identities->insert(path.referenceTargetKey);
        }
    }

    QString uniqueSiblingPath(const QString& finalPath, const QString& suffix)
    {
        return QStringLiteral("%1.plascan-mask-%2.%3")
            .arg(finalPath, QUuid::createUuid().toString(QUuid::WithoutBraces), suffix);
    }

    bool copyFileAtomically(const QString& sourcePath, const QString& destinationPath, QString* errorMessage)
    {
        QFile input(sourcePath);
        if (!input.open(QIODevice::ReadOnly))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取蒙版暂存文件：%1").arg(sourcePath);
            }
            return false;
        }

        QSaveFile output(destinationPath);
        output.setDirectWriteFallback(false);
        if (!output.open(QIODevice::WriteOnly))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法创建蒙版发布文件：%1").arg(destinationPath);
            }
            return false;
        }

        constexpr qint64 chunk_size = 64 * 1024;
        while (!input.atEnd())
        {
            const QByteArray chunk = input.read(chunk_size);
            if (chunk.isEmpty() && input.error() != QFileDevice::NoError)
            {
                output.cancelWriting();
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("读取蒙版暂存文件失败：%1").arg(sourcePath);
                }
                return false;
            }
            if (!chunk.isEmpty() && output.write(chunk) != chunk.size())
            {
                output.cancelWriting();
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("写入蒙版发布文件失败：%1").arg(destinationPath);
                }
                return false;
            }
        }
        if (!output.commit())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("原子发布蒙版失败：%1").arg(destinationPath);
            }
            return false;
        }
        return true;
    }

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

    QStringList requestedMaskTargets(const QJsonObject& settings, const QStringList& allImages)
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

    xjw::gui::project::ProjectTaskOrchestrator* maskOwner(ProjectMaskWorkflowController* controller)
    {
        return controller ? qobject_cast<xjw::gui::project::ProjectTaskOrchestrator*>(controller->parent()) : nullptr;
    }

} // namespace

ProjectMaskWorkflowController::ProjectMaskWorkflowController(xjw::gui::project::ProjectSession* session,
                                                             ProjectUiMessageAdapter* messages,
                                                             MaskSettingsProvider settingsProvider,
                                                             QObject* parent)
    : QObject(parent), _session(session), _messages(messages), _settingsProvider(std::move(settingsProvider))
{
}

ProjectMaskWorkflowController::~ProjectMaskWorkflowController()
{
    waitForActiveTask();
    cleanupOwnedArtifacts();
}

void ProjectMaskWorkflowController::setActiveImagePath(const QString& imagePath)
{
    _activeImagePath = imagePath;
}

void ProjectMaskWorkflowController::openDialog(xjw::gui::project::ProjectTaskContext context)
{
    openDialogForImages(std::move(context), _session ? _session->allImages() : QStringList{});
}

void ProjectMaskWorkflowController::openDialogForImages(xjw::gui::project::ProjectTaskContext context,
                                                        const QStringList& requestedImages)
{
    using xjw::gui::project::ProjectTaskOrchestrator;
    const QPointer<ProjectMaskWorkflowController> self_guard(this);
    const QPointer<ProjectTaskOrchestrator> owner_guard(maskOwner(this));
    const QPointer<xjw::gui::project::ProjectSession> session_guard(_session);
    ProjectUiMessageAdapter* const messages = _messages;
    const auto settle = [owner_guard, context]()
    {
        if (owner_guard)
        {
            owner_guard->settleMaskOperation(context);
        }
    };
    if (!owner_guard || !session_guard || !owner_guard->maskContextIsLive(context))
    {
        settle();
        return;
    }
    if (!session_guard->hasProject())
    {
        if (messages && owner_guard->maskContextIsLive(context))
        {
            messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("请先打开项目，再生成照片蒙版。"));
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    const QString project_path = session_guard->projectPath();
    QHash<QString, QString> project_images;
    QStringList resolved_images;
    for (const QString& path : session_guard->allImages())
    {
        const QString resolved =
            QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, path));
        const QString key = xjw::common::project::normalizePath(resolved);
        if (!key.isEmpty() && !project_images.contains(key))
        {
            project_images.insert(key, resolved);
            resolved_images.push_back(resolved);
        }
    }
    if (resolved_images.isEmpty())
    {
        if (messages && owner_guard->maskContextIsLive(context))
        {
            messages->warning(nullptr, QStringLiteral("生成蒙版"), QStringLiteral("当前项目没有照片。"));
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    QStringList selected_images;
    QSet<QString> seen;
    for (const QString& path : requestedImages)
    {
        const QString resolved =
            QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, path));
        const QString key = xjw::common::project::normalizePath(resolved);
        if (project_images.contains(key) && !seen.contains(key))
        {
            seen.insert(key);
            selected_images.push_back(project_images.value(key));
        }
    }
    if (selected_images.isEmpty())
    {
        if (messages && owner_guard->maskContextIsLive(context))
        {
            messages->warning(nullptr, QStringLiteral("生成蒙版"), QStringLiteral("没有选中可处理的照片。"));
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    const QString active =
        QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, _activeImagePath));
    const QString current_image = project_images.value(xjw::common::project::normalizePath(active));
    const MaskSettingsProvider settings_provider = _settingsProvider;
    if (!settings_provider)
    {
        settle();
        return;
    }
    const std::optional<QJsonObject> provided_settings = settings_provider(selected_images, current_image);
    if (!self_guard || !owner_guard)
    {
        return;
    }
    if (!session_guard)
    {
        settle();
        return;
    }
    if (!owner_guard->maskContextIsLive(context))
    {
        settle();
        return;
    }
    if (!provided_settings.has_value())
    {
        settle();
        return;
    }

    const QJsonObject settings = *provided_settings;
    const QStringList requested_targets = requestedMaskTargets(settings, resolved_images);
    QStringList targets;
    seen.clear();
    for (const QString& path : requested_targets)
    {
        const QString resolved =
            QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, path));
        const QString key = xjw::common::project::normalizePath(resolved);
        if (project_images.contains(key) && !seen.contains(key))
        {
            seen.insert(key);
            targets.push_back(project_images.value(key));
        }
    }
    if (targets.isEmpty())
    {
        if (messages && owner_guard->maskContextIsLive(context))
        {
            messages->warning(nullptr, QStringLiteral("生成蒙版"), QStringLiteral("没有选中可处理的照片。"));
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    QVector<FrozenMaskTarget> frozen_targets;
    frozen_targets.reserve(targets.size());
    for (const QString& image_path : targets)
    {
        const QString final_path = xjw::common::project::ProjectIO::maskOutputPathForImage(project_path, image_path);
        const QString output_directory = QFileInfo(final_path).absolutePath();
        if (final_path.isEmpty() || !QDir().mkpath(output_directory))
        {
            for (const FrozenMaskTarget& frozen_target : std::as_const(frozen_targets))
            {
                self_guard->removeOwnedArtifact(frozen_target.stagingPath);
            }
            if (messages && owner_guard->maskContextIsLive(context))
            {
                messages->warning(
                    nullptr, QStringLiteral("生成蒙版"), QStringLiteral("无法创建输出目录：%1").arg(output_directory));
            }
            if (!self_guard || !owner_guard)
            {
                return;
            }
            settle();
            return;
        }
        const QString staging_path = uniqueSiblingPath(final_path, QStringLiteral("staging.png"));
        self_guard->rememberOwnedArtifact(staging_path);
        frozen_targets.push_back({image_path, final_path, staging_path});
    }

    if (!owner_guard->maskContextIsLive(context))
    {
        for (const FrozenMaskTarget& target : frozen_targets)
        {
            self_guard->removeOwnedArtifact(target.stagingPath);
        }
        settle();
        return;
    }
    owner_guard->setMaskLaneMode(context, ProjectTaskOrchestrator::MaskLaneMode::Batch);
    emit self_guard->progressChanged(QStringLiteral("生成蒙版"), 0, frozen_targets.size());
    if (!self_guard || !owner_guard)
    {
        return;
    }
    if (!owner_guard->maskContextIsLive(context))
    {
        for (const FrozenMaskTarget& target : frozen_targets)
        {
            self_guard->removeOwnedArtifact(target.stagingPath);
        }
        const bool user_cancel =
            owner_guard->maskContextIsLive(context, true) &&
            owner_guard->maskCancelReason(context) == ProjectTaskOrchestrator::MaskCancelReason::User;
        if (user_cancel && owner_guard->maskTailIsLive(context))
        {
            if (messages)
            {
                messages->warning(nullptr, QStringLiteral("生成蒙版"), QStringLiteral("蒙版生成已取消。"));
            }
            if (!self_guard || !owner_guard)
            {
                return;
            }
            if (owner_guard->maskTailIsLive(context) &&
                owner_guard->maskCancelReason(context) == ProjectTaskOrchestrator::MaskCancelReason::User)
            {
                emit self_guard->finished(false);
            }
            if (!self_guard || !owner_guard)
            {
                return;
            }
        }
        settle();
        return;
    }

    ++self_guard->_runningWorkers;
    const QPointer<ProjectMaskWorkflowController> guard(self_guard);
    const std::function<void(int)> staged_observer = self_guard->_artifactStagedObserverForTesting;
    QFuture<void> future = xjw::gui::tasks::runGuardedWithOutcome(
        self_guard.data(),
        [settings, frozen_targets, context, guard, staged_observer]()
        {
            GenerateMaskResult result;
            const auto options = generationOptions(settings);
            const auto operation = maskOperation(settings);
            const QString method =
                settings.value(QStringLiteral("method")).toString(QStringLiteral("black_background"));
            const bool use_ai_mask = method == QLatin1String("u2net") || method == QLatin1String("birefnet_dynamic");
            std::unique_ptr<xjw::gui::project::ProjectMaskInferenceAdapter> inference;
            if (use_ai_mask)
            {
                QString error;
                inference = xjw::gui::project::ProjectMaskInferenceAdapter::create(
                    method,
                    settings,
                    [guard, context, total = frozen_targets.size()](const std::string& status_message)
                    {
                        const QString message = QString::fromStdString(status_message);
                        xjw::gui::tasks::postGuarded(
                            guard,
                            [message, context, total](ProjectMaskWorkflowController* self)
                            {
                                auto* owner = maskOwner(self);
                                if (owner && owner->maskContextIsLive(context))
                                {
                                    emit self->progressChanged(
                                        QStringLiteral("准备 AI 蒙版：%1").arg(message), 0, total);
                                }
                            });
                    },
                    &error);
                if (!inference)
                {
                    result.errors.push_back(error);
                    return result;
                }
                const auto metadata = inference->metadata();
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
            }

            int completed = 0;
            for (const FrozenMaskTarget& target : frozen_targets)
            {
                if (!context.cancelFlag || context.cancelFlag->load(std::memory_order_relaxed))
                {
                    result.cancelled = true;
                    break;
                }
                const auto report = [&]()
                {
                    ++completed;
                    xjw::gui::tasks::postGuarded(
                        guard,
                        [completed, total = frozen_targets.size(), context](ProjectMaskWorkflowController* self)
                        {
                            auto* owner = maskOwner(self);
                            if (owner && owner->maskContextIsLive(context))
                            {
                                emit self->progressChanged(QStringLiteral("生成蒙版"), completed, total);
                            }
                        });
                };

                const cv::Mat source = xjw::common::io::readImage(target.imagePath, cv::IMREAD_UNCHANGED);
                if (source.empty())
                {
                    result.errors.push_back(QFileInfo(target.imagePath).fileName() + QStringLiteral(": 读取失败"));
                    report();
                    continue;
                }

                cv::Mat generated;
                xjw::gui::project::ProjectMaskInferenceResult inference_result;
                try
                {
                    if (use_ai_mask)
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
                catch (const std::exception& exception)
                {
                    result.errors.push_back(QStringLiteral("%1: %2").arg(QFileInfo(target.imagePath).fileName(),
                                                                         QString::fromUtf8(exception.what())));
                    report();
                    continue;
                }

                if (QFileInfo::exists(target.finalPath) && operation != xjw::mask::MaskOperation::Replace)
                {
                    const cv::Mat existing = xjw::common::io::readImage(target.finalPath, cv::IMREAD_GRAYSCALE);
                    if (!existing.empty())
                    {
                        generated = xjw::mask::composeMasks(existing, generated, operation);
                    }
                }
                if (!context.cancelFlag || context.cancelFlag->load(std::memory_order_relaxed))
                {
                    result.cancelled = true;
                    break;
                }
                if (generated.empty() || !xjw::common::io::writeImage(target.stagingPath, generated))
                {
                    result.errors.push_back(QFileInfo(target.imagePath).fileName() + QStringLiteral(": 写入失败"));
                    report();
                    continue;
                }
                if (context.cancelFlag->load(std::memory_order_relaxed))
                {
                    QFile::remove(target.stagingPath);
                    result.cancelled = true;
                    break;
                }

                QJsonObject record;
                record.insert(QStringLiteral("mask_path"), QDir::cleanPath(target.finalPath));
                record.insert(QStringLiteral("mask_method"), method);
                if (use_ai_mask)
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
                result.artifacts.push_back({target.imagePath, target.finalPath, target.stagingPath, record});
                if (staged_observer)
                {
                    staged_observer(result.artifacts.size());
                }
                report();
            }
            if (context.cancelFlag && context.cancelFlag->load(std::memory_order_relaxed))
            {
                result.cancelled = true;
            }
            return result;
        },
        [context, project_path](ProjectMaskWorkflowController* self,
                                xjw::gui::tasks::TaskOutcome<GenerateMaskResult> outcome)
        {
            self->pruneFinishedFutures();
            const QPointer<ProjectMaskWorkflowController> self_guard(self);
            const QPointer<xjw::gui::project::ProjectTaskOrchestrator> owner_guard(maskOwner(self));
            const QPointer<xjw::gui::project::ProjectSession> session_guard(self->_session);
            ProjectUiMessageAdapter* const messages = self->_messages;
            --self->_runningWorkers;
            QVector<StagedMaskArtifact> artifacts;
            if (outcome.value)
            {
                artifacts = outcome.value->artifacts;
            }
            QSet<QString> completed_staging;
            for (const StagedMaskArtifact& artifact : artifacts)
            {
                completed_staging.insert(artifact.stagingPath);
            }
            const QSet<QString> owned_artifacts = self->_ownedArtifacts;
            for (const QString& path : owned_artifacts)
            {
                if (path.contains(QStringLiteral(".staging.png")) && !completed_staging.contains(path))
                {
                    self->removeOwnedArtifact(path);
                }
            }
            if (!owner_guard || !outcome.succeeded() || !owner_guard->maskContextIsLive(context, true))
            {
                for (const StagedMaskArtifact& artifact : artifacts)
                {
                    self->removeOwnedArtifact(artifact.stagingPath);
                }
                if (owner_guard && owner_guard->maskContextIsLive(context, true) && !outcome.succeeded())
                {
                    if (messages)
                    {
                        messages->warning(nullptr, QStringLiteral("生成蒙版"), outcome.errorMessage);
                    }
                    if (!self_guard || !owner_guard)
                    {
                        return;
                    }
                    if (owner_guard->maskTailIsLive(context))
                    {
                        emit self_guard->finished(false);
                    }
                    if (!self_guard || !owner_guard)
                    {
                        return;
                    }
                }
                if (owner_guard)
                {
                    owner_guard->settleMaskOperation(context);
                }
                return;
            }

            GenerateMaskResult result = std::move(*outcome.value);
            PublishMaskResult publication;
            if (!result.artifacts.isEmpty())
            {
                publication = self_guard->publishStagedArtifacts(context, result.artifacts, true);
            }
            if (!self_guard || !owner_guard)
            {
                return;
            }
            const bool committed = publication.committed;
            const auto user_cancelled = [&]() {
                return owner_guard &&
                       owner_guard->maskCancelReason(context) == ProjectTaskOrchestrator::MaskCancelReason::User;
            };
            if (committed && publication.tailAllowed && owner_guard->maskTailIsLive(context))
            {
                if (!publication.metadataDeltaImages.isEmpty())
                {
                    emit self_guard->projectMetadataUpdated(project_path);
                    if (!self_guard || !owner_guard)
                    {
                        return;
                    }
                    if (!session_guard)
                    {
                        owner_guard->settleMaskOperation(context);
                        return;
                    }
                }
                if (owner_guard->maskTailIsLive(context))
                {
                    emit self_guard->masksGenerated(publication.filesystemSucceededImages);
                }
                if (!self_guard || !owner_guard)
                {
                    return;
                }
                if (!session_guard)
                {
                    owner_guard->settleMaskOperation(context);
                    return;
                }
            }

            if (owner_guard->maskTailIsLive(context))
            {
                const bool user_cancel = user_cancelled();
                QString message;
                if (user_cancel || result.cancelled)
                {
                    message = committed ? QStringLiteral("已取消，已保留 %1 张照片的蒙版。")
                                              .arg(publication.filesystemSucceededImages.size())
                                        : QStringLiteral("蒙版生成已取消。");
                }
                else if (!publication.errorMessage.isEmpty())
                {
                    message = QStringLiteral("蒙版生成失败：%1").arg(publication.errorMessage);
                }
                else if (!committed)
                {
                    message = QStringLiteral("蒙版生成失败：%1").arg(result.errors.join(QStringLiteral("; ")));
                }
                else
                {
                    message =
                        QStringLiteral("已生成 %1 张照片的蒙版。").arg(publication.filesystemSucceededImages.size());
                }
                if (!result.errors.isEmpty() && committed)
                {
                    message += QStringLiteral("\n部分失败：%1").arg(result.errors.join(QStringLiteral("; ")));
                }
                if (!result.inferenceDevice.isEmpty() && committed)
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
                if (!result.deviceFallbackReason.isEmpty() && committed)
                {
                    message += QStringLiteral("\n后端回退原因：%1").arg(result.deviceFallbackReason);
                }
                if (messages)
                {
                    if (committed)
                    {
                        messages->information(nullptr, QStringLiteral("生成蒙版"), message);
                    }
                    else
                    {
                        messages->warning(nullptr, QStringLiteral("生成蒙版"), message);
                    }
                }
                if (!self_guard || !owner_guard)
                {
                    return;
                }
                if (!session_guard)
                {
                    owner_guard->settleMaskOperation(context);
                    return;
                }
                if (owner_guard->maskTailIsLive(context))
                {
                    const bool success = committed && !user_cancelled() && !result.cancelled && result.errors.isEmpty();
                    emit self_guard->finished(success);
                }
                if (!self_guard || !owner_guard)
                {
                    return;
                }
            }
            owner_guard->settleMaskOperation(context);
        });
    if (self_guard)
    {
        self_guard->registerFuture(std::move(future));
    }
}

void ProjectMaskWorkflowController::saveInteractiveMask(xjw::gui::project::ProjectTaskContext context,
                                                        const QString& imagePath,
                                                        const QImage& mask,
                                                        const QString& method,
                                                        quint64 revision)
{
    const QPointer<ProjectMaskWorkflowController> self_guard(this);
    const QPointer<xjw::gui::project::ProjectTaskOrchestrator> owner_guard(maskOwner(this));
    const QPointer<xjw::gui::project::ProjectSession> session_guard(_session);
    const auto settle = [owner_guard, context]()
    {
        if (owner_guard)
        {
            owner_guard->settleMaskOperation(context);
        }
    };
    if (!owner_guard || !session_guard || !owner_guard->maskContextIsLive(context))
    {
        settle();
        return;
    }
    if (!session_guard->hasProject() || imagePath.trimmed().isEmpty() || mask.isNull())
    {
        emit self_guard->interactiveMaskSaveFailed(
            imagePath, revision, QStringLiteral("当前项目或照片不可用，无法保存蒙版。"));
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    const QString project_path = session_guard->projectPath();
    const QString resolved_image =
        QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, imagePath));
    const QString resolved_key = xjw::common::project::normalizePath(resolved_image);
    bool belongs_to_project = false;
    for (const QString& project_image : session_guard->allImages())
    {
        const QString resolved =
            xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, project_image);
        if (xjw::common::project::normalizePath(resolved) == resolved_key)
        {
            belongs_to_project = true;
            break;
        }
    }
    if (!belongs_to_project)
    {
        emit self_guard->interactiveMaskSaveFailed(
            imagePath, revision, QStringLiteral("当前照片不属于活动项目，未保存蒙版。"));
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    const QString final_path = xjw::common::project::ProjectIO::maskOutputPathForImage(project_path, resolved_image);
    const QString staging_path = uniqueSiblingPath(final_path, QStringLiteral("staging.png"));
    if (final_path.isEmpty() || !QDir().mkpath(QFileInfo(final_path).absolutePath()))
    {
        emit self_guard->interactiveMaskSaveFailed(
            resolved_image,
            revision,
            QStringLiteral("无法创建蒙版输出目录：%1").arg(QFileInfo(final_path).absolutePath()));
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }
    self_guard->rememberOwnedArtifact(staging_path);
    ++self_guard->_runningWorkers;
    const QImage frozen_mask = mask.convertToFormat(QImage::Format_Grayscale8);
    const std::function<void(int)> staged_observer = self_guard->_artifactStagedObserverForTesting;
    QFuture<void> future = xjw::gui::tasks::runGuardedWithOutcome(
        self_guard.data(),
        [frozen_mask, resolved_image, final_path, staging_path, method, context, staged_observer]()
        {
            InteractiveMaskSaveResult result;
            if (!context.cancelFlag || context.cancelFlag->load(std::memory_order_relaxed))
            {
                result.cancelled = true;
                return result;
            }
            QSaveFile output(staging_path);
            if (!output.open(QIODevice::WriteOnly))
            {
                result.errorMessage = QStringLiteral("无法写入蒙版：%1").arg(staging_path);
                return result;
            }
            if (!frozen_mask.save(&output, "PNG"))
            {
                output.cancelWriting();
                result.errorMessage = QStringLiteral("蒙版 PNG 编码失败：%1").arg(staging_path);
                return result;
            }
            if (!output.commit())
            {
                result.errorMessage = QStringLiteral("无法提交蒙版暂存文件：%1").arg(staging_path);
                return result;
            }
            if (context.cancelFlag->load(std::memory_order_relaxed))
            {
                QFile::remove(staging_path);
                result.cancelled = true;
                return result;
            }
            QJsonObject record{
                {QStringLiteral("mask_path"), QDir::cleanPath(final_path)},
                {QStringLiteral("mask_method"), method},
                {QStringLiteral("mask_updated_at"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)}};
            result.artifact = StagedMaskArtifact{resolved_image, final_path, staging_path, record};
            if (staged_observer)
            {
                staged_observer(1);
            }
            return result;
        },
        [context, resolved_image, revision, staging_path](
            ProjectMaskWorkflowController* self, xjw::gui::tasks::TaskOutcome<InteractiveMaskSaveResult> outcome)
        {
            self->pruneFinishedFutures();
            const QPointer<ProjectMaskWorkflowController> self_guard(self);
            const QPointer<xjw::gui::project::ProjectTaskOrchestrator> owner_guard(maskOwner(self));
            const QPointer<xjw::gui::project::ProjectSession> session_guard(self->_session);
            --self->_runningWorkers;
            if (!owner_guard || !owner_guard->maskContextIsLive(context) || !outcome.succeeded() ||
                outcome.value->cancelled || !outcome.value->artifact.has_value())
            {
                self->removeOwnedArtifact(staging_path);
                if (owner_guard && owner_guard->maskContextIsLive(context) && outcome.succeeded() &&
                    !outcome.value->cancelled)
                {
                    emit self_guard->interactiveMaskSaveFailed(resolved_image, revision, outcome.value->errorMessage);
                }
                else if (owner_guard && owner_guard->maskContextIsLive(context) && !outcome.succeeded())
                {
                    emit self_guard->interactiveMaskSaveFailed(resolved_image, revision, outcome.errorMessage);
                }
                if (!self_guard || !owner_guard)
                {
                    return;
                }
                if (owner_guard)
                {
                    owner_guard->settleMaskOperation(context);
                }
                return;
            }

            const PublishMaskResult publication =
                self_guard->publishStagedArtifacts(context, {*outcome.value->artifact}, false);
            if (!self_guard || !owner_guard)
            {
                return;
            }
            if (publication.committed && publication.tailAllowed && owner_guard->maskTailIsLive(context))
            {
                if (!publication.metadataDeltaImages.isEmpty())
                {
                    emit self_guard->projectMetadataUpdated(context.session.projectPath);
                    if (!self_guard || !owner_guard)
                    {
                        return;
                    }
                    if (!session_guard)
                    {
                        owner_guard->settleMaskOperation(context);
                        return;
                    }
                }
                if (owner_guard->maskTailIsLive(context))
                {
                    emit self_guard->interactiveMaskSaved(resolved_image, revision);
                }
            }
            else if (owner_guard->maskTailIsLive(context))
            {
                emit self_guard->interactiveMaskSaveFailed(resolved_image, revision, publication.errorMessage);
            }
            if (!self_guard || !owner_guard)
            {
                return;
            }
            if (!session_guard)
            {
                owner_guard->settleMaskOperation(context);
                return;
            }
            owner_guard->settleMaskOperation(context);
        });
    if (self_guard)
    {
        self_guard->registerFuture(std::move(future));
    }
}

void ProjectMaskWorkflowController::clearMasksForImages(xjw::gui::project::ProjectTaskContext context,
                                                        const QStringList& requestedImages)
{
    using xjw::gui::project::ProjectTaskOrchestrator;
    const QPointer<ProjectMaskWorkflowController> self_guard(this);
    const QPointer<ProjectTaskOrchestrator> owner_guard(maskOwner(this));
    const QPointer<xjw::gui::project::ProjectSession> session_guard(_session);
    ProjectUiMessageAdapter* const messages = _messages;
    const auto settle = [owner_guard, context]()
    {
        if (owner_guard)
        {
            owner_guard->settleMaskOperation(context);
        }
    };
    if (!owner_guard || !session_guard || !owner_guard->maskContextIsLive(context))
    {
        settle();
        return;
    }
    if (!session_guard->hasProject())
    {
        if (messages)
        {
            messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("请先打开项目，再清除照片蒙版。"));
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    const QString project_path = session_guard->projectPath();
    QHash<QString, QString> project_images;
    for (const QString& path : session_guard->allImages())
    {
        const QString resolved =
            QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, path));
        const QString key = xjw::common::project::normalizePath(resolved);
        if (!key.isEmpty() && !project_images.contains(key))
        {
            project_images.insert(key, resolved);
        }
    }
    QSet<QString> selected_keys;
    for (const QString& path : requestedImages)
    {
        const QString resolved = xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, path);
        const QString key = xjw::common::project::normalizePath(resolved);
        if (project_images.contains(key))
        {
            selected_keys.insert(key);
        }
    }
    if (selected_keys.isEmpty())
    {
        if (messages && owner_guard->maskContextIsLive(context))
        {
            messages->warning(nullptr, QStringLiteral("清除蒙版"), QStringLiteral("没有选中可处理的照片。"));
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    const QString masks_directory = xjw::common::project::ProjectIO::maskOutputDir(project_path);
    std::optional<ManagedMaskRoot> authoritative_root = resolveManagedMaskRoot(masks_directory);
    const auto project_snapshot = [&project_images, &project_path, &selected_keys](
                                      const QJsonObject& metadata, const std::optional<ManagedMaskRoot>& managedRoot)
    {
        QVector<ProjectMaskSnapshot> result;
        const QJsonArray images = metadata.value(QStringLiteral("images")).toArray();
        result.reserve(images.size());
        for (const QJsonValue& value : images)
        {
            const QJsonObject image = value.toObject();
            const QString resolved_image =
                normalizedAbsolutePath(xjw::common::project::ProjectIO::resolveProjectResourcePath(
                    project_path, image.value(QStringLiteral("path")).toString()));
            const QString image_key = xjw::common::project::normalizePath(resolved_image);
            if (image_key.isEmpty())
            {
                continue;
            }

            ProjectMaskSnapshot snapshot;
            snapshot.imagePath = project_images.value(image_key, resolved_image);
            snapshot.imageKey = image_key;
            snapshot.selected = selected_keys.contains(image_key);
            snapshot.hasMaskState = hasMaskMetadata(image);
            snapshot.maskRecord = maskRecordFromImage(image);

            const QString stored_mask_value = image.value(QStringLiteral("mask_path")).toString().trimmed();
            if (!stored_mask_value.isEmpty())
            {
                const QString stored_mask = normalizedAbsolutePath(
                    xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, stored_mask_value));
                if (managedRoot)
                {
                    snapshot.explicitMaskPath = resolveManagedMaskPath(stored_mask, *managedRoot);
                }
            }

            const QString standard_mask = normalizedAbsolutePath(
                xjw::common::project::ProjectIO::maskOutputPathForImage(project_path, snapshot.imagePath));
            if (managedRoot)
            {
                if (const auto resolved_standard = resolveManagedMaskPath(standard_mask, *managedRoot))
                {
                    snapshot.standardMaskPath = *resolved_standard;
                }
            }
            result.push_back(std::move(snapshot));
        }
        return result;
    };
    const auto selected_targets = [](const QVector<ProjectMaskSnapshot>& snapshots)
    {
        QVector<ClearMaskTarget> result;
        for (const ProjectMaskSnapshot& snapshot : snapshots)
        {
            if (!snapshot.selected || (!snapshot.hasMaskState && (snapshot.standardMaskPath.physicalPath.isEmpty() ||
                                                                  !managedMaskPathExists(snapshot.standardMaskPath))))
            {
                continue;
            }
            ClearMaskTarget target;
            target.imagePath = snapshot.imagePath;
            QSet<QString> seen_paths;
            if (snapshot.explicitMaskPath && !seen_paths.contains(snapshot.explicitMaskPath->mutationKey))
            {
                seen_paths.insert(snapshot.explicitMaskPath->mutationKey);
                target.managedMaskPaths.push_back(*snapshot.explicitMaskPath);
                addManagedMaskPathIdentities(*snapshot.explicitMaskPath, &target.identityKeys);
            }
            if (!snapshot.standardMaskPath.physicalPath.isEmpty() &&
                !seen_paths.contains(snapshot.standardMaskPath.mutationKey))
            {
                seen_paths.insert(snapshot.standardMaskPath.mutationKey);
                target.managedMaskPaths.push_back(snapshot.standardMaskPath);
                addManagedMaskPathIdentities(snapshot.standardMaskPath, &target.identityKeys);
            }
            result.push_back(std::move(target));
        }
        return result;
    };

    QVector<ProjectMaskSnapshot> snapshots = project_snapshot(session_guard->coreMetadata(), authoritative_root);
    QVector<ClearMaskTarget> targets = selected_targets(snapshots);
    if (targets.isEmpty())
    {
        if (messages && owner_guard->maskContextIsLive(context))
        {
            messages->information(nullptr, QStringLiteral("清除蒙版"), QStringLiteral("所选照片没有可清除的蒙版。"));
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    const QString confirmation =
        targets.size() == 1 ? QStringLiteral("确定清除“%1”的蒙版吗？此操作无法撤销。")
                                  .arg(QFileInfo(targets.constFirst().imagePath).fileName())
                            : QStringLiteral("确定清除所选 %1 张照片的蒙版吗？此操作无法撤销。").arg(targets.size());
    const UiAnswer answer =
        messages ? messages->question(nullptr, QStringLiteral("清除蒙版"), confirmation, UiAnswer::No) : UiAnswer::No;
    if (!self_guard || !owner_guard)
    {
        return;
    }
    if (!session_guard)
    {
        settle();
        return;
    }
    if (!owner_guard->maskContextIsLive(context) || answer != UiAnswer::Yes)
    {
        settle();
        return;
    }

    // The confirmation dialog may run a nested event loop. Re-read current mask
    // records only after it returns so same-generation edits are not overwritten.
    authoritative_root = resolveManagedMaskRoot(masks_directory);
    snapshots = project_snapshot(session_guard->coreMetadata(), authoritative_root);
    targets = selected_targets(snapshots);
    if (!owner_guard->maskContextIsLive(context))
    {
        settle();
        return;
    }
    if (targets.isEmpty())
    {
        if (messages)
        {
            messages->information(nullptr, QStringLiteral("清除蒙版"), QStringLiteral("所选照片没有可清除的蒙版。"));
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    QSet<QString> protected_identities;
    for (const ProjectMaskSnapshot& snapshot : std::as_const(snapshots))
    {
        if (snapshot.selected)
        {
            continue;
        }
        if (snapshot.explicitMaskPath)
        {
            addManagedMaskPathIdentities(*snapshot.explicitMaskPath, &protected_identities);
        }
        if (!snapshot.standardMaskPath.physicalPath.isEmpty() && managedMaskPathExists(snapshot.standardMaskPath))
        {
            addManagedMaskPathIdentities(snapshot.standardMaskPath, &protected_identities);
        }
    }

    QVector<int> parents(targets.size());
    for (int index = 0; index < parents.size(); ++index)
    {
        parents[index] = index;
    }
    const auto find_root = [&parents](int index)
    {
        int root = index;
        while (parents.at(root) != root)
        {
            root = parents.at(root);
        }
        while (parents.at(index) != index)
        {
            const int parent = parents.at(index);
            parents[index] = root;
            index = parent;
        }
        return root;
    };
    const auto unite = [&parents, &find_root](int left, int right)
    {
        const int left_root = find_root(left);
        const int right_root = find_root(right);
        if (left_root != right_root)
        {
            parents[right_root] = left_root;
        }
    };
    QHash<QString, int> first_identity_owner;
    for (int index = 0; index < targets.size(); ++index)
    {
        for (const QString& identity : targets.at(index).identityKeys)
        {
            const auto owner = first_identity_owner.constFind(identity);
            if (owner == first_identity_owner.cend())
            {
                first_identity_owner.insert(identity, index);
            }
            else
            {
                unite(index, owner.value());
            }
        }
    }

    QVector<IsolatedMaskComponent> components;
    QHash<int, int> component_by_root;
    for (int index = 0; index < targets.size(); ++index)
    {
        const int root = find_root(index);
        int component_index = component_by_root.value(root, -1);
        if (component_index < 0)
        {
            component_index = components.size();
            component_by_root.insert(root, component_index);
            components.push_back({});
        }
        IsolatedMaskComponent& component = components[component_index];
        component.imagePaths.push_back(targets.at(index).imagePath);
        for (const ManagedMaskPath& path : targets.at(index).managedMaskPaths)
        {
            if (!component.mutationKeys.contains(path.mutationKey))
            {
                component.mutationKeys.insert(path.mutationKey);
                component.managedMaskPaths.push_back(path);
            }
        }
    }

    QStringList errors;
    QStringList recovery_failures;
    const auto discard_owned_copy =
        [self_guard](const QString& final_path, const QString& copy_path, bool injected_failure, QStringList* failures)
    {
        if (!self_guard || copy_path.isEmpty())
        {
            return;
        }
        if (QFileInfo::exists(copy_path) && (injected_failure || !QFile::remove(copy_path)))
        {
            if (failures)
            {
                failures->push_back(formatRecoveryCopyMapping(final_path, copy_path));
            }
            self_guard->forgetOwnedArtifact(copy_path);
            return;
        }
        self_guard->forgetOwnedArtifact(copy_path);
    };
    const auto restore_component =
        [self_guard, &discard_owned_copy](IsolatedMaskComponent* component, QStringList* failures)
    {
        if (!self_guard || !component)
        {
            return;
        }
        for (auto it = component->files.rbegin(); it != component->files.rend(); ++it)
        {
            QString recovery_source;
            if (QFileInfo::exists(it->tombstonePath))
            {
                recovery_source = it->tombstonePath;
            }
            else if (QFileInfo::exists(it->recoveryPath))
            {
                recovery_source = it->recoveryPath;
            }

            bool restored = QFileInfo::exists(it->finalPath);
            if (!restored && !recovery_source.isEmpty())
            {
                QString restore_error;
                restored = copyFileAtomically(recovery_source, it->finalPath, &restore_error);
            }
            if (!restored)
            {
                for (const QString& path : {it->tombstonePath, it->recoveryPath})
                {
                    if (!path.isEmpty() && QFileInfo::exists(path))
                    {
                        self_guard->forgetOwnedArtifact(path);
                        if (failures)
                        {
                            failures->push_back(formatRecoveryCopyMapping(it->finalPath, path));
                        }
                    }
                }
                continue;
            }

            for (const QString& path : {it->tombstonePath, it->recoveryPath})
            {
                discard_owned_copy(it->finalPath, path, false, failures);
            }
        }
    };

    const int injected_failure_index = self_guard->_clearRemovalFailureIndexForTesting;
    self_guard->_clearRemovalFailureIndexForTesting = -1;
    int removal_index = 0;
    QVector<IsolatedMaskComponent> ready_components;
    QStringList filesystem_success_images;
    const auto path_is_protected = [&protected_identities](const ManagedMaskPath& path)
    {
        return protected_identities.contains(path.mutationKey) ||
               (!path.referenceTargetKey.isEmpty() && protected_identities.contains(path.referenceTargetKey));
    };
    for (const IsolatedMaskComponent& planned_component : std::as_const(components))
    {
        IsolatedMaskComponent component = planned_component;
        const QString component_label = [&component]()
        {
            QStringList names;
            for (const QString& image_path : component.imagePaths)
            {
                names.push_back(QFileInfo(image_path).fileName());
            }
            return names.join(QStringLiteral("、"));
        }();
        QString failed_path;
        QString validation_error;
        QVector<ManagedMaskPath> files_to_isolate;
        bool has_mutable_candidate = false;
        for (const ManagedMaskPath& path : component.managedMaskPaths)
        {
            if (!path.leafLink && !path_is_protected(path))
            {
                has_mutable_candidate = true;
                break;
            }
        }
        if (has_mutable_candidate)
        {
            const std::optional<ManagedMaskRoot> current_root =
                authoritative_root ? resolveManagedMaskRoot(authoritative_root->sourcePath) : std::nullopt;
            if (!authoritative_root || !current_root || current_root->key != authoritative_root->key)
            {
                failed_path = masks_directory;
                validation_error = QStringLiteral("蒙版目录在确认后发生变化 %1").arg(failed_path);
            }
            else
            {
                for (const ManagedMaskPath& path : component.managedMaskPaths)
                {
                    if (path.leafLink || path_is_protected(path))
                    {
                        continue;
                    }
                    const std::optional<ManagedMaskPath> current_path =
                        resolveManagedMaskPath(path.sourcePath, *current_root);
                    if (!current_path || current_path->leafLink || current_path->mutationKey != path.mutationKey)
                    {
                        failed_path = path.sourcePath;
                        validation_error = QStringLiteral("蒙版路径在确认后发生变化 %1").arg(failed_path);
                        break;
                    }

                    const QFileInfo info(path.physicalPath);
                    if (isLeafLink(info))
                    {
                        failed_path = path.physicalPath;
                        validation_error = QStringLiteral("蒙版路径在确认后变为链接 %1").arg(failed_path);
                        break;
                    }
                    if (info.exists() && !info.isFile())
                    {
                        failed_path = path.physicalPath;
                        validation_error = QStringLiteral("蒙版目标不是普通文件 %1").arg(failed_path);
                        break;
                    }
                    if (info.exists())
                    {
                        files_to_isolate.push_back(path);
                    }
                }
            }
        }
        if (!validation_error.isEmpty())
        {
            errors.push_back(QStringLiteral("%1：%2").arg(component_label, validation_error));
            continue;
        }

        for (const ManagedMaskPath& path : std::as_const(files_to_isolate))
        {
            const QString tombstone = uniqueSiblingPath(path.physicalPath, QStringLiteral("tombstone"));
            self_guard->rememberOwnedArtifact(tombstone);
            if (!QFile::rename(path.physicalPath, tombstone))
            {
                self_guard->forgetOwnedArtifact(tombstone);
                failed_path = path.physicalPath;
                break;
            }
            component.files.push_back({path.physicalPath, tombstone, {}});
        }
        if (!failed_path.isEmpty())
        {
            restore_component(&component, &recovery_failures);
            errors.push_back(QStringLiteral("%1：无法隔离 %2").arg(component_label, failed_path));
            continue;
        }

        QString preparation_error;
        for (IsolatedMaskFile& file : component.files)
        {
            file.recoveryPath = uniqueSiblingPath(file.finalPath, QStringLiteral("recovery"));
            self_guard->rememberOwnedArtifact(file.recoveryPath);
            if (!copyFileAtomically(file.tombstonePath, file.recoveryPath, &preparation_error))
            {
                break;
            }
        }
        if (!preparation_error.isEmpty())
        {
            restore_component(&component, &recovery_failures);
            errors.push_back(QStringLiteral("%1：%2").arg(component_label, preparation_error));
            continue;
        }

        failed_path.clear();
        for (IsolatedMaskFile& file : component.files)
        {
            const bool injected_failure = removal_index++ == injected_failure_index;
            if (injected_failure || !QFile::remove(file.tombstonePath))
            {
                failed_path = file.finalPath;
                break;
            }
            self_guard->forgetOwnedArtifact(file.tombstonePath);
        }
        if (!failed_path.isEmpty())
        {
            restore_component(&component, &recovery_failures);
            errors.push_back(QStringLiteral("%1：无法删除 %2").arg(component_label, failed_path));
            continue;
        }
        filesystem_success_images.append(component.imagePaths);
        ready_components.push_back(std::move(component));
    }
    if (!owner_guard->maskContextIsLive(context))
    {
        for (IsolatedMaskComponent& component : ready_components)
        {
            restore_component(&component, &recovery_failures);
        }
        self_guard->showRollbackFailures(context, recovery_failures);
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    QStringList cleared_images;
    QString port_error;
    const bool committed =
        session_guard->clearImageMaskRecords(context.session, filesystem_success_images, &cleared_images, &port_error);
    if (!self_guard || !owner_guard)
    {
        return;
    }
    if (!committed)
    {
        for (IsolatedMaskComponent& component : ready_components)
        {
            restore_component(&component, &recovery_failures);
        }
        self_guard->showRollbackFailures(context, recovery_failures);
        if (!self_guard || !owner_guard)
        {
            return;
        }
        if (!session_guard)
        {
            settle();
            return;
        }
        if (messages && owner_guard->maskContextIsLive(context))
        {
            messages->warning(nullptr, QStringLiteral("清除蒙版"), port_error);
        }
        if (!self_guard || !owner_guard)
        {
            return;
        }
        settle();
        return;
    }

    QStringList cleanup_failures = recovery_failures;
    const int injected_cleanup_failure_index = self_guard->_clearRecoveryCleanupFailureIndexForTesting;
    self_guard->_clearRecoveryCleanupFailureIndexForTesting = -1;
    int cleanup_index = 0;
    for (const IsolatedMaskComponent& component : std::as_const(ready_components))
    {
        for (const IsolatedMaskFile& file : component.files)
        {
            const bool injected_failure = cleanup_index++ == injected_cleanup_failure_index;
            discard_owned_copy(file.finalPath, file.recoveryPath, injected_failure, &cleanup_failures);
        }
    }
    self_guard->showRollbackFailures(context, cleanup_failures);
    if (!self_guard || !owner_guard)
    {
        return;
    }
    if (!session_guard)
    {
        settle();
        return;
    }
    if (!owner_guard->maskTailIsLive(context))
    {
        settle();
        return;
    }
    if (!cleared_images.isEmpty())
    {
        emit self_guard->projectMetadataUpdated(project_path);
        if (!self_guard || !owner_guard)
        {
            return;
        }
        if (!session_guard)
        {
            settle();
            return;
        }
    }
    if (!filesystem_success_images.isEmpty() && owner_guard->maskTailIsLive(context))
    {
        emit self_guard->masksGenerated(filesystem_success_images);
        if (!self_guard || !owner_guard)
        {
            return;
        }
        if (!session_guard)
        {
            settle();
            return;
        }
    }
    if (!owner_guard->maskTailIsLive(context))
    {
        settle();
        return;
    }
    if (!errors.isEmpty())
    {
        if (messages)
        {
            messages->warning(nullptr,
                              QStringLiteral("清除蒙版"),
                              QStringLiteral("已清除 %1 张照片的蒙版。\n以下蒙版未能清除，项目记录已保留：\n%2")
                                  .arg(filesystem_success_images.size())
                                  .arg(errors.join(QLatin1Char('\n'))));
        }
    }
    else if (messages)
    {
        messages->information(nullptr,
                              QStringLiteral("清除蒙版"),
                              QStringLiteral("已清除 %1 张照片的蒙版。").arg(filesystem_success_images.size()));
    }
    if (!self_guard || !owner_guard)
    {
        return;
    }
    if (!session_guard)
    {
        settle();
        return;
    }
    settle();
}

PublishMaskResult ProjectMaskWorkflowController::publishStagedArtifacts(xjw::gui::project::ProjectTaskContext context,
                                                                        const QVector<StagedMaskArtifact>& artifacts,
                                                                        bool allowUserBatchCancel)
{
    PublishMaskResult result;
    const QPointer<ProjectMaskWorkflowController> self_guard(this);
    const QPointer<xjw::gui::project::ProjectTaskOrchestrator> owner_guard(maskOwner(this));
    const QPointer<xjw::gui::project::ProjectSession> session_guard(_session);
    if (!owner_guard || !session_guard || artifacts.isEmpty() ||
        !owner_guard->maskContextIsLive(context, allowUserBatchCancel))
    {
        result.errorMessage = QStringLiteral("项目会话已变化，蒙版结果未发布。");
        for (const StagedMaskArtifact& artifact : artifacts)
        {
            removeOwnedArtifact(artifact.stagingPath);
        }
        return result;
    }

    const auto discard_owned_file = [self_guard](const QString& path, QStringList* failures)
    {
        if (!self_guard || path.isEmpty())
        {
            return;
        }
        if (QFileInfo::exists(path) && !QFile::remove(path))
        {
            self_guard->forgetOwnedArtifact(path);
            if (failures)
            {
                failures->push_back(path);
            }
            return;
        }
        self_guard->forgetOwnedArtifact(path);
    };

    QVector<PublishedMaskArtifact> published;
    published.reserve(artifacts.size());
    QSet<QString> final_path_keys;
    for (const StagedMaskArtifact& artifact : artifacts)
    {
        const QString final_path = normalizedAbsolutePath(artifact.finalPath);
        const QString staging_path = normalizedAbsolutePath(artifact.stagingPath);
        const QString final_key = managedPathKey(final_path);
        const QFileInfo staging_info(staging_path);
        const QFileInfo final_info(final_path);
        if (artifact.imagePath.trimmed().isEmpty() || final_path.isEmpty() || staging_path.isEmpty() ||
            !staging_info.exists() || !staging_info.isFile())
        {
            result.errorMessage = QStringLiteral("蒙版暂存文件不可用：%1").arg(artifact.stagingPath);
            break;
        }
        if (final_key.isEmpty() || final_path_keys.contains(final_key))
        {
            result.errorMessage = QStringLiteral("蒙版发布目标重复：%1").arg(artifact.finalPath);
            break;
        }
        if (final_info.exists() && !final_info.isFile())
        {
            result.errorMessage = QStringLiteral("蒙版目标不是普通文件：%1").arg(artifact.finalPath);
            break;
        }
        final_path_keys.insert(final_key);
        PublishedMaskArtifact item;
        item.staged = artifact;
        item.staged.finalPath = final_path;
        item.staged.stagingPath = staging_path;
        item.hadOriginal = final_info.exists();
        if (item.hadOriginal)
        {
            item.backupPath = uniqueSiblingPath(final_path, QStringLiteral("backup"));
            rememberOwnedArtifact(item.backupPath);
        }
        published.push_back(std::move(item));
    }

    QStringList retained_artifacts;
    const auto cleanup_unpublished = [&]()
    {
        for (const PublishedMaskArtifact& item : std::as_const(published))
        {
            discard_owned_file(item.backupPath, &retained_artifacts);
        }
        for (const StagedMaskArtifact& artifact : artifacts)
        {
            discard_owned_file(artifact.stagingPath, &retained_artifacts);
        }
    };
    if (!result.errorMessage.isEmpty() || published.size() != artifacts.size())
    {
        cleanup_unpublished();
        showRollbackFailures(context, retained_artifacts);
        return result;
    }

    for (PublishedMaskArtifact& item : published)
    {
        if (!item.hadOriginal)
        {
            continue;
        }
        QString backup_error;
        if (!copyFileAtomically(item.staged.finalPath, item.backupPath, &backup_error))
        {
            result.errorMessage = QStringLiteral("无法备份已有蒙版 %1：%2").arg(item.staged.finalPath, backup_error);
            break;
        }
    }
    if (!result.errorMessage.isEmpty())
    {
        cleanup_unpublished();
        showRollbackFailures(context, retained_artifacts);
        return result;
    }

    const auto publication_observer = _publicationObserverForTesting;
    if (publication_observer)
    {
        for (const PublishedMaskArtifact& item : std::as_const(published))
        {
            try
            {
                publication_observer(QStringLiteral("backups_prepared"), item.staged, item.backupPath);
            }
            catch (const std::exception& exception)
            {
                result.errorMessage = QStringLiteral("蒙版发布观察器失败：%1").arg(QString::fromUtf8(exception.what()));
            }
            catch (...)
            {
                result.errorMessage = QStringLiteral("蒙版发布观察器失败。");
            }
            if (!self_guard || !owner_guard)
            {
                return result;
            }
            if (!session_guard || !owner_guard->maskContextIsLive(context, allowUserBatchCancel))
            {
                result.errorMessage = QStringLiteral("项目会话已变化，蒙版结果未发布。");
            }
            if (!result.errorMessage.isEmpty())
            {
                break;
            }
        }
    }
    if (!result.errorMessage.isEmpty())
    {
        cleanup_unpublished();
        showRollbackFailures(context, retained_artifacts);
        return result;
    }

    for (PublishedMaskArtifact& item : published)
    {
        const StagedMaskArtifact& artifact = item.staged;
        if (!owner_guard->maskContextIsLive(context, allowUserBatchCancel))
        {
            result.errorMessage = QStringLiteral("项目会话已变化，蒙版结果未发布。");
            break;
        }
        QString publish_error;
        if (!copyFileAtomically(artifact.stagingPath, artifact.finalPath, &publish_error))
        {
            result.errorMessage = publish_error;
            break;
        }
        item.published = true;
        discard_owned_file(artifact.stagingPath, &retained_artifacts);
    }

    if (!result.errorMessage.isEmpty())
    {
        QStringList rollback_failures = retained_artifacts;
        rollbackPublishedArtifacts(context, &published, &rollback_failures, allowUserBatchCancel);
        if (!self_guard || !owner_guard)
        {
            return result;
        }
        for (const StagedMaskArtifact& artifact : artifacts)
        {
            discard_owned_file(artifact.stagingPath, &rollback_failures);
        }
        showRollbackFailures(context, rollback_failures);
        return result;
    }

    QMap<QString, QJsonObject> records;
    for (const PublishedMaskArtifact& item : published)
    {
        records.insert(item.staged.imagePath, item.staged.record);
    }
    QString port_error;
    QStringList updated_images;
    const bool port_committed =
        session_guard->publishImageMaskRecords(context.session, records, &updated_images, &port_error);
    result.committed = port_committed;
    if (port_committed)
    {
        for (const PublishedMaskArtifact& item : std::as_const(published))
        {
            result.filesystemSucceededImages.push_back(item.staged.imagePath);
        }
        result.metadataDeltaImages = updated_images;
    }
    if (!self_guard || !owner_guard)
    {
        return result;
    }
    if (!port_committed)
    {
        result.errorMessage = port_error;
        QStringList rollback_failures;
        rollbackPublishedArtifacts(context, &published, &rollback_failures, allowUserBatchCancel);
        if (!self_guard || !owner_guard)
        {
            return result;
        }
        showRollbackFailures(context, rollback_failures);
        if (self_guard && owner_guard && !session_guard)
        {
            owner_guard->settleMaskOperation(context);
        }
        return result;
    }

    QStringList backup_delete_failures = retained_artifacts;
    for (const PublishedMaskArtifact& item : std::as_const(published))
    {
        if (item.backupPath.isEmpty())
        {
            continue;
        }
        if (QFileInfo::exists(item.backupPath) && !QFile::remove(item.backupPath))
        {
            backup_delete_failures.push_back(
                QStringLiteral("目标=%1，保留备份=%2").arg(item.staged.finalPath, item.backupPath));
            forgetOwnedArtifact(item.backupPath);
        }
        else
        {
            forgetOwnedArtifact(item.backupPath);
        }
    }
    showRollbackFailures(context, backup_delete_failures);
    if (!self_guard || !owner_guard)
    {
        return result;
    }
    if (!session_guard)
    {
        owner_guard->settleMaskOperation(context);
        return result;
    }
    result.tailAllowed = owner_guard->maskTailIsLive(context);
    return result;
}

bool ProjectMaskWorkflowController::rollbackPublishedArtifacts(xjw::gui::project::ProjectTaskContext context,
                                                               QVector<PublishedMaskArtifact>* artifacts,
                                                               QStringList* failures,
                                                               bool allowUserBatchCancel)
{
    if (!artifacts)
    {
        return true;
    }

    const QPointer<ProjectMaskWorkflowController> self_guard(this);
    const QPointer<xjw::gui::project::ProjectTaskOrchestrator> owner_guard(maskOwner(this));
    const QPointer<xjw::gui::project::ProjectSession> session_guard(_session);
    const auto publication_observer = _publicationObserverForTesting;
    QVector<RecoveryCopyMapping> recovery_mappings;

    // A test observer is an external boundary and may destroy the controller.
    // Detach every recovery copy first so destruction cannot erase the only
    // byte-for-byte rollback source while the nested callback is running.
    if (publication_observer)
    {
        recovery_mappings.reserve(artifacts->size());
        for (const PublishedMaskArtifact& item : std::as_const(*artifacts))
        {
            if (!item.backupPath.isEmpty())
            {
                recovery_mappings.push_back({item.staged.finalPath, item.backupPath});
                forgetOwnedArtifact(item.backupPath);
            }
        }
    }

    bool restored = true;
    for (auto it = artifacts->rbegin(); it != artifacts->rend(); ++it)
    {
        if (it->published && it->hadOriginal)
        {
            if (publication_observer)
            {
                try
                {
                    publication_observer(QStringLiteral("before_rollback_restore"), it->staged, it->backupPath);
                }
                catch (const std::exception& exception)
                {
                    restored = false;
                    if (failures)
                    {
                        failures->push_back(
                            QStringLiteral("回滚观察器失败：%1").arg(QString::fromUtf8(exception.what())));
                    }
                }
                catch (...)
                {
                    restored = false;
                    if (failures)
                    {
                        failures->push_back(QStringLiteral("回滚观察器失败。"));
                    }
                }
                if (!self_guard || !owner_guard)
                {
                    QStringList retained_recoveries;
                    for (const RecoveryCopyMapping& mapping : std::as_const(recovery_mappings))
                    {
                        if (QFileInfo::exists(mapping.copyPath))
                        {
                            retained_recoveries.push_back(
                                formatRecoveryCopyMapping(mapping.finalPath, mapping.copyPath));
                        }
                    }
                    if (failures)
                    {
                        failures->append(retained_recoveries);
                    }
                    logRetainedRecoveryFailures(retained_recoveries);
                    return false;
                }

                // Re-evaluate every lifetime/lane/cancel boundary after the
                // callback. Rollback remains mandatory even when it invalidated
                // the operation because publication has already changed disk.
                const bool boundary_live =
                    session_guard && owner_guard->maskContextIsLive(context, allowUserBatchCancel);
                Q_UNUSED(boundary_live);
            }

            QString restore_error;
            if (!copyFileAtomically(it->backupPath, it->staged.finalPath, &restore_error))
            {
                restored = false;
                forgetOwnedArtifact(it->backupPath);
                if (failures)
                {
                    failures->push_back(formatRecoveryCopyMapping(it->staged.finalPath, it->backupPath) +
                                        QStringLiteral("，恢复失败=%1").arg(restore_error));
                }
            }
            else
            {
                if (QFileInfo::exists(it->backupPath) && !QFile::remove(it->backupPath))
                {
                    restored = false;
                    forgetOwnedArtifact(it->backupPath);
                    if (failures)
                    {
                        failures->push_back(formatRecoveryCopyMapping(it->staged.finalPath, it->backupPath));
                    }
                }
                else
                {
                    forgetOwnedArtifact(it->backupPath);
                }
            }
        }
        else if (it->published && QFileInfo::exists(it->staged.finalPath) && !QFile::remove(it->staged.finalPath))
        {
            restored = false;
            if (failures)
            {
                failures->push_back(QStringLiteral("无法移除事务新建目标=%1").arg(it->staged.finalPath));
            }
        }
        else if (it->hadOriginal && !it->backupPath.isEmpty())
        {
            if (QFileInfo::exists(it->backupPath) && !QFile::remove(it->backupPath))
            {
                restored = false;
                forgetOwnedArtifact(it->backupPath);
                if (failures)
                {
                    failures->push_back(
                        QStringLiteral("未发布目标=%1，保留备份=%2").arg(it->staged.finalPath, it->backupPath));
                }
            }
            else
            {
                forgetOwnedArtifact(it->backupPath);
            }
        }

        if (QFileInfo::exists(it->staged.stagingPath) && !QFile::remove(it->staged.stagingPath))
        {
            restored = false;
            forgetOwnedArtifact(it->staged.stagingPath);
            if (failures)
            {
                failures->push_back(QStringLiteral("保留暂存=%1").arg(it->staged.stagingPath));
            }
        }
        else
        {
            forgetOwnedArtifact(it->staged.stagingPath);
        }
    }
    return restored;
}

void ProjectMaskWorkflowController::showRollbackFailures(xjw::gui::project::ProjectTaskContext context,
                                                         const QStringList& failures)
{
    if (failures.isEmpty())
    {
        return;
    }
    const QPointer<ProjectMaskWorkflowController> self_guard(this);
    const QPointer<xjw::gui::project::ProjectTaskOrchestrator> owner_guard(maskOwner(this));
    ProjectUiMessageAdapter* const messages = _messages;
    const QString message = logRetainedRecoveryFailures(failures);
    if (messages && self_guard && owner_guard && owner_guard->maskTailIsLive(context))
    {
        messages->critical(nullptr, QStringLiteral("蒙版文件恢复失败"), message);
    }
}

void ProjectMaskWorkflowController::cancelActiveTask()
{
}

void ProjectMaskWorkflowController::waitForActiveTask()
{
    for (QFuture<void>& future : _futures)
    {
        future.waitForFinished();
    }
    _futures.clear();
}

bool ProjectMaskWorkflowController::hasRunningTask() const
{
    if (_runningWorkers > 0)
    {
        return true;
    }
    return std::any_of(
        _futures.cbegin(), _futures.cend(), [](const QFuture<void>& future) { return !future.isFinished(); });
}

bool ProjectMaskWorkflowController::hasPendingWork() const noexcept
{
    return std::any_of(_futures.cbegin(),
                       _futures.cend(),
                       [](const QFuture<void>& future) { return future.isValid() && !future.isFinished(); });
}

void ProjectMaskWorkflowController::pruneFinishedFutures()
{
    _futures.erase(std::remove_if(_futures.begin(),
                                  _futures.end(),
                                  [](const QFuture<void>& future) {
                                      return future.isFinished() ||
                                             (!future.isValid() && !future.isStarted() && !future.isRunning());
                                  }),
                   _futures.end());
}

void ProjectMaskWorkflowController::registerFuture(QFuture<void> future)
{
    pruneFinishedFutures();
    _futures.push_back(std::move(future));
    pruneFinishedFutures();
}

void ProjectMaskWorkflowController::trackFutureForTesting(QFuture<void> future)
{
    registerFuture(std::move(future));
}

void ProjectMaskWorkflowController::setArtifactStagedObserverForTesting(std::function<void(int)> observer)
{
    _artifactStagedObserverForTesting = std::move(observer);
}

void ProjectMaskWorkflowController::setClearRemovalFailureIndexForTesting(int index)
{
    _clearRemovalFailureIndexForTesting = index;
}

void ProjectMaskWorkflowController::setClearRecoveryCleanupFailureIndexForTesting(int index)
{
    _clearRecoveryCleanupFailureIndexForTesting = index;
}

void ProjectMaskWorkflowController::setPublicationObserverForTesting(
    std::function<void(const QString&, const StagedMaskArtifact&, const QString&)> observer)
{
    _publicationObserverForTesting = std::move(observer);
}

int ProjectMaskWorkflowController::ownedArtifactCountForTesting() const
{
    return _ownedArtifacts.size();
}

int ProjectMaskWorkflowController::futureCountForTesting() const
{
    return _futures.size();
}

void ProjectMaskWorkflowController::rememberOwnedArtifact(const QString& path)
{
    if (!path.isEmpty())
    {
        _ownedArtifacts.insert(QDir::cleanPath(path));
    }
}

void ProjectMaskWorkflowController::forgetOwnedArtifact(const QString& path)
{
    if (!path.isEmpty())
    {
        _ownedArtifacts.remove(QDir::cleanPath(path));
    }
}

void ProjectMaskWorkflowController::removeOwnedArtifact(const QString& path)
{
    if (path.isEmpty())
    {
        return;
    }
    const QString clean_path = QDir::cleanPath(path);
    if (_ownedArtifacts.contains(clean_path))
    {
        if (QFileInfo::exists(clean_path) && !QFile::remove(clean_path))
        {
            LOG_ERROR(QStringLiteral("无法清理蒙版任务临时文件：%1").arg(clean_path));
            return;
        }
        _ownedArtifacts.remove(clean_path);
    }
}

void ProjectMaskWorkflowController::cleanupOwnedArtifacts()
{
    const QSet<QString> paths = _ownedArtifacts;
    for (const QString& path : paths)
    {
        removeOwnedArtifact(path);
    }
}
