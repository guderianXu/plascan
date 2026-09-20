#include "ProjectSession.h"

#include "ProjectMetadataOperations.h"
#include "ProjectModelResultPolicy.h"
#include "ProjectTiePointResultService.h"
#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"
#include "project/ProjectSessionModel.h"
#include "camera/models/CameraModelFactories.h"
#include "camera/models/rpc/RpcInstance.h"
#include "camera/project/CameraProjectRuntime.h"

#include <QDir>
#include <QJsonArray>
#include <QSet>
#include <QThread>

#include <algorithm>

namespace xjw::gui::project
{

    namespace
    {

        const QSet<QString>& imageMaskRecordFields()
        {
            static const QSet<QString> fields{QStringLiteral("mask_path"),
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
            return fields;
        }

        QString resolvedImagePath(const QString& projectPath, const QJsonObject& image)
        {
            return QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(
                projectPath, image.value(QStringLiteral("path")).toString()));
        }

        void clearMaskFields(QJsonObject* image)
        {
            if (!image)
            {
                return;
            }
            for (const QString& field : imageMaskRecordFields())
            {
                image->remove(field);
            }
        }

        bool hasMaskFields(const QJsonObject& image)
        {
            return std::any_of(imageMaskRecordFields().cbegin(),
                               imageMaskRecordFields().cend(),
                               [&image](const QString& field) { return image.contains(field); });
        }

    } // namespace

    ProjectSession::ProjectSession(ProjectData* projectData, QObject* parent)
        : QObject(parent), _projectData(projectData)
    {
        if (!_projectData)
        {
            return;
        }

        const auto advanceAndNotify = [this]() { advanceGeneration(); };
        connect(_projectData,
                &ProjectData::projectOpened,
                this,
                [this, advanceAndNotify](const QString& projectPath)
                {
                    advanceAndNotify();
                    emit projectOpened(projectPath);
                });
        connect(_projectData, &ProjectData::projectSaved, this, &ProjectSession::projectSaved);
        connect(_projectData,
                &ProjectData::projectClosed,
                this,
                [this, advanceAndNotify]()
                {
                    advanceAndNotify();
                    emit projectClosed();
                });
        connect(_projectData,
                &ProjectData::activeChunkChanged,
                this,
                [advanceAndNotify](const QString&, const QString&, int) { advanceAndNotify(); });
        connect(_projectData, &ProjectData::chunkListChanged, this, &ProjectSession::chunkListChanged);
        connect(_projectData,
                &ProjectData::metadataChanged,
                this,
                [this](const QJsonObject&) { emit metadataChanged(metadata()); });
        connect(_projectData, &ProjectData::dirtyStateChanged, this, &ProjectSession::dirtyStateChanged);
    }

    ProjectData* ProjectSession::data() const
    {
        return _projectData;
    }

    bool ProjectSession::hasProject() const
    {
        return _projectData && _projectData->hasProject();
    }

    QString ProjectSession::projectPath() const
    {
        return _projectData ? _projectData->currentProjectPath() : QString();
    }

    QString ProjectSession::activeChunkId() const
    {
        return _projectData ? _projectData->activeChunkId() : QString();
    }

    ProjectSessionContext ProjectSession::context() const
    {
        return {projectPath(), activeChunkId(), _generation};
    }

    bool ProjectSession::isCurrent(const ProjectSessionContext& expected) const
    {
        return expected.matches(context());
    }

    void ProjectSession::advanceGeneration()
    {
        ++_generation;
        emit sessionChanged(context());
    }

    bool ProjectSession::tryBeginOperation(const QString& operation)
    {
        if (!_busyOperation.isEmpty())
        {
            return false;
        }
        _busyOperation = operation.trimmed().isEmpty() ? QStringLiteral("项目操作") : operation.trimmed();
        return true;
    }

    void ProjectSession::endOperation(const QString& operation)
    {
        if (_busyOperation.isEmpty() || operation.isEmpty() || _busyOperation == operation)
        {
            _busyOperation.clear();
        }
    }

    bool ProjectSession::isBusy() const
    {
        return !_busyOperation.isEmpty();
    }

    QString ProjectSession::busyOperation() const
    {
        return _busyOperation;
    }

    bool ProjectSession::isDirty() const
    {
        return hasProject() && _projectData->isDirty();
    }

    QJsonObject ProjectSession::metadata() const
    {
        if (!_projectData)
        {
            return {};
        }
        return ProjectTiePointResultService::metadataWithCurrentOnly(_projectData->metadataIncludingResults(),
                                                                     projectPath());
    }

    QJsonObject ProjectSession::coreMetadata() const
    {
        return _projectData ? _projectData->coreFilesMeta() : QJsonObject();
    }

    QStringList ProjectSession::imagesByCategory(const QString& category) const
    {
        return _projectData ? _projectData->getImagesByCategory(category) : QStringList();
    }

    QStringList ProjectSession::allImages() const
    {
        return _projectData ? _projectData->getAllImages() : QStringList();
    }

    QString ProjectSession::matchFile(const QString& firstImage, const QString& secondImage) const
    {
        return _projectData ? _projectData->findMatchFile(firstImage, secondImage) : QString();
    }

    std::unordered_map<xjw::camera_core::ImageId, xjw::camera_models::frame_pinhole::FramePinholeNumericState>
    ProjectSession::pinholeNumericStatesByImageId(const std::vector<xjw::camera_core::ImageId>& imageIds,
                                                  bool* hasCamerasForAll) const
    {
        if (hasCamerasForAll)
        {
            *hasCamerasForAll = true;
        }

        std::unordered_map<xjw::camera_core::ImageId, xjw::camera_models::frame_pinhole::FramePinholeNumericState>
            result;
        result.reserve(imageIds.size());
        if (!_projectData || imageIds.empty())
        {
            if (hasCamerasForAll)
            {
                *hasCamerasForAll = false;
            }
            return result;
        }

        const QJsonObject project_files = xjw::gui::project::projectFilesMeta(_projectData);
        const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
            project_files, xjw::camera_models::makeBuiltinCameraModelRegistry());
        if (!runtime.ok())
        {
            if (hasCamerasForAll)
            {
                *hasCamerasForAll = false;
            }
            return result;
        }

        std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> states;
        std::string error;
        if (!runtime.framePinholeStatesForImages(imageIds, &states, &error) || states.size() != imageIds.size())
        {
            if (hasCamerasForAll)
            {
                *hasCamerasForAll = false;
            }
            return result;
        }
        for (std::size_t index = 0; index < imageIds.size(); ++index)
        {
            result.emplace(imageIds[index], std::move(states[index]));
        }
        return result;
    }

    QMap<QString, xjw::camera_models::frame_pinhole::FramePinholeNumericState>
    ProjectSession::getPinholeNumericStatesForImages(const QStringList& images, bool* hasCamerasForAll) const
    {
        bool all_image_ids = false;
        const std::vector<xjw::camera_core::ImageId> image_ids = getImageIdsForImages(images, &all_image_ids);
        bool all_cameras = false;
        const auto states = pinholeNumericStatesByImageId(image_ids, &all_cameras);

        QMap<QString, xjw::camera_models::frame_pinhole::FramePinholeNumericState> result;
        for (std::size_t index = 0; index < image_ids.size() && index < static_cast<std::size_t>(images.size());
             ++index)
        {
            const auto state = states.find(image_ids[index]);
            if (state != states.end())
            {
                result.insert(xjw::common::project::normalizePath(images.at(static_cast<int>(index))), state->second);
            }
        }
        if (hasCamerasForAll)
        {
            *hasCamerasForAll = all_image_ids && all_cameras && result.size() == images.size();
        }
        return result;
    }

    std::vector<xjw::camera_core::ImageId> ProjectSession::getImageIdsForImages(const QStringList& images,
                                                                                bool* allResolved) const
    {
        if (allResolved)
        {
            *allResolved = true;
        }
        std::vector<xjw::camera_core::ImageId> result;
        if (!_projectData)
        {
            if (allResolved)
            {
                *allResolved = false;
            }
            return result;
        }

        const QMap<QString, QJsonObject> image_meta_by_path =
            xjw::common::project::projectImageMetaByPath(xjw::gui::project::projectFilesMeta(_projectData), true);
        result.reserve(static_cast<std::size_t>(images.size()));
        QSet<QString> seen_ids;
        for (const QString& image_path : images)
        {
            const QString image_id = image_meta_by_path.value(xjw::common::project::normalizePath(image_path))
                                         .value(QStringLiteral("image_uuid"))
                                         .toString()
                                         .trimmed();
            if (image_id.isEmpty() || seen_ids.contains(image_id))
            {
                if (allResolved)
                {
                    *allResolved = false;
                }
                continue;
            }
            try
            {
                result.emplace_back(image_id.toStdString());
                seen_ids.insert(image_id);
            }
            catch (...)
            {
                if (allResolved)
                {
                    *allResolved = false;
                }
            }
        }
        if (allResolved && result.size() != static_cast<std::size_t>(images.size()))
        {
            *allResolved = false;
        }
        return result;
    }

    xjw::camera_reference::ReferenceCameraGeometryMap
    ProjectSession::getReferenceCameraGeometriesForImages(const QStringList& images, bool* hasCamerasForAll) const
    {
        if (hasCamerasForAll)
        {
            *hasCamerasForAll = true;
        }
        xjw::camera_reference::ReferenceCameraGeometryMap result;
        bool all_image_ids = false;
        const std::vector<xjw::camera_core::ImageId> image_ids = getImageIdsForImages(images, &all_image_ids);
        bool all_numeric = false;
        const auto states = pinholeNumericStatesByImageId(image_ids, &all_numeric);
        for (const xjw::camera_core::ImageId& image_id : image_ids)
        {
            const auto state = states.find(image_id);
            if (state == states.end() || !(state->second.imageId() == image_id))
            {
                if (hasCamerasForAll)
                {
                    *hasCamerasForAll = false;
                }
                continue;
            }
            std::string error;
            const auto geometry = xjw::camera_reference::ReferenceCameraGeometry::create(state->second, &error);
            if (!geometry)
            {
                if (hasCamerasForAll)
                {
                    *hasCamerasForAll = false;
                }
                continue;
            }
            result.emplace(image_id, std::move(*geometry));
        }
        if (hasCamerasForAll && (!all_numeric || !all_image_ids || result.size() != image_ids.size()))
        {
            *hasCamerasForAll = false;
        }
        return result;
    }

    QStringList ProjectSession::getRpcCameraImagePaths(const QStringList& images, bool* hasCamerasForAll) const
    {
        if (hasCamerasForAll)
        {
            *hasCamerasForAll = true;
        }
        QStringList result;
        bool all_image_ids = false;
        const std::vector<xjw::camera_core::ImageId> image_ids = getImageIdsForImages(images, &all_image_ids);
        if (!_projectData || !all_image_ids || image_ids.size() != static_cast<std::size_t>(images.size()))
        {
            if (hasCamerasForAll)
            {
                *hasCamerasForAll = false;
            }
            return result;
        }

        const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
            xjw::gui::project::projectFilesMeta(_projectData), xjw::camera_models::makeBuiltinCameraModelRegistry());
        if (!runtime.ok())
        {
            if (hasCamerasForAll)
            {
                *hasCamerasForAll = false;
            }
            return result;
        }
        for (std::size_t index = 0; index < image_ids.size(); ++index)
        {
            const auto lookup = runtime.instances.forImage(image_ids[index]);
            const auto rpc =
                lookup.ok() ? std::dynamic_pointer_cast<const xjw::camera_models::rpc::RpcInstance>(lookup.instance)
                            : nullptr;
            if (rpc)
            {
                result.append(xjw::common::project::normalizePath(images.at(static_cast<int>(index))));
            }
            else if (hasCamerasForAll)
            {
                *hasCamerasForAll = false;
            }
        }
        return result;
    }

    QJsonObject ProjectSession::loadUiSettings() const
    {
        return _projectData ? _projectData->loadUiSettings() : QJsonObject();
    }

    void ProjectSession::saveUiSettings(const QJsonObject& settings)
    {
        if (_projectData)
        {
            _projectData->saveUiSettings(settings);
        }
    }

    void ProjectSession::markWorkspaceDirty()
    {
        if (_projectData)
        {
            _projectData->markWorkspaceDirty();
        }
    }

    void ProjectSession::discardTemporaryMetadata()
    {
        if (_projectData)
        {
            _projectData->clearTemporaryMetadata();
        }
    }

    bool ProjectSession::persistMetadata(const ProjectSessionContext& expected,
                                         const QJsonObject& metadata,
                                         bool markDirty,
                                         QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        _projectData->persistMetadata(metadata, markDirty);
        return true;
    }

    bool ProjectSession::upsertResultRecordByPath(const ProjectSessionContext& expected,
                                                  const QString& arrayKey,
                                                  const QString& pathKey,
                                                  const QJsonObject& record,
                                                  bool markDirty,
                                                  QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        if (!_projectData->upsertResultRecordByPath(arrayKey, pathKey, record, markDirty))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("写入项目成果记录失败：%1/%2").arg(arrayKey, pathKey);
            }
            return false;
        }
        return true;
    }

    bool ProjectSession::upsertResultRecordsByPath(const ProjectSessionContext& expected,
                                                   const QVector<ProjectResultRecordUpsert>& records,
                                                   QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        if (records.isEmpty())
        {
            return true;
        }

        QVector<ProjectResultRecordPathUpsert> writes;
        writes.reserve(records.size());
        for (const ProjectResultRecordUpsert& write : records)
        {
            writes.push_back({write.arrayKey, write.pathKey, write.record, write.markDirty});
        }
        return _projectData->upsertResultRecordsByPath(writes, errorMessage);
    }

    bool ProjectSession::replaceResultRecordWithLatest(const ProjectSessionContext& expected,
                                                       const QString& arrayKey,
                                                       const QJsonObject& record,
                                                       bool markDirty,
                                                       QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        if (!_projectData->replaceResultRecordWithLatest(arrayKey, record, markDirty))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("替换项目成果记录失败：%1").arg(arrayKey);
            }
            return false;
        }
        return true;
    }

    bool ProjectSession::appendImageMatchResults(const ProjectSessionContext& expected,
                                                 const QVector<ProjectImageMatchResultRecord>& records,
                                                 QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        if (!_projectData->appendImageMatchResults(records, errorMessage))
        {
            return false;
        }
        for (const ProjectImageMatchResultRecord& record : records)
        {
            emit imageMatchResultAppended(record.image);
        }
        return true;
    }

    bool ProjectSession::requestProjectSave(const ProjectSessionContext& expected, QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        return _projectData->saveProjectAsync(errorMessage);
    }

    bool ProjectSession::registerCompletedModelRun(const ProjectSessionContext& expected,
                                                   const QJsonObject& modelRecord,
                                                   xjw::mesh::workflow::ModelOutputPolicy policy,
                                                   QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        QJsonObject metadata = _projectData->metadataIncludingResults();
        if (!xjw::gui::project::registerCompletedModelRun(&metadata, modelRecord, policy, errorMessage))
        {
            return false;
        }
        _projectData->persistMetadata(metadata, true);
        return true;
    }

    bool ProjectSession::updateCompletedModelRun(const ProjectSessionContext& expected,
                                                 const QJsonObject& modelRecord,
                                                 QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        QJsonObject metadata = _projectData->metadataIncludingResults();
        if (!xjw::gui::project::updateCompletedModelRun(&metadata, modelRecord, errorMessage))
        {
            return false;
        }
        _projectData->persistMetadata(metadata, true);
        return true;
    }

    bool ProjectSession::resolveLatestDenseCloudPath(const ProjectSessionContext& expected,
                                                     QString* denseCloudPath,
                                                     QString* errorMessage) const
    {
        return requireCurrentProjectData(expected, nullptr, nullptr, errorMessage) &&
               xjw::gui::project::resolveLatestDenseCloudPath(_projectData, denseCloudPath, errorMessage);
    }

    bool ProjectSession::stageBundleAdjustMetadata(const ProjectSessionContext& expected,
                                                   const xjw::camera_project::CameraInstanceUpdates& cameraUpdates,
                                                   const QJsonObject& bundleAdjustResult,
                                                   ProjectBundleAdjustMetadataStageToken* token,
                                                   QString* errorMessage)
    {
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            if (token)
            {
                *token = {};
            }
            return false;
        }
        return _projectData->stageBundleAdjustMetadata(cameraUpdates, bundleAdjustResult, token, errorMessage);
    }

    ProjectBundleAdjustMetadataStageResolveResult
    ProjectSession::resolveBundleAdjustMetadataStage(const ProjectSessionContext& expected,
                                                     const ProjectBundleAdjustMetadataStageToken& token,
                                                     ProjectBundleAdjustMetadataStageDecision decision)
    {
        ProjectBundleAdjustMetadataStageResolveResult result;
        if (QThread::currentThread() != thread())
        {
            result.errorMessage = QStringLiteral("BA 元数据事务必须在会话线程处理");
            return result;
        }
        if (!_projectData)
        {
            result.errorMessage = QStringLiteral("ProjectData 未初始化");
            return result;
        }
        if (decision == ProjectBundleAdjustMetadataStageDecision::Commit)
        {
            QString errorMessage;
            if (!requireCurrentProjectData(expected, nullptr, nullptr, &errorMessage))
            {
                result.errorMessage = errorMessage;
                return result;
            }
        }
        return _projectData->resolveBundleAdjustMetadataStage(token, decision);
    }

    bool ProjectSession::publishImageMaskRecords(const ProjectSessionContext& expected,
                                                 const QMap<QString, QJsonObject>& recordsByResolvedImage,
                                                 QStringList* updatedImages,
                                                 QString* errorMessage)
    {
        if (updatedImages)
        {
            updatedImages->clear();
        }
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        if (recordsByResolvedImage.isEmpty())
        {
            return true;
        }

        const QString project_path = projectPath();
        QJsonObject metadata = _projectData->coreFilesMeta();
        QJsonArray images = metadata.value(QStringLiteral("images")).toArray();
        QHash<QString, int> image_indices;
        QHash<QString, QString> resolved_images;
        for (int index = 0; index < images.size(); ++index)
        {
            const QString resolved = resolvedImagePath(project_path, images.at(index).toObject());
            const QString key = xjw::common::project::normalizePath(resolved);
            if (key.isEmpty())
            {
                continue;
            }
            if (image_indices.contains(key))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("当前项目包含重复影像路径：%1").arg(resolved);
                }
                return false;
            }
            else
            {
                image_indices.insert(key, index);
                resolved_images.insert(key, resolved);
            }
        }

        QSet<QString> requested_keys;
        struct ValidatedRecord
        {
            QString key;
            QString resolvedImage;
            QJsonObject record;
        };
        QVector<ValidatedRecord> validated_records;
        validated_records.reserve(recordsByResolvedImage.size());
        for (auto it = recordsByResolvedImage.cbegin(); it != recordsByResolvedImage.cend(); ++it)
        {
            const QString resolved =
                QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, it.key()));
            const QString key = xjw::common::project::normalizePath(resolved);
            if (key.isEmpty() || !image_indices.contains(key))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("蒙版记录目标不属于当前项目：%1").arg(it.key());
                }
                return false;
            }
            if (requested_keys.contains(key))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("蒙版记录包含重复影像：%1").arg(resolved);
                }
                return false;
            }
            requested_keys.insert(key);
            for (auto field = it.value().constBegin(); field != it.value().constEnd(); ++field)
            {
                if (!imageMaskRecordFields().contains(field.key()))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("蒙版记录包含未知字段：%1").arg(field.key());
                    }
                    return false;
                }
            }
            validated_records.push_back({key, resolved_images.value(key), it.value()});
        }

        QStringList changed_images;
        for (const ValidatedRecord& validated : validated_records)
        {
            const int index = image_indices.value(validated.key);
            QJsonObject image = images.at(index).toObject();
            const QJsonObject before = image;
            clearMaskFields(&image);
            for (auto field = validated.record.constBegin(); field != validated.record.constEnd(); ++field)
            {
                image.insert(field.key(), field.value());
            }
            if (image != before)
            {
                images.replace(index, image);
                changed_images.push_back(validated.resolvedImage);
            }
        }
        if (changed_images.isEmpty())
        {
            return true;
        }

        metadata.insert(QStringLiteral("images"), images);
        _projectData->persistMetadata(metadata, true);
        if (updatedImages)
        {
            *updatedImages = changed_images;
        }
        return true;
    }

    bool ProjectSession::clearImageMaskRecords(const ProjectSessionContext& expected,
                                               const QStringList& resolvedImages,
                                               QStringList* clearedImages,
                                               QString* errorMessage)
    {
        if (clearedImages)
        {
            clearedImages->clear();
        }
        if (!requireCurrentProjectData(expected, nullptr, nullptr, errorMessage))
        {
            return false;
        }
        if (resolvedImages.isEmpty())
        {
            return true;
        }

        const QString project_path = projectPath();
        QJsonObject metadata = _projectData->coreFilesMeta();
        QJsonArray images = metadata.value(QStringLiteral("images")).toArray();
        QHash<QString, int> image_indices;
        QHash<QString, QString> project_images;
        for (int index = 0; index < images.size(); ++index)
        {
            const QString resolved = resolvedImagePath(project_path, images.at(index).toObject());
            const QString key = xjw::common::project::normalizePath(resolved);
            if (key.isEmpty())
            {
                continue;
            }
            if (image_indices.contains(key))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("当前项目包含重复影像路径：%1").arg(resolved);
                }
                return false;
            }
            else
            {
                image_indices.insert(key, index);
                project_images.insert(key, resolved);
            }
        }

        QSet<QString> requested_keys;
        for (const QString& requested : resolvedImages)
        {
            const QString resolved =
                QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(project_path, requested));
            const QString key = xjw::common::project::normalizePath(resolved);
            if (key.isEmpty() || !image_indices.contains(key))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("清除蒙版目标不属于当前项目：%1").arg(requested);
                }
                return false;
            }
            if (requested_keys.contains(key))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("清除蒙版目标重复：%1").arg(resolved);
                }
                return false;
            }
            requested_keys.insert(key);
        }

        QStringList changed_images;
        for (const QString& key : requested_keys)
        {
            const int index = image_indices.value(key);
            QJsonObject image = images.at(index).toObject();
            if (!hasMaskFields(image))
            {
                continue;
            }
            clearMaskFields(&image);
            images.replace(index, image);
            changed_images.push_back(project_images.value(key));
        }
        if (changed_images.isEmpty())
        {
            return true;
        }

        metadata.insert(QStringLiteral("images"), images);
        _projectData->persistMetadata(metadata, true);
        if (clearedImages)
        {
            *clearedImages = changed_images;
        }
        return true;
    }

    bool ProjectSession::requireProjectData(int* updatedCount, int* clearedCount, QString* errorMessage) const
    {
        if (_projectData)
        {
            return true;
        }
        if (updatedCount)
        {
            *updatedCount = 0;
        }
        if (clearedCount)
        {
            *clearedCount = 0;
        }
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("ProjectData 未初始化");
        }
        return false;
    }

    bool ProjectSession::requireCurrentProjectData(const ProjectSessionContext& expected,
                                                   int* updatedCount,
                                                   int* clearedCount,
                                                   QString* errorMessage) const
    {
        const auto reject = [updatedCount, clearedCount, errorMessage](const QString& message)
        {
            if (updatedCount)
            {
                *updatedCount = 0;
            }
            if (clearedCount)
            {
                *clearedCount = 0;
            }
            if (errorMessage)
            {
                *errorMessage = message;
            }
            return false;
        };

        if (QThread::currentThread() != thread())
        {
            return reject(QStringLiteral("项目结果写回必须在会话线程执行"));
        }
        if (!isCurrent(expected))
        {
            return reject(QStringLiteral("项目会话已变化，已拒绝过期结果写回"));
        }
        if (!requireProjectData(updatedCount, clearedCount, errorMessage))
        {
            return false;
        }
        if (!_projectData->hasProject())
        {
            return reject(QStringLiteral("没有打开的项目"));
        }
        return true;
    }

    bool ProjectSession::setCameraInstances(const QMap<QString, QJsonObject>& cameras,
                                            int* updatedCount,
                                            QString* errorMessage)
    {
        return requireProjectData(updatedCount, nullptr, errorMessage) &&
               _projectData->setCameraInstances(cameras, updatedCount, errorMessage);
    }

    bool ProjectSession::setCameraInstances(const ProjectSessionContext& expected,
                                            const QMap<QString, QJsonObject>& cameras,
                                            int* updatedCount,
                                            QString* errorMessage)
    {
        return requireCurrentProjectData(expected, updatedCount, nullptr, errorMessage) &&
               _projectData->setCameraInstances(cameras, updatedCount, errorMessage);
    }

    bool ProjectSession::setCameraInstancesById(const xjw::camera_project::CameraInstanceUpdates& updates,
                                                int* updatedCount,
                                                QString* errorMessage)
    {
        return requireProjectData(updatedCount, nullptr, errorMessage) &&
               _projectData->setCameraInstancesById(updates, updatedCount, errorMessage);
    }

    bool ProjectSession::setCameraInstancesById(const ProjectSessionContext& expected,
                                                const xjw::camera_project::CameraInstanceUpdates& updates,
                                                int* updatedCount,
                                                QString* errorMessage)
    {
        return requireCurrentProjectData(expected, updatedCount, nullptr, errorMessage) &&
               _projectData->setCameraInstancesById(updates, updatedCount, errorMessage);
    }

    bool ProjectSession::replaceCameraInstances(const QStringList& targetImagePaths,
                                                const QMap<QString, QJsonObject>& cameras,
                                                int* updatedCount,
                                                int* clearedCount,
                                                QString* errorMessage)
    {
        return requireProjectData(updatedCount, clearedCount, errorMessage) &&
               _projectData->replaceCameraInstances(
                   targetImagePaths, cameras, updatedCount, clearedCount, errorMessage);
    }

    bool ProjectSession::replaceCameraInstances(const ProjectSessionContext& expected,
                                                const QStringList& targetImagePaths,
                                                const QMap<QString, QJsonObject>& cameras,
                                                int* updatedCount,
                                                int* clearedCount,
                                                QString* errorMessage)
    {
        return requireCurrentProjectData(expected, updatedCount, clearedCount, errorMessage) &&
               _projectData->replaceCameraInstances(
                   targetImagePaths, cameras, updatedCount, clearedCount, errorMessage);
    }

    bool ProjectSession::replaceCameraInstancesById(const xjw::camera_project::CameraImageIds& targetImageIds,
                                                    const xjw::camera_project::CameraInstanceUpdates& updates,
                                                    int* updatedCount,
                                                    int* clearedCount,
                                                    QString* errorMessage)
    {
        return requireProjectData(updatedCount, clearedCount, errorMessage) &&
               _projectData->replaceCameraInstancesById(
                   targetImageIds, updates, updatedCount, clearedCount, errorMessage);
    }

    bool ProjectSession::replaceCameraInstancesById(const ProjectSessionContext& expected,
                                                    const xjw::camera_project::CameraImageIds& targetImageIds,
                                                    const xjw::camera_project::CameraInstanceUpdates& updates,
                                                    int* updatedCount,
                                                    int* clearedCount,
                                                    QString* errorMessage)
    {
        return requireCurrentProjectData(expected, updatedCount, clearedCount, errorMessage) &&
               _projectData->replaceCameraInstancesById(
                   targetImageIds, updates, updatedCount, clearedCount, errorMessage);
    }

    bool ProjectSession::clearCameraInstances(const QStringList& imagePaths, int* updatedCount, QString* errorMessage)
    {
        return requireProjectData(updatedCount, nullptr, errorMessage) &&
               _projectData->clearCameraInstances(imagePaths, updatedCount, errorMessage);
    }

    bool ProjectSession::clearCameraInstances(const ProjectSessionContext& expected,
                                              const QStringList& imagePaths,
                                              int* updatedCount,
                                              QString* errorMessage)
    {
        return requireCurrentProjectData(expected, updatedCount, nullptr, errorMessage) &&
               _projectData->clearCameraInstances(imagePaths, updatedCount, errorMessage);
    }

    bool ProjectSession::appendIntersectionResult(const QJsonObject& result, QString* errorMessage)
    {
        return _projectData && _projectData->appendIntersectionResult(result, errorMessage);
    }

    bool ProjectSession::appendIntersectionResult(const ProjectSessionContext& expected,
                                                  const QJsonObject& result,
                                                  QString* errorMessage)
    {
        return requireCurrentProjectData(expected, nullptr, nullptr, errorMessage) &&
               _projectData->appendIntersectionResult(result, errorMessage);
    }

    TiePointMutationResult ProjectSession::replaceTiePointResult(const ProjectSessionContext& expected,
                                                                 const QString& sparseCloudPath,
                                                                 int sparsePointCount,
                                                                 const QStringList& selectedImages,
                                                                 const QString& outputDir,
                                                                 const QJsonObject& extraRecord)
    {
        TiePointMutationResult result;
        if (!requireCurrentProjectData(expected, nullptr, nullptr, &result.errorMessage))
        {
            return result;
        }
        return xjw::gui::project::replaceTiePointResult(
            _projectData, sparseCloudPath, sparsePointCount, selectedImages, outputDir, extraRecord);
    }

    QJsonArray ProjectSession::intersectionResults() const
    {
        return _projectData ? _projectData->getIntersectionResults() : QJsonArray();
    }

} // namespace xjw::gui::project
