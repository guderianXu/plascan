#pragma once

#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/project/CameraInstanceUpdate.h"
#include "camera/reference/geometry/ReferenceCameraGeometry.h"
#include "ProjectTiePointResultService.h"
#include "project/ProjectDocumentModel.h"
#include "project/ProjectSessionModel.h"
#include "project/support/ProjectSessionContext.h"

#include <QObject>
#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>

#include <unordered_map>
#include <vector>

class ProjectData;

namespace xjw::mesh::workflow
{
    enum class ModelOutputPolicy;
}

namespace xjw::gui::project
{

    struct ProjectResultRecordUpsert
    {
        QString arrayKey;
        QString pathKey;
        QJsonObject record;
        bool markDirty = true;
    };

    class ProjectSession final : public QObject
    {
        Q_OBJECT

    public:
        explicit ProjectSession(ProjectData* projectData, QObject* parent = nullptr);

        ProjectData* data() const;
        bool hasProject() const;
        QString projectPath() const;
        QString activeChunkId() const;
        ProjectSessionContext context() const;
        bool isCurrent(const ProjectSessionContext& context) const;
        void advanceGeneration();

        // A single GUI-side operation gate shared by lifecycle, resource and
        // cleanup services.  It deliberately lives beside the session context so
        // no service can mutate ProjectData while another service is replacing or
        // closing the session.
        bool tryBeginOperation(const QString& operation);
        void endOperation(const QString& operation);
        bool isBusy() const;
        QString busyOperation() const;

        bool isDirty() const;
        QJsonObject metadata() const;
        QJsonObject coreMetadata() const;
        QStringList imagesByCategory(const QString& category) const;
        QStringList allImages() const;
        QString matchFile(const QString& firstImage, const QString& secondImage) const;
        QMap<QString, xjw::camera_models::frame_pinhole::FramePinholeNumericState>
        getPinholeNumericStatesForImages(const QStringList& images, bool* hasCamerasForAll = nullptr) const;
        std::vector<xjw::camera_core::ImageId> getImageIdsForImages(const QStringList& images,
                                                                    bool* allResolved = nullptr) const;
        xjw::camera_reference::ReferenceCameraGeometryMap
        getReferenceCameraGeometriesForImages(const QStringList& images, bool* hasCamerasForAll = nullptr) const;
        QStringList getRpcCameraImagePaths(const QStringList& images, bool* hasCamerasForAll = nullptr) const;
        QJsonObject loadUiSettings() const;
        void saveUiSettings(const QJsonObject& settings);
        void markWorkspaceDirty();
        void discardTemporaryMetadata();

        bool persistMetadata(const ProjectSessionContext& expected,
                             const QJsonObject& metadata,
                             bool markDirty = true,
                             QString* errorMessage = nullptr);
        bool upsertResultRecordByPath(const ProjectSessionContext& expected,
                                      const QString& arrayKey,
                                      const QString& pathKey,
                                      const QJsonObject& record,
                                      bool markDirty = true,
                                      QString* errorMessage = nullptr);
        bool upsertResultRecordsByPath(const ProjectSessionContext& expected,
                                       const QVector<ProjectResultRecordUpsert>& records,
                                       QString* errorMessage = nullptr);
        bool replaceResultRecordWithLatest(const ProjectSessionContext& expected,
                                           const QString& arrayKey,
                                           const QJsonObject& record,
                                           bool markDirty = true,
                                           QString* errorMessage = nullptr);
        bool appendImageMatchResults(const ProjectSessionContext& expected,
                                     const QVector<ProjectImageMatchResultRecord>& records,
                                     QString* errorMessage = nullptr);
        bool requestProjectSave(const ProjectSessionContext& expected, QString* errorMessage = nullptr);
        bool registerCompletedModelRun(const ProjectSessionContext& expected,
                                       const QJsonObject& modelRecord,
                                       xjw::mesh::workflow::ModelOutputPolicy policy,
                                       QString* errorMessage = nullptr);
        bool updateCompletedModelRun(const ProjectSessionContext& expected,
                                     const QJsonObject& modelRecord,
                                     QString* errorMessage = nullptr);
        bool resolveLatestDenseCloudPath(const ProjectSessionContext& expected,
                                         QString* denseCloudPath,
                                         QString* errorMessage = nullptr) const;
        bool stageBundleAdjustMetadata(const ProjectSessionContext& expected,
                                       const xjw::camera_project::CameraInstanceUpdates& cameraUpdates,
                                       const QJsonObject& bundleAdjustResult,
                                       ProjectBundleAdjustMetadataStageToken* token,
                                       QString* errorMessage = nullptr);
        ProjectBundleAdjustMetadataStageResolveResult
        resolveBundleAdjustMetadataStage(const ProjectSessionContext& expected,
                                         const ProjectBundleAdjustMetadataStageToken& token,
                                         ProjectBundleAdjustMetadataStageDecision decision);

        bool publishImageMaskRecords(const ProjectSessionContext& expected,
                                     const QMap<QString, QJsonObject>& recordsByResolvedImage,
                                     QStringList* updatedImages = nullptr,
                                     QString* errorMessage = nullptr);
        bool clearImageMaskRecords(const ProjectSessionContext& expected,
                                   const QStringList& resolvedImages,
                                   QStringList* clearedImages = nullptr,
                                   QString* errorMessage = nullptr);

        bool setCameraInstances(const QMap<QString, QJsonObject>& cameras,
                                int* updatedCount = nullptr,
                                QString* errorMessage = nullptr);
        bool setCameraInstances(const ProjectSessionContext& expected,
                                const QMap<QString, QJsonObject>& cameras,
                                int* updatedCount = nullptr,
                                QString* errorMessage = nullptr);
        bool setCameraInstancesById(const xjw::camera_project::CameraInstanceUpdates& updates,
                                    int* updatedCount = nullptr,
                                    QString* errorMessage = nullptr);
        bool setCameraInstancesById(const ProjectSessionContext& expected,
                                    const xjw::camera_project::CameraInstanceUpdates& updates,
                                    int* updatedCount = nullptr,
                                    QString* errorMessage = nullptr);
        bool replaceCameraInstances(const QStringList& targetImagePaths,
                                    const QMap<QString, QJsonObject>& cameras,
                                    int* updatedCount = nullptr,
                                    int* clearedCount = nullptr,
                                    QString* errorMessage = nullptr);
        bool replaceCameraInstances(const ProjectSessionContext& expected,
                                    const QStringList& targetImagePaths,
                                    const QMap<QString, QJsonObject>& cameras,
                                    int* updatedCount = nullptr,
                                    int* clearedCount = nullptr,
                                    QString* errorMessage = nullptr);
        bool replaceCameraInstancesById(const xjw::camera_project::CameraImageIds& targetImageIds,
                                        const xjw::camera_project::CameraInstanceUpdates& updates,
                                        int* updatedCount = nullptr,
                                        int* clearedCount = nullptr,
                                        QString* errorMessage = nullptr);
        bool replaceCameraInstancesById(const ProjectSessionContext& expected,
                                        const xjw::camera_project::CameraImageIds& targetImageIds,
                                        const xjw::camera_project::CameraInstanceUpdates& updates,
                                        int* updatedCount = nullptr,
                                        int* clearedCount = nullptr,
                                        QString* errorMessage = nullptr);
        bool clearCameraInstances(const QStringList& imagePaths,
                                  int* updatedCount = nullptr,
                                  QString* errorMessage = nullptr);
        bool clearCameraInstances(const ProjectSessionContext& expected,
                                  const QStringList& imagePaths,
                                  int* updatedCount = nullptr,
                                  QString* errorMessage = nullptr);
        bool appendIntersectionResult(const QJsonObject& result, QString* errorMessage = nullptr);
        bool appendIntersectionResult(const ProjectSessionContext& expected,
                                      const QJsonObject& result,
                                      QString* errorMessage = nullptr);
        TiePointMutationResult replaceTiePointResult(const ProjectSessionContext& expected,
                                                     const QString& sparseCloudPath,
                                                     int sparsePointCount,
                                                     const QStringList& selectedImages,
                                                     const QString& outputDir,
                                                     const QJsonObject& extraRecord = {});
        QJsonArray intersectionResults() const;

    signals:
        void sessionChanged(const xjw::gui::project::ProjectSessionContext& context);
        void projectOpened(const QString& projectPath);
        void projectSaved(const QString& projectPath);
        void projectClosed();
        void chunkListChanged(const QJsonArray& chunks, const QString& activeChunkId);
        void metadataChanged(const QJsonObject& metadata);
        void dirtyStateChanged(bool dirty);
        void imageMatchResultAppended(const QString& imagePath);
        void matchPairReady(const QString& firstImage,
                            const QString& secondImage,
                            const QString& matchFilePath,
                            int matchCount);

    private:
        std::unordered_map<xjw::camera_core::ImageId, xjw::camera_models::frame_pinhole::FramePinholeNumericState>
        pinholeNumericStatesByImageId(const std::vector<xjw::camera_core::ImageId>& imageIds,
                                      bool* hasCamerasForAll = nullptr) const;
        bool requireProjectData(int* updatedCount, int* clearedCount, QString* errorMessage) const;
        bool requireCurrentProjectData(const ProjectSessionContext& expected,
                                       int* updatedCount,
                                       int* clearedCount,
                                       QString* errorMessage) const;

        ProjectData* _projectData = nullptr;
        quint64 _generation = 0;
        QString _busyOperation;
    };

} // namespace xjw::gui::project
