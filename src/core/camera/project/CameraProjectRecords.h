#pragma once

#include "CameraProjectStore.h"
#include "CameraInstanceUpdate.h"

#include <QJsonObject>
#include <QMap>
#include <QSet>
#include <QString>
#include <QStringList>

namespace xjw::camera_project
{

    struct CameraProjectUpdateResult
    {
        int updatedCount = 0;
        int clearedCount = 0;
        QStringList errors;

        bool ok() const noexcept
        {
            return errors.isEmpty();
        }
    };

    /**
     * Builds and updates normalized camera definition/instance collections.
     * Input metadata is model-specific parameter data from an importer or solver;
     * it is never written into an image object.
     */
    class CameraProjectRecords
    {
    public:
        static CameraProjectUpdateResult upsertByImagePath(QJsonObject* projectFiles,
                                                           const QMap<QString, QJsonObject>& metadataByImagePath);

        /** Apply solver output addressed by canonical ImageId. */
        static CameraProjectUpdateResult upsertByImageId(QJsonObject* projectFiles,
                                                         const CameraInstanceUpdates& updates);

        /** Replace the selected canonical images with solver output addressed by ImageId. */
        static CameraProjectUpdateResult replaceByImageId(QJsonObject* projectFiles,
                                                          const CameraImageIds& targetImageIds,
                                                          const CameraInstanceUpdates& updates);

        static CameraProjectUpdateResult replaceByImagePath(QJsonObject* projectFiles,
                                                            const QStringList& targetImagePaths,
                                                            const QMap<QString, QJsonObject>& metadataByImagePath);

        static CameraProjectUpdateResult clearByImagePath(QJsonObject* projectFiles, const QStringList& imagePaths);

        static QJsonObject instanceForImage(const QJsonObject& projectFiles, const QString& imageUuid);

        static QJsonObject definitionForInstance(const QJsonObject& projectFiles, const QJsonObject& instance);

        /** Return model parameters combined with instance state for a model adapter. */
        static QJsonObject modelParametersForInstance(const QJsonObject& projectFiles, const QJsonObject& instance);

        static QJsonObject modelParametersForImage(const QJsonObject& projectFiles, const QJsonObject& image);

    private:
        static CameraProjectUpdateResult updateByImageId(QJsonObject* projectFiles,
                                                         const QSet<QString>& targetImageIds,
                                                         const QMap<QString, QJsonObject>& metadataByImageId,
                                                         bool clearMissing);
    };

} // namespace xjw::camera_project
