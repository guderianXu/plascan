#include "ProjectDepthBatchLineage.h"

#include "ProjectWorkflowOperations.h"
#include "camera/models/CameraModelFactories.h"
#include "camera/project/CameraProjectRecords.h"
#include "camera/project/CameraProjectRuntime.h"
#include "project/ProjectMetadata.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>

#include <QSet>

namespace xjw::gui::project
{

    namespace
    {

        constexpr int kCurrentProjectDepthInputSignatureVersion = 4;

        QString normalizedResourcePath(const QString& path)
        {
            return QDir::fromNativeSeparators(path.trimmed()).toCaseFolded();
        }

        QString canonicalImageIdentity(const QJsonObject& image)
        {
            const QString uuid = image.value(QStringLiteral("image_uuid")).toString().trimmed();
            return uuid.isEmpty() ? QString() : QStringLiteral("uuid:") + uuid;
        }

        int imageIndexForResource(const QJsonArray& images, const QString& resource)
        {
            const QString token = resource.trimmed();
            if (token.isEmpty())
            {
                return -1;
            }

            int matchedIndex = -1;
            for (int index = 0; index < images.size(); ++index)
            {
                const QJsonObject image = images.at(index).toObject();
                if (image.value(QStringLiteral("image_uuid")).toString().trimmed() != token)
                {
                    continue;
                }
                if (matchedIndex >= 0)
                {
                    return -1;
                }
                matchedIndex = index;
            }
            if (matchedIndex >= 0)
            {
                return matchedIndex;
            }

            const QString normalizedResource = normalizedResourcePath(token);
            matchedIndex = -1;
            for (int index = 0; index < images.size(); ++index)
            {
                if (normalizedResourcePath(images.at(index).toObject().value(QStringLiteral("path")).toString()) ==
                    normalizedResource)
                {
                    if (matchedIndex >= 0)
                    {
                        return -1;
                    }
                    matchedIndex = index;
                }
            }
            return matchedIndex;
        }

        QJsonObject canonicalCameraRecord(const QJsonObject& projectFiles,
                                          const QJsonObject& image,
                                          const xjw::camera_project::CameraProjectRuntimeResult& runtime)
        {
            const QString imageId = image.value(QStringLiteral("image_uuid")).toString().trimmed();
            if (imageId.isEmpty())
            {
                return {};
            }
            const auto lookup = runtime.instances.forImage(xjw::camera_core::ImageId(imageId.toStdString()));
            if (!lookup.ok())
            {
                return {};
            }
            const QJsonObject instance = xjw::camera_project::CameraProjectRecords::instanceForImage(
                projectFiles, imageId);
            const QJsonObject definition = xjw::camera_project::CameraProjectRecords::definitionForInstance(
                projectFiles, instance);
            if (instance.isEmpty() || definition.isEmpty())
            {
                return {};
            }
            return QJsonObject{{QStringLiteral("definition"), definition},
                               {QStringLiteral("instance"), instance}};
        }

        QString selectedSparsePlyPath(const QJsonObject& atResult)
        {
            const QJsonObject files = atResult.value(QStringLiteral("files")).toObject();
            QString path = files.value(QStringLiteral("sparse_cloud_xyz")).toString().trimmed();
            if (path.isEmpty())
            {
                path = atResult.value(QStringLiteral("sparse_cloud_xyz")).toString().trimmed();
            }
            if (path.isEmpty())
            {
                path = atResult.value(QStringLiteral("sparse_cloud_path")).toString().trimmed();
            }
            return path;
        }

        QJsonObject sparsePlyContentIdentity(const QJsonObject& atResult)
        {
            const QString path = selectedSparsePlyPath(atResult);
            QFile file(path);
            if (path.isEmpty() || !file.open(QIODevice::ReadOnly))
            {
                return {{QStringLiteral("state"), QStringLiteral("unreadable")}};
            }

            QCryptographicHash hash(QCryptographicHash::Sha256);
            while (!file.atEnd())
            {
                const QByteArray chunk = file.read(1024 * 1024);
                if (chunk.isEmpty() && file.error() != QFileDevice::NoError)
                {
                    return {{QStringLiteral("state"), QStringLiteral("unreadable")}};
                }
                hash.addData(chunk);
            }
            return {{QStringLiteral("state"), QStringLiteral("sha256")},
                    {QStringLiteral("value"), QString::fromLatin1(hash.result().toHex())}};
        }

    } // namespace

    QString canonicalProjectDepthInputSignature(const QJsonObject& projectMetadata, int aerialTriangulationResultIndex)
    {
        const QJsonObject projectFiles = xjw::common::project::projectFilesRootObject(projectMetadata);
        const QJsonArray images = projectFiles.value(QStringLiteral("images")).toArray();
        const QJsonArray atResults = projectFiles.value(QStringLiteral("aerial_triangulation_results")).toArray();
        if (images.isEmpty())
        {
            return QString();
        }

        // A depth cache is only meaningful when the complete canonical camera
        // graph is valid.  In particular, do not derive an identity or
        // geometry from a path, array position, or an embedded legacy camera.
        const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
            projectFiles, xjw::camera_models::makeBuiltinCameraModelRegistry());
        if (!runtime.ok())
        {
            return QString();
        }

        QSet<QString> imageIds;
        for (const QJsonValue& value : images)
        {
            const QString imageId = value.toObject().value(QStringLiteral("image_uuid")).toString().trimmed();
            if (imageId.isEmpty() || imageIds.contains(imageId))
            {
                return QString();
            }
            imageIds.insert(imageId);
        }

        int atIndex = aerialTriangulationResultIndex;
        if (atIndex < 0 || atIndex >= atResults.size())
        {
            atIndex = xjw::core::project::findLatestProductionAtResultIndex(projectFiles);
        }
        if (atIndex < 0 && !atResults.isEmpty())
        {
            atIndex = atResults.size() - 1;
        }
        const QJsonObject atResult = atIndex >= 0 ? atResults.at(atIndex).toObject() : QJsonObject();

        QJsonArray selectedResources = atResult.value(QStringLiteral("selected_images")).toArray();
        if (selectedResources.isEmpty())
        {
            for (const QJsonValue& value : images)
            {
                selectedResources.append(value.toObject().value(QStringLiteral("image_uuid")));
            }
        }

        QJsonArray canonicalImages;
        QSet<int> selectedIndices;
        for (int selectedIndex = 0; selectedIndex < selectedResources.size(); ++selectedIndex)
        {
            const int imageIndex = imageIndexForResource(images, selectedResources.at(selectedIndex).toString());
            if (imageIndex < 0 || selectedIndices.contains(imageIndex))
            {
                return QString();
            }
            selectedIndices.insert(imageIndex);

            const QJsonObject image = images.at(imageIndex).toObject();
            const QString identity = canonicalImageIdentity(image);
            const QJsonObject camera = canonicalCameraRecord(projectFiles, image, runtime);
            if (identity.isEmpty() || camera.isEmpty())
            {
                return QString();
            }
            canonicalImages.append(QJsonObject{
                {QStringLiteral("identity"), identity},
                {QStringLiteral("camera"), camera}});
        }

        QJsonObject lineage;
        lineage[QStringLiteral("aerial_triangulation_result_index")] = atIndex;
        lineage[QStringLiteral("operation")] = atResult.value(QStringLiteral("operation")).toString();
        lineage[QStringLiteral("reconstruction_generation_id")] =
            atResult.value(QStringLiteral("reconstruction_generation_id")).toString();
        lineage[QStringLiteral("run_id")] = atResult.value(QStringLiteral("run_id")).toString();

        QJsonObject signatureInput;
        signatureInput[QStringLiteral("signature_version")] = kCurrentProjectDepthInputSignatureVersion;
        signatureInput[QStringLiteral("images")] = canonicalImages;
        signatureInput[QStringLiteral("lineage")] = lineage;
        signatureInput[QStringLiteral("sparse_ply_content")] = sparsePlyContentIdentity(atResult);
        const QByteArray payload = QJsonDocument(signatureInput).toJson(QJsonDocument::Compact);
        return QString::fromLatin1(QCryptographicHash::hash(payload, QCryptographicHash::Sha256).toHex());
    }

} // namespace xjw::gui::project
