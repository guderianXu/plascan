#include "ProjectDepthBatchLineage.h"

#include "ProjectCameraIO.h"
#include "ProjectWorkflowUtils.h"

#include <QCryptographicHash>
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>

#include <algorithm>
#include <array>
#include <cmath>

namespace xjw::gui::project
{

namespace
{

QString normalizedResourcePath(const QString &path)
{
    return QDir::fromNativeSeparators(path.trimmed()).toCaseFolded();
}

QString resourceFileName(const QString &path)
{
    const QString normalized = QDir::fromNativeSeparators(path.trimmed());
    return normalized.mid(normalized.lastIndexOf(QLatin1Char('/')) + 1).toCaseFolded();
}

QString stableImageIdentity(const QJsonObject &image, int fallbackIndex)
{
    const QString uuid = image.value(QStringLiteral("image_uuid")).toString().trimmed();
    if (!uuid.isEmpty())
    {
        return QStringLiteral("uuid:") + uuid.toCaseFolded();
    }
    const QString contentHash = image.value(QStringLiteral("sha256")).toString().trimmed();
    if (!contentHash.isEmpty())
    {
        return QStringLiteral("sha256:") + contentHash.toCaseFolded();
    }
    return QStringLiteral("file:%1:%2")
        .arg(resourceFileName(image.value(QStringLiteral("path")).toString()))
        .arg(fallbackIndex);
}

QJsonArray doubleArray(const double *values, int count)
{
    QJsonArray result;
    for (int index = 0; index < count; ++index)
    {
        result.append(values[index]);
    }
    return result;
}

QJsonObject fallbackCameraGeometry(const QJsonObject &cameraObject)
{
    QJsonObject fallback;
    const std::array<QString, 19> geometryKeys{
        QStringLiteral("C"), QStringLiteral("R"), QStringLiteral("center"),
        QStringLiteral("fu"), QStringLiteral("fv"), QStringLiteral("cu"),
        QStringLiteral("cv"), QStringLiteral("pitch"), QStringLiteral("k1"),
        QStringLiteral("k2"), QStringLiteral("k3"), QStringLiteral("p1"),
        QStringLiteral("p2"), QStringLiteral("u_direction"),
        QStringLiteral("v_direction"), QStringLiteral("depth_axis_flipped"),
        QStringLiteral("intrinsics_unit"), QStringLiteral("camera_center_unit"),
        QStringLiteral("model")};
    for (const QString &key : geometryKeys)
    {
        if (cameraObject.contains(key))
        {
            fallback[key] = cameraObject.value(key);
        }
    }
    return fallback;
}

QJsonObject canonicalCameraGeometry(const QJsonObject &cameraObject)
{
    xjw::Camera camera;
    if (!xjw::common::project::cameraFromJson(cameraObject, &camera) ||
        !camera.isValid())
    {
        return fallbackCameraGeometry(cameraObject);
    }

    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();
    const auto distortion = camera.distortion();
    return QJsonObject{
        {QStringLiteral("center_m"), doubleArray(center.data(), 3)},
        {QStringLiteral("rotation_camera_to_world"), doubleArray(rotation.data(), 9)},
        {QStringLiteral("fx_px"), camera.focalX()},
        {QStringLiteral("fy_px"), camera.focalY()},
        {QStringLiteral("cx_px"), camera.principalX()},
        {QStringLiteral("cy_px"), camera.principalY()},
        {QStringLiteral("k1"), distortion.radialK1},
        {QStringLiteral("k2"), distortion.radialK2},
        {QStringLiteral("k3"), distortion.radialK3},
        {QStringLiteral("p1"), distortion.tangentialP1},
        {QStringLiteral("p2"), distortion.tangentialP2},
        {QStringLiteral("u_direction"), camera.uAxisSign()},
        {QStringLiteral("v_direction"), camera.vAxisSign()},
        {QStringLiteral("depth_axis_flipped"), camera.depthAxisFlipped()}};
}

int imageIndexForResource(const QJsonArray &images, const QString &resource)
{
    const QString normalizedResource = normalizedResourcePath(resource);
    for (int index = 0; index < images.size(); ++index)
    {
        if (normalizedResourcePath(
                images.at(index).toObject().value(QStringLiteral("path")).toString()) ==
            normalizedResource)
        {
            return index;
        }
    }

    const QString fileName = resourceFileName(resource);
    int matchingIndex = -1;
    for (int index = 0; index < images.size(); ++index)
    {
        if (resourceFileName(
                images.at(index).toObject().value(QStringLiteral("path")).toString()) !=
            fileName)
        {
            continue;
        }
        if (matchingIndex >= 0)
        {
            return -1;
        }
        matchingIndex = index;
    }
    return matchingIndex;
}

bool nearlyEqual(double lhs, double rhs)
{
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= 1.0e-9 * scale;
}

bool jsonArrayMatches(const QJsonArray &actual,
                      const double *expected,
                      int count,
                      bool transposeThreeByThree = false)
{
    if (actual.size() != count)
    {
        return false;
    }
    for (int index = 0; index < count; ++index)
    {
        const int expectedIndex = transposeThreeByThree
            ? (index % 3) * 3 + index / 3
            : index;
        if (!nearlyEqual(actual.at(index).toDouble(), expected[expectedIndex]))
        {
            return false;
        }
    }
    return true;
}

bool depthCameraMatchesProjectImage(
    const xjw::core::project::StoredDepthFrameRecord &frame,
    const QJsonObject &image)
{
    xjw::Camera camera;
    if (frame.cameraModel.isEmpty() ||
        !xjw::common::project::cameraFromJson(
            image.value(QStringLiteral("camera")).toObject(), &camera) ||
        !camera.isValid())
    {
        return false;
    }

    const auto distortion = camera.distortion();
    const bool unverifiableProjectionTerms =
        !nearlyEqual(distortion.radialK1, 0.0) ||
        !nearlyEqual(distortion.radialK2, 0.0) ||
        !nearlyEqual(distortion.radialK3, 0.0) ||
        !nearlyEqual(distortion.tangentialP1, 0.0) ||
        !nearlyEqual(distortion.tangentialP2, 0.0) ||
        camera.uAxisSign() != 1 || camera.vAxisSign() != 1 ||
        camera.depthAxisFlipped();
    if (unverifiableProjectionTerms)
    {
        return false;
    }

    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();
    return jsonArrayMatches(
               frame.cameraModel.value(QStringLiteral("camera_center")).toArray(),
               center.data(), 3) &&
        jsonArrayMatches(
               frame.cameraModel.value(QStringLiteral("rotation_world_to_camera")).toArray(),
               rotation.data(), 9, true) &&
        nearlyEqual(frame.cameraModel.value(QStringLiteral("fx")).toDouble(), camera.focalX()) &&
        nearlyEqual(frame.cameraModel.value(QStringLiteral("fy")).toDouble(), camera.focalY()) &&
        nearlyEqual(frame.cameraModel.value(QStringLiteral("cx")).toDouble(), camera.principalX()) &&
        nearlyEqual(frame.cameraModel.value(QStringLiteral("cy")).toDouble(), camera.principalY());
}

} // namespace

QString canonicalProjectDepthInputSignature(
    const QJsonObject &projectMetadata,
    int aerialTriangulationResultIndex,
    int signatureVersion)
{
    const QJsonArray images = projectMetadata.value(QStringLiteral("images")).toArray();
    const QJsonArray atResults =
        projectMetadata.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (images.isEmpty() && atResults.isEmpty())
    {
        return QString();
    }

    int atIndex = aerialTriangulationResultIndex;
    if (atIndex < 0 || atIndex >= atResults.size())
    {
        atIndex = findLatestProductionAtResultIndex(projectMetadata);
    }
    if (atIndex < 0 && !atResults.isEmpty())
    {
        atIndex = atResults.size() - 1;
    }
    const QJsonObject atResult = atIndex >= 0
        ? atResults.at(atIndex).toObject()
        : QJsonObject();

    QJsonArray selectedResources = atResult.value(QStringLiteral("selected_images")).toArray();
    if (selectedResources.isEmpty())
    {
        for (const QJsonValue &value : images)
        {
            selectedResources.append(
                value.toObject().value(QStringLiteral("path")).toString());
        }
    }

    QJsonArray canonicalImages;
    for (int selectedIndex = 0; selectedIndex < selectedResources.size(); ++selectedIndex)
    {
        const int imageIndex = imageIndexForResource(
            images, selectedResources.at(selectedIndex).toString());
        if (imageIndex < 0)
        {
            canonicalImages.append(QJsonObject{
                {QStringLiteral("identity"),
                 QStringLiteral("unresolved:%1")
                     .arg(resourceFileName(selectedResources.at(selectedIndex).toString()))}});
            continue;
        }

        const QJsonObject image = images.at(imageIndex).toObject();
        canonicalImages.append(QJsonObject{
            {QStringLiteral("identity"), stableImageIdentity(image, imageIndex)},
            {QStringLiteral("camera"),
             canonicalCameraGeometry(image.value(QStringLiteral("camera")).toObject())}});
    }

    QJsonObject lineage;
    lineage[QStringLiteral("reconstruction_generation_id")] =
        atResult.value(QStringLiteral("reconstruction_generation_id")).toString();
    lineage[QStringLiteral("run_id")] = atResult.value(QStringLiteral("run_id")).toString();

    QJsonObject signatureInput;
    signatureInput[QStringLiteral("signature_version")] = signatureVersion;
    signatureInput[QStringLiteral("images")] = canonicalImages;
    signatureInput[QStringLiteral("lineage")] = lineage;
    const QByteArray payload =
        QJsonDocument(signatureInput).toJson(QJsonDocument::Compact);
    return QString::fromLatin1(
        QCryptographicHash::hash(payload, QCryptographicHash::Sha256).toHex());
}

bool legacyDepthCamerasMatchCurrentProject(
    const xjw::core::project::StoredDepthFramesResult &storedFrames,
    const QJsonObject &projectMetadata)
{
    const QJsonArray images = projectMetadata.value(QStringLiteral("images")).toArray();
    if (images.isEmpty())
    {
        return false;
    }
    for (const auto &frame : storedFrames.frames)
    {
        const int imageIndex = imageIndexForResource(images, frame.refImage);
        if (imageIndex < 0 ||
            !depthCameraMatchesProjectImage(frame, images.at(imageIndex).toObject()))
        {
            return false;
        }
    }
    return true;
}

} // namespace xjw::gui::project
