#include "BaInputBuilder.h"

#include "Intersection.h"
#include "project/ProjectCommonUtils.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPointF>
#include <QSet>

#include <cmath>
#include <limits>

namespace xjw::core::project
{

namespace
{

double reprojectionErrorPx(const xjw::Camera &camera,
                           const std::array<double, 3> &xyz,
                           const QPointF &pixel)
{
    const double worldPoint[3] = {xyz[0], xyz[1], xyz[2]};
    double projected[2] = {0.0, 0.0};
    if (!camera.projectWorldPoint(worldPoint, projected)
        && !camera.projectWorldPointSigned(worldPoint, projected))
    {
        return std::numeric_limits<double>::infinity();
    }

    const double dx = projected[0] - pixel.x();
    const double dy = projected[1] - pixel.y();
    return std::sqrt(dx * dx + dy * dy);
}

struct PairIntersectionCandidate
{
    std::array<double, 3> point{{0.0, 0.0, 0.0}};
    double rmsReprojectionPx = std::numeric_limits<double>::infinity();
    bool valid = false;
};

PairIntersectionCandidate triangulatePairWithDirectionFallback(const xjw::Camera &cameraA,
                                                               const QPointF &pixelA,
                                                               const xjw::Camera &cameraB,
                                                               const QPointF &pixelB)
{
    PairIntersectionCandidate bestCandidate;

    for (int flipMask = 0; flipMask < 4; ++flipMask)
    {
        xjw::Camera testCameraA = cameraA;
        xjw::Camera testCameraB = cameraB;

        if ((flipMask & 0x1) != 0)
        {
            testCameraA.setDepthAxisFlipped(!testCameraA.depthAxisFlipped());
        }
        if ((flipMask & 0x2) != 0)
        {
            testCameraB.setDepthAxisFlipped(!testCameraB.depthAxisFlipped());
        }

        const auto pairResult = xjw::Intersection::intersectPair(
            testCameraA, pixelA.x(), pixelA.y(), testCameraB, pixelB.x(), pixelB.y());
        if (!std::isfinite(pairResult.point[0])
            || !std::isfinite(pairResult.point[1])
            || !std::isfinite(pairResult.point[2]))
        {
            continue;
        }

        const double errorA = reprojectionErrorPx(testCameraA, pairResult.point, pixelA);
        const double errorB = reprojectionErrorPx(testCameraB, pairResult.point, pixelB);
        if (!std::isfinite(errorA) || !std::isfinite(errorB))
        {
            continue;
        }

        const double rmsReprojectionPx = std::sqrt(0.5 * (errorA * errorA + errorB * errorB));
        if (!bestCandidate.valid || rmsReprojectionPx < bestCandidate.rmsReprojectionPx)
        {
            bestCandidate.point = pairResult.point;
            bestCandidate.rmsReprojectionPx = rmsReprojectionPx;
            bestCandidate.valid = true;
        }
    }

    return bestCandidate;
}

} // namespace

BaInputBuildStatus buildBaInputFromMeta(const QJsonObject &meta,
                                        const QStringList &selectedImages,
                                        int minMatches,
                                        BaInputBuildResult *result)
{
    if (!result)
    {
        return BaInputBuildStatus::NoTracks;
    }

    result->cameras.clear();
    result->imagePathByIndex.clear();
    result->beforeCamMeta.clear();
    result->tracks.clear();

    QSet<QString> selectedNormalized;
    for (const QString &path : selectedImages)
    {
        selectedNormalized.insert(xjw::common::project::normalizePath(path));
    }

    QMap<QString, int> cameraIndexByPath;
    const QJsonArray imageArray = meta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : imageArray)
    {
        const QJsonObject object = value.toObject();
        const QString normalizedPath = xjw::common::project::normalizePath(object.value(QStringLiteral("path")).toString());
        if (!selectedNormalized.contains(normalizedPath))
        {
            continue;
        }

        xjw::Camera camera;
        if (!xjw::common::project::cameraFromJson(object.value(QStringLiteral("camera")).toObject(), &camera))
        {
            continue;
        }

        cameraIndexByPath[normalizedPath] = static_cast<int>(result->cameras.size());
        result->cameras.push_back(camera);
        result->imagePathByIndex.append(normalizedPath);
        result->beforeCamMeta.insert(normalizedPath, object.value(QStringLiteral("camera")).toObject());
    }

    if (result->cameras.size() < 2)
    {
        return BaInputBuildStatus::NotEnoughCameras;
    }

    const QJsonArray matchResults = meta.value(QStringLiteral("ipmatch_results")).toArray();
    for (const QJsonValue &value : matchResults)
    {
        if (!value.isObject())
        {
            continue;
        }

        const QJsonObject record = value.toObject();
        QString rawPath0 = record.value(QStringLiteral("image0")).toString();
        QString rawPath1 = record.value(QStringLiteral("image1")).toString();
        if (rawPath0.isEmpty() || rawPath1.isEmpty())
        {
            const QJsonObject settings = record.value(QStringLiteral("settings")).toObject();
            const QJsonArray imageFiles = settings.value(QStringLiteral("image_files")).toArray();
            if (imageFiles.size() >= 2)
            {
                rawPath0 = imageFiles.at(0).toString();
                rawPath1 = imageFiles.at(1).toString();
            }
        }
        if (rawPath0.isEmpty() || rawPath1.isEmpty())
        {
            continue;
        }

        QString imageA;
        QString imageB;
        for (const QString &selected : selectedImages)
        {
            if (xjw::common::project::pathTokenMatchesImage(rawPath0, selected))
            {
                imageA = xjw::common::project::normalizePath(selected);
            }
            if (xjw::common::project::pathTokenMatchesImage(rawPath1, selected))
            {
                imageB = xjw::common::project::normalizePath(selected);
            }
        }

        if (imageA.isEmpty() || imageB.isEmpty() || imageA == imageB)
        {
            continue;
        }
        if (!cameraIndexByPath.contains(imageA) || !cameraIndexByPath.contains(imageB))
        {
            continue;
        }

        QString sidecarPath = record.value(QStringLiteral("settings"))
                                  .toObject()
                                  .value(QStringLiteral("sidecar_json"))
                                  .toString();
        if (sidecarPath.isEmpty())
        {
            sidecarPath = record.value(QStringLiteral("output")).toString() + QStringLiteral(".json");
        }
        if (!QFile::exists(sidecarPath))
        {
            continue;
        }

        QFile sidecarFile(sidecarPath);
        if (!sidecarFile.open(QIODevice::ReadOnly))
        {
            continue;
        }

        const QJsonDocument sidecarDocument = QJsonDocument::fromJson(sidecarFile.readAll());
        sidecarFile.close();
        if (!sidecarDocument.isObject())
        {
            continue;
        }

        const QJsonObject sidecar = sidecarDocument.object();
        const QJsonArray matched0 = sidecar.value(QStringLiteral("matched_points0")).toArray();
        const QJsonArray matched1 = sidecar.value(QStringLiteral("matched_points1")).toArray();
        if (matched0.isEmpty() || matched0.size() != matched1.size())
        {
            continue;
        }
        if (minMatches > 0 && matched0.size() < minMatches)
        {
            continue;
        }

        const QString sideImage0 = sidecar.value(QStringLiteral("image0_path")).toString();
        const QString sideImage1 = sidecar.value(QStringLiteral("image1_path")).toString();
        const bool direct = xjw::common::project::pathTokenMatchesImage(sideImage0, imageA)
                            && xjw::common::project::pathTokenMatchesImage(sideImage1, imageB);
        const bool reverse = xjw::common::project::pathTokenMatchesImage(sideImage0, imageB)
                             && xjw::common::project::pathTokenMatchesImage(sideImage1, imageA);
        if (!direct && !reverse)
        {
            continue;
        }

        const int indexA = cameraIndexByPath.value(imageA);
        const int indexB = cameraIndexByPath.value(imageB);
        const auto &cameraA = result->cameras.at(static_cast<size_t>(indexA));
        const auto &cameraB = result->cameras.at(static_cast<size_t>(indexB));

        for (int pointIndex = 0; pointIndex < matched0.size(); ++pointIndex)
        {
            const QJsonArray a0 = matched0.at(pointIndex).toArray();
            const QJsonArray a1 = matched1.at(pointIndex).toArray();
            if (a0.size() < 2 || a1.size() < 2)
            {
                continue;
            }

            const QPointF uvA = direct
                ? QPointF(a0.at(0).toDouble(), a0.at(1).toDouble())
                : QPointF(a1.at(0).toDouble(), a1.at(1).toDouble());
            const QPointF uvB = direct
                ? QPointF(a1.at(0).toDouble(), a1.at(1).toDouble())
                : QPointF(a0.at(0).toDouble(), a0.at(1).toDouble());

            xjw::BATrack track;
            const PairIntersectionCandidate init = triangulatePairWithDirectionFallback(cameraA, uvA, cameraB, uvB);
            if (init.valid)
            {
                track.initialPoint = init.point;
            }
            else
            {
                const auto cA = cameraA.cameraCenter();
                const auto cB = cameraB.cameraCenter();
                track.initialPoint = {{
                    0.5 * (cA[0] + cB[0]),
                    0.5 * (cA[1] + cB[1]),
                    0.5 * (cA[2] + cB[2])
                }};
            }

            track.observations.push_back(xjw::BAObservation{indexA, uvA.x(), uvA.y()});
            track.observations.push_back(xjw::BAObservation{indexB, uvB.x(), uvB.y()});
            result->tracks.push_back(std::move(track));
        }
    }

    if (result->tracks.empty())
    {
        return BaInputBuildStatus::NoTracks;
    }

    return BaInputBuildStatus::Ok;
}

} // namespace xjw::core::project
