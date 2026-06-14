#include "BaInputBuilder.h"

#include "Intersection.h"
#include "project/ProjectCommonUtils.h"
#include "tracks/MultiViewTrackBuilder.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPointF>
#include <QSet>

#include <cmath>
#include <limits>
#include <map>
#include <utility>

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

struct IndexedObservation
{
    int cameraIndex = -1;
    QPointF uv;
};

using IndexedFeatureKey = std::pair<int, xjw::FeatureIdx>;

bool readFeatureIndex(const QJsonArray &indices, int index, xjw::FeatureIdx *featureIdx)
{
    if (!featureIdx || index < 0 || index >= indices.size())
    {
        return false;
    }

    bool ok = false;
    const double raw = indices.at(index).toDouble(-1.0);
    if (!std::isfinite(raw) || raw < 0.0
        || raw > static_cast<double>(std::numeric_limits<xjw::FeatureIdx>::max()))
    {
        return false;
    }

    const double rounded = std::floor(raw);
    if (std::fabs(raw - rounded) > 1e-6)
    {
        return false;
    }

    const auto parsed = static_cast<unsigned long long>(rounded);
    if (parsed > static_cast<unsigned long long>(std::numeric_limits<xjw::FeatureIdx>::max()))
    {
        return false;
    }

    *featureIdx = static_cast<xjw::FeatureIdx>(parsed);
    ok = true;
    return ok;
}

void rememberObservation(std::map<IndexedFeatureKey, IndexedObservation> *observations,
                         int cameraIndex,
                         xjw::FeatureIdx featureIdx,
                         const QPointF &uv)
{
    if (!observations || cameraIndex < 0 || featureIdx == xjw::kInvalidFeatureIdx)
    {
        return;
    }

    const IndexedFeatureKey key{cameraIndex, featureIdx};
    observations->emplace(key, IndexedObservation{cameraIndex, uv});
}

std::array<double, 3> midpointBetweenCameras(const xjw::Camera &cameraA, const xjw::Camera &cameraB)
{
    const auto cA = cameraA.cameraCenter();
    const auto cB = cameraB.cameraCenter();
    return {{
        0.5 * (cA[0] + cB[0]),
        0.5 * (cA[1] + cB[1]),
        0.5 * (cA[2] + cB[2])
    }};
}

xjw::BATrack makeBaTrackFromIndexedTrack(
    const xjw::Track &track,
    const std::map<IndexedFeatureKey, IndexedObservation> &observationsByIndexedFeature,
    const std::vector<xjw::Camera> &cameras)
{
    std::vector<IndexedObservation> observations;
    observations.reserve(track.elements.size());

    for (const xjw::TrackElement &element : track.elements)
    {
        if (element.imageId > static_cast<xjw::ImageId>(std::numeric_limits<int>::max()))
        {
            continue;
        }

        const int cameraIndex = static_cast<int>(element.imageId);
        const auto it = observationsByIndexedFeature.find(IndexedFeatureKey{cameraIndex, element.featureIdx});
        if (it != observationsByIndexedFeature.end())
        {
            observations.push_back(it->second);
        }
    }

    xjw::BATrack baTrack;
    if (observations.size() < 2)
    {
        return baTrack;
    }

    bool initialized = false;
    for (std::size_t i = 0; i < observations.size() && !initialized; ++i)
    {
        for (std::size_t j = i + 1; j < observations.size(); ++j)
        {
            const IndexedObservation &obsA = observations[i];
            const IndexedObservation &obsB = observations[j];
            if (obsA.cameraIndex < 0 || obsB.cameraIndex < 0
                || obsA.cameraIndex >= static_cast<int>(cameras.size())
                || obsB.cameraIndex >= static_cast<int>(cameras.size()))
            {
                continue;
            }

            const PairIntersectionCandidate init = triangulatePairWithDirectionFallback(
                cameras[static_cast<std::size_t>(obsA.cameraIndex)],
                obsA.uv,
                cameras[static_cast<std::size_t>(obsB.cameraIndex)],
                obsB.uv);
            if (init.valid)
            {
                baTrack.initialPoint = init.point;
                initialized = true;
                break;
            }
        }
    }

    if (!initialized)
    {
        const IndexedObservation &obsA = observations[0];
        const IndexedObservation &obsB = observations[1];
        baTrack.initialPoint = midpointBetweenCameras(
            cameras[static_cast<std::size_t>(obsA.cameraIndex)],
            cameras[static_cast<std::size_t>(obsB.cameraIndex)]);
    }

    for (const IndexedObservation &observation : observations)
    {
        baTrack.observations.push_back(xjw::BAObservation{
            observation.cameraIndex,
            observation.uv.x(),
            observation.uv.y()
        });
    }

    return baTrack;
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
    result->sidecarV2PairCount = 0;
    result->multiViewTrackCount = 0;
    result->rejectedConflictTrackCount = 0;

    xjw::MultiViewTrackBuilder multiViewTrackBuilder;
    std::map<IndexedFeatureKey, IndexedObservation> observationsByIndexedFeature;

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
        const QJsonArray indices0 = sidecar.value(QStringLiteral("matched_indices0")).toArray();
        const QJsonArray indices1 = sidecar.value(QStringLiteral("matched_indices1")).toArray();
        const QJsonArray matchedScores = sidecar.value(QStringLiteral("matched_scores")).toArray();
        const bool hasIndexedMatches =
            !indices0.isEmpty()
            && indices0.size() == matched0.size()
            && indices1.size() == matched1.size();
        const bool hasMatchScores = matchedScores.size() == matched0.size();

        std::vector<xjw::MultiViewTrackBuilder::MatchIndexPair> indexedMatches;
        if (hasIndexedMatches)
        {
            indexedMatches.reserve(static_cast<std::size_t>(matched0.size()));
        }

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

            if (hasIndexedMatches)
            {
                xjw::FeatureIdx feature0 = xjw::kInvalidFeatureIdx;
                xjw::FeatureIdx feature1 = xjw::kInvalidFeatureIdx;
                if (!readFeatureIndex(indices0, pointIndex, &feature0)
                    || !readFeatureIndex(indices1, pointIndex, &feature1))
                {
                    continue;
                }

                const xjw::FeatureIdx featureA = direct ? feature0 : feature1;
                const xjw::FeatureIdx featureB = direct ? feature1 : feature0;
                const float matchScore = hasMatchScores
                    ? static_cast<float>(matchedScores.at(pointIndex).toDouble(1.0))
                    : 1.0f;
                indexedMatches.emplace_back(featureA, featureB, matchScore);
                rememberObservation(&observationsByIndexedFeature, indexA, featureA, uvA);
                rememberObservation(&observationsByIndexedFeature, indexB, featureB, uvB);
                ++result->sidecarV2PairCount;
                continue;
            }

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

        if (!indexedMatches.empty())
        {
            multiViewTrackBuilder.addMatchPair(
                static_cast<xjw::ImageId>(indexA),
                static_cast<xjw::ImageId>(indexB),
                indexedMatches);
        }
    }

    const xjw::MultiViewTrackBuildResult multiViewResult = multiViewTrackBuilder.build();
    result->multiViewTrackCount = static_cast<int>(multiViewResult.tracks.size());
    result->rejectedConflictTrackCount = multiViewResult.rejectedConflictComponents;
    for (const xjw::Track &track : multiViewResult.tracks)
    {
        xjw::BATrack baTrack = makeBaTrackFromIndexedTrack(track, observationsByIndexedFeature, result->cameras);
        if (baTrack.observations.size() >= 2)
        {
            result->tracks.push_back(std::move(baTrack));
        }
    }

    if (result->tracks.empty())
    {
        return BaInputBuildStatus::NoTracks;
    }

    return BaInputBuildStatus::Ok;
}

} // namespace xjw::core::project
