#include "SparseSceneOverlapAnalyzer.h"

#include "io/PathIO.h"

#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace xjw::matchphotos
{
namespace
{

struct SparsePoint
{
    std::array<double, 3> xyz{};
    std::vector<int> observations;
};

std::uint64_t pairKey(int indexA, int indexB)
{
    const auto first = static_cast<std::uint32_t>(std::min(indexA, indexB));
    const auto second = static_cast<std::uint32_t>(std::max(indexA, indexB));
    return (static_cast<std::uint64_t>(first) << 32U) | second;
}

QString normalizedPath(const QString &path)
{
    return QDir::fromNativeSeparators(QDir::cleanPath(path.trimmed())).toLower();
}

bool finitePoint(const QJsonArray &values, std::array<double, 3> *point)
{
    if (!point || values.size() < 3)
    {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        (*point)[axis] = values.at(axis).toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite((*point)[axis]))
        {
            return false;
        }
    }
    return true;
}

int resolveObservationImage(const QJsonObject &observation,
                            const QHash<QString, int> &pathIndices,
                            const QHash<QString, int> &uniqueNameIndices,
                            int imageCount)
{
    const QString path = normalizedPath(observation.value(QStringLiteral("image_path")).toString());
    if (!path.isEmpty())
    {
        const auto pathIt = pathIndices.constFind(path);
        if (pathIt != pathIndices.constEnd())
        {
            return pathIt.value();
        }
    }

    const QString name = observation.value(QStringLiteral("image_name")).toString().trimmed().toLower();
    if (!name.isEmpty())
    {
        const auto nameIt = uniqueNameIndices.constFind(name);
        if (nameIt != uniqueNameIndices.constEnd())
        {
            return nameIt.value();
        }
    }

    const int imageId = observation.value(QStringLiteral("image_id")).toInt(-1);
    return imageId >= 0 && imageId < imageCount ? imageId : -1;
}

std::array<double, 2> robustLimits(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t last = values.size() - 1;
    const std::size_t low = static_cast<std::size_t>(0.02 * static_cast<double>(last));
    const std::size_t high = static_cast<std::size_t>(0.98 * static_cast<double>(last));
    return {{values[low], values[std::max(low, high)]}};
}

std::vector<int> projectionSampleIndices(const std::vector<SparsePoint> &points, int maximum)
{
    std::array<std::array<double, 2>, 3> limits{};
    for (int axis = 0; axis < 3; ++axis)
    {
        std::vector<double> values;
        values.reserve(points.size());
        for (const SparsePoint &point : points)
        {
            values.push_back(point.xyz[axis]);
        }
        limits[axis] = robustLimits(std::move(values));
    }

    std::vector<int> inliers;
    inliers.reserve(points.size());
    for (int index = 0; index < static_cast<int>(points.size()); ++index)
    {
        bool inside = true;
        for (int axis = 0; axis < 3; ++axis)
        {
            inside = inside && points[static_cast<std::size_t>(index)].xyz[axis] >= limits[axis][0] &&
                points[static_cast<std::size_t>(index)].xyz[axis] <= limits[axis][1];
        }
        if (inside)
        {
            inliers.push_back(index);
        }
    }

    const int safeMaximum = std::max(32, maximum);
    if (static_cast<int>(inliers.size()) <= safeMaximum)
    {
        return inliers;
    }

    std::vector<int> sampled;
    sampled.reserve(static_cast<std::size_t>(safeMaximum));
    for (int sample = 0; sample < safeMaximum; ++sample)
    {
        const std::size_t index = static_cast<std::size_t>(sample) * inliers.size() /
            static_cast<std::size_t>(safeMaximum);
        sampled.push_back(inliers[std::min(index, inliers.size() - 1)]);
    }
    return sampled;
}

std::array<double, 3> normalizedViewingDirection(const Camera &source)
{
    const Camera camera = source.normalizedForPositiveDepth();
    const auto rotation = camera.cameraToWorldRotation();
    std::array<double, 3> direction{{rotation[2], rotation[5], rotation[8]}};
    const double norm = std::hypot(direction[0], direction[1], direction[2]);
    if (!std::isfinite(norm) || norm <= 1.0e-12)
    {
        return {};
    }
    for (double &value : direction)
    {
        value /= norm;
    }
    return direction;
}

} // namespace

bool SparseSceneOverlapAnalyzer::analyzeFile(
    const QString &sidecarPath,
    const std::vector<OverlapImageInput> &images,
    const SparseSceneOverlapOptions &options,
    OverlapAnalysisResult *result,
    SparseSceneOverlapStats *stats,
    QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!result || images.size() < 2)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("稀疏场景重叠分析至少需要两张影像");
        }
        return false;
    }
    result->centers.clear();
    result->footprintRadii.clear();
    result->pairs.clear();
    result->detail.clear();
    if (stats)
    {
        *stats = {};
    }

    QString readError;
    const QByteArray bytes = common::io::readFileBytes(sidecarPath, &readError);
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
    if (bytes.isEmpty() || !document.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = readError.isEmpty()
                ? QStringLiteral("无法解析已有 SfM 稀疏点: %1 (%2)")
                      .arg(sidecarPath, parseError.errorString())
                : readError;
        }
        return false;
    }

    QHash<QString, int> pathIndices;
    QHash<QString, int> nameCounts;
    QHash<QString, int> uniqueNameIndices;
    for (int index = 0; index < static_cast<int>(images.size()); ++index)
    {
        const QString path = QString::fromUtf8(images[static_cast<std::size_t>(index)].imagePath);
        pathIndices.insert(normalizedPath(path), index);
        const QString name = QFileInfo(path).fileName().toLower();
        nameCounts[name] = nameCounts.value(name) + 1;
        uniqueNameIndices[name] = index;
    }
    for (auto it = nameCounts.constBegin(); it != nameCounts.constEnd(); ++it)
    {
        if (it.value() != 1)
        {
            uniqueNameIndices.remove(it.key());
        }
    }

    std::vector<SparsePoint> points;
    const QJsonArray pointValues = document.object().value(QStringLiteral("points")).toArray();
    points.reserve(static_cast<std::size_t>(pointValues.size()));
    std::vector<int> observedCounts(images.size(), 0);
    QHash<quint64, int> sharedCounts;
    for (const QJsonValue &pointValue : pointValues)
    {
        const QJsonObject pointObject = pointValue.toObject();
        SparsePoint point;
        if (!finitePoint(pointObject.value(QStringLiteral("point_xyz")).toArray(), &point.xyz))
        {
            continue;
        }

        QSet<int> observationSet;
        for (const QJsonValue &observationValue :
             pointObject.value(QStringLiteral("observations")).toArray())
        {
            const int imageIndex = resolveObservationImage(
                observationValue.toObject(),
                pathIndices,
                uniqueNameIndices,
                static_cast<int>(images.size()));
            if (imageIndex >= 0)
            {
                observationSet.insert(imageIndex);
            }
        }
        point.observations.assign(observationSet.cbegin(), observationSet.cend());
        std::sort(point.observations.begin(), point.observations.end());
        for (int imageIndex : point.observations)
        {
            ++observedCounts[static_cast<std::size_t>(imageIndex)];
        }
        for (std::size_t first = 0; first < point.observations.size(); ++first)
        {
            for (std::size_t second = first + 1; second < point.observations.size(); ++second)
            {
                const quint64 key = static_cast<quint64>(
                    pairKey(point.observations[first], point.observations[second]));
                sharedCounts[key] = sharedCounts.value(key) + 1;
            }
        }
        points.push_back(std::move(point));
    }

    if (points.size() < 32)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("已有 SfM 稀疏点不足：有效 %1，至少需要 32")
                                .arg(static_cast<qulonglong>(points.size()));
        }
        return false;
    }

    const std::vector<int> sampleIndices = projectionSampleIndices(
        points, options.maxProjectionSamples);
    const std::size_t visibilityWordCount = (sampleIndices.size() + 63U) / 64U;
    std::vector<std::vector<std::uint64_t>> visible(
        images.size(), std::vector<std::uint64_t>(visibilityWordCount, 0));
    std::vector<int> visibleCounts(images.size(), 0);
    std::vector<std::array<double, 3>> viewingDirections;
    viewingDirections.reserve(images.size());
    for (int imageIndex = 0; imageIndex < static_cast<int>(images.size()); ++imageIndex)
    {
        const OverlapImageInput &image = images[static_cast<std::size_t>(imageIndex)];
        viewingDirections.push_back(normalizedViewingDirection(image.camera));
        const double marginX = std::max(0.0, options.imageMarginFraction) * image.width;
        const double marginY = std::max(0.0, options.imageMarginFraction) * image.height;
        for (std::size_t sampleIndex = 0; sampleIndex < sampleIndices.size(); ++sampleIndex)
        {
            const auto &xyz = points[static_cast<std::size_t>(sampleIndices[sampleIndex])].xyz;
            double pixel[2]{};
            if (image.camera.projectWorldPoint(xyz.data(), pixel) &&
                pixel[0] >= -marginX && pixel[0] <= image.width + marginX &&
                pixel[1] >= -marginY && pixel[1] <= image.height + marginY)
            {
                visible[static_cast<std::size_t>(imageIndex)][sampleIndex / 64U] |=
                    std::uint64_t{1} << (sampleIndex % 64U);
                ++visibleCounts[static_cast<std::size_t>(imageIndex)];
            }
        }
    }

    int covisibilityPairs = 0;
    int frustumPairs = 0;
    for (int indexA = 0; indexA < static_cast<int>(images.size()); ++indexA)
    {
        for (int indexB = indexA + 1; indexB < static_cast<int>(images.size()); ++indexB)
        {
            const int shared = sharedCounts.value(
                static_cast<quint64>(pairKey(indexA, indexB)));
            int jointVisible = 0;
            for (std::size_t wordIndex = 0; wordIndex < visibilityWordCount; ++wordIndex)
            {
                jointVisible += std::popcount(
                    visible[static_cast<std::size_t>(indexA)][wordIndex] &
                    visible[static_cast<std::size_t>(indexB)][wordIndex]);
            }
            const int minObserved = std::min(observedCounts[static_cast<std::size_t>(indexA)],
                                             observedCounts[static_cast<std::size_t>(indexB)]);
            const int minVisible = std::min(visibleCounts[static_cast<std::size_t>(indexA)],
                                            visibleCounts[static_cast<std::size_t>(indexB)]);
            const double covisibility = minObserved > 0
                ? static_cast<double>(shared) / minObserved : 0.0;
            const double projectedOverlap = minVisible > 0
                ? static_cast<double>(jointVisible) / minVisible : 0.0;
            const auto &directionA = viewingDirections[static_cast<std::size_t>(indexA)];
            const auto &directionB = viewingDirections[static_cast<std::size_t>(indexB)];
            const double viewingDirectionCosine = std::clamp(
                directionA[0] * directionB[0] + directionA[1] * directionB[1] +
                    directionA[2] * directionB[2],
                -1.0,
                1.0);
            const bool acceptedByCovisibility = shared >= std::max(1, options.minSharedPointCount);
            const bool acceptedByFrustum =
                jointVisible >= std::max(1, options.minJointVisibleSamples) &&
                projectedOverlap >= std::clamp(options.minProjectedOverlap, 0.0, 1.0) &&
                viewingDirectionCosine >=
                    std::clamp(options.minViewingDirectionCosine, -1.0, 1.0);
            if (!acceptedByCovisibility && !acceptedByFrustum)
            {
                continue;
            }

            covisibilityPairs += acceptedByCovisibility ? 1 : 0;
            frustumPairs += acceptedByFrustum ? 1 : 0;
            const double directionAlignment = 0.5 * (1.0 + viewingDirectionCosine);
            const double score = std::clamp(
                std::max(covisibility, projectedOverlap * directionAlignment), 0.0, 1.0);
            result->pairs.push_back({indexA, indexB, 0.0, score});
        }
    }
    std::sort(result->pairs.begin(), result->pairs.end(), [](const auto &lhs, const auto &rhs)
    {
        if (lhs.overlapScore != rhs.overlapScore)
        {
            return lhs.overlapScore > rhs.overlapScore;
        }
        return std::tie(lhs.indexA, lhs.indexB) < std::tie(rhs.indexA, rhs.indexB);
    });

    result->centers.reserve(images.size());
    for (const OverlapImageInput &image : images)
    {
        result->centers.push_back(image.camera.cameraCenter());
    }
    const QString detail = QStringLiteral(
        "已有 SfM 场景预选：有效点 %1，投影样本 %2，候选 %3，共视证据 %4，视锥证据 %5")
        .arg(static_cast<qulonglong>(points.size()))
        .arg(static_cast<qulonglong>(sampleIndices.size()))
        .arg(static_cast<qulonglong>(result->pairs.size()))
        .arg(covisibilityPairs)
        .arg(frustumPairs);
    result->detail = detail.toStdString();
    if (stats)
    {
        stats->validPointCount = static_cast<int>(points.size());
        stats->sampledPointCount = static_cast<int>(sampleIndices.size());
        stats->covisibilityPairCount = covisibilityPairs;
        stats->frustumPairCount = frustumPairs;
        stats->detail = detail;
    }
    return !result->pairs.empty();
}

} // namespace xjw::matchphotos
