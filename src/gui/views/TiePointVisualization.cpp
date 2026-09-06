#include "TiePointVisualization.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xjw::gui::tie_points
{

namespace
{

constexpr std::size_t kCancellationCheckInterval = 4096;

bool isCancellationRequested(const std::atomic_bool *cancellationFlag)
{
    return cancellationFlag
        && cancellationFlag->load(std::memory_order_relaxed);
}

bool finiteNonNegativeNumber(const QJsonObject &object,
                             const QString &key,
                             double *value)
{
    if (!value)
    {
        return false;
    }
    const QJsonValue field = object.value(key);
    if (!field.isDouble())
    {
        return false;
    }
    const double parsed = field.toDouble();
    if (!std::isfinite(parsed) || parsed < 0.0)
    {
        return false;
    }
    *value = parsed;
    return true;
}

bool finitePositiveNumber(const QJsonObject &object,
                          const QString &key,
                          double *value)
{
    return finiteNonNegativeNumber(object, key, value) && *value > 0.0;
}

bool nonNegativeInteger(const QJsonObject &object,
                        const QString &key,
                        int *value)
{
    double parsed = 0.0;
    if (!value || !finiteNonNegativeNumber(object, key, &parsed)
        || parsed > static_cast<double>(std::numeric_limits<int>::max())
        || std::trunc(parsed) != parsed)
    {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

ScalarRange rangeForValues(const QVector<double> &values)
{
    bool found_finite = false;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const double value : values)
    {
        if (!std::isfinite(value))
        {
            continue;
        }
        found_finite = true;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return found_finite ? ScalarRange{minimum, maximum} : ScalarRange{nan, nan};
}

ScalarRange rangeForValues(const QVector<int> &values)
{
    if (values.isEmpty())
    {
        return {};
    }
    const auto [minimum, maximum] = std::minmax_element(
        values.cbegin(), values.cend());
    return {double(*minimum), double(*maximum)};
}

} // namespace

bool ScalarRange::isValid() const
{
    return std::isfinite(minimum) && std::isfinite(maximum) && maximum >= minimum;
}

double ScalarRange::normalize(double value) const
{
    if (!isValid() || !std::isfinite(value) || maximum <= minimum)
    {
        return 0.5;
    }
    return std::clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
}

bool ImageCountMetadata::isValidFor(qsizetype pointCount) const
{
    return pointCount > 0 && counts.size() == pointCount;
}

bool PrunePreviewQuery::isValid() const
{
    return std::isfinite(threshold) && threshold >= 0.0
        && (criterion != QualityCriterion::ImageCount
            || std::trunc(threshold) == threshold);
}

bool QualityMetadata::isValidFor(qsizetype pointCount) const
{
    return pointCount > 0 && sourcePointCount == pointCount
        && (hasCriterion(QualityCriterion::ReprojectionError, pointCount)
            || hasCriterion(QualityCriterion::ReconstructionUncertainty, pointCount)
            || hasCriterion(QualityCriterion::ImageCount, pointCount)
            || hasCriterion(QualityCriterion::ProjectionAccuracy, pointCount)
            || hasCriterion(QualityCriterion::MinimumTriangulationAngle,
                            pointCount));
}

bool QualityMetadata::hasCriterion(QualityCriterion criterion,
                                   qsizetype pointCount) const
{
    if (pointCount < -1)
    {
        return false;
    }
    const qsizetype expected = pointCount == -1 ? sourcePointCount : pointCount;
    if (expected <= 0 || sourcePointCount != expected)
    {
        return false;
    }
    switch (criterion)
    {
    case QualityCriterion::ReprojectionError:
        return reprojectionErrors.size() == expected;
    case QualityCriterion::ReconstructionUncertainty:
        return reconstructionUncertainties.size() == expected;
    case QualityCriterion::ImageCount:
        return imageCounts.size() == expected;
    case QualityCriterion::ProjectionAccuracy:
        return projectionAccuracies.size() == expected;
    case QualityCriterion::MinimumTriangulationAngle:
        return minimumTriangulationAngles.size() == expected;
    }
    return false;
}

ScalarRange QualityMetadata::range(QualityCriterion criterion) const
{
    switch (criterion)
    {
    case QualityCriterion::ReprojectionError:
        return reprojectionErrorRange;
    case QualityCriterion::ReconstructionUncertainty:
        return reconstructionUncertaintyRange;
    case QualityCriterion::ImageCount:
        return imageCountRange;
    case QualityCriterion::ProjectionAccuracy:
        return projectionAccuracyRange;
    case QualityCriterion::MinimumTriangulationAngle:
        return minimumTriangulationAngleRange;
    }
    return {};
}

QColor elevationColor(double elevation, const ScalarRange &range)
{
    return scalarRampColor(range.normalize(elevation));
}

QColor imageCountColor(int imageCount, const ScalarRange &range)
{
    return scalarRampColor(1.0 - range.normalize(static_cast<double>(imageCount)));
}

QColor scalarRampColor(double normalizedValue)
{
    // Metashape 风格的高对比色带。固定满饱和度并适度压低亮度，
    // 在白色三维背景和点精灵抗锯齿下仍能保持鲜明、清晰。
    const double value = std::clamp(normalizedValue, 0.0, 1.0);
    return QColor::fromHsvF((1.0 - value) * (2.0 / 3.0), 1.0, 0.92);
}

float pointSizeForMode(ColorMode mode)
{
    return mode == ColorMode::Color ? 1.8f : 3.0f;
}

QString inferSidecarPath(const QString &pointCloudPath)
{
    const QFileInfo pointCloudInfo(pointCloudPath);
    if (pointCloudInfo.filePath().trimmed().isEmpty())
    {
        return QString();
    }

    const QString baseName = pointCloudInfo.completeBaseName();
    QStringList candidates;
    if (baseName == QLatin1String("sfm_sparse"))
    {
        candidates.append(QStringLiteral("sfm_sparse_points.json"));
    }
    candidates.append(QStringLiteral("sparse_cloud_points.json"));
    candidates.append(baseName + QStringLiteral("_points.json"));

    const QDir directory = pointCloudInfo.absoluteDir();
    for (const QString &candidate : candidates)
    {
        const QString path = directory.filePath(candidate);
        if (QFileInfo::exists(path))
        {
            return QDir::cleanPath(path);
        }
    }
    return QString();
}

ImageCountMetadata loadImageCountMetadata(const QString &sidecarPath)
{
    ImageCountMetadata result;
    QualityMetadata quality = loadQualityMetadata(sidecarPath);
    result.counts = std::move(quality.imageCounts);
    result.errorMessage = quality.errorMessage;
    if (result.counts.isEmpty() && result.errorMessage.isEmpty())
    {
        result.errorMessage = QStringLiteral("连接点观测数据缺少有效的 track_len 字段");
    }
    return result;
}

QualityMetadata loadQualityMetadata(
    const QString &sidecarPath,
    const std::atomic_bool *cancellationFlag)
{
    QualityMetadata result;
    if (sidecarPath.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("未找到连接点观测数据文件");
        return result;
    }

    QFile file(sidecarPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        result.errorMessage = QStringLiteral("无法读取连接点观测数据: %1").arg(sidecarPath);
        return result;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (isCancellationRequested(cancellationFlag))
    {
        result.errorMessage = QStringLiteral("连接点质量元数据读取已取消");
        return result;
    }
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        result.errorMessage = QStringLiteral("连接点观测数据格式无效: %1")
                                  .arg(parseError.errorString());
        return result;
    }

    const QJsonArray points = document.object().value(QStringLiteral("points")).toArray();
    if (points.isEmpty())
    {
        result.errorMessage = QStringLiteral("连接点观测数据中没有点记录");
        return result;
    }

    result.sourcePointCount = points.size();
    result.reprojectionErrors.reserve(points.size());
    result.reconstructionUncertainties.reserve(points.size());
    result.imageCounts.reserve(points.size());
    result.projectionAccuracies.reserve(points.size());
    result.minimumTriangulationAngles.reserve(points.size());
    bool has_reprojection_errors = true;
    bool has_reconstruction_uncertainties = true;
    bool has_image_counts = true;
    bool has_projection_accuracies = true;
    bool has_minimum_angles = true;
    bool has_positive_minimum_angle = false;
    for (qsizetype index = 0; index < points.size(); ++index)
    {
        if (static_cast<std::size_t>(index) % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            result = {};
            result.errorMessage = QStringLiteral("连接点质量元数据读取已取消");
            return result;
        }
        const QJsonValue point_value = points.at(index);
        const QJsonObject point = point_value.toObject();
        const QJsonValue clean_value = point.value(QStringLiteral("clean_tie_points"));
        const QJsonObject clean = clean_value.toObject();
        if (!point_value.isObject())
        {
            has_reprojection_errors = false;
            has_reconstruction_uncertainties = false;
            has_image_counts = false;
            has_projection_accuracies = false;
            has_minimum_angles = false;
        }

        double reprojection_error = 0.0;
        if (!finiteNonNegativeNumber(clean, QStringLiteral("reprojection_error"), &reprojection_error))
        {
            has_reprojection_errors = false;
        }
        result.reprojectionErrors.push_back(reprojection_error);

        double reconstruction_uncertainty = std::numeric_limits<double>::quiet_NaN();
        const QJsonValue has_geometry_value = clean.value(QStringLiteral("has_projection_geometry"));
        if (!clean_value.isObject() || !has_geometry_value.isBool())
        {
            has_reconstruction_uncertainties = false;
        }
        else if (has_geometry_value.toBool())
        {
            if (clean.value(QStringLiteral("reconstruction_uncertainty_infinite")).toBool(false))
            {
                reconstruction_uncertainty = std::numeric_limits<double>::infinity();
            }
            else if (!finitePositiveNumber(
                         clean, QStringLiteral("reconstruction_uncertainty"), &reconstruction_uncertainty))
            {
                has_reconstruction_uncertainties = false;
            }
        }
        result.reconstructionUncertainties.push_back(reconstruction_uncertainty);

        int image_count = 0;
        if (!nonNegativeInteger(clean, QStringLiteral("image_count"), &image_count) &&
            !nonNegativeInteger(point, QStringLiteral("track_len"), &image_count))
        {
            has_image_counts = false;
        }
        result.imageCounts.push_back(image_count);

        double projection_accuracy = 0.0;
        if (!finiteNonNegativeNumber(clean, QStringLiteral("projection_accuracy"), &projection_accuracy))
        {
            has_projection_accuracies = false;
        }
        result.projectionAccuracies.push_back(projection_accuracy);

        double minimum_angle = 0.0;
        if (!finiteNonNegativeNumber(
                point, QStringLiteral("min_tri_angle_deg"), &minimum_angle))
        {
            has_minimum_angles = false;
        }
        else if (minimum_angle > 0.0)
        {
            has_positive_minimum_angle = true;
        }
        result.minimumTriangulationAngles.push_back(minimum_angle);
    }
    if (!has_reprojection_errors)
    {
        result.reprojectionErrors.clear();
    }
    if (!has_reconstruction_uncertainties)
    {
        result.reconstructionUncertainties.clear();
    }
    if (!has_image_counts)
    {
        result.imageCounts.clear();
    }
    if (!has_projection_accuracies)
    {
        result.projectionAccuracies.clear();
    }
    if (!has_minimum_angles || !has_positive_minimum_angle)
    {
        result.minimumTriangulationAngles.clear();
    }
    result.reprojectionErrorRange = rangeForValues(result.reprojectionErrors);
    result.reconstructionUncertaintyRange = rangeForValues(
        result.reconstructionUncertainties);
    result.imageCountRange = rangeForValues(result.imageCounts);
    result.projectionAccuracyRange = rangeForValues(result.projectionAccuracies);
    result.minimumTriangulationAngleRange = rangeForValues(
        result.minimumTriangulationAngles);
    if (result.reprojectionErrors.isEmpty()
        && result.reconstructionUncertainties.isEmpty()
        && result.imageCounts.isEmpty()
        && result.projectionAccuracies.isEmpty()
        && result.minimumTriangulationAngles.isEmpty())
    {
        result.errorMessage = QStringLiteral(
            "连接点质量元数据中没有可用的逐点清理指标");
    }
    return result;
}

PruneCandidateQueryResult queryPruneCandidates(
    const QualityMetadata &metadata,
    const PrunePreviewQuery &query,
    qsizetype pointCount,
    const std::atomic_bool *cancellationFlag,
    std::size_t maximumReturnedIndices,
    const std::vector<std::uint32_t> *excludedIndices)
{
    PruneCandidateQueryResult result;
    if (pointCount <= 0)
    {
        result.errorMessage = QStringLiteral("连接点数量必须大于零");
        return result;
    }
    if (!query.isValid())
    {
        result.errorMessage = QStringLiteral("连接点清理阈值不是有限数值");
        return result;
    }
    if (!metadata.hasCriterion(query.criterion, pointCount))
    {
        result.errorMessage = QStringLiteral("当前连接点缺少所选标准的逐点质量数据");
        return result;
    }
    if (pointCount > static_cast<qsizetype>(
            std::numeric_limits<std::uint32_t>::max()))
    {
        result.errorMessage = QStringLiteral("连接点数量超过候选索引表示范围");
        return result;
    }
    if (excludedIndices
        && (!std::is_sorted(excludedIndices->cbegin(), excludedIndices->cend())
            || (!excludedIndices->empty()
                && excludedIndices->back() >= static_cast<std::uint32_t>(pointCount))))
    {
        result.errorMessage = QStringLiteral("连接点暂删索引无效");
        return result;
    }

    const auto is_rejected = [&metadata, &query, excludedIndices](qsizetype index)
    {
        if (excludedIndices
            && std::binary_search(excludedIndices->cbegin(),
                                  excludedIndices->cend(),
                                  static_cast<std::uint32_t>(index)))
        {
            return false;
        }
        switch (query.criterion)
        {
        case QualityCriterion::ReprojectionError:
            return metadata.reprojectionErrors.at(index) > query.threshold;
        case QualityCriterion::ReconstructionUncertainty:
            return metadata.reconstructionUncertainties.at(index) > query.threshold;
        case QualityCriterion::ImageCount:
            return double(metadata.imageCounts.at(index)) <= query.threshold;
        case QualityCriterion::ProjectionAccuracy:
            return metadata.projectionAccuracies.at(index) > query.threshold;
        case QualityCriterion::MinimumTriangulationAngle:
            return metadata.minimumTriangulationAngles.at(index) < query.threshold;
        }
        return false;
    };

    for (qsizetype index = 0; index < pointCount; ++index)
    {
        if (static_cast<std::size_t>(index) % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            result.errorMessage = QStringLiteral("连接点清理候选查询已取消");
            return result;
        }
        if (is_rejected(index))
        {
            ++result.candidateCount;
        }
    }
    if (isCancellationRequested(cancellationFlag))
    {
        result.candidateCount = 0;
        result.errorMessage = QStringLiteral("连接点清理候选查询已取消");
        return result;
    }

    const std::size_t returned_count = std::min(
        static_cast<std::size_t>(result.candidateCount),
        maximumReturnedIndices);
    if (returned_count == 0)
    {
        return result;
    }

    result.indices.reserve(returned_count);
    const std::uint64_t last_candidate_rank = static_cast<std::uint64_t>(
        result.candidateCount - 1);
    const std::uint64_t last_sample_rank = static_cast<std::uint64_t>(
        returned_count - 1);
    auto sampled_candidate_rank = [last_candidate_rank, last_sample_rank](
                                      std::size_t sampleIndex)
    {
        if (last_sample_rank == 0)
        {
            return last_candidate_rank / 2;
        }
        return static_cast<std::uint64_t>(sampleIndex) * last_candidate_rank
            / last_sample_rank;
    };

    std::uint64_t candidate_rank = 0;
    std::size_t sample_index = 0;
    std::uint64_t target_rank = sampled_candidate_rank(sample_index);
    for (qsizetype index = 0; index < pointCount; ++index)
    {
        if (static_cast<std::size_t>(index) % kCancellationCheckInterval == 0
            && isCancellationRequested(cancellationFlag))
        {
            result.indices.clear();
            result.candidateCount = 0;
            result.errorMessage = QStringLiteral("连接点清理候选查询已取消");
            return result;
        }
        if (!is_rejected(index))
        {
            continue;
        }
        if (candidate_rank == target_rank)
        {
            result.indices.push_back(static_cast<std::uint32_t>(index));
            ++sample_index;
            if (sample_index == returned_count)
            {
                break;
            }
            target_rank = sampled_candidate_rank(sample_index);
        }
        ++candidate_rank;
    }
    if (isCancellationRequested(cancellationFlag))
    {
        result.indices.clear();
        result.candidateCount = 0;
        result.errorMessage = QStringLiteral("连接点清理候选查询已取消");
    }
    return result;
}

} // namespace xjw::gui::tie_points
