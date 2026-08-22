#include "DepthEvidenceConfidence.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace xjw::mvs
{
namespace
{

bool compatible(const cv::Mat &matrix, const cv::Size &size, int type)
{
    return !matrix.empty() && matrix.size() == size && matrix.type() == type;
}

float unit(float value)
{
    return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
}

float geometricMean(float first, float second)
{
    return std::sqrt(std::max(0.0f, first) * std::max(0.0f, second));
}

} // namespace

DepthEvidenceConfidenceResult buildDepthEvidenceConfidence(
    const cv::Mat &depth,
    const cv::Mat &photometricConfidence,
    const cv::Mat &geometrySupportCount,
    const cv::Mat &inverseDepthRelativeSpread,
    const cv::Mat &adaptiveSupportWeight,
    const cv::Mat &adaptiveEffectiveViewCount,
    const cv::Mat &adaptiveConflictRatio,
    const DepthGeometryHypothesisRerankMaps *rerankMaps,
    const DepthEvidenceConfidenceOptions &options)
{
    DepthEvidenceConfidenceResult result;
    if (!compatible(depth, depth.size(), CV_32FC1) ||
        !compatible(photometricConfidence, depth.size(), CV_32FC1))
    {
        return result;
    }

    result.photometric = photometricConfidence.clone();
    result.geometric = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.0f));
    result.combined = photometricConfidence.clone();
    const bool has_discrete = compatible(
        geometrySupportCount, depth.size(), CV_16UC1) &&
        compatible(inverseDepthRelativeSpread, depth.size(), CV_32FC1);
    const bool has_adaptive = has_discrete &&
        compatible(adaptiveSupportWeight, depth.size(), CV_32FC1) &&
        compatible(adaptiveEffectiveViewCount, depth.size(), CV_32FC1) &&
        compatible(adaptiveConflictRatio, depth.size(), CV_32FC1);
    const bool has_rerank = rerankMaps && rerankMaps->compatible(depth.size());
    if (!has_discrete && !has_rerank)
    {
        return result;
    }

    const int minimum_observations = std::max(
        2, options.minimumGeometryObservationCount);
    const float maximum_spread = std::max(
        1.0e-6f, options.maximumInverseDepthRelativeSpread);
    const float photometric_weight = std::clamp(
        options.photometricWeight, 0.0f, 1.0f);
    const float strong_threshold = unit(options.strongGeometryConfidence);
    double photometric_sum = 0.0;
    double geometric_sum = 0.0;
    double combined_sum = 0.0;
    double corrected_geometric_sum = 0.0;

    for (int row = 0; row < depth.rows; ++row)
    {
        const float *depth_row = depth.ptr<float>(row);
        const float *photometric_row = photometricConfidence.ptr<float>(row);
        float *geometric_row = result.geometric.ptr<float>(row);
        float *combined_row = result.combined.ptr<float>(row);
        const std::uint16_t *support_row = has_discrete
            ? geometrySupportCount.ptr<std::uint16_t>(row) : nullptr;
        const float *spread_row = has_discrete
            ? inverseDepthRelativeSpread.ptr<float>(row) : nullptr;
        const float *adaptive_weight_row = has_adaptive
            ? adaptiveSupportWeight.ptr<float>(row) : nullptr;
        const float *adaptive_views_row = has_adaptive
            ? adaptiveEffectiveViewCount.ptr<float>(row) : nullptr;
        const float *adaptive_conflict_row = has_adaptive
            ? adaptiveConflictRatio.ptr<float>(row) : nullptr;
        const float *rerank_advantage_row = has_rerank
            ? rerankMaps->costAdvantage.ptr<float>(row) : nullptr;
        const float *rerank_weight_row = has_rerank
            ? rerankMaps->effectiveSourceWeight.ptr<float>(row) : nullptr;
        const float *weakest_confidence_row = has_rerank
            ? rerankMaps->weakestSourceConfidence.ptr<float>(row) : nullptr;
        const std::uint8_t *rerank_sources_row = has_rerank
            ? rerankMaps->supportingSourceCount.ptr<std::uint8_t>(row) : nullptr;
        const std::uint8_t *rerank_sectors_row = has_rerank
            ? rerankMaps->baselineSectorCount.ptr<std::uint8_t>(row) : nullptr;
        const std::uint8_t *rerank_action_row = has_rerank
            ? rerankMaps->decisionAction.ptr<std::uint8_t>(row) : nullptr;

        for (int column = 0; column < depth.cols; ++column)
        {
            if (!std::isfinite(depth_row[column]) || depth_row[column] <= 0.0f)
            {
                combined_row[column] = 0.0f;
                continue;
            }
            ++result.summary.validPixelCount;
            const float photometric = unit(photometric_row[column]);
            float geometric = 0.0f;
            if (has_discrete && support_row[column] >= minimum_observations &&
                std::isfinite(spread_row[column]) && spread_row[column] >= 0.0f)
            {
                const float support_score = unit(
                    static_cast<float>(support_row[column] - 1) / 3.0f);
                const float spread_score = std::exp(
                    -spread_row[column] / maximum_spread);
                geometric = geometricMean(support_score, spread_score);
                if (has_adaptive)
                {
                    const float adaptive_score = std::sqrt(
                        geometricMean(
                            unit(adaptive_weight_row[column] / 0.75f),
                            unit(adaptive_views_row[column] / 2.5f)) *
                        unit(1.0f - adaptive_conflict_row[column]));
                    geometric = geometricMean(geometric, adaptive_score);
                }
            }

            bool corrected = false;
            if (has_rerank && rerank_sources_row[column] >= 3 &&
                rerank_sectors_row[column] >= 2)
            {
                const float advantage_score = unit(
                    rerank_advantage_row[column] / 0.20f);
                const float weight_score = unit(
                    rerank_weight_row[column] / 2.25f);
                const float rerank_score = std::cbrt(
                    std::max(0.0f, advantage_score) *
                    std::max(0.0f, weight_score) *
                    unit(weakest_confidence_row[column]));
                geometric = std::max(geometric, rerank_score);
                corrected = rerank_action_row[column] != static_cast<std::uint8_t>(
                    DepthGeometryHypothesisAction::None);
            }
            geometric = unit(geometric);
            geometric_row[column] = geometric;
            const float combined = geometric > 0.0f
                ? unit(photometric_weight * photometric +
                       (1.0f - photometric_weight) * geometric)
                : photometric;
            combined_row[column] = combined;
            photometric_sum += photometric;
            geometric_sum += geometric;
            combined_sum += combined;
            if (geometric > 0.0f)
            {
                ++result.summary.geometryObservedPixelCount;
            }
            if (geometric >= strong_threshold)
            {
                ++result.summary.strongGeometryPixelCount;
            }
            if (corrected)
            {
                ++result.summary.correctedPixelCount;
                corrected_geometric_sum += geometric;
            }
        }
    }

    if (result.summary.validPixelCount > 0)
    {
        const double divisor = static_cast<double>(result.summary.validPixelCount);
        result.summary.meanPhotometricConfidence = static_cast<float>(
            photometric_sum / divisor);
        result.summary.meanGeometricConfidence = static_cast<float>(
            geometric_sum / divisor);
        result.summary.meanCombinedConfidence = static_cast<float>(
            combined_sum / divisor);
        result.summary.strongGeometryCoverage = static_cast<float>(
            static_cast<double>(result.summary.strongGeometryPixelCount) /
            divisor);
    }
    if (result.summary.correctedPixelCount > 0)
    {
        result.summary.correctedMeanGeometricConfidence = static_cast<float>(
            corrected_geometric_sum /
            static_cast<double>(result.summary.correctedPixelCount));
    }
    result.summary.available = result.summary.geometryObservedPixelCount > 0;
    return result;
}

QJsonObject depthEvidenceConfidenceSummaryToJson(
    const DepthEvidenceConfidenceSummary &summary)
{
    return QJsonObject{
        {QStringLiteral("available"), summary.available},
        {QStringLiteral("valid_pixel_count"), summary.validPixelCount},
        {QStringLiteral("geometry_observed_pixel_count"),
         summary.geometryObservedPixelCount},
        {QStringLiteral("strong_geometry_pixel_count"),
         summary.strongGeometryPixelCount},
        {QStringLiteral("corrected_pixel_count"), summary.correctedPixelCount},
        {QStringLiteral("mean_photometric_confidence"),
         summary.meanPhotometricConfidence},
        {QStringLiteral("mean_geometric_confidence"),
         summary.meanGeometricConfidence},
        {QStringLiteral("mean_combined_confidence"),
         summary.meanCombinedConfidence},
        {QStringLiteral("strong_geometry_coverage"),
         summary.strongGeometryCoverage},
        {QStringLiteral("corrected_mean_geometric_confidence"),
         summary.correctedMeanGeometricConfidence},
        {QStringLiteral("schema_version"), 1}};
}

} // namespace xjw::mvs
