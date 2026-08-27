#include "MvsQualityReport.h"

#include <QJsonArray>
#include <QJsonObject>

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace xjw
{
namespace mvs
{

namespace
{

constexpr int kMaxQualitySamplePixels = 1024 * 1024;

cv::Mat boundedQualitySample(const cv::Mat &matrix)
{
    if (matrix.empty() || matrix.total() <= kMaxQualitySamplePixels)
    {
        return matrix;
    }

    const double scale = std::sqrt(
        static_cast<double>(kMaxQualitySamplePixels) /
        static_cast<double>(matrix.total()));
    cv::Mat sampled;
    cv::resize(matrix,
               sampled,
               cv::Size(std::max(1, static_cast<int>(std::round(matrix.cols * scale))),
                        std::max(1, static_cast<int>(std::round(matrix.rows * scale)))),
               0.0,
               0.0,
               cv::INTER_NEAREST);
    return sampled;
}

float percentile(std::vector<float> values, float ratio)
{
    if (values.empty())
    {
        return 0.0f;
    }
    ratio = std::clamp(ratio, 0.0f, 1.0f);
    const std::size_t index = static_cast<std::size_t>(
        std::round(ratio * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

float medianValue(std::vector<float> values)
{
    if (values.empty())
    {
        return 0.0f;
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    if ((values.size() % 2) == 1)
    {
        return values[mid];
    }

    const float upper = values[mid];
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1), values.end());
    return 0.5f * (values[mid - 1] + upper);
}

int countLocalDepthOutliers(const cv::Mat &depthMap)
{
    int outliers = 0;
    std::vector<float> neighbors;
    neighbors.reserve(8);

    for (int row = 0; row < depthMap.rows; ++row)
    {
        const float *depthRow = depthMap.ptr<float>(row);
        for (int col = 0; col < depthMap.cols; ++col)
        {
            const float depth = depthRow[col];
            if (depth <= 0.0f || !std::isfinite(depth))
            {
                continue;
            }

            neighbors.clear();
            for (int dy = -1; dy <= 1; ++dy)
            {
                const int nr = row + dy;
                if (nr < 0 || nr >= depthMap.rows)
                {
                    continue;
                }
                const float *neighborRow = depthMap.ptr<float>(nr);
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0)
                    {
                        continue;
                    }
                    const int nc = col + dx;
                    if (nc < 0 || nc >= depthMap.cols)
                    {
                        continue;
                    }

                    const float neighborDepth = neighborRow[nc];
                    if (neighborDepth > 0.0f && std::isfinite(neighborDepth))
                    {
                        neighbors.push_back(neighborDepth);
                    }
                }
            }

            if (neighbors.size() < 5)
            {
                continue;
            }

            const float median = medianValue(neighbors);
            if (median <= 0.0f)
            {
                continue;
            }

            const float absoluteDiff = std::fabs(depth - median);
            const float threshold = std::max(0.5f, std::fabs(median) * 0.25f);
            if (absoluteDiff > threshold)
            {
                ++outliers;
            }
        }
    }

    return outliers;
}

float largestValidComponentRatio(const cv::Mat &depthMap, int validPixelCount)
{
    if (validPixelCount <= 0)
    {
        return 0.0f;
    }

    cv::Mat valid_mask;
    cv::compare(depthMap, 0.0f, valid_mask, cv::CMP_GT);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        valid_mask,
        labels,
        stats,
        centroids,
        8,
        CV_32S);

    int largest_area = 0;
    for (int component = 1; component < component_count; ++component)
    {
        largest_area = std::max(largest_area, stats.at<int>(component, cv::CC_STAT_AREA));
    }
    return static_cast<float>(largest_area) / static_cast<float>(validPixelCount);
}

float searchBoundaryRatio(const cv::Mat &depthMap,
                          int validPixelCount,
                          float depthNear,
                          float depthFar)
{
    const float depth_range = depthFar - depthNear;
    if (validPixelCount <= 0 || depthNear <= 0.0f || depth_range <= 0.0f)
    {
        return 0.0f;
    }

    const float boundary_width = std::max(1.0e-5f, depth_range * 0.02f);
    int boundary_count = 0;
    for (int row = 0; row < depthMap.rows; ++row)
    {
        const float *depth_row = depthMap.ptr<float>(row);
        for (int column = 0; column < depthMap.cols; ++column)
        {
            const float depth = depth_row[column];
            if (depth <= 0.0f || !std::isfinite(depth))
            {
                continue;
            }
            if (depth <= depthNear + boundary_width || depth >= depthFar - boundary_width)
            {
                ++boundary_count;
            }
        }
    }
    return static_cast<float>(boundary_count) / static_cast<float>(validPixelCount);
}

} // namespace

float measureDepthDiscontinuityRatio(const cv::Mat &depthMap,
                                     float relativeThreshold)
{
    if (depthMap.empty() || depthMap.type() != CV_32F)
    {
        return 0.0f;
    }

    const cv::Mat sampled_depth = boundedQualitySample(depthMap);
    const float threshold = std::max(0.0f, relativeThreshold);
    std::uint64_t comparable_pairs = 0;
    std::uint64_t discontinuous_pairs = 0;
    const auto compare_pair = [&](float first, float second)
    {
        if (first <= 0.0f || second <= 0.0f ||
            !std::isfinite(first) || !std::isfinite(second))
        {
            return;
        }

        ++comparable_pairs;
        const float scale = std::max(1.0e-6f, std::min(std::fabs(first), std::fabs(second)));
        if (std::fabs(first - second) / scale > threshold)
        {
            ++discontinuous_pairs;
        }
    };

    for (int row = 0; row < sampled_depth.rows; ++row)
    {
        const float *depth_row = sampled_depth.ptr<float>(row);
        const float *next_row = row + 1 < sampled_depth.rows
            ? sampled_depth.ptr<float>(row + 1)
            : nullptr;
        for (int column = 0; column < sampled_depth.cols; ++column)
        {
            if (column + 1 < sampled_depth.cols)
            {
                compare_pair(depth_row[column], depth_row[column + 1]);
            }
            if (next_row)
            {
                compare_pair(depth_row[column], next_row[column]);
            }
        }
    }

    return comparable_pairs > 0
        ? static_cast<float>(discontinuous_pairs) / static_cast<float>(comparable_pairs)
        : 0.0f;
}

DepthMapQualityMetrics analyzeDepthMapQuality(const cv::Mat &depthMap,
                                              const cv::Mat &confidenceMap,
                                              int sourceViewCount,
                                              float depthNear,
                                              float depthFar)
{
    DepthMapQualityMetrics metrics;
    metrics.sourceViewCount = std::max(0, sourceViewCount);
    if (depthMap.empty() || depthMap.type() != CV_32F)
    {
        return metrics;
    }

    metrics.width = depthMap.cols;
    metrics.height = depthMap.rows;
    const int totalPixels = std::max(1, depthMap.rows * depthMap.cols);
    metrics.validPixelCount = cv::countNonZero(depthMap > 0.0f);
    metrics.validCoverage = static_cast<float>(metrics.validPixelCount) / static_cast<float>(totalPixels);
    const cv::Mat sampled_depth = boundedQualitySample(depthMap);
    const int sampled_valid_count = cv::countNonZero(sampled_depth > 0.0f);
    metrics.largestComponentRatio = largestValidComponentRatio(sampled_depth,
                                                               sampled_valid_count);
    metrics.depthAtSearchBoundaryRatio = searchBoundaryRatio(
        sampled_depth,
        sampled_valid_count,
        depthNear,
        depthFar);
    metrics.depthDiscontinuityRatio = measureDepthDiscontinuityRatio(sampled_depth);
    const int sampled_outlier_count = countLocalDepthOutliers(sampled_depth);
    metrics.localDepthOutlierRatio = sampled_valid_count > 0
        ? static_cast<float>(sampled_outlier_count) / static_cast<float>(sampled_valid_count)
        : 0.0f;
    metrics.localDepthOutlierCount = static_cast<int>(std::llround(
        static_cast<double>(metrics.localDepthOutlierRatio) *
        static_cast<double>(metrics.validPixelCount)));
    metrics.hasLocalDepthOutliers = metrics.localDepthOutlierCount > 0;

    const bool hasConfidence = !confidenceMap.empty()
        && confidenceMap.size() == depthMap.size()
        && confidenceMap.type() == CV_32F;
    if (!hasConfidence || metrics.validPixelCount <= 0)
    {
        metrics.recommendedFusionConfidence = 0.65f;
        return metrics;
    }

    const cv::Mat sampled_confidence = boundedQualitySample(confidenceMap);
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(sampled_valid_count));
    double sum = 0.0;
    for (int row = 0; row < sampled_depth.rows; ++row)
    {
        const float *depthRow = sampled_depth.ptr<float>(row);
        const float *confRow = sampled_confidence.ptr<float>(row);
        for (int col = 0; col < sampled_depth.cols; ++col)
        {
            if (depthRow[col] <= 0.0f)
            {
                continue;
            }
            const float conf = std::clamp(confRow[col], 0.0f, 1.0f);
            values.push_back(conf);
            sum += static_cast<double>(conf);
        }
    }

    if (!values.empty())
    {
        metrics.meanConfidence = static_cast<float>(sum / static_cast<double>(values.size()));
        metrics.p50Confidence = percentile(values, 0.50f);
        metrics.p75Confidence = percentile(values, 0.75f);
    }

    metrics.lowConfidenceFullCoverage =
        metrics.sourceViewCount >= 3 &&
        metrics.validCoverage >= 0.95f &&
        metrics.meanConfidence > 0.0f &&
        metrics.meanConfidence < 0.65f;

    metrics.recommendedFusionConfidence = metrics.lowConfidenceFullCoverage
        ? std::max(0.65f, metrics.p75Confidence)
        : std::max(0.25f, metrics.p50Confidence);
    metrics.recommendedFusionConfidence = std::clamp(metrics.recommendedFusionConfidence, 0.0f, 0.85f);
    return metrics;
}

QJsonObject depthMapQualityMetricsToJson(const DepthMapQualityMetrics &metrics)
{
    auto rounded = [](float value) {
        return static_cast<double>(std::round(static_cast<double>(value) * 1000000.0) / 1000000.0);
    };

    QJsonObject object;
    object.insert(QStringLiteral("width"), metrics.width);
    object.insert(QStringLiteral("height"), metrics.height);
    object.insert(QStringLiteral("valid_pixel_count"), metrics.validPixelCount);
    object.insert(QStringLiteral("source_view_count"), metrics.sourceViewCount);
    object.insert(QStringLiteral("valid_coverage"), rounded(metrics.validCoverage));
    object.insert(QStringLiteral("mean_confidence"), rounded(metrics.meanConfidence));
    object.insert(QStringLiteral("p50_confidence"), rounded(metrics.p50Confidence));
    object.insert(QStringLiteral("p75_confidence"), rounded(metrics.p75Confidence));
    object.insert(QStringLiteral("largest_component_ratio"), rounded(metrics.largestComponentRatio));
    object.insert(QStringLiteral("depth_at_search_boundary_ratio"),
                  rounded(metrics.depthAtSearchBoundaryRatio));
    object.insert(QStringLiteral("depth_discontinuity_ratio"),
                  rounded(metrics.depthDiscontinuityRatio));
    object.insert(QStringLiteral("low_confidence_full_coverage"), metrics.lowConfidenceFullCoverage);
    object.insert(QStringLiteral("local_depth_outlier_count"), metrics.localDepthOutlierCount);
    object.insert(QStringLiteral("local_depth_outlier_ratio"), rounded(metrics.localDepthOutlierRatio));
    object.insert(QStringLiteral("has_local_depth_outliers"), metrics.hasLocalDepthOutliers);
    object.insert(QStringLiteral("recommended_fusion_confidence"), rounded(metrics.recommendedFusionConfidence));
    return object;
}

QJsonObject depthFrameQualityDecisionToJson(const DepthFrameQualityDecision &decision)
{
    QJsonObject object;
    object.insert(QStringLiteral("acceptance"),
                  QString::fromLatin1(depthFrameAcceptanceId(decision.acceptance)));
    object.insert(QStringLiteral("calibrated_confidence"), decision.calibratedConfidence);
    QJsonObject confidence_components;
    confidence_components.insert(
        QStringLiteral("photometric"), decision.confidenceComponents.photometric);
    confidence_components.insert(
        QStringLiteral("source_support"), decision.confidenceComponents.support);
    confidence_components.insert(
        QStringLiteral("uniqueness"), decision.confidenceComponents.uniqueness);
    confidence_components.insert(
        QStringLiteral("multiview_geometry"), decision.confidenceComponents.geometry);
    confidence_components.insert(
        QStringLiteral("surface_coherence"), decision.confidenceComponents.texture);
    confidence_components.insert(
        QStringLiteral("absolute_geometry"),
        decision.confidenceComponents.absoluteGeometry);
    object.insert(QStringLiteral("confidence_components"), confidence_components);
    object.insert(QStringLiteral("min_component_area"),
                  decision.filterSettings.minComponentArea);
    object.insert(QStringLiteral("local_depth_outlier_rel_threshold"),
                  decision.filterSettings.localDepthOutlierRelThreshold);
    object.insert(QStringLiteral("min_consistent_views"),
                  decision.filterSettings.minConsistentViews);

    const SparseDepthResidualSummary &sparse_residual =
        decision.sparseDepthResidual;
    QJsonObject sparse_residual_object;
    sparse_residual_object.insert(QStringLiteral("available"),
                                  sparse_residual.available);
    sparse_residual_object.insert(QStringLiteral("projected_sample_count"),
                                  sparse_residual.projectedSampleCount);
    sparse_residual_object.insert(QStringLiteral("valid_sample_count"),
                                  sparse_residual.validSampleCount);
    const double valid_sample_ratio = sparse_residual.projectedSampleCount > 0
        ? static_cast<double>(sparse_residual.validSampleCount) /
              static_cast<double>(sparse_residual.projectedSampleCount)
        : 0.0;
    sparse_residual_object.insert(QStringLiteral("valid_sample_ratio"),
                                  valid_sample_ratio);
    sparse_residual_object.insert(QStringLiteral("median_absolute_log_error"), sparse_residual.medianAbsoluteLogError);
    sparse_residual_object.insert(QStringLiteral("minimum_sample_count"), kSparseDepthResidualMinimumSampleCount);
    sparse_residual_object.insert(QStringLiteral("validation_only_threshold"),
                                  decision.sparseDepthResidualValidationThreshold);
    sparse_residual_object.insert(QStringLiteral("rejection_threshold"), kSparseDepthResidualRejectionThreshold);
    sparse_residual_object.insert(QStringLiteral("neighborhood_radius_pixels"),
                                  sparse_residual.neighborhoodRadiusPixels);
    object.insert(QStringLiteral("sparse_absolute_depth_residual"), sparse_residual_object);

    QJsonArray reasons;
    for (const std::string &reason : decision.reasons)
    {
        reasons.append(QString::fromStdString(reason));
    }
    object.insert(QStringLiteral("reasons"), reasons);
    return object;
}

} // namespace mvs
} // namespace xjw
