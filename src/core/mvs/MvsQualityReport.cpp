#include "MvsQualityReport.h"

#include <QJsonObject>

#include <algorithm>
#include <cmath>
#include <vector>

namespace xjw
{
namespace mvs
{

namespace
{

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

} // namespace

DepthMapQualityMetrics analyzeDepthMapQuality(const cv::Mat &depthMap,
                                              const cv::Mat &confidenceMap,
                                              int sourceViewCount)
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
    metrics.localDepthOutlierCount = countLocalDepthOutliers(depthMap);
    metrics.localDepthOutlierRatio = metrics.validPixelCount > 0
        ? static_cast<float>(metrics.localDepthOutlierCount) / static_cast<float>(metrics.validPixelCount)
        : 0.0f;
    metrics.hasLocalDepthOutliers = metrics.localDepthOutlierCount > 0;

    const bool hasConfidence = !confidenceMap.empty()
        && confidenceMap.size() == depthMap.size()
        && confidenceMap.type() == CV_32F;
    if (!hasConfidence || metrics.validPixelCount <= 0)
    {
        metrics.recommendedFusionConfidence = 0.65f;
        return metrics;
    }

    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(metrics.validPixelCount));
    double sum = 0.0;
    for (int row = 0; row < depthMap.rows; ++row)
    {
        const float *depthRow = depthMap.ptr<float>(row);
        const float *confRow = confidenceMap.ptr<float>(row);
        for (int col = 0; col < depthMap.cols; ++col)
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
    object.insert(QStringLiteral("low_confidence_full_coverage"), metrics.lowConfidenceFullCoverage);
    object.insert(QStringLiteral("local_depth_outlier_count"), metrics.localDepthOutlierCount);
    object.insert(QStringLiteral("local_depth_outlier_ratio"), rounded(metrics.localDepthOutlierRatio));
    object.insert(QStringLiteral("has_local_depth_outliers"), metrics.hasLocalDepthOutliers);
    object.insert(QStringLiteral("recommended_fusion_confidence"), rounded(metrics.recommendedFusionConfidence));
    return object;
}

} // namespace mvs
} // namespace xjw
