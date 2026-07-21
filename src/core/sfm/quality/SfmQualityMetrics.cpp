#include "quality/SfmQualityMetrics.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace xjw
{

namespace
{

double percentile(std::vector<double> values, double ratio)
{
    if (values.empty())
    {
        return 0.0;
    }

    ratio = std::clamp(ratio, 0.0, 1.0);
    const auto index = static_cast<std::size_t>(
        std::round(ratio * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

SfmNumericSummary summarize(std::vector<double> values)
{
    SfmNumericSummary summary;
    summary.count = static_cast<int>(values.size());
    if (values.empty())
    {
        return summary;
    }

    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    summary.min = *minIt;
    summary.max = *maxIt;

    double sum = 0.0;
    for (double value : values)
    {
        sum += value;
    }
    summary.mean = sum / static_cast<double>(values.size());
    summary.p50 = percentile(values, 0.50);
    summary.p84 = percentile(values, 0.84);
    summary.p95 = percentile(values, 0.95);
    return summary;
}

double pointGridCoverage(const SfmQualityPoint &point, const SfmQualityMetricsOptions &options)
{
    if (options.imageWidth <= 0.0 || options.imageHeight <= 0.0 ||
        options.coverageGridColumns <= 0 || options.coverageGridRows <= 0)
    {
        return 0.0;
    }

    std::set<int> occupiedCells;
    for (const SfmQualityObservation &observation : point.observations)
    {
        if (observation.x < 0.0 || observation.y < 0.0 ||
            observation.x >= options.imageWidth || observation.y >= options.imageHeight)
        {
            continue;
        }

        const int col = std::clamp(static_cast<int>(observation.x / options.imageWidth *
                                                    static_cast<double>(options.coverageGridColumns)),
                                   0,
                                   options.coverageGridColumns - 1);
        const int row = std::clamp(static_cast<int>(observation.y / options.imageHeight *
                                                    static_cast<double>(options.coverageGridRows)),
                                   0,
                                   options.coverageGridRows - 1);
        occupiedCells.insert(row * options.coverageGridColumns + col);
    }

    const int totalCells = options.coverageGridColumns * options.coverageGridRows;
    return totalCells > 0
        ? static_cast<double>(occupiedCells.size()) / static_cast<double>(totalCells)
        : 0.0;
}

} // namespace

SfmQualityMetrics computeSfmQualityMetrics(const std::vector<SfmQualityPoint> &points,
                                           const SfmQualityMetricsOptions &options)
{
    SfmQualityMetrics metrics;
    metrics.totalImageCount = std::max(0, options.totalImageCount);
    metrics.registeredImageCount = std::max(0, options.registeredImageCount);
    metrics.pointCount = static_cast<int>(points.size());
    metrics.coverageGridColumns = std::max(0, options.coverageGridColumns);
    metrics.coverageGridRows = std::max(0, options.coverageGridRows);

    std::vector<double> trackLengths;
    std::vector<double> reprojectionErrors;
    std::vector<double> triangulationAngles;
    std::vector<double> coverages;
    trackLengths.reserve(points.size());
    reprojectionErrors.reserve(points.size());
    triangulationAngles.reserve(points.size());
    coverages.reserve(points.size());

    for (const SfmQualityPoint &point : points)
    {
        const int trackLength = std::max(0, point.trackLength);
        trackLengths.push_back(static_cast<double>(trackLength));
        reprojectionErrors.push_back(point.reprojectionErrorPx);
        triangulationAngles.push_back(point.triangulationAngleDeg);
        metrics.trackLengthHistogram[trackLength] += 1;

        if (trackLength == 2)
        {
            ++metrics.twoViewTrackCount;
        }
        else if (trackLength > 2)
        {
            ++metrics.multiViewTrackCount;
        }

        if (trackLength < options.minTrackLength)
        {
            ++metrics.weakTrackCount;
        }
        if (point.triangulationAngleDeg < options.minTriangulationAngleDeg)
        {
            ++metrics.weakTriangulationAngleCount;
        }
        if (point.reprojectionErrorPx > options.maxReprojectionErrorPx)
        {
            ++metrics.highReprojectionErrorCount;
        }

        coverages.push_back(pointGridCoverage(point, options));
    }

    metrics.trackLength = summarize(trackLengths);
    metrics.reprojectionError = summarize(reprojectionErrors);
    metrics.triangulationAngle = summarize(triangulationAngles);
    metrics.observationGridCoverageMean = summarize(coverages).mean;

    const double registeredRatio = metrics.totalImageCount > 0
        ? static_cast<double>(metrics.registeredImageCount) / static_cast<double>(metrics.totalImageCount)
        : 0.0;
    const double twoViewRatio = metrics.pointCount > 0
        ? static_cast<double>(metrics.twoViewTrackCount) / static_cast<double>(metrics.pointCount)
        : 0.0;
    const double highErrorRatio = metrics.pointCount > 0
        ? static_cast<double>(metrics.highReprojectionErrorCount) / static_cast<double>(metrics.pointCount)
        : 0.0;
    const double weakAngleRatio = metrics.pointCount > 0
        ? static_cast<double>(metrics.weakTriangulationAngleCount) / static_cast<double>(metrics.pointCount)
        : 0.0;

    if (metrics.totalImageCount > 0 && registeredRatio < options.minRegisteredImageRatioForMvs)
    {
        metrics.qualityWarnings.push_back("low_registered_image_coverage");
    }
    if (metrics.pointCount > 0 && twoViewRatio > options.maxTwoViewTrackRatioForMvs)
    {
        metrics.qualityWarnings.push_back("too_many_two_view_tracks");
    }
    else if (metrics.pointCount > 0 && twoViewRatio > options.warnTwoViewTrackRatioForMvs)
    {
        metrics.qualityAdvisories.push_back("high_two_view_track_ratio");
    }
    if (metrics.pointCount > 0 && highErrorRatio > options.maxHighReprojectionErrorRatioForMvs)
    {
        metrics.qualityWarnings.push_back("high_reprojection_error");
    }
    if (metrics.pointCount > 0 && weakAngleRatio > options.maxWeakTriangulationAngleRatioForMvs)
    {
        metrics.qualityWarnings.push_back("weak_triangulation_angle");
    }
    if (metrics.pointCount > 0 &&
        options.minObservationGridCoverageMeanForMvs > 0.0 &&
        metrics.observationGridCoverageMean < options.minObservationGridCoverageMeanForMvs)
    {
        metrics.qualityWarnings.push_back("poor_observation_spatial_coverage");
    }

    metrics.acceptableForMvs = metrics.qualityWarnings.empty();
    metrics.qualityStatus = !metrics.acceptableForMvs
        ? "blocked"
        : (metrics.qualityAdvisories.empty() ? "ok" : "warn");
    return metrics;
}

} // namespace xjw
