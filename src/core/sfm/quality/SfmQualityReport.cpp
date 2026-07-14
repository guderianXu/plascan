#include "quality/SfmQualityReport.h"

#include <QJsonObject>

#include <algorithm>
#include <cmath>
#include <set>

namespace xjw
{

namespace
{

double roundMetric(double value)
{
    return std::round(value * 1000000.0) / 1000000.0;
}

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

double pointGridCoverage(const SfmQualityPoint &point, const SfmQualityReportOptions &options)
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
    return totalCells > 0 ? static_cast<double>(occupiedCells.size()) / static_cast<double>(totalCells) : 0.0;
}

} // namespace

QJsonObject SfmNumericSummary::toJson(const QString &unit) const
{
    QJsonObject object;
    object.insert(QStringLiteral("count"), count);
    object.insert(QStringLiteral("min"), roundMetric(min));
    object.insert(QStringLiteral("max"), roundMetric(max));
    object.insert(QStringLiteral("mean"), roundMetric(mean));
    object.insert(QStringLiteral("p50"), roundMetric(p50));
    object.insert(QStringLiteral("p84"), roundMetric(p84));
    object.insert(QStringLiteral("p95"), roundMetric(p95));
    object.insert(QStringLiteral("unit"), unit);
    return object;
}

QJsonObject SfmQualityReport::toJson() const
{
    QJsonArray warnings;
    for (const QString &warning : qualityWarnings)
    {
        warnings.append(warning);
    }

    QJsonObject histogram;
    for (const auto &[length, count] : trackLengthHistogram)
    {
        histogram.insert(QString::number(length), count);
    }

    QJsonObject coverage;
    coverage.insert(QStringLiteral("mean"), roundMetric(observationGridCoverageMean));
    coverage.insert(QStringLiteral("grid_columns"), coverageGridColumns);
    coverage.insert(QStringLiteral("grid_rows"), coverageGridRows);

    QJsonObject qualityGate;
    qualityGate.insert(QStringLiteral("acceptable_for_mvs"), acceptableForMvs);
    qualityGate.insert(QStringLiteral("status"), qualityStatus);
    qualityGate.insert(QStringLiteral("advisories"), QJsonArray::fromStringList(qualityAdvisories));
    qualityGate.insert(QStringLiteral("warnings"), warnings);

    QJsonObject object;
    object.insert(QStringLiteral("registered_image_count"), registeredImageCount);
    object.insert(QStringLiteral("total_image_count"), totalImageCount);
    object.insert(QStringLiteral("point_count"), pointCount);
    object.insert(QStringLiteral("two_view_track_count"), twoViewTrackCount);
    object.insert(QStringLiteral("multi_view_track_count"), multiViewTrackCount);
    object.insert(QStringLiteral("weak_track_count"), weakTrackCount);
    object.insert(QStringLiteral("weak_triangulation_angle_count"), weakTriangulationAngleCount);
    object.insert(QStringLiteral("high_reprojection_error_count"), highReprojectionErrorCount);
    object.insert(QStringLiteral("track_length"), trackLength.toJson(QStringLiteral("observations")));
    object.insert(QStringLiteral("track_length_histogram"), histogram);
    object.insert(QStringLiteral("reprojection_error"), reprojectionError.toJson(QStringLiteral("px")));
    object.insert(QStringLiteral("triangulation_angle"), triangulationAngle.toJson(QStringLiteral("deg")));
    object.insert(QStringLiteral("observation_grid_coverage"), coverage);
    object.insert(QStringLiteral("quality_gate"), qualityGate);
    return object;
}

SfmQualityReport analyzeSfmQuality(const std::vector<SfmQualityPoint> &points,
                                   const SfmQualityReportOptions &options)
{
    SfmQualityReport report;
    report.totalImageCount = std::max(0, options.totalImageCount);
    report.registeredImageCount = std::max(0, options.registeredImageCount);
    report.pointCount = static_cast<int>(points.size());
    report.coverageGridColumns = std::max(0, options.coverageGridColumns);
    report.coverageGridRows = std::max(0, options.coverageGridRows);

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
        report.trackLengthHistogram[trackLength] += 1;

        if (trackLength == 2)
        {
            ++report.twoViewTrackCount;
        }
        else if (trackLength > 2)
        {
            ++report.multiViewTrackCount;
        }

        if (trackLength < options.minTrackLength)
        {
            ++report.weakTrackCount;
        }
        if (point.triangulationAngleDeg < options.minTriangulationAngleDeg)
        {
            ++report.weakTriangulationAngleCount;
        }
        if (point.reprojectionErrorPx > options.maxReprojectionErrorPx)
        {
            ++report.highReprojectionErrorCount;
        }

        coverages.push_back(pointGridCoverage(point, options));
    }

    report.trackLength = summarize(trackLengths);
    report.reprojectionError = summarize(reprojectionErrors);
    report.triangulationAngle = summarize(triangulationAngles);
    report.observationGridCoverageMean = summarize(coverages).mean;

    const double registeredRatio = report.totalImageCount > 0
        ? static_cast<double>(report.registeredImageCount) / static_cast<double>(report.totalImageCount)
        : 0.0;
    const double twoViewRatio = report.pointCount > 0
        ? static_cast<double>(report.twoViewTrackCount) / static_cast<double>(report.pointCount)
        : 0.0;
    const double highErrorRatio = report.pointCount > 0
        ? static_cast<double>(report.highReprojectionErrorCount) / static_cast<double>(report.pointCount)
        : 0.0;
    const double weakAngleRatio = report.pointCount > 0
        ? static_cast<double>(report.weakTriangulationAngleCount) / static_cast<double>(report.pointCount)
        : 0.0;

    if (report.totalImageCount > 0 && registeredRatio < options.minRegisteredImageRatioForMvs)
    {
        report.qualityWarnings.append(QStringLiteral("low_registered_image_coverage"));
    }
    if (report.pointCount > 0 && twoViewRatio > options.maxTwoViewTrackRatioForMvs)
    {
        report.qualityWarnings.append(QStringLiteral("too_many_two_view_tracks"));
    }
    else if (report.pointCount > 0 && twoViewRatio > options.warnTwoViewTrackRatioForMvs)
    {
        report.qualityAdvisories.append(QStringLiteral("high_two_view_track_ratio"));
    }
    if (report.pointCount > 0 && highErrorRatio > options.maxHighReprojectionErrorRatioForMvs)
    {
        report.qualityWarnings.append(QStringLiteral("high_reprojection_error"));
    }
    if (report.pointCount > 0 && weakAngleRatio > options.maxWeakTriangulationAngleRatioForMvs)
    {
        report.qualityWarnings.append(QStringLiteral("weak_triangulation_angle"));
    }
    if (report.pointCount > 0
        && options.minObservationGridCoverageMeanForMvs > 0.0
        && report.observationGridCoverageMean < options.minObservationGridCoverageMeanForMvs)
    {
        report.qualityWarnings.append(QStringLiteral("poor_observation_spatial_coverage"));
    }

    report.acceptableForMvs = report.qualityWarnings.isEmpty();
    report.qualityStatus = !report.acceptableForMvs
        ? QStringLiteral("blocked")
        : (report.qualityAdvisories.isEmpty() ? QStringLiteral("ok") : QStringLiteral("warn"));
    return report;
}

} // namespace xjw
