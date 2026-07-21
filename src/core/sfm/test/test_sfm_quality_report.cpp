#include <gtest/gtest.h>

#include "quality/SfmQualityMetrics.h"
#include "project/SfmQualityJsonSerializer.h"

#include <QJsonArray>
#include <QJsonObject>

#include <algorithm>
#include <string>

namespace
{

bool containsString(const std::vector<std::string> &values, const std::string &expected)
{
    return std::find(values.begin(), values.end(), expected) != values.end();
}


xjw::SfmQualityPoint makeQualityPoint(int trackLength,
                                      double reprojectionErrorPx,
                                      double triangulationAngleDeg,
                                      std::vector<xjw::SfmQualityObservation> observations)
{
    xjw::SfmQualityPoint point;
    point.trackLength = trackLength;
    point.reprojectionErrorPx = reprojectionErrorPx;
    point.triangulationAngleDeg = triangulationAngleDeg;
    point.observations = std::move(observations);
    return point;
}

} // namespace

TEST(SfmQualityReportTest, SummarizesTrackErrorAngleAndCoverage)
{
    const std::vector<xjw::SfmQualityPoint> points = {
        makeQualityPoint(2,
                         0.50,
                         5.0,
                         {{0, 10.0, 10.0}, {1, 90.0, 10.0}}),
        makeQualityPoint(3,
                         1.50,
                         12.0,
                         {{0, 10.0, 90.0}, {1, 90.0, 90.0}, {2, 50.0, 50.0}}),
        makeQualityPoint(4,
                         4.20,
                         1.0,
                         {{0, 50.0, 50.0}, {2, 95.0, 95.0}, {3, 5.0, 95.0}, {4, 95.0, 5.0}}),
    };

    xjw::SfmQualityMetricsOptions options;
    options.totalImageCount = 5;
    options.registeredImageCount = 4;
    options.imageWidth = 100.0;
    options.imageHeight = 100.0;
    options.coverageGridColumns = 2;
    options.coverageGridRows = 2;
    options.minTrackLength = 3;
    options.minTriangulationAngleDeg = 2.0;
    options.maxReprojectionErrorPx = 3.0;

    const xjw::SfmQualityMetrics metrics = xjw::computeSfmQualityMetrics(points, options);

    EXPECT_EQ(metrics.totalImageCount, 5);
    EXPECT_EQ(metrics.registeredImageCount, 4);
    EXPECT_EQ(metrics.pointCount, 3);
    EXPECT_EQ(metrics.twoViewTrackCount, 1);
    EXPECT_EQ(metrics.multiViewTrackCount, 2);
    EXPECT_EQ(metrics.weakTrackCount, 1);
    EXPECT_EQ(metrics.weakTriangulationAngleCount, 1);
    EXPECT_EQ(metrics.highReprojectionErrorCount, 1);
    EXPECT_NEAR(metrics.trackLength.mean, 3.0, 1e-9);
    EXPECT_NEAR(metrics.reprojectionError.p95, 4.2, 1e-9);
    EXPECT_NEAR(metrics.triangulationAngle.min, 1.0, 1e-9);
    EXPECT_NEAR(metrics.observationGridCoverageMean, 7.0 / 12.0, 1e-9);

    const QJsonObject json = xjw::serializeSfmQualityMetrics(metrics);
    EXPECT_EQ(json.value(QStringLiteral("registered_image_count")).toInt(), 4);
    EXPECT_EQ(json.value(QStringLiteral("total_image_count")).toInt(), 5);
    EXPECT_EQ(json.value(QStringLiteral("point_count")).toInt(), 3);
    EXPECT_EQ(json.value(QStringLiteral("two_view_track_count")).toInt(), 1);
    EXPECT_EQ(json.value(QStringLiteral("multi_view_track_count")).toInt(), 2);
    EXPECT_EQ(json.value(QStringLiteral("weak_track_count")).toInt(), 1);
    EXPECT_EQ(json.value(QStringLiteral("weak_triangulation_angle_count")).toInt(), 1);
    EXPECT_EQ(json.value(QStringLiteral("high_reprojection_error_count")).toInt(), 1);

    const QJsonObject histogram = json.value(QStringLiteral("track_length_histogram")).toObject();
    EXPECT_EQ(histogram.value(QStringLiteral("2")).toInt(), 1);
    EXPECT_EQ(histogram.value(QStringLiteral("3")).toInt(), 1);
    EXPECT_EQ(histogram.value(QStringLiteral("4")).toInt(), 1);

    const QJsonObject coverage = json.value(QStringLiteral("observation_grid_coverage")).toObject();
    EXPECT_NEAR(coverage.value(QStringLiteral("mean")).toDouble(), 7.0 / 12.0, 1e-6);
    EXPECT_EQ(coverage.value(QStringLiteral("grid_columns")).toInt(), 2);
    EXPECT_EQ(coverage.value(QStringLiteral("grid_rows")).toInt(), 2);
}

TEST(SfmQualityReportTest, EmptyInputProducesStableJson)
{
    xjw::SfmQualityMetricsOptions options;
    options.totalImageCount = 12;
    options.registeredImageCount = 0;
    options.imageWidth = 0.0;
    options.imageHeight = 0.0;

    const xjw::SfmQualityMetrics metrics = xjw::computeSfmQualityMetrics({}, options);
    const QJsonObject json = xjw::serializeSfmQualityMetrics(metrics);

    EXPECT_EQ(metrics.pointCount, 0);
    EXPECT_EQ(metrics.trackLength.count, 0);
    EXPECT_EQ(metrics.observationGridCoverageMean, 0.0);
    EXPECT_EQ(json.value(QStringLiteral("point_count")).toInt(), 0);
    EXPECT_EQ(json.value(QStringLiteral("registered_image_count")).toInt(), 0);
    EXPECT_EQ(json.value(QStringLiteral("total_image_count")).toInt(), 12);
    EXPECT_TRUE(json.value(QStringLiteral("track_length_histogram")).toObject().isEmpty());
}

TEST(SfmQualityReportTest, FlagsSparseCloudThatShouldNotFeedMvs)
{
    const std::vector<xjw::SfmQualityPoint> points = {
        makeQualityPoint(2, 4.5, 0.8, {{0, 10.0, 10.0}, {1, 20.0, 20.0}}),
        makeQualityPoint(2, 3.8, 1.0, {{1, 30.0, 30.0}, {2, 40.0, 40.0}}),
        makeQualityPoint(3, 0.9, 5.0, {{2, 50.0, 50.0}, {3, 60.0, 60.0}, {4, 70.0, 70.0}}),
    };

    xjw::SfmQualityMetricsOptions options;
    options.totalImageCount = 10;
    options.registeredImageCount = 3;
    options.imageWidth = 100.0;
    options.imageHeight = 100.0;
    options.minTrackLength = 3;
    options.minTriangulationAngleDeg = 2.0;
    options.maxReprojectionErrorPx = 3.0;
    options.minRegisteredImageRatioForMvs = 0.50;
    options.maxTwoViewTrackRatioForMvs = 0.60;
    options.maxHighReprojectionErrorRatioForMvs = 0.30;

    const xjw::SfmQualityMetrics metrics = xjw::computeSfmQualityMetrics(points, options);

    EXPECT_FALSE(metrics.acceptableForMvs);
    EXPECT_EQ(metrics.qualityStatus, "blocked");
    EXPECT_TRUE(containsString(metrics.qualityWarnings, "low_registered_image_coverage"));
    EXPECT_TRUE(containsString(metrics.qualityWarnings, "too_many_two_view_tracks"));
    EXPECT_TRUE(containsString(metrics.qualityWarnings, "high_reprojection_error"));

    const QJsonObject gate = xjw::serializeSfmQualityMetrics(metrics).value(QStringLiteral("quality_gate")).toObject();
    EXPECT_FALSE(gate.value(QStringLiteral("acceptable_for_mvs")).toBool(true));
    EXPECT_EQ(gate.value(QStringLiteral("status")).toString(), QStringLiteral("blocked"));
    EXPECT_TRUE(gate.value(QStringLiteral("warnings")).toArray().contains(
        QStringLiteral("too_many_two_view_tracks")));
}

TEST(SfmQualityReportTest, AdvisesWhenTwoViewRatioIsHighButStillUsable)
{
    std::vector<xjw::SfmQualityPoint> points;
    for (int index = 0; index < 6; ++index)
    {
        points.push_back(makeQualityPoint(2, 0.5, 6.0, {{0, 10.0, 10.0}, {1, 20.0, 20.0}}));
    }
    for (int index = 0; index < 2; ++index)
    {
        points.push_back(makeQualityPoint(
            3, 0.5, 6.0, {{0, 10.0, 10.0}, {1, 20.0, 20.0}, {2, 30.0, 30.0}}));
    }

    xjw::SfmQualityMetricsOptions options;
    options.totalImageCount = 3;
    options.registeredImageCount = 3;
    const xjw::SfmQualityMetrics metrics = xjw::computeSfmQualityMetrics(points, options);

    EXPECT_TRUE(metrics.acceptableForMvs);
    EXPECT_EQ(metrics.qualityStatus, "warn");
    EXPECT_TRUE(metrics.qualityWarnings.empty());
    EXPECT_TRUE(containsString(metrics.qualityAdvisories, "high_two_view_track_ratio"));

    const QJsonObject gate = xjw::serializeSfmQualityMetrics(metrics).value(QStringLiteral("quality_gate")).toObject();
    EXPECT_TRUE(gate.value(QStringLiteral("acceptable_for_mvs")).toBool(false));
    EXPECT_TRUE(gate.value(QStringLiteral("warnings")).toArray().isEmpty());
    EXPECT_TRUE(gate.value(QStringLiteral("advisories")).toArray().contains(
        QStringLiteral("high_two_view_track_ratio")));
}

TEST(SfmQualityReportTest, BlocksWhenTwoViewRatioExceedsProductionLimit)
{
    std::vector<xjw::SfmQualityPoint> points;
    for (int index = 0; index < 7; ++index)
    {
        points.push_back(makeQualityPoint(2, 0.5, 6.0, {{0, 10.0, 10.0}, {1, 20.0, 20.0}}));
    }
    points.push_back(makeQualityPoint(
        3, 0.5, 6.0, {{0, 10.0, 10.0}, {1, 20.0, 20.0}, {2, 30.0, 30.0}}));

    xjw::SfmQualityMetricsOptions options;
    options.totalImageCount = 3;
    options.registeredImageCount = 3;
    const xjw::SfmQualityMetrics metrics = xjw::computeSfmQualityMetrics(points, options);

    EXPECT_FALSE(metrics.acceptableForMvs);
    EXPECT_EQ(metrics.qualityStatus, "blocked");
    EXPECT_TRUE(containsString(metrics.qualityWarnings, "too_many_two_view_tracks"));
    EXPECT_FALSE(containsString(metrics.qualityAdvisories, "high_two_view_track_ratio"));
}

TEST(SfmQualityReportTest, FlagsPoorObservationSpatialCoverage)
{
    const std::vector<xjw::SfmQualityPoint> points = {
        makeQualityPoint(4,
                         0.8,
                         8.0,
                         {{0, 5.0, 5.0}, {1, 7.0, 8.0}, {2, 9.0, 6.0}, {3, 10.0, 10.0}}),
        makeQualityPoint(4,
                         0.7,
                         7.0,
                         {{0, 6.0, 6.0}, {1, 8.0, 7.0}, {2, 11.0, 9.0}, {3, 12.0, 11.0}}),
    };

    xjw::SfmQualityMetricsOptions options;
    options.totalImageCount = 4;
    options.registeredImageCount = 4;
    options.imageWidth = 100.0;
    options.imageHeight = 100.0;
    options.coverageGridColumns = 4;
    options.coverageGridRows = 4;
    options.minObservationGridCoverageMeanForMvs = 0.20;

    const xjw::SfmQualityMetrics metrics = xjw::computeSfmQualityMetrics(points, options);

    EXPECT_LT(metrics.observationGridCoverageMean, 0.20);
    EXPECT_FALSE(metrics.acceptableForMvs);
    EXPECT_TRUE(containsString(metrics.qualityWarnings, "poor_observation_spatial_coverage"));

    const QJsonObject gate = xjw::serializeSfmQualityMetrics(metrics).value(QStringLiteral("quality_gate")).toObject();
    EXPECT_FALSE(gate.value(QStringLiteral("acceptable_for_mvs")).toBool(true));
    EXPECT_TRUE(gate.value(QStringLiteral("warnings")).toArray().contains(
        QStringLiteral("poor_observation_spatial_coverage")));
}
