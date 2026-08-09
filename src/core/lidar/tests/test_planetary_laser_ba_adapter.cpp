#include "PlanetaryLaserBaAdapter.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

namespace
{

xjw::lidar::PlanetaryLaserDataset makeDataset()
{
    xjw::lidar::PlanetaryLaserDataset dataset;
    dataset.sourceFormat = xjw::lidar::PlanetaryLaserSourceFormat::PlaScanSiJsonV1;
    dataset.sensorModel = xjw::lidar::PlanetaryLaserSensorModel::Frame;
    dataset.rangeType = xjw::lidar::PlanetaryLaserRangeType::OneWay;
    dataset.reference.targetName = "MOON";
    dataset.reference.bodyFixedFrame = "IAU_MOON";
    dataset.reference.laserFrame = "CAMERA";
    dataset.reference.timeSystem = xjw::lidar::PlanetaryLaserTimeSystem::TdbEtSeconds;
    dataset.reference.latitudeType = "planetocentric";
    dataset.reference.longitudeDirection = "positive_east";

    xjw::lidar::PlanetaryLaserShot shot;
    shot.id = "shot-1";
    shot.pointMode = xjw::lidar::PlanetaryLaserPointMode::Constrained;
    shot.ephemerisTimeSeconds = 123.0;
    shot.observedRangeMeters = 100.0;
    shot.rangeSigmaMeters = 0.5;
    shot.pointBodyFixedMeters = {{1000.0, 20.0, 30.0}};
    shot.pointCovarianceBodyFixedMetersSquared = std::array<double, 9>{{
        4.0, 0.4, 0.0,
        0.4, 9.0, 0.0,
        0.0, 0.0, 1.0,
    }};
    shot.simultaneousImageIds = {"left.cub"};
    shot.leverArmSensorMeters = {{0.1, 0.0, 0.0}};
    shot.leverArmSpecified = true;
    shot.imageMeasures.push_back({
        "left.cub",
        50.0,
        60.0,
        xjw::lidar::PlanetaryLaserImageMeasureKind::ProjectedVirtual,
        std::nullopt,
    });
    shot.imageMeasures.push_back({
        "right.cub",
        52.0,
        61.0,
        xjw::lidar::PlanetaryLaserImageMeasureKind::Measured,
        std::array<double, 4>{{4.0, 0.0, 0.0, 4.0}},
    });
    dataset.shots.push_back(shot);
    return dataset;
}

xjw::lidar::PlanetaryLaserBaAdapterOptions makeOptions()
{
    xjw::lidar::PlanetaryLaserBaAdapterOptions options;
    options.imageAliasesByCameraIndex = {
        {"E:/images/left.cub", "LRO/LEFT"},
        {"E:/images/right.cub", "LRO/RIGHT"},
    };
    options.cameraCoordinateFrame = "IAU_MOON";
    options.cameraSensorFrame = "CAMERA";
    return options;
}

} // namespace

TEST(PlanetaryLaserBaAdapterTest, BuildsConstrainedShotAndIgnoresProjectedMeasure)
{
    const auto dataset = makeDataset();
    const auto options = makeOptions();
    std::vector<xjw::BALaserRangeConstraint> constraints;
    xjw::lidar::PlanetaryLaserBaAdapterSummary summary;
    std::string error;

    ASSERT_TRUE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, options, &constraints, &summary, &error)) << error;
    ASSERT_EQ(constraints.size(), 1u);
    const auto &constraint = constraints.front();
    EXPECT_EQ(constraint.cameraIndex, 0);
    EXPECT_EQ(constraint.pointMode, xjw::BALaserPointMode::Constrained);
    EXPECT_EQ(constraint.shotId, "shot-1");
    EXPECT_EQ(constraint.sourceIndex, 0);
    ASSERT_EQ(constraint.measuredImageObservations.size(), 1u);
    EXPECT_EQ(constraint.measuredImageObservations.front().cameraIndex, 1);
    EXPECT_NEAR(constraint.measuredImageObservations.front().weight, 0.25, 1.0e-12);
    EXPECT_TRUE(constraint.pointPriorSqrtInformation[0] > 0.0);
    EXPECT_TRUE(constraint.pointPriorSqrtInformation[4] > 0.0);
    EXPECT_TRUE(constraint.pointPriorSqrtInformation[8] > 0.0);
    EXPECT_EQ(summary.acceptedShots, 1);
    EXPECT_EQ(summary.ignoredProjectedMeasures, 1);
    EXPECT_EQ(summary.measuredImageObservations, 1);
}

TEST(PlanetaryLaserBaAdapterTest, RejectsLineScanInsteadOfUsingStaticPose)
{
    auto dataset = makeDataset();
    dataset.sensorModel = xjw::lidar::PlanetaryLaserSensorModel::LineScan;
    std::vector<xjw::BALaserRangeConstraint> constraints;
    std::string error;

    EXPECT_FALSE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, makeOptions(), &constraints, nullptr, &error));
    EXPECT_NE(error.find("line_scan"), std::string::npos);
}

TEST(PlanetaryLaserBaAdapterTest, RejectsCoordinateFrameMismatch)
{
    auto options = makeOptions();
    options.cameraCoordinateFrame = "LOCAL_SFM";
    std::vector<xjw::BALaserRangeConstraint> constraints;
    std::string error;

    EXPECT_FALSE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        makeDataset(), options, &constraints, nullptr, &error));
    EXPECT_NE(error.find("坐标系"), std::string::npos);
}

TEST(PlanetaryLaserBaAdapterTest, FreePointNeedsTwoRealMeasuredImages)
{
    auto dataset = makeDataset();
    auto &shot = dataset.shots.front();
    shot.pointMode = xjw::lidar::PlanetaryLaserPointMode::Free;
    shot.pointCovarianceBodyFixedMetersSquared.reset();
    std::vector<xjw::BALaserRangeConstraint> constraints;
    std::string error;

    EXPECT_FALSE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, makeOptions(), &constraints, nullptr, &error));
    EXPECT_NE(error.find("不足两幅真实 measured"), std::string::npos);

    shot.imageMeasures.push_back({
        "left.cub",
        50.0,
        60.0,
        xjw::lidar::PlanetaryLaserImageMeasureKind::Measured,
        std::nullopt,
    });
    ASSERT_TRUE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, makeOptions(), &constraints, nullptr, &error)) << error;
    ASSERT_EQ(constraints.size(), 1u);
    EXPECT_EQ(constraints.front().pointMode, xjw::BALaserPointMode::Free);
    EXPECT_EQ(constraints.front().measuredImageObservations.size(), 2u);
}

TEST(PlanetaryLaserBaAdapterTest, FixedPointDoesNotRequireCovariance)
{
    auto dataset = makeDataset();
    auto &shot = dataset.shots.front();
    shot.pointMode = xjw::lidar::PlanetaryLaserPointMode::Fixed;
    shot.pointCovarianceBodyFixedMetersSquared.reset();
    std::vector<xjw::BALaserRangeConstraint> constraints;
    std::string error;

    ASSERT_TRUE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, makeOptions(), &constraints, nullptr, &error)) << error;
    ASSERT_EQ(constraints.size(), 1u);
    EXPECT_EQ(constraints.front().pointMode, xjw::BALaserPointMode::Fixed);
}

TEST(PlanetaryLaserBaAdapterTest, CanSkipShotsOutsideSelectedCameraSetExplicitly)
{
    auto dataset = makeDataset();
    auto second = dataset.shots.front();
    second.id = "shot-outside";
    second.simultaneousImageIds = {"outside.cub"};
    dataset.shots.push_back(second);
    auto options = makeOptions();
    options.allowUnmappedShots = true;
    std::vector<xjw::BALaserRangeConstraint> constraints;
    xjw::lidar::PlanetaryLaserBaAdapterSummary summary;
    std::string error;

    ASSERT_TRUE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, options, &constraints, &summary, &error)) << error;
    EXPECT_EQ(constraints.size(), 1u);
    EXPECT_EQ(summary.skippedUnmappedShots, 1);
}

TEST(PlanetaryLaserBaAdapterTest, ExactSerialNumberWinsBeforeAmbiguousTailFallback)
{
    auto dataset = makeDataset();
    dataset.shots.front().simultaneousImageIds = {"LRO/1/NACL"};
    auto options = makeOptions();
    options.imageAliasesByCameraIndex = {
        {"LRO/1/NACL"},
        {"LRO/2/NACL", "right.cub"},
    };
    std::vector<xjw::BALaserRangeConstraint> constraints;
    std::string error;

    ASSERT_TRUE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, options, &constraints, nullptr, &error)) << error;
    ASSERT_EQ(constraints.size(), 1u);
    EXPECT_EQ(constraints.front().cameraIndex, 0);

    dataset.shots.front().simultaneousImageIds = {"NACL"};
    EXPECT_FALSE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, options, &constraints, nullptr, &error));
    EXPECT_NE(error.find("歧义"), std::string::npos);
}

TEST(PlanetaryLaserBaAdapterTest, DoesNotSilentlyDropUnmappedMeasuredImage)
{
    auto dataset = makeDataset();
    dataset.shots.front().imageMeasures.front().kind =
        xjw::lidar::PlanetaryLaserImageMeasureKind::Measured;
    dataset.shots.front().imageMeasures.front().imageId = "outside.cub";
    std::vector<xjw::BALaserRangeConstraint> constraints;
    std::string error;

    EXPECT_FALSE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, makeOptions(), &constraints, nullptr, &error));
    EXPECT_NE(error.find("不能静默丢弃"), std::string::npos);

    auto options = makeOptions();
    options.allowUnmappedMeasuredImages = true;
    ASSERT_TRUE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, options, &constraints, nullptr, &error)) << error;
}

TEST(PlanetaryLaserBaAdapterTest, RejectsAnisotropicImageCovarianceInsteadOfAveraging)
{
    auto dataset = makeDataset();
    dataset.shots.front().imageMeasures.back().covariancePixelsSquared =
        std::array<double, 4>{{1.0, 0.2, 0.2, 4.0}};
    std::vector<xjw::BALaserRangeConstraint> constraints;
    std::string error;

    EXPECT_FALSE(xjw::lidar::buildPlanetaryLaserRangeConstraints(
        dataset, makeOptions(), &constraints, nullptr, &error));
    EXPECT_NE(error.find("各向同性"), std::string::npos);
}
