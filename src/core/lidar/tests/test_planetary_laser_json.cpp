#include "PlanetaryLaserJson.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace
{

using xjw::lidar::PlanetaryLaserDataset;
using xjw::lidar::PlanetaryLaserImageMeasureKind;
using xjw::lidar::PlanetaryLaserIsisContext;
using xjw::lidar::PlanetaryLaserJsonParseOptions;
using xjw::lidar::PlanetaryLaserPointMode;
using xjw::lidar::PlanetaryLaserRangeType;
using xjw::lidar::PlanetaryLaserReferenceSystem;
using xjw::lidar::PlanetaryLaserSensorModel;
using xjw::lidar::PlanetaryLaserSourceFormat;
using xjw::lidar::PlanetaryLaserTimeSystem;
using xjw::lidar::parsePlanetaryLaserJson;

PlanetaryLaserIsisContext moonIsisContext()
{
    PlanetaryLaserIsisContext context;
    context.reference.targetName = "Moon";
    context.reference.bodyFixedFrame = "IAU_MOON";
    context.reference.laserFrame = "LRO_LOLA";
    context.reference.timeSystem = PlanetaryLaserTimeSystem::TdbEtSeconds;
    context.reference.latitudeType = "planetocentric";
    context.reference.longitudeDirection = "positive_east";
    context.sensorModel = PlanetaryLaserSensorModel::LineScan;
    context.rangeType = PlanetaryLaserRangeType::OneWay;
    context.leverArmSensorMeters = {{1.0, 2.0, 3.0}};
    return context;
}

const char *validPlaScanJson()
{
    return R"json(
{
  "schema": "plascan.planetary_laser_dataset",
  "version": 1,
  "sensor_model": "frame",
  "range_type": "one_way",
  "units": {"length": "m", "angle": "deg", "time": "s", "pixel": "px"},
  "reference": {
    "target": "MARS",
    "body_fixed_frame": "IAU_MARS",
    "laser_frame": "LANDER_LASER",
    "time_system": "TDB_ET_SECONDS",
    "latitude_type": "planetocentric",
    "longitude_direction": "positive_east"
  },
  "shots": [
    {
      "id": "shot-1",
      "point_mode": "constrained",
      "ephemeris_time_s": 123.25,
      "range_m": 25.5,
      "range_sigma_m": 0.04,
      "point_body_fixed_m": [1000.0, 2000.0, 3000.0],
      "point_covariance_body_fixed_m2": [[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 16.0]],
      "simultaneous_image_ids": ["camera-1"],
      "image_measures": [
        {
          "image_id": "camera-1",
          "sample_px": 400.25,
          "line_px": 300.75,
          "kind": "measured",
          "covariance_px2": [[1.0, 0.1], [0.1, 2.0]]
        }
      ],
      "lever_arm_sensor_m": [0.1, -0.2, 0.3]
    }
  ]
}
)json";
}

TEST(PlanetaryLaserJsonTest, ParsesPlaScanSiV1WithoutChangingMeters)
{
    PlanetaryLaserDataset dataset;
    std::string error;
    ASSERT_TRUE(parsePlanetaryLaserJson(validPlaScanJson(), {}, &dataset, &error)) << error;
    ASSERT_EQ(dataset.sourceFormat, PlanetaryLaserSourceFormat::PlaScanSiJsonV1);
    EXPECT_EQ(dataset.sensorModel, PlanetaryLaserSensorModel::Frame);
    EXPECT_EQ(dataset.rangeType, PlanetaryLaserRangeType::OneWay);
    EXPECT_EQ(dataset.reference.targetName, "MARS");
    ASSERT_EQ(dataset.shots.size(), 1U);

    const auto &shot = dataset.shots.front();
    EXPECT_EQ(shot.id, "shot-1");
    EXPECT_EQ(shot.pointMode, PlanetaryLaserPointMode::Constrained);
    EXPECT_DOUBLE_EQ(shot.ephemerisTimeSeconds, 123.25);
    EXPECT_DOUBLE_EQ(shot.observedRangeMeters, 25.5);
    EXPECT_DOUBLE_EQ(shot.rangeSigmaMeters, 0.04);
    EXPECT_EQ(shot.pointBodyFixedMeters, (std::array<double, 3>{{1000.0, 2000.0, 3000.0}}));
    ASSERT_TRUE(shot.pointCovarianceBodyFixedMetersSquared.has_value());
    EXPECT_DOUBLE_EQ((*shot.pointCovarianceBodyFixedMetersSquared)[4], 9.0);
    EXPECT_TRUE(shot.leverArmSpecified);
    ASSERT_EQ(shot.imageMeasures.size(), 1U);
    EXPECT_EQ(shot.imageMeasures.front().kind, PlanetaryLaserImageMeasureKind::Measured);
    ASSERT_TRUE(shot.imageMeasures.front().covariancePixelsSquared.has_value());
}

TEST(PlanetaryLaserJsonTest, ConvertsPlanetocentricPointToBodyFixedXyz)
{
    const std::string json = R"json(
{
  "schema": "plascan.planetary_laser_dataset", "version": 1,
  "sensor_model": "unknown", "range_type": "round_trip",
  "units": {"length": "m", "angle": "deg", "time": "s", "pixel": "px"},
  "reference": {
    "target": "TEST", "body_fixed_frame": "IAU_TEST", "laser_frame": "TEST_LASER",
    "time_system": "TDB_ET_SECONDS", "latitude_type": "planetocentric",
    "longitude_direction": "positive_east"
  },
  "shots": [{
    "id": "spherical", "point_mode": "free",
    "ephemeris_time_s": 0.0, "range_m": 10.0, "range_sigma_m": 0.1,
    "point_planetocentric": {"latitude_deg": 0.0, "longitude_deg": 90.0, "radius_m": 100.0},
    "simultaneous_image_ids": ["image"], "lever_arm_sensor_m": [0.0, 0.0, 0.0]
  }]
}
)json";

    PlanetaryLaserDataset dataset;
    std::string error;
    ASSERT_TRUE(parsePlanetaryLaserJson(json, {}, &dataset, &error)) << error;
    EXPECT_EQ(dataset.sensorModel, PlanetaryLaserSensorModel::Unknown);
    EXPECT_EQ(dataset.rangeType, PlanetaryLaserRangeType::RoundTrip);
    EXPECT_NEAR(dataset.shots[0].pointBodyFixedMeters[0], 0.0, 1.0e-12);
    EXPECT_NEAR(dataset.shots[0].pointBodyFixedMeters[1], 100.0, 1.0e-12);
    EXPECT_NEAR(dataset.shots[0].pointBodyFixedMeters[2], 0.0, 1.0e-12);
}

TEST(PlanetaryLaserJsonTest, RejectsImplicitOrMixedUnits)
{
    std::string json = validPlaScanJson();
    const std::size_t position = json.find("\"length\": \"m\"");
    ASSERT_NE(position, std::string::npos);
    json.replace(position, std::string("\"length\": \"m\"").size(), "\"length\": \"km\"");

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("length=m"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, RequiresExplicitImageMeasureKind)
{
    std::string json = validPlaScanJson();
    const std::string kind = "\"kind\": \"measured\",";
    const std::size_t position = json.find(kind);
    ASSERT_NE(position, std::string::npos);
    json.erase(position, kind.size());

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("kind"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, RequiresExplicitPointMode)
{
    std::string json = validPlaScanJson();
    const std::string point_mode = "\"point_mode\": \"constrained\",";
    const std::size_t position = json.find(point_mode);
    ASSERT_NE(position, std::string::npos);
    json.erase(position, point_mode.size());

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("point_mode"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, RejectsUnknownPlaScanFieldInsteadOfIgnoringATypo)
{
    std::string json = validPlaScanJson();
    const std::string covariance = "\"covariance_px2\"";
    const std::size_t position = json.find(covariance);
    ASSERT_NE(position, std::string::npos);
    json.replace(position, covariance.size(), "\"covaraince_px2\"");

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("unsupported field"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, FreePointCannotSilentlyUseCovarianceAsSoftConstraint)
{
    std::string json = validPlaScanJson();
    const std::string constrained = "\"point_mode\": \"constrained\"";
    const std::size_t position = json.find(constrained);
    ASSERT_NE(position, std::string::npos);
    json.replace(position, constrained.size(), "\"point_mode\": \"free\"");

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("free point cannot carry a soft covariance"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, FixedPointRetainsCovarianceWithoutChangingItsMode)
{
    std::string json = validPlaScanJson();
    const std::string constrained = "\"point_mode\": \"constrained\"";
    const std::size_t position = json.find(constrained);
    ASSERT_NE(position, std::string::npos);
    json.replace(position, constrained.size(), "\"point_mode\": \"fixed\"");

    PlanetaryLaserDataset dataset;
    std::string error;
    ASSERT_TRUE(parsePlanetaryLaserJson(json, {}, &dataset, &error)) << error;
    EXPECT_EQ(dataset.shots.front().pointMode, PlanetaryLaserPointMode::Fixed);
    EXPECT_TRUE(dataset.shots.front().pointCovarianceBodyFixedMetersSquared.has_value());
}

TEST(PlanetaryLaserJsonTest, RejectsNonSymmetricPointCovariance)
{
    std::string json = validPlaScanJson();
    const std::string row = "[1.0, 9.0, 0.0]";
    const std::size_t position = json.find(row);
    ASSERT_NE(position, std::string::npos);
    json.replace(position, row.size(), "[2.0, 9.0, 0.0]");

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("symmetric"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, ConstrainedPointRequiresInvertibleCovariance)
{
    std::string json = validPlaScanJson();
    const std::string covariance =
        "[[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 16.0]]";
    const std::size_t position = json.find(covariance);
    ASSERT_NE(position, std::string::npos);
    json.replace(position, covariance.size(),
                 "[[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 16.0]]");

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("positive definite"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, IsisImportRequiresExternalReferenceContext)
{
    const std::string json = R"json({"points": []})json";
    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("requires explicit target/frame/time-system"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, IsisImportRequiresExplicitLeverArmEvenWhenItIsZero)
{
    const std::string json = R"json(
{"points":[{
  "id":"L1", "time":1.0, "range":1.0, "sigmaRange":1.0,
  "latitude":0.0, "longitude":0.0, "radius":1.0,
  "simultaneousImages":["image"],
  "measures":[{"serialNumber":"image", "sample":1.0, "line":1.0}]
}]}
)json";
    PlanetaryLaserJsonParseOptions options;
    options.isisContext = moonIsisContext();
    options.isisContext->leverArmSensorMeters.reset();

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, options, &dataset, &error));
    EXPECT_NE(error.find("explicitly specify the laser lever arm"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, ConvertsIsisUnitsAndMarksMeasuresProjected)
{
    const std::string json = R"json(
{
  "points": [{
    "id": "Lidar0001",
    "time": 317845268.5,
    "range": 56.5,
    "sigmaRange": 10.0,
    "latitude": 0.0,
    "longitude": 0.0,
    "radius": 2.0,
    "aprioriMatrix": [1.0e-12, 0.0, 0.0, 4.0e-12, 0.0, 9.0],
    "simultaneousImages": ["LRO/NACL"],
    "measures": [{"serialNumber": "LRO/NACL", "sample": 12.5, "line": 34.5, "kind": "measured"}]
  }]
}
)json";
    PlanetaryLaserJsonParseOptions options;
    options.isisContext = moonIsisContext();

    PlanetaryLaserDataset dataset;
    std::string error;
    ASSERT_TRUE(parsePlanetaryLaserJson(json, options, &dataset, &error)) << error;
    ASSERT_EQ(dataset.sourceFormat, PlanetaryLaserSourceFormat::IsisLidarDataJson);
    EXPECT_EQ(dataset.sensorModel, PlanetaryLaserSensorModel::LineScan);
    EXPECT_EQ(dataset.rangeType, PlanetaryLaserRangeType::OneWay);
    ASSERT_EQ(dataset.shots.size(), 1U);

    const auto &shot = dataset.shots.front();
    EXPECT_DOUBLE_EQ(shot.observedRangeMeters, 56500.0);
    EXPECT_DOUBLE_EQ(shot.rangeSigmaMeters, 10.0);
    EXPECT_EQ(shot.pointMode, PlanetaryLaserPointMode::Constrained);
    EXPECT_NEAR(shot.pointBodyFixedMeters[0], 2000.0, 1.0e-12);
    EXPECT_NEAR(shot.pointBodyFixedMeters[1], 0.0, 1.0e-12);
    EXPECT_NEAR(shot.pointBodyFixedMeters[2], 0.0, 1.0e-12);
    EXPECT_EQ(shot.leverArmSensorMeters, (std::array<double, 3>{{1.0, 2.0, 3.0}}));
    ASSERT_EQ(shot.imageMeasures.size(), 1U);
    EXPECT_EQ(shot.imageMeasures.front().kind, PlanetaryLaserImageMeasureKind::ProjectedVirtual);

    ASSERT_TRUE(shot.pointCovarianceBodyFixedMetersSquared.has_value());
    const auto &covariance = *shot.pointCovarianceBodyFixedMetersSquared;
    EXPECT_NEAR(covariance[0], 9.0, 1.0e-12);
    EXPECT_NEAR(covariance[4], 1.6e-5, 1.0e-15);
    EXPECT_NEAR(covariance[8], 4.0e-6, 1.0e-15);
}

TEST(PlanetaryLaserJsonTest, IsisDefaultsDoNotPretendTheSensorIsAFrameCamera)
{
    const std::string json = R"json(
{"points":[{
  "id":"L1", "time":1.0, "range":1.0, "sigmaRange":1.0,
  "latitude":0.0, "longitude":0.0, "radius":1.0,
  "simultaneousImages":["image"],
  "measures":[{"serialNumber":"image", "sample":1.0, "line":1.0}]
}]}
)json";
    PlanetaryLaserIsisContext context = moonIsisContext();
    context.sensorModel = PlanetaryLaserSensorModel::Unknown;
    context.rangeType = PlanetaryLaserRangeType::Unknown;
    PlanetaryLaserJsonParseOptions options;
    options.isisContext = context;

    PlanetaryLaserDataset dataset;
    std::string error;
    ASSERT_TRUE(parsePlanetaryLaserJson(json, options, &dataset, &error)) << error;
    EXPECT_EQ(dataset.sensorModel, PlanetaryLaserSensorModel::Unknown);
    EXPECT_EQ(dataset.rangeType, PlanetaryLaserRangeType::Unknown);
    EXPECT_EQ(dataset.shots.front().pointMode, PlanetaryLaserPointMode::Free);
}

TEST(PlanetaryLaserJsonTest, RejectsIsisSimultaneousImageWithoutProjectedMeasure)
{
    const std::string json = R"json(
{"points":[{
  "id":"L1", "time":1.0, "range":1.0, "sigmaRange":1.0,
  "latitude":0.0, "longitude":0.0, "radius":1.0,
  "simultaneousImages":["simultaneous"],
  "measures":[{"serialNumber":"other", "sample":1.0, "line":1.0}]
}]}
)json";
    PlanetaryLaserJsonParseOptions options;
    options.isisContext = moonIsisContext();
    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, options, &dataset, &error));
    EXPECT_NE(error.find("has no projected measure"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, RejectsDuplicateShotIds)
{
    std::string json = validPlaScanJson();
    const std::size_t closing = json.rfind("]");
    ASSERT_NE(closing, std::string::npos);
    const std::size_t shot_start = json.find("    {");
    const std::size_t shot_end = json.rfind("    }");
    ASSERT_NE(shot_start, std::string::npos);
    ASSERT_NE(shot_end, std::string::npos);
    const std::string duplicate = json.substr(shot_start, shot_end + 5 - shot_start);
    json.insert(closing, ",\n" + duplicate);

    PlanetaryLaserDataset dataset;
    std::string error;
    EXPECT_FALSE(parsePlanetaryLaserJson(json, {}, &dataset, &error));
    EXPECT_NE(error.find("duplicate id"), std::string::npos);
}

TEST(PlanetaryLaserJsonTest, PublicDatasetValidationRejectsInvalidEnums)
{
    PlanetaryLaserDataset dataset;
    std::string error;
    ASSERT_TRUE(parsePlanetaryLaserJson(validPlaScanJson(), {}, &dataset, &error)) << error;

    dataset.shots.front().pointMode = PlanetaryLaserPointMode::Unspecified;
    EXPECT_FALSE(dataset.validate(&error));
    EXPECT_NE(error.find("invalid point mode"), std::string::npos);

    ASSERT_TRUE(parsePlanetaryLaserJson(validPlaScanJson(), {}, &dataset, &error)) << error;
    dataset.shots.front().pointMode = static_cast<PlanetaryLaserPointMode>(999);
    EXPECT_FALSE(dataset.validate(&error));
    EXPECT_NE(error.find("invalid point mode"), std::string::npos);

    ASSERT_TRUE(parsePlanetaryLaserJson(validPlaScanJson(), {}, &dataset, &error)) << error;
    dataset.shots.front().imageMeasures.front().kind =
        static_cast<PlanetaryLaserImageMeasureKind>(999);
    EXPECT_FALSE(dataset.validate(&error));
    EXPECT_NE(error.find("invalid or duplicate image measure"), std::string::npos);
}

} // namespace
