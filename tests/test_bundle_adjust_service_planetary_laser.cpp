#include <gtest/gtest.h>

#include "BundleAdjustService.h"
#include "FramePinholeCamera.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

#include <string>
#include <vector>

namespace
{

xjw::FramePinholeCamera makePlanetaryCamera(double centerX)
{
    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{centerX, 0.0, 0.0}});
    return camera;
}

xjw::BATrack makePlanetaryTieTrack()
{
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 10.0}};
    track.observations.push_back({0, 512.0, 384.0, 1.0});
    track.observations.push_back({1, 412.0, 384.0, 1.0});
    return track;
}

QString writePlanetaryLaserJson(const QString &directory,
                                const QString &sensorModel = QStringLiteral("frame"),
                                const QString &imageId = QStringLiteral("img0.cub"))
{
    QString json = QString::fromUtf8(R"json(
{
  "schema": "plascan.planetary_laser_dataset",
  "version": 1,
  "sensor_model": "SENSOR_MODEL",
  "range_type": "one_way",
  "units": {"length": "m", "angle": "deg", "time": "s", "pixel": "px"},
  "reference": {
    "target": "MOON",
    "body_fixed_frame": "IAU_MOON",
    "laser_frame": "CAMERA",
    "time_system": "TDB_ET_SECONDS",
    "latitude_type": "planetocentric",
    "longitude_direction": "positive_east"
  },
  "shots": [{
    "id": "lola-shot-1",
    "point_mode": "constrained",
    "ephemeris_time_s": 123456.5,
    "range_m": 99.0,
    "range_sigma_m": 0.1,
    "point_body_fixed_m": [0.0, 0.0, 100.0],
    "point_covariance_body_fixed_m2": [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
    "simultaneous_image_ids": ["img0.cub"],
    "image_measures": [{
      "image_id": "img0.cub",
      "sample_px": 512.0,
      "line_px": 384.0,
      "kind": "projected"
    }],
    "lever_arm_sensor_m": [0.0, 0.0, 0.0]
  }]
}
)json");
    json.replace(QStringLiteral("SENSOR_MODEL"), sensorModel);
    json.replace(QStringLiteral("img0.cub"), imageId);
    const QString path = QDir(directory).filePath(QStringLiteral("planetary_laser.json"));
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    EXPECT_GT(file.write(json.toUtf8()), 0);
    return path;
}

QString writeIsisPlanetaryLaserJson(const QString &directory)
{
    const QByteArray json = R"json(
{
  "points": [{
    "id": "lola-isis-1",
    "time": 123456.5,
    "range": 0.099,
    "sigmaRange": 0.1,
    "latitude": 0.0,
    "longitude": 0.0,
    "radius": 0.1,
    "aprioriMatrix": [0.0001, 0.0, 0.0, 0.0001, 0.0, 100.0],
    "simultaneousImages": ["LRO/1/NACL"],
    "measures": [{
      "serialNumber": "LRO/1/NACL",
      "sample": 512.0,
      "line": 384.0
    }]
  }]
}
)json";
    const QString path = QDir(directory).filePath(QStringLiteral("isis_lidar.json"));
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    EXPECT_GT(file.write(json), 0);
    return path;
}

xjw::gui::BaServiceOptions makePlanetaryServiceOptions(const QString &directory,
                                                        const QString &jsonPath)
{
    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(directory).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = {
        QDir(directory).filePath(QStringLiteral("img0.cub")),
        QDir(directory).filePath(QStringLiteral("img1.cub")),
    };
    options.selectedImages = options.imagePathByIndex;
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.enablePlanetaryLaserRangeConstraints = true;
    options.planetaryLaserDataPath = jsonPath;
    options.planetaryLaserCameraCoordinateFrame = QStringLiteral("IAU_MOON");
    options.planetaryLaserCameraSensorFrame = QStringLiteral("CAMERA");
    options.planetaryLaserRangeWeight = 1.0;
    options.planetaryLaserRangeHuberDeltaSigma = 10.0;
    options.baOpt.backend = xjw::BABackend::CeresCpu;
    options.baOpt.refineCameraPose = false;
    options.baOpt.enablePointFilter = false;
    options.baOpt.maxIterations = 20;
    return options;
}

} // namespace

TEST(BundleAdjustServicePlanetaryLaserTest, RunsRangeShotWithoutPollutingTrackMetrics)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres backend is not available";
    }

    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());
    const QString jsonPath = writePlanetaryLaserJson(temporaryDirectory.path());
    std::vector<xjw::FramePinholeCamera> cameras{
        makePlanetaryCamera(0.0), makePlanetaryCamera(1.0)};
    std::vector<xjw::BATrack> tracks{makePlanetaryTieTrack()};
    const xjw::gui::BaServiceOptions options =
        makePlanetaryServiceOptions(temporaryDirectory.path(), jsonPath);

    const xjw::gui::BaServiceResult result =
        xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.resultJson.value(QStringLiteral("track_count")).toInt(), 1);
    const QJsonObject summary = result.resultJson
                                    .value(QStringLiteral("planetary_laser_range_summary"))
                                    .toObject();
    EXPECT_EQ(summary.value(QStringLiteral("accepted_shots")).toInt(), 1);
    EXPECT_EQ(summary.value(QStringLiteral("range_constraint_count")).toInt(), 1);
    EXPECT_EQ(summary.value(QStringLiteral("ignored_projected_measures")).toInt(), 1);
    EXPECT_NEAR(summary.value(QStringLiteral("range_rms_before_m")).toDouble(), 1.0, 1.0e-9);
    EXPECT_LT(summary.value(QStringLiteral("range_rms_after_m")).toDouble(), 0.01);
    const QJsonArray shots = summary.value(QStringLiteral("shots")).toArray();
    ASSERT_EQ(shots.size(), 1);
    EXPECT_EQ(shots.at(0).toObject().value(QStringLiteral("id")).toString(),
              QStringLiteral("lola-shot-1"));
    EXPECT_EQ(shots.at(0).toObject()
                  .value(QStringLiteral("lever_arm_camera_m"))
                  .toArray()
                  .size(),
              3);
}

TEST(BundleAdjustServicePlanetaryLaserTest, RejectsLineScanAsStaticFrameCamera)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());
    const QString jsonPath = writePlanetaryLaserJson(
        temporaryDirectory.path(), QStringLiteral("line_scan"));
    std::vector<xjw::FramePinholeCamera> cameras{
        makePlanetaryCamera(0.0), makePlanetaryCamera(1.0)};
    std::vector<xjw::BATrack> tracks{makePlanetaryTieTrack()};
    xjw::gui::BaServiceOptions options =
        makePlanetaryServiceOptions(temporaryDirectory.path(), jsonPath);

    const xjw::gui::BaServiceResult result =
        xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("line_scan")));

    options.dryRun = true;
    const xjw::gui::BaServiceResult dryRunResult =
        xjw::gui::BundleAdjustService::run(cameras, tracks, options);
    EXPECT_FALSE(dryRunResult.success);
    EXPECT_TRUE(dryRunResult.errorMessage.contains(QStringLiteral("line_scan")));
}

TEST(BundleAdjustServicePlanetaryLaserTest, MapsExplicitIsisSerialAliasToCamera)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres backend is not available";
    }

    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());
    const QString jsonPath = writeIsisPlanetaryLaserJson(temporaryDirectory.path());
    std::vector<xjw::FramePinholeCamera> cameras{
        makePlanetaryCamera(0.0), makePlanetaryCamera(1.0)};
    std::vector<xjw::BATrack> tracks{makePlanetaryTieTrack()};
    xjw::gui::BaServiceOptions options =
        makePlanetaryServiceOptions(temporaryDirectory.path(), jsonPath);
    xjw::lidar::PlanetaryLaserIsisContext context;
    context.reference.targetName = "MOON";
    context.reference.bodyFixedFrame = "IAU_MOON";
    context.reference.laserFrame = "CAMERA";
    context.reference.timeSystem =
        xjw::lidar::PlanetaryLaserTimeSystem::TdbEtSeconds;
    context.reference.latitudeType = "planetocentric";
    context.reference.longitudeDirection = "positive_east";
    context.sensorModel = xjw::lidar::PlanetaryLaserSensorModel::Frame;
    context.rangeType = xjw::lidar::PlanetaryLaserRangeType::OneWay;
    context.leverArmSensorMeters = std::array<double, 3>{{0.0, 0.0, 0.0}};
    options.planetaryLaserParseOptions.isisContext = context;
    options.planetaryLaserImageAliasesByCameraIndex = {
        QStringList{QStringLiteral("LRO/1/NACL")},
        QStringList{},
    };
    options.planetaryLaserAllowUnmappedMeasuredImages = true;

    const xjw::gui::BaServiceResult result =
        xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.resultJson
                  .value(QStringLiteral("planetary_laser_range_summary"))
                  .toObject()
                  .value(QStringLiteral("accepted_shots"))
                  .toInt(),
              1);
    const QJsonObject savedOptions =
        result.resultJson.value(QStringLiteral("options")).toObject();
    EXPECT_TRUE(savedOptions
                    .value(QStringLiteral(
                        "planetary_laser_allow_unmapped_measured_images"))
                    .toBool());
    const QJsonArray savedAliases = savedOptions
                                        .value(QStringLiteral(
                                            "planetary_laser_image_aliases_by_camera_index"))
                                        .toArray();
    ASSERT_EQ(savedAliases.size(), 2);
    EXPECT_EQ(savedAliases.at(0)
                  .toObject()
                  .value(QStringLiteral("aliases"))
                  .toArray()
                  .at(0)
                  .toString(),
              QStringLiteral("LRO/1/NACL"));
}

TEST(BundleAdjustServicePlanetaryLaserTest, RejectsSelectedImagesAsImplicitCameraOrder)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());
    const QString jsonPath = writePlanetaryLaserJson(temporaryDirectory.path());
    std::vector<xjw::FramePinholeCamera> cameras{
        makePlanetaryCamera(0.0), makePlanetaryCamera(1.0)};
    std::vector<xjw::BATrack> tracks{makePlanetaryTieTrack()};
    xjw::gui::BaServiceOptions options =
        makePlanetaryServiceOptions(temporaryDirectory.path(), jsonPath);
    options.imagePathByIndex.clear();

    const xjw::gui::BaServiceResult result =
        xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("imagePathByIndex")));
}

TEST(BundleAdjustServicePlanetaryLaserTest, MapsProjectImageUuidsInBundleAdjustCameraOrder)
{
    const QJsonArray images{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/../data/a.cub")},
                    {QStringLiteral("image_uuid"), QStringLiteral("uuid-a")}},
        QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/skipped.cub")},
                    {QStringLiteral("image_uuid"), QStringLiteral("uuid-skipped")}},
        QJsonObject{{QStringLiteral("path"), QStringLiteral("E:\\data\\c.cub")},
                    {QStringLiteral("image_uuid"), QStringLiteral("uuid-c")}},
    };
    const QJsonObject meta{{QStringLiteral("images"), images}};
    QVector<QStringList> aliases{
        QStringList{QStringLiteral("LRO/1/NACL")},
        QStringList{},
    };
    QString error;

    ASSERT_TRUE(xjw::gui::mergePlanetaryLaserProjectImageAliases(
        meta,
        {QStringLiteral("E:/data/a.cub"), QStringLiteral("E:/data/c.cub")},
        &aliases,
        &error)) << error.toStdString();
    ASSERT_EQ(aliases.size(), 2);
    EXPECT_EQ(aliases.at(0),
              (QStringList{QStringLiteral("LRO/1/NACL"),
                           QStringLiteral("uuid-a")}));
    EXPECT_EQ(aliases.at(1), QStringList{QStringLiteral("uuid-c")});
}

TEST(BundleAdjustServicePlanetaryLaserTest, RejectsConflictingProjectImageUuidAliases)
{
    const QJsonArray images{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/a.cub")},
                    {QStringLiteral("image_uuid"), QStringLiteral("uuid-a")}},
        QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/./a.cub")},
                    {QStringLiteral("image_uuid"), QStringLiteral("uuid-b")}},
    };
    QVector<QStringList> aliases;
    QString error;

    EXPECT_FALSE(xjw::gui::mergePlanetaryLaserProjectImageAliases(
        QJsonObject{{QStringLiteral("images"), images}},
        {QStringLiteral("E:/data/a.cub")},
        &aliases,
        &error));
    EXPECT_TRUE(error.contains(QStringLiteral("不是一一对应")));
}
