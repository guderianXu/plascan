#include <gtest/gtest.h>

#include "BundleAdjustService.h"
#include "Camera.h"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

#include <array>
#include <fstream>
#include <vector>

namespace
{

xjw::Camera makeCamera()
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{0.0, 0.0, 0.0}});
    return camera;
}

xjw::BATrack makeTrack()
{
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 12.0}};
    track.observations.push_back(xjw::BAObservation{0, 512.0, 384.0});
    track.observations.push_back(xjw::BAObservation{1, 512.0, 384.0});
    return track;
}

QString writeLaserPlanePly(const QString &dir)
{
    const QString path = QDir(dir).filePath(QStringLiteral("laser_plane.ply"));
    std::ofstream out(path.toStdString(), std::ios::binary | std::ios::trunc);
    EXPECT_TRUE(out.good());
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex 1\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property float normal_x\n"
        << "property float normal_y\n"
        << "property float normal_z\n"
        << "property float curvature\n"
        << "end_header\n"
        << "0 0 10 0 0 1 0.02\n";
    return path;
}

QString writeLaserHeightPly(const QString &dir)
{
    const QString path = QDir(dir).filePath(QStringLiteral("laser_height_xyz.ply"));
    std::ofstream out(path.toStdString(), std::ios::binary | std::ios::trunc);
    EXPECT_TRUE(out.good());
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex 1\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "end_header\n"
        << "0 0 10\n";
    return path;
}

} // namespace

TEST(BundleAdjustServiceLidarTest, RunLoadsLaserCloudAndWritesLaserSummary)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack()};

    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = QStringList{QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.enableLaserConstraints = true;
    options.laserConstraintCloudPath = writeLaserPlanePly(tempDir.path());
    options.laserAssociationMaxDistanceMeters = 3.0;
    options.laserWeight = 5.0;
    options.laserHuberDeltaMeters = 10.0;
    options.baOpt.refineCameraPose = false;
    options.baOpt.enablePointFilter = false;
    options.baOpt.maxIterations = 4;

    const xjw::gui::BaServiceResult result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    ASSERT_TRUE(result.resultJson.contains(QStringLiteral("laser_constraints_summary")));
    const QJsonObject summary = result.resultJson.value(QStringLiteral("laser_constraints_summary")).toObject();
    EXPECT_EQ(summary.value(QStringLiteral("enabled")).toBool(), true);
    EXPECT_EQ(summary.value(QStringLiteral("associated_tracks")).toInt(), 1);
    EXPECT_EQ(summary.value(QStringLiteral("laser_constraint_count")).toInt(), 1);
    EXPECT_NEAR(summary.value(QStringLiteral("laser_rms_before_m")).toDouble(), 2.0, 1e-9);
    EXPECT_LT(summary.value(QStringLiteral("laser_rms_after_m")).toDouble(), 0.05);
}

TEST(BundleAdjustServiceLidarTest, RunAppliesQualityWeightWithoutMultiplyingUserLaserWeight)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack()};

    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = QStringList{QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.enableLaserConstraints = true;
    options.laserConstraintCloudPath = writeLaserPlanePly(tempDir.path());
    options.laserAssociationMaxDistanceMeters = 3.0;
    options.laserWeight = 7.0;
    options.laserHuberDeltaMeters = 10.0;
    options.baOpt.refineCameraPose = false;
    options.baOpt.enablePointFilter = false;
    options.baOpt.maxIterations = 1;

    const xjw::gui::BaServiceResult result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    ASSERT_EQ(tracks.size(), 1u);
    ASSERT_EQ(tracks.front().laserPlaneConstraints.size(), 1u);
    EXPECT_NEAR(tracks.front().laserPlaneConstraints.front().weight, 0.3, 1e-12);

    const QJsonObject optionsJson = result.resultJson.value(QStringLiteral("options")).toObject();
    EXPECT_DOUBLE_EQ(optionsJson.value(QStringLiteral("laser_weight")).toDouble(), 7.0);
}

TEST(BundleAdjustServiceLidarTest, RunDerivesStatisticalWeightFromSigma)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack()};
    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = {QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.exportObservationDetails = false;
    options.enableLaserConstraints = true;
    options.laserConstraintCloudPath = writeLaserPlanePly(tempDir.path());
    options.laserAssociationMaxDistanceMeters = 3.0;
    options.laserWeight = 0.0;
    options.laserSigmaMeters = 0.1;
    options.laserHuberDeltaMeters = 10.0;
    options.baOpt.refineCameraPose = false;
    options.baOpt.enablePointFilter = false;

    const auto result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    const QJsonObject resultOptions =
        result.resultJson.value(QStringLiteral("options")).toObject();
    EXPECT_NEAR(resultOptions.value(QStringLiteral("laser_effective_weight")).toDouble(),
                100.0,
                1.0e-9);
    EXPECT_FALSE(resultOptions.value(QStringLiteral("export_observation_details")).toBool());
    const QJsonArray points = result.resultJson.value(QStringLiteral("points")).toArray();
    ASSERT_EQ(points.size(), 1);
    EXPECT_TRUE(points.at(0).toObject().value(QStringLiteral("observations")).toArray().isEmpty());
}

TEST(BundleAdjustServiceLidarTest, RunRejectsWritebackWhenAllLaserConstraintsAreFiltered)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack()};
    tracks.front().observations[0].u += 1000.0;
    tracks.front().observations[1].u -= 1000.0;

    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = {QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.enableLaserConstraints = true;
    options.laserConstraintCloudPath = writeLaserPlanePly(tempDir.path());
    options.laserAssociationMaxDistanceMeters = 3.0;
    options.baOpt.backend = xjw::BABackend::CeresCpu;
    options.baOpt.refineCameraPose = false;
    options.baOpt.maxCeresInitialTrackRms = 0.0;
    options.baOpt.filterMaxReprojError = 2.5;
    options.baOpt.filterSigmaFactor = 0.0;

    const auto result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.pendingCamUpdates.isEmpty());
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("LiDAR 约束在求解或质量过滤后全部失效")));
    const QJsonObject summary =
        result.resultJson.value(QStringLiteral("laser_constraints_summary")).toObject();
    EXPECT_EQ(summary.value(QStringLiteral("associated_tracks")).toInt(), 1);
    EXPECT_EQ(summary.value(QStringLiteral("laser_constraint_count")).toInt(), 0);
}

TEST(BundleAdjustServiceLidarTest, RunUsesXyzLaserCloudAsHeightPlanesWhenExplicitlyEnabled)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack()};

    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = QStringList{QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.enableLaserConstraints = true;
    options.laserConstraintCloudPath = writeLaserHeightPly(tempDir.path());
    options.laserAssociationMaxDistanceMeters = 3.0;
    options.laserUseMissingNormalsAsHeightPlanes = true;
    options.laserWeight = 5.0;
    options.laserHuberDeltaMeters = 10.0;
    options.baOpt.refineCameraPose = false;
    options.baOpt.enablePointFilter = false;
    options.baOpt.maxIterations = 4;

    const xjw::gui::BaServiceResult result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    const QJsonObject summary = result.resultJson.value(QStringLiteral("laser_constraints_summary")).toObject();
    EXPECT_EQ(summary.value(QStringLiteral("map_sample_count")).toInt(), 1);
    EXPECT_EQ(summary.value(QStringLiteral("laser_constraint_count")).toInt(), 1);
    EXPECT_NEAR(summary.value(QStringLiteral("laser_rms_before_m")).toDouble(), 2.0, 1e-9);
    EXPECT_LT(summary.value(QStringLiteral("laser_rms_after_m")).toDouble(), 0.05);

    const QJsonObject optionsJson = result.resultJson.value(QStringLiteral("options")).toObject();
    EXPECT_TRUE(optionsJson.value(QStringLiteral("laser_missing_normals_as_height_planes")).toBool());
}

TEST(BundleAdjustServiceLidarTest, RunWritesControlPointConstraintSummary)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack()};

    xjw::BAControlPointConstraint constraint;
    constraint.point = {{0.0, 0.0, 10.0}};
    constraint.sigmaMeters = 0.05;
    constraint.weight = 1.0;
    tracks.front().controlPointConstraints.push_back(constraint);

    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = QStringList{QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.baOpt.refineCameraPose = false;
    options.baOpt.enablePointFilter = false;
    options.baOpt.enableControlPointConstraints = true;
    options.baOpt.controlPointHuberDeltaMeters = 10.0;
    options.baOpt.maxIterations = 4;

    const xjw::gui::BaServiceResult result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    ASSERT_TRUE(result.resultJson.contains(QStringLiteral("control_point_constraints_summary")));
    const QJsonObject summary =
        result.resultJson.value(QStringLiteral("control_point_constraints_summary")).toObject();
    EXPECT_TRUE(summary.value(QStringLiteral("enabled")).toBool());
    EXPECT_EQ(summary.value(QStringLiteral("control_point_constraint_count")).toInt(), 1);
    EXPECT_NEAR(summary.value(QStringLiteral("control_point_rms_before_m")).toDouble(), 2.0, 1e-9);
    EXPECT_LT(summary.value(QStringLiteral("control_point_rms_after_m")).toDouble(), 0.05);

    const QJsonObject optionsJson = result.resultJson.value(QStringLiteral("options")).toObject();
    EXPECT_TRUE(optionsJson.value(QStringLiteral("enable_control_point_constraints")).toBool());
    EXPECT_DOUBLE_EQ(optionsJson.value(QStringLiteral("control_point_huber_delta_m")).toDouble(), 10.0);
}

TEST(BundleAdjustServiceLidarTest, RunWritesScaleBarConstraintSummary)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack(), makeTrack()};
    tracks[0].initialPoint = {{0.0, 0.0, 10.0}};
    tracks[1].initialPoint = {{12.0, 0.0, 10.0}};
    tracks[1].observations[0].u = 1712.0;
    tracks[1].observations[1].u = 1712.0;

    xjw::BAScaleBarConstraint constraint;
    constraint.trackIndexA = 0;
    constraint.trackIndexB = 1;
    constraint.measuredDistanceMeters = 10.0;
    constraint.sigmaMeters = 0.05;

    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = QStringList{QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.baOpt.refineCameraPose = false;
    options.baOpt.enablePointFilter = false;
    options.baOpt.enableScaleBarConstraints = true;
    options.baOpt.scaleBarWeight = 1000.0;
    options.baOpt.scaleBarHuberDeltaMeters = 10.0;
    options.baOpt.scaleBarConstraints.push_back(constraint);
    options.baOpt.maxIterations = 8;
    options.baOpt.maxPointIterations = 20;

    const xjw::gui::BaServiceResult result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    ASSERT_TRUE(result.resultJson.contains(QStringLiteral("scale_bar_constraints_summary")));
    const QJsonObject summary =
        result.resultJson.value(QStringLiteral("scale_bar_constraints_summary")).toObject();
    EXPECT_TRUE(summary.value(QStringLiteral("enabled")).toBool());
    EXPECT_EQ(summary.value(QStringLiteral("scale_bar_constraint_count")).toInt(), 1);
    EXPECT_NEAR(summary.value(QStringLiteral("scale_bar_rms_before_m")).toDouble(), 2.0, 1e-9);
    EXPECT_LT(summary.value(QStringLiteral("scale_bar_rms_after_m")).toDouble(), 0.2);

    const QJsonObject optionsJson = result.resultJson.value(QStringLiteral("options")).toObject();
    EXPECT_TRUE(optionsJson.value(QStringLiteral("enable_scale_bar_constraints")).toBool());
    EXPECT_DOUBLE_EQ(optionsJson.value(QStringLiteral("scale_bar_huber_delta_m")).toDouble(), 10.0);
}

TEST(BundleAdjustServiceMarkerTest, WritesSeparateControlAndCheckResiduals)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack(), makeTrack()};
    tracks[0].initialPoint = {{0.0, 0.0, 10.0}};
    tracks[1].initialPoint = {{2.0, 0.0, 10.0}};
    for (xjw::BAObservation &observation : tracks[1].observations)
    {
        observation.u = 712.0;
    }

    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = QStringList{QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.exportTsai = false;
    options.exportEvalPlot = false;
    options.baOpt.refineCameraPose = false;
    options.baOpt.enablePointFilter = false;
    options.baOpt.maxIterations = 1;
    options.markerTrackQualityInputs = {
        {QStringLiteral("C1"), xjw::control_points::MarkerRole::ControlPoint,
         0, {{0.0, 0.0, 10.0}}, {{0.01, 0.01, 0.01}}, true},
        {QStringLiteral("K1"), xjw::control_points::MarkerRole::CheckPoint,
         1, {{3.0, 0.0, 10.0}}, {{0.01, 0.01, 0.01}}, false},
    };
    options.markerScaleBarQualityInputs = {
        {QStringLiteral("SB-C"), xjw::control_points::ScaleBarRole::Control, 0, 1, 2.0},
        {QStringLiteral("SB-K"), xjw::control_points::ScaleBarRole::Check, 0, 1, 3.0},
    };

    const auto result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    const QJsonObject report =
        result.resultJson.value(QStringLiteral("marker_quality_report")).toObject();
    EXPECT_EQ(report.value(QStringLiteral("controls")).toObject()
                  .value(QStringLiteral("count")).toInt(), 1);
    EXPECT_EQ(report.value(QStringLiteral("check_points")).toObject()
                  .value(QStringLiteral("count")).toInt(), 1);
    EXPECT_NEAR(report.value(QStringLiteral("check_points")).toObject()
                    .value(QStringLiteral("rms")).toDouble(), 1.0, 1.0e-6);
    EXPECT_NEAR(report.value(QStringLiteral("check_scale_bars")).toObject()
                    .value(QStringLiteral("rms")).toDouble(), 1.0, 1.0e-6);
}

TEST(BundleAdjustServiceLidarTest, RunFailsClearlyWhenLaserCloudPathIsMissing)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack()};

    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = QStringList{QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.enableLaserConstraints = true;
    options.laserConstraintCloudPath = QDir(tempDir.path()).filePath(QStringLiteral("missing.ply"));

    const xjw::gui::BaServiceResult result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("LiDAR"))
                || result.errorMessage.contains(QStringLiteral("激光")));
}


TEST(BundleAdjustServiceLidarTest, RunFailsWhenNoTrackCanAssociateWithLaserCloud)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    std::vector<xjw::BATrack> tracks{makeTrack()};
    xjw::gui::BaServiceOptions options;
    options.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("ba"));
    options.imagePathByIndex = {QStringLiteral("img0.jpg"), QStringLiteral("img1.jpg")};
    options.enableLaserConstraints = true;
    options.laserConstraintCloudPath = writeLaserPlanePly(tempDir.path());
    options.laserAssociationMaxDistanceMeters = 0.01;

    const auto result = xjw::gui::BundleAdjustService::run(cameras, tracks, options);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("未关联")));
}
