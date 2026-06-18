#include <gtest/gtest.h>

#include "BundleAdjustService.h"
#include "Camera.h"

#include <QDir>
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
