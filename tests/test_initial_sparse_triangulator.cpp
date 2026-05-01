// ============================================================
// test_initial_sparse_triangulator.cpp — 初始稀疏点云三角化器测试
// ============================================================

#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "Camera.h"
#include "triangulation/InitialSparsePointCloudTriangulator.h"

#include <array>
#include <vector>

namespace {

xjw::Camera makeCamera(double cx, double cy, double cz)
{
    xjw::Camera camera;
    camera.setIntrinsics(1200.0, 1200.0, 512.0, 384.0);
    const std::array<double, 9> rotation = {1.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0,
                                            0.0, 0.0, 1.0};
    const std::array<double, 3> center = {cx, cy, cz};
    camera.setPose(rotation, center);
    return camera;
}

bool projectPoint(const xjw::Camera &camera,
                  const std::array<double, 3> &xyz,
                  double *u,
                  double *v)
{
    if (!u || !v)
    {
        return false;
    }

    const double worldPoint[3] = {xyz[0], xyz[1], xyz[2]};
    double projected[2] = {0.0, 0.0};
    if (!camera.projectWorldPoint(worldPoint, projected))
    {
        return false;
    }

    *u = projected[0];
    *v = projected[1];
    return true;
}

} // namespace

TEST(InitialSparsePointCloudTriangulatorTest, KeepsValidTracks)
{
    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    double u0 = 0.0;
    double v0 = 0.0;
    double u1 = 0.0;
    double v1 = 0.0;
    ASSERT_TRUE(projectPoint(camera0, xyz, &u0, &v0));
    ASSERT_TRUE(projectPoint(camera1, xyz, &u1, &v1));

    xjw::BATrack track;
    track.initialPoint = xyz;
    track.observations.push_back(xjw::BAObservation{0, u0, v0});
    track.observations.push_back(xjw::BAObservation{1, u1, v1});

    xjw::InitialSparseTriangulationOptions options;
    options.minTriAngleDeg = 1.0;
    options.maxReprojErrorPx = 2.0;

    const auto result = xjw::InitialSparsePointCloudTriangulator::triangulate(
        std::vector<xjw::Camera>{camera0, camera1},
        std::vector<xjw::BATrack>{track},
        options);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.exportedPointCount, 1);
    EXPECT_EQ(result.rejectedByReprojCount, 0);
}

TEST(InitialSparsePointCloudTriangulatorTest, RejectsLargeReprojectionError)
{
    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    double u0 = 0.0;
    double v0 = 0.0;
    double u1 = 0.0;
    double v1 = 0.0;
    ASSERT_TRUE(projectPoint(camera0, xyz, &u0, &v0));
    ASSERT_TRUE(projectPoint(camera1, xyz, &u1, &v1));

    xjw::BATrack track;
    track.initialPoint = xyz;
    track.observations.push_back(xjw::BAObservation{0, u0 + 20.0, v0});
    track.observations.push_back(xjw::BAObservation{1, u1, v1});

    xjw::InitialSparseTriangulationOptions options;
    options.minTriAngleDeg = 1.0;
    options.maxReprojErrorPx = 2.0;

    const auto result = xjw::InitialSparsePointCloudTriangulator::triangulate(
        std::vector<xjw::Camera>{camera0, camera1},
        std::vector<xjw::BATrack>{track},
        options);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.exportedPointCount, 0);
    EXPECT_EQ(result.rejectedByReprojCount, 1);
}