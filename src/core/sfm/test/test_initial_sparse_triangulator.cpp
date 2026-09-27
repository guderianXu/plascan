// ============================================================
// test_initial_sparse_triangulator.cpp — 初始稀疏点云三角化器测试
// ============================================================

#include <gtest/gtest.h>

#include "triangulation/InitialSparsePointFilter.h"

#include <plabundle/constraints.h>
#include <placamera/frame_camera.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace
{

    std::shared_ptr<const placamera::FramePinholeModel> makeCamera(double cx, double cy, double cz)
    {
        static int next_id = 0;
        const std::string id = std::to_string(next_id++);
        const placamera::FrameId frame("sparse-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("sparse-definition-" + id),
                                                      {1200.0, 1200.0, 512.0, 384.0, 1.0, 1, 1},
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        const std::array<double, 9> rotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        const std::array<double, 3> center = {cx, cy, cz};
        return std::make_shared<const placamera::FramePinholeModel>(
            placamera::FramePinholeModel::create(placamera::CameraInstanceId("sparse-instance-" + id),
                                                 placamera::ImageId("sparse-image-" + id),
                                                 definition,
                                                 {1024, 768},
                                                 placamera::Pose::create(frame, center, rotation)));
    }

    bool
    projectPoint(const placamera::FramePinholeModel& camera, const std::array<double, 3>& xyz, double* u, double* v)
    {
        if (!u || !v)
        {
            return false;
        }

        const auto projected = camera.groundToImage(placamera::GroundCoordinate{camera.groundFrame(), xyz});
        if (!projected)
        {
            return false;
        }

        *u = projected.value().image.sample;
        *v = projected.value().image.line;
        return true;
    }

} // namespace

TEST(InitialSparsePointCloudTriangulatorTest, KeepsValidTracks)
{
    const auto camera0 = makeCamera(0.0, 0.0, 0.0);
    const auto camera1 = makeCamera(8.0, 0.0, 0.0);
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    double u0 = 0.0;
    double v0 = 0.0;
    double u1 = 0.0;
    double v1 = 0.0;
    ASSERT_TRUE(projectPoint(*camera0, xyz, &u0, &v0));
    ASSERT_TRUE(projectPoint(*camera1, xyz, &u1, &v1));

    plabundle::Track track;
    track.initialPoint = xyz;
    track.observations.push_back(plabundle::Observation{0, u0, v0});
    track.observations.push_back(plabundle::Observation{1, u1, v1});

    xjw::InitialSparseTriangulationOptions options;
    options.minTriAngleDeg = 1.0;
    options.maxReprojErrorPx = 2.0;

    const auto result = xjw::InitialSparsePointFilter::filter(
        std::vector<std::shared_ptr<const placamera::FramePinholeModel>>{camera0, camera1},
        std::vector<plabundle::Track>{track},
        options);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.exportedPointCount, 1);
    EXPECT_EQ(result.rejectedByReprojCount, 0);
}

TEST(InitialSparsePointCloudTriangulatorTest, RejectsLargeReprojectionError)
{
    const auto camera0 = makeCamera(0.0, 0.0, 0.0);
    const auto camera1 = makeCamera(8.0, 0.0, 0.0);
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    double u0 = 0.0;
    double v0 = 0.0;
    double u1 = 0.0;
    double v1 = 0.0;
    ASSERT_TRUE(projectPoint(*camera0, xyz, &u0, &v0));
    ASSERT_TRUE(projectPoint(*camera1, xyz, &u1, &v1));

    plabundle::Track track;
    track.initialPoint = xyz;
    track.observations.push_back(plabundle::Observation{0, u0 + 20.0, v0});
    track.observations.push_back(plabundle::Observation{1, u1, v1});

    xjw::InitialSparseTriangulationOptions options;
    options.minTriAngleDeg = 1.0;
    options.maxReprojErrorPx = 2.0;

    const auto result = xjw::InitialSparsePointFilter::filter(
        std::vector<std::shared_ptr<const placamera::FramePinholeModel>>{camera0, camera1},
        std::vector<plabundle::Track>{track},
        options);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.exportedPointCount, 0);
    EXPECT_EQ(result.rejectedByReprojCount, 1);
}
