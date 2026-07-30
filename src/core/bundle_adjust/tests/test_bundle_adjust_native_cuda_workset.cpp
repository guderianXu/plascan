#include "BundleAdjustNativeCudaWorkset.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

namespace
{

xjw::Camera makeCamera(double cx = 0.0)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 320.0, 240.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{cx, 0.0, 0.0}});
    return camera;
}

} // namespace

TEST(NativeCudaWorksetTest, BuildsContiguousWorksetFromValidTracks)
{
    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera(1.0)};
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 5.0}};
    track.observations.push_back({0, 320.0, 240.0, 1.0});
    track.observations.push_back({1, 319.0, 240.0, 0.5});

    xjw::BAOptions options;
    options.fixedCameraIndices.push_back(0);

    const auto build = xjw::detail::native_cuda::buildWorkset(cameras, {track}, options);
    ASSERT_TRUE(build.ok) << build.message;
    EXPECT_EQ(build.workset.cameras.size(), 2u);
    EXPECT_EQ(build.workset.points.size(), 1u);
    EXPECT_EQ(build.workset.observations.size(), 2u);
    EXPECT_EQ(build.workset.cameras[0].fixed, 1);
    EXPECT_EQ(build.workset.cameras[1].fixed, 0);
    EXPECT_EQ(build.workset.originalTrackToPoint[0], 0);
    EXPECT_EQ(build.workset.points[0].observationBegin, 0);
    EXPECT_EQ(build.workset.points[0].observationCount, 2);
}

TEST(NativeCudaWorksetTest, PreservesCameraDepthAxisConvention)
{
    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera(1.0)};
    cameras[1].setDepthAxisFlipped(true);

    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, -5.0}};
    track.observations.push_back({0, 320.0, 240.0, 1.0});
    track.observations.push_back({1, 320.0, 240.0, 1.0});

    const auto build =
        xjw::detail::native_cuda::buildWorkset(cameras, {track}, xjw::BAOptions{});
    ASSERT_TRUE(build.ok) << build.message;
    ASSERT_EQ(build.workset.cameras.size(), 2U);
    EXPECT_EQ(build.workset.cameras[0].depthAxisFlipped, 0);
    EXPECT_EQ(build.workset.cameras[1].depthAxisFlipped, 1);
}

TEST(NativeCudaWorksetTest, FiltersInvalidTracksAndKeepsOriginalMapping)
{
    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera(1.0)};

    xjw::BATrack invalidPoint;
    invalidPoint.initialPoint = {{0.0, 0.0, std::numeric_limits<double>::quiet_NaN()}};
    invalidPoint.observations.push_back({0, 320.0, 240.0, 1.0});
    invalidPoint.observations.push_back({1, 319.0, 240.0, 1.0});

    xjw::BATrack sameCameraOnly;
    sameCameraOnly.initialPoint = {{0.0, 0.0, 5.0}};
    sameCameraOnly.observations.push_back({0, 320.0, 240.0, 1.0});
    sameCameraOnly.observations.push_back({0, 321.0, 240.0, 1.0});

    xjw::BATrack valid;
    valid.initialPoint = {{0.0, 0.0, 6.0}};
    valid.observations.push_back({0, 320.0, 240.0, 1.0});
    valid.observations.push_back({42, 320.0, 240.0, 1.0});
    valid.observations.push_back({1, 319.0, 240.0, 0.0});

    xjw::BAOptions options;
    const auto build = xjw::detail::native_cuda::buildWorkset(cameras,
                                                              {invalidPoint, sameCameraOnly, valid},
                                                              options);

    ASSERT_TRUE(build.ok) << build.message;
    ASSERT_EQ(build.workset.originalTrackToPoint.size(), 3u);
    EXPECT_EQ(build.workset.originalTrackToPoint[0], -1);
    EXPECT_EQ(build.workset.originalTrackToPoint[1], -1);
    EXPECT_EQ(build.workset.originalTrackToPoint[2], 0);
    ASSERT_EQ(build.workset.points.size(), 1u);
    EXPECT_EQ(build.workset.points[0].originalTrackIndex, 2);
    ASSERT_EQ(build.workset.observations.size(), 2u);
    EXPECT_EQ(build.workset.observations[0].cameraIndex, 0);
    EXPECT_EQ(build.workset.observations[1].cameraIndex, 1);
}

TEST(NativeCudaWorksetTest, RejectsUnsupportedSoftConstraints)
{
    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera(1.0)};
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 5.0}};
    track.observations.push_back({0, 320.0, 240.0, 1.0});
    track.observations.push_back({1, 319.0, 240.0, 1.0});
    track.controlPointConstraints.push_back({{{0.0, 0.0, 5.0}}, 1.0, 1.0, 0});

    xjw::BAOptions options;
    options.enableControlPointConstraints = true;

    const auto build = xjw::detail::native_cuda::buildWorkset(cameras, {track}, options);
    EXPECT_FALSE(build.ok);
    EXPECT_NE(build.message.find("控制点"), std::string::npos);
}
