#include <gtest/gtest.h>

#include "pipeline/SfmCalibrationPreviewSampler.h"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace
{

xjw::Camera makeCamera()
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 500.0, 400.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{0.0, 0.0, 0.0}});
    return camera;
}

} // namespace

TEST(SfmCalibrationPreviewSamplerTest, BalancesCamerasAndImageRadiusDeterministically)
{
    const std::vector<xjw::Camera> cameras(4, makeCamera());
    std::vector<xjw::BATrack> tracks;
    for (int camera_index = 0; camera_index < 4; ++camera_index)
    {
        for (int radius_index = 0; radius_index < 100; ++radius_index)
        {
            xjw::BATrack track;
            track.initialPoint = {{0.0, 0.0, 5.0}};
            track.observations.push_back(
                {camera_index,
                 500.0 + static_cast<double>(radius_index) * 5.0,
                 400.0,
                 1.0});
            tracks.push_back(std::move(track));
        }
    }

    const std::vector<std::size_t> first =
        xjw::sfm_calibration_preview::selectTrackIndices(
            cameras, tracks, 20);
    const std::vector<std::size_t> second =
        xjw::sfm_calibration_preview::selectTrackIndices(
            cameras, tracks, 20);

    EXPECT_EQ(first, second);
    ASSERT_EQ(first.size(), 20U);
    std::array<int, 4> counts{{0, 0, 0, 0}};
    std::array<int, 4> minimum_radius{{100, 100, 100, 100}};
    std::array<int, 4> maximum_radius{{-1, -1, -1, -1}};
    for (const std::size_t track_index : first)
    {
        const int camera_index = static_cast<int>(track_index / 100);
        const int radius_index = static_cast<int>(track_index % 100);
        ++counts[static_cast<std::size_t>(camera_index)];
        minimum_radius[static_cast<std::size_t>(camera_index)] =
            std::min(minimum_radius[static_cast<std::size_t>(camera_index)], radius_index);
        maximum_radius[static_cast<std::size_t>(camera_index)] =
            std::max(maximum_radius[static_cast<std::size_t>(camera_index)], radius_index);
    }
    for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
    {
        EXPECT_EQ(counts[camera_index], 5);
        EXPECT_LE(minimum_radius[camera_index], 15);
        EXPECT_GE(maximum_radius[camera_index], 85);
    }
}

TEST(SfmCalibrationPreviewSamplerTest, ReturnsAllTracksWhenBelowLimit)
{
    const std::vector<xjw::Camera> cameras(1, makeCamera());
    std::vector<xjw::BATrack> tracks(3);
    const std::vector<std::size_t> selected =
        xjw::sfm_calibration_preview::selectTrackIndices(
            cameras, tracks, 10);
    EXPECT_EQ(selected, (std::vector<std::size_t>{0, 1, 2}));
}
