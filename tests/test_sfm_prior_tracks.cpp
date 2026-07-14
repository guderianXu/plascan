#include "pipeline/IncrementalSfm.h"
#include "registration/PriorTrack.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace
{

xjw::Camera makeCamera(double centerX)
{
    xjw::Camera camera;
    camera.setIntrinsics(900.0, 900.0, 512.0, 384.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {centerX, 0.0, 0.0});
    return camera;
}

xjw::control_points::PriorObservation observation(xjw::ImageId imageId,
                                                   const xjw::Camera &camera,
                                                   const std::array<double, 3> &point,
                                                   xjw::control_points::PriorObservationState state)
{
    double xyz[3] = {point[0], point[1], point[2]};
    double uv[2] = {0.0, 0.0};
    EXPECT_TRUE(camera.projectWorldPoint(xyz, uv));
    xjw::control_points::PriorObservation result;
    result.imageId = imageId;
    result.x = uv[0];
    result.y = uv[1];
    result.state = state;
    result.confidence = 0.95;
    return result;
}

std::array<double, 3> toReferenceFrame(const std::array<double, 3> &local)
{
    return {{10.0 - 2.0 * local[1],
             -5.0 + 2.0 * local[0],
             3.0 + 2.0 * local[2]}};
}

} // namespace

TEST(SfmPriorTrackTest, InjectsPinnedTracksWithoutChangingFeatureCaches)
{
    xjw::IncrementalSfmOptions options;
    options.useKnownCameraPoses = true;
    options.iterativeBARounds = 1;
    options.maxTracksPerImage = 1;
    options.maxTracksPerGridCell = 1;
    options.filterMinTrackLen = 2;
    options.triangulatorOptions.minTriAngle = 0.1;
    options.triangulatorOptions.maxReprojError = 1.0;

    const std::vector<xjw::Camera> cameras = {makeCamera(-2.0), makeCamera(0.0), makeCamera(2.0)};
    xjw::IncrementalSfm sfm(options);
    for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
    {
        sfm.addImageWithCamera(imageId,
                               "synthetic_" + std::to_string(imageId) + ".png",
                               cameras[imageId],
                               {});
    }

    constexpr int kTrackCount = 12;
    for (int index = 0; index < kTrackCount; ++index)
    {
        const std::array<double, 3> point = {
            -1.2 + 0.22 * index,
            -0.8 + 0.13 * (index % 7),
            24.0 + 0.4 * (index % 3)};
        xjw::control_points::PriorTrack track;
        track.markerId = "marker-" + std::to_string(index);
        track.observations = {
            observation(0, cameras[0], point,
                        xjw::control_points::PriorObservationState::ManualPinned),
            observation(1, cameras[1], point,
                        xjw::control_points::PriorObservationState::AutoDetected),
            observation(2, cameras[2], point,
                        xjw::control_points::PriorObservationState::ManualPinned)};
        sfm.addPriorTrack(track);
    }

    const xjw::IncrementalSfmResult result = sfm.run();

    ASSERT_TRUE(result.success) << result.summary;
    EXPECT_EQ(result.numRegisteredImages, 3);
    EXPECT_EQ(result.priorTracksAccepted, kTrackCount);
    EXPECT_EQ(result.priorTracksRejected, 0);
    EXPECT_GE(result.numPoints3D, kTrackCount);
    ASSERT_NE(result.reconstruction, nullptr);
    EXPECT_EQ(result.reconstruction->image(0).keypoints.size(), kTrackCount);
}

TEST(SfmPriorTrackTest, RejectsPredictedBlockedStaleAndDuplicateImageObservations)
{
    xjw::IncrementalSfmOptions options;
    options.useKnownCameraPoses = true;
    options.iterativeBARounds = 1;
    const std::vector<xjw::Camera> cameras = {makeCamera(-1.0), makeCamera(1.0)};
    xjw::IncrementalSfm sfm(options);
    sfm.addImageWithCamera(0, "a.png", cameras[0], {});
    sfm.addImageWithCamera(1, "b.png", cameras[1], {});

    const std::array<double, 3> point = {0.0, 0.0, 20.0};
    xjw::control_points::PriorTrack rejected;
    rejected.markerId = "rejected";
    rejected.observations = {
        observation(0, cameras[0], point,
                    xjw::control_points::PriorObservationState::Predicted),
        observation(1, cameras[1], point,
                    xjw::control_points::PriorObservationState::Blocked)};
    sfm.addPriorTrack(rejected);

    xjw::control_points::PriorTrack duplicate;
    duplicate.markerId = "duplicate";
    duplicate.observations = {
        observation(0, cameras[0], point,
                    xjw::control_points::PriorObservationState::ManualPinned),
        observation(0, cameras[0], point,
                    xjw::control_points::PriorObservationState::AutoDetected),
        observation(1, cameras[1], point,
                    xjw::control_points::PriorObservationState::ManualPinned)};
    sfm.addPriorTrack(duplicate);

    xjw::control_points::PriorTrack stale;
    stale.markerId = "stale";
    auto staleObservation = observation(
        0, cameras[0], point, xjw::control_points::PriorObservationState::ManualPinned);
    staleObservation.stale = true;
    stale.observations = {
        staleObservation,
        observation(1, cameras[1], point,
                    xjw::control_points::PriorObservationState::ManualPinned)};
    sfm.addPriorTrack(stale);

    const xjw::IncrementalSfmResult result = sfm.run();

    EXPECT_EQ(result.priorTracksAccepted, 0);
    EXPECT_EQ(result.priorTracksRejected, 3);
    EXPECT_EQ(result.priorObservationsAccepted, 0);
    EXPECT_FALSE(result.priorTrackDiagnostics.empty());
}

TEST(SfmPriorTrackTest, AppliesControlNetworkButKeepsCheckPointsOutOfBaConstraints)
{
    xjw::IncrementalSfmOptions options;
    options.useKnownCameraPoses = true;
    options.iterativeBARounds = 1;
    options.filterMinTrackLen = 2;
    options.triangulatorOptions.minTriAngle = 0.1;
    options.triangulatorOptions.maxReprojError = 1.0;

    const std::vector<xjw::Camera> cameras = {makeCamera(-2.0), makeCamera(0.0), makeCamera(2.0)};
    xjw::IncrementalSfm sfm(options);
    for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
    {
        sfm.addImageWithCamera(imageId,
                               "control_" + std::to_string(imageId) + ".png",
                               cameras[imageId],
                               {});
    }

    const std::vector<std::array<double, 3>> points = {
        {{-1.0, -0.5, 20.0}},
        {{1.0, -0.5, 20.0}},
        {{-0.5, 1.0, 20.5}},
        {{0.5, 0.5, 22.0}},
        {{0.0, 0.0, 21.0}},
    };
    for (int index = 0; index < static_cast<int>(points.size()); ++index)
    {
        xjw::control_points::PriorTrack track;
        track.markerId = "network-" + std::to_string(index);
        track.role = index < 4
            ? xjw::control_points::MarkerRole::ControlPoint
            : xjw::control_points::MarkerRole::CheckPoint;
        track.hasReference = true;
        track.referenceUsable = true;
        track.referencePoint = toReferenceFrame(points[static_cast<std::size_t>(index)]);
        track.referenceSigma = {{0.01, 0.01, 0.01}};
        track.observations = {
            observation(0, cameras[0], points[static_cast<std::size_t>(index)],
                        xjw::control_points::PriorObservationState::ManualPinned),
            observation(1, cameras[1], points[static_cast<std::size_t>(index)],
                        xjw::control_points::PriorObservationState::ManualPinned),
            observation(2, cameras[2], points[static_cast<std::size_t>(index)],
                        xjw::control_points::PriorObservationState::ManualPinned),
        };
        sfm.addPriorTrack(track);
    }
    xjw::control_points::PriorScaleBar control_scale;
    control_scale.scaleBarId = "control-scale";
    control_scale.firstMarkerId = "network-0";
    control_scale.secondMarkerId = "network-1";
    control_scale.role = xjw::control_points::ScaleBarRole::Control;
    control_scale.measuredDistance = 4.0;
    control_scale.sigma = 0.01;
    sfm.addPriorScaleBar(control_scale);

    xjw::control_points::PriorScaleBar check_scale;
    check_scale.scaleBarId = "check-scale";
    check_scale.firstMarkerId = "network-2";
    check_scale.secondMarkerId = "network-4";
    check_scale.role = xjw::control_points::ScaleBarRole::Check;
    const auto check_first = toReferenceFrame(points[2]);
    const auto check_second = toReferenceFrame(points[4]);
    check_scale.measuredDistance = std::sqrt(
        (check_first[0] - check_second[0]) * (check_first[0] - check_second[0])
        + (check_first[1] - check_second[1]) * (check_first[1] - check_second[1])
        + (check_first[2] - check_second[2]) * (check_first[2] - check_second[2])) + 1.0;
    check_scale.sigma = 0.01;
    sfm.addPriorScaleBar(check_scale);

    const xjw::IncrementalSfmResult result = sfm.run();

    ASSERT_TRUE(result.success) << result.summary;
    ASSERT_TRUE(result.controlNetworkApplied) << result.controlNetworkError;
    EXPECT_EQ(result.controlPointConstraintCount, 4);
    EXPECT_EQ(result.checkPointResidualCount, 1);
    EXPECT_EQ(result.controlScaleBarConstraintCount, 1);
    EXPECT_EQ(result.checkScaleBarResidualCount, 1);
    EXPECT_NEAR(result.checkScaleBarRms, 1.0, 1.0e-3);
    EXPECT_LT(result.controlPointRms, 1.0e-4);
    ASSERT_NE(result.reconstruction, nullptr);
    const auto transformed_center = result.reconstruction->camera(0).cameraCenter();
    const auto expected_center = toReferenceFrame({{-2.0, 0.0, 0.0}});
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_NEAR(transformed_center[axis], expected_center[axis], 1.0e-3);
    }
}
