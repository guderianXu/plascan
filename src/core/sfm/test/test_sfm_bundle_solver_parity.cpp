#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <plabundle/photo_alignment.h>
#include <plabundle/solver.h>

#include "pipeline/SfmBundleCameraCodec.h"

namespace
{

    placamera::FramePinholeNumericState makeCamera(double center_x, double center_y, double center_z, int index)
    {
        const placamera::FrameId frame("solver-world");
        const auto suffix = std::to_string(index);
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("solver-definition-" + suffix),
                                                      {1000.0, 1000.0, 512.0, 384.0, 0.005, 1, 1},
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        return placamera::FramePinholeNumericState::fromModel(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("solver-instance-" + suffix),
            placamera::ImageId("solver-image-" + suffix),
            definition,
            {1024, 768},
            placamera::Pose::create(
                frame, {center_x, center_y, center_z}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
    }

    std::vector<placamera::FramePinholeNumericState> makeCameras(int count)
    {
        std::vector<placamera::FramePinholeNumericState> cameras;
        cameras.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            const double position = (static_cast<double>(index) / std::max(1, count - 1) - 0.5) * 18.0;
            cameras.push_back(makeCamera(position, std::sin(index * 0.45) * 3.0, 0.0, index));
        }
        return cameras;
    }

    std::vector<plabundle::Track>
    makeTracks(const std::vector<placamera::FramePinholeNumericState>& cameras, int track_count, int views_per_track)
    {
        std::mt19937 generator(7U);
        std::uniform_real_distribution<double> xy_distribution(-4.0, 4.0);
        std::uniform_real_distribution<double> z_distribution(32.0, 52.0);
        std::normal_distribution<double> initial_noise(0.0, 0.30);
        std::normal_distribution<double> image_noise(0.0, 0.05);

        std::vector<plabundle::Track> tracks;
        tracks.reserve(static_cast<std::size_t>(track_count));
        for (int track_index = 0; track_index < track_count; ++track_index)
        {
            const std::array<double, 3> truth{
                {xy_distribution(generator), xy_distribution(generator), z_distribution(generator)}};
            plabundle::Track track;
            track.initialPoint = {truth[0] + initial_noise(generator),
                                  truth[1] + initial_noise(generator),
                                  truth[2] + initial_noise(generator)};
            const int start = track_index % static_cast<int>(cameras.size());
            for (int view_index = 0; view_index < views_per_track; ++view_index)
            {
                const int camera_index = (start + view_index * 3) % static_cast<int>(cameras.size());
                const auto& camera = cameras[static_cast<std::size_t>(camera_index)];
                const auto projection = camera.groundToImage({camera.groundFrame(), truth});
                if (projection)
                {
                    track.observations.push_back({camera_index,
                                                  projection.value().image.sample + image_noise(generator),
                                                  projection.value().image.line + image_noise(generator),
                                                  1.0,
                                                  1.0});
                }
            }
            if (track.observations.size() >= 2)
            {
                tracks.push_back(std::move(track));
            }
        }
        return tracks;
    }

} // namespace

TEST(SfmBundleSolverIntegrationTest, MatchesFrozenCpuSyntheticBaseline)
{
    const auto cameras = makeCameras(8);
    const auto tracks = makeTracks(cameras, 100, 4);
    ASSERT_EQ(tracks.size(), 100U);

    plabundle::Problem problem;
    std::string error;
    ASSERT_TRUE(xjw::sfm_bundle_camera::encodeAll(cameras, &problem.cameras, &error)) << error;
    problem.tracks = tracks;

    plabundle::SolveOptions options;
    options.backend.requested = plabundle::Backend::PlaMatrixCpu;
    options.solver.numThreads = 2;
    options.solver.maxIterations = 4;
    options.calibration.refineCameraPose = false;
    options.solver.enablePointFilter = false;

    const plabundle::Result result = plabundle::Solver().solve(problem, options);

    plabundle::PhotoAlignmentProblem alignment;
    alignment.problem = problem;
    alignment.cameraIds.reserve(problem.cameras.size());
    alignment.trackIds.reserve(problem.tracks.size());
    for (std::size_t index = 0; index < problem.cameras.size(); ++index)
    {
        alignment.cameraIds.push_back("synthetic-camera-" + std::to_string(index));
    }
    for (std::size_t index = 0; index < problem.tracks.size(); ++index)
    {
        alignment.trackIds.push_back("synthetic-track-" + std::to_string(index));
    }
    const plabundle::Result compatibility_result =
        plabundle::Solver().solve(problem, plabundle::makeCompatibilityOptions(options));
    plabundle::PhotoAlignmentOutcome structured_outcome;
    plabundle::PhotoAlignmentOutcome compatibility_outcome;
    ASSERT_TRUE(plabundle::makePhotoAlignmentOutcome(alignment, result, &structured_outcome, &error)) << error;
    ASSERT_TRUE(plabundle::makePhotoAlignmentOutcome(alignment, compatibility_result, &compatibility_outcome, &error))
        << error;
    plabundle::PhotoAlignmentComparison comparison;
    ASSERT_TRUE(plabundle::comparePhotoAlignmentOutcomes(
        compatibility_outcome, structured_outcome, plabundle::PhotoAlignmentComparisonTolerance{}, &comparison, &error))
        << error;
    EXPECT_TRUE(comparison.equivalent) << comparison.message;
    EXPECT_TRUE(comparison.validTrackMaskMismatchIds.empty());
    EXPECT_EQ(comparison.cameraDifferences.size(), problem.cameras.size());
    EXPECT_EQ(comparison.trackDifferences.size(), problem.tracks.size());

    ASSERT_TRUE(result.usable()) << result.backendMessage;
    EXPECT_EQ(result.status, plabundle::SolveStatus::Success);
    EXPECT_EQ(result.requestedBackend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_EQ(result.usedBackend, plabundle::Backend::PlaMatrixCpu);
    EXPECT_FALSE(result.usedGpu);
    EXPECT_FALSE(result.backendFallback);
    EXPECT_EQ(result.observationCount, 400);
    EXPECT_EQ(result.quality.totalTracks, 100);
    EXPECT_EQ(result.quality.optimizedTracks, 100);
    ASSERT_EQ(result.points.size(), tracks.size());
    EXPECT_NEAR(result.quality.meanRmsBefore, 6.717254889, 1.0e-9);
    EXPECT_NEAR(result.quality.meanRmsAfter, 0.03662643036, 1.0e-10);
    EXPECT_NEAR(result.plaMatrix.initialCost, 23355.95625, 1.0e-5);
    EXPECT_NEAR(result.plaMatrix.finalCost, 0.591483299, 1.0e-9);
    EXPECT_EQ(result.plaMatrix.acceptedSteps, 3);
    EXPECT_EQ(result.plaMatrix.rejectedSteps, 0);
    EXPECT_EQ(result.plaMatrix.linearizations, 4);
    EXPECT_EQ(result.plaMatrix.objectiveEvaluations, 10);
}
