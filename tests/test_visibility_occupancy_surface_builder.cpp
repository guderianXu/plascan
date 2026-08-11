#include "DepthTsdfSurfaceBuilder.h"
#include "Mc33IsoSurfaceExtractor.h"
#include "VisibilityOccupancyCleanup.h"
#include "VisibilityOccupancyHandleRepair.h"
#include "VisibilityOccupancyTsdfCompletion.h"
#include "VisibilityOccupancySurfaceBuilder.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

std::array<double, 3> normalize(std::array<double, 3> value)
{
    const double length = std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]);
    for (double &component : value)
    {
        component /= length;
    }
    return value;
}

std::array<double, 3> cross(
    const std::array<double, 3> &lhs,
    const std::array<double, 3> &rhs)
{
    return {
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

xjw::FramePinholeCamera makeLookAtCamera(const std::array<double, 3> &center)
{
    const std::array<double, 3> forward =
        normalize({-center[0], -center[1], -center[2]});
    const std::array<double, 3> provisional_up =
        std::fabs(forward[1]) > 0.9
        ? std::array<double, 3>{0.0, 0.0, 1.0}
        : std::array<double, 3>{0.0, 1.0, 0.0};
    const std::array<double, 3> right =
        normalize(cross(forward, provisional_up));
    const std::array<double, 3> down =
        normalize(cross(forward, right));
    const std::array<std::array<double, 3>, 3> rows{
        right, down, forward};
    std::array<double, 9> camera_to_world{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            camera_to_world[static_cast<std::size_t>(column * 3 + row)] =
                rows[static_cast<std::size_t>(row)][column];
        }
    }

    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(80.0, 80.0, 32.0, 32.0);
    camera.setPose(camera_to_world, center);
    return camera;
}

struct SyntheticFrame
{
    xjw::FramePinholeCamera camera;
    cv::Mat depth;
    cv::Mat confidence;
    cv::Mat valid;
    cv::Mat support;
};

SyntheticFrame renderSphere(
    const std::array<double, 3> &camera_center,
    double radius)
{
    SyntheticFrame frame;
    frame.camera = makeLookAtCamera(camera_center);
    frame.depth = cv::Mat::zeros(64, 64, CV_32FC1);
    frame.confidence = cv::Mat(64, 64, CV_32FC1, cv::Scalar(1.0f));
    frame.valid = cv::Mat::zeros(64, 64, CV_8UC1);
    frame.support = cv::Mat::zeros(64, 64, CV_8UC1);
    for (int row = 0; row < 64; ++row)
    {
        for (int column = 0; column < 64; ++column)
        {
            const double pixel[2] = {
                static_cast<double>(column),
                static_cast<double>(row)};
            double ray_point[3]{};
            if (!frame.camera.unprojectPixel(pixel, 1.0, ray_point))
            {
                continue;
            }
            std::array<double, 3> direction = normalize({
                ray_point[0] - camera_center[0],
                ray_point[1] - camera_center[1],
                ray_point[2] - camera_center[2]});
            const double projection =
                camera_center[0] * direction[0] +
                camera_center[1] * direction[1] +
                camera_center[2] * direction[2];
            const double center_squared =
                camera_center[0] * camera_center[0] +
                camera_center[1] * camera_center[1] +
                camera_center[2] * camera_center[2];
            const double discriminant =
                projection * projection - (center_squared - radius * radius);
            if (discriminant < 0.0)
            {
                continue;
            }
            const double ray_distance =
                -projection - std::sqrt(discriminant);
            if (!(ray_distance > 0.0))
            {
                continue;
            }
            const double world[3] = {
                camera_center[0] + direction[0] * ray_distance,
                camera_center[1] + direction[1] * ray_distance,
                camera_center[2] + direction[2] * ray_distance};
            const double depth = frame.camera.positiveDepth(world);
            frame.depth.at<float>(row, column) =
                static_cast<float>(depth);
            frame.valid.at<std::uint8_t>(row, column) = 255;
            frame.support.at<std::uint8_t>(row, column) = 255;
        }
    }
    return frame;
}

std::vector<SyntheticFrame> makeSphereFrames()
{
    const std::array<std::array<double, 3>, 6> centers{{
        {{3.0, 0.0, 0.0}}, {{-3.0, 0.0, 0.0}},
        {{0.0, 3.0, 0.0}}, {{0.0, -3.0, 0.0}},
        {{0.0, 0.0, 3.0}}, {{0.0, 0.0, -3.0}}}};
    std::vector<SyntheticFrame> frames;
    frames.reserve(centers.size());
    for (const auto &center : centers)
    {
        frames.push_back(renderSphere(center, 0.62));
    }
    return frames;
}

std::vector<xjw::mesh::VisibilityOccupancyFrameView> makeViews(
    const std::vector<SyntheticFrame> &frames)
{
    std::vector<xjw::mesh::VisibilityOccupancyFrameView> views;
    views.reserve(frames.size());
    for (const SyntheticFrame &frame : frames)
    {
        xjw::mesh::VisibilityOccupancyFrameView view;
        view.camera = &frame.camera;
        view.depth = &frame.depth;
        view.confidence = &frame.confidence;
        view.depthValidMask = &frame.valid;
        view.supportMask = &frame.support;
        views.push_back(view);
    }
    return views;
}

xjw::mesh::VisibilityOccupancyOptions makeOptions()
{
    xjw::mesh::VisibilityOccupancyOptions options;
    options.resolution = 24;
    options.minimumVisibleViews = 2;
    options.minimumSilhouetteViews = 2;
    options.allowedSilhouetteViolations = 1;
    options.pairwiseCapacity = 4;
    return options;
}

std::size_t occupancyIndex(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return static_cast<std::size_t>(
        x + dimensions[0] * (y + dimensions[1] * z));
}

std::vector<std::uint8_t> solidOccupancyBlock(
    const std::array<int, 3> &dimensions)
{
    const std::size_t count = static_cast<std::size_t>(
        dimensions[0] * dimensions[1] * dimensions[2]);
    std::vector<std::uint8_t> occupied(count, 0);
    for (int z = 1; z + 1 < dimensions[2]; ++z)
    {
        for (int y = 1; y + 1 < dimensions[1]; ++y)
        {
            for (int x = 1; x + 1 < dimensions[0]; ++x)
            {
                occupied[occupancyIndex(dimensions, x, y, z)] = 1;
            }
        }
    }
    return occupied;
}

std::vector<std::uint8_t> occupancyWithStraightTunnel(
    const std::array<int, 3> &dimensions)
{
    std::vector<std::uint8_t> occupied = solidOccupancyBlock(dimensions);
    const int y = dimensions[1] / 2;
    const int z = dimensions[2] / 2;
    for (int x = 1; x + 1 < dimensions[0]; ++x)
    {
        occupied[occupancyIndex(dimensions, x, y, z)] = 0;
    }
    return occupied;
}

} // namespace

TEST(VisibilityOccupancySurfaceBuilderTest, ReconstructsClosedSphereOccupancy)
{
    const std::vector<SyntheticFrame> frames = makeSphereFrames();
    const auto result = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        makeOptions());

    ASSERT_TRUE(result.ok) << result.error;
    const int center_x = result.sampleDimensions[0] / 2;
    const int center_y = result.sampleDimensions[1] / 2;
    const int center_z = result.sampleDimensions[2] / 2;
    const std::size_t center =
        result.index(center_x, center_y, center_z);
    EXPECT_EQ(result.occupied[center], 1);
    EXPECT_LT(result.signedDistanceSamples[center], 0.0f);
    EXPECT_EQ(result.occupied[result.index(0, 0, 0)], 0);
    EXPECT_GT(result.signedDistanceSamples[result.index(0, 0, 0)], 0.0f);
    EXPECT_GT(result.statistics.depthEmptyVoteCount, 0U);
    EXPECT_GT(result.statistics.depthFullVoteCount, 0U);
    EXPECT_GT(result.statistics.fullSampleCountAfterCleanup, 100U);

    for (int z = 0; z < result.sampleDimensions[2]; ++z)
    {
        for (int y = 0; y < result.sampleDimensions[1]; ++y)
        {
            EXPECT_EQ(result.occupied[result.index(0, y, z)], 0);
            EXPECT_EQ(
                result.occupied[
                    result.index(result.sampleDimensions[0] - 1, y, z)],
                0);
        }
    }
}

TEST(VisibilityOccupancySurfaceBuilderTest, HonorsAlignedSampleDimensions)
{
    const std::vector<SyntheticFrame> frames = makeSphereFrames();
    auto options = makeOptions();
    options.sampleDimensions = {19, 25, 21};

    const auto result = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        options);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.sampleDimensions, options.sampleDimensions);
}

TEST(VisibilityOccupancySurfaceBuilderTest, HonorsConfiguredClosingRadius)
{
    const std::vector<SyntheticFrame> frames = makeSphereFrames();
    auto options = makeOptions();
    options.closingIterations = 6;
    options.maximumHandleRepairPasses = 1;

    const auto result = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        options);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.statistics.effectiveClosingIterations, 6);
}

TEST(VisibilityOccupancySurfaceBuilderTest, MissingDepthStaysUnknownNotEmpty)
{
    std::vector<SyntheticFrame> frames = makeSphereFrames();
    frames[0].valid.setTo(cv::Scalar(0));
    frames[1].valid.setTo(cv::Scalar(0));
    const auto result = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        makeOptions());

    ASSERT_TRUE(result.ok) << result.error;
    const std::size_t center = result.index(
        result.sampleDimensions[0] / 2,
        result.sampleDimensions[1] / 2,
        result.sampleDimensions[2] / 2);
    EXPECT_EQ(result.occupied[center], 1);
    EXPECT_GT(result.statistics.fullSampleCountAfterCleanup, 100U);
}

TEST(VisibilityOccupancySurfaceBuilderTest, DepthSupportedPriorKeepsClosedSphereInterior)
{
    const std::vector<SyntheticFrame> frames = makeSphereFrames();
    auto options = makeOptions();
    options.minimumDepthFullViewsForSilhouettePrior = 2;
    const auto result = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        options);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(
        result.occupied[result.index(
            result.sampleDimensions[0] / 2,
            result.sampleDimensions[1] / 2,
            result.sampleDimensions[2] / 2)],
        1);
    EXPECT_GT(result.statistics.silhouetteFullPriorSampleCount, 0U);
    EXPECT_GT(
        result.statistics
            .silhouetteFullPriorRejectedWithoutDepthSupportSampleCount,
        0U);
    EXPECT_GT(result.statistics.silhouetteFullPriorCapacityTotal, 0U);
}

TEST(VisibilityOccupancySurfaceBuilderTest, SilhouettePriorCanRequireNearSurfaceDepthEvidence)
{
    std::vector<SyntheticFrame> frames = makeSphereFrames();
    for (SyntheticFrame &frame : frames)
    {
        frame.valid.setTo(cv::Scalar(0));
    }

    auto unrestricted_options = makeOptions();
    unrestricted_options.minimumDepthFullViewsForSilhouettePrior = 0;
    const auto unrestricted =
        xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
            {-1.0f, -1.0f, -1.0f},
            {1.0f, 1.0f, 1.0f},
            makeViews(frames),
            unrestricted_options);
    ASSERT_TRUE(unrestricted.ok) << unrestricted.error;
    EXPECT_GT(
        unrestricted.statistics.silhouetteFullPriorSampleCount,
        0U);

    auto depth_supported_options = makeOptions();
    depth_supported_options.minimumDepthFullViewsForSilhouettePrior = 1;
    const auto depth_supported =
        xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
            {-1.0f, -1.0f, -1.0f},
            {1.0f, 1.0f, 1.0f},
            makeViews(frames),
            depth_supported_options);
    EXPECT_FALSE(depth_supported.ok);
    EXPECT_EQ(
        depth_supported.statistics.silhouetteFullPriorSampleCount,
        0U);
    EXPECT_GT(
        depth_supported.statistics
            .silhouetteFullPriorRejectedWithoutDepthSupportSampleCount,
        0U);
    EXPECT_EQ(
        depth_supported.error,
        "visibility occupancy cut contains no full samples");
}

TEST(VisibilityOccupancySurfaceBuilderTest, CancellationReturnsNoPartialField)
{
    const std::vector<SyntheticFrame> frames = makeSphereFrames();
    auto options = makeOptions();
    options.isCancelled = []()
    {
        return true;
    };
    const auto result = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        options);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.cancelled);
    EXPECT_TRUE(result.occupied.empty());
}

TEST(VisibilityOccupancySurfaceBuilderTest, ZeroSilhouetteCapacityIsNotReportedAsApplied)
{
    const std::vector<SyntheticFrame> frames = makeSphereFrames();
    auto options = makeOptions();
    options.silhouetteFullPriorCapacity = 0;

    const auto result = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        options);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_GT(
        result.statistics.silhouetteFullPriorCandidateSampleCount,
        0U);
    EXPECT_EQ(result.statistics.silhouetteFullPriorSampleCount, 0U);
    EXPECT_EQ(result.statistics.silhouetteFullPriorCapacityTotal, 0U);
}

TEST(VisibilityOccupancySurfaceBuilderTest, RejectsMismatchedFrameImageDimensions)
{
    std::vector<SyntheticFrame> frames = makeSphereFrames();
    frames.front().support = cv::Mat(32, 32, CV_8UC1, cv::Scalar(255));

    const auto result = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        makeOptions());

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(
        result.error,
        "visibility occupancy frame images have inconsistent dimensions");
}

TEST(VisibilityOccupancyCleanupTest, ClosingSealsSingleSampleTunnel)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    const auto index = [&dimensions](int x, int y, int z)
    {
        return static_cast<std::size_t>(
            x + dimensions[0] * (y + dimensions[1] * z));
    };
    std::vector<std::uint8_t> occupied(343, 0);
    for (int z = 1; z <= 5; ++z)
    {
        for (int y = 1; y <= 5; ++y)
        {
            for (int x = 1; x <= 5; ++x)
            {
                occupied[index(x, y, z)] = 1;
            }
        }
    }
    for (int x = 1; x <= 5; ++x)
    {
        occupied[index(x, 3, 3)] = 0;
    }

    const std::uint64_t changed =
        xjw::mesh::detail::closeVisibilityOccupancySixConnected(
            dimensions, 1, &occupied);

    EXPECT_GT(changed, 0U);
    EXPECT_EQ(occupied[index(3, 3, 3)], 1);
    EXPECT_EQ(occupied[index(0, 3, 3)], 0);
    EXPECT_EQ(occupied[index(6, 3, 3)], 0);
}

TEST(VisibilityOccupancyCleanupTest, ClosingPreservesObservedEmptyTunnel)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    const auto index = [&dimensions](int x, int y, int z)
    {
        return static_cast<std::size_t>(
            x + dimensions[0] * (y + dimensions[1] * z));
    };
    std::vector<std::uint8_t> occupied(343, 0);
    for (int z = 1; z <= 5; ++z)
    {
        for (int y = 1; y <= 5; ++y)
        {
            for (int x = 1; x <= 5; ++x)
            {
                occupied[index(x, y, z)] = 1;
            }
        }
    }
    std::vector<std::uint8_t> protected_empty(occupied.size(), 0);
    for (int x = 1; x <= 5; ++x)
    {
        occupied[index(x, 3, 3)] = 0;
        protected_empty[index(x, 3, 3)] = 1;
    }
    const std::vector<std::uint8_t> original = occupied;

    xjw::mesh::detail::closeVisibilityOccupancySixConnected(
        dimensions, 2, &occupied, &protected_empty);

    for (int x = 1; x <= 5; ++x)
    {
        EXPECT_EQ(occupied[index(x, 3, 3)], 0);
    }
    for (std::size_t sample = 0; sample < occupied.size(); ++sample)
    {
        if (original[sample] != 0)
        {
            EXPECT_NE(occupied[sample], 0);
        }
    }
}

TEST(VisibilityOccupancyCleanupTest, ClosingRadiusGrowsWithIterations)
{
    const std::array<int, 3> dimensions{11, 11, 11};
    const auto index = [&dimensions](int x, int y, int z)
    {
        return static_cast<std::size_t>(
            x + dimensions[0] * (y + dimensions[1] * z));
    };
    std::vector<std::uint8_t> occupied(1331, 0);
    for (int z = 2; z <= 8; ++z)
    {
        for (int y = 2; y <= 8; ++y)
        {
            for (int x = 2; x <= 8; ++x)
            {
                occupied[index(x, y, z)] = 1;
            }
        }
    }
    for (int x = 0; x <= 8; ++x)
    {
        for (int z = 4; z <= 6; ++z)
        {
            for (int y = 4; y <= 6; ++y)
            {
                occupied[index(x, y, z)] = 0;
            }
        }
    }

    auto radius_one = occupied;
    xjw::mesh::detail::closeVisibilityOccupancySixConnected(
        dimensions, 1, &radius_one);
    EXPECT_EQ(radius_one[index(5, 5, 5)], 0);

    auto radius_two = occupied;
    xjw::mesh::detail::closeVisibilityOccupancySixConnected(
        dimensions, 2, &radius_two);
    EXPECT_EQ(radius_two[index(5, 5, 5)], 1);
}

TEST(VisibilityOccupancyCleanupTest, FillsOnlyUnprotectedInteriorBubbles)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    const std::size_t center = occupancyIndex(dimensions, 3, 3, 3);
    auto unprotected = solidOccupancyBlock(dimensions);
    unprotected[center] = 0;

    const std::uint64_t filled =
        xjw::mesh::detail::fillInteriorVisibilityEmptyBubbles(
            dimensions, &unprotected);

    EXPECT_EQ(filled, 1U);
    EXPECT_EQ(unprotected[center], 1);

    auto protected_occupancy = solidOccupancyBlock(dimensions);
    protected_occupancy[center] = 0;
    std::vector<std::uint8_t> protected_empty(
        protected_occupancy.size(), 0);
    protected_empty[center] = 1;

    const std::uint64_t protected_filled =
        xjw::mesh::detail::fillInteriorVisibilityEmptyBubbles(
            dimensions,
            &protected_occupancy,
            &protected_empty);

    EXPECT_EQ(protected_filled, 0U);
    EXPECT_EQ(protected_occupancy[center], 0);
}

TEST(VisibilityOccupancyHandleRepairTest, RepairsUnsupportedTunnel)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    const auto occupied = occupancyWithStraightTunnel(dimensions);
    const auto proposal = solidOccupancyBlock(dimensions);
    const std::vector<std::uint8_t> protected_empty(occupied.size(), 0);
    const int euler_before =
        xjw::mesh::VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
            dimensions, occupied);

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(euler_before, 0);
    EXPECT_EQ(result.statistics.bodyEulerBefore, 0);
    EXPECT_EQ(result.statistics.bodyEulerAfter, 1);
    EXPECT_EQ(result.statistics.acceptedCandidateCount, 1);
    EXPECT_EQ(result.statistics.filledSampleCount, 5U);
    for (std::size_t index = 0; index < occupied.size(); ++index)
    {
        if (occupied[index] != 0)
        {
            EXPECT_NE(result.occupied[index], 0);
        }
    }
}

TEST(VisibilityOccupancyHandleRepairTest, PreservesProtectedTunnel)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    const auto occupied = occupancyWithStraightTunnel(dimensions);
    const auto proposal = solidOccupancyBlock(dimensions);
    std::vector<std::uint8_t> protected_empty(occupied.size(), 0);
    for (int x = 1; x <= 5; ++x)
    {
        protected_empty[occupancyIndex(dimensions, x, 3, 3)] = 1;
    }

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.occupied, occupied);
    EXPECT_EQ(result.statistics.acceptedCandidateCount, 0);
    EXPECT_EQ(result.statistics.rejectedProtectedCandidateCount, 1);
    EXPECT_EQ(result.statistics.bodyEulerAfter, 0);
}

TEST(VisibilityOccupancyHandleRepairTest, RejectsExteriorConcavity)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    auto occupied = solidOccupancyBlock(dimensions);
    const auto proposal = occupied;
    occupied[occupancyIndex(dimensions, 1, 3, 3)] = 0;
    const std::vector<std::uint8_t> protected_empty(occupied.size(), 0);

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.occupied, occupied);
    EXPECT_EQ(result.statistics.acceptedCandidateCount, 0);
    EXPECT_EQ(result.statistics.rejectedTopologyCandidateCount, 1);
    EXPECT_EQ(result.statistics.bodyEulerBefore, 1);
    EXPECT_EQ(result.statistics.bodyEulerAfter, 1);
}

TEST(
    VisibilityOccupancyHandleRepairTest,
    RejectsEulerIncreaseThatMergesOccupiedComponents)
{
    const std::array<int, 3> dimensions{5, 5, 5};
    std::vector<std::uint8_t> occupied(125, 0);
    const std::array<std::array<int, 3>, 14> full_samples{{
        {1, 1, 3}, {1, 2, 1}, {1, 2, 3}, {1, 3, 3},
        {2, 1, 1}, {2, 1, 3}, {2, 3, 1}, {2, 3, 3},
        {3, 1, 1}, {3, 1, 3}, {3, 2, 1}, {3, 2, 3},
        {3, 3, 2}, {3, 3, 3}}};
    for (const auto &sample : full_samples)
    {
        occupied[occupancyIndex(
            dimensions, sample[0], sample[1], sample[2])] = 1;
    }
    auto proposal = occupied;
    const std::size_t improving =
        occupancyIndex(dimensions, 2, 2, 1);
    const std::size_t harmful =
        occupancyIndex(dimensions, 3, 1, 2);
    proposal[improving] = 1;
    proposal[harmful] = 1;
    const std::vector<std::uint8_t> protected_empty(occupied.size(), 0);

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.statistics.candidateComponentCount, 2);
    EXPECT_EQ(result.statistics.acceptedCandidateCount, 0);
    EXPECT_GT(result.statistics.rejectedTopologyCandidateCount, 0);
    EXPECT_EQ(result.statistics.bodyEulerBefore, -1);
    EXPECT_EQ(result.statistics.bodyEulerAfter, -1);
    EXPECT_EQ(result.occupied[improving], 0);
    EXPECT_EQ(result.occupied[harmful], 0);
}

TEST(VisibilityOccupancyHandleRepairTest,
     ExtractsImprovingSubsetFromNeutralConnectedProposal)
{
    const std::array<int, 3> dimensions{8, 8, 8};
    std::vector<std::uint8_t> occupied(512, 0);
    const std::array<std::array<int, 3>, 35> full_samples{{
        {1, 3, 1}, {1, 4, 1}, {1, 4, 2}, {1, 4, 3},
        {2, 1, 4}, {2, 2, 3}, {2, 2, 4}, {2, 3, 3},
        {2, 3, 4}, {2, 4, 3}, {2, 4, 4}, {3, 4, 2},
        {3, 4, 3}, {3, 4, 4}, {3, 5, 2}, {3, 6, 4},
        {4, 3, 4}, {4, 4, 4}, {4, 4, 5}, {4, 4, 6},
        {4, 5, 4}, {4, 5, 5}, {4, 5, 6}, {4, 6, 1},
        {4, 6, 2}, {4, 6, 3}, {4, 6, 4}, {4, 6, 5},
        {4, 6, 6}, {5, 3, 3}, {5, 3, 4}, {5, 4, 4},
        {5, 4, 5}, {5, 5, 4}, {5, 6, 4}}};
    for (const auto &sample : full_samples)
    {
        occupied[occupancyIndex(
            dimensions, sample[0], sample[1], sample[2])] = 1;
    }
    auto proposal = occupied;
    const std::array<std::array<int, 3>, 9> proposal_samples{{
        {1, 5, 6}, {2, 5, 4}, {2, 5, 5}, {2, 5, 6},
        {3, 4, 6}, {3, 5, 3}, {3, 5, 4}, {3, 5, 6},
        {3, 6, 6}}};
    for (const auto &sample : proposal_samples)
    {
        proposal[occupancyIndex(
            dimensions, sample[0], sample[1], sample[2])] = 1;
    }
    const std::vector<std::uint8_t> protected_empty(occupied.size(), 0);

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.statistics.candidateComponentCount, 1);
    EXPECT_EQ(result.statistics.acceptedCandidateCount, 1);
    EXPECT_EQ(result.statistics.acceptedSubsetCandidateCount, 1);
    EXPECT_EQ(result.statistics.acceptedPlateauSubsetCandidateCount, 0);
    EXPECT_EQ(result.statistics.filledSampleCount, 1U);
    EXPECT_EQ(result.statistics.bodyEulerBefore, 0);
    EXPECT_EQ(result.statistics.bodyEulerAfter, 1);
}

TEST(VisibilityOccupancyHandleRepairTest,
     GrowsNeutralVoxelsIntoTopologyImprovingWideTunnelCut)
{
    const std::array<int, 3> dimensions{9, 9, 9};
    auto occupied = solidOccupancyBlock(dimensions);
    for (int x = 1; x <= 7; ++x)
    {
        for (int y = 4; y <= 5; ++y)
        {
            for (int z = 4; z <= 5; ++z)
            {
                occupied[occupancyIndex(dimensions, x, y, z)] = 0;
            }
        }
    }
    auto proposal = occupied;
    for (int y = 4; y <= 5; ++y)
    {
        for (int z = 4; z <= 5; ++z)
        {
            proposal[occupancyIndex(dimensions, 4, y, z)] = 1;
        }
    }
    const std::vector<std::uint8_t> protected_empty(occupied.size(), 0);
    xjw::mesh::VisibilityOccupancyHandleRepairOptions options;
    options.maximumCandidateSampleCount = 1;
    options.maximumSubsetSampleCount = 8;
    options.maximumSubsetSeedCount = 16;

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty, options);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.statistics.acceptedCandidateCount, 1);
    EXPECT_EQ(result.statistics.acceptedSubsetCandidateCount, 1);
    EXPECT_EQ(result.statistics.acceptedPlateauSubsetCandidateCount, 1);
    EXPECT_GT(result.statistics.attemptedSubsetSeedCount, 0);
    EXPECT_EQ(result.statistics.filledSampleCount, 4U);
    EXPECT_EQ(result.statistics.bodyEulerBefore, 0);
    EXPECT_EQ(result.statistics.bodyEulerAfter, 1);
}

TEST(VisibilityOccupancyHandleRepairTest, PreservesProtectedExteriorReachability)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    auto occupied = solidOccupancyBlock(dimensions);
    for (int x = 0; x <= 3; ++x)
    {
        occupied[occupancyIndex(dimensions, x, 3, 3)] = 0;
    }
    auto proposal = occupied;
    proposal[occupancyIndex(dimensions, 1, 3, 3)] = 1;
    std::vector<std::uint8_t> protected_empty(occupied.size(), 0);
    protected_empty[occupancyIndex(dimensions, 3, 3, 3)] = 1;

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.occupied, occupied);
    EXPECT_EQ(result.statistics.rejectedTopologyCandidateCount, 0);
    EXPECT_EQ(
        result.statistics.rejectedProtectedReachabilityCandidateCount,
        1);
    EXPECT_EQ(result.statistics.protectedExteriorSampleCountBefore, 1U);
    EXPECT_EQ(result.statistics.protectedExteriorSampleCountAfter, 1U);
}

TEST(VisibilityOccupancySurfaceBuilderTest,
     ParallelProjectionMatchesSingleThreadResult)
{
    const std::vector<SyntheticFrame> frames = makeSphereFrames();
    auto single_options = makeOptions();
    single_options.workerCount = 1;
    const auto single = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        single_options);
    auto parallel_options = single_options;
    parallel_options.workerCount = 4;
    const auto parallel = xjw::mesh::VisibilityOccupancySurfaceBuilder::build(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        makeViews(frames),
        parallel_options);

    ASSERT_TRUE(single.ok) << single.error;
    ASSERT_TRUE(parallel.ok) << parallel.error;
    EXPECT_EQ(single.occupied, parallel.occupied);
    EXPECT_EQ(single.signedDistanceSamples, parallel.signedDistanceSamples);
    EXPECT_EQ(single.statistics.depthEmptyVoteCount,
              parallel.statistics.depthEmptyVoteCount);
    EXPECT_EQ(single.statistics.depthFullVoteCount,
              parallel.statistics.depthFullVoteCount);
    EXPECT_EQ(single.statistics.effectiveWorkerCount, 1);
    EXPECT_EQ(parallel.statistics.effectiveWorkerCount, 4);
}

TEST(VisibilityOccupancyHandleRepairTest,
     SealsUnprotectedExteriorPocketWhenTopologyImproves)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    auto occupied = solidOccupancyBlock(dimensions);
    for (int x = 0; x <= 3; ++x)
    {
        occupied[occupancyIndex(dimensions, x, 3, 3)] = 0;
    }
    auto proposal = occupied;
    const std::size_t seal = occupancyIndex(dimensions, 1, 3, 3);
    proposal[seal] = 1;
    const std::vector<std::uint8_t> protected_empty(occupied.size(), 0);

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.occupied[seal], 1);
    EXPECT_EQ(result.statistics.acceptedCandidateCount, 1);
    EXPECT_EQ(result.statistics.rejectedProtectedReachabilityCandidateCount, 0);
    EXPECT_GT(
        result.statistics.bodyEulerAfter,
        result.statistics.bodyEulerBefore);
}

TEST(VisibilityOccupancyHandleRepairTest, RejectsProposalThatRemovesFullSample)
{
    const std::array<int, 3> dimensions{5, 5, 5};
    const auto occupied = solidOccupancyBlock(dimensions);
    auto proposal = occupied;
    proposal[occupancyIndex(dimensions, 2, 2, 2)] = 0;
    const std::vector<std::uint8_t> protected_empty(occupied.size(), 0);

    const auto result =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty);

    EXPECT_FALSE(result.ok);
    EXPECT_FALSE(result.cancelled);
    EXPECT_EQ(result.occupied, occupied);
    EXPECT_EQ(
        result.error,
        "visibility occupancy closing proposal removes full samples");
}

TEST(VisibilityOccupancyHandleRepairTest, IsDeterministicAndCancellationIsAtomic)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    const auto occupied = occupancyWithStraightTunnel(dimensions);
    const auto proposal = solidOccupancyBlock(dimensions);
    const std::vector<std::uint8_t> protected_empty(occupied.size(), 0);

    const auto first = xjw::mesh::VisibilityOccupancyHandleRepair::repair(
        dimensions, occupied, proposal, protected_empty);
    const auto second = xjw::mesh::VisibilityOccupancyHandleRepair::repair(
        dimensions, occupied, proposal, protected_empty);
    ASSERT_TRUE(first.ok);
    ASSERT_TRUE(second.ok);
    EXPECT_EQ(first.occupied, second.occupied);
    EXPECT_EQ(
        first.statistics.acceptedCandidateCount,
        second.statistics.acceptedCandidateCount);
    EXPECT_EQ(
        first.statistics.bodyEulerAfter,
        second.statistics.bodyEulerAfter);

    xjw::mesh::VisibilityOccupancyHandleRepairOptions options;
    options.isCancelled = []()
    {
        return true;
    };
    const auto cancelled =
        xjw::mesh::VisibilityOccupancyHandleRepair::repair(
            dimensions, occupied, proposal, protected_empty, options);
    EXPECT_FALSE(cancelled.ok);
    EXPECT_TRUE(cancelled.cancelled);
    EXPECT_EQ(cancelled.occupied, occupied);
}

TEST(VisibilityOccupancyTsdfCompletionTest, ClosesSupportAndBlendsAgreedSurface)
{
    xjw::mesh::DepthTsdfLayout layout;
    layout.ok = true;
    layout.boundsMin = {-1.0f, -1.0f, -1.0f};
    layout.boundsMax = {1.0f, 1.0f, 1.0f};
    layout.cells = {4, 4, 4};
    layout.voxelSize = {0.5f, 0.5f, 0.5f};
    layout.sampleCount = 125;

    xjw::mesh::VisibilityOccupancyResult occupancy;
    occupancy.ok = true;
    occupancy.sampleDimensions = {3, 3, 3};
    occupancy.occupied.assign(27, 0);
    occupancy.signedDistanceSamples.assign(27, 1.0f);
    const std::size_t coarse_center = occupancy.index(1, 1, 1);
    occupancy.occupied[coarse_center] = 1;
    occupancy.signedDistanceSamples[coarse_center] = -1.0f;

    std::vector<float> tsdf(layout.sampleCount, 1.0f);
    std::vector<std::uint8_t> supported(layout.sampleCount, 0);
    const std::size_t center = 2U + 5U * (2U + 5U * 2U);
    tsdf[center] = -0.1f;
    supported[center] = 1;

    const auto statistics =
        xjw::mesh::VisibilityOccupancyTsdfCompletion::apply(
            layout,
            occupancy,
            {},
            &tsdf,
            &supported);

    EXPECT_EQ(statistics.trustedObservationSampleCount, 1U);
    EXPECT_EQ(statistics.blendedSampleCount, 1U);
    EXPECT_LT(tsdf[center], 0.0f);
    EXPECT_GT(tsdf[center], -1.0f / 3.0f);
    EXPECT_EQ(statistics.carrierSignMismatchSampleCount, 0U);
    EXPECT_GT(statistics.recoveredUnsupportedSampleCount, 0U);
    EXPECT_TRUE(std::all_of(
        supported.begin(),
        supported.end(),
        [](std::uint8_t value)
        {
            return value != 0;
    }));
    EXPECT_GT(tsdf.front(), 0.0f);
    EXPECT_GT(statistics.adjustedExactIsoValueSampleCount, 0U);
    EXPECT_TRUE(std::none_of(
        tsdf.begin(),
        tsdf.end(),
        [](float value)
        {
            return std::fabs(value) < 5.0e-5f;
        }));
}

TEST(VisibilityOccupancyTsdfCompletionTest, RejectsOppositeSignObservation)
{
    xjw::mesh::DepthTsdfLayout layout;
    layout.ok = true;
    layout.cells = {2, 2, 2};
    layout.sampleCount = 27;

    xjw::mesh::VisibilityOccupancyResult occupancy;
    occupancy.ok = true;
    occupancy.sampleDimensions = {3, 3, 3};
    occupancy.occupied.assign(27, 1);
    occupancy.signedDistanceSamples.assign(27, -1.0f);

    std::vector<float> tsdf(layout.sampleCount, 0.8f);
    std::vector<std::uint8_t> supported(layout.sampleCount, 1);
    const std::size_t center = 1U + 3U * (1U + 3U);

    const auto statistics =
        xjw::mesh::VisibilityOccupancyTsdfCompletion::apply(
            layout,
            occupancy,
            {},
            &tsdf,
            &supported);

    EXPECT_LT(tsdf[center], 0.0f);
    EXPECT_EQ(statistics.ignoredSignConflictObservationCount, 1U);
    EXPECT_EQ(statistics.overriddenObservedSampleCount, 1U);
    EXPECT_EQ(statistics.carrierSignMismatchSampleCount, 0U);
}

TEST(Mc33IsoSurfaceExtractorTest, FullySupportedFieldNeverRejectsFaces)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }

    const std::array<int, 3> cells{8, 8, 8};
    const int samples = cells[0] + 1;
    std::vector<float> field(
        static_cast<std::size_t>(samples * samples * samples),
        1.0f);
    std::vector<std::uint8_t> support(field.size(), 1);
    const auto index = [samples](int x, int y, int z)
    {
        return static_cast<std::size_t>(
            x + samples * (y + samples * z));
    };
    for (int z = 0; z < samples; ++z)
    {
        for (int y = 0; y < samples; ++y)
        {
            for (int x = 0; x < samples; ++x)
            {
                const float dx = static_cast<float>(x - 4);
                const float dy = static_cast<float>(y - 4);
                const float dz = static_cast<float>(z - 4);
                field[index(x, y, z)] =
                    std::sqrt(dx * dx + dy * dy + dz * dz) - 2.5f;
            }
        }
    }

    xjw::mesh::Mc33IsoSurfaceOptions options;
    options.requireSupportedSignChange = true;
    const auto result = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        cells,
        field,
        support,
        options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    EXPECT_FALSE(result.mesh.empty());
    EXPECT_EQ(result.statistics.supportMaskedSampleCount, 0U);
    EXPECT_EQ(result.statistics.rejectedUnsupportedCellFaceCount, 0U);
}
