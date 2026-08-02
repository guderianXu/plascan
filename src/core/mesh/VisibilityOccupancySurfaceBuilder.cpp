#include "VisibilityOccupancySurfaceBuilder.h"

#include "DepthRayMetric.h"
#include "VisibilityOccupancyDistanceField.h"
#include "VisibilityOccupancyCleanup.h"
#include "VisibilityOccupancyHandleRepair.h"
#include "VisibilityOccupancyWellComposedRepair.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::mesh
{
namespace
{

bool isCancelled(const VisibilityOccupancyOptions &options)
{
    return options.isCancelled && options.isCancelled();
}

std::array<int, 3> makeSampleDimensions(
    const std::array<float, 3> &bounds_min,
    const std::array<float, 3> &bounds_max,
    int resolution)
{
    std::array<float, 3> extents{};
    for (int axis = 0; axis < 3; ++axis)
    {
        extents[axis] = bounds_max[axis] - bounds_min[axis];
    }
    const float maximum_extent =
        std::max({extents[0], extents[1], extents[2]});
    std::array<int, 3> dimensions{};
    for (int axis = 0; axis < 3; ++axis)
    {
        const float ratio = extents[axis] / maximum_extent;
        dimensions[axis] =
            std::max(5, static_cast<int>(std::lround(resolution * ratio)) + 1);
    }
    return dimensions;
}

std::size_t gridIndex(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) *
                static_cast<std::size_t>(dimensions[1]) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(dimensions[0]) +
           static_cast<std::size_t>(x);
}

bool frameImageSize(
    const VisibilityOccupancyFrameView &frame,
    int *width,
    int *height)
{
    const cv::Mat *images[] = {
        frame.depth,
        frame.supportMask,
        frame.depthValidMask,
        frame.confidence};
    for (const cv::Mat *image : images)
    {
        if (image != nullptr && !image->empty())
        {
            *width = image->cols;
            *height = image->rows;
            return *width > 0 && *height > 0;
        }
    }
    return false;
}

bool frameImagesHaveMatchingSize(
    const VisibilityOccupancyFrameView &frame,
    int width,
    int height)
{
    const cv::Mat *images[] = {
        frame.depth,
        frame.supportMask,
        frame.depthValidMask,
        frame.confidence};
    for (const cv::Mat *image : images)
    {
        if (image != nullptr && !image->empty() &&
            (image->cols != width || image->rows != height))
        {
            return false;
        }
    }
    return true;
}

bool maskAllows(const cv::Mat *mask, int row, int column)
{
    return mask == nullptr || mask->empty() ||
           (mask->type() == CV_8UC1 &&
            mask->at<std::uint8_t>(row, column) != 0);
}

float confidenceAt(
    const cv::Mat *confidence,
    int row,
    int column)
{
    if (confidence == nullptr || confidence->empty())
    {
        return 1.0f;
    }
    if (confidence->type() != CV_32FC1)
    {
        return 0.0f;
    }
    const float value = confidence->at<float>(row, column);
    return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f;
}

BinaryGridCapacity weightedCapacity(
    BinaryGridCapacity base,
    float weight)
{
    if (base <= 0 || !(weight > 0.0f) || !std::isfinite(weight))
    {
        return 0;
    }
    const double value =
        static_cast<double>(base) * static_cast<double>(weight);
    return static_cast<BinaryGridCapacity>(std::clamp(
        std::llround(value),
        1LL,
        static_cast<long long>(
            std::numeric_limits<BinaryGridCapacity>::max() / 16)));
}

void addCapacity(
    BinaryGridCapacity addition,
    BinaryGridCapacity *value)
{
    const BinaryGridCapacity limit =
        std::numeric_limits<BinaryGridCapacity>::max() / 16;
    *value = std::min(limit, *value + std::max<BinaryGridCapacity>(0, addition));
}

} // namespace

std::size_t VisibilityOccupancyResult::index(int x, int y, int z) const
{
    return gridIndex(sampleDimensions, x, y, z);
}

VisibilityOccupancyResult VisibilityOccupancySurfaceBuilder::build(
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::vector<VisibilityOccupancyFrameView> &frames,
    const VisibilityOccupancyOptions &options)
{
    VisibilityOccupancyResult result;
    result.boundsMin = boundsMin;
    result.boundsMax = boundsMax;
    const float maximum_extent = std::max(
        {boundsMax[0] - boundsMin[0],
         boundsMax[1] - boundsMin[1],
         boundsMax[2] - boundsMin[2]});
    if (!(maximum_extent > 0.0f) || frames.empty() ||
        options.resolution < 4)
    {
        result.error = "visibility occupancy input is empty or invalid";
        return result;
    }
    const bool has_explicit_dimensions = std::all_of(
        options.sampleDimensions.cbegin(),
        options.sampleDimensions.cend(),
        [](int dimension)
        {
            return dimension >= 5;
        });
    result.sampleDimensions = has_explicit_dimensions
        ? options.sampleDimensions
        : makeSampleDimensions(boundsMin, boundsMax, options.resolution);
    for (const VisibilityOccupancyFrameView &frame : frames)
    {
        if (frame.camera == nullptr || frame.frameWeight <= 0.0f)
        {
            continue;
        }
        int width = 0;
        int height = 0;
        if (frameImageSize(frame, &width, &height) &&
            !frameImagesHaveMatchingSize(frame, width, height))
        {
            result.error =
                "visibility occupancy frame images have inconsistent dimensions";
            return result;
        }
    }

    BinaryGridMinCutProblem problem;
    problem.sizeX = result.sampleDimensions[0];
    problem.sizeY = result.sampleDimensions[1];
    problem.sizeZ = result.sampleDimensions[2];
    const std::size_t node_count = problem.nodeCount();
    result.statistics.sampleCount = node_count;
    problem.sourceCapacities.assign(node_count, 0);
    problem.sinkCapacities.assign(node_count, 0);
    problem.positiveXCapacities.assign(node_count, 0);
    problem.positiveYCapacities.assign(node_count, 0);
    problem.positiveZCapacities.assign(node_count, 0);
    std::vector<std::uint8_t> depth_protected_empty(node_count, 0);
    std::vector<std::uint8_t> silhouette_protected_empty(node_count, 0);
    std::vector<std::uint8_t> depth_empty_view_counts(node_count, 0);
    std::vector<std::uint8_t> depth_full_view_counts(node_count, 0);
    std::vector<std::uint8_t> silhouette_outside_view_counts(node_count, 0);

    for (int z = 0; z < problem.sizeZ; ++z)
    {
        if ((z & 3) == 0 && isCancelled(options))
        {
            result.cancelled = true;
            return result;
        }
        const float wz = boundsMin[2] +
            (boundsMax[2] - boundsMin[2]) *
                static_cast<float>(z) / static_cast<float>(problem.sizeZ - 1);
        for (int y = 0; y < problem.sizeY; ++y)
        {
            const float wy = boundsMin[1] +
                (boundsMax[1] - boundsMin[1]) *
                    static_cast<float>(y) / static_cast<float>(problem.sizeY - 1);
            for (int x = 0; x < problem.sizeX; ++x)
            {
                const std::size_t index = problem.index(x, y, z);
                if (x + 1 < problem.sizeX)
                {
                    problem.positiveXCapacities[index] =
                        options.pairwiseCapacity;
                }
                if (y + 1 < problem.sizeY)
                {
                    problem.positiveYCapacities[index] =
                        options.pairwiseCapacity;
                }
                if (z + 1 < problem.sizeZ)
                {
                    problem.positiveZCapacities[index] =
                        options.pairwiseCapacity;
                }
                const bool boundary = x == 0 || y == 0 || z == 0 ||
                    x + 1 == problem.sizeX || y + 1 == problem.sizeY ||
                    z + 1 == problem.sizeZ;
                if (boundary)
                {
                    problem.sinkCapacities[index] =
                        options.hardBoundaryCapacity;
                    continue;
                }

                const float wx = boundsMin[0] +
                    (boundsMax[0] - boundsMin[0]) *
                        static_cast<float>(x) /
                        static_cast<float>(problem.sizeX - 1);
                const double world[3] = {wx, wy, wz};
                int projected_views = 0;
                int silhouette_inside_views = 0;
                int silhouette_violations = 0;
                int depth_empty_views = 0;
                int depth_full_views = 0;
                for (const VisibilityOccupancyFrameView &frame : frames)
                {
                    if (frame.camera == nullptr || frame.frameWeight <= 0.0f)
                    {
                        continue;
                    }
                    int width = 0;
                    int height = 0;
                    if (!frameImageSize(frame, &width, &height))
                    {
                        continue;
                    }
                    double pixel[2]{};
                    double voxel_depth = 0.0;
                    if (!frame.camera->projectWorldPointWithDepth(
                            world, pixel, voxel_depth))
                    {
                        continue;
                    }
                    const int column = static_cast<int>(std::lround(pixel[0]));
                    const int row = static_cast<int>(std::lround(pixel[1]));
                    if (row < 0 || row >= height ||
                        column < 0 || column >= width)
                    {
                        continue;
                    }
                    ++projected_views;
                    ++result.statistics.projectedViewCount;
                    if (!maskAllows(frame.supportMask, row, column))
                    {
                        ++silhouette_violations;
                        ++result.statistics.silhouetteOutsideVoteCount;
                        addCapacity(
                            weightedCapacity(
                                options.silhouetteEmptyCapacity,
                                frame.frameWeight),
                            &problem.sinkCapacities[index]);
                        continue;
                    }
                    ++silhouette_inside_views;
                    ++result.statistics.silhouetteInsideVoteCount;
                    if (frame.depth == nullptr || frame.depth->empty() ||
                        frame.depth->type() != CV_32FC1 ||
                        !maskAllows(frame.depthValidMask, row, column))
                    {
                        continue;
                    }
                    const float observed_depth =
                        frame.depth->at<float>(row, column);
                    const float confidence =
                        confidenceAt(frame.confidence, row, column);
                    const DepthRayMetricSample ray = DepthRayMetric::evaluate(
                        *frame.camera,
                        {pixel[0], pixel[1]},
                        observed_depth);
                    if (!ray.valid || !(confidence > 0.0f))
                    {
                        continue;
                    }
                    const double footprint = std::max(
                        1.0e-9, ray.worldPixelFootprint);
                    const double signed_distance =
                        (static_cast<double>(observed_depth) - voxel_depth) *
                        ray.rayDistancePerCameraZ;
                    const float vote_weight =
                        frame.frameWeight * confidence;
                    if (signed_distance >
                        options.frontTolerancePixelFootprints * footprint)
                    {
                        ++depth_empty_views;
                        ++result.statistics.depthEmptyVoteCount;
                        addCapacity(
                            weightedCapacity(
                                options.depthEmptyCapacity, vote_weight),
                            &problem.sinkCapacities[index]);
                    }
                    else if (signed_distance >=
                             -options.behindSurfaceBandPixelFootprints *
                                 footprint)
                    {
                        ++depth_full_views;
                        ++result.statistics.depthFullVoteCount;
                        addCapacity(
                            weightedCapacity(
                                options.depthFullCapacity, vote_weight),
                            &problem.sourceCapacities[index]);
                    }
                }
                const bool silhouette_prior_candidate =
                    projected_views >= options.minimumVisibleViews &&
                    silhouette_inside_views >=
                        options.minimumSilhouetteViews &&
                    silhouette_violations <=
                        options.allowedSilhouetteViolations;
                if (silhouette_prior_candidate)
                {
                    ++result.statistics
                          .silhouetteFullPriorCandidateSampleCount;
                    if (depth_full_views >=
                        options.minimumDepthFullViewsForSilhouettePrior)
                    {
                        if (options.silhouetteFullPriorCapacity > 0)
                        {
                            ++result.statistics
                                  .silhouetteFullPriorSampleCount;
                            const int prior_support = std::clamp(
                                silhouette_inside_views -
                                    silhouette_violations,
                                1,
                                16);
                            const BinaryGridCapacity prior_capacity =
                                options.silhouetteFullPriorCapacity *
                                prior_support;
                            addCapacity(
                                prior_capacity,
                                &problem.sourceCapacities[index]);
                            result.statistics
                                .silhouetteFullPriorCapacityTotal +=
                                static_cast<std::uint64_t>(
                                    std::max<BinaryGridCapacity>(
                                        0,
                                        prior_capacity));
                        }
                    }
                    else
                    {
                        ++result.statistics
                              .silhouetteFullPriorRejectedWithoutDepthSupportSampleCount;
                    }
                }
                const bool protect_depth_empty =
                    depth_full_views == 0 &&
                    depth_empty_views >= std::max(
                        1,
                        options.closingMinimumDepthEmptyViewsToProtect);
                const bool protect_silhouette_empty =
                    silhouette_inside_views == 0 &&
                    silhouette_violations >= std::max(
                        options.allowedSilhouetteViolations + 1,
                        std::max(
                            1,
                            options
                                .closingMinimumSilhouetteOutsideViewsToProtect));
                if (protect_depth_empty)
                {
                    depth_protected_empty[index] = 1;
                }
                if (protect_silhouette_empty)
                {
                    silhouette_protected_empty[index] = 1;
                }
                depth_empty_view_counts[index] = static_cast<std::uint8_t>(
                    std::clamp(depth_empty_views, 0, 255));
                depth_full_view_counts[index] = static_cast<std::uint8_t>(
                    std::clamp(depth_full_views, 0, 255));
                silhouette_outside_view_counts[index] =
                    static_cast<std::uint8_t>(
                        std::clamp(silhouette_violations, 0, 255));
            }
        }
    }

    BinaryGridMinCutResult cut =
        BinaryGridMinCutSolver::solve(problem, options.isCancelled);
    if (!cut.solved)
    {
        result.cancelled = cut.cancelled;
        result.error = cut.cancelled
            ? "visibility occupancy min-cut cancelled"
            : cut.error;
        return result;
    }
    result.statistics.minCut = cut.statistics;
    result.occupied.resize(node_count, 0);
    std::vector<std::uint8_t> protected_empty(node_count, 0);
    for (std::size_t index = 0; index < node_count; ++index)
    {
        result.occupied[index] =
            cut.labels[index] == BinaryGridLabel::Full ? 1 : 0;
        result.statistics.fullSampleCountBeforeCleanup +=
            result.occupied[index] != 0;
        if (result.occupied[index] != 0)
        {
            continue;
        }
        const bool protected_by_depth = depth_protected_empty[index] != 0;
        const bool protected_by_silhouette =
            silhouette_protected_empty[index] != 0;
        protected_empty[index] =
            protected_by_depth || protected_by_silhouette;
        result.statistics.closingDepthEmptyProtectedSampleCount +=
            protected_by_depth;
        result.statistics.closingSilhouetteEmptyProtectedSampleCount +=
            protected_by_silhouette;
        result.statistics.closingProtectedEmptySampleCount +=
            protected_empty[index] != 0;
    }
    if (options.retainLargestFullComponent)
    {
        result.statistics.removedFullDustSampleCount =
            detail::retainLargestVisibilityFullComponent(
                result.sampleDimensions, &result.occupied);
    }
    if (options.fillInteriorEmptyBubbles)
    {
        result.statistics.filledInteriorEmptySampleCount =
            detail::fillInteriorVisibilityEmptyBubbles(
                result.sampleDimensions,
                &result.occupied,
                &protected_empty);
    }
    result.statistics.closingPreservedOriginalFullSampleCount =
        static_cast<std::uint64_t>(std::count_if(
            result.occupied.cbegin(),
            result.occupied.cend(),
            [](std::uint8_t value)
            {
                return value != 0;
            }));
    const int closing_iterations =
        std::clamp(options.closingIterations, 0, 8);
    result.statistics.effectiveClosingIterations = closing_iterations;
    bool recorded_initial_handle_euler = false;
    const auto run_handle_repair = [&]() -> bool
    {
        if (closing_iterations <= 0)
        {
            return true;
        }
        VisibilityOccupancyHandleRepairOptions repair_options;
        repair_options.maximumAcceptedCandidateCount = std::max(
            0, options.maximumHandleRepairAcceptedCandidateCount);
        repair_options.maximumCandidateSampleCount = std::max<std::size_t>(
            1, options.maximumHandleRepairCandidateSampleCount);
        repair_options.maximumSubsetSampleCount = std::max<std::size_t>(
            1, options.maximumHandleRepairSubsetSampleCount);
        repair_options.maximumSubsetSeedCount = std::max(
            0, options.maximumHandleRepairSubsetSeedCount);
        repair_options.isCancelled = options.isCancelled;
        const int maximum_passes = std::clamp(
            options.maximumHandleRepairPasses, 1, 16);
        for (int pass = 0; pass < maximum_passes; ++pass)
        {
            std::vector<std::uint8_t> closing_proposal = result.occupied;
            detail::closeVisibilityOccupancySixConnected(
                result.sampleDimensions,
                closing_iterations,
                &closing_proposal,
                &protected_empty);
            for (std::size_t index = 0; index < node_count; ++index)
            {
                if (result.occupied[index] != 0 ||
                    closing_proposal[index] == 0)
                {
                    continue;
                }
                ++result.statistics.closingProposalAddedSampleCount;
                result.statistics
                    .closingProposalDepthEmptyAtLeastTwoSampleCount +=
                    depth_empty_view_counts[index] >= 2;
                result.statistics
                    .closingProposalDepthEmptyAtLeastThreeSampleCount +=
                    depth_empty_view_counts[index] >= 3;
                result.statistics
                    .closingProposalDepthEmptyAtLeastFourSampleCount +=
                    depth_empty_view_counts[index] >= 4;
                result.statistics.closingProposalDepthFullSampleCount +=
                    depth_full_view_counts[index] > 0;
                result.statistics
                    .closingProposalSilhouetteOutsideAtLeastTwoSampleCount +=
                    silhouette_outside_view_counts[index] >= 2;
            }
            const VisibilityOccupancyHandleRepairResult repair =
                VisibilityOccupancyHandleRepair::repair(
                    result.sampleDimensions,
                    result.occupied,
                    closing_proposal,
                    protected_empty,
                    repair_options);
            if (!repair.ok)
            {
                result.cancelled = repair.cancelled;
                result.error = repair.cancelled
                    ? "visibility occupancy handle repair cancelled"
                    : repair.error;
                return false;
            }
            if (!recorded_initial_handle_euler)
            {
                result.statistics.handleRepairBodyEulerBefore =
                    repair.statistics.bodyEulerBefore;
                recorded_initial_handle_euler = true;
            }
            result.occupied = repair.occupied;
            result.statistics.closingChangedSampleCount +=
                repair.statistics.filledSampleCount;
            result.statistics.handleRepairCandidateComponentCount +=
                repair.statistics.candidateComponentCount;
            result.statistics.handleRepairAcceptedCandidateCount +=
                repair.statistics.acceptedCandidateCount;
            result.statistics.handleRepairAcceptedSubsetCandidateCount +=
                repair.statistics.acceptedSubsetCandidateCount;
            result.statistics
                .handleRepairAcceptedPlateauSubsetCandidateCount +=
                repair.statistics.acceptedPlateauSubsetCandidateCount;
            result.statistics.handleRepairAttemptedSubsetSeedCount +=
                repair.statistics.attemptedSubsetSeedCount;
            result.statistics.handleRepairRejectedProtectedCandidateCount +=
                repair.statistics.rejectedProtectedCandidateCount;
            result.statistics.handleRepairRejectedOversizedCandidateCount +=
                repair.statistics.rejectedOversizedCandidateCount;
            result.statistics.handleRepairRejectedTopologyCandidateCount +=
                repair.statistics.rejectedTopologyCandidateCount;
            result.statistics
                .handleRepairRejectedProtectedReachabilityCandidateCount +=
                repair.statistics
                    .rejectedProtectedReachabilityCandidateCount;
            result.statistics.handleRepairBodyEulerAfter =
                repair.statistics.bodyEulerAfter;
            if (repair.statistics.acceptedCandidateCount == 0)
            {
                break;
            }
        }
        return true;
    };

    bool recorded_initial_well_composed_euler = false;
    const auto run_well_composed_repair = [&]() -> bool
    {
        if (!options.repairNonManifoldConfigurations)
        {
            return true;
        }
        VisibilityOccupancyWellComposedRepairOptions repair_options;
        repair_options.maximumPasses = std::clamp(
            options.wellComposedRepairMaximumPasses, 1, 16);
        repair_options.maximumFilledSampleCount = std::max<std::size_t>(
            1, options.wellComposedRepairMaximumFilledSampleCount);
        repair_options.isCancelled = options.isCancelled;
        const VisibilityOccupancyWellComposedRepairResult repair =
            VisibilityOccupancyWellComposedRepair::repair(
                result.sampleDimensions,
                result.occupied,
                protected_empty,
                repair_options);
        if (!repair.ok)
        {
            result.cancelled = repair.cancelled;
            result.error = repair.cancelled
                ? "visibility occupancy well-composed repair cancelled"
                : repair.error;
            return false;
        }
        result.occupied = repair.occupied;
        result.statistics.wellComposedRepairFilledSampleCount +=
            repair.statistics.filledSampleCount;
        result.statistics.wellComposedRepairAcceptedPassCount +=
            repair.statistics.acceptedPassCount;
        if (!recorded_initial_well_composed_euler)
        {
            result.statistics.wellComposedRepairBodyEulerBefore =
                repair.statistics.bodyEulerBefore;
            recorded_initial_well_composed_euler = true;
        }
        result.statistics.wellComposedRepairBodyEulerAfter =
            repair.statistics.bodyEulerAfter;
        result.statistics.wellComposedRepairRemainingEdgeCheckerboardCount =
            repair.statistics.remainingEdgeCheckerboardCount;
        result.statistics
            .wellComposedRepairRemainingVertexOccupiedComponentDefectCount =
            repair.statistics
                .remainingVertexOccupiedComponentDefectCount;
        result.statistics
            .wellComposedRepairRemainingVertexEmptyComponentDefectCount =
            repair.statistics.remainingVertexEmptyComponentDefectCount;
        return true;
    };

    // Handle repair and well-composed repair only add occupied samples. One
    // pass may convert an exterior-connected pocket into an internal cavity;
    // removing that cavity can in turn expose a simpler handle-repair
    // candidate. Iterate this sequence to a fixed point while every observed
    // exterior ray remains protected by the repair transactions.
    constexpr int maximum_topology_cleanup_cycles = 8;
    for (int cycle = 0; cycle < maximum_topology_cleanup_cycles; ++cycle)
    {
        const std::vector<std::uint8_t> before_cycle = result.occupied;
        if (!run_handle_repair() || !run_well_composed_repair())
        {
            return result;
        }
        if (options.fillInteriorEmptyBubbles)
        {
            result.statistics.filledInteriorEmptySampleCount +=
                detail::fillInteriorVisibilityEmptyBubbles(
                    result.sampleDimensions,
                    &result.occupied,
                    &protected_empty);
        }
        if (result.occupied == before_cycle)
        {
            break;
        }
    }
    for (const std::uint8_t value : result.occupied)
    {
        result.statistics.fullSampleCountAfterCleanup += value != 0;
    }
    if (result.statistics.fullSampleCountAfterCleanup == 0)
    {
        result.error = "visibility occupancy cut contains no full samples";
        result.occupied.clear();
        return result;
    }
    if (!options.buildSignedDistanceSamples)
    {
        result.ok = true;
        return result;
    }
    const VisibilityOccupancyDistanceFieldResult distance_field =
        VisibilityOccupancyDistanceField::build(
            result.sampleDimensions,
            result.boundsMin,
            result.boundsMax,
            result.occupied);
    if (!distance_field.ok)
    {
        result.error = distance_field.error;
        result.occupied.clear();
        return result;
    }
    result.signedDistanceSamples = distance_field.signedWorldDistance;
    result.signedDistanceSamplesAreWorldUnits = true;
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
