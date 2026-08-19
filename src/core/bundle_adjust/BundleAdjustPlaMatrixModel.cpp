#include "BundleAdjustPlaMatrixProblem.h"

#include <plamatrix/ops/statistics.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw::detail::plamatrix_ba
{
namespace
{

std::size_t parameterIndex(BAIntrinsicParameter parameter)
{
    return static_cast<std::size_t>(parameter);
}

double medianOr(std::vector<double> values, double fallback)
{
    return plamatrix::finiteMedian(std::move(values)).value_or(fallback);
}

struct GroupSamples
{
    std::array<std::vector<double>, 9> initial;
    std::array<std::vector<double>, 9> reference;
};

void appendSamples(const FramePinholeCamera& camera,
                   const FramePinholeCamera& reference,
                   GroupSamples* samples)
{
    const auto intrinsics = camera.intrinsics();
    const auto reference_intrinsics = reference.intrinsics();
    const auto distortion = camera.distortion();
    const auto reference_distortion = reference.distortion();
    samples->initial[0].push_back(intrinsics.focalX);
    samples->reference[0].push_back(reference_intrinsics.focalX);
    samples->initial[1].push_back(intrinsics.focalY / intrinsics.focalX);
    samples->reference[1].push_back(
        reference_intrinsics.focalY / reference_intrinsics.focalX);
    samples->initial[2].push_back(intrinsics.principalX - reference_intrinsics.principalX);
    samples->initial[3].push_back(intrinsics.principalY - reference_intrinsics.principalY);
    samples->reference[2].push_back(0.0);
    samples->reference[3].push_back(0.0);
    const std::array<double, 5> initial_distortion{{
        distortion.radialK1,
        distortion.radialK2,
        distortion.radialK3,
        distortion.tangentialP1,
        distortion.tangentialP2}};
    const std::array<double, 5> reference_values{{
        reference_distortion.radialK1,
        reference_distortion.radialK2,
        reference_distortion.radialK3,
        reference_distortion.tangentialP1,
        reference_distortion.tangentialP2}};
    for (std::size_t index = 0; index < initial_distortion.size(); ++index)
    {
        samples->initial[index + 4].push_back(initial_distortion[index]);
        samples->reference[index + 4].push_back(reference_values[index]);
    }
}

} // namespace

bool hasSharedIntrinsics(const BAOptions& options)
{
    for (std::size_t index = 0; index < kBAIntrinsicParameterCount; ++index)
    {
        if (sharedIntrinsicParameterEnabled(
                options, static_cast<BAIntrinsicParameter>(index)))
        {
            return true;
        }
    }
    return false;
}

std::vector<IntrinsicGroupState> initializeIntrinsicGroups(
    const std::vector<FramePinholeCamera>& cameras,
    const BAOptions& options,
    const ActiveProblem& active)
{
    std::vector<IntrinsicGroupState> groups(
        static_cast<std::size_t>(active.intrinsicBlockCount));
    if (groups.empty())
    {
        return groups;
    }
    const auto& references = options.sharedIntrinsicReferenceCameras.empty()
        ? cameras
        : options.sharedIntrinsicReferenceCameras;
    std::vector<GroupSamples> samples(groups.size());
    for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
    {
        appendSamples(cameras[camera_index],
                      references[camera_index],
                      &samples[static_cast<std::size_t>(
                          active.calibrationGroupByCamera[camera_index])]);
    }

    const double low_order_scale = std::max(1.0, options.sharedLowOrderDistortionScale);
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index)
    {
        auto& group = groups[group_index];
        group.lower.fill(-std::numeric_limits<double>::infinity());
        group.upper.fill(std::numeric_limits<double>::infinity());
        for (std::size_t parameter = 0; parameter < 9; ++parameter)
        {
            group.enabled[parameter] = sharedIntrinsicParameterEnabled(
                options, static_cast<BAIntrinsicParameter>(parameter));
        }
        group.focalReference = medianOr(
            samples[group_index].reference[0], 1.0);
        group.aspectReference = medianOr(
            samples[group_index].reference[1], 1.0);
        group.parameters[0] = std::log(std::max(
            1e-12, medianOr(samples[group_index].initial[0], group.focalReference)));
        group.parameters[1] = std::log(std::max(
            1e-12, medianOr(samples[group_index].initial[1], group.aspectReference)));
        group.prior[0] = std::log(group.focalReference);
        group.prior[1] = std::log(group.aspectReference);
        for (std::size_t parameter = 2; parameter < 9; ++parameter)
        {
            group.prior[parameter] = medianOr(
                samples[group_index].reference[parameter], 0.0);
            group.parameters[parameter] = medianOr(
                samples[group_index].initial[parameter], group.prior[parameter]);
        }

        group.lower[0] = std::log(group.focalReference * options.minSharedFocalScale);
        group.upper[0] = std::log(group.focalReference * options.maxSharedFocalScale);
        group.lower[1] = std::log(
            group.aspectReference * options.minSharedFocalAspectScale);
        group.upper[1] = std::log(
            group.aspectReference * options.maxSharedFocalAspectScale);
        const double principal_limit = group.focalReference *
            options.maxSharedPrincipalPointOffsetFraction;
        group.lower[2] = group.lower[3] = -principal_limit;
        group.upper[2] = group.upper[3] = principal_limit;
        const std::array<double, 5> distortion_limits{{
            options.maxSharedRadialK1Abs * low_order_scale,
            options.maxSharedRadialK2Abs,
            options.maxSharedRadialK3Abs,
            options.maxSharedTangentialP1Abs * low_order_scale,
            options.maxSharedTangentialP2Abs * low_order_scale}};
        for (std::size_t index = 0; index < distortion_limits.size(); ++index)
        {
            group.lower[index + 4] = -distortion_limits[index];
            group.upper[index + 4] = distortion_limits[index];
        }

        const double principal_sigma = std::max(
            1e-6, group.focalReference * options.sharedPrincipalPointPriorSigmaFraction);
        const std::array<double, 9> sigma{{
            options.sharedFocalPriorSigma,
            options.sharedFocalAspectPriorSigma,
            principal_sigma,
            principal_sigma,
            options.sharedRadialK1PriorSigma * low_order_scale,
            options.sharedRadialK2PriorSigma,
            options.sharedRadialK3PriorSigma,
            options.sharedTangentialP1PriorSigma * low_order_scale,
            options.sharedTangentialP2PriorSigma * low_order_scale}};
        for (std::size_t parameter = 0; parameter < 9; ++parameter)
        {
            group.parameters[parameter] = std::clamp(
                group.parameters[parameter], group.lower[parameter], group.upper[parameter]);
            group.inverseSigma[parameter] = group.enabled[parameter]
                ? 1.0 / std::max(1e-9, sigma[parameter])
                : 0.0;
        }
    }
    return groups;
}

BAIntrinsicParameterMask activeIntrinsicParameters(
    const BAOptions& options,
    const BAIntrinsicParameterMask& enabled,
    int iteration)
{
    const int total_iteration_budget = std::max(1, options.maxIterations);
    const bool staged = options.stageSharedFocalRefinement &&
        options.sharedFocalWarmupFraction > 0.0 &&
        total_iteration_budget >= 2;
    if (!staged)
    {
        return enabled;
    }
    const int warmup_iterations = std::clamp(
        static_cast<int>(std::lround(
            total_iteration_budget * options.sharedFocalWarmupFraction)),
        1,
        total_iteration_budget - 1);
    if (iteration < warmup_iterations)
    {
        return {};
    }
    const bool has_high_order = enabled[parameterIndex(BAIntrinsicParameter::RadialK2)] ||
        enabled[parameterIndex(BAIntrinsicParameter::RadialK3)] ||
        enabled[parameterIndex(BAIntrinsicParameter::TangentialP1)] ||
        enabled[parameterIndex(BAIntrinsicParameter::TangentialP2)];
    const int refinement_iterations = total_iteration_budget - warmup_iterations;
    const bool run_low_order_stage = has_high_order && refinement_iterations >= 6;
    const int low_order_iterations = run_low_order_stage
        ? std::max(2, refinement_iterations / 2)
        : 0;
    if (run_low_order_stage &&
        iteration < warmup_iterations + low_order_iterations)
    {
        auto low_order = enabled;
        low_order[parameterIndex(BAIntrinsicParameter::RadialK2)] = false;
        low_order[parameterIndex(BAIntrinsicParameter::RadialK3)] = false;
        low_order[parameterIndex(BAIntrinsicParameter::TangentialP1)] = false;
        low_order[parameterIndex(BAIntrinsicParameter::TangentialP2)] = false;
        return low_order;
    }
    return enabled;
}

int intrinsicStageCount(const BAOptions& options,
                        const BAIntrinsicParameterMask& enabled)
{
    const int total_iteration_budget = std::max(1, options.maxIterations);
    const bool staged = options.stageSharedFocalRefinement &&
        options.sharedFocalWarmupFraction > 0.0 &&
        total_iteration_budget >= 2;
    if (!staged)
    {
        return 1;
    }
    const int warmup_iterations = std::clamp(
        static_cast<int>(std::lround(
            total_iteration_budget * options.sharedFocalWarmupFraction)),
        1,
        total_iteration_budget - 1);
    const int refinement_iterations = total_iteration_budget - warmup_iterations;
    const bool has_high_order = enabled[parameterIndex(BAIntrinsicParameter::RadialK2)] ||
        enabled[parameterIndex(BAIntrinsicParameter::RadialK3)] ||
        enabled[parameterIndex(BAIntrinsicParameter::TangentialP1)] ||
        enabled[parameterIndex(BAIntrinsicParameter::TangentialP2)];
    return has_high_order && refinement_iterations >= 6 ? 3 : 2;
}

void applyIntrinsicStep(const ActiveProblem& active,
                        const std::vector<double>& primary_step,
                        std::vector<IntrinsicGroupState>* groups)
{
    for (std::size_t group_index = 0; group_index < groups->size(); ++group_index)
    {
        auto& group = (*groups)[group_index];
        const int block = active.cameraBlockCount + static_cast<int>(group_index);
        for (std::size_t parameter = 0; parameter < 9; ++parameter)
        {
            if (!group.enabled[parameter])
            {
                continue;
            }
            group.parameters[parameter] = std::clamp(
                group.parameters[parameter] +
                    primary_step[static_cast<std::size_t>(
                        block * kPrimaryBlockSize + static_cast<int>(parameter))],
                group.lower[parameter],
                group.upper[parameter]);
        }
    }
}

void publishIntrinsics(const std::vector<FramePinholeCamera>& input_cameras,
                       const BAOptions& options,
                       const ActiveProblem& active,
                       const std::vector<IntrinsicGroupState>& groups,
                       BAResult* result)
{
    if (groups.empty())
    {
        return;
    }
    const auto& references = options.sharedIntrinsicReferenceCameras.empty()
        ? input_cameras
        : options.sharedIntrinsicReferenceCameras;
    double focal_scale_sum = 0.0;
    double aspect_scale_sum = 0.0;
    std::array<double, 7> remaining_sums{};
    for (std::size_t camera_index = 0; camera_index < result->refinedCameras.size(); ++camera_index)
    {
        auto& camera = result->refinedCameras[camera_index];
        const auto source = input_cameras[camera_index].intrinsics();
        const auto reference = references[camera_index].intrinsics();
        const auto& group = groups[static_cast<std::size_t>(
            active.calibrationGroupByCamera[camera_index])];
        const double focal = group.enabled[0] ? std::exp(group.parameters[0]) : source.focalX;
        const double source_aspect = source.focalY / source.focalX;
        const double aspect = group.enabled[1] ? std::exp(group.parameters[1]) : source_aspect;
        const double principal_x = group.enabled[2]
            ? reference.principalX + group.parameters[2]
            : source.principalX;
        const double principal_y = group.enabled[3]
            ? reference.principalY + group.parameters[3]
            : source.principalY;
        camera.setIntrinsics(focal, focal * aspect, principal_x, principal_y);
        const auto source_distortion = input_cameras[camera_index].distortion();
        auto distortion = source_distortion;
        double* distortion_values[5] = {
            &distortion.radialK1, &distortion.radialK2, &distortion.radialK3,
            &distortion.tangentialP1, &distortion.tangentialP2};
        for (std::size_t index = 0; index < 5; ++index)
        {
            if (group.enabled[index + 4])
            {
                *distortion_values[index] = group.parameters[index + 4];
            }
        }
        camera.setDistortion(distortion);
        focal_scale_sum += focal / group.focalReference;
        aspect_scale_sum += aspect / group.aspectReference;
        remaining_sums[0] += principal_x - reference.principalX;
        remaining_sums[1] += principal_y - reference.principalY;
        remaining_sums[2] += distortion.radialK1;
        remaining_sums[3] += distortion.radialK2;
        remaining_sums[4] += distortion.radialK3;
        remaining_sums[5] += distortion.tangentialP1;
        remaining_sums[6] += distortion.tangentialP2;
        const bool changed =
            std::abs(focal - source.focalX) > 1e-8 * std::max(1.0, std::abs(source.focalX)) ||
            std::abs(focal * aspect - source.focalY) >
                1e-8 * std::max(1.0, std::abs(source.focalY)) ||
            std::abs(principal_x - source.principalX) > 1e-8 ||
            std::abs(principal_y - source.principalY) > 1e-8 ||
            std::abs(distortion.radialK1 - source_distortion.radialK1) > 1e-10 ||
            std::abs(distortion.radialK2 - source_distortion.radialK2) > 1e-10 ||
            std::abs(distortion.radialK3 - source_distortion.radialK3) > 1e-10 ||
            std::abs(distortion.tangentialP1 - source_distortion.tangentialP1) > 1e-10 ||
            std::abs(distortion.tangentialP2 - source_distortion.tangentialP2) > 1e-10;
        if (changed)
        {
            ++result->refinedIntrinsicCount;
        }
    }
    const double count = static_cast<double>(result->refinedCameras.size());
    result->refinedCalibrationGroupCount = static_cast<int>(groups.size());
    result->refinedSharedFocalScale = focal_scale_sum / count;
    result->refinedSharedFocalAspectScale = aspect_scale_sum / count;
    result->refinedSharedPrincipalOffsetX = remaining_sums[0] / count;
    result->refinedSharedPrincipalOffsetY = remaining_sums[1] / count;
    result->refinedSharedRadialK1 = remaining_sums[2] / count;
    result->refinedSharedRadialK2 = remaining_sums[3] / count;
    result->refinedSharedRadialK3 = remaining_sums[4] / count;
    result->refinedSharedTangentialP1 = remaining_sums[5] / count;
    result->refinedSharedTangentialP2 = remaining_sums[6] / count;
}

} // namespace xjw::detail::plamatrix_ba
