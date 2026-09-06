#include "ReferenceResectionSolver.h"

#include "ReferenceP3p.h"
#include "ReferencePoseRefiner.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace xjw
{
    namespace
    {

        cv::Matx33d matrixFromArray(const std::array<double, 9>& values)
        {
            return {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]};
        }

        std::array<double, 9> arrayFromMatrix(const cv::Matx33d& matrix)
        {
            return {{matrix(0, 0),
                     matrix(0, 1),
                     matrix(0, 2),
                     matrix(1, 0),
                     matrix(1, 1),
                     matrix(1, 2),
                     matrix(2, 0),
                     matrix(2, 1),
                     matrix(2, 2)}};
        }

        cv::Vec3d vectorFromArray(const std::array<double, 3>& values)
        {
            return {values[0], values[1], values[2]};
        }

        bool project(const FramePinholeCamera& camera,
                     const ReferenceWorldToCameraPose& pose,
                     const std::array<double, 3>& world,
                     cv::Vec2d* pixel,
                     double* positiveDepth = nullptr)
        {
            if (!pixel)
            {
                return false;
            }
            const cv::Matx33d rotation = matrixFromArray(pose.rotation);
            const cv::Vec3d translation = vectorFromArray(pose.translation);
            const cv::Vec3d local = rotation * vectorFromArray(world) + translation;
            if (positiveDepth)
            {
                *positiveDepth = local[2];
            }
            if (!(local[2] > 1e-9))
            {
                return false;
            }
            const double inverse_z = 1.0 / local[2];
            const double x = local[0] * inverse_z;
            const double y = local[1] * inverse_z;
            const double r2 = x * x + y * y;
            const FramePinholeCamera::Distortion distortion = camera.distortion();
            const double radial =
                1.0 + distortion.radialK1 * r2 + distortion.radialK2 * r2 * r2 + distortion.radialK3 * r2 * r2 * r2;
            const double distorted_x =
                x * radial + 2.0 * distortion.tangentialP1 * x * y + distortion.tangentialP2 * (r2 + 2.0 * x * x);
            const double distorted_y =
                y * radial + distortion.tangentialP1 * (r2 + 2.0 * y * y) + 2.0 * distortion.tangentialP2 * x * y;
            (*pixel)[0] = camera.focalX() * distorted_x + camera.principalX();
            (*pixel)[1] = camera.focalY() * distorted_y + camera.principalY();
            return std::isfinite((*pixel)[0]) && std::isfinite((*pixel)[1]);
        }

        double squaredReprojectionError(const FramePinholeCamera& camera,
                                        const ReferenceWorldToCameraPose& pose,
                                        const std::array<double, 3>& world,
                                        const std::array<double, 2>& observed)
        {
            cv::Vec2d predicted;
            if (!project(camera, pose, world, &predicted))
            {
                return std::numeric_limits<double>::infinity();
            }
            const double dx = predicted[0] - observed[0];
            const double dy = predicted[1] - observed[1];
            return dx * dx + dy * dy;
        }

        std::array<double, 9> restoreOriginalCameraAxes(const std::array<double, 9>& normalizedCameraToWorld,
                                                        const FramePinholeCamera& originalCamera)
        {
            const double z_sign = originalCamera.depthAxisFlipped() ? -1.0 : 1.0;
            const cv::Matx33d axis(z_sign * static_cast<double>(originalCamera.uAxisSign()),
                                   0.0,
                                   0.0,
                                   0.0,
                                   z_sign * static_cast<double>(originalCamera.vAxisSign()),
                                   0.0,
                                   0.0,
                                   0.0,
                                   z_sign);
            return arrayFromMatrix(matrixFromArray(normalizedCameraToWorld) * axis);
        }

    } // namespace

    ReferenceResectionResult solveReferenceResection(const std::vector<std::array<double, 3>>& worldPoints,
                                                     const std::vector<std::array<double, 2>>& imagePoints,
                                                     const FramePinholeCamera& camera,
                                                     double resectionThresholdPixels)
    {
        ReferenceResectionResult result;
        const std::size_t count = worldPoints.size();
        if (count < 4 || count != imagePoints.size() || !(resectionThresholdPixels > 0.0) ||
            !std::isfinite(resectionThresholdPixels))
        {
            return result;
        }

        const FramePinholeCamera normalized_camera = camera.normalizedForPositiveDepth();
        std::vector<std::array<double, 3>> bearing_vectors(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            double normalized[2]{};
            if (!normalized_camera.undistortPixel(imagePoints[index].data(), normalized))
            {
                return result;
            }
            bearing_vectors[index] = {{normalized[0], normalized[1], 1.0}};
        }

        constexpr std::size_t level_count = 10;
        std::array<double, level_count> thresholds{};
        std::array<double, level_count> squared_thresholds{};
        thresholds[0] = resectionThresholdPixels;
        for (std::size_t level = 1; level < level_count; ++level)
        {
            thresholds[level] = thresholds[level - 1] * 0.75;
        }
        for (std::size_t level = 0; level < level_count; ++level)
        {
            squared_thresholds[level] = thresholds[level] * thresholds[level];
        }

        std::array<std::size_t, level_count> best_counts{};
        std::array<ReferenceWorldToCameraPose, level_count> best_poses{};
        std::uint64_t random_state = 1;
        constexpr int ransac_iterations = 500;
        for (int iteration = 0; iteration < ransac_iterations; ++iteration)
        {
            std::array<std::size_t, 3> selected{};
            bool sample_ok = true;
            for (std::size_t slot = 0; slot < selected.size(); ++slot)
            {
                bool distinct = false;
                for (int attempt = 0; attempt < 1000 && !distinct; ++attempt)
                {
                    random_state = (16807ULL * random_state) % 0x7FFFFFFFULL;
                    const std::size_t candidate = static_cast<std::size_t>(random_state % count);
                    distinct =
                        std::find(selected.begin(), selected.begin() + static_cast<std::ptrdiff_t>(slot), candidate) ==
                        selected.begin() + static_cast<std::ptrdiff_t>(slot);
                    if (distinct)
                    {
                        selected[slot] = candidate;
                    }
                }
                sample_ok = sample_ok && distinct;
            }
            if (!sample_ok)
            {
                break;
            }

            std::array<std::array<double, 3>, 3> sample_world{};
            std::array<std::array<double, 3>, 3> sample_bearings{};
            for (std::size_t slot = 0; slot < selected.size(); ++slot)
            {
                sample_world[slot] = worldPoints[selected[slot]];
                sample_bearings[slot] = bearing_vectors[selected[slot]];
            }
            for (const ReferenceWorldToCameraPose& candidate : solveReferenceP3p(sample_world, sample_bearings))
            {
                std::array<std::size_t, level_count> candidate_counts{};
                for (std::size_t index = 0; index < count; ++index)
                {
                    const double error =
                        squaredReprojectionError(normalized_camera, candidate, worldPoints[index], imagePoints[index]);
                    for (std::size_t level = 0; level < level_count; ++level)
                    {
                        candidate_counts[level] += error < squared_thresholds[level];
                    }
                }
                for (std::size_t level = 0; level < level_count; ++level)
                {
                    if (candidate_counts[level] > best_counts[level])
                    {
                        best_counts[level] = candidate_counts[level];
                        best_poses[level] = candidate;
                    }
                }
            }
        }
        result.ransacIterations = ransac_iterations;
        if (best_counts[0] == 0)
        {
            return result;
        }

        std::size_t selected_level = 0;
        if (best_counts[0] > 64)
        {
            while (selected_level + 1 < level_count && best_counts[selected_level + 1] * 2 >= best_counts[0])
            {
                ++selected_level;
            }
        }
        ReferenceWorldToCameraPose pose = best_poses[selected_level];
        std::vector<unsigned char> mask(count, 0);
        const double initial_mask_threshold = thresholds[selected_level];
        for (std::size_t index = 0; index < count; ++index)
        {
            mask[index] = squaredReprojectionError(normalized_camera, pose, worldPoints[index], imagePoints[index]) <
                          initial_mask_threshold;
        }

        const double base_threshold_squared = resectionThresholdPixels * resectionThresholdPixels;
        for (int outer = 0; outer < 5; ++outer)
        {
            std::vector<std::size_t> inliers;
            inliers.reserve(count);
            for (std::size_t index = 0; index < count; ++index)
            {
                if (mask[index])
                {
                    inliers.push_back(index);
                }
            }
            if (inliers.size() < 3)
            {
                return result;
            }
            refineReferencePose(normalized_camera, worldPoints, imagePoints, inliers, &pose, 10);
            std::size_t changed = 0;
            for (std::size_t index = 0; index < count; ++index)
            {
                const unsigned char next =
                    squaredReprojectionError(normalized_camera, pose, worldPoints[index], imagePoints[index]) <
                    base_threshold_squared;
                changed += next != mask[index];
                mask[index] = next;
            }
            if (changed == 0)
            {
                break;
            }
        }

        result.numInliers = static_cast<int>(std::count(mask.begin(), mask.end(), static_cast<unsigned char>(1)));
        result.inlierMask = std::move(mask);
        result.selectedThresholdLevel = static_cast<int>(selected_level);
        if (result.numInliers <= 4)
        {
            return result;
        }

        const cv::Matx33d normalized_world_to_camera = matrixFromArray(pose.rotation);
        const cv::Matx33d normalized_camera_to_world = normalized_world_to_camera.t();
        const cv::Vec3d translation = vectorFromArray(pose.translation);
        const cv::Vec3d center = -(normalized_camera_to_world * translation);
        result.cameraToWorldRotation = restoreOriginalCameraAxes(arrayFromMatrix(normalized_camera_to_world), camera);
        result.cameraCenter = {{center[0], center[1], center[2]}};
        result.success = std::all_of(result.cameraToWorldRotation.begin(),
                                     result.cameraToWorldRotation.end(),
                                     [](double value) { return std::isfinite(value); }) &&
                         std::all_of(result.cameraCenter.begin(),
                                     result.cameraCenter.end(),
                                     [](double value) { return std::isfinite(value); });
        return result;
    }

} // namespace xjw
