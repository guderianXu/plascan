#include "ReferenceModelQuality.h"

#include "concurrency/SafeWorkerGroup.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace xjw
{
    namespace
    {

        std::size_t referenceWorkerCount(std::size_t itemCount, int requestedThreadCount)
        {
            return std::min(itemCount, static_cast<std::size_t>(std::max(1, requestedThreadCount)));
        }

        template <typename Function>
        void parallelReferenceChunks(std::size_t itemCount, int requestedThreadCount, Function&& function)
        {
            constexpr std::size_t kChunkSize = 100;
            const std::size_t worker_count = referenceWorkerCount(itemCount, requestedThreadCount);
            if (worker_count <= 1)
            {
                function(0, itemCount, 0);
                return;
            }

            std::atomic<std::size_t> next_item{0};
            common::concurrency::runWorkerGroup(
                worker_count,
                [&](std::size_t worker_index, std::stop_token stop_token)
                {
                    while (!stop_token.stop_requested())
                    {
                        const std::size_t begin = next_item.fetch_add(kChunkSize, std::memory_order_relaxed);
                        if (begin >= itemCount)
                        {
                            break;
                        }
                        function(begin, std::min(itemCount, begin + kChunkSize), worker_index);
                    }
                });
        }

        double observationScale(const FeatureKeypoint& keypoint)
        {
            return std::isfinite(keypoint.scale) && keypoint.scale > 0.0f ? keypoint.scale : 1.0;
        }

        bool observationResidual(const SfmReconstruction& reconstruction,
                                 const ScenePoint3D& point,
                                 const TrackElement& observation,
                                 double* normalizedResidual)
        {
            if (!reconstruction.isRegistered(observation.imageId))
            {
                return false;
            }
            const ImageData& image = reconstruction.image(observation.imageId);
            if (observation.featureIdx >= image.keypoints.size())
            {
                return false;
            }
            double projected[2]{};
            double depth = 0.0;
            if (!reconstruction.camera(observation.imageId)
                     .projectWorldPointWithDepth(point.xyz.data(), projected, depth) ||
                !(depth > 0.0) || !std::isfinite(projected[0]) || !std::isfinite(projected[1]))
            {
                return false;
            }
            const FeatureKeypoint& keypoint = image.keypoints[observation.featureIdx];
            const double dx = projected[0] - static_cast<double>(keypoint.x);
            const double dy = projected[1] - static_cast<double>(keypoint.y);
            *normalizedResidual = std::hypot(dx, dy) / observationScale(keypoint);
            return std::isfinite(*normalizedResidual);
        }

        std::array<double, 3> symmetricEigenvalues(const std::array<double, 9>& matrix)
        {
            const double off_diagonal_squared = matrix[1] * matrix[1] + matrix[2] * matrix[2] + matrix[5] * matrix[5];
            if (off_diagonal_squared == 0.0)
            {
                std::array<double, 3> values{{matrix[0], matrix[4], matrix[8]}};
                std::sort(values.begin(), values.end(), std::greater<>());
                return values;
            }

            const double mean = (matrix[0] + matrix[4] + matrix[8]) / 3.0;
            const double centered_squared = (matrix[0] - mean) * (matrix[0] - mean) +
                                            (matrix[4] - mean) * (matrix[4] - mean) +
                                            (matrix[8] - mean) * (matrix[8] - mean) + 2.0 * off_diagonal_squared;
            const double scale = std::sqrt(centered_squared / 6.0);
            if (!(scale > 0.0) || !std::isfinite(scale))
            {
                return {{0.0, 0.0, 0.0}};
            }

            std::array<double, 9> normalized = matrix;
            normalized[0] -= mean;
            normalized[4] -= mean;
            normalized[8] -= mean;
            for (double& value : normalized)
            {
                value /= scale;
            }
            const double determinant = normalized[0] * (normalized[4] * normalized[8] - normalized[5] * normalized[7]) -
                                       normalized[1] * (normalized[3] * normalized[8] - normalized[5] * normalized[6]) +
                                       normalized[2] * (normalized[3] * normalized[7] - normalized[4] * normalized[6]);
            const double angle = std::acos(std::clamp(determinant * 0.5, -1.0, 1.0)) / 3.0;
            constexpr double twoPiOverThree = 2.0943951023931954923;
            const double largest = mean + 2.0 * scale * std::cos(angle);
            const double smallest = mean + 2.0 * scale * std::cos(angle + twoPiOverThree);
            const double middle = 3.0 * mean - largest - smallest;
            return {{largest, middle, smallest}};
        }

        double pointGeometryAccuracy(const SfmReconstruction& reconstruction, const ScenePoint3D& point)
        {
            std::array<double, 9> information{};
            int valid_observations = 0;

            for (const TrackElement& observation : point.track.elements)
            {
                if (!reconstruction.isRegistered(observation.imageId))
                {
                    continue;
                }
                const ImageData& image = reconstruction.image(observation.imageId);
                if (observation.featureIdx >= image.keypoints.size())
                {
                    continue;
                }
                const FramePinholeCamera camera =
                    reconstruction.camera(observation.imageId).normalizedForPositiveDepth();
                double local[3]{};
                camera.worldToCamera(point.xyz.data(), local);
                if (!(local[2] > 0.0) || !std::isfinite(local[0]) || !std::isfinite(local[1]) ||
                    !std::isfinite(local[2]))
                {
                    continue;
                }

                const FramePinholeCamera::Intrinsics intrinsics = camera.intrinsics();
                const FramePinholeCamera::Distortion distortion = camera.distortion();
                const double x = local[0] / local[2];
                const double y = local[1] / local[2];
                const double radius_squared = x * x + y * y;
                const double radius_fourth = radius_squared * radius_squared;
                const double radius_sixth = radius_fourth * radius_squared;
                const double radial = 1.0 + distortion.radialK1 * radius_squared + distortion.radialK2 * radius_fourth +
                                      distortion.radialK3 * radius_sixth;
                const double radial_derivative = distortion.radialK1 + 2.0 * distortion.radialK2 * radius_squared +
                                                 3.0 * distortion.radialK3 * radius_fourth;
                const double dxx = radial + 2.0 * x * x * radial_derivative + 2.0 * distortion.tangentialP1 * y +
                                   6.0 * distortion.tangentialP2 * x;
                const double dxy = 2.0 * x * y * radial_derivative + 2.0 * distortion.tangentialP1 * x +
                                   2.0 * distortion.tangentialP2 * y;
                const double dyx = dxy;
                const double dyy = radial + 2.0 * y * y * radial_derivative + 6.0 * distortion.tangentialP1 * y +
                                   2.0 * distortion.tangentialP2 * x;
                const double inverse_depth = 1.0 / local[2];
                const std::array<double, 3> local_x{{intrinsics.focalX * inverse_depth * dxx,
                                                     intrinsics.focalX * inverse_depth * dxy,
                                                     -intrinsics.focalX * inverse_depth * (dxx * x + dxy * y)}};
                const std::array<double, 3> local_y{{intrinsics.focalY * inverse_depth * dyx,
                                                     intrinsics.focalY * inverse_depth * dyy,
                                                     -intrinsics.focalY * inverse_depth * (dyx * x + dyy * y)}};
                const std::array<double, 9> rotation = camera.worldToCameraRotation();
                std::array<double, 3> world_x{};
                std::array<double, 3> world_y{};
                for (int axis = 0; axis < 3; ++axis)
                {
                    world_x[axis] =
                        local_x[0] * rotation[axis] + local_x[1] * rotation[3 + axis] + local_x[2] * rotation[6 + axis];
                    world_y[axis] =
                        local_y[0] * rotation[axis] + local_y[1] * rotation[3 + axis] + local_y[2] * rotation[6 + axis];
                }

                const double jacobian_norm =
                    std::sqrt(world_x[0] * world_x[0] + world_x[1] * world_x[1] + world_x[2] * world_x[2] +
                              world_y[0] * world_y[0] + world_y[1] * world_y[1] + world_y[2] * world_y[2]);
                const std::array<double, 3> center = camera.cameraCenter();
                const double dx = point.xyz[0] - center[0];
                const double dy = point.xyz[1] - center[1];
                const double dz = point.xyz[2] - center[2];
                const double rounding_bound = jacobian_norm * std::sqrt(dx * dx + dy * dy + dz * dz) *
                                              static_cast<double>(std::numeric_limits<float>::epsilon());
                if (!(rounding_bound <= observationScale(image.keypoints[observation.featureIdx])))
                {
                    continue;
                }
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        information[static_cast<std::size_t>(row * 3 + column)] +=
                            world_x[row] * world_x[column] + world_y[row] * world_y[column];
                    }
                }
                ++valid_observations;
            }

            if (valid_observations < 2)
            {
                return 0.0;
            }
            const std::array<double, 3> eigenvalues = symmetricEigenvalues(information);
            const double largest = std::max(0.0, eigenvalues[0]);
            const double middle = std::max(0.0, eigenvalues[1]);
            const double smallest = std::max(0.0, eigenvalues[2]);
            const double magnitude = std::sqrt(largest * largest + middle * middle + smallest * smallest);
            return magnitude > 0.0 && std::isfinite(magnitude) ? smallest / magnitude : 0.0;
        }

        double sensorResidualThreshold(const SfmReconstruction& reconstruction, ImageId imageId)
        {
            const FramePinholeCamera& camera = reconstruction.camera(imageId);
            if (camera.imageSize() && camera.imageSize()->samples > 0 && camera.imageSize()->lines > 0)
            {
                return 0.002 * 0.5 * static_cast<double>(camera.imageSize()->samples + camera.imageSize()->lines);
            }
            const ImageData& image = reconstruction.image(imageId);
            double maximum_x = 0.0;
            double maximum_y = 0.0;
            for (const FeatureKeypoint& keypoint : image.keypoints)
            {
                maximum_x = std::max(maximum_x, static_cast<double>(keypoint.x));
                maximum_y = std::max(maximum_y, static_cast<double>(keypoint.y));
            }
            return 0.002 * 0.5 * (std::max(1.0, maximum_x + 1.0) + std::max(1.0, maximum_y + 1.0));
        }

        std::vector<ImageId> registeredObservationImages(const SfmReconstruction& reconstruction,
                                                         const ScenePoint3D& point)
        {
            std::vector<ImageId> image_ids;
            for (const TrackElement& observation : point.track.elements)
            {
                if (reconstruction.isRegistered(observation.imageId))
                {
                    image_ids.push_back(observation.imageId);
                }
            }
            std::sort(image_ids.begin(), image_ids.end());
            image_ids.erase(std::unique(image_ids.begin(), image_ids.end()), image_ids.end());
            return image_ids;
        }

    } // namespace

    bool referenceInitialPairScoreBetter(const ReferenceInitialPairScore& candidate,
                                         const ReferenceInitialPairScore& incumbent)
    {
        if (candidate.alignedCameraCount != incumbent.alignedCameraCount)
        {
            return candidate.alignedCameraCount > incumbent.alignedCameraCount;
        }
        if (candidate.pointCountTier != incumbent.pointCountTier)
        {
            return candidate.pointCountTier;
        }
        if (candidate.accuracyTier != incumbent.accuracyTier)
        {
            return candidate.accuracyTier;
        }
        if (candidate.maximumCameraReprojectionRms != incumbent.maximumCameraReprojectionRms)
        {
            return candidate.maximumCameraReprojectionRms < incumbent.maximumCameraReprojectionRms;
        }
        if (candidate.geometryAccuracy != incumbent.geometryAccuracy)
        {
            return candidate.geometryAccuracy > incumbent.geometryAccuracy;
        }
        if (candidate.pointCount != incumbent.pointCount)
        {
            return candidate.pointCount > incumbent.pointCount;
        }
        return std::tie(candidate.firstImageId, candidate.secondImageId) <
               std::tie(incumbent.firstImageId, incumbent.secondImageId);
    }

    bool referenceInitialPairScoreStable(const ReferenceInitialPairScore& score, int totalImages)
    {
        return score.alignedCameraCount >= std::min(3, std::max(0, totalImages)) && score.pointCount >= 100 &&
               score.maximumCameraReprojectionRms <= 0.5 && score.geometryAccuracy >= 0.0025;
    }

    ReferenceInitialPairScore evaluateReferenceInitialPairScore(const SfmReconstruction& reconstruction,
                                                                ImageId firstImageId,
                                                                ImageId secondImageId)
    {
        ReferenceInitialPairScore score;
        score.alignedCameraCount = static_cast<int>(reconstruction.numRegisteredImages());
        score.firstImageId = firstImageId;
        score.secondImageId = secondImageId;
        std::unordered_map<ImageId, std::pair<double, int>> camera_squared_residuals;
        double accuracy_sum = 0.0;

        for (const auto& [point_id, point] : reconstruction.points3D())
        {
            (void)point_id;
            if (registeredObservationImages(reconstruction, point).size() < 2)
            {
                continue;
            }
            ++score.pointCount;
            accuracy_sum += pointGeometryAccuracy(reconstruction, point);
            for (const TrackElement& observation : point.track.elements)
            {
                double residual = 0.0;
                if (!observationResidual(reconstruction, point, observation, &residual))
                {
                    continue;
                }
                auto& accumulator = camera_squared_residuals[observation.imageId];
                accumulator.first += residual * residual;
                ++accumulator.second;
            }
        }
        score.geometryAccuracy = score.pointCount > 0 ? accuracy_sum / static_cast<double>(score.pointCount) : 0.0;
        for (const auto& [image_id, accumulator] : camera_squared_residuals)
        {
            (void)image_id;
            if (accumulator.second > 0)
            {
                score.maximumCameraReprojectionRms =
                    std::max(score.maximumCameraReprojectionRms,
                             std::sqrt(accumulator.first / static_cast<double>(accumulator.second)));
            }
        }
        if (camera_squared_residuals.size() < reconstruction.numRegisteredImages())
        {
            score.maximumCameraReprojectionRms = std::numeric_limits<double>::infinity();
        }
        score.pointCountTier = score.pointCount >= 100;
        score.accuracyTier = score.geometryAccuracy >= 0.0025;
        return score;
    }

    ReferenceStructureFilterResult filterReferenceStructurePoints(SfmReconstruction& reconstruction,
                                                                  double requestedReprojectionThreshold,
                                                                  int threadCount)
    {
        ReferenceStructureFilterResult result;
        std::vector<Point3DId> point_ids = reconstruction.allPoint3DIds();
        std::vector<std::uint8_t> remove_far(point_ids.size(), 0);
        parallelReferenceChunks(
            point_ids.size(),
            threadCount,
            [&](std::size_t begin, std::size_t end, std::size_t)
            {
                for (std::size_t index = begin; index < end; ++index)
                {
                    const ScenePoint3D& point = reconstruction.point3D(point_ids[index]);
                    for (const TrackElement& observation : point.track.elements)
                    {
                        if (!reconstruction.isRegistered(observation.imageId))
                        {
                            continue;
                        }
                        double residual = 0.0;
                        const double sensor_threshold = sensorResidualThreshold(reconstruction, observation.imageId);
                        const double threshold = requestedReprojectionThreshold > 0.0
                                                     ? std::min(requestedReprojectionThreshold, sensor_threshold)
                                                     : sensor_threshold;
                        if (!observationResidual(reconstruction, point, observation, &residual) || residual > threshold)
                        {
                            remove_far[index] = 1;
                            break;
                        }
                    }
                }
            });
        for (std::size_t index = 0; index < point_ids.size(); ++index)
        {
            if (remove_far[index] != 0)
            {
                reconstruction.deactivatePoint3D(point_ids[index]);
                ++result.farPoints;
            }
        }

        point_ids = reconstruction.allPoint3DIds();
        std::vector<double> accuracies(point_ids.size(), 0.0);
        parallelReferenceChunks(point_ids.size(),
                                threadCount,
                                [&](std::size_t begin, std::size_t end, std::size_t)
                                {
                                    for (std::size_t index = begin; index < end; ++index)
                                    {
                                        accuracies[index] = pointGeometryAccuracy(
                                            reconstruction, reconstruction.point3D(point_ids[index]));
                                    }
                                });
        std::vector<double> values;
        values.reserve(accuracies.size());
        for (const double accuracy : accuracies)
        {
            if (std::isfinite(accuracy))
            {
                values.push_back(accuracy);
            }
        }
        std::sort(values.begin(), values.end());
        const double upper_median = values.empty() ? 0.0 : values[values.size() / 2];
        const double accuracy_threshold = std::clamp(0.1 * upper_median, 1.0e-6, 1.0e-4);
        for (std::size_t index = 0; index < point_ids.size(); ++index)
        {
            const double accuracy = accuracies[index];
            if (!std::isfinite(accuracy) || accuracy < accuracy_threshold)
            {
                reconstruction.deactivatePoint3D(point_ids[index]);
                ++result.inaccuratePoints;
            }
        }

        // 双视种子阶段尚不存在可证明连通性的三视点；此时执行 weak 判据会把
        // 启动 P3P 所需的全部种子删除。参考流程只在第三台相机进入模型后启用该层。
        if (reconstruction.numRegisteredImages() < 3)
        {
            return result;
        }

        point_ids = reconstruction.allPoint3DIds();
        const std::vector<ImageId> registered_image_ids = reconstruction.registeredImageIds();
        std::unordered_map<ImageId, std::size_t> camera_indices;
        camera_indices.reserve(registered_image_ids.size());
        for (std::size_t index = 0; index < registered_image_ids.size(); ++index)
        {
            camera_indices.emplace(registered_image_ids[index], index);
        }

        const std::size_t camera_count = registered_image_ids.size();
        const std::size_t worker_count = referenceWorkerCount(point_ids.size(), threadCount);
        std::vector<std::vector<std::uint8_t>> worker_connected(
            worker_count, std::vector<std::uint8_t>(camera_count * camera_count, 0));
        parallelReferenceChunks(point_ids.size(),
                                threadCount,
                                [&](std::size_t begin, std::size_t end, std::size_t worker_index)
                                {
                                    std::vector<std::size_t> image_indices;
                                    for (std::size_t point_index = begin; point_index < end; ++point_index)
                                    {
                                        const std::vector<ImageId> image_ids = registeredObservationImages(
                                            reconstruction, reconstruction.point3D(point_ids[point_index]));
                                        if (image_ids.size() < 3)
                                        {
                                            continue;
                                        }
                                        image_indices.clear();
                                        image_indices.reserve(image_ids.size());
                                        for (const ImageId image_id : image_ids)
                                        {
                                            image_indices.push_back(camera_indices.at(image_id));
                                        }
                                        for (const std::size_t first : image_indices)
                                        {
                                            for (const std::size_t second : image_indices)
                                            {
                                                worker_connected[worker_index][first * camera_count + second] = 1;
                                            }
                                        }
                                    }
                                });
        std::vector<std::uint8_t> connected(camera_count * camera_count, 0);
        for (const std::vector<std::uint8_t>& local : worker_connected)
        {
            for (std::size_t index = 0; index < connected.size(); ++index)
            {
                connected[index] = connected[index] || local[index];
            }
        }

        std::vector<std::uint8_t> remove_weak(point_ids.size(), 0);
        parallelReferenceChunks(point_ids.size(),
                                threadCount,
                                [&](std::size_t begin, std::size_t end, std::size_t)
                                {
                                    for (std::size_t point_index = begin; point_index < end; ++point_index)
                                    {
                                        const std::vector<ImageId> image_ids = registeredObservationImages(
                                            reconstruction, reconstruction.point3D(point_ids[point_index]));
                                        if (image_ids.size() != 2)
                                        {
                                            continue;
                                        }
                                        const std::size_t first = camera_indices.at(image_ids[0]);
                                        const std::size_t second = camera_indices.at(image_ids[1]);
                                        remove_weak[point_index] = connected[first * camera_count + second] == 0;
                                    }
                                });
        for (std::size_t index = 0; index < point_ids.size(); ++index)
        {
            if (remove_weak[index] != 0)
            {
                reconstruction.deactivatePoint3D(point_ids[index]);
                ++result.weakPoints;
            }
        }
        return result;
    }

} // namespace xjw
