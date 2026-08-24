#include "VisualHullFieldBackend.h"

#include "VisualHullFieldEvaluator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <thread>

namespace xjw::mesh::detail
{
    namespace
    {

        int resolveWorkerCount(int requested)
        {
            if (requested > 0)
            {
                return std::clamp(requested, 1, 128);
            }
            const unsigned int hardware = std::thread::hardware_concurrency();
            return std::clamp(static_cast<int>(hardware > 0 ? hardware : 8U), 1, 128);
        }

        const char* backendName(VisualHullComputeBackend backend)
        {
            switch (backend)
            {
            case VisualHullComputeBackend::Auto:
                return "Auto";
            case VisualHullComputeBackend::Cpu:
                return "CPU";
            case VisualHullComputeBackend::Cuda:
                return "CUDA";
            case VisualHullComputeBackend::OpenCL:
                return "OpenCL";
            }
            return "unknown";
        }

        bool appendImageAsFloat(const cv::Mat& image,
                                int expectedType,
                                std::vector<float>* samples,
                                std::int32_t* offset,
                                std::string* errorMessage)
        {
            if (!samples || !offset || image.empty() || image.type() != expectedType)
            {
                if (errorMessage)
                {
                    *errorMessage = "visual hull GPU input image has an invalid type";
                }
                return false;
            }
            if (samples->size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            {
                if (errorMessage)
                {
                    *errorMessage = "visual hull GPU image buffer exceeds 32-bit indexing";
                }
                return false;
            }
            *offset = static_cast<std::int32_t>(samples->size());
            const std::size_t additional = static_cast<std::size_t>(image.rows) * image.cols;
            if (additional > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - samples->size())
            {
                if (errorMessage)
                {
                    *errorMessage = "visual hull GPU image buffer exceeds 32-bit indexing";
                }
                return false;
            }
            samples->reserve(samples->size() + additional);
            for (int row = 0; row < image.rows; ++row)
            {
                if (expectedType == CV_8UC1)
                {
                    const std::uint8_t* source = image.ptr<std::uint8_t>(row);
                    for (int column = 0; column < image.cols; ++column)
                    {
                        samples->push_back(source[column] == 0 ? 0.0f : 1.0f);
                    }
                }
                else
                {
                    const float* source = image.ptr<float>(row);
                    samples->insert(samples->end(), source, source + image.cols);
                }
            }
            return true;
        }

        bool packDeviceInput(const std::vector<VisualHullView>& views,
                             const VisualHullConfig& config,
                             const RegularGrid3D& grid,
                             VisualHullFieldDeviceInput* input,
                             std::string* errorMessage)
        {
            if (!input || !grid.isValid())
            {
                if (errorMessage)
                {
                    *errorMessage = "visual hull GPU grid is invalid";
                }
                return false;
            }
            if (config.useContinuousSilhouetteField &&
                config.allowedSilhouetteViolations > kVisualHullMaximumGpuSilhouetteViolations)
            {
                if (errorMessage)
                {
                    *errorMessage = "continuous visual hull GPU evaluation supports at most " +
                                    std::to_string(kVisualHullMaximumGpuSilhouetteViolations) +
                                    " allowed silhouette violations";
                }
                return false;
            }

            *input = VisualHullFieldDeviceInput{};
            input->grid = grid;
            input->minimumVisibleViews = config.minimumVisibleViews;
            input->allowedSilhouetteViolations = config.allowedSilhouetteViolations;
            input->continuousSilhouetteField = config.useContinuousSilhouetteField;
            input->enableDepthFreeSpaceCarving = config.enableDepthFreeSpaceCarving;
            input->minimumDepthFreeSpaceViolations = config.minimumDepthFreeSpaceViolations;
            input->relativeDepthTolerance = config.relativeDepthTolerance;
            input->closeVolumeBoundary = config.closeVolumeBoundary;
            input->gpuSlabDepth = std::clamp(config.gpuSlabDepth, 1, 64);
            input->isCancelled = config.isCancelled;

            std::array<double, 3> world_origin{};
            const std::size_t coordinate_count = static_cast<std::size_t>(grid.sampleSize(0)) +
                                                 static_cast<std::size_t>(grid.sampleSize(1)) +
                                                 static_cast<std::size_t>(grid.sampleSize(2));
            input->gridCoordinates.reserve(coordinate_count);
            // Match the CPU grid's float rounding first, then shift both grid
            // coordinates and double-precision camera centers to one local
            // origin. This avoids catastrophic world-center cancellation on
            // planetary absolute coordinates without requiring device FP64.
            for (int axis = 0; axis < 3; ++axis)
            {
                world_origin[axis] =
                    0.5 * (static_cast<double>(grid.boundsMin[axis]) + static_cast<double>(grid.boundsMax[axis]));
                for (int index = 0; index < grid.sampleSize(axis); ++index)
                {
                    const float absolute_coordinate =
                        grid.boundsMin[axis] + grid.spacing(axis) * static_cast<float>(index);
                    input->gridCoordinates.push_back(
                        static_cast<float>(static_cast<double>(absolute_coordinate) - world_origin[axis]));
                }
            }

            const std::vector<PreparedVisualHullView> prepared = config.useContinuousSilhouetteField
                                                                     ? prepareVisualHullFieldViews(views)
                                                                     : std::vector<PreparedVisualHullView>{};
            for (std::size_t view_index = 0; view_index < views.size(); ++view_index)
            {
                const VisualHullView& view = views[view_index];
                if (!view.camera.isValid() || view.silhouetteMask.empty() || view.silhouetteMask.type() != CV_8UC1)
                {
                    continue;
                }
                const cv::Mat& silhouette = config.useContinuousSilhouetteField
                                                ? prepared[view_index].signedSilhouetteDistance
                                                : view.silhouetteMask;
                const int silhouette_type = config.useContinuousSilhouetteField ? CV_32FC1 : CV_8UC1;
                std::int32_t silhouette_offset = -1;
                if (!appendImageAsFloat(
                        silhouette, silhouette_type, &input->silhouetteSamples, &silhouette_offset, errorMessage))
                {
                    return false;
                }

                std::int32_t depth_offset = -1;
                int depth_columns = 0;
                int depth_rows = 0;
                if (config.enableDepthFreeSpaceCarving && !view.depthMap.empty())
                {
                    if (!appendImageAsFloat(view.depthMap, CV_32FC1, &input->depthSamples, &depth_offset, errorMessage))
                    {
                        return false;
                    }
                    depth_columns = view.depthMap.cols;
                    depth_rows = view.depthMap.rows;
                }

                const FramePinholeCamera::Intrinsics intrinsics = view.camera.intrinsics();
                const FramePinholeCamera::Distortion distortion = view.camera.distortion();
                const FramePinholeCamera::Pose pose = view.camera.pose();
                const std::size_t parameter_offset = input->cameraParameters.size();
                input->cameraParameters.resize(parameter_offset + kVisualHullCameraParameterStride, 0.0f);
                float* parameters = input->cameraParameters.data() + parameter_offset;
                for (int index = 0; index < 9; ++index)
                {
                    parameters[index] = static_cast<float>(pose.cameraToWorldRotation[index]);
                }
                for (int index = 0; index < 3; ++index)
                {
                    parameters[9 + index] = static_cast<float>(pose.cameraCenter[index] - world_origin[index]);
                }
                parameters[12] = static_cast<float>(intrinsics.focalX);
                parameters[13] = static_cast<float>(intrinsics.focalY);
                parameters[14] = static_cast<float>(intrinsics.principalX);
                parameters[15] = static_cast<float>(intrinsics.principalY);
                parameters[16] = static_cast<float>(distortion.radialK1);
                parameters[17] = static_cast<float>(distortion.radialK2);
                parameters[18] = static_cast<float>(distortion.radialK3);
                parameters[19] = static_cast<float>(distortion.tangentialP1);
                parameters[20] = static_cast<float>(distortion.tangentialP2);
                parameters[21] = intrinsics.uAxisSign < 0 ? -1.0f : 1.0f;
                parameters[22] = intrinsics.vAxisSign < 0 ? -1.0f : 1.0f;
                parameters[23] = pose.depthAxisFlipped ? -1.0f : 1.0f;
                parameters[24] = static_cast<float>(0.5 * (std::abs(intrinsics.focalX) + std::abs(intrinsics.focalY)));

                input->viewMetadata.insert(input->viewMetadata.end(),
                                           {silhouette_offset,
                                            static_cast<std::int32_t>(silhouette.cols),
                                            static_cast<std::int32_t>(silhouette.rows),
                                            depth_offset,
                                            static_cast<std::int32_t>(depth_columns),
                                            static_cast<std::int32_t>(depth_rows)});
                ++input->viewCount;
            }
            return true;
        }

        bool evaluateCpu(const std::vector<VisualHullView>& views,
                         const VisualHullConfig& config,
                         const RegularGrid3D& grid,
                         std::vector<float>* field,
                         std::string* errorMessage)
        {
            field->assign(grid.sampleCount(), 1.0f);
            const std::vector<PreparedVisualHullView> prepared = config.useContinuousSilhouetteField
                                                                     ? prepareVisualHullFieldViews(views)
                                                                     : std::vector<PreparedVisualHullView>{};
            std::atomic_bool cancelled{false};
            std::atomic<int> completed_layer_count{0};
            std::mutex progress_mutex;
            int reported_progress_percent = 5;
            const int size_x = grid.sampleSize(0);
            const int size_y = grid.sampleSize(1);
            const int size_z = grid.sampleSize(2);
            const int workers = resolveWorkerCount(config.workerCount);

#if defined(MESHING_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
#endif
            for (int z = 0; z < size_z; ++z)
            {
                if (cancelled.load(std::memory_order_relaxed))
                {
                    continue;
                }
                if (config.isCancelled && config.isCancelled())
                {
                    cancelled.store(true, std::memory_order_relaxed);
                    continue;
                }
                for (int y = 0; y < size_y; ++y)
                {
                    for (int x = 0; x < size_x; ++x)
                    {
                        const std::size_t offset = grid.linearIndex(x, y, z);
                        if (config.closeVolumeBoundary && grid.isBoundarySample(x, y, z))
                        {
                            (*field)[offset] = 1.0f;
                            continue;
                        }
                        const std::array<float, 3> world = grid.samplePosition(x, y, z);
                        (*field)[offset] =
                            config.useContinuousSilhouetteField
                                ? evaluateContinuousVisualHullField(world[0], world[1], world[2], prepared, config)
                                : (evaluateBinaryVisualHullField(world[0], world[1], world[2], views, config) ? -1.0f
                                                                                                              : 1.0f);
                    }
                }
                const int completed_layers = completed_layer_count.fetch_add(1, std::memory_order_relaxed) + 1;
                const int progress_percent = 5 + completed_layers * 65 / size_z;
                if (config.progressFn)
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    if (progress_percent > reported_progress_percent)
                    {
                        reported_progress_percent = progress_percent;
                        config.progressFn("正在评估多视轮廓体素（" + std::to_string(completed_layers) + "/" +
                                              std::to_string(size_z) + " 层）...",
                                          static_cast<float>(progress_percent) / 100.0f);
                    }
                }
            }
            if (cancelled.load(std::memory_order_relaxed))
            {
                if (errorMessage)
                {
                    *errorMessage = "visual hull reconstruction cancelled";
                }
                return false;
            }
            return true;
        }

    } // namespace

    bool isVisualHullFieldBackendAvailable(VisualHullComputeBackend backend, int deviceIndex) noexcept
    {
        switch (backend)
        {
        case VisualHullComputeBackend::Auto:
        case VisualHullComputeBackend::Cpu:
            return true;
        case VisualHullComputeBackend::Cuda:
            return cudaVisualHullFieldAvailable(deviceIndex);
        case VisualHullComputeBackend::OpenCL:
            return openClVisualHullFieldAvailable(deviceIndex);
        }
        return false;
    }

    bool evaluateVisualHullFieldGrid(const std::vector<VisualHullView>& views,
                                     const VisualHullConfig& config,
                                     const RegularGrid3D& grid,
                                     std::vector<float>* field,
                                     VisualHullComputeBackend* usedBackend,
                                     std::string* errorMessage,
                                     VisualHullExecutionInfo* executionInfo)
    {
        VisualHullExecutionInfo info;
        info.requestedBackend = config.computeBackend;
        info.requestedDeviceIndex = config.computeDeviceIndex;
        if (executionInfo)
        {
            *executionInfo = info;
        }
        if (usedBackend)
        {
            *usedBackend = VisualHullComputeBackend::Auto;
        }
        if (errorMessage)
        {
            errorMessage->clear();
        }

        const auto publish_execution =
            [&](VisualHullComputeBackend actual_backend, int actual_device_index, const std::string& fallback_reason)
        {
            info.actualBackend = actual_backend;
            info.actualDeviceIndex = actual_device_index;
            info.usedFallback = config.computeBackend == VisualHullComputeBackend::Auto && !fallback_reason.empty();
            info.fallbackReason = fallback_reason;
            if (executionInfo)
            {
                *executionInfo = info;
            }
            if (config.executionInfoFn)
            {
                config.executionInfoFn(info);
            }
        };

        if (!field || !grid.isValid())
        {
            if (errorMessage)
            {
                *errorMessage = "visual hull field output or grid is invalid";
            }
            return false;
        }
        field->clear();
        if (config.isCancelled && config.isCancelled())
        {
            if (errorMessage)
            {
                *errorMessage = "visual hull reconstruction cancelled";
            }
            return false;
        }

        const auto run_cpu = [&](const std::string& fallback_reason)
        {
            if (!fallback_reason.empty() && config.progressFn)
            {
                config.progressFn("GPU 体素场执行失败，已回退 CPU：" + fallback_reason, 0.05f);
            }
            const bool ok = evaluateCpu(views, config, grid, field, errorMessage);
            if (ok)
            {
                if (usedBackend)
                {
                    *usedBackend = VisualHullComputeBackend::Cpu;
                }
                publish_execution(VisualHullComputeBackend::Cpu, -1, fallback_reason);
            }
            return ok;
        };
        if (config.computeBackend == VisualHullComputeBackend::Cpu)
        {
            return run_cpu({});
        }

        const bool strict = config.computeBackend != VisualHullComputeBackend::Auto;
        const VisualHullComputeBackend candidates[] = {VisualHullComputeBackend::Cuda,
                                                       VisualHullComputeBackend::OpenCL};
        std::string fallback_reason;
        const auto append_failure = [&](VisualHullComputeBackend backend, const std::string& reason)
        {
            if (!fallback_reason.empty())
            {
                fallback_reason += "; ";
            }
            fallback_reason += std::string(backendName(backend)) + ": " + reason;
        };
        for (VisualHullComputeBackend candidate : candidates)
        {
            if (strict && config.computeBackend != candidate)
            {
                continue;
            }
            if (!isVisualHullFieldBackendAvailable(candidate, config.computeDeviceIndex))
            {
                if (strict)
                {
                    if (errorMessage)
                    {
                        *errorMessage = std::string(backendName(candidate)) + " visual hull backend is unavailable";
                    }
                    return false;
                }
                append_failure(candidate, "backend is unavailable for the requested device");
                continue;
            }

            VisualHullFieldDeviceInput input;
            std::string backend_error;
            if (!packDeviceInput(views, config, grid, &input, &backend_error))
            {
                if (strict)
                {
                    if (errorMessage)
                    {
                        *errorMessage = backend_error;
                    }
                    return false;
                }
                append_failure(candidate, backend_error);
                break;
            }
            if (config.progressFn)
            {
                config.progressFn(std::string("正在使用 ") + backendName(candidate) + " 评估多视轮廓体素...", 0.05f);
            }
            int actual_device_index = -1;
            const bool ok = candidate == VisualHullComputeBackend::Cuda
                                ? evaluateVisualHullFieldCuda(
                                      input, config.computeDeviceIndex, field, &actual_device_index, &backend_error)
                                : evaluateVisualHullFieldOpenCl(
                                      input, config.computeDeviceIndex, field, &actual_device_index, &backend_error);
            if (ok)
            {
                if (config.isCancelled && config.isCancelled())
                {
                    if (errorMessage)
                    {
                        *errorMessage = "visual hull reconstruction cancelled";
                    }
                    return false;
                }
                if (config.progressFn)
                {
                    config.progressFn("多视轮廓体素场评估完成", 0.70f);
                }
                if (usedBackend)
                {
                    *usedBackend = candidate;
                }
                publish_execution(candidate, actual_device_index, fallback_reason);
                return true;
            }
            if ((config.isCancelled && config.isCancelled()) || backend_error.find("cancelled") != std::string::npos)
            {
                if (errorMessage)
                {
                    *errorMessage = "visual hull reconstruction cancelled";
                }
                return false;
            }
            if (strict)
            {
                if (errorMessage)
                {
                    *errorMessage =
                        std::string(backendName(candidate)) + " visual hull evaluation failed: " + backend_error;
                }
                return false;
            }
            append_failure(candidate, backend_error);
        }

        if (strict)
        {
            if (errorMessage)
            {
                *errorMessage = "unsupported visual hull compute backend";
            }
            return false;
        }
        const bool cpu_ok = run_cpu(fallback_reason);
        if (!cpu_ok && errorMessage && !fallback_reason.empty())
        {
            *errorMessage += "; accelerator attempts: " + fallback_reason;
        }
        return cpu_ok;
    }

} // namespace xjw::mesh::detail
