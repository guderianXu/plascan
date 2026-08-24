#include "OrthoProjector.h"

#include "OrthoProjectorGpu.h"
#include "OrthoProjectorInternal.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace xjw
{

    namespace
    {

        bool isCancelled(const std::atomic_bool* cancelFlag)
        {
            return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
        }

        bool sampleDemElevation(const DemGridData& demGrid, double worldX, double worldY, double* elevation)
        {
            const double grid_x = (worldX - demGrid.minX) / demGrid.stepX;
            const double grid_y = (worldY - demGrid.minY) / demGrid.stepY;
            if (grid_x < -0.5 || grid_y < -0.5 || grid_x > static_cast<double>(demGrid.width) - 0.5 ||
                grid_y > static_cast<double>(demGrid.height) - 0.5)
            {
                return false;
            }

            const int nearest_x = std::clamp(static_cast<int>(std::lround(grid_x)), 0, demGrid.width - 1);
            const int nearest_y = std::clamp(static_cast<int>(std::lround(grid_y)), 0, demGrid.height - 1);
            const int x0 = std::clamp(static_cast<int>(std::floor(grid_x)), 0, demGrid.width - 1);
            const int y0 = std::clamp(static_cast<int>(std::floor(grid_y)), 0, demGrid.height - 1);
            const int x1 = std::min(x0 + 1, demGrid.width - 1);
            const int y1 = std::min(y0 + 1, demGrid.height - 1);

            const bool bilinear_valid =
                demGrid.validMask.at<uchar>(y0, x0) != 0 && demGrid.validMask.at<uchar>(y0, x1) != 0 &&
                demGrid.validMask.at<uchar>(y1, x0) != 0 && demGrid.validMask.at<uchar>(y1, x1) != 0;
            if (bilinear_valid)
            {
                const double fx = std::clamp(grid_x - static_cast<double>(x0), 0.0, 1.0);
                const double fy = std::clamp(grid_y - static_cast<double>(y0), 0.0, 1.0);
                const double z00 = demGrid.elevation.at<float>(y0, x0);
                const double z10 = demGrid.elevation.at<float>(y0, x1);
                const double z01 = demGrid.elevation.at<float>(y1, x0);
                const double z11 = demGrid.elevation.at<float>(y1, x1);
                *elevation =
                    (1.0 - fx) * (1.0 - fy) * z00 + fx * (1.0 - fy) * z10 + (1.0 - fx) * fy * z01 + fx * fy * z11;
                return std::isfinite(*elevation);
            }

            if (demGrid.validMask.at<uchar>(nearest_y, nearest_x) == 0)
            {
                return false;
            }
            *elevation = demGrid.elevation.at<float>(nearest_y, nearest_x);
            return std::isfinite(*elevation);
        }

        cv::Vec3b byteColor(const cv::Vec3f& color)
        {
            return cv::Vec3b(static_cast<uchar>(std::clamp(color[0], 0.0f, 255.0f)),
                             static_cast<uchar>(std::clamp(color[1], 0.0f, 255.0f)),
                             static_cast<uchar>(std::clamp(color[2], 0.0f, 255.0f)));
        }

        bool blendCandidates(const std::vector<ortho_internal::ColorCandidate>& candidates,
                             OrthoBlendMode blendMode,
                             cv::Vec3f* color,
                             std::vector<int>* contributingFrames)
        {
            if (candidates.empty() || !color || !contributingFrames)
            {
                return false;
            }

            contributingFrames->clear();
            if (blendMode == OrthoBlendMode::FirstValid)
            {
                *color = candidates.front().color;
                contributingFrames->push_back(candidates.front().frameIndex);
                return true;
            }
            if (blendMode == OrthoBlendMode::Mosaic)
            {
                const auto best =
                    std::max_element(candidates.begin(),
                                     candidates.end(),
                                     [](const auto& left, const auto& right) { return left.weight < right.weight; });
                *color = best->color;
                contributingFrames->push_back(best->frameIndex);
                return true;
            }

            cv::Vec3d sum(0.0, 0.0, 0.0);
            double total_weight = 0.0;
            for (const auto& candidate : candidates)
            {
                sum += cv::Vec3d(candidate.color) * candidate.weight;
                total_weight += candidate.weight;
                contributingFrames->push_back(candidate.frameIndex);
            }
            if (total_weight <= 0.0)
            {
                return false;
            }
            const cv::Vec3d averaged = sum / total_weight;
            *color = cv::Vec3f(
                static_cast<float>(averaged[0]), static_cast<float>(averaged[1]), static_cast<float>(averaged[2]));
            return true;
        }

    } // namespace

    bool OrthoProjector::project(const DemGridData& demGrid,
                                 const std::vector<OrthoImageInput>& inputs,
                                 const OrthoGenerationOptions& options,
                                 double demElevationOffset,
                                 OrthoProjectionResult* result,
                                 QString* errorMsg,
                                 const std::atomic_bool* cancelFlag,
                                 const ProgressCallback& progressCallback)
    {
        if (!result)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射投影结果对象为空");
            }
            return false;
        }
        if (isCancelled(cancelFlag))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射影像生成已取消");
            }
            return false;
        }
        if (!demGrid.isValid() || demGrid.elevation.type() != CV_32FC1 || demGrid.validMask.type() != CV_8UC1)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射投影要求有效的 CV_32FC1 DEM 和 CV_8UC1 有效掩膜");
            }
            return false;
        }

        OrthoProjectionResult projected;
        projected.selectedCameraCount = static_cast<int>(inputs.size());
        if (!planOutputGrid(demGrid, options, &projected.outputGrid, errorMsg))
        {
            return false;
        }
        if (options.ghostFilter && options.computeBackend != TerrainComputeBackend::Auto &&
            options.computeBackend != TerrainComputeBackend::Cpu)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("显式 %1 正射后端不支持 ghostFilter；请选择 CPU，或使用 Auto 允许回退")
                                .arg(terrainComputeBackendDisplayName(options.computeBackend));
            }
            return false;
        }

        std::vector<ortho_internal::LoadedFrame> frames;
        if (progressCallback)
        {
            progressCallback(QStringLiteral("加载正射影像"), 0);
        }
        if (!ortho_internal::loadFrames(inputs, options, &frames, cancelFlag, errorMsg))
        {
            return false;
        }
        projected.loadedCameraCount = static_cast<int>(frames.size());

        const int width = projected.outputGrid.reference.width;
        const int height = projected.outputGrid.reference.height;
        qint64 surface_pixels = 0;
        qint64 filled_pixels = 0;
        bool projected_on_gpu = false;
        QStringList fallback_reasons;

        auto try_gpu = [&](TerrainComputeBackend backend) -> bool
        {
            if (!isTerrainComputeBackendAvailable(backend, options.computeDeviceIndex))
            {
                fallback_reasons.push_back(
                    QStringLiteral("%1 后端或设备不可用").arg(terrainComputeBackendDisplayName(backend)));
                return false;
            }
            QString gpu_error;
            QString device_name;
            int resolved_device_index = -1;
            if (!ortho_internal::projectPixelsOnGpu(demGrid,
                                                    projected.outputGrid,
                                                    &frames,
                                                    options,
                                                    demElevationOffset,
                                                    backend,
                                                    options.computeDeviceIndex,
                                                    &projected,
                                                    &surface_pixels,
                                                    &resolved_device_index,
                                                    &device_name,
                                                    &gpu_error))
            {
                fallback_reasons.push_back(
                    QStringLiteral("%1 执行失败: %2").arg(terrainComputeBackendDisplayName(backend), gpu_error));
                return false;
            }
            projected.computeExecution.backend = backend;
            projected.computeExecution.deviceIndex = resolved_device_index;
            projected.computeExecution.deviceName = device_name;
            filled_pixels = projected.filledPixelCount;
            projected_on_gpu = true;
            return true;
        };

        if (options.computeBackend == TerrainComputeBackend::Cuda ||
            options.computeBackend == TerrainComputeBackend::OpenCl)
        {
            if (!try_gpu(options.computeBackend))
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("显式 %1 正射后端失败: %2")
                                    .arg(terrainComputeBackendDisplayName(options.computeBackend),
                                         fallback_reasons.join(QStringLiteral("；")));
                }
                return false;
            }
        }
        else if (options.computeBackend == TerrainComputeBackend::Auto)
        {
            if (options.ghostFilter)
            {
                fallback_reasons.push_back(QStringLiteral("ghostFilter 尚未实现 GPU 路径，Auto 已回退 CPU"));
            }
            else if (!try_gpu(TerrainComputeBackend::Cuda))
            {
                try_gpu(TerrainComputeBackend::OpenCl);
            }
        }

        if (!projected_on_gpu)
        {
            projected.computeExecution.backend = TerrainComputeBackend::Cpu;
            projected.computeExecution.deviceIndex = -1;
            projected.computeExecution.deviceName = QStringLiteral("CPU");
            projected.imageBgr = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
            projected.surfaceMask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
            projected.coverageMask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

            std::vector<ortho_internal::ColorCandidate> candidates;
            candidates.reserve(frames.size());
            std::vector<int> contributing_frames;
            contributing_frames.reserve(frames.size());

            for (int row = 0; row < height; ++row)
            {
                if (isCancelled(cancelFlag))
                {
                    if (errorMsg)
                    {
                        *errorMsg = QStringLiteral("正射影像生成已取消");
                    }
                    return false;
                }
                for (int col = 0; col < width; ++col)
                {
                    const double world_x = projected.outputGrid.minEdgeX +
                                           (static_cast<double>(col) + 0.5) * projected.outputGrid.reference.stepX;
                    const double world_y = projected.outputGrid.minEdgeY +
                                           (static_cast<double>(row) + 0.5) * projected.outputGrid.reference.stepY;
                    double elevation = 0.0;
                    if (!sampleDemElevation(demGrid, world_x, world_y, &elevation))
                    {
                        continue;
                    }
                    projected.surfaceMask.at<uchar>(row, col) = 255;
                    ++surface_pixels;

                    const double world[3]{world_x, world_y, elevation + demElevationOffset};
                    candidates.clear();
                    for (int frame_index = 0; frame_index < static_cast<int>(frames.size()); ++frame_index)
                    {
                        ortho_internal::ColorCandidate candidate;
                        candidate.frameIndex = frame_index;
                        if (ortho_internal::sampleFrame(
                                frames[static_cast<std::size_t>(frame_index)], world, &candidate))
                        {
                            candidates.push_back(candidate);
                            if (options.blendMode == OrthoBlendMode::FirstValid)
                            {
                                break;
                            }
                        }
                    }
                    if (options.ghostFilter && options.blendMode != OrthoBlendMode::FirstValid)
                    {
                        ortho_internal::filterGhostCandidates(&candidates);
                    }

                    cv::Vec3f color;
                    if (!blendCandidates(candidates, options.blendMode, &color, &contributing_frames))
                    {
                        continue;
                    }
                    projected.imageBgr.at<cv::Vec3b>(row, col) = byteColor(color);
                    projected.coverageMask.at<uchar>(row, col) = 255;
                    ++filled_pixels;
                    for (int frame_index : contributing_frames)
                    {
                        frames[static_cast<std::size_t>(frame_index)].contributed = true;
                    }
                }
                if (progressCallback)
                {
                    const int percent =
                        5 + static_cast<int>(85.0 * static_cast<double>(row + 1) / static_cast<double>(height));
                    progressCallback(QStringLiteral("正射投影"), percent);
                }
            }
        }
        else if (progressCallback)
        {
            progressCallback(QStringLiteral("正射投影（%1）").arg(projected.computeExecution.deviceName), 90);
        }
        projected.computeExecution.fallbackReason = fallback_reasons.join(QStringLiteral("；"));

        if (isCancelled(cancelFlag))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射影像生成已取消");
            }
            return false;
        }

        if (filled_pixels <= 0)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射投影没有产生有效像素，请检查 DEM、相机和影像坐标系");
            }
            return false;
        }

        if (progressCallback)
        {
            progressCallback(QStringLiteral("正射孔洞处理"), 92);
        }
        projected.holeFilledPixelCount = ortho_internal::fillSmallInteriorHoles(&projected.imageBgr,
                                                                                projected.surfaceMask,
                                                                                projected.coverageMask,
                                                                                options,
                                                                                &projected.holeFilledMask,
                                                                                cancelFlag);
        if (projected.holeFilledPixelCount < 0)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射影像生成已取消");
            }
            return false;
        }
        projected.filledPixelCount = filled_pixels;
        projected.coverageRatio =
            surface_pixels > 0 ? static_cast<double>(filled_pixels) / static_cast<double>(surface_pixels) : 0.0;
        for (const auto& frame : frames)
        {
            if (frame.contributed)
            {
                ++projected.contributingCameraCount;
            }
        }
        cv::bitwise_or(projected.coverageMask, projected.holeFilledMask, projected.outputGrid.reference.validMask);
        *result = std::move(projected);
        if (progressCallback)
        {
            progressCallback(QStringLiteral("正射投影完成"), 100);
        }
        return true;
    }

} // namespace xjw
