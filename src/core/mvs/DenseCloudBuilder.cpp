// =============================================================================
// 文件: DenseCloudBuilder.cpp
// 模块: MVS - 稠密点云构建 (CPU)
// =============================================================================

#include "DenseCloudBuilder.h"
#include "DensePointCloudCUDA.h"
#include "DensePointCloudOpenCL.h"
#include "io/PathIO.h"
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/io/ply_io.h>
#include "log/Logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <sstream>
#include <stdexcept>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace xjw
{
    namespace mvs
    {

        namespace
        {
            static plapoint::PointCloud<float, plamatrix::Device::CPU>
            buildPointCloud(const std::vector<DensePoint>& cloud);

            std::vector<DensePoint> unprojectCpu(const cv::Mat& depth,
                                                 const cv::Mat& mask,
                                                 const FramePinholeCamera& cam,
                                                 const cv::Mat& colorImg,
                                                 const DenseCloudOptions& options);
        } // namespace

        // =============================================================================
        const char* denseCloudComputeBackendId(DenseCloudComputeBackend backend) noexcept
        {
            return patchMatchBackendId(backend);
        }

        namespace
        {

            std::vector<DensePoint> unprojectCpu(const cv::Mat& depth,
                                                 const cv::Mat& mask,
                                                 const FramePinholeCamera& cam,
                                                 const cv::Mat& colorImg,
                                                 const DenseCloudOptions& options)
            {
                std::vector<DensePoint> cloud;
                if (depth.empty())
                {
                    return cloud;
                }

                const int W = depth.cols, H = depth.rows;
                const bool hasColor = !colorImg.empty() && colorImg.cols == W && colorImg.rows == H;
                const bool isRGB = hasColor && colorImg.channels() == 3;
                const bool isGray = hasColor && colorImg.channels() == 1;

                const int step = std::max(1, options.subsample);
                const std::size_t sampledColumns =
                    (static_cast<std::size_t>(W) + static_cast<std::size_t>(step) - 1) / static_cast<std::size_t>(step);
                const std::size_t sampledRows =
                    (static_cast<std::size_t>(H) + static_cast<std::size_t>(step) - 1) / static_cast<std::size_t>(step);
                cloud.reserve(sampledColumns * sampledRows);

                for (int v = 0; v < H; v += step)
                {
                    for (int u = 0; u < W; u += step)
                    {
                        // 检查掩码
                        if (!mask.empty() && mask.at<uint8_t>(v, u) == 0)
                        {
                            continue;
                        }

                        float d = depth.at<float>(v, u);
                        if (d < options.minDepth || d > options.maxDepth)
                        {
                            continue;
                        }

                        const double pixel[2] = {static_cast<double>(u), static_cast<double>(v)};
                        double world[3] = {0.0, 0.0, 0.0};
                        if (!cam.unprojectPixel(pixel, static_cast<double>(d), world))
                        {
                            continue;
                        }
                        const float Xw = static_cast<float>(world[0]);
                        const float Yw = static_cast<float>(world[1]);
                        const float Zw = static_cast<float>(world[2]);

                        // AABB 裁剪
                        if (options.clipAABB)
                        {
                            if (Xw < options.minX || Xw > options.maxX)
                            {
                                continue;
                            }
                            if (Yw < options.minY || Yw > options.maxY)
                            {
                                continue;
                            }
                            if (Zw < options.minZ || Zw > options.maxZ)
                            {
                                continue;
                            }
                        }

                        DensePoint pt;
                        pt.x = Xw;
                        pt.y = Yw;
                        pt.z = Zw;

                        // 颜色（无颜色时保持默认 0，避免未初始化的栈垃圾值）
                        if (isRGB)
                        {
                            const cv::Vec3b& bgr = colorImg.at<cv::Vec3b>(v, u);
                            pt.r = bgr[2];
                            pt.g = bgr[1];
                            pt.b = bgr[0];
                        }
                        else if (isGray)
                        {
                            uint8_t g = colorImg.at<uint8_t>(v, u);
                            pt.r = pt.g = pt.b = g;
                        }
                        else
                        {
                            pt.r = pt.g = pt.b = 128;
                        }

                        cloud.push_back(pt);
                    }
                }

                return cloud;
            }

            bool validateUnprojectionInput(const cv::Mat& depth,
                                           const cv::Mat& mask,
                                           const FramePinholeCamera& camera,
                                           const cv::Mat& color,
                                           const DenseCloudOptions& options,
                                           std::string* errorMsg)
            {
                const auto fail = [errorMsg](const std::string& message)
                {
                    if (errorMsg)
                    {
                        *errorMsg = message;
                    }
                    return false;
                };
                if (depth.empty())
                {
                    return true;
                }
                if (depth.type() != CV_32FC1)
                {
                    return fail("dense-cloud unprojection requires a CV_32FC1 depth map");
                }
                if (!camera.isValid())
                {
                    return fail("dense-cloud unprojection requires a valid pinhole camera");
                }
                if (!mask.empty() && (mask.type() != CV_8UC1 || mask.size() != depth.size()))
                {
                    return fail("dense-cloud mask must be CV_8UC1 and match the depth map");
                }
                if (!color.empty() &&
                    ((color.type() != CV_8UC1 && color.type() != CV_8UC3) || color.size() != depth.size()))
                {
                    return fail("dense-cloud color image must be CV_8UC1/CV_8UC3 and match the depth map");
                }
                if (!std::isfinite(options.minDepth) || !std::isfinite(options.maxDepth) || options.minDepth <= 0.0f ||
                    options.maxDepth < options.minDepth)
                {
                    return fail("dense-cloud depth range is invalid");
                }
                if (options.subsample <= 0)
                {
                    return fail("dense-cloud subsample must be positive");
                }
                return true;
            }

        } // namespace

        std::vector<DensePoint> DenseCloudBuilder::unproject(const cv::Mat& depth,
                                                             const cv::Mat& mask,
                                                             const FramePinholeCamera& cameraModel,
                                                             const cv::Mat& colorImg,
                                                             const DenseCloudOptions& options)
        {
            std::vector<DensePoint> cloud;
            std::string error;
            if (!unprojectWithReport(depth, mask, cameraModel, colorImg, options, &cloud, nullptr, &error))
            {
                throw std::runtime_error(error.empty() ? "dense-cloud unprojection failed" : error);
            }
            return cloud;
        }

        bool DenseCloudBuilder::unprojectWithReport(const cv::Mat& depth,
                                                    const cv::Mat& mask,
                                                    const FramePinholeCamera& cameraModel,
                                                    const cv::Mat& colorImg,
                                                    const DenseCloudOptions& options,
                                                    std::vector<DensePoint>* cloud,
                                                    DenseCloudExecutionReport* report,
                                                    std::string* errorMsg)
        {
            DenseCloudExecutionReport localReport;
            localReport.requestedBackend = options.useGPU ? options.computeBackend : DenseCloudComputeBackend::Cpu;
            localReport.actualBackend = DenseCloudComputeBackend::Cpu;
            localReport.deviceIndex = -1;
            localReport.deviceName = "CPU";
            if (report)
            {
                *report = localReport;
            }
            if (!cloud)
            {
                if (errorMsg)
                {
                    *errorMsg = "dense-cloud output pointer is null";
                }
                return false;
            }
            cloud->clear();
            if (errorMsg)
            {
                errorMsg->clear();
            }
            if (!validateUnprojectionInput(depth, mask, cameraModel, colorImg, options, errorMsg))
            {
                return false;
            }

            const auto finishCpu = [&](const std::string& fallbackReason)
            {
                *cloud = unprojectCpu(depth, mask, cameraModel, colorImg, options);
                localReport.actualBackend = DenseCloudComputeBackend::Cpu;
                localReport.deviceIndex = -1;
                localReport.deviceName = "CPU";
                localReport.workSubmitted = !depth.empty();
                localReport.fallbackUsed = !fallbackReason.empty();
                localReport.fallbackReason = fallbackReason;
                if (report)
                {
                    *report = localReport;
                }
                return true;
            };

            if (depth.empty() || localReport.requestedBackend == DenseCloudComputeBackend::Cpu)
            {
                return finishCpu({});
            }

            const auto runCuda = [&](std::string* failure)
            {
                std::string availabilityError;
                if (!DensePointCloudCUDA::isAvailable(options.cudaDeviceIndex, &availabilityError))
                {
                    if (failure)
                    {
                        *failure =
                            availabilityError.empty() ? "CUDA dense-cloud backend is unavailable" : availabilityError;
                    }
                    return false;
                }
                std::string executionError;
                std::vector<DensePoint> candidate = DensePointCloudCUDA::unprojectGPU(
                    depth, mask, cameraModel, colorImg, options.minDepth, options.maxDepth, &executionError, &options);
                if (!executionError.empty())
                {
                    if (failure)
                    {
                        *failure = executionError;
                    }
                    return false;
                }
                *cloud = std::move(candidate);
                localReport.actualBackend = DenseCloudComputeBackend::Cuda;
                localReport.deviceIndex = options.cudaDeviceIndex;
                localReport.deviceName = DensePointCloudCUDA::deviceName(options.cudaDeviceIndex);
                localReport.workSubmitted = true;
                return true;
            };

            const auto runOpenCl = [&](std::string* failure)
            {
                std::string availabilityError;
                if (!DensePointCloudOpenCL::isAvailable(options.openClDeviceIndex, &availabilityError))
                {
                    if (failure)
                    {
                        *failure =
                            availabilityError.empty() ? "OpenCL dense-cloud backend is unavailable" : availabilityError;
                    }
                    return false;
                }
                std::string executionError;
                std::vector<DensePoint> candidate = DensePointCloudOpenCL::unproject(
                    depth, mask, cameraModel, colorImg, options.minDepth, options.maxDepth, &executionError, &options);
                if (!executionError.empty())
                {
                    if (failure)
                    {
                        *failure = executionError;
                    }
                    return false;
                }
                *cloud = std::move(candidate);
                localReport.actualBackend = DenseCloudComputeBackend::OpenCl;
                localReport.deviceIndex = options.openClDeviceIndex;
                localReport.deviceName = DensePointCloudOpenCL::deviceName(options.openClDeviceIndex);
                localReport.workSubmitted = true;
                return true;
            };

            std::string cudaFailure;
            std::string openClFailure;
            bool ok = false;
            switch (localReport.requestedBackend)
            {
            case DenseCloudComputeBackend::Cuda:
                ok = runCuda(&cudaFailure);
                break;
            case DenseCloudComputeBackend::OpenCl:
                ok = runOpenCl(&openClFailure);
                break;
            case DenseCloudComputeBackend::Auto:
                ok = runCuda(&cudaFailure);
                if (!ok)
                {
                    ok = runOpenCl(&openClFailure);
                    if (ok)
                    {
                        localReport.fallbackUsed = true;
                        localReport.fallbackReason = cudaFailure;
                    }
                }
                break;
            case DenseCloudComputeBackend::Cpu:
                break;
            }
            if (ok)
            {
                if (report)
                {
                    *report = localReport;
                }
                return true;
            }

            std::ostringstream reason;
            if (!cudaFailure.empty())
            {
                reason << cudaFailure;
            }
            if (!openClFailure.empty())
            {
                if (!cudaFailure.empty())
                {
                    reason << "; ";
                }
                reason << openClFailure;
            }
            if (localReport.requestedBackend == DenseCloudComputeBackend::Auto)
            {
                return finishCpu(reason.str());
            }
            if (errorMsg)
            {
                *errorMsg = reason.str().empty()
                                ? std::string(denseCloudComputeBackendId(localReport.requestedBackend)) +
                                      " dense-cloud backend failed"
                                : reason.str();
            }
            if (report)
            {
                *report = localReport;
            }
            return false;
        }

        // =============================================================================
        std::vector<DensePoint> DenseCloudBuilder::merge(const std::vector<std::vector<DensePoint>>& clouds)
        {
            std::vector<DensePoint> result;
            size_t total = 0;
            for (const auto& c : clouds)
            {
                total += c.size();
            }
            result.reserve(total);
            for (const auto& c : clouds)
            {
                result.insert(result.end(), c.begin(), c.end());
            }
            return result;
        }

        // =============================================================================
        bool
        DenseCloudBuilder::savePLY(const std::string& path, const std::vector<DensePoint>& cloud, std::string* errorMsg)
        {
            try
            {
                plapoint::io::writePly(xjw::common::io::toNativeNarrowPath(path),
                                       buildPointCloud(cloud),
                                       plapoint::io::PlyFormat::BinaryLE);
                return true;
            }
            catch (const std::exception& e)
            {
                if (errorMsg)
                    *errorMsg = e.what();
                return false;
            }
        }

        // =============================================================================
        // 辅助：在 PlaScan 的 DensePoint 和 plapoint::PointCloud 之间转换，保留颜色属性。
        // =============================================================================
        namespace
        {
            using DensePlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

            static DensePlaCloud buildPointCloud(const std::vector<DensePoint>& cloud)
            {
                plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(cloud.size(), 3);
                plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(cloud.size(), 3);
                for (size_t i = 0; i < cloud.size(); ++i)
                {
                    const auto row = static_cast<plamatrix::Index>(i);
                    pts(row, 0) = cloud[i].x;
                    pts(row, 1) = cloud[i].y;
                    pts(row, 2) = cloud[i].z;
                    colors(row, 0) = cloud[i].r;
                    colors(row, 1) = cloud[i].g;
                    colors(row, 2) = cloud[i].b;
                }
                DensePlaCloud pointCloud(std::move(pts));
                pointCloud.setColors(std::move(colors));
                return pointCloud;
            }

            static std::vector<DensePoint> fromPointCloud(const DensePlaCloud& cloud)
            {
                std::vector<DensePoint> points;
                points.reserve(cloud.size());
                const auto& matrix = cloud.points();
                for (std::size_t i = 0; i < cloud.size(); ++i)
                {
                    const auto row = static_cast<plamatrix::Index>(i);
                    DensePoint point;
                    point.x = matrix.getValue(row, 0);
                    point.y = matrix.getValue(row, 1);
                    point.z = matrix.getValue(row, 2);
                    if (cloud.hasColors())
                    {
                        point.r = cloud.colors()->getValue(row, 0);
                        point.g = cloud.colors()->getValue(row, 1);
                        point.b = cloud.colors()->getValue(row, 2);
                    }
                    points.push_back(point);
                }
                return points;
            }

            const char* processingDeviceName(plapoint::ProcessingDevice device)
            {
                switch (device)
                {
                case plapoint::ProcessingDevice::CPU:
                    return "CPU";
                case plapoint::ProcessingDevice::CUDA:
                    return "CUDA";
                case plapoint::ProcessingDevice::OpenCL:
                    return "OpenCL";
                case plapoint::ProcessingDevice::Auto:
                    return "Auto";
                }
                return "Unknown";
            }

            void initializeProcessingReport(plapoint::ProcessingReport* report,
                                            plapoint::ProcessingDevice requestedDevice)
            {
                if (!report)
                {
                    return;
                }

                *report = {};
                report->requestedDevice = requestedDevice;
                report->actualDevice = plapoint::ProcessingDevice::CPU;
                report->usedDevice = plapoint::ProcessingDevice::CPU;
            }

            void markProcessingSkipped(plapoint::ProcessingReport* report, const char* reason)
            {
                if (report)
                {
                    report->fallbackReason = reason;
                }
            }

        } // anonymous namespace

        // =============================================================================
        // Voxel Downsample
        // =============================================================================
        std::vector<DensePoint> DenseCloudBuilder::voxelDownsample(const std::vector<DensePoint>& cloud,
                                                                   float voxelSize,
                                                                   plapoint::ProcessingDevice processingDevice,
                                                                   plapoint::ProcessingReport* processingReport)
        {
            initializeProcessingReport(processingReport, processingDevice);
            if (cloud.empty() || voxelSize <= 0.0f)
            {
                markProcessingSkipped(processingReport, "skipped: empty cloud or invalid voxel size");
                return cloud;
            }

            const int N = static_cast<int>(cloud.size());
            auto t0 = std::chrono::steady_clock::now();
            LOG_INFO("[Voxel] 开始 plapoint 体素下采样: %d 点, voxelSize=%.4f", N, voxelSize);

            DensePlaCloud pointCloud = buildPointCloud(cloud);
            plapoint::ProcessingReport report;
            DensePlaCloud filtered = plapoint::voxelDownsample(pointCloud, voxelSize, processingDevice, &report);
            if (processingReport)
            {
                *processingReport = report;
            }
            std::vector<DensePoint> result = fromPointCloud(filtered);

            auto t1 = std::chrono::steady_clock::now();
            LOG_INFO("[Voxel] plapoint 下采样完成: %d → %d 点 (请求=%s, 实际=%s) 耗时 %.3f s",
                     N,
                     static_cast<int>(result.size()),
                     processingDeviceName(report.requestedDevice),
                     processingDeviceName(report.actualDevice),
                     std::chrono::duration<double>(t1 - t0).count());
            return result;
        }

        // =============================================================================
        // Statistical Outlier Removal
        // =============================================================================
        std::vector<DensePoint>
        DenseCloudBuilder::statisticalOutlierRemoval(const std::vector<DensePoint>& cloud,
                                                     int kNeighbors,
                                                     float stdRatio,
                                                     plapoint::ProcessingDevice processingDevice,
                                                     plapoint::ProcessingReport* processingReport)
        {
            initializeProcessingReport(processingReport, processingDevice);
            if (cloud.size() < (size_t)kNeighbors + 1)
            {
                markProcessingSkipped(processingReport, "skipped: point count is smaller than k + 1");
                return cloud;
            }

            const int N = (int)cloud.size();
            auto t0 = std::chrono::steady_clock::now();

            LOG_INFO("[SOR] 开始 plapoint 统计离群点过滤: %d 点, k=%d, stdRatio=%.2f", N, kNeighbors, stdRatio);

            DensePlaCloud pointCloud = buildPointCloud(cloud);
            std::vector<int> removedIndices;
            plapoint::ProcessingReport report;
            DensePlaCloud filtered = plapoint::statisticalOutlierRemoval(
                pointCloud, kNeighbors, stdRatio, processingDevice, &removedIndices, &report);
            if (processingReport)
            {
                *processingReport = report;
            }
            std::vector<DensePoint> result = fromPointCloud(filtered);

            auto t3 = std::chrono::steady_clock::now();
            LOG_INFO("[SOR] plapoint 过滤完成: %d → %d 点 (移除 %zu, %.1f%%, 请求=%s, 实际=%s) 总耗时 %.3f s",
                     N,
                     (int)result.size(),
                     removedIndices.size(),
                     100.f * removedIndices.size() / N,
                     processingDeviceName(report.requestedDevice),
                     processingDeviceName(report.actualDevice),
                     std::chrono::duration<double>(t3 - t0).count());
            return result;
        }

        // =============================================================================
        // Radius Outlier Removal
        // =============================================================================
        std::vector<DensePoint> DenseCloudBuilder::radiusOutlierRemoval(const std::vector<DensePoint>& cloud,
                                                                        float radius,
                                                                        int minNeighbors,
                                                                        plapoint::ProcessingDevice processingDevice,
                                                                        plapoint::ProcessingReport* processingReport)
        {
            initializeProcessingReport(processingReport, processingDevice);
            if (cloud.empty())
            {
                markProcessingSkipped(processingReport, "skipped: empty cloud");
                return cloud;
            }

            const int N = (int)cloud.size();
            auto t0 = std::chrono::steady_clock::now();

            LOG_INFO("[RadiusOR] 开始 plapoint 半径离群点过滤: %d 点, radius=%.4f, minNeighbors=%d",
                     N,
                     radius,
                     minNeighbors);

            DensePlaCloud pointCloud = buildPointCloud(cloud);
            std::vector<int> removedIndices;
            plapoint::ProcessingReport report;
            DensePlaCloud filtered = plapoint::radiusOutlierRemoval(
                pointCloud, radius, minNeighbors, processingDevice, &removedIndices, &report);
            if (processingReport)
            {
                *processingReport = report;
            }
            std::vector<DensePoint> result = fromPointCloud(filtered);

            auto t2 = std::chrono::steady_clock::now();
            LOG_INFO("[RadiusOR] plapoint 过滤完成: %d → %d 点 (移除 %zu, %.1f%%, 请求=%s, 实际=%s) 耗时 %.3f s",
                     N,
                     (int)result.size(),
                     removedIndices.size(),
                     100.f * removedIndices.size() / N,
                     processingDeviceName(report.requestedDevice),
                     processingDeviceName(report.actualDevice),
                     std::chrono::duration<double>(t2 - t0).count());
            return result;
        }

    } // namespace mvs
} // namespace xjw
