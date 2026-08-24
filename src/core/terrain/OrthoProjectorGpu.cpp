#include "OrthoProjectorGpu.h"

#include "TerrainGpuBackend.h"

#include <opencv2/core.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <string>

namespace xjw::ortho_internal
{

    namespace
    {

        template <typename T> void appendMat(const cv::Mat& mat, int columnsPerPixel, std::vector<T>* values)
        {
            const std::size_t row_values =
                static_cast<std::size_t>(mat.cols) * static_cast<std::size_t>(columnsPerPixel);
            const std::size_t old_size = values->size();
            values->resize(old_size + row_values * static_cast<std::size_t>(mat.rows));
            T* destination = values->data() + old_size;
            for (int row = 0; row < mat.rows; ++row)
            {
                std::memcpy(
                    destination + static_cast<std::size_t>(row) * row_values, mat.ptr<T>(row), row_values * sizeof(T));
            }
        }

        template <typename T> void copyVectorToMat(const std::vector<T>& values, int columnsPerPixel, cv::Mat* mat)
        {
            const std::size_t row_values =
                static_cast<std::size_t>(mat->cols) * static_cast<std::size_t>(columnsPerPixel);
            for (int row = 0; row < mat->rows; ++row)
            {
                std::memcpy(mat->ptr<T>(row),
                            values.data() + static_cast<std::size_t>(row) * row_values,
                            row_values * sizeof(T));
            }
        }

        constexpr std::size_t kMaximumKernelIndexCount = static_cast<std::size_t>(std::numeric_limits<int>::max());

        bool checkedKernelElementCount(
            std::size_t first, std::size_t second, const QString& label, std::size_t* count, QString* errorMsg)
        {
            if (first != 0 && second > kMaximumKernelIndexCount / first)
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("正射 GPU %1 元素总数超过 32 位 kernel 索引限制").arg(label);
                }
                return false;
            }
            if (count)
            {
                *count = first * second;
            }
            return true;
        }

        bool checkedBufferAppend(
            std::size_t currentSize, std::size_t appendSize, const QString& label, int* offset, QString* errorMsg)
        {
            if (currentSize > kMaximumKernelIndexCount || appendSize > kMaximumKernelIndexCount - currentSize)
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("正射 GPU %1 缓冲总长度（当前 %2，追加 %3）超过 32 位 kernel 索引限制")
                                    .arg(label)
                                    .arg(static_cast<qulonglong>(currentSize))
                                    .arg(static_cast<qulonglong>(appendSize));
                }
                return false;
            }
            *offset = static_cast<int>(currentSize);
            return true;
        }

        bool packInput(const DemGridData& demGrid,
                       const OrthoOutputGrid& outputGrid,
                       const std::vector<LoadedFrame>& frames,
                       const OrthoGenerationOptions& options,
                       double elevationOffset,
                       terrain_internal::PackedOrthoProjection* packed,
                       QString* errorMsg)
        {
            std::size_t dem_pixel_count = 0;
            std::size_t output_pixel_count = 0;
            std::size_t camera_value_count = 0;
            std::size_t camera_metadata_count = 0;
            if (demGrid.width <= 0 || demGrid.height <= 0 || outputGrid.reference.width <= 0 ||
                outputGrid.reference.height <= 0)
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("正射 GPU 输入或输出网格尺寸无效");
                }
                return false;
            }
            if (!checkedKernelElementCount(static_cast<std::size_t>(demGrid.width),
                                           static_cast<std::size_t>(demGrid.height),
                                           QStringLiteral("DEM"),
                                           &dem_pixel_count,
                                           errorMsg) ||
                !checkedKernelElementCount(static_cast<std::size_t>(outputGrid.reference.width),
                                           static_cast<std::size_t>(outputGrid.reference.height),
                                           QStringLiteral("输出像元"),
                                           &output_pixel_count,
                                           errorMsg) ||
                !checkedKernelElementCount(output_pixel_count, 3, QStringLiteral("输出 BGR"), nullptr, errorMsg) ||
                !checkedKernelElementCount(frames.size(),
                                           terrain_internal::kTerrainCameraValueStride,
                                           QStringLiteral("相机参数"),
                                           &camera_value_count,
                                           errorMsg) ||
                !checkedKernelElementCount(frames.size(),
                                           terrain_internal::kTerrainCameraMetadataStride,
                                           QStringLiteral("相机元数据"),
                                           &camera_metadata_count,
                                           errorMsg))
            {
                return false;
            }
            if (demGrid.elevation.type() != CV_32FC1 || demGrid.validMask.type() != CV_8UC1 ||
                demGrid.elevation.total() != dem_pixel_count || demGrid.validMask.total() != dem_pixel_count)
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("正射 GPU DEM 高程或有效掩膜的类型、尺寸与网格不一致");
                }
                return false;
            }

            packed->demWidth = demGrid.width;
            packed->demHeight = demGrid.height;
            packed->outputWidth = outputGrid.reference.width;
            packed->outputHeight = outputGrid.reference.height;
            packed->frameCount = static_cast<int>(frames.size());
            packed->blendMode = static_cast<int>(options.blendMode);
            packed->demMinX = demGrid.minX;
            packed->demMinY = demGrid.minY;
            packed->demStepX = demGrid.stepX;
            packed->demStepY = demGrid.stepY;
            packed->outputMinEdgeX = outputGrid.minEdgeX;
            packed->outputMinEdgeY = outputGrid.minEdgeY;
            packed->outputStepX = outputGrid.reference.stepX;
            packed->outputStepY = outputGrid.reference.stepY;
            packed->elevationOffset = elevationOffset;
            appendMat<float>(demGrid.elevation, 1, &packed->demElevation);
            appendMat<std::uint8_t>(demGrid.validMask, 1, &packed->demValid);

            packed->cameraValues.reserve(camera_value_count);
            packed->cameraMetadata.reserve(camera_metadata_count);
            packed->maskData.push_back(0);
            for (const LoadedFrame& frame : frames)
            {
                if (frame.imageBgr.type() != CV_8UC3 || frame.imageBgr.rows <= 0 || frame.imageBgr.cols <= 0)
                {
                    if (errorMsg)
                    {
                        *errorMsg = QStringLiteral("正射 GPU 输入影像必须为非空 CV_8UC3");
                    }
                    return false;
                }
                if (!frame.exclusionMask.empty() &&
                    (frame.exclusionMask.type() != CV_8UC1 || frame.exclusionMask.size() != frame.imageBgr.size()))
                {
                    if (errorMsg)
                    {
                        *errorMsg = QStringLiteral("正射 GPU 排除掩膜必须为与输入影像同尺寸的 CV_8UC1");
                    }
                    return false;
                }

                std::size_t image_pixel_count = 0;
                std::size_t image_byte_count = 0;
                if (!checkedKernelElementCount(static_cast<std::size_t>(frame.imageBgr.cols),
                                               static_cast<std::size_t>(frame.imageBgr.rows),
                                               QStringLiteral("单幅输入影像像元"),
                                               &image_pixel_count,
                                               errorMsg) ||
                    !checkedKernelElementCount(
                        image_pixel_count, 3, QStringLiteral("单幅输入影像 BGR"), &image_byte_count, errorMsg))
                {
                    return false;
                }
                int image_offset = 0;
                int mask_offset = -1;
                if (!checkedBufferAppend(packed->imageData.size(),
                                         image_byte_count,
                                         QStringLiteral("输入影像"),
                                         &image_offset,
                                         errorMsg))
                {
                    return false;
                }
                appendMat<std::uint8_t>(frame.imageBgr, 3, &packed->imageData);
                if (!frame.exclusionMask.empty())
                {
                    if (!checkedBufferAppend(packed->maskData.size(),
                                             image_pixel_count,
                                             QStringLiteral("排除掩膜"),
                                             &mask_offset,
                                             errorMsg))
                    {
                        return false;
                    }
                    appendMat<std::uint8_t>(frame.exclusionMask, 1, &packed->maskData);
                }

                const std::array<double, 9> rotation = frame.input.camera.cameraToWorldRotation();
                const std::array<double, 3> center = frame.input.camera.cameraCenter();
                const FramePinholeCamera::Distortion distortion = frame.input.camera.distortion();
                packed->cameraValues.insert(packed->cameraValues.end(), rotation.begin(), rotation.end());
                packed->cameraValues.insert(packed->cameraValues.end(), center.begin(), center.end());
                packed->cameraValues.push_back(frame.input.camera.focalX());
                packed->cameraValues.push_back(frame.input.camera.focalY());
                packed->cameraValues.push_back(frame.input.camera.principalX());
                packed->cameraValues.push_back(frame.input.camera.principalY());
                packed->cameraValues.push_back(distortion.radialK1);
                packed->cameraValues.push_back(distortion.radialK2);
                packed->cameraValues.push_back(distortion.radialK3);
                packed->cameraValues.push_back(distortion.tangentialP1);
                packed->cameraValues.push_back(distortion.tangentialP2);
                packed->cameraValues.push_back(frame.gain);
                packed->cameraValues.push_back(frame.sharpnessWeight);
                packed->cameraMetadata.push_back(frame.imageBgr.cols);
                packed->cameraMetadata.push_back(frame.imageBgr.rows);
                packed->cameraMetadata.push_back(image_offset);
                packed->cameraMetadata.push_back(mask_offset);
                packed->cameraMetadata.push_back(frame.input.camera.uAxisSign());
                packed->cameraMetadata.push_back(frame.input.camera.vAxisSign());
                packed->cameraMetadata.push_back(frame.input.camera.depthAxisFlipped() ? 1 : 0);
            }
            return true;
        }

    } // namespace

    bool projectPixelsOnGpu(const DemGridData& demGrid,
                            const OrthoOutputGrid& outputGrid,
                            std::vector<LoadedFrame>* frames,
                            const OrthoGenerationOptions& options,
                            double demElevationOffset,
                            TerrainComputeBackend backend,
                            int deviceIndex,
                            OrthoProjectionResult* result,
                            qint64* surfacePixelCount,
                            int* resolvedDeviceIndex,
                            QString* deviceName,
                            QString* errorMsg)
    {
        terrain_internal::PackedOrthoProjection input;
        if (!packInput(demGrid, outputGrid, *frames, options, demElevationOffset, &input, errorMsg))
        {
            return false;
        }

        const terrain_internal::TerrainDeviceInfo device =
            backend == TerrainComputeBackend::Cuda ? terrain_internal::queryTerrainCudaDevice(deviceIndex)
                                                   : terrain_internal::queryTerrainOpenClDevice(deviceIndex);
        if (!device.available)
        {
            if (errorMsg)
            {
                *errorMsg = QString::fromStdString(device.error);
            }
            return false;
        }

        terrain_internal::PackedOrthoProjectionResult projected;
        std::string backend_error;
        const bool ok =
            backend == TerrainComputeBackend::Cuda
                ? terrain_internal::runTerrainCudaOrtho(input, device.resolvedIndex, &projected, &backend_error)
                : terrain_internal::runTerrainOpenClOrtho(input, device.resolvedIndex, &projected, &backend_error);
        if (!ok)
        {
            if (errorMsg)
            {
                *errorMsg = QString::fromStdString(backend_error);
            }
            return false;
        }

        result->imageBgr = cv::Mat(input.outputHeight, input.outputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        result->surfaceMask = cv::Mat(input.outputHeight, input.outputWidth, CV_8UC1, cv::Scalar(0));
        result->coverageMask = cv::Mat(input.outputHeight, input.outputWidth, CV_8UC1, cv::Scalar(0));
        copyVectorToMat(projected.imageBgr, 3, &result->imageBgr);
        copyVectorToMat(projected.surfaceMask, 1, &result->surfaceMask);
        copyVectorToMat(projected.coverageMask, 1, &result->coverageMask);
        result->filledPixelCount = cv::countNonZero(result->coverageMask);
        *surfacePixelCount = cv::countNonZero(result->surfaceMask);
        for (std::size_t index = 0; index < frames->size() && index < projected.contributedFrames.size(); ++index)
        {
            (*frames)[index].contributed = projected.contributedFrames[index] != 0;
        }
        if (resolvedDeviceIndex)
        {
            *resolvedDeviceIndex = device.resolvedIndex;
        }
        if (deviceName)
        {
            *deviceName = QString::fromStdString(device.name);
        }
        return true;
    }

} // namespace xjw::ortho_internal
