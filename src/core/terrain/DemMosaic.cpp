#include "DemMosaic.h"

#include "TerrainGpuBackend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <string>

namespace xjw
{

    namespace
    {

        bool gridCompatible(const DemGridData& a, const DemGridData& b)
        {
            constexpr double kEps = 1e-9;
            return a.width == b.width && a.height == b.height && std::abs(a.minX - b.minX) <= kEps &&
                   std::abs(a.minY - b.minY) <= kEps && std::abs(a.stepX - b.stepX) <= kEps &&
                   std::abs(a.stepY - b.stepY) <= kEps;
        }

        bool validRasterTypes(const DemGridData& grid)
        {
            const bool base_valid = grid.elevation.type() == CV_32FC1 && grid.validMask.type() == CV_8UC1;
            const bool confidence_valid =
                grid.confidence.empty() || (grid.confidence.type() == CV_32FC1 && grid.confidence.rows == grid.height &&
                                            grid.confidence.cols == grid.width);
            const bool error_valid = grid.triangulationError.empty() || (grid.triangulationError.type() == CV_32FC1 &&
                                                                         grid.triangulationError.rows == grid.height &&
                                                                         grid.triangulationError.cols == grid.width);
            return base_valid && confidence_valid && error_valid;
        }

        bool gpuMosaicIndexingFits(const std::vector<DemGridData>& tiles, QString* errorMsg)
        {
            const std::size_t maximum_count = static_cast<std::size_t>(std::numeric_limits<int>::max());
            const std::size_t width = static_cast<std::size_t>(tiles.front().width);
            const std::size_t height = static_cast<std::size_t>(tiles.front().height);
            if (width != 0 && height > maximum_count / width)
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("DEM mosaic 单瓦片像元总数超过 32 位 kernel 索引限制");
                }
                return false;
            }

            const std::size_t pixel_count = width * height;
            if (tiles.size() > maximum_count || (pixel_count != 0 && tiles.size() > maximum_count / pixel_count))
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("DEM mosaic GPU 输入总元素数（%1 个瓦片 × %2 个像元）超过 32 位 "
                                               "kernel 索引限制")
                                    .arg(static_cast<qulonglong>(tiles.size()))
                                    .arg(static_cast<qulonglong>(pixel_count));
                }
                return false;
            }
            return true;
        }

        template <typename T> void appendRaster(const cv::Mat& raster, std::vector<T>* values)
        {
            const std::size_t row_size = static_cast<std::size_t>(raster.cols);
            const std::size_t offset = values->size();
            values->resize(offset + row_size * static_cast<std::size_t>(raster.rows));
            for (int row = 0; row < raster.rows; ++row)
            {
                std::memcpy(values->data() + offset + static_cast<std::size_t>(row) * row_size,
                            raster.ptr<T>(row),
                            row_size * sizeof(T));
            }
        }

        template <typename T> void copyRaster(const std::vector<T>& values, cv::Mat* raster)
        {
            const std::size_t row_size = static_cast<std::size_t>(raster->cols);
            for (int row = 0; row < raster->rows; ++row)
            {
                std::memcpy(raster->ptr<T>(row),
                            values.data() + static_cast<std::size_t>(row) * row_size,
                            row_size * sizeof(T));
            }
        }

        terrain_internal::PackedDemMosaic packMosaic(const std::vector<DemGridData>& tiles,
                                                     DemMosaicBlendMode blendMode)
        {
            terrain_internal::PackedDemMosaic packed;
            packed.width = tiles.front().width;
            packed.height = tiles.front().height;
            packed.tileCount = static_cast<int>(tiles.size());
            packed.blendMode = static_cast<int>(blendMode);
            const std::size_t pixel_count =
                static_cast<std::size_t>(packed.width) * static_cast<std::size_t>(packed.height);
            packed.elevation.reserve(pixel_count * tiles.size());
            packed.valid.reserve(pixel_count * tiles.size());
            packed.confidence.reserve(pixel_count * tiles.size());
            packed.triangulationError.reserve(pixel_count * tiles.size());
            for (const DemGridData& tile : tiles)
            {
                appendRaster<float>(tile.elevation, &packed.elevation);
                appendRaster<std::uint8_t>(tile.validMask, &packed.valid);
                if (tile.confidence.empty())
                {
                    packed.confidence.insert(packed.confidence.end(), pixel_count, 1.0f);
                }
                else
                {
                    appendRaster<float>(tile.confidence, &packed.confidence);
                }
                if (tile.triangulationError.empty())
                {
                    packed.triangulationError.insert(packed.triangulationError.end(), pixel_count, 1.0f);
                }
                else
                {
                    appendRaster<float>(tile.triangulationError, &packed.triangulationError);
                }
            }
            return packed;
        }

        void initializeOutput(const DemGridData& reference, DemGridData* output)
        {
            *output = DemGridData();
            output->width = reference.width;
            output->height = reference.height;
            output->minX = reference.minX;
            output->minY = reference.minY;
            output->stepX = reference.stepX;
            output->stepY = reference.stepY;
            output->projection = reference.projection;
            output->elevation = cv::Mat::zeros(output->height, output->width, CV_32FC1);
            output->validMask = cv::Mat::zeros(output->height, output->width, CV_8UC1);
            output->coverageMask = cv::Mat::zeros(output->height, output->width, CV_8UC1);
            output->pointCount = cv::Mat::zeros(output->height, output->width, CV_32SC1);
            output->confidence = cv::Mat::zeros(output->height, output->width, CV_32FC1);
            output->triangulationError = cv::Mat::zeros(output->height, output->width, CV_32FC1);
        }

        void unpackMosaic(const terrain_internal::PackedDemMosaicResult& packed, DemGridData* output)
        {
            copyRaster(packed.elevation, &output->elevation);
            copyRaster(packed.valid, &output->validMask);
            copyRaster(packed.valid, &output->coverageMask);
            copyRaster(packed.pointCount, &output->pointCount);
            copyRaster(packed.confidence, &output->confidence);
            copyRaster(packed.triangulationError, &output->triangulationError);
        }

        bool isValidCell(const DemGridData& grid, int row, int col)
        {
            if (grid.elevation.empty() || row < 0 || col < 0 || row >= grid.elevation.rows ||
                col >= grid.elevation.cols)
            {
                return false;
            }
            if (!grid.validMask.empty() && grid.validMask.at<uchar>(row, col) == 0)
            {
                return false;
            }
            return std::isfinite(grid.elevation.at<float>(row, col));
        }

        float medianValue(std::vector<float> values)
        {
            if (values.empty())
            {
                return 0.0f;
            }
            std::sort(values.begin(), values.end());
            const size_t mid = values.size() / 2;
            if (values.size() % 2 == 1)
            {
                return values[mid];
            }
            return (values[mid - 1] + values[mid]) * 0.5f;
        }

        float confidenceAt(const DemGridData& grid, int row, int col)
        {
            if (grid.hasConfidence())
            {
                return std::max(0.0f, grid.confidence.at<float>(row, col));
            }
            return 1.0f;
        }

        float errorAt(const DemGridData& grid, int row, int col)
        {
            if (grid.hasTriangulationError())
            {
                return std::max(0.0f, grid.triangulationError.at<float>(row, col));
            }
            return 1.0f;
        }

        float blendValues(const std::vector<float>& values, const std::vector<float>& weights, DemMosaicBlendMode mode)
        {
            if (values.empty())
            {
                return 0.0f;
            }

            switch (mode)
            {
            case DemMosaicBlendMode::First:
                return values.front();
            case DemMosaicBlendMode::Last:
                return values.back();
            case DemMosaicBlendMode::Median:
                return medianValue(values);
            case DemMosaicBlendMode::Min:
                return *std::min_element(values.begin(), values.end());
            case DemMosaicBlendMode::Max:
                return *std::max_element(values.begin(), values.end());
            case DemMosaicBlendMode::ConfidenceWeighted:
            case DemMosaicBlendMode::InverseErrorWeighted:
            {
                float weightedSum = 0.0f;
                float weightSum = 0.0f;
                for (size_t i = 0; i < values.size(); ++i)
                {
                    const float weight = i < weights.size() ? weights[i] : 1.0f;
                    weightedSum += values[i] * weight;
                    weightSum += weight;
                }
                return weightSum > 0.0f ? weightedSum / weightSum : values.back();
            }
            case DemMosaicBlendMode::Mean:
            default:
                return std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
            }
        }

    } // namespace

    bool DemMosaic::mosaicSameGrid(const std::vector<DemGridData>& tiles,
                                   DemMosaicBlendMode blendMode,
                                   DemGridData* output,
                                   QString* errorMsg,
                                   const TerrainComputeOptions& computeOptions,
                                   TerrainComputeExecution* computeExecution)
    {
        if (!output)
        {
            if (errorMsg)
                *errorMsg = QStringLiteral("DEM mosaic 输出对象为空");
            return false;
        }
        if (tiles.empty())
        {
            if (errorMsg)
                *errorMsg = QStringLiteral("DEM mosaic 输入瓦片为空");
            return false;
        }
        if (!tiles.front().isValid() || !validRasterTypes(tiles.front()))
        {
            if (errorMsg)
                *errorMsg = QStringLiteral("DEM mosaic 第一个瓦片无效");
            return false;
        }

        for (size_t i = 1; i < tiles.size(); ++i)
        {
            if (!tiles[i].isValid() || !validRasterTypes(tiles[i]))
            {
                if (errorMsg)
                    *errorMsg = QStringLiteral("DEM mosaic 瓦片 %1 无效").arg(i);
                return false;
            }
            if (!gridCompatible(tiles.front(), tiles[i]))
            {
                if (errorMsg)
                    *errorMsg = QStringLiteral("DEM mosaic 瓦片 %1 网格不一致").arg(i);
                return false;
            }
        }
        if (computeOptions.deviceIndex < -1)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("DEM mosaic 设备索引必须为 -1 或非负整数");
            }
            return false;
        }

        TerrainComputeExecution execution;
        QStringList fallback_reasons;
        bool gpu_indexing_compatible = true;
        if (computeOptions.backend != TerrainComputeBackend::Cpu)
        {
            QString indexing_error;
            gpu_indexing_compatible = gpuMosaicIndexingFits(tiles, &indexing_error);
            if (!gpu_indexing_compatible)
            {
                if (computeOptions.backend == TerrainComputeBackend::Cuda ||
                    computeOptions.backend == TerrainComputeBackend::OpenCl)
                {
                    if (errorMsg)
                    {
                        *errorMsg = QStringLiteral("显式 %1 DEM mosaic 后端失败: %2")
                                        .arg(terrainComputeBackendDisplayName(computeOptions.backend), indexing_error);
                    }
                    return false;
                }
                fallback_reasons.push_back(indexing_error);
            }
        }

        initializeOutput(tiles.front(), output);
        bool completed_on_gpu = false;
        std::optional<terrain_internal::PackedDemMosaic> packed;
        auto try_gpu = [&](TerrainComputeBackend backend) -> bool
        {
            if (!gpu_indexing_compatible)
            {
                return false;
            }
            const terrain_internal::TerrainDeviceInfo device =
                backend == TerrainComputeBackend::Cuda
                    ? terrain_internal::queryTerrainCudaDevice(computeOptions.deviceIndex)
                    : terrain_internal::queryTerrainOpenClMosaicDevice(computeOptions.deviceIndex);
            if (!device.available)
            {
                fallback_reasons.push_back(
                    QStringLiteral("%1 后端或设备不可用: %2")
                        .arg(terrainComputeBackendDisplayName(backend), QString::fromStdString(device.error)));
                return false;
            }
            if (!packed.has_value())
            {
                packed = packMosaic(tiles, blendMode);
            }
            terrain_internal::PackedDemMosaicResult gpu_output;
            std::string backend_error;
            const bool ok = backend == TerrainComputeBackend::Cuda
                                ? terrain_internal::runTerrainCudaDemMosaic(
                                      *packed, device.resolvedIndex, &gpu_output, &backend_error)
                                : terrain_internal::runTerrainOpenClDemMosaic(
                                      *packed, device.resolvedIndex, &gpu_output, &backend_error);
            if (!ok)
            {
                fallback_reasons.push_back(
                    QStringLiteral("%1 执行失败: %2")
                        .arg(terrainComputeBackendDisplayName(backend), QString::fromStdString(backend_error)));
                return false;
            }
            unpackMosaic(gpu_output, output);
            execution.backend = backend;
            execution.deviceIndex = device.resolvedIndex;
            execution.deviceName = QString::fromStdString(device.name);
            completed_on_gpu = true;
            return true;
        };

        if (computeOptions.backend == TerrainComputeBackend::Cuda ||
            computeOptions.backend == TerrainComputeBackend::OpenCl)
        {
            if (!try_gpu(computeOptions.backend))
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("显式 %1 DEM mosaic 后端失败: %2")
                                    .arg(terrainComputeBackendDisplayName(computeOptions.backend),
                                         fallback_reasons.join(QStringLiteral("；")));
                }
                return false;
            }
        }
        else if (computeOptions.backend == TerrainComputeBackend::Auto && !try_gpu(TerrainComputeBackend::Cuda))
        {
            try_gpu(TerrainComputeBackend::OpenCl);
        }

        if (completed_on_gpu)
        {
            execution.fallbackReason = fallback_reasons.join(QStringLiteral("；"));
            if (computeExecution)
            {
                *computeExecution = execution;
            }
            return true;
        }

        execution.backend = TerrainComputeBackend::Cpu;
        execution.deviceIndex = -1;
        execution.deviceName = QStringLiteral("CPU");
        execution.fallbackReason = fallback_reasons.join(QStringLiteral("；"));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int row = 0; row < output->height; ++row)
        {
            for (int col = 0; col < output->width; ++col)
            {
                std::vector<float> values;
                std::vector<float> weights;
                float confidenceSum = 0.0f;
                float errorSum = 0.0f;

                for (const DemGridData& tile : tiles)
                {
                    if (!isValidCell(tile, row, col))
                    {
                        continue;
                    }
                    values.push_back(tile.elevation.at<float>(row, col));
                    confidenceSum += confidenceAt(tile, row, col);
                    errorSum += errorAt(tile, row, col);

                    if (blendMode == DemMosaicBlendMode::ConfidenceWeighted)
                    {
                        weights.push_back(confidenceAt(tile, row, col));
                    }
                    else if (blendMode == DemMosaicBlendMode::InverseErrorWeighted)
                    {
                        weights.push_back(1.0f / std::max(errorAt(tile, row, col), 1e-6f));
                    }
                }

                if (values.empty())
                {
                    continue;
                }

                output->elevation.at<float>(row, col) = blendValues(values, weights, blendMode);
                output->validMask.at<uchar>(row, col) = 255;
                output->coverageMask.at<uchar>(row, col) = 255;
                output->pointCount.at<int>(row, col) = static_cast<int>(values.size());
                output->confidence.at<float>(row, col) = confidenceSum / static_cast<float>(values.size());
                output->triangulationError.at<float>(row, col) = errorSum / static_cast<float>(values.size());
            }
        }

        if (computeExecution)
        {
            *computeExecution = execution;
        }

        return true;
    }

} // namespace xjw
