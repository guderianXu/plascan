#include "DemGenerator.h"

#include <plapoint/mesh/height_grid.h>

#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw
{

namespace
{

struct PointCloudBounds
{
    bool valid = false;
    struct { float x = 0, y = 0, z = 0; } minCorner;
    struct { float x = 0, y = 0, z = 0; } maxCorner;
};

PointCloudBounds computeBounds(const PlaPointCloud &pointCloud)
{
    PointCloudBounds result;
    if (pointCloud.size() == 0)
    {
        return result;
    }

    float minX = std::numeric_limits<float>::max(), maxX = -std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max(), maxY = -std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max(), maxZ = -std::numeric_limits<float>::max();
    bool hasFinitePoint = false;
    for (size_t i = 0; i < pointCloud.size(); ++i)
    {
        auto pt = pointCloud[i];
        if (!std::isfinite(pt.x()) || !std::isfinite(pt.y()) || !std::isfinite(pt.z()))
        {
            continue;
        }

        minX = std::min(minX, pt.x()); maxX = std::max(maxX, pt.x());
        minY = std::min(minY, pt.y()); maxY = std::max(maxY, pt.y());
        minZ = std::min(minZ, pt.z()); maxZ = std::max(maxZ, pt.z());
        hasFinitePoint = true;
    }

    if (!hasFinitePoint)
    {
        return result;
    }

    result.valid = true;
    result.minCorner = {minX, minY, minZ};
    result.maxCorner = {maxX, maxY, maxZ};
    return result;
}

plapoint::mesh::ElevationAggregation toPlaAggregation(
    DemGenerationOptions::ElevationAggregation aggregation)
{
    if (aggregation == DemGenerationOptions::ElevationAggregation::Min)
    {
        return plapoint::mesh::ElevationAggregation::Min;
    }
    if (aggregation == DemGenerationOptions::ElevationAggregation::Max)
    {
        return plapoint::mesh::ElevationAggregation::Max;
    }
    return plapoint::mesh::ElevationAggregation::Mean;
}

void assignHeightGridToDemGrid(const plapoint::mesh::HeightGrid<float> &grid,
                               const DemProjectionParameters &projection,
                               DemGridData *demGrid)
{
    demGrid->width = grid.width;
    demGrid->height = grid.height;
    demGrid->minX = grid.minX;
    demGrid->minY = grid.minY;
    demGrid->stepX = grid.stepX;
    demGrid->stepY = grid.stepY;
    demGrid->projection = projection;
    demGrid->elevation = cv::Mat(grid.height, grid.width, CV_32F, cv::Scalar(0));
    demGrid->worldX = cv::Mat(grid.height, grid.width, CV_32F, cv::Scalar(0));
    demGrid->worldY = cv::Mat(grid.height, grid.width, CV_32F, cv::Scalar(0));
    demGrid->validMask = cv::Mat(grid.height, grid.width, CV_8U, cv::Scalar(0));
    demGrid->triangulationError.release();
    demGrid->color.release();
    if (grid.hasColors())
    {
        demGrid->color = cv::Mat(grid.height, grid.width, CV_8UC3, cv::Scalar(128, 128, 128));
    }

    for (int row = 0; row < grid.height; ++row)
    {
        for (int col = 0; col < grid.width; ++col)
        {
            const float x = grid.minX + static_cast<float>(col) * grid.stepX;
            const float y = grid.minY + static_cast<float>(row) * grid.stepY;
            demGrid->worldX.at<float>(row, col) = x;
            demGrid->worldY.at<float>(row, col) = y;
            if (!grid.isValid(col, row))
            {
                continue;
            }
            demGrid->elevation.at<float>(row, col) = grid.at(col, row);
            demGrid->validMask.at<uchar>(row, col) = 255;
            if (grid.hasColors())
            {
                demGrid->color.at<cv::Vec3b>(row, col) =
                    cv::Vec3b(grid.colorAt(col, row, 2),
                              grid.colorAt(col, row, 1),
                              grid.colorAt(col, row, 0));
            }
        }
    }
}

} // namespace

int DemGenerator::estimateGridResolution(const PlaPointCloud &pointCloud,
                                         const DemGenerationOptions &options)
{
    if (pointCloud.size() == 0)
    {
        return 0;
    }

    const PointCloudBounds bounds = computeBounds(pointCloud);
    const double spanX = std::max(1e-6, static_cast<double>(bounds.maxCorner.x - bounds.minCorner.x));

    if (options.gridResolution > 0.0)
    {
        const int widthFromCellSize = static_cast<int>(std::ceil(spanX / options.gridResolution)) + 1;
        return std::max(2, std::min(widthFromCellSize, options.maxGridSize));
    }

    const int autoWidth = static_cast<int>(std::sqrt(static_cast<double>(pointCloud.size())) * 1.8);
    return qBound(options.minGridSize, autoWidth, options.maxGridSize);
}

bool DemGenerator::generateFromPointCloud(const PlaPointCloud &pointCloud,
                                          const DemGenerationOptions &options,
                                          DemGridData *demGrid,
                                          PlaPointCloud *denseCloud,
                                          QString *errorMsg)
{
    if (!demGrid)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 输出对象为空");
        }
        return false;
    }

    if (pointCloud.size() == 0)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("输入点云为空");
        }
        return false;
    }

    const PointCloudBounds bounds = computeBounds(pointCloud);
    if (!bounds.valid)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法计算点云边界");
        }
        return false;
    }

    // Use percentile-based robust bounds to exclude outliers that inflate the grid
    double robustMinX = static_cast<double>(bounds.minCorner.x);
    double robustMaxX = static_cast<double>(bounds.maxCorner.x);
    double robustMinY = static_cast<double>(bounds.minCorner.y);
    double robustMaxY = static_cast<double>(bounds.maxCorner.y);
    {
        const size_t n = pointCloud.size();
        if (n > 100)
        {
            std::vector<float> xs;
            std::vector<float> ys;
            xs.reserve(n);
            ys.reserve(n);
            for (size_t i = 0; i < n; ++i)
            {
                auto pt = pointCloud[i];
                if (!std::isfinite(pt.x()) || !std::isfinite(pt.y()) || !std::isfinite(pt.z()))
                {
                    continue;
                }

                xs.push_back(pt.x());
                ys.push_back(pt.y());
            }
            if (xs.size() > 100)
            {
                std::sort(xs.begin(), xs.end());
                std::sort(ys.begin(), ys.end());
                const size_t lo = xs.size() / 200;        // 0.5th percentile
                const size_t hi = xs.size() - 1 - lo;     // 99.5th percentile
                const float padX = (xs[hi] - xs[lo]) * 0.02f;
                const float padY = (ys[hi] - ys[lo]) * 0.02f;
                robustMinX = static_cast<double>(xs[lo] - padX);
                robustMaxX = static_cast<double>(xs[hi] + padX);
                robustMinY = static_cast<double>(ys[lo] - padY);
                robustMaxY = static_cast<double>(ys[hi] + padY);
            }
        }
    }

    const double spanX = std::max(1e-6, robustMaxX - robustMinX);
    const double spanY = std::max(1e-6, robustMaxY - robustMinY);

    int width = 0;
    if (options.gridResolution > 0.0)
    {
        width = std::max(2,
                         std::min(static_cast<int>(std::ceil(spanX / options.gridResolution)) + 1,
                                  options.maxGridSize));
    }
    else
    {
        const int autoWidth = static_cast<int>(std::sqrt(static_cast<double>(pointCloud.size())) * 1.8);
        width = qBound(options.minGridSize, autoWidth, options.maxGridSize);
    }

    int height = 0;
    if (options.gridResolution > 0.0)
    {
        height = std::max(2,
                          std::min(static_cast<int>(std::ceil(spanY / options.gridResolution)) + 1,
                                   options.maxGridSize));
    }
    else
    {
        height = qBound(options.minGridSize,
                        static_cast<int>(std::round(static_cast<double>(width) * (spanY / spanX))),
                        options.maxGridSize);
    }

    if (width <= 1 || height <= 1)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 栅格尺寸无效");
        }
        return false;
    }

    demGrid->width = width;
    demGrid->height = height;
    demGrid->minX = robustMinX;
    demGrid->minY = robustMinY;
    demGrid->stepX = options.gridResolution > 0.0 ? options.gridResolution : spanX / std::max(1, width - 1);
    demGrid->stepY = options.gridResolution > 0.0 ? options.gridResolution : spanY / std::max(1, height - 1);

    if (demGrid->stepX <= 0.0 || demGrid->stepY <= 0.0)
    {
        if (errorMsg)
            *errorMsg = QStringLiteral("DEM 步长计算异常（stepX=%1 stepY=%2 须 > 0）")
                            .arg(demGrid->stepX).arg(demGrid->stepY);
        return false;
    }

    plapoint::mesh::HeightGridOptions<float> heightGridOptions;
    heightGridOptions.width = width;
    heightGridOptions.height = height;
    heightGridOptions.useExplicitBounds = true;
    heightGridOptions.minX = static_cast<float>(demGrid->minX);
    heightGridOptions.maxX = static_cast<float>(demGrid->minX + demGrid->stepX * static_cast<double>(width - 1));
    heightGridOptions.minY = static_cast<float>(demGrid->minY);
    heightGridOptions.maxY = static_cast<float>(demGrid->minY + demGrid->stepY * static_cast<double>(height - 1));
    heightGridOptions.useBilinearSplat = options.useSubPixelBilinearSplat;
    heightGridOptions.elevationAggregation = toPlaAggregation(options.elevationAggregation);
    heightGridOptions.skipNonFinite = true;
    heightGridOptions.holeFillSearchRadius = std::max(1, options.holeFillSearchRadius);
    heightGridOptions.holeFillMinNeighbors = std::max(1, options.holeFillMinNeighbors);

    try
    {
        auto heightGrid = plapoint::mesh::buildHeightGrid(pointCloud, heightGridOptions);
        plapoint::mesh::fillHoles(heightGrid,
                                  options.holeFillIterations,
                                  heightGridOptions.holeFillMinNeighbors,
                                  heightGridOptions.holeFillSearchRadius);

        assignHeightGridToDemGrid(heightGrid, options.projection, demGrid);
        if (denseCloud)
        {
            *denseCloud = plapoint::mesh::heightGridToPointCloud(heightGrid);
        }
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = QStringLiteral("DEM 栅格化失败: %1")
                                      .arg(QString::fromStdString(e.what()));
        return false;
    }

    return demGrid->isValid();
}

} // namespace xjw
