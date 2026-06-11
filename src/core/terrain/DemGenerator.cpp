#include "DemGenerator.h"

#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw
{

namespace
{

struct AccumulatorCell
{
    double sumWeightedZ = 0.0;
    double sumWeight = 0.0;
    double sumWeightedB = 0.0;
    double sumWeightedG = 0.0;
    double sumWeightedR = 0.0;
    double sumColorWeight = 0.0;
    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();
    int count = 0;
};

void accumulateCell(AccumulatorCell *cell,
                    double z,
                    double weight,
                    DemGenerationOptions::ElevationAggregation aggregation,
                    const cv::Vec3b *color = nullptr)
{
    if (!cell || weight <= 0.0)
    {
        return;
    }

    cell->sumWeightedZ += z * weight;
    cell->sumWeight += weight;
    cell->minZ = std::min(cell->minZ, z);
    cell->maxZ = std::max(cell->maxZ, z);
    ++cell->count;
    if (color)
    {
        cell->sumWeightedB += static_cast<double>((*color)[0]) * weight;
        cell->sumWeightedG += static_cast<double>((*color)[1]) * weight;
        cell->sumWeightedR += static_cast<double>((*color)[2]) * weight;
        cell->sumColorWeight += weight;
    }

    Q_UNUSED(aggregation);
}

float resolveCellElevation(const AccumulatorCell &cell,
                           DemGenerationOptions::ElevationAggregation aggregation)
{
    if (cell.count <= 0 || cell.sumWeight <= 0.0)
    {
        return 0.0f;
    }

    if (aggregation == DemGenerationOptions::ElevationAggregation::Min)
    {
        return static_cast<float>(cell.minZ);
    }

    if (aggregation == DemGenerationOptions::ElevationAggregation::Max)
    {
        return static_cast<float>(cell.maxZ);
    }

    return static_cast<float>(cell.sumWeightedZ / cell.sumWeight);
}

cv::Vec3b resolveCellColor(const AccumulatorCell &cell)
{
    if (cell.count <= 0 || cell.sumColorWeight <= 0.0)
    {
        return cv::Vec3b(128, 128, 128);
    }

    auto channel = [](double value) {
        return static_cast<uchar>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
    };
    const double invWeight = 1.0 / cell.sumColorWeight;
    return cv::Vec3b(channel(cell.sumWeightedB * invWeight),
                     channel(cell.sumWeightedG * invWeight),
                     channel(cell.sumWeightedR * invWeight));
}

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

PlaPointCloud buildDenseCloud(const DemGridData &demGrid)
{
    std::vector<float> xs, ys, zs;
    std::vector<cv::Vec3b> colors;
    const size_t n = static_cast<std::size_t>(demGrid.width) * static_cast<std::size_t>(demGrid.height);
    xs.reserve(n);
    ys.reserve(n);
    zs.reserve(n);
    if (demGrid.hasColor())
    {
        colors.reserve(n);
    }

    for (int row = 0; row < demGrid.height; ++row)
    {
        for (int col = 0; col < demGrid.width; ++col)
        {
            if (demGrid.validMask.at<uchar>(row, col) == 0)
            {
                continue;
            }

            xs.push_back(static_cast<float>(demGrid.minX + demGrid.stepX * static_cast<double>(col)));
            ys.push_back(static_cast<float>(demGrid.minY + demGrid.stepY * static_cast<double>(row)));
            zs.push_back(demGrid.elevation.at<float>(row, col));
            if (demGrid.hasColor())
            {
                colors.push_back(demGrid.color.at<cv::Vec3b>(row, col));
            }
        }
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(xs.size(), 3);
    for (size_t i = 0; i < xs.size(); ++i)
    {
        pts(static_cast<plamatrix::Index>(i), 0) = xs[i];
        pts(static_cast<plamatrix::Index>(i), 1) = ys[i];
        pts(static_cast<plamatrix::Index>(i), 2) = zs[i];
    }
    PlaPointCloud cloud(std::move(pts));
    if (!colors.empty() && colors.size() == xs.size())
    {
        plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colorMatrix(colors.size(), 3);
        for (size_t i = 0; i < colors.size(); ++i)
        {
            colorMatrix(static_cast<plamatrix::Index>(i), 0) = colors[i][2];
            colorMatrix(static_cast<plamatrix::Index>(i), 1) = colors[i][1];
            colorMatrix(static_cast<plamatrix::Index>(i), 2) = colors[i][0];
        }
        cloud.setColors(std::move(colorMatrix));
    }
    return cloud;
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

    demGrid->projection = options.projection;
    demGrid->elevation = cv::Mat(height, width, CV_32F, cv::Scalar(0));
    demGrid->color.release();
    demGrid->validMask = cv::Mat(height, width, CV_8U, cv::Scalar(0));
    const bool hasSourceColors = pointCloud.hasColors() && pointCloud.colors()
                              && pointCloud.colors()->rows() == static_cast<plamatrix::Index>(pointCloud.size())
                              && pointCloud.colors()->cols() >= 3;
    if (hasSourceColors)
    {
        demGrid->color = cv::Mat(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
    }

    std::vector<AccumulatorCell> accumulators(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    auto cellIndex = [width](int col, int row) {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(col);
    };

    auto splatPointToGrid = [&](double x, double y, double z, const cv::Vec3b *color) {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            return;
        }

        const double gridX = (x - demGrid->minX) / demGrid->stepX;
        const double gridY = (y - demGrid->minY) / demGrid->stepY;
        if (gridX < 0.0 || gridX > static_cast<double>(width - 1) ||
            gridY < 0.0 || gridY > static_cast<double>(height - 1))
        {
            return;
        }

        if (options.useSubPixelBilinearSplat)
        {
            const int baseCol = static_cast<int>(std::floor(gridX));
            const int baseRow = static_cast<int>(std::floor(gridY));
            const double fracX = gridX - static_cast<double>(baseCol);
            const double fracY = gridY - static_cast<double>(baseRow);

            for (int rowOffset = 0; rowOffset <= 1; ++rowOffset)
            {
                for (int colOffset = 0; colOffset <= 1; ++colOffset)
                {
                    const int col = baseCol + colOffset;
                    const int row = baseRow + rowOffset;
                    if (col < 0 || col >= width || row < 0 || row >= height)
                    {
                        continue;
                    }

                    const double weightX = colOffset == 0 ? (1.0 - fracX) : fracX;
                    const double weightY = rowOffset == 0 ? (1.0 - fracY) : fracY;
                    const double weight = std::max(0.0, weightX * weightY);

                    if (weight <= 1e-9)
                    {
                        continue;
                    }

                    AccumulatorCell &accumulator = accumulators[cellIndex(col, row)];
                    accumulateCell(&accumulator, z, weight, options.elevationAggregation, color);
                    demGrid->validMask.at<uchar>(row, col) = 255;
                }
            }
            return;
        }

        const int col = static_cast<int>(std::round(gridX));
        const int row = static_cast<int>(std::round(gridY));
        if (col < 0 || col >= width || row < 0 || row >= height)
        {
            return;
        }

        AccumulatorCell &accumulator = accumulators[cellIndex(col, row)];
        accumulateCell(&accumulator, z, 1.0, options.elevationAggregation, color);
        demGrid->validMask.at<uchar>(row, col) = 255;
    };

    for (size_t i = 0; i < pointCloud.size(); ++i)
    {
        auto pt = pointCloud[i];
        cv::Vec3b sourceColor;
        const cv::Vec3b *sourceColorPtr = nullptr;
        if (hasSourceColors)
        {
            const auto *colors = pointCloud.colors();
            sourceColor = cv::Vec3b(
                colors->getValue(static_cast<plamatrix::Index>(i), 2),
                colors->getValue(static_cast<plamatrix::Index>(i), 1),
                colors->getValue(static_cast<plamatrix::Index>(i), 0));
            sourceColorPtr = &sourceColor;
        }
        splatPointToGrid(static_cast<double>(pt.x()),
                         static_cast<double>(pt.y()),
                         static_cast<double>(pt.z()),
                         sourceColorPtr);
    }

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            const AccumulatorCell &accumulator = accumulators[cellIndex(col, row)];
            if (accumulator.count > 0)
            {
                demGrid->elevation.at<float>(row, col) =
                    resolveCellElevation(accumulator, options.elevationAggregation);
                if (hasSourceColors)
                {
                    demGrid->color.at<cv::Vec3b>(row, col) = resolveCellColor(accumulator);
                }
            }
        }
    }

    const int holeFillRadius = std::max(1, options.holeFillSearchRadius);
    for (int iteration = 0; iteration < options.holeFillIterations; ++iteration)
    {
        cv::Mat elevationCopy = demGrid->elevation.clone();
        cv::Mat colorCopy = demGrid->hasColor() ? demGrid->color.clone() : cv::Mat();
        cv::Mat validCopy = demGrid->validMask.clone();
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                if (demGrid->validMask.at<uchar>(row, col) != 0)
                {
                    continue;
                }

                double weightedNeighborSum = 0.0;
                double weightedB = 0.0;
                double weightedG = 0.0;
                double weightedR = 0.0;
                double weightSum = 0.0;
                int neighborCount = 0;
                const int minRow = std::max(0, row - holeFillRadius);
                const int maxRow = std::min(height - 1, row + holeFillRadius);
                const int minCol = std::max(0, col - holeFillRadius);
                const int maxCol = std::min(width - 1, col + holeFillRadius);

                for (int neighborRow = minRow; neighborRow <= maxRow; ++neighborRow)
                {
                    for (int neighborCol = minCol; neighborCol <= maxCol; ++neighborCol)
                    {
                        if (neighborCol == col && neighborRow == row)
                        {
                            continue;
                        }

                        if (demGrid->validMask.at<uchar>(neighborRow, neighborCol) != 0)
                        {
                            const int dx = neighborCol - col;
                            const int dy = neighborRow - row;
                            const double distance = std::sqrt(static_cast<double>(dx * dx + dy * dy));
                            const double weight = 1.0 / std::max(1e-6, distance);
                            weightedNeighborSum +=
                                static_cast<double>(demGrid->elevation.at<float>(neighborRow, neighborCol)) * weight;
                            if (demGrid->hasColor())
                            {
                                const cv::Vec3b neighborColor = demGrid->color.at<cv::Vec3b>(neighborRow, neighborCol);
                                weightedB += static_cast<double>(neighborColor[0]) * weight;
                                weightedG += static_cast<double>(neighborColor[1]) * weight;
                                weightedR += static_cast<double>(neighborColor[2]) * weight;
                            }
                            weightSum += weight;
                            ++neighborCount;
                        }
                    }
                }

                if (neighborCount >= options.holeFillMinNeighbors && weightSum > 0.0)
                {
                    elevationCopy.at<float>(row, col) = static_cast<float>(weightedNeighborSum / weightSum);
                    if (!colorCopy.empty())
                    {
                        auto channel = [](double value) {
                            return static_cast<uchar>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
                        };
                        const double invWeight = 1.0 / weightSum;
                        colorCopy.at<cv::Vec3b>(row, col) = cv::Vec3b(channel(weightedB * invWeight),
                                                                       channel(weightedG * invWeight),
                                                                       channel(weightedR * invWeight));
                    }
                    validCopy.at<uchar>(row, col) = 255;
                }
            }
        }

        demGrid->elevation = elevationCopy;
        if (!colorCopy.empty())
        {
            demGrid->color = colorCopy;
        }
        demGrid->validMask = validCopy;
    }

    if (denseCloud)
    {
        *denseCloud = buildDenseCloud(*demGrid);
    }

    return demGrid->isValid();
}

} // namespace xjw
