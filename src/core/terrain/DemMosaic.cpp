#include "DemMosaic.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace xjw
{

namespace
{

bool gridCompatible(const DemGridData &a, const DemGridData &b)
{
    constexpr double kEps = 1e-9;
    return a.width == b.width
        && a.height == b.height
        && std::abs(a.minX - b.minX) <= kEps
        && std::abs(a.minY - b.minY) <= kEps
        && std::abs(a.stepX - b.stepX) <= kEps
        && std::abs(a.stepY - b.stepY) <= kEps;
}

bool isValidCell(const DemGridData &grid, int row, int col)
{
    if (grid.elevation.empty() || row < 0 || col < 0
        || row >= grid.elevation.rows || col >= grid.elevation.cols)
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

float confidenceAt(const DemGridData &grid, int row, int col)
{
    if (grid.hasConfidence())
    {
        return std::max(0.0f, grid.confidence.at<float>(row, col));
    }
    return 1.0f;
}

float errorAt(const DemGridData &grid, int row, int col)
{
    if (grid.hasTriangulationError())
    {
        return std::max(0.0f, grid.triangulationError.at<float>(row, col));
    }
    return 1.0f;
}

float blendValues(const std::vector<float> &values,
                  const std::vector<float> &weights,
                  DemMosaicBlendMode mode)
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
        return std::accumulate(values.begin(), values.end(), 0.0f)
            / static_cast<float>(values.size());
    }
}

} // namespace

bool DemMosaic::mosaicSameGrid(const std::vector<DemGridData> &tiles,
                               DemMosaicBlendMode blendMode,
                               DemGridData *output,
                               QString *errorMsg)
{
    if (!output)
    {
        if (errorMsg) *errorMsg = QStringLiteral("DEM mosaic 输出对象为空");
        return false;
    }
    if (tiles.empty())
    {
        if (errorMsg) *errorMsg = QStringLiteral("DEM mosaic 输入瓦片为空");
        return false;
    }
    if (!tiles.front().isValid())
    {
        if (errorMsg) *errorMsg = QStringLiteral("DEM mosaic 第一个瓦片无效");
        return false;
    }

    for (size_t i = 1; i < tiles.size(); ++i)
    {
        if (!tiles[i].isValid())
        {
            if (errorMsg) *errorMsg = QStringLiteral("DEM mosaic 瓦片 %1 无效").arg(i);
            return false;
        }
        if (!gridCompatible(tiles.front(), tiles[i]))
        {
            if (errorMsg) *errorMsg = QStringLiteral("DEM mosaic 瓦片 %1 网格不一致").arg(i);
            return false;
        }
    }

    *output = DemGridData();
    output->width = tiles.front().width;
    output->height = tiles.front().height;
    output->minX = tiles.front().minX;
    output->minY = tiles.front().minY;
    output->stepX = tiles.front().stepX;
    output->stepY = tiles.front().stepY;
    output->projection = tiles.front().projection;
    output->elevation = cv::Mat::zeros(output->height, output->width, CV_32FC1);
    output->validMask = cv::Mat::zeros(output->height, output->width, CV_8UC1);
    output->coverageMask = cv::Mat::zeros(output->height, output->width, CV_8UC1);
    output->pointCount = cv::Mat::zeros(output->height, output->width, CV_32SC1);
    output->confidence = cv::Mat::zeros(output->height, output->width, CV_32FC1);
    output->triangulationError = cv::Mat::zeros(output->height, output->width, CV_32FC1);

    for (int row = 0; row < output->height; ++row)
    {
        for (int col = 0; col < output->width; ++col)
        {
            std::vector<float> values;
            std::vector<float> weights;
            float confidenceSum = 0.0f;
            float errorSum = 0.0f;

            for (const DemGridData &tile : tiles)
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

    return true;
}

} // namespace xjw
