#include "DemGridAggregator.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace xjw
{

namespace
{

struct CellAccumulator
{
    std::vector<float> elevations;
    float confidenceSum = 0.0f;
    float errorSum = 0.0f;
};

float medianOfSorted(const std::vector<float> &sorted)
{
    if (sorted.empty())
    {
        return 0.0f;
    }
    const size_t mid = sorted.size() / 2;
    if (sorted.size() % 2 == 1)
    {
        return sorted[mid];
    }
    return (sorted[mid - 1] + sorted[mid]) * 0.5f;
}

float medianValue(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return medianOfSorted(values);
}

float percentile80Value(std::vector<float> values)
{
    if (values.empty())
    {
        return 0.0f;
    }
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(0.8 * static_cast<double>(values.size() + 1));
    const size_t index = static_cast<size_t>(std::clamp<int>(
        static_cast<int>(rank) - 1,
        0,
        static_cast<int>(values.size()) - 1));
    return values[index];
}

float nmadValue(std::vector<float> values)
{
    if (values.empty())
    {
        return 0.0f;
    }
    const float median = medianValue(values);
    std::vector<float> deviations;
    deviations.reserve(values.size());
    for (float value : values)
    {
        deviations.push_back(std::abs(value - median));
    }
    return medianValue(std::move(deviations)) * 1.4826f;
}

float stddevValue(const std::vector<float> &values)
{
    if (values.empty())
    {
        return 0.0f;
    }
    const float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    const float mean = sum / static_cast<float>(values.size());
    float variance = 0.0f;
    for (float value : values)
    {
        const float d = value - mean;
        variance += d * d;
    }
    variance /= static_cast<float>(values.size());
    return std::sqrt(variance);
}

float weightedAverageValue(const CellAccumulator &cell,
                           const std::vector<DemGridSample> &cellSamples)
{
    float weightedSum = 0.0f;
    float weightSum = 0.0f;
    for (const DemGridSample &sample : cellSamples)
    {
        float weight = sample.confidence > 0.0f ? sample.confidence : 1.0f;
        if (sample.triangulationError > 0.0f)
        {
            weight /= std::max(sample.triangulationError, 1e-6f);
        }
        weightedSum += sample.elevation * weight;
        weightSum += weight;
    }
    if (weightSum <= 0.0f)
    {
        return medianValue(cell.elevations);
    }
    return weightedSum / weightSum;
}

float aggregateElevation(const CellAccumulator &cell,
                         const std::vector<DemGridSample> &cellSamples,
                         DemGenerationOptions::ElevationAggregation aggregation)
{
    switch (aggregation)
    {
    case DemGenerationOptions::ElevationAggregation::Min:
        return *std::min_element(cell.elevations.begin(), cell.elevations.end());
    case DemGenerationOptions::ElevationAggregation::Max:
        return *std::max_element(cell.elevations.begin(), cell.elevations.end());
    case DemGenerationOptions::ElevationAggregation::WeightedAverage:
        return weightedAverageValue(cell, cellSamples);
    case DemGenerationOptions::ElevationAggregation::Median:
        return medianValue(cell.elevations);
    case DemGenerationOptions::ElevationAggregation::StdDev:
        return stddevValue(cell.elevations);
    case DemGenerationOptions::ElevationAggregation::Count:
        return static_cast<float>(cell.elevations.size());
    case DemGenerationOptions::ElevationAggregation::Nmad:
        return nmadValue(cell.elevations);
    case DemGenerationOptions::ElevationAggregation::Percentile80:
        return percentile80Value(cell.elevations);
    case DemGenerationOptions::ElevationAggregation::Mean:
    default:
    {
        const float sum = std::accumulate(cell.elevations.begin(), cell.elevations.end(), 0.0f);
        return sum / static_cast<float>(cell.elevations.size());
    }
    }
}

} // namespace

bool DemGridAggregator::aggregateSamples(int width,
                                         int height,
                                         const std::vector<DemGridSample> &samples,
                                         DemGenerationOptions::ElevationAggregation aggregation,
                                         DemGridData *grid,
                                         QString *errorMsg)
{
    if (!grid)
    {
        if (errorMsg) *errorMsg = QStringLiteral("DEM 聚合输出对象为空");
        return false;
    }
    if (width <= 0 || height <= 0)
    {
        if (errorMsg) *errorMsg = QStringLiteral("DEM 聚合网格尺寸无效");
        return false;
    }

    std::vector<CellAccumulator> cells(static_cast<size_t>(width * height));
    std::vector<std::vector<DemGridSample>> samplesByCell(static_cast<size_t>(width * height));

    for (const DemGridSample &sample : samples)
    {
        if (sample.row < 0 || sample.row >= height || sample.col < 0 || sample.col >= width)
        {
            continue;
        }
        if (!std::isfinite(sample.elevation))
        {
            continue;
        }

        const size_t index = static_cast<size_t>(sample.row * width + sample.col);
        cells[index].elevations.push_back(sample.elevation);
        cells[index].confidenceSum += sample.confidence;
        cells[index].errorSum += sample.triangulationError;
        samplesByCell[index].push_back(sample);
    }

    grid->width = width;
    grid->height = height;
    grid->elevation = cv::Mat::zeros(height, width, CV_32FC1);
    grid->validMask = cv::Mat::zeros(height, width, CV_8UC1);
    grid->coverageMask = cv::Mat::zeros(height, width, CV_8UC1);
    grid->pointCount = cv::Mat::zeros(height, width, CV_32SC1);
    grid->confidence = cv::Mat::zeros(height, width, CV_32FC1);
    grid->triangulationError = cv::Mat::zeros(height, width, CV_32FC1);

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            const size_t index = static_cast<size_t>(row * width + col);
            const CellAccumulator &cell = cells[index];
            if (cell.elevations.empty())
            {
                continue;
            }

            const int count = static_cast<int>(cell.elevations.size());
            grid->elevation.at<float>(row, col) =
                aggregateElevation(cell, samplesByCell[index], aggregation);
            grid->validMask.at<uchar>(row, col) = 255;
            grid->coverageMask.at<uchar>(row, col) = 255;
            grid->pointCount.at<int>(row, col) = count;
            grid->confidence.at<float>(row, col) = cell.confidenceSum / static_cast<float>(count);
            grid->triangulationError.at<float>(row, col) = cell.errorSum / static_cast<float>(count);
        }
    }

    return true;
}

} // namespace xjw
