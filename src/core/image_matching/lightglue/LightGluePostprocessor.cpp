#include "LightGluePostprocessor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xjw::image_matching
{
namespace
{

float logSigmoid(float value)
{
    if (value >= 0.0f)
    {
        return -std::log1p(std::exp(-value));
    }
    return value - std::log1p(std::exp(value));
}

} // namespace

MatchResult postprocessLightGlueCoreOutputs(const float *similarity,
                                            std::int64_t similarityRows,
                                            std::int64_t similarityColumns,
                                            const float *matchability0,
                                            const float *matchability1,
                                            int keypointCount0,
                                            int keypointCount1,
                                            float scoreThreshold,
                                            const char *sourceAlgorithm)
{
    if (keypointCount0 < 0 || keypointCount1 < 0)
    {
        throw std::invalid_argument("LightGlue keypoint counts must be non-negative");
    }

    MatchResult result;
    result.sourceAlgorithm = sourceAlgorithm ? sourceAlgorithm : "lightglue";
    result.matches0.assign(static_cast<std::size_t>(keypointCount0), -1);
    result.matches1.assign(static_cast<std::size_t>(keypointCount1), -1);
    result.matchingScores0.assign(static_cast<std::size_t>(keypointCount0), 0.0f);
    result.matchingScores1.assign(static_cast<std::size_t>(keypointCount1), 0.0f);
    if (keypointCount0 == 0 || keypointCount1 == 0)
    {
        return result;
    }
    if (!similarity || !matchability0 || !matchability1)
    {
        throw std::invalid_argument("LightGlue core output data is null");
    }
    if (similarityRows < keypointCount0 || similarityColumns < keypointCount1)
    {
        throw std::runtime_error(
            "LightGlue core output shape is smaller than the input keypoint counts");
    }

    const std::size_t rows = static_cast<std::size_t>(keypointCount0);
    const std::size_t columns = static_cast<std::size_t>(keypointCount1);
    const std::size_t stride = static_cast<std::size_t>(similarityColumns);
    std::vector<float> rowLogSumExp(rows);
    std::vector<float> columnLogSumExp(columns);

    for (std::size_t row = 0; row < rows; ++row)
    {
        const float *values = similarity + row * stride;
        const float maximum = *std::max_element(values, values + columns);
        double sum = 0.0;
        for (std::size_t column = 0; column < columns; ++column)
        {
            sum += std::exp(static_cast<double>(values[column] - maximum));
        }
        rowLogSumExp[row] = maximum + static_cast<float>(std::log(sum));
    }
    for (std::size_t column = 0; column < columns; ++column)
    {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t row = 0; row < rows; ++row)
        {
            maximum = std::max(maximum, similarity[row * stride + column]);
        }
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row)
        {
            sum += std::exp(static_cast<double>(
                similarity[row * stride + column] - maximum));
        }
        columnLogSumExp[column] = maximum + static_cast<float>(std::log(sum));
    }

    std::vector<int> bestColumn(rows, -1);
    std::vector<int> bestRow(columns, -1);
    std::vector<float> bestRowScore(rows, -std::numeric_limits<float>::infinity());
    std::vector<float> bestColumnScore(
        columns, -std::numeric_limits<float>::infinity());
    for (std::size_t row = 0; row < rows; ++row)
    {
        const float rowConstant = logSigmoid(matchability0[row]) - rowLogSumExp[row];
        for (std::size_t column = 0; column < columns; ++column)
        {
            const float score = 2.0f * similarity[row * stride + column]
                + rowConstant
                + logSigmoid(matchability1[column])
                - columnLogSumExp[column];
            if (score > bestRowScore[row])
            {
                bestRowScore[row] = score;
                bestColumn[row] = static_cast<int>(column);
            }
            if (score > bestColumnScore[column])
            {
                bestColumnScore[column] = score;
                bestRow[column] = static_cast<int>(row);
            }
        }
    }

    result.cvMatches.reserve(std::min(rows, columns));
    for (std::size_t row = 0; row < rows; ++row)
    {
        const int column = bestColumn[row];
        if (column < 0 || bestRow[static_cast<std::size_t>(column)] != static_cast<int>(row))
        {
            continue;
        }
        const float confidence = std::exp(bestRowScore[row]);
        if (!std::isfinite(confidence) || confidence <= scoreThreshold)
        {
            continue;
        }

        const std::size_t columnIndex = static_cast<std::size_t>(column);
        result.matches0[row] = column;
        result.matches1[columnIndex] = static_cast<int>(row);
        result.matchingScores0[row] = confidence;
        result.matchingScores1[columnIndex] = confidence;
        result.cvMatches.emplace_back(static_cast<int>(row), column, 1.0f - confidence);
    }
    result.numMatches = static_cast<int>(result.cvMatches.size());
    return result;
}

} // namespace xjw::image_matching
