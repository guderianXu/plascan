#include "DemDifference.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace xjw::qc
{

namespace
{

bool isNoData(double value, double nodata)
{
    return !std::isfinite(value) || std::abs(value - nodata) <= 1e-12;
}

double percentileSorted(const std::vector<double> &sortedValues, double q)
{
    if (sortedValues.empty())
    {
        return 0.0;
    }

    const double clamped = std::max(0.0, std::min(1.0, q));
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(clamped * static_cast<double>(sortedValues.size() - 1)));
    return sortedValues[std::min(index, sortedValues.size() - 1)];
}

} // namespace

DemDifferenceResult DemDifference::compareSameGrid(const DemGrid &candidate,
                                                   const DemGrid &reference,
                                                   bool allowProjectionMismatch)
{
    DemDifferenceResult result;
    if (candidate.width <= 0 || candidate.height <= 0
        || reference.width <= 0 || reference.height <= 0)
    {
        result.error = QStringLiteral("DEM 尺寸无效");
        return result;
    }
    if (candidate.width != reference.width || candidate.height != reference.height)
    {
        result.error = QStringLiteral("DEM 网格尺寸不一致");
        return result;
    }
    if (candidate.values.size() != reference.values.size()
        || candidate.values.size() != static_cast<std::size_t>(candidate.width * candidate.height))
    {
        result.error = QStringLiteral("DEM 栅格数据长度与尺寸不一致");
        return result;
    }
    if (!allowProjectionMismatch
        && !candidate.projection.isEmpty()
        && !reference.projection.isEmpty()
        && candidate.projection != reference.projection)
    {
        result.error = QStringLiteral("DEM 投影不一致");
        return result;
    }

    result.differences.assign(candidate.values.size(), reference.nodata);
    std::vector<double> validDiffs;
    validDiffs.reserve(candidate.values.size());

    for (std::size_t i = 0; i < candidate.values.size(); ++i)
    {
        const double candidateValue = candidate.values[i];
        const double referenceValue = reference.values[i];
        if (isNoData(candidateValue, candidate.nodata) || isNoData(referenceValue, reference.nodata))
        {
            continue;
        }

        const double diff = candidateValue - referenceValue;
        result.differences[i] = diff;
        validDiffs.push_back(diff);
    }

    if (validDiffs.empty())
    {
        result.error = QStringLiteral("DEM 没有共同有效像元");
        return result;
    }

    double sumSquared = 0.0;
    for (double diff : validDiffs)
    {
        sumSquared += diff * diff;
    }

    result.validCount = static_cast<int>(validDiffs.size());
    result.mean = std::accumulate(validDiffs.begin(), validDiffs.end(), 0.0)
        / static_cast<double>(validDiffs.size());
    result.rmse = std::sqrt(sumSquared / static_cast<double>(validDiffs.size()));
    std::sort(validDiffs.begin(), validDiffs.end());
    result.median = percentileSorted(validDiffs, 0.5);
    result.p95 = percentileSorted(validDiffs, 0.95);
    result.success = true;
    return result;
}

} // namespace xjw::qc
