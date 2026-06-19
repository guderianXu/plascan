#pragma once

#include "DemDomTypes.h"

#include <QString>

#include <vector>

namespace xjw
{

enum class DemMosaicBlendMode
{
    First,
    Last,
    Mean,
    Median,
    Min,
    Max,
    ConfidenceWeighted,
    InverseErrorWeighted
};

class DemMosaic
{
public:
    static bool mosaicSameGrid(const std::vector<DemGridData> &tiles,
                               DemMosaicBlendMode blendMode,
                               DemGridData *output,
                               QString *errorMsg = nullptr);
};

} // namespace xjw
