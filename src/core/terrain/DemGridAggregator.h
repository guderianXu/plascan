#pragma once

#include "DemDomTypes.h"

#include <QString>

#include <vector>

namespace xjw
{

struct DemGridSample
{
    int row = -1;
    int col = -1;
    float elevation = 0.0f;
    float confidence = 1.0f;
    float triangulationError = 0.0f;
};

class DemGridAggregator
{
public:
    static bool aggregateSamples(int width,
                                 int height,
                                 const std::vector<DemGridSample> &samples,
                                 DemGenerationOptions::ElevationAggregation aggregation,
                                 DemGridData *grid,
                                 QString *errorMsg = nullptr);
};

} // namespace xjw
