#pragma once

#include "BundleAdjust.h"

#include <vector>

namespace xjw
{

struct ReferenceTerrainGrid
{
    int width = 0;
    int height = 0;
    double originX = 0.0;
    double originY = 0.0;
    double pixelSizeX = 1.0;
    double pixelSizeY = 1.0;
    double nodata = -9999.0;
    std::vector<double> heights;
};

struct ReferenceTerrainPriorOptions
{
    bool enabled = true;
    double sigmaMeters = 1.0;
    double maxAssociationDistanceMeters = 2.0;
    double huberDeltaMeters = 0.5;
};

struct ReferenceTerrainPriorStats
{
    int inputTrackCount = 0;
    int associatedTrackCount = 0;
    int rejectedNoHeightCount = 0;
    int rejectedByDistanceCount = 0;
    double rmsBeforeMeters = 0.0;
    double medianAbsBeforeMeters = 0.0;
};

class ReferenceTerrainPrior
{
public:
    static double sampleHeight(const ReferenceTerrainGrid &grid,
                               double x,
                               double y,
                               bool *ok = nullptr);

    static ReferenceTerrainPriorStats attachHeightPlaneConstraints(const ReferenceTerrainGrid &grid,
                                                                   std::vector<BATrack> *tracks,
                                                                   const ReferenceTerrainPriorOptions &options);

    static BAOptions makeBundleAdjustOptions(const ReferenceTerrainPriorOptions &options);
};

} // namespace xjw
