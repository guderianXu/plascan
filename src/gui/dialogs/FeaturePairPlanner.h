#pragma once

#include <QStringList>

#include <array>
#include <vector>

namespace xjw::gui
{

struct FeaturePairPlannerOptions
{
    int exhaustiveMaxImages = 80;
    int sequentialWindow = 4;
    int spatialNeighborCount = 0;
    std::vector<std::array<double, 3>> knownCameraCenters;
    std::vector<std::array<int, 2>> knownCameraOverlapPairs;
    double knownCameraOverlapMaxExpansion = 2.0;
};

QStringList planFeatureMatchPairs(const QStringList &imageBaseNames,
                                  const FeaturePairPlannerOptions &options = {});

} // namespace xjw::gui
