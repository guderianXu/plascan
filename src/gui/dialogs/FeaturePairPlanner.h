#pragma once

#include "SfmPairPlanner.h"

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

struct FeaturePairPlan
{
    QStringList pairs;
    QStringList corePairKeys;
    SfmPairPlan corePlan;
};

FeaturePairPlan planFeatureMatchPairPlan(const QStringList &imageBaseNames,
                                         const FeaturePairPlannerOptions &options = {});

FeaturePairPlan planFeatureMatchPairPathPlan(const QStringList &imagePaths,
                                             const FeaturePairPlannerOptions &options = {});

QStringList planFeatureMatchPairs(const QStringList &imageBaseNames,
                                  const FeaturePairPlannerOptions &options = {});

QStringList planFeatureMatchPairPaths(const QStringList &imagePaths,
                                      const FeaturePairPlannerOptions &options = {});

} // namespace xjw::gui
