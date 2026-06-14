#pragma once

#include <QStringList>

namespace xjw::gui
{

struct FeaturePairPlannerOptions
{
    int exhaustiveMaxImages = 80;
    int sequentialWindow = 4;
};

QStringList planFeatureMatchPairs(const QStringList &imageBaseNames,
                                  const FeaturePairPlannerOptions &options = {});

} // namespace xjw::gui
