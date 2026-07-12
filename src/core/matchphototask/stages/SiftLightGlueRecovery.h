#pragma once

#include "MatchGeometryFilter.h"
#include "MatchPhotosAlgorithmPlan.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

#include <QSet>

#include <vector>

namespace xjw::feature_extractors
{
struct FeatureData;
}

namespace xjw::feature_match
{
struct MatchResult;
}

namespace xjw::matchphotos
{

bool shouldAugmentDenseSiftPair(const MatchPhotosOptions &options,
                                const MatchPhotosMatchRecord &record,
                                int rawMatchCount);

bool runFullSiftRecovery(const MatchPhotosOptions &options,
                         const xjw::feature_extractors::FeatureData &feature0,
                         const xjw::feature_extractors::FeatureData &feature1,
                         const xjw::feature_match::OutlierFilterConfig &filterConfig,
                         xjw::feature_match::MatchResult *rawRecovered,
                         xjw::feature_match::MatchResult *filteredRecovered,
                         bool *usedCuda);

bool persistRecoveredMatch(const MatchPhotosOptions &options,
                           const MatchPhotosAlgorithmPlan &plan,
                           const xjw::feature_extractors::FeatureData &feature0,
                           const xjw::feature_extractors::FeatureData &feature1,
                           const xjw::feature_match::MatchResult &rawRecovered,
                           const QString &reason,
                           int primaryRawCount,
                           int primaryInlierCount,
                           int recoveredInlierCount,
                           bool usedCuda,
                           MatchPhotosMatchRecord *record);

std::vector<int> disconnectedRecoveryCandidates(
    const std::vector<MatchPhotosMatchRecord> &records,
    const QSet<int> &attempted);

} // namespace xjw::matchphotos
