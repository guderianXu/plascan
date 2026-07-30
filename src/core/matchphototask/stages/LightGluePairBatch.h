#pragma once

#include "LightGlueMatcher.h"
#include "MatchPhotosAlgorithmPlan.h"
#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosResult.h"
#include "PairTypes.h"

#include <QString>

#include <vector>

namespace xjw
{
namespace matchphotos
{

struct LightGluePairBatchConfig
{
    xjw::feature_match::LightGlueConfig matcherConfig;
    MatchPhotosGpuMemoryInfo gpuMemory;
    int primaryKeypointBudget = 0;
    int requestedWorkers = 0;
    int effectiveWorkers = 1;
    float effectiveMatchThreshold = 0.15f;
    bool applyTiepointMask = false;
};

struct LightGluePairBatchResult
{
    bool cancelled = false;
    bool usedSerialRecovery = false;
    bool usedSerialOomRecovery = false;
    int matchedPairs = 0;
    int failedPairs = 0;
    int totalMatches = 0;
    QString fatalError;
    QString serialRecoveryReason;
    std::vector<MatchPhotosMatchRecord> records;
};

LightGluePairBatchResult runLightGluePairBatch(
    const MatchPhotosContext &context,
    const MatchPhotosOptions &options,
    const MatchPhotosAlgorithmPlan &algorithmPlan,
    const PairSelectionResult &pairSelection,
    const LightGluePairBatchConfig &config);

} // namespace matchphotos
} // namespace xjw
