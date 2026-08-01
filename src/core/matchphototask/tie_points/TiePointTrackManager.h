#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"
#include "common/SfmTypes.h"

#include <QJsonObject>
#include <QString>

#include <vector>

namespace xjw
{
namespace matchphotos
{

struct TiePointTrackBuildResult
{
    bool success = false;
    QString errorMessage;
    std::vector<Track> tracks;
    int consumedPairCount = 0;
    int skippedPairCount = 0;
    int totalComponents = 0;
    int acceptedComponents = 0;
    int rejectedConflictComponents = 0;
    int rejectedConflictEdges = 0;
    int prunedByQualityThinning = 0;
    int prunedStationaryTracks = 0;
    double meanTrackConfidence = 0.0;
    QJsonObject trackSummary;
    QString tiePointPath;
};

// 管理“匹配照片”阶段最终产出的多视图连接点轨迹。
// 两两匹配、特征提取、sidecar 写入和 GUI 状态更新不属于这个类的职责。
class TiePointTrackManager
{
public:
    TiePointTrackBuildResult build(const MatchPhotosContext &context,
                                   const MatchPhotosOptions &options,
                                   std::vector<MatchPhotosMatchRecord> *matchRecords) const;
};

} // namespace matchphotos
} // namespace xjw
