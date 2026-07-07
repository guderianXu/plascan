#pragma once

#include "MatchPhotosAlgorithmPlan.h"
#include "PairTypes.h"

#include <QJsonObject>
#include <QString>

#include <array>
#include <vector>

namespace xjw
{
namespace matchphotos
{

// 阶段报告面向 GUI 进度、日志和人工审查。
// 即使阶段内部实现变化，这里的 stageId 和语义也应尽量保持稳定。
enum class MatchPhotosStageStatus
{
    Pending,
    Completed,
    Skipped,
    Failed
};

struct MatchPhotosStageReport
{
    QString stageId;
    QString displayName;
    MatchPhotosStageStatus status = MatchPhotosStageStatus::Pending;
    QString message;

    // 通用计数字段。含义由 stageId 决定，可以表示候选影像对、特征数量、
    // 匹配数量、轨迹数量或几何验证通过的影像对数量。
    int itemCount = 0;
};

struct MatchPhotosFeatureRecord
{
    QString imagePath;
    QString featurePath;
    int keypointCount = 0;
    QJsonObject settings;
};

struct MatchPhotosMatchRecord
{
    QString image0Path;
    QString image1Path;
    QString matchPath;
    QString sidecarPath;
    int matchCount = 0;
    int geometricInlierCount = 0;
    bool passedGeometry = false;
    std::vector<std::array<int, 2>> inlierIndexPairs;
    QJsonObject settings;
};

struct MatchPhotosResult
{
    bool success = false;
    QString errorMessage;
    MatchPhotosAlgorithmPlan algorithmPlan;
    PairSelectionResult pairSelection;
    std::vector<MatchPhotosStageReport> stages;
    std::vector<MatchPhotosFeatureRecord> features;
    std::vector<MatchPhotosMatchRecord> matches;
    int trackCount = 0;
    int acceptedTrackComponents = 0;
    int rejectedTrackConflictComponents = 0;
    QJsonObject trackSummary;
    QString tiePointPath;
};

} // namespace matchphotos
} // namespace xjw
