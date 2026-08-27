#pragma once

#include "MatchPhotosAlgorithmPlan.h"
#include "ImageMatchTypes.h"
#include "PairTypes.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <cstdint>
#include <memory>
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
    int keypointCount = 0;
    QJsonObject settings;
};

struct MatchPhotosMatchRecord
{
    QString image0Path;
    QString image1Path;
    // 同一像对会对称写入两幅影像各自的 `.pimatch` 分片。GUI 可以从任意一侧
    // 读取当前影像的全部邻接匹配，不再依赖“图 A__图 B”命名规则。
    QString image0MatchFilePath;
    QString image1MatchFilePath;
    QString algorithmId;
    std::uint32_t algorithmVersion = 0;
    int matchCount = 0;
    int geometricInlierCount = 0;
    bool passedGeometry = false;
    // 仅在任务执行期间存在。几何验证、轨迹构建和最终分片提交都消费同一对象，
    // 避免阶段间再次解析磁盘文件；任务返回 GUI 前会释放该大对象。
    std::shared_ptr<image_matching::PairMatchData> pairData;
    QJsonObject settings;
};

/**
 * @brief 一幅影像对应的最终匹配分片记录。
 *
 * MatchPhotosMatchRecord 描述算法运行期间的“像对”，而本结构描述持久化边界上的
 * “影像”。项目元数据、匹配查看器和 SfM 入口只应登记这里的 `.pimatch` 文件，
 * 不能再根据像对文件名推断两端影像，也不能登记临时 SIFT 特征文件。
 */
struct MatchPhotosImageMatchRecord
{
    QString imagePath;
    QString matchFilePath;
    QStringList neighborImagePaths;
    int neighborVariantCount = 0;
    QJsonObject settings;
};

/**
 * @brief 从候选像对到多视轨迹的统一漏斗诊断。
 *
 * 绝对数用于定位损失阶段，比率用于跨项目对比；所有比率均为 [0, 1]，
 * guidedGainRatio 除外，它表示新增内点相对 guided 前内点数的增幅。
 */
struct MatchPhotosFunnelDiagnostics
{
    int allPairCount = 0;
    int selectedPairCount = 0;
    int matchedPairCount = 0;
    std::int64_t rawMatchCount = 0;
    int geometryPassedPairCount = 0;
    std::int64_t geometryInlierCount = 0;
    std::int64_t guidedAddedInlierCount = 0;
    int finalPassedPairCount = 0;
    std::int64_t finalGeometryInlierCount = 0;
    int trackCount = 0;
    double pairSelectionRatio = 0.0;
    double matchingYieldRatio = 0.0;
    double geometryPairRetentionRatio = 0.0;
    double geometryInlierRatio = 0.0;
    double guidedGainRatio = 0.0;

    QJsonObject toJson() const;
};

MatchPhotosFunnelDiagnostics summarizeMatchPhotosFunnel(
    const PairSelectionResult &pairSelection,
    const std::vector<MatchPhotosMatchRecord> &finalMatches,
    int matchedPairCount,
    std::int64_t rawMatchCount,
    int geometryPassedPairCount,
    std::int64_t geometryInlierCount,
    std::int64_t guidedAddedInlierCount,
    int trackCount);

struct MatchPhotosResult
{
    bool success = false;
    QString errorMessage;
    MatchPhotosAlgorithmPlan algorithmPlan;
    PairSelectionResult pairSelection;
    std::vector<MatchPhotosStageReport> stages;
    std::vector<MatchPhotosFeatureRecord> features;
    std::vector<MatchPhotosMatchRecord> matches;
    std::vector<MatchPhotosImageMatchRecord> imageMatchFiles;
    int trackCount = 0;
    int acceptedTrackComponents = 0;
    int rejectedTrackConflictComponents = 0;
    MatchPhotosFunnelDiagnostics matchingFunnel;
    QJsonObject trackSummary;
    QString tiePointPath;
};

} // namespace matchphotos
} // namespace xjw
