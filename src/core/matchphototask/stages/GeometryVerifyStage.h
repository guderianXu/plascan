#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

#include <QByteArray>

#include <vector>

namespace xjw::image_matching
{
struct PairMatchData;
}

namespace xjw
{
namespace matchphotos
{

struct GeometryQualityMetrics
{
    int rawMatchCount = 0;
    int inlierCount = 0;
    double inlierRatio = 0.0;
    double image0GridCoverage = 0.0;
    double image1GridCoverage = 0.0;
    bool adjacentImages = false;
};

struct GeometryQualityDecision
{
    bool passed = false;
    double score = 0.0;
    double requiredInlierRatio = 1.0;
    double requiredGridCoverage = 1.0;
};

GeometryQualityDecision evaluateGeometryQuality(
    const GeometryQualityMetrics &metrics,
    const MatchPhotosOptions &options);
GeometryQualityMetrics measureGeometryQuality(
    const image_matching::PairMatchData &pair,
    const MatchPhotosOptions &options,
    bool adjacentImages);
bool areSequenceAdjacent(const MatchPhotosContext &context,
                         const MatchPhotosOptions &options,
                         const QString &image0Path,
                         const QString &image1Path);

// 几何结果只在全部影响 USAC 与质量门的参数、OpenCV 实现版本均一致时复用。
// 该指纹与原始特征/匹配指纹分离，使几何调参无需重新运行 LightGlue。
QByteArray geometryVerificationFingerprint(const MatchPhotosOptions &options);

// F/E/H 几何验证和内点过滤阶段边界。
// 它独立于 MatchingStage，便于传统匹配器和学习型匹配器共用几何检查。
class GeometryVerifyStage
{
public:
    MatchPhotosStageReport run(const MatchPhotosContext &context,
                               const MatchPhotosOptions &options,
                               std::vector<MatchPhotosMatchRecord> *matchRecords) const;
};

} // namespace matchphotos
} // namespace xjw
