#pragma once

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosResult.h"

#include <QByteArray>

#include <vector>

namespace xjw
{
namespace matchphotos
{

// 低支持度的两视几何模型容易被重复结构伪造。强支持像对依据内点数，
// 弱支持像对还必须有足够高的内点率，避免错误边污染多视轨迹。
bool passesGeometryQualityGate(int rawMatchCount,
                               int inlierCount,
                               int minimumInliers);

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
