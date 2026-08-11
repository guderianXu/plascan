#pragma once

#include "OverlapAnalyzer.h"

#include <QString>
#include <vector>

namespace xjw::matchphotos
{

struct SparseSceneOverlapOptions
{
    int maxProjectionSamples = 4096;
    int minSharedPointCount = 8;
    int minJointVisibleSamples = 12;
    double minProjectedOverlap = 0.03;
    double imageMarginFraction = 0.03;
    // 仅靠视锥补对时限制观察方向夹角；更宽基线仍可由真实稀疏点共视证据召回。
    double minViewingDirectionCosine = 0.0;
};

struct SparseSceneOverlapStats
{
    int validPointCount = 0;
    int sampledPointCount = 0;
    int covisibilityPairCount = 0;
    int frustumPairCount = 0;
    QString detail;
};

/**
 * @brief 从已有 SfM 稀疏点逐观测 sidecar 计算任意三维场景的候选影像对。
 *
 * 共视轨迹提供强证据；对尚无共同轨迹的影像对，再用鲁棒稀疏场景样本在两台
 * 相机中的联合可见率估计视锥重叠。该分析不假设固定地面、DEM 或行星球面。
 */
class SparseSceneOverlapAnalyzer
{
public:
    static bool analyzeFile(const QString &sidecarPath,
                            const std::vector<OverlapImageInput> &images,
                            const SparseSceneOverlapOptions &options,
                            OverlapAnalysisResult *result,
                            SparseSceneOverlapStats *stats = nullptr,
                            QString *errorMessage = nullptr);
};

} // namespace xjw::matchphotos
