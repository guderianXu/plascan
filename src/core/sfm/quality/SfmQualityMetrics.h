#pragma once

/**
 * @file SfmQualityMetrics.h
 * @brief 与 GUI/工程格式解耦的稀疏重建质量统计和 MVS 门控。
 *
 * 指标同时衡量注册覆盖率、轨迹冗余、交会角、重投影误差和影像空间覆盖。
 * `acceptableForMvs` 是保守生产门控，不代表点云绝对精度；具体拒绝原因保存在
 * advisories/warnings，供上层生成可读报告。
 */

#include <map>
#include <string>
#include <vector>

namespace xjw
{

/// 一个三维点在单幅影像上的二维观测，用于网格覆盖率统计。
struct SfmQualityObservation
{
    int imageId = -1; ///< 当前质量输入中的影像标识。
    double x = 0.0; ///< 像素 u。
    double y = 0.0; ///< 像素 v。
};

/// 质量计算所需的最小三维点快照。
struct SfmQualityPoint
{
    int trackLength = 0; ///< 有效观测影像数。
    double reprojectionErrorPx = 0.0; ///< 该点多视平均重投影误差。
    double triangulationAngleDeg = 0.0; ///< 保守三角化角，度。
    std::vector<SfmQualityObservation> observations; ///< 用于覆盖率，不参与重算误差。
};

/// 阈值和影像范围；所有比例门控都在 [0,1] 上解释。
struct SfmQualityMetricsOptions
{
    int totalImageCount = 0; ///< 当前输入影像总数。
    int registeredImageCount = 0; ///< 成功恢复位姿数。
    double imageWidth = 0.0; ///< 覆盖率归一化宽度；<=0 时跳过网格指标。
    double imageHeight = 0.0; ///< 覆盖率归一化高度。
    int coverageGridColumns = 4; ///< 每幅影像覆盖网格列数。
    int coverageGridRows = 4; ///< 每幅影像覆盖网格行数。
    int minTrackLength = 3; ///< 低于此值计为弱轨迹。
    double minTriangulationAngleDeg = 2.0; ///< 低于此值计为弱交会点。
    double maxReprojectionErrorPx = 3.0; ///< 高于此值计为高误差点。
    double minRegisteredImageRatioForMvs = 0.80; ///< MVS 最低注册覆盖率；低于 80% 通常会造成大面积深度缺口。
    double warnTwoViewTrackRatioForMvs = 0.70; ///< 双视轨迹比例提示阈值。
    double maxTwoViewTrackRatioForMvs = 0.85; ///< 双视轨迹比例拒绝阈值。
    double maxHighReprojectionErrorRatioForMvs = 0.30; ///< 高误差点比例上限。
    double maxWeakTriangulationAngleRatioForMvs = 0.60; ///< 弱交会点比例上限。
    double minObservationGridCoverageMeanForMvs = 0.0; ///< 平均网格覆盖率下限。
};

/// 一组有限数值的描述统计；百分位使用排序后的线性/离散实现约定。
struct SfmNumericSummary
{
    int count = 0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double p50 = 0.0;
    double p84 = 0.0;
    double p95 = 0.0;
};

/// 稀疏模型质量输出，计数均针对传入 points。
struct SfmQualityMetrics
{
    int totalImageCount = 0;
    int registeredImageCount = 0;
    double registeredImageRatio = 0.0; ///< 注册影像数/输入影像总数，便于报告直接展示覆盖率。
    int pointCount = 0;
    int twoViewTrackCount = 0;
    int multiViewTrackCount = 0;
    int weakTrackCount = 0;
    int weakTriangulationAngleCount = 0;
    int highReprojectionErrorCount = 0;
    int coverageGridColumns = 0;
    int coverageGridRows = 0;
    double observationGridCoverageMean = 0.0;
    bool acceptableForMvs = true; ///< 是否通过生产 MVS 门控。
    std::string qualityStatus = "ok"; ///< 稳定机器状态，如 ok/warning/rejected。
    std::vector<std::string> qualityAdvisories; ///< 非阻塞改进建议。
    std::vector<std::string> qualityWarnings; ///< 导致风险或门控失败的原因。

    SfmNumericSummary trackLength;
    SfmNumericSummary reprojectionError;
    SfmNumericSummary triangulationAngle;
    std::map<int, int> trackLengthHistogram;
};

/**
 * @brief 一次性计算全部统计和 MVS 门控。
 *
 * 调用方必须提供有限的误差和角度；当前实现不会在统计层静默丢弃非有限样本，
 * 以免质量问题被隐藏。函数纯计算、无日志和文件 IO，可安全用于候选模型并行排序。
 */
SfmQualityMetrics computeSfmQualityMetrics(const std::vector<SfmQualityPoint> &points,
                                           const SfmQualityMetricsOptions &options);

} // namespace xjw
