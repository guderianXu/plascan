#pragma once

/**
 * @file IncrementalSfmDetail.h
 * @brief IncrementalSfm 多个执行器共享的纯几何和策略辅助接口。
 *
 * 这里的函数不负责日志、文件 IO 或 GUI；它们用于初始对多模型评分、已知位姿
 * 三角化阈值自适应、PnP 点资格判断以及相机中心相似变换估计。
 */

#include "IncrementalSfm.h"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace xjw::incremental_sfm_detail
{

/// 已知位姿路径根据当前交会几何选择出的三角化策略和诊断。
struct KnownPoseTriangulationPolicy
{
    TriangulatorOptions triangulatorOptions; ///< 创建点/补轨迹使用阈值。
    double filterMinTriAngle = 2.0; ///< 最终点过滤阈值。
    bool adapted = false; ///< 是否因当前数据交会角分布放宽默认值。
    int validCandidates = 0; ///< 可评估正深度候选数。
    int acceptedWithDefault = 0; ///< 默认阈值可接受候选数。
    int acceptedWithAdapted = 0; ///< 自适应阈值可接受候选数。
    double chosenMinTriAngle = 2.0; ///< 最终采用角度，度。
};

inline constexpr int kKnownPoseMinLongInputTracksForQualityGate = 10;
inline constexpr int kKnownPoseMinPointsForTrackRatioGate = 20;
inline constexpr double kKnownPoseMaxTwoViewTrackRatio = 0.95;

/// 三维相似变换 `target = scale * rotation * source + translation`。
struct SimilarityTransform3d
{
    bool valid = false; ///< 通过最少样本、有限值和鲁棒内点门控。
    double scale = 1.0; ///< 必须为正。
    std::array<double, 9> rotation{{1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0}};
    std::array<double, 3> translation{{0.0, 0.0, 0.0}};
    int inlierCount = 0; ///< 鲁棒估计内点数。
    double rmse = 0.0; ///< 内点相机中心对齐 RMS。
};

/// 小规模候选集是否值得逐一试算初始模型，而不是只采用排序第一名。
bool shouldEvaluateMultipleInitialPairModels(const IncrementalSfmOptions &options,
                                             int totalImages,
                                             std::size_t candidateCount);

/// 对初始模型按注册覆盖、点数和重投影质量形成可比较高分优先分数。
double scoreInitialPairTrial(const IncrementalSfmResult &result, int totalImages);

/// 随注册规模增长提高 PnP 所用三维点的最小轨迹长度，抑制弱双视点。
int effectivePnpMinTrackLength(const IncrementalSfmOptions &options,
                               std::size_t registeredImageCount);

/// 检查点存在且轨迹长度满足 PnP 门控；坐标/误差由 PnP 对应构建阶段继续验证。
bool pointUsableForPnp(const SfmReconstruction &reconstruction,
                       Point3DId pointId,
                       int minTrackLength);

/// 两个三维点的欧氏距离。
double distance3d(const std::array<double, 3> &a, const std::array<double, 3> &b);

/// 对值副本排序后取最近样本分位数；空输入返回 0。
double percentile(std::vector<double> values, double ratio);

/**
 * @brief 对 camera-to-world 旋转执行最短弧四元数插值/有限外推。
 * @param ratio 0 对应 rotationA，1 对应 rotationB，内部限制在 [-2, 2]。
 */
std::array<double, 9> interpolateCameraRotation(const std::array<double, 9> &rotationA,
                                                const std::array<double, 9> &rotationB,
                                                double ratio);
/// 依据已知相机的交会角分布选择三角化和最终过滤阈值。
KnownPoseTriangulationPolicy resolveKnownPoseTriangulationPolicy(
    const std::shared_ptr<SfmReconstruction> &reconstruction,
    const CorrespondenceGraph &correspondenceGraph,
    const std::vector<ImageId> &imageIds,
    const IncrementalSfmOptions &options);
/// 使用当前已知位姿检查一条匹配能否通过双视正深度、角度和误差门控。
bool knownPoseMatchPassesGeometry(const SfmReconstruction &reconstruction,
                                  ImageId imageId,
                                  ImageId otherImageId,
                                  const FeatureMatch &match,
                                  const TriangulatorOptions &options);

/// 应用 `scale * R * point + t`。
std::array<double, 3> transformPoint(const SimilarityTransform3d &transform,
                                     const std::array<double, 3> &point);

/// 计算两个 3x3 行主序旋转矩阵的乘积。
std::array<double, 9> multiplyRotation(const std::array<double, 9> &left,
                                       const std::array<double, 9> &right);

/// 两个三维点的欧氏距离；保留该命名供 Sim(3) 实现使用。
double pointDistance(const std::array<double, 3> &a, const std::array<double, 3> &b);

/// 返回点集任意两点距离的最大值，作为鲁棒阈值的尺度参考。
double centerExtent(const std::vector<std::array<double, 3>> &points);
/**
 * @brief 用带离群点抑制的相机中心对应估计 Sim(3)。
 *
 * 主要用于将无先验增量 SfM 的任意相似坐标系对齐到可信参考相机中心；
 * 不使用影像序号强制圆轨迹，也不会改变观测拓扑。
 */
SimilarityTransform3d estimateRobustCameraCenterSimilarity(
    const std::vector<std::array<double, 3>> &source,
    const std::vector<std::array<double, 3>> &target);

} // namespace xjw::incremental_sfm_detail
