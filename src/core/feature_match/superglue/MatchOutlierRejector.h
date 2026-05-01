// =============================================================================
// 文件: MatchOutlierRejector.h
// 模块: 匹配外点副除
// 说明:
//   对 SuperGlueMatcher 输出的匹配结果进行几何一致性验证，副除外点匹配。
//   支持三种 RANSAC 算法模式：
//     - FundamentalRansac : 基于基础矩阵（适用于一般相有场景）
//     - HomographyRansac  : 基于单应矩阵（适用于平面场景或纯旋转）
//     - AffineRansac      : 基于仿射变换（适用于功能相模场景）
//   副除后重建匹配结果，保留 MatchResult 接口一致性。
//
//   依赖: OpenCV calib3d (基础矩阵、单应矩阵、仿射变换估计)
// =============================================================================
#pragma once

#include "SuperGlueMatcher.h"

namespace superglue 
{

// 外点剔除方法枚举
// None                  : 不做任何过滤，直接返回原始匹配
// FundamentalRansac     : 使用基础矩阵 RANSAC（需要至少 8 对点）
// FundamentalUsacMagsac : 使用基础矩阵 USAC_MAGSAC（自适应阈值，更强粗差剔除能力）
// HomographyRansac      : 使用单应矩阵 RANSAC（需要至少 4 对点）
// AffineRansac          : 使用仿射变换 RANSAC（需要至少 4 对点）
enum class OutlierMethod 
{
    None,
    FundamentalRansac,
    FundamentalUsacMagsac,
    HomographyRansac,
    AffineRansac
};

// 外点剔除配置结构体
struct OutlierFilterConfig 
{
    // 剔除方法（默认 FundamentalUsacMagsac，自适应阈值更强粗差剔除）
    OutlierMethod method = OutlierMethod::FundamentalUsacMagsac;
    // RANSAC 重投影误差阈值（像素），值越小则内点要求越严格
    // 对于 USAC_MAGSAC，此值作为 sigma 参数；建议 1.0-2.0
    double reprojThreshold = 1.5;
    // RANSAC 置信度（越高则迭代越多，默认 0.9999）
    double confidence = 0.9999;
    // RANSAC 最大迭代次数
    int maxIters = 10000;
    // 最少内点数：如果内点少于此值，返回原始匹配不过滤
    int minInliers = 20;
};

// 匹配外点副除工具类（纯静态方法，无需实例化）
class MatchOutlierRejector
{
public:
    // 对匹配结果进行几何副除
    // 入参:
    //   input  : SuperGlue 输出的原始匹配结果
    //   kpts0  : 图像0 的关键点列表（与 input.cv_matches 的 queryIdx 对应）
    //   kpts1  : 图像1 的关键点列表（与 input.cv_matches 的 trainIdx 对应）
    //   config : 副除配置（方法、阈值、迭代次数等）
    //   inlierCount: [out] 可选，返回内点匹配数量
    // 返回值: 副除外点后的 MatchResult（外点匹配的 matches0/matches1 被置-1）
    static MatchResult filter(const MatchResult &input,
                              const std::vector<cv::KeyPoint> &kpts0,
                              const std::vector<cv::KeyPoint> &kpts1,
                              const OutlierFilterConfig &config,
                              int *inlierCount = nullptr);
};

} // namespace superglue
