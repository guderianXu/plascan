#include "OverlapAnalyzer.h"

// ============================================================
// 文件：OverlapAnalyzer.cpp
// 功能：实现多影像地面重叠分析算法。
//
// 算法流程：
//   1) 逐影像反投影中心点到地面，得到地面中心坐标集合
//   2) 逐影像估算地面覆盖等效半径（footprint radius）
//   3) 构建 KD 树（基于地面中心点），加速邻域查询
//   4) 对每张影像，在 searchRadius = neighborFactor * r_i * 2.5 的范围内
//      用 KD 树检索邻近影像；对每对邻近影像（j > i）：
//        - 计算地面中心水平距离 d
//        - threshold = neighborFactor * (r_i + r_j)
//        - 若 d <= threshold，则认为有重叠，得分 = max(0, 1 - d/threshold)
//   5) 所有重叠对按得分降序排列输出
//
// 重叠得分模型（线性衰减）：
//   score = max(0, 1 - d / threshold)
//   score = 1.0 表示两影像中心完全重合（最大重叠）
//   score = 0.0 表示距离恰好等于阈值（临界无重叠）
// ============================================================

#include "KDTree2D.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace
{

// 计算两个三维点在 XY 平面（水平面）的投影距离（忽略高程 Z 分量）
// 用于比较影像地面中心点之间的水平间距
double dist2D(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace

namespace xjw
{

// ============================================================
// 函数：OverlapAnalyzer::analyze
// ============================================================
bool OverlapAnalyzer::analyze(const std::vector<OverlapImageInput> &images,
                              const DemSurface *dem,
                              bool useFixedZ,
                              double fixedZ,
                              double neighborFactor,
                              OverlapAnalysisResult *result,
                              std::string *errorMsg)
{
    if (!result)
    {
        return false;
    }

    // 清除上次运行结果
    result->centers.clear();
    result->footprintRadii.clear();
    result->pairs.clear();

    // 至少 2 张影像才有意义进行重叠分析
    if (images.size() < 2)
    {
        if (errorMsg)
        {
            *errorMsg = "至少需要两张影像";
        }
        return false;
    }

    // 确保邻域系数为正（避免搜索范围为 0 或负数）
    const double kNeighbor = std::max(0.1, neighborFactor);

    // 预分配输出数组
    result->centers.resize(images.size());
    result->footprintRadii.resize(images.size(), 0.0);

    // 用于构建 KD 树的 2D 地面中心点集
    std::vector<common::spatial::KDPoint2D> centerPts;
    centerPts.reserve(images.size());

    // ---- Step 1 & 2：逐影像反投影中心点并估算地面覆盖半径 ----
    for (size_t i = 0; i < images.size(); ++i)
    {
        std::string err;

        // 将影像中心像素反投影到地面，得到地面中心坐标
        if (!GroundBackProjector::imageCenterToGround(images[i].camera,
                                                      images[i].width,
                                                      images[i].height,
                                                      dem,
                                                      useFixedZ,
                                                      fixedZ,
                                                      &result->centers[i],
                                                      &err))
        {
            if (errorMsg)
            {
                *errorMsg = "中心点反投影失败: " + images[i].imagePath + " | " + err;
            }
            return false;
        }

        // 估算影像地面覆盖等效半径（四角到中心平均距离）
        double radius = 0.0;
        if (!GroundBackProjector::estimateFootprintRadius(images[i].camera,
                                                          images[i].width,
                                                          images[i].height,
                                                          dem,
                                                          useFixedZ,
                                                          fixedZ,
                                                          &radius,
                                                          &err))
        {
            radius = 1.0; // 估算失败时使用默认值 1.0（兜底，避免除零）
        }
        // 确保半径至少为 1e-3，防止后续除零
        result->footprintRadii[i] = std::max(1e-3, radius);

        // 将地面中心点加入 KD 树点集（附影像索引，用于回溯）
        centerPts.push_back(common::spatial::KDPoint2D{result->centers[i][0], result->centers[i][1], static_cast<int>(i)});
    }

    // ---- Step 3：构建 KD 树，用于快速邻域搜索 ----
    common::spatial::KDTree2D tree(centerPts);

    // ---- Step 4：遍历所有影像，寻找重叠对 ----
    for (size_t i = 0; i < images.size(); ++i)
    {
        // 以当前影像地面半径的 neighborFactor * 2.5 倍作为 KD 树搜索半径
        // 乘以 2.5 是为了保守地覆盖两影像半径之和的最大可能范围
        const double searchRadius = kNeighbor * result->footprintRadii[i] * 2.5;

        // KD 树半径搜索：返回搜索范围内所有影像的索引
        std::vector<int> nearby = tree.radiusSearch(result->centers[i][0], result->centers[i][1], searchRadius);

        for (int j : nearby)
        {
            // 只处理 j > i 的对，避免重复统计 (i,j) 和 (j,i)
            if (j <= static_cast<int>(i)) continue;

            // 计算两影像地面中心的水平距离
            const double distance = dist2D(result->centers[i], result->centers[static_cast<size_t>(j)]);

            // 计算重叠判断阈值：
            //   threshold = neighborFactor * (r_i + r_j)
            //   当 d ≤ threshold 时认为两影像有重叠覆盖
            const double threshold = std::max(1e-6, kNeighbor * (result->footprintRadii[i] + result->footprintRadii[static_cast<size_t>(j)]));
            if (distance > threshold) continue; // 距离超出阈值，无重叠

            // 线性重叠得分：中心重合时为 1.0，距离达到阈值时为 0.0
            const double score = std::max(0.0, 1.0 - distance / threshold);

            result->pairs.push_back(OverlapPairResult{
                static_cast<int>(i), // indexA（较小索引）
                j,                   // indexB（较大索引）
                distance,
                score});
        }
    }

    // ---- Step 5：按重叠得分降序排列（高重叠优先输出）----
    std::sort(
        result->pairs.begin(),
        result->pairs.end(),
        [](const OverlapPairResult &a, const OverlapPairResult &b)
        {
            return a.overlapScore > b.overlapScore;
        });

    // 生成分析摘要字符串
    std::ostringstream oss;
    oss << "输入影像: " << images.size()
        << ", 重叠对: " << result->pairs.size()
        << ", 邻域系数: " << kNeighbor;
    result->detail = oss.str();
    return true;
}

} // namespace xjw
