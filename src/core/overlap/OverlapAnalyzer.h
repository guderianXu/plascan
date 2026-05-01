#pragma once

// ============================================================
// 文件：OverlapAnalyzer.h
// 功能：分析多张航空/无人机影像之间的地面重叠关系。
//   流程：
//     1) 将每张影像中心反投影到地面，得到地面中心坐标；
//     2) 估计每张影像的地面覆盖半径（footprint radius）；
//     3) 利用 KD 树快速筛选邻近影像对，计算重叠得分；
//     4) 按重叠得分降序排列输出。
//   重叠得分模型：
//     score = max(0, 1 - d / threshold)
//     其中 d 为两影像地面中心距离，
//     threshold = neighborFactor * (r1 + r2)（r1/r2 为各自地面覆盖半径）。
// ============================================================

#include "GroundBackProjector.h"

#include <string>
#include <vector>

namespace xjw {

// ============================================================
// 结构体：OverlapImageInput
// 描述：重叠分析所需的单张影像输入数据，包含影像路径、相机标定及分辨率。
// ============================================================
struct OverlapImageInput
{
    // 影像文件路径（用于错误信息报告，不做实际读取）
    std::string imagePath;

    // 该影像对应的相机标定参数（内参 + 外参）
    Camera camera;

    // 影像宽度（像素），用于计算中心像素坐标及四角坐标
    int width = 0;

    // 影像高度（像素）
    int height = 0;
};

// ============================================================
// 结构体：OverlapPairResult
// 描述：一对影像之间的重叠分析结果。
// ============================================================
struct OverlapPairResult
{
    // 第一张影像在输入向量中的索引（满足 indexA < indexB）
    int indexA = -1;

    // 第二张影像在输入向量中的索引
    int indexB = -1;

    // 两张影像地面中心点之间的水平距离（单位与坐标系一致，如米）
    double centerDistance = 0.0;

    // 重叠得分：[0,1]，越接近 1 表示地面重叠越大；0 表示无重叠
    double overlapScore = 0.0;
};

// ============================================================
// 结构体：OverlapAnalysisResult
// 描述：整体重叠分析的输出结果集合。
// ============================================================
struct OverlapAnalysisResult
{
    // 各影像对应的地面中心三维坐标（世界坐标系），顺序与输入一致
    std::vector<std::array<double, 3>> centers;

    // 各影像估计的地面覆盖半径（影像四角到中心的平均地面距离），顺序与输入一致
    std::vector<double> footprintRadii;

    // 所有满足重叠条件的影像对结果，按 overlapScore 降序排列
    std::vector<OverlapPairResult> pairs;

    // 分析摘要信息字符串（含影像数量、重叠对数、邻域系数等）
    std::string detail;
};

// ============================================================
// 类：OverlapAnalyzer
// 描述：影像重叠分析的静态工具类，提供单一入口函数 analyze()。
//   所有计算逻辑封装于 analyze() 中，无需实例化。
// ============================================================
class OverlapAnalyzer
{
public:
    // --------------------------------------------------------
    // 函数：analyze
    // 功能：对输入影像集合执行地面重叠分析，输出所有重叠影像对及得分。
    // 参数：
    //   images         - 输入影像列表（影像路径 + 相机参数 + 分辨率）
    //   dem            - DEM 曲面指针（可为 nullptr，useFixedZ=true 时不需要）
    //   useFixedZ      - true = 使用固定高程面，false = 使用 DEM
    //   fixedZ         - 固定高程值（useFixedZ=true 时使用）
    //   neighborFactor - 邻域搜索倍数系数（控制重叠判断的阈值宽松程度，典型值 1.5~3.0）
    //                    threshold = neighborFactor * (r_i + r_j)
    //   result         - 分析结果输出指针（非 nullptr）
    //   errorMsg       - 可选错误信息输出
    // 返回值：成功返回 true，输入不足或反投影失败返回 false
    // --------------------------------------------------------
    static bool analyze(const std::vector<OverlapImageInput> &images,
                        const DemSurface *dem,
                        bool useFixedZ,
                        double fixedZ,
                        double neighborFactor,
                        OverlapAnalysisResult *result,
                        std::string *errorMsg = nullptr);
};

} // namespace xjw
