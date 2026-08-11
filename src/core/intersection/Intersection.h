#pragma once

// ============================================================
// 文件：Intersection.h
// 功能：提供基于最短距离法（Midpoint / 前方交汇）的立体视觉三角化接口。
//       给定两台已标定相机及对应的匹配像素坐标，求解三维空间中的对应点坐标。
// 典型应用：摄影测量前方交汇、SfM/MVS 点云生成、相机标定检验。
// ============================================================

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "FramePinholeCamera.h"

namespace xjw {

// ============================================================
// 类：Intersection
// 职责：封装前方交汇（Front Intersection）算法的核心逻辑，提供单点和批量两种接口。
//       算法采用最短距离中点法（Midpoint Method），即寻找两条空间射线之间的
//       公垂线中点作为三维点估计，同时计算交汇质量指标。
// ============================================================
class Intersection
{
public:
    // --------------------------------------------------------
    // 结构体：Result
    // 描述：单对点的前方交汇结果，包含三维坐标及若干质量评价指标。
    // --------------------------------------------------------
    struct Result
    {
        // 三角化是否成功（深度为正、坐标有限）
        bool valid = false;

        // 三角化得到的三维点坐标（世界坐标系，与相机标定使用的坐标系一致）
        std::array<double, 3> point{{NAN, NAN, NAN}};

        // 两条视线的夹角（度），越大表示三角化几何条件越好（基线/深度比大）
        double angle_deg = NAN;

        // 两射线最近点之间的距离（公垂线长度），理论为 0，数值越小说明匹配越好
        double ray_miss_distance = NAN;

        // 三维点重投影回第一台相机的像素重投影误差（像素）
        double reproj_error_cam1 = NAN;

        // 三维点重投影回第二台相机的像素重投影误差（像素）
        double reproj_error_cam2 = NAN;

        // 两台相机重投影误差的均方根（RMS），综合质量评价指标
        double reproj_error_rms = NAN;
    };

    // --------------------------------------------------------
    // 函数：intersectPair
    // 功能：对单对匹配像素坐标执行前方交汇，返回三维点及质量评价。
    // 参数：
    //   cam1       - 第一台相机的标定参数（内参 + 外参）
    //   u1, v1     - 第一张影像上的像素坐标（列、行）
    //   cam2       - 第二台相机的标定参数
    //   u2, v2     - 第二张影像上对应的像素坐标
    // 返回值：Result 结构体，若成功则 valid=true
    // --------------------------------------------------------
    static Result intersectPair(const FramePinholeCamera &cam1,
                                double u1,
                                double v1,
                                const FramePinholeCamera &cam2,
                                double u2,
                                double v2);

    // --------------------------------------------------------
    // 函数：intersectBatch
    // 功能：对一批匹配点对执行前方交汇，内部使用多线程并行加速。
    // 参数：
    //   cam1       - 第一台相机
    //   pts1       - 第一张影像上的像素坐标列表（u,v 对）
    //   cam2       - 第二台相机
    //   pts2       - 第二张影像上对应像素坐标列表（与 pts1 等长）
    //   numThreads - 使用的线程数，0 表示自动使用硬件并发度
    // 返回值：与输入等长的 Result 向量
    // --------------------------------------------------------
    static std::vector<Result> intersectBatch(const FramePinholeCamera &cam1,
                                              const std::vector<std::pair<double, double>> &pts1,
                                              const FramePinholeCamera &cam2,
                                              const std::vector<std::pair<double, double>> &pts2,
                                              unsigned int numThreads = 0);
};

} // namespace xjw
