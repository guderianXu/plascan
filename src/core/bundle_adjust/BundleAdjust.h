#pragma once

// ============================================================
// 文件：BundleAdjust.h
// 功能：定义光束法平差（Bundle Adjustment）相关数据结构和核心类。
//
// 算法概述：
//   - 采用交替优化策略：先固定相机优化三维点，再固定三维点优化相机位姿
//   - 数值集加速：采用高斯牛顿 + LM 风格週界，雅可比通过有限差分构造雅可比
//   - 鲁棒性：使用 Huber 损失挺压粗差省点影响
// ============================================================

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "Camera.h"

namespace xjw {

/**
 * @brief 单条观测：某个三维点在某台相机中对应的像点坐标。
 *
 * 每条观测将一个三维轨迹点与某台相机中的匹配点关联起来，
 * 是重投影误差计算的最小单元。
 */
struct BAObservation
{
    int cameraIndex = -1; ///< 观测所属相机的索引（对应 cameras 向量下标）
    double u = 0.0;       ///< 相机像平面上的 u（列）坐标（像素）
    double v = 0.0;       ///< 相机像平面上的 v（行）坐标（像素）
};

/**
 * @brief 轨迹：一个三维点及其在多幅图像中的观测集合。
 *
 * 轨迹表示多相机共观的同一个物方点，是光束法平差的核心数据单元。
 * 每条轨迹至少需要 2 个观测才能参与优化。
 */
struct BATrack
{
    std::array<double, 3> initialPoint{{0.0, 0.0, 0.0}}; ///< 三维点的初始坐标（优化起始值）
    std::vector<BAObservation> observations;              ///< 所有相机中对该点的观测列表
};

/**
 * @brief 优化选项：控制光束法平差迭代策略和收敛条件的参数集。
 */
struct BAOptions
{
    int maxIterations = 20;       ///< 外层交替优化最大迭代轮数（点+相机各一次为一轮）
    int maxPointIterations = 12;  ///< 每轮内点位置优化的最大迭代次数
    int maxCameraIterations = 10; ///< 每轮内相机位姿优化的最大迭代次数
    bool refineCameraPose = true; ///< 是否同时优化相机位姿（false 则仅优化三维点）

    double huberDelta = 3.0;        ///< Huber 损失阈值（像素），残差>delta 则降低权重以抑制粗差
    double finiteDiffEps = 1e-6;   ///< 有限差分步长（中央差分: ±eps），用于近似雅可比
    double damping = 1e-3;         ///< Levenberg-Marquardt 初始阻尼因子（自适应调整）
    double stepTolerance = 1e-8;   ///< 收敛判断阈值：步长小于此值则认为已收敛

    // ── Gauge 固定 ──────────────────────────────────────────────────────────
    /// 固定这些索引对应的相机位姿（不参与 camera 优化阶段）。
    /// 全局 BA 时通常固定索引 0，消除坐标系漂移（7 自由度 gauge 问题）。
    std::vector<int> fixedCameraIndices;

    // ── 离群点过滤 ─────────────────────────────────────────────────────────
    /// 是否在每轮 BA 结束后自动过滤高重投影误差点。
    bool enablePointFilter = true;
    /// 过滤阈值（像素）：rmsAfter 超过此值的点会被标记为 invalid。
    double filterMaxReprojError = 2.5;  ///< 重投影误差上限（像素），超过此值的点标记为无效
    /// 过滤倍率：同时过滤 rmsAfter > filterSigmaFactor × 全局中位数 rms 的点。
    /// 设为 0 则禁用此规则，仅使用固定阈值。
    double filterSigmaFactor = 3.0;
    // ── 并行计算 ───────────────────────────────────────────
    /// OpenMP 线程数；0 表示使用系统默认（OMP_NUM_THREADS 或 CPU 核心数）。
    int numThreads = 0;

    // ── 任务控制 ──────────────────────────────────────────────────────────
    /// 外部取消标志；GUI/CLI 长任务可在外层迭代边界中止 BA。
    std::shared_ptr<std::atomic<bool>> cancelFlag;
    /// 外层迭代进度回调，返回 false 表示调用方要求停止后续迭代。
    std::function<bool(int currentIteration, int maxIterations, double avgRms, int validPoints)> progressCallback;
};

/**
 * @brief 单个轨迹/三维点的优化结果。
 *
 * 记录优化前后重投影 RMS 误差及收敛状态，方便谨歌分析。
 */
struct BARefinedPoint
{
    bool valid = false;                              ///< 优化结果是否有效（点坐标有限且 rmsAfter 有限）
    bool converged = false;                          ///< 是否在容差内收敛
    std::array<double, 3> point{{0.0, 0.0, 0.0}};   ///< 优化后的三维坐标
    double rmsBefore = 0.0;  ///< 优化前重投影 RMS 误差（像素）
    double rmsAfter = 0.0;   ///< 优化后重投影 RMS 误差（像素）
    int iterations = 0;      ///< 实际迭代次数
};

/**
 * @brief 整个光束法平差的综合结果结构。
 *
 * 包含所有轨迹的优化结果、全局统计指标和优化后的相机列表。
 */
struct BAResult
{
    int totalTracks = 0;        ///< 输入轨迹总数
    int optimizedTracks = 0;    ///< 成功优化的轨迹数
    double meanRmsBefore = 0.0; ///< 优化前所有轩迹重投影 RMS 的均値
    double meanRmsAfter = 0.0;  ///< 优化后所有轨迹重投影 RMS 的均値
    int refinedCameraCount = 0; ///< 最后一轮中实际被更新的相机数量

    std::vector<BARefinedPoint> points;   ///< 每条轨迹对应的点优化结果（与输入 tracks 索引一一对应）
    std::vector<Camera> refinedCameras;   ///< 优化后的相机列表（与输入 cameras 長度相同）
};

/**
 * @brief 光束法平差核心类（静态接口）。
 *
 * 提供一个静态方法 optimizePoints，负责接收相机列表和轨迹列表，
 * 返回优化后的点坐标、相机位姿及统计信息。
 */
class BundleAdjust
{
public:
    /**
     * @brief 执行光束法平差优化（交替优化点位置和相机位姿）。
     *
     * 优化流程：
     *   1. 初始化点集，计算前置重投影误差
     *   2. 外层迭代 maxIterations 轮：
     *      a) 对每条轨迹优化三维点坐标
     *      b) （可选）对每台相机优化 6-DOF 位姿
     *   3. 收集全局统计数据
     *
     * @param cameras  初始相机列表（优化过程中将作副本修改）
     * @param tracks   轨迹列表，每条轨迹包含初始三维点和多相机观测
     * @param options  优化选项（可选，默认使用 BAOptions）
     * @return         BAResult 结个，包含优化后点坐标、相机位姿及误差统计
     */
    static BAResult optimizePoints(const std::vector<Camera> &cameras,
                                   const std::vector<BATrack> &tracks,
                                   const BAOptions &options = BAOptions());
};

} // namespace xjw
