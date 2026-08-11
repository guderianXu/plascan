#pragma once

/**
 * @file BundleAdjustNativeCudaTypes.h
 * @brief native CUDA BA 的主机侧扁平数据布局。
 *
 * 这些结构是通用 FramePinholeCamera/BATrack 与设备 POD 结构之间的中间表示。索引均指向当前
 * Workset，而 `originalTrackIndex` 用于恢复调用方顺序。
 */

#include <array>
#include <vector>

namespace xjw::detail::native_cuda
{

/**
 * @brief 一台相机的只读投影快照。
 *
 * `cameraToWorldRotation` 按行存储 camera-to-world 旋转。投影时实际使用其转置，
 * 将 `(X - C)` 从世界坐标变换到相机坐标。内参和 Brown-Conrady 畸变在当前
 * native CUDA 点优化阶段保持固定。
 */
struct HostCamera
{
    std::array<double, 9> cameraToWorldRotation{}; ///< 行优先 3x3 camera-to-world 旋转。
    std::array<double, 3> cameraCenter{}; ///< 世界坐标系相机中心 C。
    double focalX = 1.0; ///< 水平焦距，像素。
    double focalY = 1.0; ///< 垂直焦距，像素。
    double principalX = 0.0; ///< 主点 u 坐标，像素。
    double principalY = 0.0; ///< 主点 v 坐标，像素。
    double radialK1 = 0.0; ///< Brown-Conrady 一阶径向畸变。
    double radialK2 = 0.0; ///< Brown-Conrady 二阶径向畸变。
    double radialK3 = 0.0; ///< Brown-Conrady 三阶径向畸变。
    double tangentialP1 = 0.0; ///< 第一切向畸变系数。
    double tangentialP2 = 0.0; ///< 第二切向畸变系数。
    int uAxisSign = 1; ///< 归一化横坐标到像素 u 的方向符号。
    int vAxisSign = 1; ///< 归一化纵坐标到像素 v 的方向符号。
    int depthAxisFlipped = 0; ///< 非零表示相机前方沿局部 -Z。
};

/// 压缩后的一个待优化三维点及其连续观测区间。
struct HostPoint
{
    std::array<double, 3> xyz{}; ///< 当前世界坐标。
    int originalTrackIndex = -1; ///< 输入 tracks 中的索引。
    int observationBegin = 0; ///< Workset::observations 中首个观测下标。
    int observationCount = 0; ///< 属于该点的连续观测数量。
};

/// 一条二维重投影观测，索引均指向当前 Workset。
struct HostObservation
{
    int cameraIndex = -1; ///< Workset::cameras 下标。
    int pointIndex = -1; ///< Workset::points 下标。
    double u = 0.0; ///< 实测像素横坐标。
    double v = 0.0; ///< 实测像素纵坐标。
    double weight = 1.0; ///< 非负残差权重，在鲁棒核之前生效。
};

/**
 * @brief 一次 native CUDA 调用的完整主机工作集。
 *
 * observations 按 point 分段连续排列，使每个 CUDA block/线程可以顺序读取同一点
 * 的观测。HostPoint::originalTrackIndex 用于恢复输入轨迹顺序。
 */
struct Workset
{
    std::vector<HostCamera> cameras;
    std::vector<HostPoint> points;
    std::vector<HostObservation> observations;
};

} // namespace xjw::detail::native_cuda
