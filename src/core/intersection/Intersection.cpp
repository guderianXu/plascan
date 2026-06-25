#include "Intersection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

// ============================================================
// 文件：Intersection.cpp
// 功能：实现前方交汇（Front Intersection / 最短距离三角化）算法的
//       CPU 多线程并行版本。
//
// 算法概述（Midpoint / 最短公垂线法）：
//   给定两台相机中心 C1、C2 及对应像素发出的单位方向向量 d1、d2，
//   射线方程：
//       R1(s) = C1 + s * d1
//       R2(t) = C2 + t * d2
//   目标：求 s,t 使 ||R1(s) - R2(t)||² 最小（最小二乘意义下的交汇）。
//   对 s,t 分别求导（令梯度为零）可得 2×2 线性方程组，用 Cramer 法求解。
//   最终三维点估计为两最近点的中点：X = (R1(s) + R2(t)) / 2
//
// 注释详尽说明了每一步的数学推导与数值处理要点。
// ============================================================

namespace xjw
{

namespace
{

// 向量 3 点积：a·b
inline double dot3(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// 向量相加: a + b
inline std::array<double, 3> add3(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

// 向量相减: a - b
inline std::array<double, 3> sub3(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// 向量与标量相乘: a * s
inline std::array<double, 3> mul3(const std::array<double, 3> &a, double s)
{
    return {a[0] * s, a[1] * s, a[2] * s};
}

// 向量范数（欧几里得范数）
inline double norm3(const std::array<double, 3> &a)
{
    return std::sqrt(dot3(a, a));
}

// 向量归一化；若长度接近 0 或非数则返回 {NAN,NAN,NAN}
inline std::array<double, 3> normalize3(const std::array<double, 3> &a)
{
    double n = norm3(a);
    if (n < 1e-15 || !std::isfinite(n))
    {
        return {NAN, NAN, NAN};
    }
    return mul3(a, 1.0 / n);
}

// 判断向量元素是否均为有限数
inline bool isFinite3(const std::array<double, 3> &a)
{
    return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]);
}

// 将像素坐标 (u,v) 转换为世界坐标系下的射线方向向量（已归一化）
// 说明：
//  1) 先将像素坐标反内参得到相机坐标系下的归一化像平面点 (x,y,1)
//     x = (u - cu) / (udir * fu), y = (v - cv) / (vdir * fv)
//  2) 使用相机的 R（这里 R 约定为 cam->world）乘以相机坐标向量得到
//     世界方向（未归一化），再做归一化返回。归一化是为了数值稳定和
//     在后续最小距离解中简化符号（令 a = d1·d1 ≈ 1）
inline std::array<double, 3> pixelToWorldRay(const Camera &cam, double u, double v) 
{
    const double x = (u - cam.principalX()) / (cam.uAxisSign() * cam.focalX());
    const double y = (v - cam.principalY()) / (cam.vAxisSign() * cam.focalY());
    const double zSign = cam.depthAxisFlipped() ? -1.0 : 1.0;

    const std::array<double, 3> ray_cam{x * zSign, y * zSign, zSign};
    const auto R = cam.cameraToWorldRotation();
    // R 存储为行主序的 3x3 cam->world 矩阵，因此 world_dir = R * ray_cam
    std::array<double, 3> ray_world{
        R[0] * ray_cam[0] + R[1] * ray_cam[1] + R[2] * ray_cam[2],
        R[3] * ray_cam[0] + R[4] * ray_cam[1] + R[5] * ray_cam[2],
        R[6] * ray_cam[0] + R[7] * ray_cam[1] + R[8] * ray_cam[2]};
    return normalize3(ray_world);
}

// 计算两单位向量之间的夹角（度）
inline double angleDegrees(const std::array<double, 3> &d1, const std::array<double, 3> &d2)
{
    double c = dot3(d1, d2);
    // 裁剪以防数值超出 [-1,1] 导致 acos NaN
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c) * 180.0 / M_PI;
}

} // namespace

/*
 * 三角化单点函数说明（最短距离 / 中点法）
 *
 * 给定两台相机中心 C1, C2 以及两条单位方向向量 d1, d2，射线表示为：
 *   P1(s) = C1 + s * d1
 *   P2(t) = C2 + t * d2
 * 要求 s,t 使得两点间距离 ||P1(s) - P2(t)||^2 最小。对 s,t 求导得到线性方程组：
 *   [ d1·d1   -d1·d2 ] [ s ] = [ -w·d1 ]
 *   [-d1·d2    d2·d2 ] [ t ]   [  w·d2 ]
 * 其中 w = C2 - C1。
 * 上式可重写为常见的 2x2 线性系统（本文实现采用等价形式并直接用 Cramer 法解）。
 * 解出 s,t 后取 P1(s) 与 P2(t) 的中点作为最终三维估计 X。
 *
 * 本实现要点与鲁棒性处理：
 *  - 在计算中我们使用归一化方向向量，因而 a = d1·d1 ≈ 1, c = d2·d2 ≈ 1。
 *  - 当两条射线几乎平行时，2x2 系数矩阵的行列式 den 会接近 0，
 *    本实现通过阈值检测 degeneracy（|den| < 1e-12）并退化到简单的深度估计
 *   （用 t = e/c 或类似启发式），以避免数值不稳定。
 *  - 计算出的 P1,P2 之间的距离存为 `ray_miss_distance`，用于衡量两射线未严格相交时的最短距离。
 *  - 最后计算重投影残差（把 X 投影回两个相机像平面，比较与原始观测像素差）
 */

Intersection::Result Intersection::intersectPair(const Camera &cam1,
                                                 double u1,
                                                 double v1,
                                                 const Camera &cam2,
                                                 double u2,
                                                 double v2)
{
    Result out; // 默认 invalid

    // 相机中心
    const auto C1 = cam1.cameraCenter();
    const auto C2 = cam2.cameraCenter();
    // 将像素坐标映射为世界坐标系下的射线方向（已归一化）
    const auto d1 = pixelToWorldRay(cam1, u1, v1);
    const auto d2 = pixelToWorldRay(cam2, u2, v2);

    // 如果任意一个方向非法（包含 NaN 或 非有限数），返回无效结果
    if (!isFinite3(d1) || !isFinite3(d2))
    {
        return out;
    }

    // angle_deg 在三维点求出后精确计算（见下方），此处先置 NAN
    out.angle_deg = NAN;

    // 准备解线性系统的各个内积项
    // w0 = C1 - C2（注意与数学推导记号方向关系，只要一致即可）
    const auto w0 = sub3(C1, C2);
    const double a = dot3(d1, d1); // ≈1
    const double b = dot3(d1, d2);
    const double c = dot3(d2, d2); // ≈1
    const double d = dot3(d1, w0);
    const double e = dot3(d2, w0);

    const double den = a * c - b * b; // 行列式
    double s = 0.0;
    double t = 0.0;

    // 数值退化处理：当 den 很小时，说明视线近似平行/共面，无法稳定解出 s,t。
    if (std::abs(den) < 1e-12)
    {
        // 在退化情况下，使用一个简单的 t = e / c 作为备选解（投影到第二条射线）
        t = (c > 1e-12) ? (e / c) : 0.0;
    } else {
        // Cramer 法或者解析解，等价于最小二乘求解
        s = (b * e - c * d) / den;
        t = (a * e - b * d) / den;
    }

    // 最近点与中点：P1 = C1 + s*d1, P2 = C2 + t*d2, X = (P1 + P2) / 2
    const auto P1 = add3(C1, mul3(d1, s));
    const auto P2 = add3(C2, mul3(d2, t));
    const auto X = mul3(add3(P1, P2), 0.5);

    out.point = X;
    // 两射线最近点之间的距离，用作”几何一致性残差”
    out.ray_miss_distance = norm3(sub3(P1, P2));

    // 交会角：从三维点 X 分别看向两相机中心 C1、C2 的方向夹角。
    // 这才是摄影测量意义上的交会角（convergence angle），反映三角化的几何强度。
    // 对于正常立体像对，该角通常为几度到几十度；射线方向夹角 d1·d2 接近 0°，不能用。
    {
        const auto toC1 = normalize3(sub3(C1, X));
        const auto toC2 = normalize3(sub3(C2, X));
        if (isFinite3(toC1) && isFinite3(toC2))
            out.angle_deg = angleDegrees(toC1, toC2);
    }

    // 计算重投影残差：将 X 投影回像面，与原始观测像素差的欧氏距离
    double uv_proj[2];
    double e1 = std::numeric_limits<double>::quiet_NaN();
    double e2 = std::numeric_limits<double>::quiet_NaN();
    double du1 = NAN, dv1 = NAN, du2 = NAN, dv2 = NAN;

    const double world[3] = {X[0], X[1], X[2]};
    if (cam1.projectWorldPoint(world, uv_proj))
    {
        du1 = uv_proj[0] - u1;
        dv1 = uv_proj[1] - v1;
        e1 = std::sqrt(du1 * du1 + dv1 * dv1);
    }

    if (cam2.projectWorldPoint(world, uv_proj))
    {
        du2 = uv_proj[0] - u2;
        dv2 = uv_proj[1] - v2;
        e2 = std::sqrt(du2 * du2 + dv2 * dv2);
    }

    out.reproj_error_cam1 = e1;
    out.reproj_error_cam2 = e2;

    // RMS：两个相机重投影误差（欧氏距离）的均方根
    // e1 = sqrt(du1²+dv1²)，e2 = sqrt(du2²+dv2²)
    // RMS = sqrt((e1² + e2²) / 2)
    if (std::isfinite(e1) && std::isfinite(e2))
    {
        out.reproj_error_rms = std::sqrt((e1 * e1 + e2 * e2) / 2.0);
    }

    // 最后做一些健康性检查：要求深度 s,t 为正（点在相机前方），并且关键量为有限数
    const bool positive_depth = (s > 0.0 && t > 0.0);
    const bool finite_main = isFinite3(out.point)
                             && std::isfinite(out.angle_deg)
                             && std::isfinite(out.ray_miss_distance);
    out.valid = finite_main && positive_depth;
    return out;
}

/*
 * 批量三角化：将点集合划分为若干块并为每块创建线程进行并行计算。
 * 设计要点：
 *  - 使用 std::thread 而非更复杂的线程池以保持实现简单且跨平台；
 *  - chunk 分块策略为均匀划分，线程数默认使用硬件并发度（hardware_concurrency），
 *    但不会超过样本数。
 *  - 每个线程对其负责的区间内逐点调用 intersectPair，因此单点函数的鲁棒性
 *    直接传导到批量结果。
 */
std::vector<Intersection::Result> Intersection::intersectBatch(const Camera &cam1,
                                                               const std::vector<std::pair<double, double>> &pts1,
                                                               const Camera &cam2,
                                                               const std::vector<std::pair<double, double>> &pts2,
                                                               unsigned int numThreads) {
    const size_t n = std::min(pts1.size(), pts2.size());
    std::vector<Result> out(n);
    if (n == 0) {
        return out;
    }

    unsigned int threads = numThreads;
    if (threads == 0) {
        // 默认并行度：硬件并发数与 1 取较大值（防止返回 0）
        threads = std::max(1u, std::thread::hardware_concurrency());
    }
    // 不要创建多于样本数的线程
    threads = std::min<unsigned int>(threads, static_cast<unsigned int>(n));

    std::vector<std::thread> workers;
    workers.reserve(threads);

    const size_t chunk = (n + threads - 1) / threads; // 划分块大小（向上取整）
    for (unsigned int tid = 0; tid < threads; ++tid) {
        const size_t begin = tid * chunk;
        const size_t end = std::min(n, begin + chunk);
        if (begin >= end) {
            continue;
        }

        // 每个线程处理 [begin, end) 区间的点
        workers.emplace_back([&, begin, end]() {
            for (size_t i = begin; i < end; ++i) {
                out[i] = intersectPair(cam1,
                                       pts1[i].first,
                                       pts1[i].second,
                                       cam2,
                                       pts2[i].first,
                                       pts2[i].second);
            }
        });
    }

    // 等待所有线程完成
    for (auto &worker : workers) {
        worker.join();
    }
    return out;
}

} // namespace xjw
