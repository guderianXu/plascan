#pragma once

// =============================================================================
// 文件: math/Vec3Ops.h
// 功能: 三维向量基本运算工具（内联、无依赖）。
//
// 供 BundleAdjust、SparsePointCloudProcessor 等模块共用，
// 避免在各匿名命名空间中重复定义相同的 norm3/dot3/cross3 等函数。
// =============================================================================

#include <array>
#include <cmath>

namespace xjw
{
namespace vec3
{

inline double dot(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm(const std::array<double, 3> &v)
{
    return std::sqrt(dot(v, v));
}

inline std::array<double, 3> cross(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

inline std::array<double, 3> subtract(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline bool isFinite(const std::array<double, 3> &v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

inline std::array<double, 3> normalize(const std::array<double, 3> &v)
{
    const double n = norm(v);
    if (n < 1e-12) return v;
    return {v[0] / n, v[1] / n, v[2] / n};
}

} // namespace vec3
} // namespace xjw
