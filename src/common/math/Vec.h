#pragma once

// =============================================================================
// 文件: math/Vec.h
// 功能: 通用轻量几何基础类型。
//
// 这些类型在摄影测量、点云、网格、SfM 等多个模块中均有使用，因此统一放于
// common/math 下，避免各模块重复定义。
//
// 包含:
//   - Point3f    三维坐标 / 向量（位置、法向量、平移量等）
//   - Point2f    二维纹理坐标（UV）
//   - ColorRGBA  RGBA 颜色（各通道 8bit）
// =============================================================================

#include <cstdint>

namespace xjw
{

/**
 * @brief 三维浮点坐标或方向向量。
 *
 * 可用于表示点位置、法向量、平移量等任意三维量。
 */
struct Point3f
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

/**
 * @brief 二维浮点坐标（UV / 像素 / 纹理坐标）。
 */
struct Point2f
{
    float u = 0.0f;
    float v = 0.0f;
};

/**
 * @brief RGBA 颜色，各通道 8bit 无符号整型。
 *
 * 默认值为不透明白色（255, 255, 255, 255）。
 */
struct ColorRGBA
{
    uint8_t r = 255;
    uint8_t g = 255;
    uint8_t b = 255;
    uint8_t a = 255;
};

// ── Point3f 运算符 ──────────────────────────────────────────────────────────

inline Point3f operator+(const Point3f &a, const Point3f &b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Point3f operator-(const Point3f &a, const Point3f &b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Point3f operator*(const Point3f &a, float s)          { return {a.x*s,   a.y*s,   a.z*s};   }
inline Point3f operator*(float s, const Point3f &a)          { return {a.x*s,   a.y*s,   a.z*s};   }
inline Point3f operator/(const Point3f &a, float s)          { return {a.x/s,   a.y/s,   a.z/s};   }
inline Point3f &operator+=(Point3f &a, const Point3f &b)     { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
inline Point3f &operator-=(Point3f &a, const Point3f &b)     { a.x-=b.x; a.y-=b.y; a.z-=b.z; return a; }

inline float dot(const Point3f &a, const Point3f &b)         { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Point3f cross(const Point3f &a, const Point3f &b)
{
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

// ── Point2f 运算符 ──────────────────────────────────────────────────────────

inline Point2f operator+(const Point2f &a, const Point2f &b) { return {a.u+b.u, a.v+b.v}; }
inline Point2f operator-(const Point2f &a, const Point2f &b) { return {a.u-b.u, a.v-b.v}; }
inline Point2f operator*(const Point2f &a, float s)          { return {a.u*s,   a.v*s};   }
inline Point2f operator*(float s, const Point2f &a)          { return {a.u*s,   a.v*s};   }
inline Point2f &operator+=(Point2f &a, const Point2f &b)     { a.u+=b.u; a.v+=b.v; return a; }
inline float dot(const Point2f &a, const Point2f &b)         { return a.u*b.u + a.v*b.v; }

} // namespace xjw
