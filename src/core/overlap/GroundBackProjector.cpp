#include "GroundBackProjector.h"
#include "io/PathIO.h"

// ============================================================
// 文件：GroundBackProjector.cpp
// 功能：实现 DemSurface（DEM 高程曲面）和 GroundBackProjector（反投影工具）。
//
// 主要模块：
//   1) DemSurface      - 从 XYZ 文件加载 DEM 点云并提供最近邻高程查询
//   2) pixelRayWorld   - 像素坐标 → 世界系射线（相机内外参解算）
//   3) backProjectToFixedZ  - 射线与水平面求交（解析解）
//   4) backProjectWithDem   - 射线与 DEM 曲面求交（迭代法）
//   5) imageCenterToGround  - 影像中心点地面坐标
//   6) estimateFootprintRadius - 影像地面覆盖半径估算
//
// 坐标系约定：
//   - 像素坐标：(u=列, v=行)，左上角为原点
//   - 相机坐标系的轴方向和物理前方由 FramePinholeNumericState 统一定义，
//     不在此处假设固定的深度轴方向
//   - 世界坐标系：由相机标定确定，通常为 UTM 或本地水平坐标系
//   - 高程 Z：向上为正
// ============================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace {

// 计算两个三维点在 XY 平面（水平面）上的投影距离（忽略 Z 分量）
// 用于将地面点之间的水平距离与影像覆盖半径比较
double distance2D(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    return std::sqrt(dx * dx + dy * dy);
}

double distance3D(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool finitePoint(const std::array<double, 3> &point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

} // namespace

namespace xjw {

// ============================================================
// 函数：DemSurface::loadFromXYZ
// 功能：从 XYZ 格式文本文件（每行：x y z，# 为注释）加载 DEM 点云，
//       并在加载完成后：
//         1) 计算所有点的 Z 均值（用于初始射线步长估计）
//         2) 构建 KD 树（用于高效最近邻高程查询）
// ============================================================
bool DemSurface::loadFromXYZ(const std::string &path, std::string *errorMsg)
{
    // 清空已有数据，准备重新加载
    _points.clear();
    _xyPoints.clear();
    _meanHeight = 0.0;

    std::ifstream ifs = xjw::common::io::openInputFile(path, std::ios::in);
    if (!ifs.is_open()) {
        if (errorMsg) *errorMsg = "无法打开 DEM 文件: " + path;
        return false;
    }

    // 逐行解析：跳过空行和 # 开头的注释行
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double x = 0.0, y = 0.0, z = 0.0;
        if (!(iss >> x >> y >> z) || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            continue; // 解析失败或非有限数据则跳过该行
        }
        _points.push_back({x, y, z});
    }

    if (_points.empty()) {
        if (errorMsg) *errorMsg = "DEM 中未读取到有效 XYZ 点";
        return false;
    }

    // 构建 KD 树所需的 2D 点集（只需 x, y 坐标），同时累加 Z 值以计算均值
    _xyPoints.reserve(_points.size());
    double sumZ = 0.0;
    for (size_t i = 0; i < _points.size(); ++i) {
        // 将 3D 点压缩为 2D 点（附原始下标），用于 KD 树水平查询
        _xyPoints.push_back(DemKdTree2D::Point{{_points[i][0], _points[i][1]}, static_cast<int>(i)});
        sumZ += _points[i][2];
    }

    // 建立 KD 树空间索引，后续 sampleHeight 调用依赖此索引加速
    _index.build(_xyPoints);

    // 计算平均高程，用作射线行进初始深度估计（避免迭代从零开始）
    _meanHeight = sumZ / static_cast<double>(_points.size());
    return true;
}

// 在 (x,y) 处查询最近邻高程：通过 KD 树找到平面最近点，返回其 Z 值
// idx 是 PlaPoint KDTree 点的 payload，即 _points 中的原始下标
bool DemSurface::sampleHeight(double x, double y, double *z, double *xyDistance) const
{
    if (!z || !std::isfinite(x) || !std::isfinite(y) || _index.empty()) return false;
    double dist = 0.0;
    // KD 树最近邻查询：返回 PlaPoint KDTree 点的 payload（即 _points 的下标），dist 为水平距离
    const int idx = _index.nearest(DemKdTree2D::CoordinateArray{x, y}, &dist);
    if (idx < 0 || idx >= static_cast<int>(_points.size())) return false;
    *z = _points[static_cast<size_t>(idx)][2]; // 取对应点的高程 Z
    if (!std::isfinite(*z) || !std::isfinite(dist)) return false;
    if (xyDistance) *xyDistance = dist;          // 可选：返回水平距离（评估外推精度）
    return true;
}

// 检查 DEM 是否已有有效数据（至少加载了一个点）
bool DemSurface::valid() const
{
    return !_points.empty();
}

// 返回 DEM 点云的平均高程
double DemSurface::meanHeight() const
{
    return _meanHeight;
}

// ============================================================
// 函数：GroundBackProjector::pixelRayWorld（私有）
// 功能：将像素坐标 (u,v) 转换为世界坐标系下的射线（起点 + 归一化方向）。
//   数学步骤：
//     ① 通过 FramePinholeNumericState::rayForPixel() 完成去畸变、轴方向、
//        深度轴和相机到世界坐标变换；这里不再复制一套针孔公式。
//     ② 返回统一数值状态定义的世界系单位射线。
// ============================================================
bool GroundBackProjector::pixelRayWorld(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                                        double u,
                                        double v,
                                        std::array<double, 3>* origin,
                                        std::array<double, 3>* dir,
                                        std::string* errorMsg)
{
    if (!origin || !dir)
    {
        return false;
    }

    std::string validationError;
    if (!camera.validateNumericalState(&validationError))
    {
        if (errorMsg)
        {
            *errorMsg = "数值相机状态无效";
            if (!validationError.empty())
            {
                *errorMsg += ": " + validationError;
            }
        }
        return false;
    }

    xjw::camera_models::frame_pinhole::FramePinholeNumericState::Ray ray;
    if (!camera.rayForPixel({u, v}, &ray))
    {
        if (errorMsg)
        {
            *errorMsg = "无法计算有效射线方向";
        }
        return false;
    }

    *origin = ray.origin;
    *dir = ray.direction;
    return true;
}

// ============================================================
// 函数：GroundBackProjector::backProjectToFixedZ
// 功能：射线与水平面 Z=fixedZ 求交（解析解）。
//   数学：
//     射线方程：P(t) = origin + t * dir
//     令 P_z = fixedZ，解得：t = (fixedZ - origin.z) / dir.z
//     交点：ground = origin + t * dir（z 分量赋为 fixedZ）
//   条件：
//     - |dir.z| 不能接近 0（否则射线与高程面平行，无交点或无穷远）
//     - t > 0（交点必须在相机前方，t ≤ 0 表示面在相机后方）
// ============================================================
bool GroundBackProjector::backProjectToFixedZ(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                                              double u,
                                              double v,
                                              double fixedZ,
                                              std::array<double, 3>* ground,
                                              std::string* errorMsg)
{
    if (!ground) return false;
    if (!std::isfinite(fixedZ))
    {
        if (errorMsg)
        {
            *errorMsg = "固定高程必须是有限数值";
        }
        return false;
    }

    // 计算世界坐标系下的射线（起点 + 归一化方向）
    std::array<double, 3> origin;
    std::array<double, 3> dir;
    if (!pixelRayWorld(camera, u, v, &origin, &dir, errorMsg)) return false;

    // 检查射线 Z 分量：若接近 0，则射线近乎水平，无法与固定高程面相交
    if (std::abs(dir[2]) < 1e-12) {
        if (errorMsg) *errorMsg = "射线与固定高程平面近平行";
        return false;
    }

    // 计算参数 t（射线行进长度）
    const double t = (fixedZ - origin[2]) / dir[2];

    if (!std::isfinite(t))
    {
        if (errorMsg) *errorMsg = "固定高程交点距离无效";
        return false;
    }

    // t ≤ 0 意味着交点在相机后方（反向延伸），物理上无意义
    if (t <= 0.0) {
        if (errorMsg) *errorMsg = "固定高程交点在相机后方";
        return false;
    }

    // 计算交点，Z 分量直接赋值为 fixedZ（避免浮点误差累积）
    const std::array<double, 3> candidate{origin[0] + t * dir[0], origin[1] + t * dir[1], fixedZ};
    if (!finitePoint(candidate))
    {
        if (errorMsg) *errorMsg = "固定高程交点坐标无效";
        return false;
    }
    *ground = candidate;
    return true;
}

// ============================================================
// 函数：GroundBackProjector::backProjectWithDem
// 功能：射线与 DEM 曲面迭代求交（类牛顿步迭代）。
//   原理：
//     射线上任意一点 p(t) = origin + t * dir
//     DEM 在 (p.x, p.y) 处的高程为 dem_z
//     令 f(t) = p(t).z - dem_z(p(t).x, p(t).y) = 0
//     牛顿步更新：t_new = t - f(t) / (df/dt) ≈ t - diff / dir.z
//       其中 diff = p.z - dem_z，忽略 DEM 的水平梯度（近似假设 DEM 较平坦）
//   初始化：以 DEM 均值高程作为初始深度估计
//   迭代限制：最多 32 次，收敛条件 |diff| < 1e-3（毫米级精度）
//   DEM 查询失败时扩大 t 继续搜索；超过迭代预算或没有前向交点则失败。
// ============================================================
bool GroundBackProjector::backProjectWithDem(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                                             double u,
                                             double v,
                                             const DemSurface& dem,
                                             std::array<double, 3>* ground,
                                             std::string* errorMsg)
{
    if (!ground) return false;
    if (!dem.valid()) {
        if (errorMsg) *errorMsg = "DEM 数据不可用";
        return false;
    }

    // 计算世界坐标系下的射线
    std::array<double, 3> origin;
    std::array<double, 3> dir;
    if (!pixelRayWorld(camera, u, v, &origin, &dir, errorMsg)) return false;

    // 射线 Z 分量接近 0（射线近似水平），无法与高程面稳定求交
    if (std::abs(dir[2]) < 1e-12) {
        if (errorMsg) *errorMsg = "DEM 求交失败：射线方向异常";
        return false;
    }

    // 初始 t：用 DEM 均值高程估算初始交点深度
    const double zMean = dem.meanHeight();
    if (!std::isfinite(zMean))
    {
        if (errorMsg)
        {
            *errorMsg = "DEM 平均高程无效";
        }
        return false;
    }
    double t = (zMean - origin[2]) / dir[2];
    if (!std::isfinite(t))
    {
        if (errorMsg)
        {
            *errorMsg = "DEM 初始交点距离无效";
        }
        return false;
    }
    if (t <= 0.0)
    {
        if (errorMsg)
        {
            *errorMsg = "DEM 交点在相机后方";
        }
        return false;
    }

    bool found = false;
    std::array<double, 3> best = origin; // 保存每次迭代的最优近似点

    // 迭代求交（最多 32 次）
    for (int iter = 0; iter < 32; ++iter) {
        // 计算当前 t 对应的射线上的点
        const std::array<double, 3> p{origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]};

        // 查询该水平位置 (p.x, p.y) 的 DEM 高程
        if (!finitePoint(p))
        {
            if (errorMsg) *errorMsg = "DEM 迭代点坐标无效";
            return false;
        }
        double zDem = 0.0;
        if (!dem.sampleHeight(p[0], p[1], &zDem)) {
            // DEM 查询失败（可能超出覆盖范围），扩大 t 继续延射线前进
            t *= 1.3;
            if (!std::isfinite(t) || t <= 0.0)
            {
                if (errorMsg) *errorMsg = "DEM 搜索距离无效";
                return false;
            }
            continue;
        }
        if (!std::isfinite(zDem))
        {
            if (errorMsg) *errorMsg = "DEM 高程无效";
            return false;
        }

        // 残差：当前射线点的 Z 与 DEM 高程的差值
        const double diff = p[2] - zDem;
        if (!std::isfinite(diff))
        {
            if (errorMsg) *errorMsg = "DEM 求交残差无效";
            return false;
        }
        // 记录当前最优近似点（用于未完全收敛时的返回值）
        best = {p[0], p[1], zDem};

        // 收敛判断：残差绝对值小于 1mm
        if (std::abs(diff) < 1e-3) {
            found = true;
            break;
        }

        // 牛顿步更新 t：t_new = t - diff / dir.z
        //   推导：p.z(t) = origin.z + t * dir.z，dp.z/dt = dir.z
        //   f(t) = p.z(t) - zDem，f'(t) ≈ dir.z （忽略 DEM 梯度项）
        if (std::abs(dir[2]) < 1e-12) break;
        t -= diff / dir[2];
        if (t <= 0.0 || !std::isfinite(t))
        {
            if (errorMsg)
            {
                *errorMsg = "DEM 迭代交点在相机后方或距离无效";
            }
            return false;
        }
    }

    if (!found)
    {
        if (errorMsg)
        {
            *errorMsg = "DEM 求交未收敛";
        }
        return false;
    }

    if (!finitePoint(best))
    {
        if (errorMsg) *errorMsg = "DEM 交点坐标无效";
        return false;
    }
    *ground = best;
    return true;
}

// ============================================================
// 函数：GroundBackProjector::backProjectToSphere
// 功能：射线与基准球面求交（解析二次方程）。
// ============================================================
bool GroundBackProjector::backProjectToSphere(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                                              double u,
                                              double v,
                                              const ReferenceSphereSurface& sphere,
                                              std::array<double, 3>* ground,
                                              std::string* errorMsg)
{
    if (!ground)
    {
        return false;
    }
    if (!std::isfinite(sphere.radiusMeters) || sphere.radiusMeters <= 0.0 ||
        !std::isfinite(sphere.center[0]) || !std::isfinite(sphere.center[1]) ||
        !std::isfinite(sphere.center[2]))
    {
        if (errorMsg)
        {
            *errorMsg = "基准球参数无效";
        }
        return false;
    }

    std::array<double, 3> origin;
    std::array<double, 3> dir;
    if (!pixelRayWorld(camera, u, v, &origin, &dir, errorMsg))
    {
        return false;
    }

    const std::array<double, 3> oc{
        origin[0] - sphere.center[0],
        origin[1] - sphere.center[1],
        origin[2] - sphere.center[2]};

    const double b = 2.0 * (oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2]);
    const double c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] -
                     sphere.radiusMeters * sphere.radiusMeters;
    const double disc = b * b - 4.0 * c;
    if (!std::isfinite(b) || !std::isfinite(c) || !std::isfinite(disc) || disc < 0.0)
    {
        if (errorMsg)
        {
            *errorMsg = std::isfinite(disc) ? "射线与基准球无交点" : "基准球交点计算无效";
        }
        return false;
    }

    const double root = std::sqrt(std::max(0.0, disc));
    const double t0 = (-b - root) * 0.5;
    const double t1 = (-b + root) * 0.5;
    if (!std::isfinite(t0) || !std::isfinite(t1))
    {
        if (errorMsg) *errorMsg = "基准球交点距离无效";
        return false;
    }
    double t = 0.0;
    if (t0 > 1e-9)
    {
        t = t0;
    }
    else if (t1 > 1e-9)
    {
        t = t1;
    }
    else
    {
        if (errorMsg)
        {
            *errorMsg = "基准球交点在相机后方";
        }
        return false;
    }

    const std::array<double, 3> candidate{origin[0] + t * dir[0],
                                           origin[1] + t * dir[1],
                                           origin[2] + t * dir[2]};
    if (!finitePoint(candidate))
    {
        if (errorMsg) *errorMsg = "基准球交点坐标无效";
        return false;
    }
    *ground = candidate;
    return true;
}

// ============================================================
// 函数：GroundBackProjector::imageCenterToGround
// 功能：将影像中心像素反投影为地面坐标，作为影像的地面中心点。
//   像素坐标选取策略：
//     - imageWidth/imageHeight > 0 时：使用影像几何中心（w/2, h/2）
//     - 调用方应在进入重叠分析前提供有效影像尺寸
//   地面模型选择：useFixedZ=true 用固定高程面，否则用 DEM（需有效）
// ============================================================
bool GroundBackProjector::imageCenterToGround(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                                              int imageWidth,
                                              int imageHeight,
                                              const DemSurface* dem,
                                              bool useFixedZ,
                                              double fixedZ,
                                              std::array<double, 3>* ground,
                                              std::string* errorMsg)
{
    if (imageWidth <= 0 || imageHeight <= 0)
    {
        if (errorMsg)
        {
            *errorMsg = "影像尺寸必须为正";
        }
        return false;
    }

    const double u = 0.5 * double(imageWidth);
    const double v = 0.5 * double(imageHeight);

    // 根据模式选择反投影方法
    if (useFixedZ)
    {
        return backProjectToFixedZ(camera, u, v, fixedZ, ground, errorMsg);
    }
    if (!dem)
    {
        if (errorMsg)
        {
            *errorMsg = "DEM 模式需要有效的 DEM 曲面";
        }
        return false;
    }
    return backProjectWithDem(camera, u, v, *dem, ground, errorMsg);
}

bool GroundBackProjector::imageCenterToSphere(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                                              int imageWidth,
                                              int imageHeight,
                                              const ReferenceSphereSurface& sphere,
                                              std::array<double, 3>* ground,
                                              std::string* errorMsg)
{
    if (imageWidth <= 0 || imageHeight <= 0)
    {
        if (errorMsg)
        {
            *errorMsg = "影像尺寸必须为正";
        }
        return false;
    }

    const double u = 0.5 * double(imageWidth);
    const double v = 0.5 * double(imageHeight);
    return backProjectToSphere(camera, u, v, sphere, ground, errorMsg);
}

// ============================================================
// 函数：GroundBackProjector::estimateFootprintRadius
// 功能：估算影像地面覆盖区域的等效半径（单位与坐标系一致）。
//   方法：
//     1) 将影像中心点反投影到地面，得到 center
//     2) 将影像四角 {(0,0), (w,0), (w,h), (0,h)} 分别反投影到地面
//     3) 计算四角到中心的水平距离（XY平面距离），取平均值为等效半径
//   用途：此半径用于重叠度分析中邻域搜索的初始半径估计。
//         搜索半径 = neighborFactor * radius * 2.5（见 OverlapAnalyzer）
// ============================================================
bool GroundBackProjector::estimateFootprintRadius(
    const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
    int imageWidth,
    int imageHeight,
    const DemSurface* dem,
    bool useFixedZ,
    double fixedZ,
    double* radius,
    std::string* errorMsg)
{
    if (!radius)
    {
        return false;
    }
    if (imageWidth <= 0 || imageHeight <= 0)
    {
        if (errorMsg)
        {
            *errorMsg = "影像尺寸必须为正";
        }
        return false;
    }

    // Step 1：获取影像地面中心点
    std::array<double, 3> center;
    if (!imageCenterToGround(camera, imageWidth, imageHeight, dem, useFixedZ, fixedZ, &center, errorMsg))
    {
        return false;
    }

    const double w = double(imageWidth);
    const double h = double(imageHeight);

    // 四角像素坐标（左上、右上、右下、左下）
    std::array<double, 3> corners[4];
    const std::pair<double, double> uv[4] = {{0.0, 0.0}, {w, 0.0}, {w, h}, {0.0, h}};

    // Step 2：将四角反投影到地面，累加与中心的水平距离
    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < 4; ++i) {
        std::string tmpErr;
        // 选择对应的反投影方法
        const bool ok = useFixedZ
                            ? backProjectToFixedZ(camera, uv[i].first, uv[i].second, fixedZ, &corners[i], &tmpErr)
                            : (dem ? backProjectWithDem(camera, uv[i].first, uv[i].second, *dem, &corners[i], &tmpErr)
                                   : false);
        if (!ok)
        {
            if (errorMsg)
            {
                *errorMsg = "影像角点反投影失败: corner=" + std::to_string(i) +
                             (tmpErr.empty() ? std::string() : " | " + tmpErr);
            }
            return false;
        }
        const double distance = distance2D(center, corners[i]);
        if (!std::isfinite(distance))
        {
            if (errorMsg)
            {
                *errorMsg = "影像角点距离无效: corner=" + std::to_string(i);
            }
            return false;
        }
        sum += distance; // 累加水平距离
        if (!std::isfinite(sum))
        {
            if (errorMsg) *errorMsg = "影像覆盖半径累计值无效";
            return false;
        }
        ++valid;
    }

    if (valid != 4)
    {
        if (errorMsg)
        {
            *errorMsg = "无法估计影像地面覆盖半径：四个角点必须全部有效";
        }
        return false;
    }

    // Step 3：平均距离作为等效半径
    *radius = sum / double(valid);
    if (!std::isfinite(*radius) || *radius <= 0.0)
    {
        if (errorMsg) *errorMsg = "影像覆盖半径无效";
        return false;
    }
    return true;
}

bool GroundBackProjector::estimateFootprintRadiusOnSphere(
    const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
    int imageWidth,
    int imageHeight,
    const ReferenceSphereSurface& sphere,
    double* radius,
    std::string* errorMsg)
{
    if (!radius)
    {
        return false;
    }
    if (imageWidth <= 0 || imageHeight <= 0)
    {
        if (errorMsg)
        {
            *errorMsg = "影像尺寸必须为正";
        }
        return false;
    }

    std::array<double, 3> center;
    if (!imageCenterToSphere(camera, imageWidth, imageHeight, sphere, &center, errorMsg))
    {
        return false;
    }

    const double w = double(imageWidth);
    const double h = double(imageHeight);
    const std::pair<double, double> uv[4] = {{0.0, 0.0}, {w, 0.0}, {w, h}, {0.0, h}};

    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < 4; ++i)
    {
        std::array<double, 3> corner;
        std::string tmpErr;
        if (!backProjectToSphere(camera, uv[i].first, uv[i].second, sphere, &corner, &tmpErr))
        {
            if (errorMsg)
            {
                *errorMsg = "影像角点球面反投影失败: corner=" + std::to_string(i) +
                             (tmpErr.empty() ? std::string() : " | " + tmpErr);
            }
            return false;
        }
        const double distance = distance3D(center, corner);
        if (!std::isfinite(distance))
        {
            if (errorMsg)
            {
                *errorMsg = "基准球角点距离无效: corner=" + std::to_string(i);
            }
            return false;
        }
        sum += distance;
        if (!std::isfinite(sum))
        {
            if (errorMsg) *errorMsg = "基准球覆盖半径累计值无效";
            return false;
        }
        ++valid;
    }

    if (valid != 4)
    {
        if (errorMsg)
        {
            *errorMsg = "无法估计基准球面影像覆盖半径：四个角点必须全部有效";
        }
        return false;
    }

    *radius = sum / double(valid);
    if (!std::isfinite(*radius) || *radius <= 0.0)
    {
        if (errorMsg) *errorMsg = "基准球覆盖半径无效";
        return false;
    }
    return true;
}

} // namespace xjw
