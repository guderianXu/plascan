#include "GroundBackProjector.h"

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
//   - 相机坐标系：X 向右，Y 向下，Z 向前（光轴方向）
//   - 世界坐标系：由相机标定确定，通常为 UTM 或本地水平坐标系
//   - 高程 Z：向上为正
// ============================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace {

// 计算三维向量的欧氏范数（L2 范数）：||v|| = sqrt(x²+y²+z²)
double norm3(const std::array<double, 3> &v)
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

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
    m_points.clear();
    m_xyPoints.clear();
    m_meanHeight = 0.0;

    std::ifstream ifs(path);
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
        if (!(iss >> x >> y >> z)) continue; // 解析失败则跳过该行
        m_points.push_back({x, y, z});
    }

    if (m_points.empty()) {
        if (errorMsg) *errorMsg = "DEM 中未读取到有效 XYZ 点";
        return false;
    }

    // 构建 KD 树所需的 2D 点集（只需 x, y 坐标），同时累加 Z 值以计算均值
    m_xyPoints.reserve(m_points.size());
    double sumZ = 0.0;
    for (size_t i = 0; i < m_points.size(); ++i) {
        // 将 3D 点压缩为 2D 点（附原始下标），用于 KD 树水平查询
        m_xyPoints.push_back(DemKdTree2D::Point{{m_points[i][0], m_points[i][1]}, static_cast<int>(i)});
        sumZ += m_points[i][2];
    }

    // 建立 KD 树空间索引，后续 sampleHeight 调用依赖此索引加速
    m_index.build(m_xyPoints);

    // 计算平均高程，用作射线行进初始深度估计（避免迭代从零开始）
    m_meanHeight = sumZ / static_cast<double>(m_points.size());
    return true;
}

// 在 (x,y) 处查询最近邻高程：通过 KD 树找到平面最近点，返回其 Z 值
// idx 是 PlaPoint KDTree 点的 payload，即 m_points 中的原始下标
bool DemSurface::sampleHeight(double x, double y, double *z, double *xyDistance) const
{
    if (!z || m_index.empty()) return false;
    double dist = 0.0;
    // KD 树最近邻查询：返回 PlaPoint KDTree 点的 payload（即 m_points 的下标），dist 为水平距离
    const int idx = m_index.nearest(DemKdTree2D::CoordinateArray{x, y}, &dist);
    if (idx < 0 || idx >= static_cast<int>(m_points.size())) return false;
    *z = m_points[static_cast<size_t>(idx)][2]; // 取对应点的高程 Z
    if (xyDistance) *xyDistance = dist;          // 可选：返回水平距离（评估外推精度）
    return true;
}

// 检查 DEM 是否已有有效数据（至少加载了一个点）
bool DemSurface::valid() const
{
    return !m_points.empty();
}

// 返回 DEM 点云的平均高程
double DemSurface::meanHeight() const
{
    return m_meanHeight;
}

// ============================================================
// 函数：GroundBackProjector::pixelRayWorld（私有）
// 功能：将像素坐标 (u,v) 转换为世界坐标系下的射线（起点 + 归一化方向）。
//   数学步骤：
//     ① 去畸变（此处假设已畸变校正或使用线性针孔模型，无显式畸变处理）
//     ② 反内参变换（像素 → 相机归一化坐标）：
//          x = (u - cu) / (uDir * fu)
//          y = (v - cv) / (vDir * fv)
//        其中 uDir/vDir 为轴方向符号（±1），处理不同图像坐标轴定义
//     ③ 构造相机系射线方向：ray_cam = (x, y, 1)
//     ④ 使用旋转矩阵 R（cam→world，行主序 3×3）变换到世界系：
//          ray_world = R * ray_cam
//     ⑤ 归一化：dir = ray_world / ||ray_world||
// ============================================================
bool GroundBackProjector::pixelRayWorld(const Camera &camera,
                                        double u,
                                        double v,
                                        std::array<double, 3> *origin,
                                        std::array<double, 3> *dir,
                                        std::string *errorMsg)
{
    if (!origin || !dir) return false;

    // 获取相机外参：中心坐标 C（世界系）和旋转矩阵 R（cam→world）
    const auto C = camera.cameraCenter();
    const auto R = camera.cameraToWorldRotation();

    // 获取相机内参
    const double fu = camera.focalX(); // X 方向焦距（像素）
    const double fv = camera.focalY(); // Y 方向焦距（像素）
    const double cu = camera.principalX(); // 主点 X（像素）
    const double cv = camera.principalY(); // 主点 Y（像素）
    const int uDir = camera.uAxisSign(); // X 轴方向符号（+1 或 -1）
    const int vDir = camera.vAxisSign(); // Y 轴方向符号（+1 或 -1）

    // 焦距有效性检查（防止除零）
    if (std::abs(fu) < 1e-12 || std::abs(fv) < 1e-12) {
        if (errorMsg) *errorMsg = "相机焦距 fu/fv 无效";
        return false;
    }

    // ② 反内参：像素坐标 → 相机归一化坐标（Z=1 平面上的点）
    const double x = (u - cu) / (double(uDir) * fu);
    const double y = (v - cv) / (double(vDir) * fv);

    // ③ 相机坐标系下的射线方向（齐次表示，Z=1）
    std::array<double, 3> rayCam{x, y, 1.0};

    // ④ 旋转到世界坐标系：ray_world = R * ray_cam
    //    R 为行主序 3×3 矩阵（cam→world），即 R[row*3+col]
    std::array<double, 3> rayWorld{
        R[0] * rayCam[0] + R[1] * rayCam[1] + R[2] * rayCam[2],
        R[3] * rayCam[0] + R[4] * rayCam[1] + R[5] * rayCam[2],
        R[6] * rayCam[0] + R[7] * rayCam[1] + R[8] * rayCam[2]};

    // ⑤ 归一化，得到单位方向向量
    const double n = norm3(rayWorld);
    if (n < 1e-12) {
        if (errorMsg) *errorMsg = "无法计算有效射线方向";
        return false;
    }

    // 射线起点为相机中心（世界坐标）
    *origin = C;
    *dir = {rayWorld[0] / n, rayWorld[1] / n, rayWorld[2] / n};
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
bool GroundBackProjector::backProjectToFixedZ(const Camera &camera,
                                              double u,
                                              double v,
                                              double fixedZ,
                                              std::array<double, 3> *ground,
                                              std::string *errorMsg)
{
    if (!ground) return false;

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

    // t ≤ 0 意味着交点在相机后方（反向延伸），物理上无意义
    if (t <= 0.0) {
        if (errorMsg) *errorMsg = "固定高程交点在相机后方";
        return false;
    }

    // 计算交点，Z 分量直接赋值为 fixedZ（避免浮点误差累积）
    *ground = {origin[0] + t * dir[0], origin[1] + t * dir[1], fixedZ};
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
//   退化处理：若 DEM 查询失败则将 t 扩大 1.3 倍继续搜索
// ============================================================
bool GroundBackProjector::backProjectWithDem(const Camera &camera,
                                             double u,
                                             double v,
                                             const DemSurface &dem,
                                             std::array<double, 3> *ground,
                                             std::string *errorMsg)
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
    double t = (zMean - origin[2]) / dir[2];
    if (t <= 0.0) t = 1.0; // 如果均值估计为负（相机在 DEM 下方），给一个默认正值

    bool found = false;
    std::array<double, 3> best = origin; // 保存每次迭代的最优近似点

    // 迭代求交（最多 32 次）
    for (int iter = 0; iter < 32; ++iter) {
        // 计算当前 t 对应的射线上的点
        const std::array<double, 3> p{origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]};

        // 查询该水平位置 (p.x, p.y) 的 DEM 高程
        double zDem = 0.0;
        if (!dem.sampleHeight(p[0], p[1], &zDem)) {
            // DEM 查询失败（可能超出覆盖范围），扩大 t 继续延射线前进
            t *= 1.3;
            continue;
        }

        // 残差：当前射线点的 Z 与 DEM 高程的差值
        const double diff = p[2] - zDem;
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
        if (t <= 0.0) t = 0.5; // 防止 t 退化为非正值
    }

    if (!found) {
        // 未完全收敛，使用最后一次近似作为输出（附警告）
        *ground = best;
        if (errorMsg) *errorMsg = "DEM 求交未完全收敛，返回近似交点";
        return true;
    }

    *ground = best;
    return true;
}

// ============================================================
// 函数：GroundBackProjector::backProjectToSphere
// 功能：射线与基准球面求交（解析二次方程）。
// ============================================================
bool GroundBackProjector::backProjectToSphere(const Camera &camera,
                                              double u,
                                              double v,
                                              const ReferenceSphereSurface &sphere,
                                              std::array<double, 3> *ground,
                                              std::string *errorMsg)
{
    if (!ground)
    {
        return false;
    }
    if (sphere.radiusMeters <= 0.0)
    {
        if (errorMsg)
        {
            *errorMsg = "基准球半径无效";
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
    if (disc < 0.0)
    {
        if (errorMsg)
        {
            *errorMsg = "射线与基准球无交点";
        }
        return false;
    }

    const double root = std::sqrt(std::max(0.0, disc));
    const double t0 = (-b - root) * 0.5;
    const double t1 = (-b + root) * 0.5;
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

    *ground = {origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]};
    return true;
}

// ============================================================
// 函数：GroundBackProjector::imageCenterToGround
// 功能：将影像中心像素反投影为地面坐标，作为影像的地面中心点。
//   像素坐标选取策略：
//     - imageWidth/imageHeight > 0 时：使用影像几何中心（w/2, h/2）
//     - 否则：使用相机主点（cu, cv）作为退化情况的备选
//   地面模型选择：useFixedZ=true 用固定高程面，否则用 DEM（需有效）
// ============================================================
bool GroundBackProjector::imageCenterToGround(const Camera &camera,
                                              int imageWidth,
                                              int imageHeight,
                                              const DemSurface *dem,
                                              bool useFixedZ,
                                              double fixedZ,
                                              std::array<double, 3> *ground,
                                              std::string *errorMsg)
{
    // 计算中心像素坐标：优先使用影像尺寸，退化时使用主点
    const double u = imageWidth > 0 ? 0.5 * double(imageWidth) : camera.principalX();
    const double v = imageHeight > 0 ? 0.5 * double(imageHeight) : camera.principalY();

    // 根据模式选择反投影方法
    if (useFixedZ || !dem)
    {
        return backProjectToFixedZ(camera, u, v, fixedZ, ground, errorMsg);
    }
    return backProjectWithDem(camera, u, v, *dem, ground, errorMsg);
}

bool GroundBackProjector::imageCenterToSphere(const Camera &camera,
                                              int imageWidth,
                                              int imageHeight,
                                              const ReferenceSphereSurface &sphere,
                                              std::array<double, 3> *ground,
                                              std::string *errorMsg)
{
    const double u = imageWidth > 0 ? 0.5 * double(imageWidth) : camera.principalX();
    const double v = imageHeight > 0 ? 0.5 * double(imageHeight) : camera.principalY();
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
bool GroundBackProjector::estimateFootprintRadius(const Camera &camera,
                                                  int imageWidth,
                                                  int imageHeight,
                                                  const DemSurface *dem,
                                                  bool useFixedZ,
                                                  double fixedZ,
                                                  double *radius,
                                                  std::string *errorMsg)
{
    if (!radius)
    {
        return false;
    }

    // Step 1：获取影像地面中心点
    std::array<double, 3> center;
    if (!imageCenterToGround(camera, imageWidth, imageHeight, dem, useFixedZ, fixedZ, &center, errorMsg))
    {
        return false;
    }

    // 计算影像尺寸：优先使用传入值，退化时使用主点近似
    const double w = imageWidth > 0 ? double(imageWidth) : camera.principalX() * 2.0;
    const double h = imageHeight > 0 ? double(imageHeight) : camera.principalY() * 2.0;

    // 四角像素坐标（左上、右上、右下、左下）
    std::array<double, 3> corners[4];
    const std::pair<double, double> uv[4] = {{0.0, 0.0}, {w, 0.0}, {w, h}, {0.0, h}};

    // Step 2：将四角反投影到地面，累加与中心的水平距离
    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < 4; ++i) {
        std::string tmpErr;
        // 选择对应的反投影方法
        const bool ok = (useFixedZ || !dem)
                            ? backProjectToFixedZ(camera, uv[i].first, uv[i].second, fixedZ, &corners[i], &tmpErr)
                            : backProjectWithDem(camera, uv[i].first, uv[i].second, *dem, &corners[i], &tmpErr);
        if (!ok) continue; // 某角点失败则跳过（如射线朝向天空）
        sum += distance2D(center, corners[i]); // 累加水平距离
        ++valid;
    }

    // 至少需要 1 个有效角点才能估算
    if (valid <= 0) {
        if (errorMsg) *errorMsg = "无法估计影像地面覆盖半径";
        return false;
    }

    // Step 3：平均距离作为等效半径
    *radius = sum / double(valid);
    return true;
}

bool GroundBackProjector::estimateFootprintRadiusOnSphere(const Camera &camera,
                                                          int imageWidth,
                                                          int imageHeight,
                                                          const ReferenceSphereSurface &sphere,
                                                          double *radius,
                                                          std::string *errorMsg)
{
    if (!radius)
    {
        return false;
    }

    std::array<double, 3> center;
    if (!imageCenterToSphere(camera, imageWidth, imageHeight, sphere, &center, errorMsg))
    {
        return false;
    }

    const double w = imageWidth > 0 ? double(imageWidth) : camera.principalX() * 2.0;
    const double h = imageHeight > 0 ? double(imageHeight) : camera.principalY() * 2.0;
    const std::pair<double, double> uv[4] = {{0.0, 0.0}, {w, 0.0}, {w, h}, {0.0, h}};

    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < 4; ++i)
    {
        std::array<double, 3> corner;
        std::string tmpErr;
        if (!backProjectToSphere(camera, uv[i].first, uv[i].second, sphere, &corner, &tmpErr))
        {
            continue;
        }
        sum += distance3D(center, corner);
        ++valid;
    }

    if (valid <= 0)
    {
        if (errorMsg)
        {
            *errorMsg = "无法估计基准球面影像覆盖半径";
        }
        return false;
    }

    *radius = sum / double(valid);
    return true;
}

} // namespace xjw
