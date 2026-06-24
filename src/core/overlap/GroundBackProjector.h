#pragma once

// ============================================================
// 文件：GroundBackProjector.h
// 功能：将影像像素坐标反投影为地面三维坐标（反向投影 / Back-Projection）。
//       支持三种地面模型：
//         1) 固定高程面（Flat Earth / Plane at fixed Z）
//         2) 数字高程模型（DEM，通过迭代射线-曲面求交实现）
//         3) 基准球面（Reference sphere，用于行星/局部切平面近似）
//       另提供 DemSurface 类，用于从 XYZ 文本文件加载 DEM 点云并快速查询最近邻高程。
// 典型应用：影像地面覆盖范围估算、重叠度分析、相机姿态验证。
// ============================================================

#include "Camera.h"
#include <plapoint/search/spatial_kdtree.h>

#include <array>
#include <string>
#include <vector>

namespace xjw {

// ============================================================
// 结构体：ReferenceSphereSurface
// 描述：射线-基准球面求交所需的几何参数。
//   center       - 球心坐标；行星中心坐标通常为 (0,0,0)，本地坐标可放在局部切面下方
//   radiusMeters - 球半径，单位与相机坐标一致（项目中通常为米）
// ============================================================
struct ReferenceSphereSurface
{
    std::array<double, 3> center{{0.0, 0.0, 0.0}};
    double radiusMeters = 0.0;
};

// ============================================================
// 类：DemSurface
// 描述：数字高程模型（DEM）的轻量封装。
//   - 从 XYZ 格式点云文件中加载地面高程数据
//   - 内部使用 PlaPoint SpatialKdTree 建立空间索引以支持快速最近邻高程查询
//   - 提供均值高程作为缺省/初始值使用
// 坐标系说明：XY 为水平坐标（如 UTM 东北向），Z 为高程（通常为 CGCS2000 或 WGS84 椭球高）
// ============================================================
class DemSurface
{
public:
    // 从 XYZ 文本文件加载 DEM 点云；每行格式："x y z"，# 开头为注释行
    // 成功返回 true，失败将错误信息写入 errorMsg（若非 nullptr）
    bool loadFromXYZ(const std::string &path, std::string *errorMsg = nullptr);

    // 在 (x,y) 水平坐标处查询最近邻高程 z
    // xyDistance（可选）返回查询点到最近 DEM 点的水平距离，用于评估插值质量
    // 若未加载数据则返回 false
    bool sampleHeight(double x, double y, double *z, double *xyDistance = nullptr) const;

    // 判断 DEM 是否已成功加载（点云非空）
    bool valid() const;

    // 返回加载点云的平均高程（Z 均值），可用作初始射线行进深度估计
    double meanHeight() const;

private:
    // 原始 XYZ 三维点集合
    std::vector<std::array<double, 3>> _points;

    using DemKdTree2D = plapoint::search::SpatialKdTree<2, double>;

    // 对应 2D 平面点（仅含 XY 坐标+索引），用于构建 KD 树
    std::vector<DemKdTree2D::Point> _xyPoints;

    // 2D KD 树空间索引，加速水平方向最近邻查询
    DemKdTree2D _index;

    // 所有点的 Z 均值，作为 DEM 整体高程的近似估计
    double _meanHeight = 0.0;
};

// ============================================================
// 类：GroundBackProjector
// 描述：提供影像像素坐标 → 地面三维坐标的反投影功能（静态工具类）。
//   反投影原理：
//     1) 根据相机内参将像素坐标转换为相机坐标系下的归一化射线方向
//     2) 使用相机外参（旋转矩阵 R）将射线变换到世界坐标系
//     3) 从相机中心沿射线方向行进，与地面模型（水平面或 DEM）求交
//   所有方法均为静态方法，无需实例化。
// ============================================================
class GroundBackProjector
{
public:
    // --------------------------------------------------------
    // 函数：backProjectToFixedZ
    // 功能：将像素坐标 (u,v) 沿射线方向与固定高程平面 Z=fixedZ 求交。
    //   数学推导：
    //     射线方程：P(t) = C + t * dir
    //     令 P_z = fixedZ：t = (fixedZ - C.z) / dir.z
    //     交点：ground = C + t * dir，其中 z 分量强制为 fixedZ
    // 参数：
    //   camera   - 相机标定参数
    //   u, v     - 像素坐标（列、行）
    //   fixedZ   - 固定高程值（世界坐标系 Z，如平均地面高程）
    //   ground   - 输出三维交点坐标
    //   errorMsg - 可选错误信息输出
    // 返回值：成功 true，失败 false（射线平行于高程面或交点在相机后方）
    // --------------------------------------------------------
    static bool backProjectToFixedZ(const Camera &camera,
                                    double u,
                                    double v,
                                    double fixedZ,
                                    std::array<double, 3> *ground,
                                    std::string *errorMsg = nullptr);

    // --------------------------------------------------------
    // 函数：backProjectWithDem
    // 功能：将像素坐标 (u,v) 沿射线方向与 DEM 曲面迭代求交（类牛顿迭代）。
    //   方法：以 DEM 均值高程作为初始 t，然后迭代更新：
    //         t_new = t - (ray_z(t) - dem_z(x,y)) / dir.z
    //         最多 32 次迭代，收敛条件：|ray_z - dem_z| < 1e-3
    // 参数：
    //   dem      - 已加载的 DEM 曲面对象
    //   ground   - 输出三维交点坐标（Z 取 DEM 高程值）
    // 返回值：迭代收敛或未完全收敛时均返回 true（附近似结果），加载失败返回 false
    // --------------------------------------------------------
    static bool backProjectWithDem(const Camera &camera,
                                   double u,
                                   double v,
                                   const DemSurface &dem,
                                   std::array<double, 3> *ground,
                                   std::string *errorMsg = nullptr);

    // --------------------------------------------------------
    // 函数：backProjectToSphere
    // 功能：将像素坐标 (u,v) 沿射线方向与基准球面求交。
    //   射线方程 P(t)=C+t*dir，球面方程 |P-center|=radius。
    //   选择最小正根作为相机前方的近端交点。
    // --------------------------------------------------------
    static bool backProjectToSphere(const Camera &camera,
                                    double u,
                                    double v,
                                    const ReferenceSphereSurface &sphere,
                                    std::array<double, 3> *ground,
                                    std::string *errorMsg = nullptr);

    // --------------------------------------------------------
    // 函数：imageCenterToGround
    // 功能：将影像中心像素反投影到地面，得到影像的地面投影中心点。
    //   若 imageWidth/imageHeight > 0 则使用影像中心坐标（w/2, h/2），
    //   否则使用相机主点坐标（cu, cv）。
    //   根据 useFixedZ 标志选择固定高程面模式或 DEM 模式。
    // 参数：
    //   imageWidth/Height - 影像分辨率（像素），用于计算中心像素坐标
    //   dem               - DEM 指针，可为 nullptr（useFixedZ=true 时可不提供）
    //   useFixedZ         - true = 固定高程面，false = DEM
    //   fixedZ            - 固定高程值（useFixedZ=true 时有效）
    // --------------------------------------------------------
    static bool imageCenterToGround(const Camera &camera,
                                    int imageWidth,
                                    int imageHeight,
                                    const DemSurface *dem,
                                    bool useFixedZ,
                                    double fixedZ,
                                    std::array<double, 3> *ground,
                                    std::string *errorMsg = nullptr);

    // --------------------------------------------------------
    // 函数：imageCenterToSphere
    // 功能：将影像中心像素反投影到基准球面。
    // --------------------------------------------------------
    static bool imageCenterToSphere(const Camera &camera,
                                    int imageWidth,
                                    int imageHeight,
                                    const ReferenceSphereSurface &sphere,
                                    std::array<double, 3> *ground,
                                    std::string *errorMsg = nullptr);

    // --------------------------------------------------------
    // 函数：estimateFootprintRadius
    // 功能：估计影像地面覆盖区域的近似"等效半径"（单位与坐标系一致，如米）。
    //   方法：将影像四角 (0,0)、(w,0)、(w,h)、(0,h) 反投影到地面，
    //         计算四角到地面中心点的平均水平距离作为等效半径。
    //   该半径用于重叠度分析中的邻域搜索半径初始化。
    // 参数：
    //   radius - 输出：估算的地面覆盖等效半径
    // --------------------------------------------------------
    static bool estimateFootprintRadius(const Camera &camera,
                                        int imageWidth,
                                        int imageHeight,
                                        const DemSurface *dem,
                                        bool useFixedZ,
                                        double fixedZ,
                                        double *radius,
                                        std::string *errorMsg = nullptr);

    // --------------------------------------------------------
    // 函数：estimateFootprintRadiusOnSphere
    // 功能：估计影像在基准球面上的近似覆盖半径。
    // --------------------------------------------------------
    static bool estimateFootprintRadiusOnSphere(const Camera &camera,
                                                int imageWidth,
                                                int imageHeight,
                                                const ReferenceSphereSurface &sphere,
                                                double *radius,
                                                std::string *errorMsg = nullptr);

private:
    // --------------------------------------------------------
    // 函数：pixelRayWorld（私有工具方法）
    // 功能：将像素坐标 (u,v) 转换为世界坐标系下的射线（起点 + 归一化方向）。
    //   步骤：
    //     1) 反内参：x = (u - cu) / (uDir * fu)，y = (v - cv) / (vDir * fv)
    //     2) 相机系射线方向：ray_cam = (x, y, 1)
    //     3) 旋转到世界系：ray_world = R * ray_cam
    //     4) 归一化：dir = ray_world / ||ray_world||
    // 参数：
    //   origin - 输出射线起点（= 相机中心 C）
    //   dir    - 输出射线归一化方向向量（世界坐标系）
    // --------------------------------------------------------
    static bool pixelRayWorld(const Camera &camera,
                              double u,
                              double v,
                              std::array<double, 3> *origin,
                              std::array<double, 3> *dir,
                              std::string *errorMsg = nullptr);
};

} // namespace xjw
