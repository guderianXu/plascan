#pragma once

// =============================================================================
// 文件: AsteroidProjection.h
// 功能: 小行星地形投影——将三维笛卡尔坐标转换为若干适合小天体制图的平面坐标，
//       并生成带地理参考元数据的 GeoTIFF DEM/DOM 产品。
//
// 支持三种投影：
//   1. 极射赤平投影  (Stereographic)        ——保角，适合极区观测
//   2. 方位等距投影  (Azimuthal Equidistant) ——保距，适合全球制图
//   3. 三轴椭球投影  (Triaxial Ellipsoid)    ——先 PCA 拟合不规则体形状，
//                                              再在归一化椭球单位球面上做方位等距投影
// =============================================================================

#include <QString>
#include <array>

#include <plamatrix/core/types.h>

// 前向声明，避免将 PointCloud 的重型依赖传播到整个地形模块头文件
namespace plapoint
{
template <typename Scalar, plamatrix::Device Dev>
class PointCloud;
}

namespace xjw
{

// =============================================================================
// 投影类型枚举
// =============================================================================

/**
 * @brief 小天体地形投影类型。
 */
enum class AsteroidProjectionType
{
    Stereographic,        ///< 极射赤平投影（从南极切线平面投影到北极切线平面）
    AzimuthalEquidistant, ///< 方位等距投影（北极为中心保持角距）
    TriaxialEllipsoid,    ///< 三轴椭球方位等距投影（PCA 拟合体形 + 方位等距）
};

/** @brief 返回投影类型的简短英文名称，用于文件命名。 */
const char *asteroidProjectionName(AsteroidProjectionType type);

// =============================================================================
// 投影参数结构
// =============================================================================

/**
 * @brief 三轴椭球的半轴长度（a ≥ b ≥ c）。
 *
 * - a：最长半轴（主轴，PCA 第一分量方向）
 * - b：中间半轴
 * - c：最短半轴（极轴方向）
 */
struct TriaxialEllipsoidParams
{
    double a = 1.0;
    double b = 1.0;
    double c = 1.0;
};

/**
 * @brief 小行星投影参数。
 */
struct AsteroidProjectionParams
{
    AsteroidProjectionType type = AsteroidProjectionType::Stereographic;

    /** 参考球半径（用于 Stere/AEQD 投影）；0 = 使用 AsteroidBodyCenter::referenceRadius 自动估算 */
    double referenceRadius = 0.0;

    /** 对三轴椭球投影是否在运行时自动用 PCA 拟合半轴 */
    bool autoFitEllipsoid = true;

    /** 主动指定或 fitEllipsoid() 填写的三轴参数 */
    TriaxialEllipsoidParams ellipsoid;
};

/**
 * @brief 由点云估算的小天体质心与参考半径。
 */
struct AsteroidBodyCenter
{
    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    double referenceRadius = 1.0; ///< 点云各点到质心距离的均值
};

// =============================================================================
// AsteroidProjection 类
// =============================================================================

/**
 * @brief 小行星地形投影转换器（纯静态工具类）。
 *
 * 典型调用流程：
 * @code
 *   auto center  = AsteroidProjection::computeCenter(pc);
 *   std::array<double,9> rot{};
 *   TriaxialEllipsoidParams ell = AsteroidProjection::fitEllipsoid(pc, center, &rot);
 *
 *   AsteroidProjectionParams params;
 *   params.type = AsteroidProjectionType::TriaxialEllipsoid;
 *   params.ellipsoid = ell;
 *
 *   double u, v, elev;
 *   AsteroidProjection::projectPoint(x, y, z, center, params, rot, &u, &v, &elev);
 * @endcode
 */
class AsteroidProjection
{
public:
    /**
     * @brief 计算点云的质心坐标和参考半径（各点到质心的均值距离）。
     */
    static AsteroidBodyCenter computeCenter(const plapoint::PointCloud<float, plamatrix::Device::CPU> &pc);

    /**
     * @brief 通过 PCA 拟合三轴椭球体，返回半轴参数并可选地输出旋转矩阵。
     *
     * @param[out] rotationMatrix  3×3 行主序旋转矩阵（每行为一个主轴方向单位向量）。
     *                             从体固系 (dx,dy,dz) 变换到 PCA 主轴系 (dx',dy',dz') 的方式为：
     *                               [dx',dy',dz'] = rotationMatrix * [dx,dy,dz]^T
     *                             第 0 行对应最长半轴 a，第 2 行对应最短半轴 c。
     * @return 拟合的三轴椭球半轴，a ≥ b ≥ c。
     */
    static TriaxialEllipsoidParams fitEllipsoid(
        const plapoint::PointCloud<float, plamatrix::Device::CPU> &pc,
        const AsteroidBodyCenter &center,
        std::array<double, 9> *rotationMatrix = nullptr);

    /**
     * @brief 将单个三维笛卡尔点投影到指定坐标系。
     *
     * @param x,y,z         点的体固系笛卡尔坐标
     * @param center        质心与参考半径
     * @param params        投影参数（类型、参考半径覆盖等）
     * @param rotMatrix     三轴椭球投影使用的 3×3 旋转矩阵；其他投影传入单位矩阵
     * @param[out] u        投影后 X 坐标（米）
     * @param[out] v        投影后 Y 坐标（米）
     * @param[out] elevation 相对于参考球/椭球面的高程（米）；正值表示在表面之上
     */
    static void projectPoint(
        double x, double y, double z,
        const AsteroidBodyCenter &center,
        const AsteroidProjectionParams &params,
        const std::array<double, 9> &rotMatrix,
        double *u, double *v, double *elevation);

    /**
     * @brief 构建嵌入 GeoTIFF 的 OGC WKT 投影字符串。
     *
     * Stereographic/AzimuthalEquidistant 使用标准 PROJCS 格式；
     * TriaxialEllipsoid 使用 LOCAL_CS（GDAL 通用回退）。
     */
    static QString buildProjectionWkt(
        AsteroidProjectionType type,
        const AsteroidBodyCenter &center,
        const TriaxialEllipsoidParams &ellipsoid);

    /** @brief 返回恒等旋转矩阵（供 Stere/AEQD 调用方传入 rotMatrix）。 */
    static std::array<double, 9> identityRotation();
};

} // namespace xjw
