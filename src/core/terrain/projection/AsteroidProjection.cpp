#include "AsteroidProjection.h"

#include "data/PointCloud.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace xjw
{

// =============================================================================
// 公共辅助函数
// =============================================================================

namespace
{

constexpr double kPi = 3.14159265358979323846;

inline double clampd(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace

// =============================================================================
// AsteroidProjection 实现
// =============================================================================

const char *asteroidProjectionName(AsteroidProjectionType type)
{
    switch (type)
    {
    case AsteroidProjectionType::Stereographic:
        return "stere";
    case AsteroidProjectionType::AzimuthalEquidistant:
        return "aeqd";
    case AsteroidProjectionType::TriaxialEllipsoid:
        return "triaxial";
    }
    return "unknown";
}

std::array<double, 9> AsteroidProjection::identityRotation()
{
    return {1, 0, 0,
            0, 1, 0,
            0, 0, 1};
}

// -----------------------------------------------------------------------------
// computeCenter
// -----------------------------------------------------------------------------

AsteroidBodyCenter AsteroidProjection::computeCenter(const pointcloud::PointCloud &pc)
{
    AsteroidBodyCenter result;
    if (pc.empty())
    {
        return result;
    }

    const auto &positions = pc.positions();
    const std::size_t n = positions.size();

    double sumX = 0, sumY = 0, sumZ = 0;
    for (const auto &p : positions)
    {
        sumX += static_cast<double>(p.x);
        sumY += static_cast<double>(p.y);
        sumZ += static_cast<double>(p.z);
    }
    result.cx = sumX / static_cast<double>(n);
    result.cy = sumY / static_cast<double>(n);
    result.cz = sumZ / static_cast<double>(n);

    double sumDist = 0.0;
    for (const auto &p : positions)
    {
        const double dx = static_cast<double>(p.x) - result.cx;
        const double dy = static_cast<double>(p.y) - result.cy;
        const double dz = static_cast<double>(p.z) - result.cz;
        sumDist += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    result.referenceRadius = sumDist / static_cast<double>(n);

    return result;
}

// -----------------------------------------------------------------------------
// fitEllipsoid  (PCA-based)
// -----------------------------------------------------------------------------

TriaxialEllipsoidParams AsteroidProjection::fitEllipsoid(
    const pointcloud::PointCloud &pc,
    const AsteroidBodyCenter &center,
    std::array<double, 9> *rotationMatrix)
{
    TriaxialEllipsoidParams result;

    if (rotationMatrix)
    {
        *rotationMatrix = identityRotation();
    }

    if (pc.size() < 4)
    {
        result.a = result.b = result.c = center.referenceRadius;
        return result;
    }

    const auto &positions = pc.positions();
    const int n = static_cast<int>(positions.size());

    // 构建以质心为原点的去中心化坐标矩阵 (n × 3)
    cv::Mat data(n, 3, CV_64F);
    for (int i = 0; i < n; ++i)
    {
        data.at<double>(i, 0) = static_cast<double>(positions[static_cast<std::size_t>(i)].x) - center.cx;
        data.at<double>(i, 1) = static_cast<double>(positions[static_cast<std::size_t>(i)].y) - center.cy;
        data.at<double>(i, 2) = static_cast<double>(positions[static_cast<std::size_t>(i)].z) - center.cz;
    }

    // PCA：特征值按降序排列（方差最大轴 → 最长半轴 a）
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 3);

    // 特征向量（行 i 是第 i 主轴单位向量）
    const cv::Mat &eigVecs = pca.eigenvectors;   // shape: 3×3, CV_64F
    const cv::Mat &eigVals = pca.eigenvalues;    // shape: 3×1, CV_64F

    // 将数据投影到 PCA 主轴坐标系
    cv::Mat dataPca = data * eigVecs.t();  // n×3 in PCA frame

    // 对每个投影点计算归一化椭球半径坐标
    // s(i) = sqrt( (x'/sqrt(λ0))² + (y'/sqrt(λ1))² + (z'/sqrt(λ2))² )
    // 对于真实椭球体表面的均匀分布点，s 的统计分布的某个百分位 k 对应椭球
    // 表面 —— 用该百分位调整半轴绝对值，使椭球面通过约 80% 的点群
    const double sqrtLam0 = std::sqrt(std::max(1e-9, eigVals.at<double>(0, 0)));
    const double sqrtLam1 = std::sqrt(std::max(1e-9, eigVals.at<double>(1, 0)));
    const double sqrtLam2 = std::sqrt(std::max(1e-9, eigVals.at<double>(2, 0)));

    std::vector<double> rho(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double xn = dataPca.at<double>(i, 0) / sqrtLam0;
        const double yn = dataPca.at<double>(i, 1) / sqrtLam1;
        const double zn = dataPca.at<double>(i, 2) / sqrtLam2;
        rho[static_cast<std::size_t>(i)] = std::sqrt(xn * xn + yn * yn + zn * zn);
    }

    // 使用第 80 百分位作为椭球"表面"定标因子（约 80% 的点在椭球内部或表面）
    std::sort(rho.begin(), rho.end());
    const double kPercentile = rho[static_cast<std::size_t>(n) * 4 / 5];  // 80th percentile
    const double scale = (kPercentile > 1e-9) ? (1.0 / kPercentile) : 1.0;

    result.a = scale * sqrtLam0;
    result.b = scale * sqrtLam1;
    result.c = scale * sqrtLam2;

    // 保证 a ≥ b ≥ c（PCA 按降序，通常已满足，但以防万一）
    if (result.a < result.b) std::swap(result.a, result.b);
    if (result.b < result.c) std::swap(result.b, result.c);
    if (result.a < result.b) std::swap(result.a, result.b);

    // 写出旋转矩阵（行主序，行 i 为主轴 i 的单位向量）
    if (rotationMatrix)
    {
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                (*rotationMatrix)[static_cast<std::size_t>(row * 3 + col)] =
                    eigVecs.at<double>(row, col);
            }
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// projectPoint
// -----------------------------------------------------------------------------

void AsteroidProjection::projectPoint(
    double x, double y, double z,
    const AsteroidBodyCenter &center,
    const AsteroidProjectionParams &params,
    const std::array<double, 9> &rotMatrix,
    double *u, double *v, double *elevation)
{
    const double dx = x - center.cx;
    const double dy = y - center.cy;
    const double dz = z - center.cz;
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

    const double R = (params.referenceRadius > 0.0)
                         ? params.referenceRadius
                         : center.referenceRadius;

    if (u)       *u = 0.0;
    if (v)       *v = 0.0;
    if (elevation) *elevation = r - R;

    if (r < 1e-12)
    {
        return;
    }

    switch (params.type)
    {
    // -----------------------------------------------------------------------
    // 极射赤平投影（Stereographic）
    // 以 +Z 为北极切点平面；从南极 (0,0,-1) 投影到北极邻域平面
    // -----------------------------------------------------------------------
    case AsteroidProjectionType::Stereographic:
    {
        const double xn = dx / r;
        const double yn = dy / r;
        const double zn = dz / r;

        // 南极检查：zn ≈ -1 时分母趋零，裁剪到大但有限的值
        const double denom = 1.0 + zn;
        if (std::abs(denom) < 1e-9)
        {
            // 南极点映射到极远处，给出一个有限大值
            const double sign = (dx >= 0.0) ? 1.0 : -1.0;
            if (u)       *u = sign * R * 1e6;
            if (v)       *v = 0.0;
            if (elevation) *elevation = r - R;
            return;
        }

        const double k = 2.0 / denom;
        if (u)       *u = k * xn * R;
        if (v)       *v = k * yn * R;
        if (elevation) *elevation = r - R;
        break;
    }

    // -----------------------------------------------------------------------
    // 方位等距投影（Azimuthal Equidistant）
    // 以 +Z 为中心，保持角距离与方位角
    // -----------------------------------------------------------------------
    case AsteroidProjectionType::AzimuthalEquidistant:
    {
        const double xn = dx / r;
        const double yn = dy / r;
        const double zn = dz / r;

        const double c = std::acos(clampd(zn, -1.0, 1.0));  // 与北极的角距
        const double theta = std::atan2(yn, xn);             // 方位角

        const double rho = R * c;
        if (u)       *u = rho * std::cos(theta);
        if (v)       *v = rho * std::sin(theta);
        if (elevation) *elevation = r - R;
        break;
    }

    // -----------------------------------------------------------------------
    // 三轴椭球方位等距投影（Triaxial Ellipsoid）
    // 1. PCA 旋转到椭球主轴系
    // 2. 归一化到单位椭球面
    // 3. 以归一化 +Z_ell 为北极做方位等距投影
    // -----------------------------------------------------------------------
    case AsteroidProjectionType::TriaxialEllipsoid:
    {
        const TriaxialEllipsoidParams &ell = params.ellipsoid;
        const double a = std::max(ell.a, 1e-12);
        const double b = std::max(ell.b, 1e-12);
        const double c = std::max(ell.c, 1e-12);
        const double R_eff = (a + b + c) / 3.0;

        // 旋转到 PCA 主轴系（rotMatrix 为行主序，行 i 是主轴 i 方向）
        const double dpx = rotMatrix[0] * dx + rotMatrix[1] * dy + rotMatrix[2] * dz;
        const double dpy = rotMatrix[3] * dx + rotMatrix[4] * dy + rotMatrix[5] * dz;
        const double dpz = rotMatrix[6] * dx + rotMatrix[7] * dy + rotMatrix[8] * dz;

        // 归一化到椭球：xe² + ye² + ze² ≠ 1，但方向给出"椭球面点"
        const double xe = dpx / a;
        const double ye = dpy / b;
        const double ze = dpz / c;

        // 归一化椭球坐标的径向距离（=1 表示在椭球面）
        const double re = std::sqrt(xe * xe + ye * ye + ze * ze);

        if (re < 1e-12)
        {
            if (elevation) *elevation = -R_eff;
            return;
        }

        // 将椭球面点映射到单位球：(xs,ys,zs) = unit(xe,ye,ze)
        const double xs = xe / re;
        const double ys = ye / re;
        const double zs = ze / re;

        // 以 +Z_ell 轴为"北极"做方位等距投影
        const double cAng = std::acos(clampd(zs, -1.0, 1.0));
        const double theta = std::atan2(ys, xs);

        const double rho = R_eff * cAng;
        if (u)       *u = rho * std::cos(theta);
        if (v)       *v = rho * std::sin(theta);
        if (elevation) *elevation = (re - 1.0) * R_eff;
        break;
    }
    }
}

// -----------------------------------------------------------------------------
// buildProjectionWkt
// -----------------------------------------------------------------------------

QString AsteroidProjection::buildProjectionWkt(
    AsteroidProjectionType type,
    const AsteroidBodyCenter &center,
    const TriaxialEllipsoidParams &ellipsoid)
{
    const double R = center.referenceRadius;
    // 半轴取到整数米精度即可；避免写入极端精度造成 WKT 过长
    const double Rf = std::round(R * 1000.0) / 1000.0;

    switch (type)
    {
    case AsteroidProjectionType::Stereographic:
        return QString(
                   "PROJCS[\"Asteroid_Stereo_NorthPole\","
                   "GEOGCS[\"GCS_Asteroid\","
                   "DATUM[\"D_Asteroid_Sphere\","
                   "SPHEROID[\"Asteroid_Sphere\",%1,0]],"
                   "PRIMEM[\"Greenwich\",0],"
                   "UNIT[\"Degree\",0.0174532925199433]],"
                   "PROJECTION[\"Stereographic\"],"
                   "PARAMETER[\"False_Easting\",0],"
                   "PARAMETER[\"False_Northing\",0],"
                   "PARAMETER[\"Central_Meridian\",0],"
                   "PARAMETER[\"Scale_Factor\",1],"
                   "PARAMETER[\"Latitude_Of_Origin\",90],"
                   "UNIT[\"Metre\",1]]")
            .arg(Rf, 0, 'f', 3);

    case AsteroidProjectionType::AzimuthalEquidistant:
        return QString(
                   "PROJCS[\"Asteroid_AziEquidistant_NorthPole\","
                   "GEOGCS[\"GCS_Asteroid\","
                   "DATUM[\"D_Asteroid_Sphere\","
                   "SPHEROID[\"Asteroid_Sphere\",%1,0]],"
                   "PRIMEM[\"Greenwich\",0],"
                   "UNIT[\"Degree\",0.0174532925199433]],"
                   "PROJECTION[\"Azimuthal_Equidistant\"],"
                   "PARAMETER[\"False_Easting\",0],"
                   "PARAMETER[\"False_Northing\",0],"
                   "PARAMETER[\"Latitude_Of_Center\",90],"
                   "PARAMETER[\"Longitude_Of_Center\",0],"
                   "UNIT[\"Metre\",1]]")
            .arg(Rf, 0, 'f', 3);

    case AsteroidProjectionType::TriaxialEllipsoid:
    {
        // GDAL 无标准三轴椭球投影定义，使用 LOCAL_CS 携带自定义元数据
        const double af = std::round(ellipsoid.a * 1000.0) / 1000.0;
        const double bf = std::round(ellipsoid.b * 1000.0) / 1000.0;
        const double cf = std::round(ellipsoid.c * 1000.0) / 1000.0;
        return QString(
                   "LOCAL_CS[\"Asteroid_Triaxial_AziEquidistant\","
                   "LOCAL_DATUM[\"Asteroid_Triaxial_Body\",32767],"
                   "UNIT[\"Metre\",1],"
                   "AXIS[\"X\",EAST],"
                   "AXIS[\"Y\",NORTH],"
                   "AUTHORITY[\"CUSTOM\",\"Triaxial_a=%1_b=%2_c=%3\"]]")
            .arg(af, 0, 'f', 3)
            .arg(bf, 0, 'f', 3)
            .arg(cf, 0, 'f', 3);
    }
    }

    return {};
}

} // namespace xjw
