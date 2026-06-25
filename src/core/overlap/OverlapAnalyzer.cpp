#include "OverlapAnalyzer.h"

// ============================================================
// 文件：OverlapAnalyzer.cpp
// 功能：实现多影像地面重叠分析算法。
//
// 算法流程：
//   1) 逐影像反投影中心点到地面，得到地面中心坐标集合
//   2) 逐影像估算地面覆盖等效半径（footprint radius）
//   3) 构建 KD 树（基于地面中心点），加速邻域查询
//   4) 对每张影像，在 searchRadius = neighborFactor * r_i * 2.5 的范围内
//      用 KD 树检索邻近影像；对每对邻近影像（j > i）：
//        - 计算地面中心水平距离 d
//        - threshold = neighborFactor * (r_i + r_j)
//        - 若 d <= threshold，则认为有重叠，得分 = max(0, 1 - d/threshold)
//   5) 所有重叠对按得分降序排列输出
//
// 重叠得分模型（线性衰减）：
//   score = max(0, 1 - d / threshold)
//   score = 1.0 表示两影像中心完全重合（最大重叠）
//   score = 0.0 表示距离恰好等于阈值（临界无重叠）
// ============================================================

#include <plapoint/search/spatial_kdtree.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace
{

struct GroundProjection
{
    double x = 0.0;
    double y = 0.0;
};

struct SphereSurfaceBuildResult
{
    xjw::ReferenceSphereSurface surface;
    xjw::ReferenceSphereCenterMode centerMode = xjw::ReferenceSphereCenterMode::LocalTangent;
    double tangentElevationMeters = 0.0;
};

double distance2D(const GroundProjection &a, const GroundProjection &b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

double distance2D(double ax, double ay, double bx, double by)
{
    const double dx = ax - bx;
    const double dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

double dot3(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> sub3(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

std::array<double, 3> cross3(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

double norm3(const std::array<double, 3> &v)
{
    return std::sqrt(dot3(v, v));
}

std::array<double, 3> normalize3(const std::array<double, 3> &v, const std::array<double, 3> &fallback)
{
    const double n = norm3(v);
    if (n < 1e-12)
    {
        return fallback;
    }
    return {v[0] / n, v[1] / n, v[2] / n};
}

double medianValue(std::vector<double> values, double fallback)
{
    if (values.empty())
    {
        return fallback;
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool centerRayWorldDirection(const xjw::OverlapImageInput &image, std::array<double, 3> *dir)
{
    if (!dir)
    {
        return false;
    }

    const xjw::Camera &camera = image.camera;
    const double fu = camera.focalX();
    const double fv = camera.focalY();
    if (std::abs(fu) < 1e-12 || std::abs(fv) < 1e-12)
    {
        return false;
    }

    const double u = image.width > 0 ? 0.5 * double(image.width) : camera.principalX();
    const double v = image.height > 0 ? 0.5 * double(image.height) : camera.principalY();
    const double x = (u - camera.principalX()) / (double(camera.uAxisSign()) * fu);
    const double y = (v - camera.principalY()) / (double(camera.vAxisSign()) * fv);
    const std::array<double, 3> rayCam{x, y, 1.0};
    const auto R = camera.cameraToWorldRotation();
    const std::array<double, 3> rayWorld{
        R[0] * rayCam[0] + R[1] * rayCam[1] + R[2] * rayCam[2],
        R[3] * rayCam[0] + R[4] * rayCam[1] + R[5] * rayCam[2],
        R[6] * rayCam[0] + R[7] * rayCam[1] + R[8] * rayCam[2]};
    *dir = normalize3(rayWorld, {0.0, 0.0, -1.0});
    return true;
}

double medianNearestSpacingXY(const std::vector<xjw::OverlapImageInput> &images)
{
    if (images.size() < 2)
    {
        return 50.0;
    }

    std::vector<double> nearest;
    nearest.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i)
    {
        const auto ci = images[i].camera.cameraCenter();
        double best = std::numeric_limits<double>::max();
        for (size_t j = 0; j < images.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            const auto cj = images[j].camera.cameraCenter();
            best = std::min(best, distance2D(ci[0], ci[1], cj[0], cj[1]));
        }
        if (best < std::numeric_limits<double>::max())
        {
            nearest.push_back(best);
        }
    }
    return medianValue(nearest, 50.0);
}

SphereSurfaceBuildResult buildReferenceSphereSurface(const std::vector<xjw::OverlapImageInput> &images,
                                                     const xjw::ReferenceSphereOptions &options)
{
    SphereSurfaceBuildResult result;
    const double radius = options.radiusMeters > 0.0
        ? options.radiusMeters
        : xjw::referenceBodyRadiusMeters(options.body);

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    std::vector<double> norms;
    std::vector<double> dirZs;
    xs.reserve(images.size());
    ys.reserve(images.size());
    zs.reserve(images.size());
    norms.reserve(images.size());
    dirZs.reserve(images.size());

    for (const xjw::OverlapImageInput &image : images)
    {
        const auto c = image.camera.cameraCenter();
        xs.push_back(c[0]);
        ys.push_back(c[1]);
        zs.push_back(c[2]);
        norms.push_back(norm3(c));

        std::array<double, 3> dir;
        if (centerRayWorldDirection(image, &dir))
        {
            dirZs.push_back(dir[2]);
        }
    }

    xjw::ReferenceSphereCenterMode mode = options.centerMode;
    if (mode == xjw::ReferenceSphereCenterMode::Auto)
    {
        const double medianNorm = medianValue(norms, 0.0);
        mode = medianNorm > radius * 0.25
            ? xjw::ReferenceSphereCenterMode::PlanetCentered
            : xjw::ReferenceSphereCenterMode::LocalTangent;
    }
    result.centerMode = mode;

    if (mode == xjw::ReferenceSphereCenterMode::PlanetCentered)
    {
        result.surface.center = {0.0, 0.0, 0.0};
        result.surface.radiusMeters = radius + options.elevationMeters;
        result.tangentElevationMeters = options.elevationMeters;
        return result;
    }

    const double medianX = medianValue(xs, 0.0);
    const double medianY = medianValue(ys, 0.0);
    const double medianZ = medianValue(zs, 0.0);
    double tangentZ = options.elevationMeters;
    if (options.autoLocalTangentHeight)
    {
        const double spacing = medianNearestSpacingXY(images);
        const double altitudeGuess = std::max(10.0, 3.0 * spacing);
        const double medianDirZ = medianValue(dirZs, -1.0);
        tangentZ = medianDirZ < 0.0 ? medianZ - altitudeGuess : medianZ + altitudeGuess;
    }

    result.surface.center = {medianX, medianY, tangentZ - radius};
    result.surface.radiusMeters = radius;
    result.tangentElevationMeters = tangentZ;
    return result;
}

std::vector<GroundProjection> projectCenters(const std::vector<std::array<double, 3>> &centers,
                                             const xjw::ReferenceSphereSurface *sphere,
                                             xjw::ReferenceSphereCenterMode sphereCenterMode)
{
    std::vector<GroundProjection> projected;
    projected.reserve(centers.size());
    if (!sphere || sphereCenterMode != xjw::ReferenceSphereCenterMode::PlanetCentered)
    {
        for (const auto &center : centers)
        {
            projected.push_back({center[0], center[1]});
        }
        return projected;
    }

    std::array<double, 3> mean{0.0, 0.0, 0.0};
    for (const auto &center : centers)
    {
        mean[0] += center[0] - sphere->center[0];
        mean[1] += center[1] - sphere->center[1];
        mean[2] += center[2] - sphere->center[2];
    }
    if (!centers.empty())
    {
        mean[0] /= double(centers.size());
        mean[1] /= double(centers.size());
        mean[2] /= double(centers.size());
    }

    const std::array<double, 3> up = normalize3(mean, {0.0, 0.0, 1.0});
    std::array<double, 3> east = normalize3(cross3({0.0, 0.0, 1.0}, up), {1.0, 0.0, 0.0});
    if (norm3(east) < 1e-12)
    {
        east = normalize3(cross3({0.0, 1.0, 0.0}, up), {1.0, 0.0, 0.0});
    }
    const std::array<double, 3> north = normalize3(cross3(up, east), {0.0, 1.0, 0.0});
    const std::array<double, 3> origin{
        sphere->center[0] + up[0] * sphere->radiusMeters,
        sphere->center[1] + up[1] * sphere->radiusMeters,
        sphere->center[2] + up[2] * sphere->radiusMeters};

    for (const auto &center : centers)
    {
        const std::array<double, 3> delta = sub3(center, origin);
        projected.push_back({dot3(delta, east), dot3(delta, north)});
    }
    return projected;
}

} // namespace

namespace xjw
{

double referenceBodyRadiusMeters(ReferenceBody body)
{
    switch (body)
    {
    case ReferenceBody::Moon:
        return 1737400.0;
    case ReferenceBody::Mars:
        return 3389500.0;
    case ReferenceBody::Earth:
    default:
        return 6378137.0;
    }
}

const char *referenceBodyName(ReferenceBody body)
{
    switch (body)
    {
    case ReferenceBody::Moon:
        return "moon";
    case ReferenceBody::Mars:
        return "mars";
    case ReferenceBody::Earth:
    default:
        return "earth";
    }
}

// ============================================================
// 函数：OverlapAnalyzer::analyze
// ============================================================
bool OverlapAnalyzer::analyze(const std::vector<OverlapImageInput> &images,
                              const DemSurface *dem,
                              bool useFixedZ,
                              double fixedZ,
                              double neighborFactor,
                              OverlapAnalysisResult *result,
                              std::string *errorMsg)
{
    OverlapAnalysisOptions options;
    options.groundModel = (useFixedZ || !dem) ? OverlapGroundModel::FixedZPlane : OverlapGroundModel::Dem;
    options.dem = dem;
    options.fixedZ = fixedZ;
    options.neighborFactor = neighborFactor;
    return analyze(images, options, result, errorMsg);
}

bool OverlapAnalyzer::analyze(const std::vector<OverlapImageInput> &images,
                              const OverlapAnalysisOptions &options,
                              OverlapAnalysisResult *result,
                              std::string *errorMsg)
{
    if (!result)
    {
        return false;
    }

    // 清除上次运行结果
    result->centers.clear();
    result->footprintRadii.clear();
    result->pairs.clear();

    // 至少 2 张影像才有意义进行重叠分析
    if (images.size() < 2)
    {
        if (errorMsg)
        {
            *errorMsg = "至少需要两张影像";
        }
        return false;
    }

    // 确保邻域系数为正（避免搜索范围为 0 或负数）
    const double kNeighbor = std::max(0.1, options.neighborFactor);

    // 预分配输出数组
    result->centers.resize(images.size());
    result->footprintRadii.resize(images.size(), 0.0);

    ReferenceSphereSurface sphere;
    bool useSphere = options.groundModel == OverlapGroundModel::ReferenceSphere;
    ReferenceSphereCenterMode sphereCenterMode = ReferenceSphereCenterMode::LocalTangent;
    double tangentElevation = 0.0;
    if (useSphere)
    {
        SphereSurfaceBuildResult sphereBuild = buildReferenceSphereSurface(images, options.referenceSphere);
        sphere = sphereBuild.surface;
        sphereCenterMode = sphereBuild.centerMode;
        tangentElevation = sphereBuild.tangentElevationMeters;
        if (sphere.radiusMeters <= 0.0)
        {
            if (errorMsg)
            {
                *errorMsg = "基准球半径无效";
            }
            return false;
        }
    }

    // ---- Step 1 & 2：逐影像反投影中心点并估算地面覆盖半径 ----
    for (size_t i = 0; i < images.size(); ++i)
    {
        std::string err;

        // 将影像中心像素反投影到地面，得到地面中心坐标
        bool centerOk = false;
        if (useSphere)
        {
            centerOk = GroundBackProjector::imageCenterToSphere(images[i].camera,
                                                                images[i].width,
                                                                images[i].height,
                                                                sphere,
                                                                &result->centers[i],
                                                                &err);
        }
        else
        {
            const bool useFixedZ = options.groundModel != OverlapGroundModel::Dem || !options.dem;
            centerOk = GroundBackProjector::imageCenterToGround(images[i].camera,
                                                                images[i].width,
                                                                images[i].height,
                                                                options.dem,
                                                                useFixedZ,
                                                                options.fixedZ,
                                                                &result->centers[i],
                                                                &err);
        }
        if (!centerOk)
        {
            if (errorMsg)
            {
                *errorMsg = "中心点反投影失败: " + images[i].imagePath + " | " + err;
            }
            return false;
        }

        // 估算影像地面覆盖等效半径（四角到中心平均距离）
        double radius = 0.0;
        bool radiusOk = false;
        if (useSphere)
        {
            radiusOk = GroundBackProjector::estimateFootprintRadiusOnSphere(images[i].camera,
                                                                            images[i].width,
                                                                            images[i].height,
                                                                            sphere,
                                                                            &radius,
                                                                            &err);
        }
        else
        {
            const bool useFixedZ = options.groundModel != OverlapGroundModel::Dem || !options.dem;
            radiusOk = GroundBackProjector::estimateFootprintRadius(images[i].camera,
                                                                    images[i].width,
                                                                    images[i].height,
                                                                    options.dem,
                                                                    useFixedZ,
                                                                    options.fixedZ,
                                                                    &radius,
                                                                    &err);
        }
        if (!radiusOk)
        {
            radius = 1.0; // 估算失败时使用默认值 1.0（兜底，避免除零）
        }
        // 确保半径至少为 1e-3，防止后续除零
        result->footprintRadii[i] = std::max(1e-3, radius);
    }

    const std::vector<GroundProjection> projectedCenters = projectCenters(result->centers,
                                                                          useSphere ? &sphere : nullptr,
                                                                          sphereCenterMode);

    using CenterKdTree2D = plapoint::search::SpatialKdTree<2, double>;

    // 用于构建 KD 树的 2D 地面中心点集
    std::vector<CenterKdTree2D::Point> centerPts;
    centerPts.reserve(images.size());
    for (size_t i = 0; i < projectedCenters.size(); ++i)
    {
        centerPts.push_back(CenterKdTree2D::Point{{projectedCenters[i].x, projectedCenters[i].y},
                                                  static_cast<int>(i)});
    }

    // ---- Step 3：构建 KD 树，用于快速邻域搜索 ----
    CenterKdTree2D tree(centerPts);

    // ---- Step 4：遍历所有影像，寻找重叠对 ----
    for (size_t i = 0; i < images.size(); ++i)
    {
        // 以当前影像地面半径的 neighborFactor * 2.5 倍作为 KD 树搜索半径
        // 乘以 2.5 是为了保守地覆盖两影像半径之和的最大可能范围
        const double searchRadius = kNeighbor * result->footprintRadii[i] * 2.5;

        // KD 树半径搜索：返回搜索范围内所有影像的索引
        std::vector<int> nearby =
            tree.radiusSearch(CenterKdTree2D::CoordinateArray{projectedCenters[i].x, projectedCenters[i].y},
                              searchRadius);

        for (int j : nearby)
        {
            // 只处理 j > i 的对，避免重复统计 (i,j) 和 (j,i)
            if (j <= static_cast<int>(i)) continue;

            // 计算两影像地面中心的水平距离
            const double distance = distance2D(projectedCenters[i], projectedCenters[static_cast<size_t>(j)]);

            // 计算重叠判断阈值：
            //   threshold = neighborFactor * (r_i + r_j)
            //   当 d ≤ threshold 时认为两影像有重叠覆盖
            const double threshold = std::max(
                1e-6,
                kNeighbor * (result->footprintRadii[i]
                             + result->footprintRadii[static_cast<size_t>(j)]));
            if (distance > threshold) continue; // 距离超出阈值，无重叠

            // 线性重叠得分：中心重合时为 1.0，距离达到阈值时为 0.0
            const double score = std::max(0.0, 1.0 - distance / threshold);

            result->pairs.push_back(OverlapPairResult{
                static_cast<int>(i), // indexA（较小索引）
                j,                   // indexB（较大索引）
                distance,
                score});
        }
    }

    // ---- Step 5：按重叠得分降序排列（高重叠优先输出）----
    std::sort(
        result->pairs.begin(),
        result->pairs.end(),
        [](const OverlapPairResult &a, const OverlapPairResult &b)
        {
            return a.overlapScore > b.overlapScore;
        });

    // 生成分析摘要字符串
    std::ostringstream oss;
    oss << "输入影像: " << images.size()
        << ", 重叠对: " << result->pairs.size()
        << ", 邻域系数: " << kNeighbor;
    if (useSphere)
    {
        oss << ", ground=reference_sphere"
            << ", body=" << referenceBodyName(options.referenceSphere.body)
            << ", radius_m=" << sphere.radiusMeters
            << ", center_mode="
            << (sphereCenterMode == ReferenceSphereCenterMode::PlanetCentered ? "planet_centered" : "local_tangent")
            << ", tangent_z=" << tangentElevation;
    }
    else if (options.groundModel == OverlapGroundModel::Dem && options.dem)
    {
        oss << ", ground=dem";
    }
    else
    {
        oss << ", ground=fixed_z"
            << ", fixed_z=" << options.fixedZ;
    }
    result->detail = oss.str();
    return true;
}

} // namespace xjw
