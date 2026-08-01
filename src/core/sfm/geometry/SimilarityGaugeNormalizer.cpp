/**
 * @file SimilarityGaugeNormalizer.cpp
 * @brief 两相机基线规范的验证和原子应用实现。
 *
 * 变换形式为 `X' = C_ref + s * (X - C_refined)`。因为只进行统一平移和尺度，
 * 相机旋转、内参及所有重投影像素保持不变。
 */

#include "SimilarityGaugeNormalizer.h"

#include <array>
#include <cmath>

namespace xjw
{
namespace
{

bool finitePoint(const std::array<double, 3> &point)
{
    return std::isfinite(point[0]) &&
           std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

double distance(const std::array<double, 3> &left,
                const std::array<double, 3> &right)
{
    const double dx = left[0] - right[0];
    const double dy = left[1] - right[1];
    const double dz = left[2] - right[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::array<double, 3> transformPoint(const std::array<double, 3> &point,
                                     const std::array<double, 3> &sourceAnchor,
                                     const std::array<double, 3> &targetAnchor,
                                     double scale)
{
    return {{
        targetAnchor[0] + scale * (point[0] - sourceAnchor[0]),
        targetAnchor[1] + scale * (point[1] - sourceAnchor[1]),
        targetAnchor[2] + scale * (point[2] - sourceAnchor[2]),
    }};
}

} // namespace

SimilarityGaugeNormalizationResult normalizeSimilarityGauge(
    const std::vector<Camera> &referenceCameras,
    int anchorCameraIndex,
    int scaleCameraIndex,
    std::vector<Camera> *refinedCameras,
    std::vector<BARefinedPoint> *refinedPoints)
{
    SimilarityGaugeNormalizationResult result;
    if (!refinedCameras || !refinedPoints)
    {
        result.reason = "null_output";
        return result;
    }
    if (referenceCameras.size() != refinedCameras->size())
    {
        result.reason = "camera_count_mismatch";
        return result;
    }
    if (anchorCameraIndex < 0 ||
        scaleCameraIndex < 0 ||
        anchorCameraIndex == scaleCameraIndex ||
        anchorCameraIndex >= static_cast<int>(referenceCameras.size()) ||
        scaleCameraIndex >= static_cast<int>(referenceCameras.size()))
    {
        result.reason = "invalid_gauge_camera_indices";
        return result;
    }

    const auto referenceAnchor =
        referenceCameras[static_cast<std::size_t>(anchorCameraIndex)].cameraCenter();
    const auto referenceScaleCamera =
        referenceCameras[static_cast<std::size_t>(scaleCameraIndex)].cameraCenter();
    const auto refinedAnchor =
        (*refinedCameras)[static_cast<std::size_t>(anchorCameraIndex)].cameraCenter();
    const auto refinedScaleCamera =
        (*refinedCameras)[static_cast<std::size_t>(scaleCameraIndex)].cameraCenter();
    if (!finitePoint(referenceAnchor) ||
        !finitePoint(referenceScaleCamera) ||
        !finitePoint(refinedAnchor) ||
        !finitePoint(refinedScaleCamera))
    {
        result.reason = "non_finite_gauge_camera";
        return result;
    }

    // 参考基线来自 BA 前状态，细化基线来自 BA 输出；二者都必须非退化。
    const double referenceBaseline = distance(referenceAnchor, referenceScaleCamera);
    const double refinedBaseline = distance(refinedAnchor, refinedScaleCamera);
    constexpr double kMinimumBaseline = 1.0e-10;
    if (!(referenceBaseline > kMinimumBaseline) ||
        !(refinedBaseline > kMinimumBaseline))
    {
        result.reason = "degenerate_gauge_baseline";
        return result;
    }

    // 统一正尺度排除镜像；本步骤不估计旋转。
    const double scale = referenceBaseline / refinedBaseline;
    if (!std::isfinite(scale) || !(scale > 0.0))
    {
        result.reason = "invalid_gauge_scale";
        return result;
    }

    // 先在副本中完成全部变换，失败路径不会部分修改 BA 输出。
    std::vector<Camera> normalizedCameras = *refinedCameras;
    for (Camera &camera : normalizedCameras)
    {
        const auto center = camera.cameraCenter();
        if (!finitePoint(center))
        {
            result.reason = "non_finite_refined_camera";
            return result;
        }
        camera.setCameraCenter(transformPoint(center, refinedAnchor, referenceAnchor, scale));
    }

    std::vector<BARefinedPoint> normalizedPoints = *refinedPoints;
    for (BARefinedPoint &point : normalizedPoints)
    {
        if (finitePoint(point.point))
        {
            point.point = transformPoint(point.point, refinedAnchor, referenceAnchor, scale);
        }
    }

    refinedCameras->swap(normalizedCameras);
    refinedPoints->swap(normalizedPoints);
    result.applied = true;
    result.scale = scale;
    result.reason = "ok";
    return result;
}

} // namespace xjw
