#pragma once

#include "BundleAdjust.h"
#include "Camera.h"

#include <string>
#include <vector>

namespace xjw
{

struct SimilarityGaugeNormalizationResult
{
    bool applied = false;
    double scale = 1.0;
    std::string reason;
};

/**
 * @brief 用 BA 前的一条相机基线恢复单目重建的尺度规范。
 *
 * 第一台相机的中心恢复到 BA 前的位置，所有其它相机中心和三维点相对该中心
 * 做同一尺度变换。相机旋转和内参保持不变，因此不会改变重投影几何。
 */
SimilarityGaugeNormalizationResult normalizeSimilarityGauge(
    const std::vector<Camera> &referenceCameras,
    int anchorCameraIndex,
    int scaleCameraIndex,
    std::vector<Camera> *refinedCameras,
    std::vector<BARefinedPoint> *refinedPoints);

} // namespace xjw
