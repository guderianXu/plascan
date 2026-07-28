#pragma once

#include <array>
#include <optional>

#include "Camera.h"

namespace xjw
{

/**
 * @brief 两台相机之间的摄影测量基线及针对指定空间点的几何指标。
 *
 * `length()` 仅表示两个相机光心的欧氏距离；指定空间点后，
 * `triangulationAngleDeg()` 表示该点处两条观测射线的夹角，
 * `meanDepthToBaselineRatio()` 则用于判断深度估计的相对几何强度。
 */
class CameraBaseline
{
public:
    /// 仅根据两个相机光心计算物理基线长度。
    static CameraBaseline evaluate(const Camera &first, const Camera &second);

    /// 额外计算指定空间点的观测夹角、前方性和深度/基线比。
    static CameraBaseline evaluate(const Camera &first,
                                   const Camera &second,
                                   const std::array<double, 3> &worldPoint);

    /// 两个光心均为有限数且不重合时返回 true。
    bool isValid() const { return _valid; }

    /// 两个相机光心的欧氏距离，单位与相机中心坐标一致。
    double length() const { return _length; }

    /// 是否已针对某一空间点完成几何计算。
    bool hasPointGeometry() const { return _hasPointGeometry; }

    /// 指定点处两条观测射线的夹角，单位为度；无指定点时为空。
    std::optional<double> triangulationAngleDeg() const { return _triangulationAngleDeg; }

    /// 指定点是否同时位于两台相机的物理前方。
    bool isPointInFrontOfBothCameras() const { return _pointInFrontOfBothCameras; }

    /// 两相机物理深度均为正时的平均深度/基线比；否则为空。
    std::optional<double> meanDepthToBaselineRatio() const { return _meanDepthToBaselineRatio; }

private:
    bool _valid = false;
    double _length = 0.0;
    bool _hasPointGeometry = false;
    std::optional<double> _triangulationAngleDeg;
    bool _pointInFrontOfBothCameras = false;
    std::optional<double> _meanDepthToBaselineRatio;
};

} // namespace xjw
