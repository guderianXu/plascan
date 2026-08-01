#pragma once

#include <array>

namespace xjw
{
class Camera;
}

namespace xjw::mesh
{

/**
 * @brief A depth observation expressed in both camera-Z and Euclidean ray units.
 *
 * PlaScan depth maps store positive physical camera-Z depth.  This remains true
 * for cameras whose physical forward axis is negative camera Z.  The ray scale
 * converts that depth to the Euclidean distance from the camera centre.
 */
struct DepthRayMetricSample
{
    bool valid = false;
    double cameraZDepth = 0.0;
    double rayDistance = 0.0;
    double rayDistancePerCameraZ = 0.0;
    double worldPixelFootprint = 0.0;
    double worldPixelFootprintX = 0.0;
    double worldPixelFootprintY = 0.0;
    std::array<double, 3> cameraCenter{};
    std::array<double, 3> unitWorldRay{};
    std::array<double, 3> worldPoint{};
};

/**
 * @brief Converts camera-Z depth into ray-aware geometry for depth fusion.
 */
class DepthRayMetric final
{
public:
    /**
     * @brief Evaluates the ray and the world-space footprint of one image pixel.
     *
     * The scalar footprint is the area-equivalent geometric mean of the
     * horizontal and vertical footprints.  Each axis footprint is measured
     * symmetrically against the two half-pixel boundary rays.
     */
    static DepthRayMetricSample evaluate(
        const Camera &camera,
        const std::array<double, 2> &pixel,
        double positiveCameraZDepth);

    /**
     * @brief Moves the sample by a signed camera-Z depth offset.
     *
     * Positive offsets move away from the camera.  The function rejects
     * non-finite offsets and results at or behind the camera centre.
     */
    static bool pointAtCameraZOffset(
        const DepthRayMetricSample &sample,
        double signedCameraZOffset,
        std::array<double, 3> *worldPoint,
        double *rayDistance = nullptr);

    /**
     * @brief Moves the sample by a signed Euclidean distance along its ray.
     *
     * Positive offsets move away from the camera.  The resulting positive
     * camera-Z depth can optionally be returned.
     */
    static bool pointAtRayDistanceOffset(
        const DepthRayMetricSample &sample,
        double signedRayDistanceOffset,
        std::array<double, 3> *worldPoint,
        double *positiveCameraZDepth = nullptr);
};

} // namespace xjw::mesh
