#pragma once

#include "placamera/frame_camera.h"

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace placamera
{

    /** One cameras.txt model; parameter order follows COLMAP's text format. */
    struct ColmapCamera
    {
        std::string model;
        int width = 0;
        int height = 0;
        std::vector<double> parameters;
        int cameraId = 0;
    };

    /** One images.txt header, converted to a camera-to-world PlaCamera pose. */
    struct ColmapImage
    {
        int imageId = 0;
        int cameraId = 0;
        std::string imageName;
        Pose pose;
    };

    /** Parse and validate one non-comment cameras.txt row. */
    Result<ColmapCamera> parseColmapCameraLine(std::string_view line);

    /** Parse one images.txt header. The caller consumes the following POINTS2D row. */
    Result<ColmapImage> parseColmapImageLine(std::string_view line, FrameId worldFrame);

    bool isSupportedColmapCamera(const ColmapCamera& camera) noexcept;
    Result<bool> colmapHasDistortion(const ColmapCamera& camera, double tolerance = 1.0e-12);

    /** Return an exact Brown-Conrady representation or an UnsupportedModel failure. */
    Result<BrownConradyDistortion> colmapBrownConradyDistortion(const ColmapCamera& camera, double tolerance = 1.0e-12);

    /** Convert COLMAP's first pixel center (0.5, 0.5) to raster index center (0, 0). */
    Result<FrameIntrinsics> colmapRasterIntrinsics(const ColmapCamera& camera);

    /** Validates once, then projects many pixels without repeating model checks. */
    class ColmapProjector
    {
    public:
        static Result<ColmapProjector> create(ColmapCamera camera);

        const FrameIntrinsics& rasterIntrinsics() const noexcept;
        std::array<double, 2> project(double x, double y) const noexcept;

    private:
        ColmapProjector(ColmapCamera camera, FrameIntrinsics intrinsics);

        ColmapCamera _camera;
        FrameIntrinsics _intrinsics;
    };

    /** Project an undistorted normalized ray to the source raster, including fisheye distortion. */
    Result<std::array<double, 2>> projectColmapNormalized(const ColmapCamera& camera, double x, double y);

} // namespace placamera
