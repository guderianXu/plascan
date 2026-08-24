#pragma once

#include <array>
#include <optional>
#include <string_view>

namespace xjw
{

/**
 * @brief Public category of a camera geometry model.
 *
 * The category describes how an image observation is formed. It is deliberately
 * independent from bundle-adjustment parameter blocks and file formats.
 */
enum class CameraModelType
{
    FramePinhole,
    PlanetaryLineScan,
    RationalPolynomial
};

/** @brief Image coordinate using OpenCV's zero-based pixel-centre convention. */
struct CameraImageCoordinate
{
    double sample = 0.0;
    double line = 0.0;
};

/** @brief Optional raster dimensions associated with a camera model. */
struct CameraImageSize
{
    int samples = 0;
    int lines = 0;
};

/** @brief A world-space imaging ray for one image observation. */
struct CameraImagingRay
{
    std::array<double, 3> originMeters{{0.0, 0.0, 0.0}};
    std::array<double, 3> direction{{0.0, 0.0, 1.0}}; ///< Unit vector in the world frame.
    std::optional<double> ephemerisTimeSeconds;
};

/** @brief Result of projecting a world point into an image. */
struct CameraGroundProjection
{
    CameraImageCoordinate image;
    double positiveDepthMeters = 0.0; ///< Instantaneous optical-axis depth, not slant range.
    std::optional<double> ephemerisTimeSeconds;
};

/**
 * @brief Read-only geometry shared by frame, time-dependent line-scan and RPC cameras.
 *
 * The common boundary intentionally contains only pixel/ray/ground operations.
 * Intrinsic parameters, pose updates and bundle-adjustment residuals remain on
 * the concrete models because pushbroom and RPC images have no single static pose.
 * RPC ray queries return a documented local chord over the model's height range.
 */
class CameraModel
{
public:
    virtual ~CameraModel();

    virtual CameraModelType modelType() const noexcept = 0;
    virtual bool isValid() const noexcept = 0;
    virtual std::optional<CameraImageSize> imageSize() const noexcept = 0;
    virtual std::string_view worldFrameName() const noexcept = 0;

    virtual bool rayForPixel(const CameraImageCoordinate &pixel,
                             CameraImagingRay *ray) const = 0;
    // Time-dependent models may solve the image line iteratively to their
    // model-specific default precision.
    virtual bool groundToImage(const std::array<double, 3> &groundMeters,
                               CameraGroundProjection *projection) const = 0;
};

} // namespace xjw
