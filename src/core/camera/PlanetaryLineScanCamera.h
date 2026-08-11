#pragma once

#include "CameraModel.h"

#include <array>
#include <string>
#include <vector>

namespace xjw
{

/**
 * @brief Time-dependent planetary pushbroom camera loaded from a USGSCSM ISD.
 *
 * This class is intentionally separate from the static Tsai FramePinholeCamera. Positions
 * and ground coordinates use target body-fixed metres; ephemeris times use TDB
 * seconds. The P0 implementation supports the USGS Astro line-scanner model
 * and the LRO LROC NAC optical-distortion model used by the ISIS lidar fixture.
 */
class PlanetaryLineScanCamera final : public CameraModel
{
public:
    using Vector3 = std::array<double, 3>;
    using Matrix3 = std::array<double, 9>;

    enum class PixelConvention
    {
        CsmPixelCenter, ///< Upper-left pixel centre is (sample, line) = (0.5, 0.5).
        OpenCvZeroBased ///< Upper-left pixel centre is (sample, line) = (0.0, 0.0).
    };

    struct ImageCoordinate
    {
        double sample = 0.0;
        double line = 0.0;
    };

    struct ImagingRay
    {
        Vector3 centerBodyFixedMeters{{0.0, 0.0, 0.0}};
        Vector3 directionBodyFixed{{0.0, 0.0, 1.0}};
        double ephemerisTimeSeconds = 0.0;
    };

    /**
     * @brief Optional bundle-adjustment bias in the target body-fixed frame.
     *
     * The corrected pose is `R' = bodyFixedRotation * R` and
     * `C' = C + bodyFixedTranslationMeters`. This is suitable for a
     * left-multiplicative small-angle attitude update without changing the
     * nominal trajectory stored in the ISD.
     */
    struct PoseBias
    {
        Matrix3 bodyFixedRotation{{1.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0,
                                   0.0, 0.0, 1.0}};
        Vector3 bodyFixedTranslationMeters{{0.0, 0.0, 0.0}};
    };

    struct FixedLineProjection
    {
        double sample = 0.0;
        double line = 0.0;
        double detectorLineResidualPixels = 0.0;
        double undistortedFocalXMillimeters = 0.0;
        double undistortedFocalYMillimeters = 0.0;
        double sensorDepthMeters = 0.0;
        double ephemerisTimeSeconds = 0.0;
    };

    struct GroundToImageOptions
    {
        double desiredLinePrecisionPixels = 1.0e-3;
        int maximumIterations = 20;
        bool requireInsideImage = false;
    };

    PlanetaryLineScanCamera() = default;

    bool loadFromIsd(const std::string &path, std::string *errorMessage = nullptr);
    CameraModelType modelType() const noexcept override;
    bool isValid() const noexcept override { return _isLoaded; }
    std::optional<CameraImageSize> imageSize() const noexcept override;
    std::string_view worldFrameName() const noexcept override;
    bool rayForPixel(const CameraImageCoordinate &pixel,
                     CameraImagingRay *ray) const override;
    bool groundToImage(const Vector3 &groundBodyFixedMeters,
                       CameraGroundProjection *projection) const override;

    int imageLines() const { return _imageLines; }
    int imageSamples() const { return _imageSamples; }
    const std::string &modelName() const { return _modelName; }
    const std::string &platformName() const { return _platformName; }
    const std::string &sensorName() const { return _sensorName; }
    const std::string &declaredInterpolationMethod() const { return _declaredInterpolationMethod; }
    const std::string &targetName() const { return _targetName; }
    const std::string &bodyFixedFrameName() const { return _bodyFixedFrameName; }
    int bodyFixedFrameCode() const { return _bodyFixedFrameCode; }
    double startingEphemerisTimeSeconds() const { return _startingEphemerisTimeSeconds; }
    double centerEphemerisTimeSeconds() const { return _centerEphemerisTimeSeconds; }
    double focalLengthMillimeters() const { return _focalLengthMillimeters; }
    double detectorSampleSumming() const { return _detectorSampleSumming; }
    double detectorLineSumming() const { return _detectorLineSumming; }

    static PoseBias bodyFixedSmallAngleBias(const Vector3 &angleAxisRadians,
                                            const Vector3 &translationMeters);

    bool absoluteEtForLine(double line,
                           PixelConvention convention,
                           double *ephemerisTimeSeconds) const;
    bool lineForAbsoluteEt(double ephemerisTimeSeconds,
                           PixelConvention convention,
                           double *line) const;

    bool sensorCenterBodyFixedAtEt(double ephemerisTimeSeconds,
                                   Vector3 *centerMeters) const;
    bool sensorToBodyFixedAtEt(double ephemerisTimeSeconds,
                               Matrix3 *rotation) const;

    bool pixelToDistortedFocalPlane(double sample,
                                    PixelConvention convention,
                                    std::array<double, 2> *focalMillimeters) const;
    bool distortedFocalPlaneToPixel(const std::array<double, 2> &focalMillimeters,
                                    PixelConvention convention,
                                    double *sample,
                                    double *detectorLineOffsetPixels) const;
    bool removeOpticalDistortion(const std::array<double, 2> &distortedMillimeters,
                                 std::array<double, 2> *undistortedMillimeters) const;
    bool applyOpticalDistortion(const std::array<double, 2> &undistortedMillimeters,
                                std::array<double, 2> *distortedMillimeters) const;

    bool pixelRayBodyFixed(double sample,
                           double line,
                           PixelConvention convention,
                           ImagingRay *ray) const;
    bool pixelRayBodyFixed(double sample,
                           double line,
                           PixelConvention convention,
                           ImagingRay *ray,
                           const PoseBias &bias) const;

    /**
     * @brief Project a body-fixed point at an already observed pushbroom line.
     *
     * The returned detector-line residual is the along-track collinearity
     * residual in pixels. It is zero when the 3D point lies in the instantaneous
     * detector plane. This method is the direct observation function for BA.
     */
    bool projectAtObservedLine(const Vector3 &groundBodyFixedMeters,
                               double observedLine,
                               PixelConvention convention,
                               FixedLineProjection *projection) const;
    bool projectAtObservedLine(const Vector3 &groundBodyFixedMeters,
                               double observedLine,
                               PixelConvention convention,
                               FixedLineProjection *projection,
                               const PoseBias &bias) const;

    bool groundToImage(const Vector3 &groundBodyFixedMeters,
                       PixelConvention convention,
                       ImageCoordinate *image) const;
    bool groundToImage(const Vector3 &groundBodyFixedMeters,
                       PixelConvention convention,
                       ImageCoordinate *image,
                       const GroundToImageOptions &options) const;
    bool groundToImage(const Vector3 &groundBodyFixedMeters,
                       PixelConvention convention,
                       ImageCoordinate *image,
                       const GroundToImageOptions &options,
                       const PoseBias &bias) const;

private:
    struct LineRateRecord
    {
        double startLineCsm = 0.5;
        double startTimeRelativeToCenterSeconds = 0.0;
        double secondsPerLine = 0.0;
    };

    struct QuaternionSample
    {
        double ephemerisTimeSeconds = 0.0;
        std::array<double, 4> scalarFirst{{1.0, 0.0, 0.0, 0.0}};
    };

    struct StateSample
    {
        double ephemerisTimeSeconds = 0.0;
        Vector3 positionKilometers{{0.0, 0.0, 0.0}};
        Vector3 velocityKilometersPerSecond{{0.0, 0.0, 0.0}};
    };

    bool correctedPoseAtEt(double ephemerisTimeSeconds,
                           const PoseBias &bias,
                           Vector3 *centerMeters,
                           Matrix3 *sensorToBodyFixed) const;
    bool isLineInsideImage(double line, PixelConvention convention) const;

    int _imageLines = 0;
    int _imageSamples = 0;
    std::string _modelName;
    std::string _platformName;
    std::string _sensorName;
    std::string _declaredInterpolationMethod;
    std::string _targetName;
    std::string _bodyFixedFrameName;
    int _bodyFixedFrameCode = 0;
    double _startingEphemerisTimeSeconds = 0.0;
    double _centerEphemerisTimeSeconds = 0.0;
    double _focalLengthMillimeters = 0.0;
    double _detectorSampleSumming = 1.0;
    double _detectorLineSumming = 1.0;
    double _detectorSampleOrigin = 0.0;
    double _detectorLineOrigin = 0.0;
    double _startingDetectorSample = 0.0;
    double _startingDetectorLine = 0.0;
    std::array<double, 3> _focalToPixelSamples{{0.0, 0.0, 0.0}};
    std::array<double, 3> _focalToPixelLines{{0.0, 0.0, 0.0}};
    double _lroNacDistortionK1 = 0.0;
    Matrix3 _bodyConstantRotation{{1.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0,
                                   0.0, 0.0, 1.0}};
    Matrix3 _instrumentConstantRotation{{1.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0,
                                         0.0, 0.0, 1.0}};
    std::vector<LineRateRecord> _lineRates;
    std::vector<QuaternionSample> _bodyRotations;
    std::vector<QuaternionSample> _instrumentPointing;
    std::vector<StateSample> _instrumentStates;
    bool _isLoaded = false;
};

} // namespace xjw
