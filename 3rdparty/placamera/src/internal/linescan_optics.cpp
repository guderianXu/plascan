#include "linescan_projection.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace placamera
{

    namespace
    {

        struct CalibratedPixel
        {
            double sample = 0.0;
            double line = 0.0;
        };

        double coordinateToCsm(double coordinate, LineScanPixelConvention convention) noexcept
        {
            return convention == LineScanPixelConvention::ZeroBased ? coordinate + 0.5 : coordinate;
        }

        double csmToCoordinate(double coordinate, LineScanPixelConvention convention) noexcept
        {
            return convention == LineScanPixelConvention::ZeroBased ? coordinate - 0.5 : coordinate;
        }

        double coordinateToCompleteCalibration(double coordinate, LineScanPixelConvention convention) noexcept
        {
            return convention == LineScanPixelConvention::PixelCenter ? coordinate - 0.5 : coordinate;
        }

        double completeCalibrationToCoordinate(double coordinate, LineScanPixelConvention convention) noexcept
        {
            return convention == LineScanPixelConvention::PixelCenter ? coordinate + 0.5 : coordinate;
        }

        bool validOptions(const EvaluationOptions& options) noexcept
        {
            return std::isfinite(options.desiredPrecisionPixels) && options.desiredPrecisionPixels > 0.0 &&
                   options.maximumIterations > 0;
        }

        void applyCompleteDistortion(const MetashapeCalibration& calibration,
                                     double x,
                                     double y,
                                     double* distortedX,
                                     double* distortedY) noexcept
        {
            const double x_squared = x * x;
            const double y_squared = y * y;
            const double radius_squared = x_squared + y_squared;
            const double radius_fourth = radius_squared * radius_squared;
            const double radial_delta = calibration.k1 * radius_squared + calibration.k2 * radius_fourth +
                                        calibration.k3 * radius_fourth * radius_squared +
                                        calibration.k4 * radius_fourth * radius_fourth;
            double diagonal_x = 3.0 * x;
            diagonal_x *= x;
            diagonal_x += y_squared;
            double diagonal_y = 3.0 * y;
            diagonal_y *= y;
            diagonal_y += x_squared;
            double cross_x = calibration.p2 + calibration.p2;
            cross_x *= x;
            cross_x *= y;
            double cross_y = calibration.p1 + calibration.p1;
            cross_y *= x;
            cross_y *= y;
            const double tangential_scale = 1.0 + calibration.p3 * radius_squared + calibration.p4 * radius_fourth;
            const double tangential_x = (calibration.p1 * diagonal_x + cross_x) * tangential_scale;
            const double tangential_y = (cross_y + calibration.p2 * diagonal_y) * tangential_scale;
            *distortedX = x + (x * radial_delta + tangential_x);
            *distortedY = y + (y * radial_delta + tangential_y);
        }

        CalibratedPixel calibratedPixelFromPlane(const MetashapeCalibration& calibration, double x, double y) noexcept
        {
            return {(calibration.f + calibration.b1) * x + calibration.b2 * y + calibration.cx,
                    calibration.f * y + calibration.cy};
        }

        std::array<double, 2> calibratedPlaneFromPixel(const MetashapeCalibration& calibration,
                                                       const CalibratedPixel& pixel) noexcept
        {
            double center_x = calibration.cx;
            double center_y = calibration.cy;
            double cx_offset = 0.0;
            double cy_offset = 0.0;
            if (calibration.principalPointDecomposition)
            {
                const PrincipalPointDecomposition& principal = *calibration.principalPointDecomposition;
                if (principal.imageCenterX + principal.cxOffset == calibration.cx)
                {
                    center_x = principal.imageCenterX;
                    cx_offset = principal.cxOffset;
                }
                if (principal.imageCenterY + principal.cyOffset == calibration.cy)
                {
                    center_y = principal.imageCenterY;
                    cy_offset = principal.cyOffset;
                }
            }
            const double y = ((pixel.line - center_y) - cy_offset) / calibration.f;
            const double x =
                ((pixel.sample - center_x) - cx_offset - calibration.b2 * y) / (calibration.f + calibration.b1);
            return {x, y};
        }

        EvaluationResult<FocalPlaneCoordinate>
        pixelToDistortedFocal(const LineScanOptics& optics, LineScanPixelConvention convention, double sample)
        {
            if (!std::isfinite(sample))
            {
                return EvaluationResult<FocalPlaneCoordinate>::failure(CameraErrorCode::InvalidArgument,
                                                                       "line-scan sample must be finite");
            }
            const double csm_sample = coordinateToCsm(sample, convention);
            if (!optics.detectorGeometry)
            {
                return EvaluationResult<FocalPlaneCoordinate>::success(
                    {(csm_sample - optics.principalSample) * optics.samplePitchMillimeters, 0.0});
            }

            const LineScanDetectorGeometry& detector = *optics.detectorGeometry;
            const double detector_sample =
                csm_sample * detector.detectorSampleSumming + detector.startingDetectorSample;
            const double line_offset =
                detector.startingDetectorLine - detector.detectorLineOrigin - detector.focalToPixelLines[0];
            const double sample_offset =
                detector_sample - detector.detectorSampleOrigin - detector.focalToPixelSamples[0];
            const double determinant = detector.focalToPixelLines[1] * detector.focalToPixelSamples[2] -
                                       detector.focalToPixelLines[2] * detector.focalToPixelSamples[1];
            if (std::abs(determinant) < 1.0e-15)
            {
                return EvaluationResult<FocalPlaneCoordinate>::failure(CameraErrorCode::OutsideModelDomain,
                                                                       "line-scan detector mapping is singular");
            }

            const FocalPlaneCoordinate focal{
                (detector.focalToPixelSamples[2] * line_offset - detector.focalToPixelLines[2] * sample_offset) /
                    determinant,
                (-detector.focalToPixelSamples[1] * line_offset + detector.focalToPixelLines[1] * sample_offset) /
                    determinant};
            if (!std::isfinite(focal.xMillimeters) || !std::isfinite(focal.yMillimeters))
            {
                return EvaluationResult<FocalPlaneCoordinate>::failure(
                    CameraErrorCode::OutsideModelDomain,
                    "line-scan detector mapping produced a non-finite focal coordinate");
            }
            return EvaluationResult<FocalPlaneCoordinate>::success(focal);
        }

        EvaluationResult<LineScanDetectorProjection> distortedFocalToPixel(const LineScanOptics& optics,
                                                                           LineScanPixelConvention convention,
                                                                           const FocalPlaneCoordinate& focal)
        {
            if (!std::isfinite(focal.xMillimeters) || !std::isfinite(focal.yMillimeters))
            {
                return EvaluationResult<LineScanDetectorProjection>::failure(
                    CameraErrorCode::InvalidArgument, "line-scan focal coordinate must be finite");
            }

            double csm_sample = 0.0;
            double line_residual = 0.0;
            if (!optics.detectorGeometry)
            {
                csm_sample = optics.principalSample + focal.xMillimeters / optics.samplePitchMillimeters;
                line_residual = focal.yMillimeters / optics.samplePitchMillimeters;
            }
            else
            {
                const LineScanDetectorGeometry& detector = *optics.detectorGeometry;
                const double detector_line = detector.focalToPixelLines[0] +
                                             detector.focalToPixelLines[1] * focal.xMillimeters +
                                             detector.focalToPixelLines[2] * focal.yMillimeters;
                const double detector_sample = detector.focalToPixelSamples[0] +
                                               detector.focalToPixelSamples[1] * focal.xMillimeters +
                                               detector.focalToPixelSamples[2] * focal.yMillimeters;
                csm_sample = (detector_sample + detector.detectorSampleOrigin - detector.startingDetectorSample) /
                             detector.detectorSampleSumming;
                line_residual = (detector_line + detector.detectorLineOrigin - detector.startingDetectorLine) /
                                detector.detectorLineSumming;
            }

            const LineScanDetectorProjection projection{csmToCoordinate(csm_sample, convention), line_residual};
            if (!std::isfinite(projection.sample) || !std::isfinite(projection.lineResidualPixels))
            {
                return EvaluationResult<LineScanDetectorProjection>::failure(
                    CameraErrorCode::OutsideModelDomain,
                    "line-scan detector projection produced a non-finite coordinate");
            }
            return EvaluationResult<LineScanDetectorProjection>::success(projection);
        }

        EvaluationResult<FocalPlaneCoordinate> completePixelToUndistortedFocal(const LineScanOptics& optics,
                                                                               LineScanPixelConvention convention,
                                                                               double sample,
                                                                               const EvaluationOptions& options)
        {
            const MetashapeCalibration& calibration = *optics.completeCalibration;
            CalibratedPixel pixel{coordinateToCompleteCalibration(sample, convention), calibration.cy};
            if (optics.detectorGeometry)
            {
                const auto centered = pixelToDistortedFocal(optics, convention, sample);
                if (!centered)
                {
                    return centered;
                }
                pixel.sample = calibration.cx + centered.value().xMillimeters / optics.samplePitchMillimeters;
                pixel.line = calibration.cy + centered.value().yMillimeters / optics.samplePitchMillimeters;
            }

            const std::array<double, 2> target = calibratedPlaneFromPixel(calibration, pixel);
            double x = target[0];
            double y = target[1];
            double achieved_precision = 0.0;
            for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
            {
                double distorted_x = 0.0;
                double distorted_y = 0.0;
                applyCompleteDistortion(calibration, x, y, &distorted_x, &distorted_y);
                const CalibratedPixel current = calibratedPixelFromPlane(calibration, distorted_x, distorted_y);
                const CalibratedPixel target_pixel = calibratedPixelFromPlane(calibration, target[0], target[1]);
                const double residual_sample = current.sample - target_pixel.sample;
                const double residual_line = current.line - target_pixel.line;
                achieved_precision = std::max(std::abs(residual_sample), std::abs(residual_line));
                if (achieved_precision <= options.desiredPrecisionPixels)
                {
                    return EvaluationResult<FocalPlaneCoordinate>::success(
                        {x * optics.focalLengthMillimeters, y * optics.focalLengthMillimeters}, achieved_precision);
                }

                constexpr double step = 1.0e-7;
                double shifted_x = 0.0;
                double shifted_y = 0.0;
                applyCompleteDistortion(calibration, x + step, y, &shifted_x, &shifted_y);
                const CalibratedPixel shifted_pixel_x = calibratedPixelFromPlane(calibration, shifted_x, shifted_y);
                const double j00 = (shifted_pixel_x.sample - current.sample) / step;
                const double j10 = (shifted_pixel_x.line - current.line) / step;
                applyCompleteDistortion(calibration, x, y + step, &shifted_x, &shifted_y);
                const CalibratedPixel shifted_pixel_y = calibratedPixelFromPlane(calibration, shifted_x, shifted_y);
                const double j01 = (shifted_pixel_y.sample - current.sample) / step;
                const double j11 = (shifted_pixel_y.line - current.line) / step;
                const double determinant = j00 * j11 - j01 * j10;
                if (!std::isfinite(determinant) || std::abs(determinant) < 1.0e-15)
                {
                    return EvaluationResult<FocalPlaneCoordinate>::failure(
                        CameraErrorCode::NonConvergence,
                        "complete line-scan distortion inverse has a singular numerical Jacobian");
                }
                x -= (j11 * residual_sample - j01 * residual_line) / determinant;
                y -= (-j10 * residual_sample + j00 * residual_line) / determinant;
                if (!std::isfinite(x) || !std::isfinite(y))
                {
                    break;
                }
            }
            return EvaluationResult<FocalPlaneCoordinate>::failure(
                CameraErrorCode::NonConvergence,
                "complete line-scan distortion inverse did not reach the requested pixel precision");
        }

        EvaluationResult<LineScanDetectorProjection> completeUndistortedFocalToPixel(const LineScanOptics& optics,
                                                                                     LineScanPixelConvention convention,
                                                                                     const FocalPlaneCoordinate& focal)
        {
            const MetashapeCalibration& calibration = *optics.completeCalibration;
            const double x = focal.xMillimeters / optics.focalLengthMillimeters;
            const double y = focal.yMillimeters / optics.focalLengthMillimeters;
            double distorted_x = 0.0;
            double distorted_y = 0.0;
            applyCompleteDistortion(calibration, x, y, &distorted_x, &distorted_y);
            const CalibratedPixel pixel = calibratedPixelFromPlane(calibration, distorted_x, distorted_y);
            if (!std::isfinite(pixel.sample) || !std::isfinite(pixel.line))
            {
                return EvaluationResult<LineScanDetectorProjection>::failure(
                    CameraErrorCode::OutsideModelDomain,
                    "complete line-scan calibration produced a non-finite coordinate");
            }
            if (!optics.detectorGeometry)
            {
                return EvaluationResult<LineScanDetectorProjection>::success(
                    {completeCalibrationToCoordinate(pixel.sample, convention), pixel.line - calibration.cy});
            }
            const FocalPlaneCoordinate centered{(pixel.sample - calibration.cx) * optics.samplePitchMillimeters,
                                                (pixel.line - calibration.cy) * optics.samplePitchMillimeters};
            return distortedFocalToPixel(optics, convention, centered);
        }

        EvaluationResult<FocalPlaneCoordinate> removeDistortion(const LineScanOptics& optics,
                                                                const FocalPlaneCoordinate& distorted,
                                                                const EvaluationOptions& options)
        {
            if (optics.distortionModel == LineScanDistortionModel::LroNacFocalPlane)
            {
                const double denominator = 1.0 + optics.distortionK1 * distorted.yMillimeters * distorted.yMillimeters;
                if (std::abs(denominator) < 1.0e-15)
                {
                    return EvaluationResult<FocalPlaneCoordinate>::failure(
                        CameraErrorCode::OutsideModelDomain, "line-scan focal-plane distortion is singular");
                }
                return EvaluationResult<FocalPlaneCoordinate>::success(
                    {distorted.xMillimeters, distorted.yMillimeters / denominator});
            }

            double x = distorted.xMillimeters / optics.focalLengthMillimeters;
            double y = distorted.yMillimeters / optics.focalLengthMillimeters;
            const double target_x = x;
            const double target_y = y;
            for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
            {
                const double radial = 1.0 + optics.distortionK1 * (x * x + y * y);
                if (std::abs(radial) < 1.0e-15 || !std::isfinite(radial))
                {
                    break;
                }
                const double next_x = target_x / radial;
                const double next_y = target_y / radial;
                if (std::abs(next_x - x) + std::abs(next_y - y) <= 1.0e-14)
                {
                    return EvaluationResult<FocalPlaneCoordinate>::success(
                        {next_x * optics.focalLengthMillimeters, next_y * optics.focalLengthMillimeters});
                }
                x = next_x;
                y = next_y;
            }
            return EvaluationResult<FocalPlaneCoordinate>::failure(
                CameraErrorCode::NonConvergence, "line-scan radial distortion inverse did not converge");
        }

        EvaluationResult<FocalPlaneCoordinate> applyDistortion(const LineScanOptics& optics,
                                                               const FocalPlaneCoordinate& undistorted,
                                                               const EvaluationOptions& options)
        {
            if (optics.distortionModel == LineScanDistortionModel::LroNacFocalPlane)
            {
                if (std::abs(undistorted.yMillimeters) > 40.0)
                {
                    return EvaluationResult<FocalPlaneCoordinate>::failure(
                        CameraErrorCode::OutsideModelDomain,
                        "line-scan LRO NAC focal coordinate lies outside the supported distortion domain");
                }
                double y = undistorted.yMillimeters;
                for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
                {
                    const double next = undistorted.yMillimeters * (1.0 + optics.distortionK1 * y * y);
                    if (!std::isfinite(next))
                    {
                        break;
                    }
                    if (std::abs(next - y) <= 1.0e-10)
                    {
                        return EvaluationResult<FocalPlaneCoordinate>::success({undistorted.xMillimeters, next});
                    }
                    y = next;
                }
                return EvaluationResult<FocalPlaneCoordinate>::failure(CameraErrorCode::NonConvergence,
                                                                       "line-scan LRO NAC distortion did not converge");
            }

            const double x = undistorted.xMillimeters / optics.focalLengthMillimeters;
            const double y = undistorted.yMillimeters / optics.focalLengthMillimeters;
            const double radial = 1.0 + optics.distortionK1 * (x * x + y * y);
            const FocalPlaneCoordinate distorted{undistorted.xMillimeters * radial, undistorted.yMillimeters * radial};
            if (!std::isfinite(distorted.xMillimeters) || !std::isfinite(distorted.yMillimeters))
            {
                return EvaluationResult<FocalPlaneCoordinate>::failure(
                    CameraErrorCode::OutsideModelDomain,
                    "line-scan radial distortion produced a non-finite focal coordinate");
            }
            return EvaluationResult<FocalPlaneCoordinate>::success(distorted);
        }

    } // namespace

} // namespace placamera

namespace placamera::internal
{

    EvaluationResult<FocalPlaneCoordinate> lineScanPixelToUndistortedFocal(const LineScanOptics& optics,
                                                                           LineScanPixelConvention convention,
                                                                           double sample,
                                                                           const EvaluationOptions& options)
    {
        if (!validOptions(options))
        {
            return EvaluationResult<FocalPlaneCoordinate>::failure(
                CameraErrorCode::InvalidArgument, "line-scan evaluation options must be finite and positive");
        }
        if (optics.completeCalibration)
        {
            return completePixelToUndistortedFocal(optics, convention, sample, options);
        }
        const auto distorted = pixelToDistortedFocal(optics, convention, sample);
        if (!distorted)
        {
            return distorted;
        }
        return removeDistortion(optics, distorted.value(), options);
    }

    EvaluationResult<LineScanDetectorProjection> lineScanUndistortedFocalToPixel(const LineScanOptics& optics,
                                                                                 LineScanPixelConvention convention,
                                                                                 const FocalPlaneCoordinate& focal,
                                                                                 const EvaluationOptions& options)
    {
        if (!validOptions(options))
        {
            return EvaluationResult<LineScanDetectorProjection>::failure(
                CameraErrorCode::InvalidArgument, "line-scan evaluation options must be finite and positive");
        }
        if (optics.completeCalibration)
        {
            return completeUndistortedFocalToPixel(optics, convention, focal);
        }
        const auto distorted = applyDistortion(optics, focal, options);
        if (!distorted)
        {
            return EvaluationResult<LineScanDetectorProjection>::failure(distorted.errorCode(), distorted.message());
        }
        return distortedFocalToPixel(optics, convention, distorted.value());
    }

} // namespace placamera::internal

namespace placamera
{

    EvaluationResult<FocalPlaneCoordinate> lineScanPixelToUndistortedFocal(const LineScanDefinition& definition,
                                                                           double sample,
                                                                           const EvaluationOptions& options)
    {
        return internal::lineScanPixelToUndistortedFocal(
            definition.optics(), definition.pixelConvention(), sample, options);
    }

    EvaluationResult<LineScanDetectorProjection> lineScanUndistortedFocalToPixel(const LineScanDefinition& definition,
                                                                                 const FocalPlaneCoordinate& focal,
                                                                                 const EvaluationOptions& options)
    {
        return internal::lineScanUndistortedFocalToPixel(
            definition.optics(), definition.pixelConvention(), focal, options);
    }

} // namespace placamera
