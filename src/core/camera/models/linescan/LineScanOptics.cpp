#include "LineScanOptics.h"

#include <cmath>

namespace xjw::camera_models::linescan
{
    namespace
    {

        double coordinateToCsm(double coordinate, PixelConvention convention)
        {
            return convention == PixelConvention::ZeroBased ? coordinate + 0.5 : coordinate;
        }

        double csmToCoordinate(double coordinate, PixelConvention convention)
        {
            return convention == PixelConvention::ZeroBased ? coordinate - 0.5 : coordinate;
        }

        bool pixelToDistortedFocal(const LineScanDefinition& definition,
                                   double sample,
                                   FocalPlaneCoordinate* focal)
        {
            if (!focal || !std::isfinite(sample))
            {
                return false;
            }
            const LineScanOptics& optics = definition.optics();
            const double csmSample = coordinateToCsm(sample, definition.pixelConvention());
            if (!optics.detectorGeometry)
            {
                *focal = {(csmSample - optics.principalSample) * optics.samplePitchMillimeters, 0.0};
                return std::isfinite(focal->xMillimeters);
            }
            const LineScanDetectorGeometry& detector = *optics.detectorGeometry;
            const double detectorSample = csmSample * detector.detectorSampleSumming +
                                          detector.startingDetectorSample;
            const double lineOffset = detector.startingDetectorLine - detector.detectorLineOrigin -
                                      detector.focalToPixelLines[0];
            const double sampleOffset = detectorSample - detector.detectorSampleOrigin -
                                        detector.focalToPixelSamples[0];
            const double determinant = detector.focalToPixelLines[1] * detector.focalToPixelSamples[2] -
                                       detector.focalToPixelLines[2] * detector.focalToPixelSamples[1];
            if (std::abs(determinant) < 1.0e-15)
            {
                return false;
            }
            focal->xMillimeters = (detector.focalToPixelSamples[2] * lineOffset -
                                   detector.focalToPixelLines[2] * sampleOffset) /
                                  determinant;
            focal->yMillimeters = (-detector.focalToPixelSamples[1] * lineOffset +
                                   detector.focalToPixelLines[1] * sampleOffset) /
                                  determinant;
            return std::isfinite(focal->xMillimeters) && std::isfinite(focal->yMillimeters);
        }

        bool distortedFocalToPixel(const LineScanDefinition& definition,
                                   const FocalPlaneCoordinate& focal,
                                   double* sample,
                                   double* lineResidual)
        {
            if (!sample || !lineResidual || !std::isfinite(focal.xMillimeters) ||
                !std::isfinite(focal.yMillimeters))
            {
                return false;
            }
            const LineScanOptics& optics = definition.optics();
            double csmSample = 0.0;
            if (!optics.detectorGeometry)
            {
                csmSample = optics.principalSample + focal.xMillimeters / optics.samplePitchMillimeters;
                *lineResidual = focal.yMillimeters / optics.samplePitchMillimeters;
            }
            else
            {
                const LineScanDetectorGeometry& detector = *optics.detectorGeometry;
                const double detectorLine = detector.focalToPixelLines[0] +
                                            detector.focalToPixelLines[1] * focal.xMillimeters +
                                            detector.focalToPixelLines[2] * focal.yMillimeters;
                const double detectorSample = detector.focalToPixelSamples[0] +
                                              detector.focalToPixelSamples[1] * focal.xMillimeters +
                                              detector.focalToPixelSamples[2] * focal.yMillimeters;
                csmSample = (detectorSample + detector.detectorSampleOrigin - detector.startingDetectorSample) /
                            detector.detectorSampleSumming;
                *lineResidual = (detectorLine + detector.detectorLineOrigin - detector.startingDetectorLine) /
                                detector.detectorLineSumming;
            }
            *sample = csmToCoordinate(csmSample, definition.pixelConvention());
            return std::isfinite(*sample) && std::isfinite(*lineResidual);
        }

        bool removeDistortion(const LineScanOptics& optics,
                              const FocalPlaneCoordinate& distorted,
                              FocalPlaneCoordinate* undistorted)
        {
            if (!undistorted || !std::isfinite(distorted.xMillimeters) || !std::isfinite(distorted.yMillimeters))
            {
                return false;
            }
            if (optics.distortionModel == LineScanDistortionModel::LroNacFocalPlane)
            {
                const double denominator =
                    1.0 + optics.distortionK1 * distorted.yMillimeters * distorted.yMillimeters;
                if (std::abs(denominator) < 1.0e-15)
                {
                    return false;
                }
                *undistorted = {distorted.xMillimeters, distorted.yMillimeters / denominator};
                return std::isfinite(undistorted->yMillimeters);
            }
            double x = distorted.xMillimeters / optics.focalLengthMillimeters;
            double y = distorted.yMillimeters / optics.focalLengthMillimeters;
            const double targetX = x;
            const double targetY = y;
            for (int iteration = 0; iteration < 30; ++iteration)
            {
                const double radial = 1.0 + optics.distortionK1 * (x * x + y * y);
                if (std::abs(radial) < 1.0e-15 || !std::isfinite(radial))
                {
                    return false;
                }
                const double nextX = targetX / radial;
                const double nextY = targetY / radial;
                if (std::abs(nextX - x) + std::abs(nextY - y) <= 1.0e-14)
                {
                    *undistorted = {nextX * optics.focalLengthMillimeters,
                                    nextY * optics.focalLengthMillimeters};
                    return true;
                }
                x = nextX;
                y = nextY;
            }
            return false;
        }

        bool applyDistortion(const LineScanOptics& optics,
                             const FocalPlaneCoordinate& undistorted,
                             FocalPlaneCoordinate* distorted)
        {
            if (!distorted || !std::isfinite(undistorted.xMillimeters) ||
                !std::isfinite(undistorted.yMillimeters))
            {
                return false;
            }
            if (optics.distortionModel == LineScanDistortionModel::LroNacFocalPlane)
            {
                if (std::abs(undistorted.yMillimeters) > 40.0)
                {
                    return false;
                }
                double y = undistorted.yMillimeters;
                for (int iteration = 0; iteration < 50; ++iteration)
                {
                    const double next =
                        undistorted.yMillimeters * (1.0 + optics.distortionK1 * y * y);
                    if (!std::isfinite(next))
                    {
                        return false;
                    }
                    if (std::abs(next - y) <= 1.0e-10)
                    {
                        *distorted = {undistorted.xMillimeters, next};
                        return true;
                    }
                    y = next;
                }
                return false;
            }
            const double x = undistorted.xMillimeters / optics.focalLengthMillimeters;
            const double y = undistorted.yMillimeters / optics.focalLengthMillimeters;
            const double radial = 1.0 + optics.distortionK1 * (x * x + y * y);
            *distorted = {undistorted.xMillimeters * radial, undistorted.yMillimeters * radial};
            return std::isfinite(distorted->xMillimeters) && std::isfinite(distorted->yMillimeters);
        }

    } // namespace

    bool LineScanOpticsTransform::pixelToUndistortedFocal(const LineScanDefinition& definition,
                                                          double sample,
                                                          FocalPlaneCoordinate* focal)
    {
        FocalPlaneCoordinate distorted;
        return pixelToDistortedFocal(definition, sample, &distorted) &&
               removeDistortion(definition.optics(), distorted, focal);
    }

    bool LineScanOpticsTransform::undistortedFocalToPixel(const LineScanDefinition& definition,
                                                          const FocalPlaneCoordinate& focal,
                                                          double* sample,
                                                          double* detectorLineResidualPixels)
    {
        FocalPlaneCoordinate distorted;
        return applyDistortion(definition.optics(), focal, &distorted) &&
               distortedFocalToPixel(definition, distorted, sample, detectorLineResidualPixels);
    }

} // namespace xjw::camera_models::linescan
