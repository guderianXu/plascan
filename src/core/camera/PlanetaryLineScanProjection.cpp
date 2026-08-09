#include "PlanetaryLineScanCamera.h"

#include <algorithm>
#include <cmath>

namespace xjw
{
namespace
{

using Matrix3 = PlanetaryLineScanCamera::Matrix3;
using Vector3 = PlanetaryLineScanCamera::Vector3;

double csmCoordinate(double coordinate, PlanetaryLineScanCamera::PixelConvention convention)
{
    return convention == PlanetaryLineScanCamera::PixelConvention::OpenCvZeroBased
               ? coordinate + 0.5
               : coordinate;
}

double requestedCoordinate(double csm_coordinate, PlanetaryLineScanCamera::PixelConvention convention)
{
    return convention == PlanetaryLineScanCamera::PixelConvention::OpenCvZeroBased
               ? csm_coordinate - 0.5
               : csm_coordinate;
}

Vector3 multiply(const Matrix3 &matrix, const Vector3 &vector)
{
    return {
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]
    };
}

Vector3 transposeMultiply(const Matrix3 &matrix, const Vector3 &vector)
{
    return {
        matrix[0] * vector[0] + matrix[3] * vector[1] + matrix[6] * vector[2],
        matrix[1] * vector[0] + matrix[4] * vector[1] + matrix[7] * vector[2],
        matrix[2] * vector[0] + matrix[5] * vector[1] + matrix[8] * vector[2]
    };
}

bool normalize(Vector3 *vector)
{
    const double length = std::sqrt((*vector)[0] * (*vector)[0]
                                  + (*vector)[1] * (*vector)[1]
                                  + (*vector)[2] * (*vector)[2]);
    if (!(length > 0.0) || !std::isfinite(length))
    {
        return false;
    }
    for (double &value : *vector)
    {
        value /= length;
    }
    return true;
}

} // namespace

bool PlanetaryLineScanCamera::pixelToDistortedFocalPlane(
    double sample,
    PixelConvention convention,
    std::array<double, 2> *focalMillimeters) const
{
    if (!_isLoaded || focalMillimeters == nullptr || !std::isfinite(sample))
    {
        return false;
    }
    const double detector_sample = csmCoordinate(sample, convention) * _detectorSampleSumming
                                 + _startingDetectorSample;
    const double detector_line = _startingDetectorLine;
    const double matrix_11 = _focalToPixelLines[1];
    const double matrix_12 = _focalToPixelLines[2];
    const double matrix_21 = _focalToPixelSamples[1];
    const double matrix_22 = _focalToPixelSamples[2];
    const double determinant = matrix_11 * matrix_22 - matrix_12 * matrix_21;
    if (std::abs(determinant) < 1.0e-15)
    {
        return false;
    }
    const double line_offset = detector_line - _detectorLineOrigin - _focalToPixelLines[0];
    const double sample_offset = detector_sample - _detectorSampleOrigin - _focalToPixelSamples[0];
    (*focalMillimeters)[0] = (matrix_22 * line_offset - matrix_12 * sample_offset) / determinant;
    (*focalMillimeters)[1] = (-matrix_21 * line_offset + matrix_11 * sample_offset) / determinant;
    return std::isfinite((*focalMillimeters)[0]) && std::isfinite((*focalMillimeters)[1]);
}

bool PlanetaryLineScanCamera::distortedFocalPlaneToPixel(
    const std::array<double, 2> &focalMillimeters,
    PixelConvention convention,
    double *sample,
    double *detectorLineOffsetPixels) const
{
    if (!_isLoaded || sample == nullptr || detectorLineOffsetPixels == nullptr
        || !std::isfinite(focalMillimeters[0]) || !std::isfinite(focalMillimeters[1]))
    {
        return false;
    }
    const double detector_line = _focalToPixelLines[0]
        + _focalToPixelLines[1] * focalMillimeters[0]
        + _focalToPixelLines[2] * focalMillimeters[1];
    const double detector_sample = _focalToPixelSamples[0]
        + _focalToPixelSamples[1] * focalMillimeters[0]
        + _focalToPixelSamples[2] * focalMillimeters[1];
    const double sample_csm = (detector_sample + _detectorSampleOrigin - _startingDetectorSample)
                            / _detectorSampleSumming;
    *sample = requestedCoordinate(sample_csm, convention);
    *detectorLineOffsetPixels =
        (detector_line + _detectorLineOrigin - _startingDetectorLine) / _detectorLineSumming;
    return std::isfinite(*sample) && std::isfinite(*detectorLineOffsetPixels);
}

bool PlanetaryLineScanCamera::removeOpticalDistortion(
    const std::array<double, 2> &distortedMillimeters,
    std::array<double, 2> *undistortedMillimeters) const
{
    if (!_isLoaded || undistortedMillimeters == nullptr
        || !std::isfinite(distortedMillimeters[0]) || !std::isfinite(distortedMillimeters[1]))
    {
        return false;
    }
    const double denominator = 1.0 + _lroNacDistortionK1
        * distortedMillimeters[1] * distortedMillimeters[1];
    if (std::abs(denominator) < 1.0e-15)
    {
        return false;
    }
    (*undistortedMillimeters)[0] = distortedMillimeters[0];
    (*undistortedMillimeters)[1] = distortedMillimeters[1] / denominator;
    return std::isfinite((*undistortedMillimeters)[1]);
}

bool PlanetaryLineScanCamera::applyOpticalDistortion(
    const std::array<double, 2> &undistortedMillimeters,
    std::array<double, 2> *distortedMillimeters) const
{
    if (!_isLoaded || distortedMillimeters == nullptr
        || !std::isfinite(undistortedMillimeters[0]) || !std::isfinite(undistortedMillimeters[1])
        || std::abs(undistortedMillimeters[1]) > 40.0)
    {
        return false;
    }
    double distorted_y = undistortedMillimeters[1];
    for (int iteration = 0; iteration < 50; ++iteration)
    {
        const double next = undistortedMillimeters[1]
            * (1.0 + _lroNacDistortionK1 * distorted_y * distorted_y);
        if (!std::isfinite(next))
        {
            return false;
        }
        if (std::abs(next - distorted_y) <= 1.0e-10)
        {
            (*distortedMillimeters)[0] = undistortedMillimeters[0];
            (*distortedMillimeters)[1] = next;
            return true;
        }
        distorted_y = next;
    }
    return false;
}

bool PlanetaryLineScanCamera::pixelRayBodyFixed(double sample,
                                                double line,
                                                PixelConvention convention,
                                                ImagingRay *ray,
                                                const PoseBias &bias) const
{
    if (ray == nullptr)
    {
        return false;
    }
    double ephemeris_time = 0.0;
    std::array<double, 2> distorted{};
    std::array<double, 2> undistorted{};
    Matrix3 sensor_to_body{};
    if (!absoluteEtForLine(line, convention, &ephemeris_time)
        || !pixelToDistortedFocalPlane(sample, convention, &distorted)
        || !removeOpticalDistortion(distorted, &undistorted)
        || !correctedPoseAtEt(ephemeris_time, bias, &ray->centerBodyFixedMeters, &sensor_to_body))
    {
        return false;
    }
    Vector3 sensor_direction{{undistorted[0], undistorted[1], _focalLengthMillimeters}};
    if (!normalize(&sensor_direction))
    {
        return false;
    }
    ray->directionBodyFixed = multiply(sensor_to_body, sensor_direction);
    ray->ephemerisTimeSeconds = ephemeris_time;
    return normalize(&ray->directionBodyFixed);
}

bool PlanetaryLineScanCamera::projectAtObservedLine(
    const Vector3 &groundBodyFixedMeters,
    double observedLine,
    PixelConvention convention,
    FixedLineProjection *projection,
    const PoseBias &bias) const
{
    if (projection == nullptr || !isLineInsideImage(observedLine, convention))
    {
        return false;
    }
    double ephemeris_time = 0.0;
    Vector3 center{};
    Matrix3 sensor_to_body{};
    if (!absoluteEtForLine(observedLine, convention, &ephemeris_time)
        || !correctedPoseAtEt(ephemeris_time, bias, &center, &sensor_to_body))
    {
        return false;
    }
    const Vector3 body_look{{groundBodyFixedMeters[0] - center[0],
                             groundBodyFixedMeters[1] - center[1],
                             groundBodyFixedMeters[2] - center[2]}};
    const Vector3 sensor_look = transposeMultiply(sensor_to_body, body_look);
    if (!(sensor_look[2] > 1.0e-9) || !std::isfinite(sensor_look[2]))
    {
        return false;
    }
    const std::array<double, 2> undistorted{{
        _focalLengthMillimeters * sensor_look[0] / sensor_look[2],
        _focalLengthMillimeters * sensor_look[1] / sensor_look[2]
    }};
    std::array<double, 2> distorted{};
    double predicted_sample = 0.0;
    double line_residual = 0.0;
    if (!applyOpticalDistortion(undistorted, &distorted)
        || !distortedFocalPlaneToPixel(distorted, convention,
                                       &predicted_sample, &line_residual))
    {
        return false;
    }
    projection->sample = predicted_sample;
    projection->line = observedLine;
    projection->detectorLineResidualPixels = line_residual;
    projection->undistortedFocalXMillimeters = undistorted[0];
    projection->undistortedFocalYMillimeters = undistorted[1];
    projection->sensorDepthMeters = sensor_look[2];
    projection->ephemerisTimeSeconds = ephemeris_time;
    return true;
}

bool PlanetaryLineScanCamera::groundToImage(const Vector3 &groundBodyFixedMeters,
                                            PixelConvention convention,
                                            ImageCoordinate *image,
                                            const GroundToImageOptions &options,
                                            const PoseBias &bias) const
{
    if (!_isLoaded || image == nullptr || options.maximumIterations < 1
        || !(options.desiredLinePrecisionPixels > 0.0)
        || !std::isfinite(options.desiredLinePrecisionPixels))
    {
        return false;
    }
    const double lower = convention == PixelConvention::CsmPixelCenter ? 0.5 : 0.0;
    const double upper = lower + static_cast<double>(_imageLines - 1);
    double line0 = 0.5 * (lower + upper);
    double line1 = std::min(upper, line0 + 0.1);
    FixedLineProjection projection0;
    FixedLineProjection projection1;
    if (!projectAtObservedLine(groundBodyFixedMeters, line0, convention, &projection0, bias)
        || !projectAtObservedLine(groundBodyFixedMeters, line1, convention, &projection1, bias))
    {
        return false;
    }
    for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
    {
        if (std::abs(projection1.detectorLineResidualPixels)
            <= options.desiredLinePrecisionPixels)
        {
            image->line = line1;
            image->sample = projection1.sample;
            if (!options.requireInsideImage)
            {
                return true;
            }
            const double sample_lower = convention == PixelConvention::CsmPixelCenter ? 0.5 : 0.0;
            return image->sample >= sample_lower
                && image->sample <= sample_lower + static_cast<double>(_imageSamples - 1);
        }
        const double denominator = projection1.detectorLineResidualPixels
                                 - projection0.detectorLineResidualPixels;
        if (std::abs(denominator) < 1.0e-15)
        {
            return false;
        }
        const double next_line = std::clamp(
            line1 - projection1.detectorLineResidualPixels * (line1 - line0) / denominator,
            lower, upper);
        line0 = line1;
        projection0 = projection1;
        line1 = next_line;
        if (!projectAtObservedLine(groundBodyFixedMeters, line1, convention, &projection1, bias))
        {
            return false;
        }
    }
    return false;
}

} // namespace xjw
