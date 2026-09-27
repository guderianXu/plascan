#include "rpc_math.h"

#include <algorithm>
#include <cmath>

namespace placamera
{

    namespace
    {

        constexpr double kDenominatorEpsilon = 1.0e-14;
        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        RpcCoefficients polynomialTerms(double longitude, double latitude, double height) noexcept
        {
            const double longitude2 = longitude * longitude;
            const double latitude2 = latitude * latitude;
            const double height2 = height * height;
            return {{1.0,
                     longitude,
                     latitude,
                     height,
                     longitude * latitude,
                     longitude * height,
                     latitude * height,
                     longitude2,
                     latitude2,
                     height2,
                     longitude * latitude * height,
                     longitude2 * longitude,
                     longitude * latitude2,
                     longitude * height2,
                     longitude2 * latitude,
                     latitude2 * latitude,
                     latitude * height2,
                     longitude2 * height,
                     latitude2 * height,
                     height2 * height}};
        }

        double dot(const RpcCoefficients& first, const RpcCoefficients& second) noexcept
        {
            double result = 0.0;
            for (std::size_t index = 0; index < first.size(); ++index)
            {
                result += first[index] * second[index];
            }
            return result;
        }

        double longitudeDifference(double longitude, double reference) noexcept
        {
            double difference = longitude - reference;
            if (difference < -270.0)
            {
                difference += 360.0;
            }
            else if (difference > 270.0)
            {
                difference -= 360.0;
            }
            return difference;
        }

        ImageCoordinate applyImageCorrection(const ImageCoordinate& uncorrected,
                                             const RpcParameters& parameters,
                                             const RpcImageCorrection& correction) noexcept
        {
            const double normalized_sample = (uncorrected.sample - parameters.sampleOffset) / parameters.sampleScale;
            const double normalized_line = (uncorrected.line - parameters.lineOffset) / parameters.lineScale;
            return {uncorrected.sample + correction.sampleOffsetPixels +
                        correction.sampleSamplePixels * normalized_sample +
                        correction.sampleLinePixels * normalized_line,
                    uncorrected.line + correction.lineOffsetPixels + correction.lineSamplePixels * normalized_sample +
                        correction.lineLinePixels * normalized_line};
        }

        ImageCoordinate applyGroundCorrection(const ImageCoordinate& uncorrected,
                                              const RpcParameters& parameters,
                                              const GeodeticCoordinate& ground,
                                              const RpcGroundCorrection& correction) noexcept
        {
            const double longitude_delta = ground.longitudeDegrees - parameters.longitudeOffset;
            const double latitude_delta = ground.latitudeDegrees - parameters.latitudeOffset;
            const double height_delta = ground.heightMeters - parameters.heightOffset;
            return {uncorrected.sample + correction.sampleOffsetPixels +
                        correction.sampleLongitudePixelsPerDegree * longitude_delta +
                        correction.sampleLatitudePixelsPerDegree * latitude_delta +
                        correction.sampleHeightPixelsPerMeter * height_delta,
                    uncorrected.line + correction.lineOffsetPixels +
                        correction.lineLongitudePixelsPerDegree * longitude_delta +
                        correction.lineLatitudePixelsPerDegree * latitude_delta +
                        correction.lineHeightPixelsPerMeter * height_delta};
        }

        bool evaluate(const RpcDefinition& definition,
                      const RpcCorrection& correction,
                      double normalizedLongitude,
                      double normalizedLatitude,
                      double normalizedHeight,
                      bool applyCorrection,
                      ImageCoordinate* image) noexcept
        {
            if (!image || !finite(normalizedLongitude) || !finite(normalizedLatitude) || !finite(normalizedHeight))
            {
                return false;
            }

            const RpcParameters& parameters = definition.parameters();
            const RpcCoefficients terms = polynomialTerms(normalizedLongitude, normalizedLatitude, normalizedHeight);
            const double line_denominator = dot(parameters.lineDenominator, terms);
            const double sample_denominator = dot(parameters.sampleDenominator, terms);
            if (std::abs(line_denominator) < kDenominatorEpsilon || std::abs(sample_denominator) < kDenominatorEpsilon)
            {
                return false;
            }

            const ImageCoordinate uncorrected{
                parameters.sampleOffset +
                    parameters.sampleScale * dot(parameters.sampleNumerator, terms) / sample_denominator,
                parameters.lineOffset + parameters.lineScale * dot(parameters.lineNumerator, terms) / line_denominator};
            if (!finite(uncorrected.sample) || !finite(uncorrected.line))
            {
                return false;
            }
            if (!applyCorrection)
            {
                *image = uncorrected;
            }
            else if (const RpcImageCorrection* image_correction = correction.normalizedImageValue())
            {
                *image = applyImageCorrection(uncorrected, parameters, *image_correction);
            }
            else
            {
                const GeodeticCoordinate ground{
                    parameters.longitudeOffset + normalizedLongitude * parameters.longitudeScale,
                    parameters.latitudeOffset + normalizedLatitude * parameters.latitudeScale,
                    parameters.heightOffset + normalizedHeight * parameters.heightScale};
                *image = applyGroundCorrection(uncorrected, parameters, ground, *correction.groundCoordinateValue());
            }
            return finite(image->sample) && finite(image->line);
        }

    } // namespace

    namespace internal
    {

        EvaluationResult<ImageCoordinate> projectRpc(const RpcDefinition& definition,
                                                     const RpcCorrection& correction,
                                                     const GeodeticCoordinate& ground,
                                                     bool applyCorrection)
        {
            if (!finite(ground.longitudeDegrees) || !finite(ground.latitudeDegrees) || !finite(ground.heightMeters) ||
                ground.latitudeDegrees < -90.0 || ground.latitudeDegrees > 90.0)
            {
                return EvaluationResult<ImageCoordinate>::failure(CameraErrorCode::InvalidArgument,
                                                                  "RPC geodetic coordinate is invalid");
            }

            const RpcParameters& parameters = definition.parameters();
            const double normalized_longitude =
                longitudeDifference(ground.longitudeDegrees, parameters.longitudeOffset) / parameters.longitudeScale;
            const double normalized_latitude =
                (ground.latitudeDegrees - parameters.latitudeOffset) / parameters.latitudeScale;
            const double normalized_height = (ground.heightMeters - parameters.heightOffset) / parameters.heightScale;
            ImageCoordinate image;
            if (!evaluate(definition,
                          correction,
                          normalized_longitude,
                          normalized_latitude,
                          normalized_height,
                          applyCorrection,
                          &image))
            {
                return EvaluationResult<ImageCoordinate>::failure(
                    CameraErrorCode::OutsideModelDomain,
                    "RPC polynomial is singular or non-finite at the requested coordinate");
            }
            return EvaluationResult<ImageCoordinate>::success(image);
        }

        EvaluationResult<GeodeticCoordinate> invertRpcAtHeight(const RpcDefinition& definition,
                                                               const RpcCorrection& correction,
                                                               const ImageCoordinate& image,
                                                               double ellipsoidalHeightMeters,
                                                               const EvaluationOptions& options)
        {
            if (!finite(image.sample) || !finite(image.line) || !finite(ellipsoidalHeightMeters) ||
                !finite(options.desiredPrecisionPixels) || options.desiredPrecisionPixels <= 0.0 ||
                options.maximumIterations <= 0)
            {
                return EvaluationResult<GeodeticCoordinate>::failure(
                    CameraErrorCode::InvalidArgument,
                    "RPC inverse inputs and evaluation options must be finite and positive");
            }

            const RpcParameters& parameters = definition.parameters();
            const double normalized_height =
                (ellipsoidalHeightMeters - parameters.heightOffset) / parameters.heightScale;
            double normalized_longitude = 0.0;
            double normalized_latitude = 0.0;
            constexpr double derivative_step = 1.0e-6;

            for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
            {
                ImageCoordinate current;
                if (!evaluate(definition,
                              correction,
                              normalized_longitude,
                              normalized_latitude,
                              normalized_height,
                              true,
                              &current))
                {
                    break;
                }
                const double sample_residual = image.sample - current.sample;
                const double line_residual = image.line - current.line;
                const double current_error = std::hypot(sample_residual, line_residual);
                if (current_error <= options.desiredPrecisionPixels)
                {
                    const GeodeticCoordinate ground{
                        parameters.longitudeOffset + normalized_longitude * parameters.longitudeScale,
                        parameters.latitudeOffset + normalized_latitude * parameters.latitudeScale,
                        ellipsoidalHeightMeters};
                    if (ground.latitudeDegrees < -90.0 || ground.latitudeDegrees > 90.0)
                    {
                        break;
                    }
                    return EvaluationResult<GeodeticCoordinate>::success(ground, current_error);
                }

                ImageCoordinate longitude_plus;
                ImageCoordinate longitude_minus;
                ImageCoordinate latitude_plus;
                ImageCoordinate latitude_minus;
                if (!evaluate(definition,
                              correction,
                              normalized_longitude + derivative_step,
                              normalized_latitude,
                              normalized_height,
                              true,
                              &longitude_plus) ||
                    !evaluate(definition,
                              correction,
                              normalized_longitude - derivative_step,
                              normalized_latitude,
                              normalized_height,
                              true,
                              &longitude_minus) ||
                    !evaluate(definition,
                              correction,
                              normalized_longitude,
                              normalized_latitude + derivative_step,
                              normalized_height,
                              true,
                              &latitude_plus) ||
                    !evaluate(definition,
                              correction,
                              normalized_longitude,
                              normalized_latitude - derivative_step,
                              normalized_height,
                              true,
                              &latitude_minus))
                {
                    break;
                }

                const double d_sample_d_longitude =
                    (longitude_plus.sample - longitude_minus.sample) / (2.0 * derivative_step);
                const double d_line_d_longitude =
                    (longitude_plus.line - longitude_minus.line) / (2.0 * derivative_step);
                const double d_sample_d_latitude =
                    (latitude_plus.sample - latitude_minus.sample) / (2.0 * derivative_step);
                const double d_line_d_latitude = (latitude_plus.line - latitude_minus.line) / (2.0 * derivative_step);
                const double determinant =
                    d_sample_d_longitude * d_line_d_latitude - d_sample_d_latitude * d_line_d_longitude;
                if (!finite(determinant) || std::abs(determinant) < 1.0e-12)
                {
                    break;
                }

                const double longitude_update =
                    (sample_residual * d_line_d_latitude - line_residual * d_sample_d_latitude) / determinant;
                const double latitude_update =
                    (d_sample_d_longitude * line_residual - d_line_d_longitude * sample_residual) / determinant;
                bool accepted = false;
                double step_scale = 1.0;
                for (int line_search = 0; line_search < 10; ++line_search)
                {
                    const double trial_longitude = normalized_longitude + step_scale * longitude_update;
                    const double trial_latitude = normalized_latitude + step_scale * latitude_update;
                    ImageCoordinate trial;
                    if (evaluate(
                            definition, correction, trial_longitude, trial_latitude, normalized_height, true, &trial) &&
                        std::hypot(image.sample - trial.sample, image.line - trial.line) < current_error)
                    {
                        normalized_longitude = trial_longitude;
                        normalized_latitude = trial_latitude;
                        accepted = true;
                        break;
                    }
                    step_scale *= 0.5;
                }
                if (!accepted || !finite(normalized_longitude) || !finite(normalized_latitude))
                {
                    break;
                }
            }

            return EvaluationResult<GeodeticCoordinate>::failure(
                CameraErrorCode::NonConvergence, "RPC inverse did not reach the requested pixel precision");
        }

    } // namespace internal

} // namespace placamera
