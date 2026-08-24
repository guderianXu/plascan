#include "RpcCameraModel.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{
    namespace
    {

        constexpr double kDenominatorEpsilon = 1.0e-14;

        bool finite(double value)
        {
            return std::isfinite(value);
        }

        bool finiteCoefficients(const RpcCameraModel::Coefficients& coefficients)
        {
            return std::all_of(coefficients.begin(), coefficients.end(), finite);
        }

        double longitudeDifference(double longitude, double reference)
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

        RpcCameraModel::Coefficients rpc00bTerms(double longitude, double latitude, double height)
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

        double dot(const RpcCameraModel::Coefficients& first, const RpcCameraModel::Coefficients& second)
        {
            double result = 0.0;
            for (std::size_t index = 0; index < first.size(); ++index)
            {
                result += first[index] * second[index];
            }
            return result;
        }

    } // namespace

    CameraModelType RpcCameraModel::modelType() const noexcept
    {
        return CameraModelType::RationalPolynomial;
    }

    std::string_view RpcCameraModel::worldFrameName() const noexcept
    {
        return "EPSG:4978";
    }

    bool RpcCameraModel::setParameters(const Parameters& parameters, std::string* errorMessage)
    {
        const std::array<double, 10> scalars{{parameters.lineOffset,
                                              parameters.sampleOffset,
                                              parameters.latitudeOffset,
                                              parameters.longitudeOffset,
                                              parameters.heightOffset,
                                              parameters.lineScale,
                                              parameters.sampleScale,
                                              parameters.latitudeScale,
                                              parameters.longitudeScale,
                                              parameters.heightScale}};
        const bool finite_scalars = std::all_of(scalars.begin(), scalars.end(), finite);
        const bool positive_scales = parameters.lineScale > 0.0 && parameters.sampleScale > 0.0 &&
                                     parameters.latitudeScale > 0.0 && parameters.longitudeScale > 0.0 &&
                                     parameters.heightScale > 0.0;
        const bool finite_coefficients =
            finiteCoefficients(parameters.lineNumerator) && finiteCoefficients(parameters.lineDenominator) &&
            finiteCoefficients(parameters.sampleNumerator) && finiteCoefficients(parameters.sampleDenominator);
        const bool valid_errors = (!parameters.errorBiasMeters ||
                                   (finite(*parameters.errorBiasMeters) && *parameters.errorBiasMeters >= 0.0)) &&
                                  (!parameters.errorRandomMeters ||
                                   (finite(*parameters.errorRandomMeters) && *parameters.errorRandomMeters >= 0.0));
        if (!finite_scalars || !positive_scales || !finite_coefficients || !valid_errors)
        {
            _isValid = false;
            if (errorMessage)
            {
                *errorMessage =
                    "RPC values must be finite, scales must be positive, and optional errors must be non-negative";
            }
            return false;
        }
        if (std::abs(parameters.lineDenominator[0]) < kDenominatorEpsilon ||
            std::abs(parameters.sampleDenominator[0]) < kDenominatorEpsilon)
        {
            _isValid = false;
            if (errorMessage)
            {
                *errorMessage = "RPC denominator is singular at its normalization origin";
            }
            return false;
        }

        _parameters = parameters;
        _isValid = true;
        if (errorMessage)
        {
            errorMessage->clear();
        }
        return true;
    }

    void RpcCameraModel::setImageSize(std::optional<CameraImageSize> imageSize)
    {
        if (imageSize && (imageSize->samples <= 0 || imageSize->lines <= 0))
        {
            _imageSize.reset();
            return;
        }
        _imageSize = imageSize;
    }

    bool RpcCameraModel::evaluateNormalizedUncorrected(double normalizedLongitude,
                                                       double normalizedLatitude,
                                                       double normalizedHeight,
                                                       CameraImageCoordinate* image) const
    {
        if (!_isValid || !image || !finite(normalizedLongitude) || !finite(normalizedLatitude) ||
            !finite(normalizedHeight))
        {
            return false;
        }

        const Coefficients terms = rpc00bTerms(normalizedLongitude, normalizedLatitude, normalizedHeight);
        const double line_denominator = dot(_parameters.lineDenominator, terms);
        const double sample_denominator = dot(_parameters.sampleDenominator, terms);
        if (std::abs(line_denominator) < kDenominatorEpsilon || std::abs(sample_denominator) < kDenominatorEpsilon)
        {
            return false;
        }

        image->line =
            _parameters.lineOffset + _parameters.lineScale * dot(_parameters.lineNumerator, terms) / line_denominator;
        image->sample = _parameters.sampleOffset +
                        _parameters.sampleScale * dot(_parameters.sampleNumerator, terms) / sample_denominator;
        return finite(image->sample) && finite(image->line);
    }

    bool RpcCameraModel::evaluateNormalized(double normalizedLongitude,
                                            double normalizedLatitude,
                                            double normalizedHeight,
                                            CameraImageCoordinate* image) const
    {
        CameraImageCoordinate uncorrected;
        return evaluateNormalizedUncorrected(normalizedLongitude, normalizedLatitude, normalizedHeight, &uncorrected) &&
               applyImageCorrection(uncorrected, image);
    }

    bool RpcCameraModel::groundToImageGeodetic(const GeodeticCoordinate& ground, CameraImageCoordinate* image) const
    {
        if (!_isValid || !image || !finite(ground[0]) || !finite(ground[1]) || !finite(ground[2]))
        {
            return false;
        }
        const double normalized_longitude =
            longitudeDifference(ground[0], _parameters.longitudeOffset) / _parameters.longitudeScale;
        const double normalized_latitude = (ground[1] - _parameters.latitudeOffset) / _parameters.latitudeScale;
        const double normalized_height = (ground[2] - _parameters.heightOffset) / _parameters.heightScale;
        return evaluateNormalized(normalized_longitude, normalized_latitude, normalized_height, image);
    }

    bool RpcCameraModel::groundToImageGeodeticUncorrected(const GeodeticCoordinate& ground,
                                                          CameraImageCoordinate* image) const
    {
        if (!_isValid || !image || !finite(ground[0]) || !finite(ground[1]) || !finite(ground[2]))
        {
            return false;
        }
        const double normalized_longitude =
            longitudeDifference(ground[0], _parameters.longitudeOffset) / _parameters.longitudeScale;
        const double normalized_latitude = (ground[1] - _parameters.latitudeOffset) / _parameters.latitudeScale;
        const double normalized_height = (ground[2] - _parameters.heightOffset) / _parameters.heightScale;
        return evaluateNormalizedUncorrected(normalized_longitude, normalized_latitude, normalized_height, image);
    }

    bool RpcCameraModel::groundToImage(const EcefCoordinate& groundMeters, CameraGroundProjection* projection) const
    {
        if (!projection)
        {
            return false;
        }
        GeodeticCoordinate geodetic;
        if (!ecefToGeodetic(groundMeters, &geodetic) || !groundToImageGeodetic(geodetic, &projection->image))
        {
            return false;
        }

        CameraImagingRay ray;
        if (rayForPixel(projection->image, &ray))
        {
            projection->positiveDepthMeters = (groundMeters[0] - ray.originMeters[0]) * ray.direction[0] +
                                              (groundMeters[1] - ray.originMeters[1]) * ray.direction[1] +
                                              (groundMeters[2] - ray.originMeters[2]) * ray.direction[2];
        }
        else
        {
            projection->positiveDepthMeters = std::numeric_limits<double>::quiet_NaN();
        }
        projection->ephemerisTimeSeconds.reset();
        return true;
    }

    bool RpcCameraModel::imageToGroundAtHeight(const CameraImageCoordinate& image,
                                               double ellipsoidalHeightMeters,
                                               GeodeticCoordinate* ground) const
    {
        return imageToGroundAtHeight(image, ellipsoidalHeightMeters, ground, InverseOptions{});
    }

    bool RpcCameraModel::imageToGroundAtHeight(const CameraImageCoordinate& image,
                                               double ellipsoidalHeightMeters,
                                               GeodeticCoordinate* ground,
                                               const InverseOptions& options) const
    {
        if (!_isValid || !ground || !finite(image.sample) || !finite(image.line) || !finite(ellipsoidalHeightMeters) ||
            options.pixelTolerance <= 0.0 || options.maximumIterations <= 0)
        {
            return false;
        }

        const double normalized_height = (ellipsoidalHeightMeters - _parameters.heightOffset) / _parameters.heightScale;
        double normalized_longitude = 0.0;
        double normalized_latitude = 0.0;
        constexpr double derivative_step = 1.0e-6;

        for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
        {
            CameraImageCoordinate current;
            if (!evaluateNormalized(normalized_longitude, normalized_latitude, normalized_height, &current))
            {
                return false;
            }
            const double sample_residual = image.sample - current.sample;
            const double line_residual = image.line - current.line;
            if (std::hypot(sample_residual, line_residual) <= options.pixelTolerance)
            {
                (*ground)[0] = _parameters.longitudeOffset + normalized_longitude * _parameters.longitudeScale;
                (*ground)[1] = _parameters.latitudeOffset + normalized_latitude * _parameters.latitudeScale;
                (*ground)[2] = ellipsoidalHeightMeters;
                return finite((*ground)[0]) && finite((*ground)[1]);
            }

            CameraImageCoordinate longitude_plus;
            CameraImageCoordinate longitude_minus;
            CameraImageCoordinate latitude_plus;
            CameraImageCoordinate latitude_minus;
            if (!evaluateNormalized(
                    normalized_longitude + derivative_step, normalized_latitude, normalized_height, &longitude_plus) ||
                !evaluateNormalized(
                    normalized_longitude - derivative_step, normalized_latitude, normalized_height, &longitude_minus) ||
                !evaluateNormalized(
                    normalized_longitude, normalized_latitude + derivative_step, normalized_height, &latitude_plus) ||
                !evaluateNormalized(
                    normalized_longitude, normalized_latitude - derivative_step, normalized_height, &latitude_minus))
            {
                return false;
            }

            const double d_sample_d_longitude =
                (longitude_plus.sample - longitude_minus.sample) / (2.0 * derivative_step);
            const double d_line_d_longitude = (longitude_plus.line - longitude_minus.line) / (2.0 * derivative_step);
            const double d_sample_d_latitude = (latitude_plus.sample - latitude_minus.sample) / (2.0 * derivative_step);
            const double d_line_d_latitude = (latitude_plus.line - latitude_minus.line) / (2.0 * derivative_step);
            const double determinant =
                d_sample_d_longitude * d_line_d_latitude - d_sample_d_latitude * d_line_d_longitude;
            if (!finite(determinant) || std::abs(determinant) < 1.0e-12)
            {
                return false;
            }

            const double longitude_update =
                (sample_residual * d_line_d_latitude - line_residual * d_sample_d_latitude) / determinant;
            const double latitude_update =
                (d_sample_d_longitude * line_residual - d_line_d_longitude * sample_residual) / determinant;
            bool accepted = false;
            double step_scale = 1.0;
            const double current_error = std::hypot(sample_residual, line_residual);
            for (int line_search = 0; line_search < 10; ++line_search)
            {
                const double trial_longitude = normalized_longitude + step_scale * longitude_update;
                const double trial_latitude = normalized_latitude + step_scale * latitude_update;
                CameraImageCoordinate trial;
                if (evaluateNormalized(trial_longitude, trial_latitude, normalized_height, &trial) &&
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
                return false;
            }
        }
        return false;
    }

    bool RpcCameraModel::imageToEcefAtHeight(const CameraImageCoordinate& image,
                                             double ellipsoidalHeightMeters,
                                             EcefCoordinate* ground) const
    {
        GeodeticCoordinate geodetic;
        return ground && imageToGroundAtHeight(image, ellipsoidalHeightMeters, &geodetic) &&
               geodeticToEcef(geodetic, ground);
    }

    bool RpcCameraModel::rayForPixel(const CameraImageCoordinate& pixel, CameraImagingRay* ray) const
    {
        if (!_isValid || !ray)
        {
            return false;
        }
        EcefCoordinate upper;
        EcefCoordinate lower;
        const double upper_height = _parameters.heightOffset + _parameters.heightScale;
        const double lower_height = _parameters.heightOffset - _parameters.heightScale;
        if (!imageToEcefAtHeight(pixel, upper_height, &upper) || !imageToEcefAtHeight(pixel, lower_height, &lower))
        {
            return false;
        }

        EcefCoordinate direction{{lower[0] - upper[0], lower[1] - upper[1], lower[2] - upper[2]}};
        const double norm = std::hypot(direction[0], std::hypot(direction[1], direction[2]));
        if (!finite(norm) || norm <= 1.0e-9)
        {
            return false;
        }
        for (double& component : direction)
        {
            component /= norm;
        }
        ray->originMeters = upper;
        ray->direction = direction;
        ray->ephemerisTimeSeconds.reset();
        return true;
    }

} // namespace xjw
