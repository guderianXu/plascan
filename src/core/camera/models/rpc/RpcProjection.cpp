#include "RpcProjection.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::camera_models::rpc
{
    namespace
    {

        constexpr double denominatorEpsilon = 1.0e-14;
        constexpr double wgs84SemiMajorMeters = 6378137.0;
        constexpr double wgs84Flattening = 1.0 / 298.257223563;
        constexpr double degreesToRadians = 3.14159265358979323846 / 180.0;
        constexpr double radiansToDegrees = 180.0 / 3.14159265358979323846;

        bool finite(double value)
        {
            return std::isfinite(value);
        }

        RpcDefinition::Coefficients terms(double longitude, double latitude, double height)
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

        double dot(const RpcDefinition::Coefficients& first, const RpcDefinition::Coefficients& second)
        {
            double result = 0.0;
            for (std::size_t index = 0; index < first.size(); ++index)
            {
                result += first[index] * second[index];
            }
            return result;
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

        ImagePoint applyCorrection(const ImagePoint& uncorrected,
                                   const RpcDefinition::Parameters& parameters,
                                   const ImageCorrection& correction)
        {
            const double normalizedSample = (uncorrected.sample - parameters.sampleOffset) / parameters.sampleScale;
            const double normalizedLine = (uncorrected.line - parameters.lineOffset) / parameters.lineScale;
            return {uncorrected.sample + correction.sampleOffsetPixels +
                        correction.sampleSamplePixels * normalizedSample + correction.sampleLinePixels * normalizedLine,
                    uncorrected.line + correction.lineOffsetPixels + correction.lineSamplePixels * normalizedSample +
                        correction.lineLinePixels * normalizedLine};
        }

    } // namespace

    bool RpcProjection::groundToImage(const RpcInstance& instance,
                                      const RpcDefinition::GeodeticCoordinate& ground,
                                      ImagePoint* image)
    {
        if (!image || !finite(ground[0]) || !finite(ground[1]) || !finite(ground[2]))
        {
            return false;
        }
        const RpcDefinition::Parameters& parameters = instance.rpcDefinition().parameters();
        const double normalizedLongitude =
            longitudeDifference(ground[0], parameters.longitudeOffset) / parameters.longitudeScale;
        const double normalizedLatitude = (ground[1] - parameters.latitudeOffset) / parameters.latitudeScale;
        const double normalizedHeight = (ground[2] - parameters.heightOffset) / parameters.heightScale;
        return evaluate(instance, normalizedLongitude, normalizedLatitude, normalizedHeight, image, true);
    }

    bool RpcProjection::groundToImageUncorrected(const RpcInstance& instance,
                                                 const RpcDefinition::GeodeticCoordinate& ground,
                                                 ImagePoint* image)
    {
        if (!image || !finite(ground[0]) || !finite(ground[1]) || !finite(ground[2]))
        {
            return false;
        }
        const RpcDefinition::Parameters& parameters = instance.rpcDefinition().parameters();
        const double normalizedLongitude =
            longitudeDifference(ground[0], parameters.longitudeOffset) / parameters.longitudeScale;
        const double normalizedLatitude = (ground[1] - parameters.latitudeOffset) / parameters.latitudeScale;
        const double normalizedHeight = (ground[2] - parameters.heightOffset) / parameters.heightScale;
        return evaluate(instance, normalizedLongitude, normalizedLatitude, normalizedHeight, image, false);
    }

    bool RpcProjection::groundToImageEcef(const RpcInstance& instance,
                                          const EcefCoordinate& groundMeters,
                                          ImagePoint* image)
    {
        RpcDefinition::GeodeticCoordinate geodetic;
        return ecefToGeodetic(groundMeters, &geodetic) && groundToImage(instance, geodetic, image);
    }

    bool RpcProjection::imageToGroundAtHeight(const RpcInstance& instance,
                                              const ImagePoint& image,
                                              double ellipsoidalHeightMeters,
                                              RpcDefinition::GeodeticCoordinate* ground)
    {
        return imageToGroundAtHeight(instance, image, ellipsoidalHeightMeters, ground, InverseOptions{});
    }

    bool RpcProjection::imageToGroundAtHeight(const RpcInstance& instance,
                                              const ImagePoint& image,
                                              double ellipsoidalHeightMeters,
                                              RpcDefinition::GeodeticCoordinate* ground,
                                              InverseOptions options)
    {
        if (!ground || !finite(image.sample) || !finite(image.line) || !finite(ellipsoidalHeightMeters) ||
            options.pixelTolerance <= 0.0 || options.maximumIterations <= 0)
        {
            return false;
        }
        const RpcDefinition::Parameters& parameters = instance.rpcDefinition().parameters();
        const double normalizedHeight = (ellipsoidalHeightMeters - parameters.heightOffset) / parameters.heightScale;
        double normalizedLongitude = 0.0;
        double normalizedLatitude = 0.0;
        constexpr double derivativeStep = 1.0e-6;

        for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
        {
            ImagePoint current;
            if (!evaluate(instance, normalizedLongitude, normalizedLatitude, normalizedHeight, &current, true))
            {
                return false;
            }
            const double sampleResidual = image.sample - current.sample;
            const double lineResidual = image.line - current.line;
            if (std::hypot(sampleResidual, lineResidual) <= options.pixelTolerance)
            {
                *ground = {parameters.longitudeOffset + normalizedLongitude * parameters.longitudeScale,
                           parameters.latitudeOffset + normalizedLatitude * parameters.latitudeScale,
                           ellipsoidalHeightMeters};
                return finite((*ground)[0]) && finite((*ground)[1]);
            }

            ImagePoint longitudePlus;
            ImagePoint longitudeMinus;
            ImagePoint latitudePlus;
            ImagePoint latitudeMinus;
            if (!evaluate(instance,
                          normalizedLongitude + derivativeStep,
                          normalizedLatitude,
                          normalizedHeight,
                          &longitudePlus,
                          true) ||
                !evaluate(instance,
                          normalizedLongitude - derivativeStep,
                          normalizedLatitude,
                          normalizedHeight,
                          &longitudeMinus,
                          true) ||
                !evaluate(instance,
                          normalizedLongitude,
                          normalizedLatitude + derivativeStep,
                          normalizedHeight,
                          &latitudePlus,
                          true) ||
                !evaluate(instance,
                          normalizedLongitude,
                          normalizedLatitude - derivativeStep,
                          normalizedHeight,
                          &latitudeMinus,
                          true))
            {
                return false;
            }

            const double dSampleDLongitude = (longitudePlus.sample - longitudeMinus.sample) / (2.0 * derivativeStep);
            const double dLineDLongitude = (longitudePlus.line - longitudeMinus.line) / (2.0 * derivativeStep);
            const double dSampleDLatitude = (latitudePlus.sample - latitudeMinus.sample) / (2.0 * derivativeStep);
            const double dLineDLatitude = (latitudePlus.line - latitudeMinus.line) / (2.0 * derivativeStep);
            const double determinant = dSampleDLongitude * dLineDLatitude - dSampleDLatitude * dLineDLongitude;
            if (!finite(determinant) || std::abs(determinant) < 1.0e-12)
            {
                return false;
            }

            const double longitudeUpdate =
                (sampleResidual * dLineDLatitude - lineResidual * dSampleDLatitude) / determinant;
            const double latitudeUpdate =
                (dSampleDLongitude * lineResidual - dLineDLongitude * sampleResidual) / determinant;
            bool accepted = false;
            double stepScale = 1.0;
            const double currentError = std::hypot(sampleResidual, lineResidual);
            for (int lineSearch = 0; lineSearch < 10; ++lineSearch)
            {
                const double trialLongitude = normalizedLongitude + stepScale * longitudeUpdate;
                const double trialLatitude = normalizedLatitude + stepScale * latitudeUpdate;
                ImagePoint trial;
                if (evaluate(instance, trialLongitude, trialLatitude, normalizedHeight, &trial, true) &&
                    std::hypot(image.sample - trial.sample, image.line - trial.line) < currentError)
                {
                    normalizedLongitude = trialLongitude;
                    normalizedLatitude = trialLatitude;
                    accepted = true;
                    break;
                }
                stepScale *= 0.5;
            }
            if (!accepted || !finite(normalizedLongitude) || !finite(normalizedLatitude))
            {
                return false;
            }
        }
        return false;
    }

    bool RpcProjection::geodeticToEcef(const RpcDefinition::GeodeticCoordinate& geodetic, EcefCoordinate* ecef)
    {
        if (!ecef || !finite(geodetic[0]) || !finite(geodetic[1]) || !finite(geodetic[2]) || geodetic[1] < -90.0 ||
            geodetic[1] > 90.0)
        {
            return false;
        }
        const double longitude = geodetic[0] * degreesToRadians;
        const double latitude = geodetic[1] * degreesToRadians;
        const double eccentricitySquared = wgs84Flattening * (2.0 - wgs84Flattening);
        const double sineLatitude = std::sin(latitude);
        const double cosineLatitude = std::cos(latitude);
        const double primeVertical =
            wgs84SemiMajorMeters / std::sqrt(1.0 - eccentricitySquared * sineLatitude * sineLatitude);
        (*ecef)[0] = (primeVertical + geodetic[2]) * cosineLatitude * std::cos(longitude);
        (*ecef)[1] = (primeVertical + geodetic[2]) * cosineLatitude * std::sin(longitude);
        (*ecef)[2] = (primeVertical * (1.0 - eccentricitySquared) + geodetic[2]) * sineLatitude;
        return finite((*ecef)[0]) && finite((*ecef)[1]) && finite((*ecef)[2]);
    }

    bool RpcProjection::ecefToGeodetic(const EcefCoordinate& ecef, RpcDefinition::GeodeticCoordinate* geodetic)
    {
        if (!geodetic || !finite(ecef[0]) || !finite(ecef[1]) || !finite(ecef[2]))
        {
            return false;
        }
        const double horizontal = std::hypot(ecef[0], ecef[1]);
        if (horizontal < 1.0e-9 && std::abs(ecef[2]) < 1.0e-9)
        {
            return false;
        }
        const double eccentricitySquared = wgs84Flattening * (2.0 - wgs84Flattening);
        const double longitude = std::atan2(ecef[1], ecef[0]);
        double latitude = std::atan2(ecef[2], horizontal * (1.0 - eccentricitySquared));
        double height = 0.0;
        for (int iteration = 0; iteration < 15; ++iteration)
        {
            const double sineLatitude = std::sin(latitude);
            const double primeVertical =
                wgs84SemiMajorMeters / std::sqrt(1.0 - eccentricitySquared * sineLatitude * sineLatitude);
            if (horizontal > 1.0e-9)
            {
                height = horizontal / std::cos(latitude) - primeVertical;
            }
            else
            {
                height = std::abs(ecef[2]) - primeVertical * (1.0 - eccentricitySquared);
            }
            const double nextLatitude = std::atan2(
                ecef[2], horizontal * (1.0 - eccentricitySquared * primeVertical / (primeVertical + height)));
            if (std::abs(nextLatitude - latitude) < 1.0e-14)
            {
                latitude = nextLatitude;
                break;
            }
            latitude = nextLatitude;
        }
        const double sineLatitude = std::sin(latitude);
        const double primeVertical =
            wgs84SemiMajorMeters / std::sqrt(1.0 - eccentricitySquared * sineLatitude * sineLatitude);
        height = horizontal > 1.0e-9 ? horizontal / std::cos(latitude) - primeVertical
                                     : std::abs(ecef[2]) - primeVertical * (1.0 - eccentricitySquared);
        *geodetic = {longitude * radiansToDegrees, latitude * radiansToDegrees, height};
        return finite((*geodetic)[0]) && finite((*geodetic)[1]) && finite((*geodetic)[2]);
    }

    bool RpcProjection::ray(const RpcInstance& instance, const ImagePoint& image, RpcRay* rayResult)
    {
        if (!rayResult)
        {
            return false;
        }
        const RpcDefinition::Parameters& parameters = instance.rpcDefinition().parameters();
        EcefCoordinate upper;
        EcefCoordinate lower;
        const double upperHeight = parameters.heightOffset + parameters.heightScale;
        const double lowerHeight = parameters.heightOffset - parameters.heightScale;
        RpcDefinition::GeodeticCoordinate upperGeodetic;
        RpcDefinition::GeodeticCoordinate lowerGeodetic;
        if (!imageToGroundAtHeight(instance, image, upperHeight, &upperGeodetic) ||
            !imageToGroundAtHeight(instance, image, lowerHeight, &lowerGeodetic) ||
            !geodeticToEcef(upperGeodetic, &upper) || !geodeticToEcef(lowerGeodetic, &lower))
        {
            return false;
        }
        rayResult->direction = {lower[0] - upper[0], lower[1] - upper[1], lower[2] - upper[2]};
        const double norm =
            std::hypot(rayResult->direction[0], std::hypot(rayResult->direction[1], rayResult->direction[2]));
        if (!finite(norm) || norm <= 1.0e-9)
        {
            return false;
        }
        for (double& component : rayResult->direction)
        {
            component /= norm;
        }
        rayResult->origin = upper;
        return true;
    }

    bool RpcProjection::evaluate(const RpcInstance& instance,
                                 double normalizedLongitude,
                                 double normalizedLatitude,
                                 double normalizedHeight,
                                 ImagePoint* image,
                                 bool applyCorrectionFlag)
    {
        if (!image || !finite(normalizedLongitude) || !finite(normalizedLatitude) || !finite(normalizedHeight))
        {
            return false;
        }
        const RpcDefinition::Parameters& parameters = instance.rpcDefinition().parameters();
        const RpcDefinition::Coefficients polynomialTerms =
            terms(normalizedLongitude, normalizedLatitude, normalizedHeight);
        const double lineDenominator = dot(parameters.lineDenominator, polynomialTerms);
        const double sampleDenominator = dot(parameters.sampleDenominator, polynomialTerms);
        if (std::abs(lineDenominator) < denominatorEpsilon || std::abs(sampleDenominator) < denominatorEpsilon)
        {
            return false;
        }
        const ImagePoint uncorrected{
            parameters.sampleOffset +
                parameters.sampleScale * dot(parameters.sampleNumerator, polynomialTerms) / sampleDenominator,
            parameters.lineOffset +
                parameters.lineScale * dot(parameters.lineNumerator, polynomialTerms) / lineDenominator};
        if (!finite(uncorrected.sample) || !finite(uncorrected.line))
        {
            return false;
        }
        *image =
            applyCorrectionFlag ? applyCorrection(uncorrected, parameters, instance.imageCorrection()) : uncorrected;
        return finite(image->sample) && finite(image->line);
    }

} // namespace xjw::camera_models::rpc
