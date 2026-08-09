#include "PlanetaryLineScanCamera.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{
namespace
{

using Matrix3 = PlanetaryLineScanCamera::Matrix3;
using Vector3 = PlanetaryLineScanCamera::Vector3;
using Quaternion = std::array<double, 4>;

constexpr double kSupportToleranceSeconds = 1.0e-7;

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

Matrix3 multiply(const Matrix3 &left, const Matrix3 &right)
{
    Matrix3 result{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                result[static_cast<std::size_t>(row * 3 + column)] +=
                    left[static_cast<std::size_t>(row * 3 + inner)]
                    * right[static_cast<std::size_t>(inner * 3 + column)];
            }
        }
    }
    return result;
}

Matrix3 transpose(const Matrix3 &matrix)
{
    return {matrix[0], matrix[3], matrix[6],
            matrix[1], matrix[4], matrix[7],
            matrix[2], matrix[5], matrix[8]};
}

Vector3 multiply(const Matrix3 &matrix, const Vector3 &vector)
{
    return {
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]
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

Quaternion normalizedQuaternion(Quaternion quaternion)
{
    double norm = 0.0;
    for (double value : quaternion)
    {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (!(norm > 0.0))
    {
        return {1.0, 0.0, 0.0, 0.0};
    }
    for (double &value : quaternion)
    {
        value /= norm;
    }
    return quaternion;
}

Quaternion slerp(Quaternion first, Quaternion second, double fraction)
{
    first = normalizedQuaternion(first);
    second = normalizedQuaternion(second);
    double dot = 0.0;
    for (int index = 0; index < 4; ++index)
    {
        dot += first[static_cast<std::size_t>(index)] * second[static_cast<std::size_t>(index)];
    }
    if (dot < 0.0)
    {
        dot = -dot;
        for (double &value : second)
        {
            value = -value;
        }
    }
    dot = std::clamp(dot, -1.0, 1.0);
    if (dot > 0.9995)
    {
        Quaternion result{};
        for (int index = 0; index < 4; ++index)
        {
            result[static_cast<std::size_t>(index)] =
                first[static_cast<std::size_t>(index)]
                + fraction * (second[static_cast<std::size_t>(index)]
                              - first[static_cast<std::size_t>(index)]);
        }
        return normalizedQuaternion(result);
    }
    const double angle = std::acos(dot);
    const double denominator = std::sin(angle);
    const double first_weight = std::sin((1.0 - fraction) * angle) / denominator;
    const double second_weight = std::sin(fraction * angle) / denominator;
    Quaternion result{};
    for (int index = 0; index < 4; ++index)
    {
        result[static_cast<std::size_t>(index)] =
            first_weight * first[static_cast<std::size_t>(index)]
            + second_weight * second[static_cast<std::size_t>(index)];
    }
    return result;
}

Matrix3 quaternionToMatrix(const Quaternion &scalar_first)
{
    const Quaternion q = normalizedQuaternion(scalar_first);
    const double w = q[0];
    const double x = q[1];
    const double y = q[2];
    const double z = q[3];
    return {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
            2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
            2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)};
}

template <typename Sample>
std::size_t lowerInterval(const std::vector<Sample> &samples, double ephemeris_time)
{
    const auto upper = std::upper_bound(samples.begin(), samples.end(), ephemeris_time,
        [](double time, const Sample &sample)
        {
            return time < sample.ephemerisTimeSeconds;
        });
    if (upper == samples.begin())
    {
        return 0;
    }
    if (upper == samples.end())
    {
        return samples.size() - 2;
    }
    return static_cast<std::size_t>(std::distance(samples.begin(), upper) - 1);
}

template <typename Sample>
bool supports(const std::vector<Sample> &samples, double ephemeris_time)
{
    return samples.size() >= 2
        && ephemeris_time >= samples.front().ephemerisTimeSeconds - kSupportToleranceSeconds
        && ephemeris_time <= samples.back().ephemerisTimeSeconds + kSupportToleranceSeconds;
}

} // namespace

PlanetaryLineScanCamera::PoseBias PlanetaryLineScanCamera::bodyFixedSmallAngleBias(
    const Vector3 &angleAxisRadians, const Vector3 &translationMeters)
{
    PoseBias bias;
    bias.bodyFixedTranslationMeters = translationMeters;
    const double angle = std::sqrt(angleAxisRadians[0] * angleAxisRadians[0]
                                 + angleAxisRadians[1] * angleAxisRadians[1]
                                 + angleAxisRadians[2] * angleAxisRadians[2]);
    if (!(angle > 0.0) || !std::isfinite(angle))
    {
        return bias;
    }
    const double x = angleAxisRadians[0] / angle;
    const double y = angleAxisRadians[1] / angle;
    const double z = angleAxisRadians[2] / angle;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double one_minus_cosine = 1.0 - cosine;
    bias.bodyFixedRotation = {
        cosine + x * x * one_minus_cosine,
        x * y * one_minus_cosine - z * sine,
        x * z * one_minus_cosine + y * sine,
        y * x * one_minus_cosine + z * sine,
        cosine + y * y * one_minus_cosine,
        y * z * one_minus_cosine - x * sine,
        z * x * one_minus_cosine - y * sine,
        z * y * one_minus_cosine + x * sine,
        cosine + z * z * one_minus_cosine
    };
    return bias;
}

bool PlanetaryLineScanCamera::isLineInsideImage(double line, PixelConvention convention) const
{
    const double csm_line = csmCoordinate(line, convention);
    return std::isfinite(csm_line) && csm_line >= 0.5
        && csm_line <= static_cast<double>(_imageLines) - 0.5;
}

bool PlanetaryLineScanCamera::absoluteEtForLine(double line,
                                                PixelConvention convention,
                                                double *ephemerisTimeSeconds) const
{
    if (!_isLoaded || ephemerisTimeSeconds == nullptr || !isLineInsideImage(line, convention))
    {
        return false;
    }
    const double csm_line = csmCoordinate(line, convention);
    auto record = std::upper_bound(_lineRates.begin(), _lineRates.end(), csm_line,
        [](double candidate, const LineRateRecord &rate)
        {
            return candidate < rate.startLineCsm;
        });
    if (record != _lineRates.begin())
    {
        --record;
    }
    *ephemerisTimeSeconds = _centerEphemerisTimeSeconds
        + record->startTimeRelativeToCenterSeconds
        + record->secondsPerLine * (csm_line - record->startLineCsm + 0.5);
    return std::isfinite(*ephemerisTimeSeconds);
}

bool PlanetaryLineScanCamera::lineForAbsoluteEt(double ephemerisTimeSeconds,
                                                PixelConvention convention,
                                                double *line) const
{
    if (!_isLoaded || line == nullptr || !std::isfinite(ephemerisTimeSeconds))
    {
        return false;
    }
    const double relative_time = ephemerisTimeSeconds - _centerEphemerisTimeSeconds;
    auto record = _lineRates.begin();
    for (auto candidate = _lineRates.begin(); candidate != _lineRates.end(); ++candidate)
    {
        const double start_center_time = candidate->startTimeRelativeToCenterSeconds
                                       + 0.5 * candidate->secondsPerLine;
        if (relative_time >= start_center_time)
        {
            record = candidate;
        }
    }
    double csm_line = record->startLineCsm - 0.5
        + (relative_time - record->startTimeRelativeToCenterSeconds) / record->secondsPerLine;
    const double minimum_csm_line = 0.5;
    const double maximum_csm_line = static_cast<double>(_imageLines) - 0.5;
    if (csm_line < minimum_csm_line - 1.0e-8 || csm_line > maximum_csm_line + 1.0e-8)
    {
        return false;
    }
    csm_line = std::clamp(csm_line, minimum_csm_line, maximum_csm_line);
    const double converted = requestedCoordinate(csm_line, convention);
    if (!isLineInsideImage(converted, convention))
    {
        return false;
    }
    *line = converted;
    return true;
}

bool PlanetaryLineScanCamera::sensorCenterBodyFixedAtEt(double ephemerisTimeSeconds,
                                                        Vector3 *centerMeters) const
{
    if (!_isLoaded || centerMeters == nullptr
        || !supports(_instrumentStates, ephemerisTimeSeconds)
        || !supports(_bodyRotations, ephemerisTimeSeconds))
    {
        return false;
    }
    const std::size_t state_index = lowerInterval(_instrumentStates, ephemerisTimeSeconds);
    const StateSample &first = _instrumentStates[state_index];
    const StateSample &second = _instrumentStates[state_index + 1];
    const double duration = second.ephemerisTimeSeconds - first.ephemerisTimeSeconds;
    const double t = std::clamp((ephemerisTimeSeconds - first.ephemerisTimeSeconds) / duration, 0.0, 1.0);
    const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
    const double h10 = t * t * t - 2.0 * t * t + t;
    const double h01 = -2.0 * t * t * t + 3.0 * t * t;
    const double h11 = t * t * t - t * t;
    Vector3 inertial_meters{};
    for (int axis = 0; axis < 3; ++axis)
    {
        inertial_meters[static_cast<std::size_t>(axis)] = 1000.0 * (
            h00 * first.positionKilometers[static_cast<std::size_t>(axis)]
            + h10 * duration * first.velocityKilometersPerSecond[static_cast<std::size_t>(axis)]
            + h01 * second.positionKilometers[static_cast<std::size_t>(axis)]
            + h11 * duration * second.velocityKilometersPerSecond[static_cast<std::size_t>(axis)]);
    }
    const std::size_t rotation_index = lowerInterval(_bodyRotations, ephemerisTimeSeconds);
    const QuaternionSample &rotation_first = _bodyRotations[rotation_index];
    const QuaternionSample &rotation_second = _bodyRotations[rotation_index + 1];
    const double rotation_fraction = std::clamp(
        (ephemerisTimeSeconds - rotation_first.ephemerisTimeSeconds)
            / (rotation_second.ephemerisTimeSeconds - rotation_first.ephemerisTimeSeconds),
        0.0, 1.0);
    const Matrix3 inertial_to_body = multiply(
        _bodyConstantRotation,
        quaternionToMatrix(slerp(rotation_first.scalarFirst,
                                 rotation_second.scalarFirst,
                                 rotation_fraction)));
    *centerMeters = multiply(inertial_to_body, inertial_meters);
    return true;
}

bool PlanetaryLineScanCamera::sensorToBodyFixedAtEt(double ephemerisTimeSeconds,
                                                    Matrix3 *rotation) const
{
    if (!_isLoaded || rotation == nullptr
        || !supports(_bodyRotations, ephemerisTimeSeconds)
        || !supports(_instrumentPointing, ephemerisTimeSeconds))
    {
        return false;
    }
    const auto interpolated_rotation = [&](const std::vector<QuaternionSample> &samples)
    {
        const std::size_t index = lowerInterval(samples, ephemerisTimeSeconds);
        const double fraction = std::clamp(
            (ephemerisTimeSeconds - samples[index].ephemerisTimeSeconds)
                / (samples[index + 1].ephemerisTimeSeconds - samples[index].ephemerisTimeSeconds),
            0.0, 1.0);
        return quaternionToMatrix(slerp(samples[index].scalarFirst,
                                        samples[index + 1].scalarFirst,
                                        fraction));
    };
    const Matrix3 inertial_to_body = multiply(_bodyConstantRotation,
                                               interpolated_rotation(_bodyRotations));
    const Matrix3 inertial_to_sensor = multiply(_instrumentConstantRotation,
                                                 interpolated_rotation(_instrumentPointing));
    *rotation = multiply(inertial_to_body, transpose(inertial_to_sensor));
    return true;
}

bool PlanetaryLineScanCamera::correctedPoseAtEt(double ephemerisTimeSeconds,
                                                const PoseBias &bias,
                                                Vector3 *centerMeters,
                                                Matrix3 *sensorToBodyFixed) const
{
    if (!sensorCenterBodyFixedAtEt(ephemerisTimeSeconds, centerMeters)
        || !sensorToBodyFixedAtEt(ephemerisTimeSeconds, sensorToBodyFixed))
    {
        return false;
    }
    *sensorToBodyFixed = multiply(bias.bodyFixedRotation, *sensorToBodyFixed);
    for (int axis = 0; axis < 3; ++axis)
    {
        (*centerMeters)[static_cast<std::size_t>(axis)] +=
            bias.bodyFixedTranslationMeters[static_cast<std::size_t>(axis)];
    }
    return true;
}

} // namespace xjw
