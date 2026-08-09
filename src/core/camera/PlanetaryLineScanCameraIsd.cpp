#include "PlanetaryLineScanCamera.h"

#include "io/PathIO.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QString>

#include <cmath>
#include <limits>
#include <utility>

namespace xjw
{
namespace
{

void setError(std::string *error_message, const std::string &message)
{
    if (error_message)
    {
        *error_message = message;
    }
}

bool finiteNumber(const QJsonObject &object,
                  const char *key,
                  double *value,
                  std::string *error_message)
{
    const QJsonValue item = object.value(QLatin1String(key));
    if (!item.isDouble() || !std::isfinite(item.toDouble()))
    {
        setError(error_message, std::string("ISD field '") + key + "' must be a finite number");
        return false;
    }
    *value = item.toDouble();
    return true;
}

bool positiveInteger(const QJsonObject &object,
                     const char *key,
                     int *value,
                     std::string *error_message)
{
    double parsed = 0.0;
    if (!finiteNumber(object, key, &parsed, error_message)
        || parsed < 1.0 || parsed > static_cast<double>(std::numeric_limits<int>::max())
        || std::floor(parsed) != parsed)
    {
        setError(error_message, std::string("ISD field '") + key + "' must be a positive integer");
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool nonEmptyString(const QJsonObject &object,
                    const char *key,
                    std::string *value,
                    std::string *error_message)
{
    const QJsonValue item = object.value(QLatin1String(key));
    if (!item.isString() || item.toString().trimmed().isEmpty())
    {
        setError(error_message, std::string("ISD field '") + key + "' must be a non-empty string");
        return false;
    }
    *value = item.toString().toUtf8().toStdString();
    return true;
}

template <std::size_t Size>
bool finiteArray(const QJsonValue &value,
                 const std::string &path,
                 std::array<double, Size> *output,
                 std::string *error_message)
{
    if (!value.isArray() || value.toArray().size() != static_cast<int>(Size))
    {
        setError(error_message, path + " must contain exactly " + std::to_string(Size) + " numbers");
        return false;
    }
    const QJsonArray values = value.toArray();
    for (std::size_t index = 0; index < Size; ++index)
    {
        const QJsonValue item = values.at(static_cast<int>(index));
        if (!item.isDouble() || !std::isfinite(item.toDouble()))
        {
            setError(error_message, path + " contains a non-finite or non-numeric value");
            return false;
        }
        (*output)[index] = item.toDouble();
    }
    return true;
}

bool parseConstantRotation(const QJsonObject &orientation,
                           const std::string &path,
                           PlanetaryLineScanCamera::Matrix3 *rotation,
                           std::string *error_message)
{
    const QJsonValue value = orientation.value(QStringLiteral("constant_rotation"));
    if (value.isUndefined())
    {
        return true;
    }
    if (!finiteArray<9>(value, path + ".constant_rotation", rotation, error_message))
    {
        return false;
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int other_row = row; other_row < 3; ++other_row)
        {
            double dot = 0.0;
            for (int column = 0; column < 3; ++column)
            {
                dot += (*rotation)[static_cast<std::size_t>(row * 3 + column)]
                     * (*rotation)[static_cast<std::size_t>(other_row * 3 + column)];
            }
            const double expected = row == other_row ? 1.0 : 0.0;
            if (std::abs(dot - expected) > 1.0e-5)
            {
                setError(error_message, path + ".constant_rotation is not orthonormal");
                return false;
            }
        }
    }
    const double determinant =
        (*rotation)[0] * ((*rotation)[4] * (*rotation)[8] - (*rotation)[5] * (*rotation)[7])
        - (*rotation)[1] * ((*rotation)[3] * (*rotation)[8] - (*rotation)[5] * (*rotation)[6])
        + (*rotation)[2] * ((*rotation)[3] * (*rotation)[7] - (*rotation)[4] * (*rotation)[6]);
    if (std::abs(determinant - 1.0) > 1.0e-5)
    {
        setError(error_message, path + ".constant_rotation must be a proper rotation");
        return false;
    }
    return true;
}

} // namespace

bool PlanetaryLineScanCamera::loadFromIsd(const std::string &path, std::string *errorMessage)
{
    QString read_error;
    const QByteArray bytes = common::io::readFileBytes(path, &read_error);
    if (!read_error.isEmpty())
    {
        setError(errorMessage, read_error.toUtf8().toStdString());
        return false;
    }

    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject())
    {
        setError(errorMessage, "Unable to parse USGSCSM ISD JSON object: "
                                   + parse_error.errorString().toUtf8().toStdString());
        return false;
    }

    const QJsonObject root = document.object();
    PlanetaryLineScanCamera parsed;
    if (!positiveInteger(root, "image_lines", &parsed._imageLines, errorMessage)
        || !positiveInteger(root, "image_samples", &parsed._imageSamples, errorMessage)
         || !nonEmptyString(root, "name_model", &parsed._modelName, errorMessage)
         || !nonEmptyString(root, "name_platform", &parsed._platformName, errorMessage)
         || !nonEmptyString(root, "name_sensor", &parsed._sensorName, errorMessage)
         || !nonEmptyString(root, "interpolation_method",
                            &parsed._declaredInterpolationMethod, errorMessage)
         || !finiteNumber(root, "starting_ephemeris_time",
                          &parsed._startingEphemerisTimeSeconds, errorMessage)
        || !finiteNumber(root, "center_ephemeris_time",
                         &parsed._centerEphemerisTimeSeconds, errorMessage))
    {
        return false;
    }
    if (parsed._modelName != "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL")
    {
        setError(errorMessage, "ISD camera model is not USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL");
        return false;
    }
    if (parsed._declaredInterpolationMethod != "lagrange")
    {
        setError(errorMessage,
                 "P0 line-scan camera only accepts ISD interpolation_method='lagrange'; "
                 "its raw-table Hermite/SLERP evaluation remains an explicit approximation");
        return false;
    }

    const QJsonObject naifKeywords = root.value(QStringLiteral("naif_keywords")).toObject();
    double bodyFrameCode = 0.0;
    if (naifKeywords.isEmpty() ||
        !finiteNumber(naifKeywords, "BODY_FRAME_CODE", &bodyFrameCode, errorMessage))
    {
        setError(errorMessage, "P0 line-scan ISD requires an integer NAIF BODY_FRAME_CODE");
        return false;
    }
    if (bodyFrameCode != 31001.0)
    {
        setError(errorMessage, "P0 line-scan camera currently requires BODY_FRAME_CODE=31001 (MOON_ME)");
        return false;
    }
    parsed._bodyFixedFrameCode = 31001;
    parsed._targetName = "MOON";
    parsed._bodyFixedFrameName = "MOON_ME";

    const QJsonObject focal_model = root.value(QStringLiteral("focal_length_model")).toObject();
    const QJsonObject detector_center = root.value(QStringLiteral("detector_center")).toObject();
    if (focal_model.isEmpty() || detector_center.isEmpty()
        || !finiteNumber(focal_model, "focal_length", &parsed._focalLengthMillimeters, errorMessage)
        || !finiteNumber(detector_center, "sample", &parsed._detectorSampleOrigin, errorMessage)
        || !finiteNumber(detector_center, "line", &parsed._detectorLineOrigin, errorMessage)
        || !finiteNumber(root, "detector_sample_summing", &parsed._detectorSampleSumming, errorMessage)
        || !finiteNumber(root, "detector_line_summing", &parsed._detectorLineSumming, errorMessage)
        || !finiteNumber(root, "starting_detector_sample", &parsed._startingDetectorSample, errorMessage)
        || !finiteNumber(root, "starting_detector_line", &parsed._startingDetectorLine, errorMessage)
        || !finiteArray<3>(root.value(QStringLiteral("focal2pixel_samples")),
                           "focal2pixel_samples", &parsed._focalToPixelSamples, errorMessage)
        || !finiteArray<3>(root.value(QStringLiteral("focal2pixel_lines")),
                           "focal2pixel_lines", &parsed._focalToPixelLines, errorMessage))
    {
        return false;
    }
    if (!(parsed._focalLengthMillimeters > 0.0)
        || !(parsed._detectorSampleSumming > 0.0)
        || !(parsed._detectorLineSumming > 0.0))
    {
        setError(errorMessage, "ISD focal length and detector summing factors must be positive");
        return false;
    }

    const QJsonArray rates = root.value(QStringLiteral("line_scan_rate")).toArray();
    if (rates.isEmpty())
    {
        setError(errorMessage, "ISD line_scan_rate must contain at least one record");
        return false;
    }
    for (int index = 0; index < rates.size(); ++index)
    {
        std::array<double, 3> values{};
        if (!finiteArray<3>(rates.at(index), "line_scan_rate[" + std::to_string(index) + "]",
                            &values, errorMessage))
        {
            return false;
        }
        if (!(values[2] > 0.0)
            || (!parsed._lineRates.empty() && values[0] <= parsed._lineRates.back().startLineCsm))
        {
            setError(errorMessage, "ISD line_scan_rate records require positive rates and increasing lines");
            return false;
        }
        parsed._lineRates.push_back({values[0], values[1], values[2]});
    }

    const QJsonObject distortion = root.value(QStringLiteral("optical_distortion")).toObject();
    const QJsonObject lro_distortion = distortion.value(QStringLiteral("lrolrocnac")).toObject();
    std::array<double, 1> distortion_coefficients{};
    if (lro_distortion.isEmpty()
        || !finiteArray<1>(lro_distortion.value(QStringLiteral("coefficients")),
                           "optical_distortion.lrolrocnac.coefficients",
                           &distortion_coefficients, errorMessage))
    {
        setError(errorMessage, "P0 line-scan camera requires the LRO LROC NAC distortion model");
        return false;
    }
    parsed._lroNacDistortionK1 = distortion_coefficients[0];

    const QJsonObject body = root.value(QStringLiteral("body_rotation")).toObject();
    const QJsonObject pointing = root.value(QStringLiteral("instrument_pointing")).toObject();
    if (body.isEmpty() || pointing.isEmpty()
        || !parseConstantRotation(body, "body_rotation", &parsed._bodyConstantRotation, errorMessage)
        || !parseConstantRotation(pointing, "instrument_pointing",
                                  &parsed._instrumentConstantRotation, errorMessage))
    {
        return false;
    }

    auto parse_orientation = [&](const QJsonObject &orientation,
                                 const std::string &name,
                                 std::vector<QuaternionSample> *samples)
    {
        const QJsonArray times = orientation.value(QStringLiteral("ephemeris_times")).toArray();
        const QJsonArray quaternions = orientation.value(QStringLiteral("quaternions")).toArray();
        if (times.size() < 2 || times.size() != quaternions.size())
        {
            setError(errorMessage, name + " requires matching ephemeris_times and quaternions");
            return false;
        }
        for (int index = 0; index < times.size(); ++index)
        {
            if (!times.at(index).isDouble() || !std::isfinite(times.at(index).toDouble()))
            {
                setError(errorMessage, name + ".ephemeris_times contains an invalid value");
                return false;
            }
            QuaternionSample sample;
            sample.ephemerisTimeSeconds = times.at(index).toDouble();
            if (!finiteArray<4>(quaternions.at(index), name + ".quaternions",
                                &sample.scalarFirst, errorMessage))
            {
                return false;
            }
            double quaternion_norm_squared = 0.0;
            for (double value : sample.scalarFirst)
            {
                quaternion_norm_squared += value * value;
            }
            if (!(quaternion_norm_squared > 1.0e-20)
                || (!samples->empty()
                    && sample.ephemerisTimeSeconds <= samples->back().ephemerisTimeSeconds))
            {
                setError(errorMessage, name + " quaternion times must be strictly increasing");
                return false;
            }
            samples->push_back(sample);
        }
        return true;
    };
    if (!parse_orientation(body, "body_rotation", &parsed._bodyRotations)
        || !parse_orientation(pointing, "instrument_pointing", &parsed._instrumentPointing))
    {
        return false;
    }

    const QJsonObject position = root.value(QStringLiteral("instrument_position")).toObject();
    const QJsonArray state_times = position.value(QStringLiteral("ephemeris_times")).toArray();
    const QJsonArray positions = position.value(QStringLiteral("positions")).toArray();
    const QJsonArray velocities = position.value(QStringLiteral("velocities")).toArray();
    if (state_times.size() < 2 || state_times.size() != positions.size()
        || state_times.size() != velocities.size())
    {
        setError(errorMessage, "instrument_position requires matching times, positions and velocities");
        return false;
    }
    for (int index = 0; index < state_times.size(); ++index)
    {
        if (!state_times.at(index).isDouble() || !std::isfinite(state_times.at(index).toDouble()))
        {
            setError(errorMessage, "instrument_position.ephemeris_times contains an invalid value");
            return false;
        }
        StateSample state;
        state.ephemerisTimeSeconds = state_times.at(index).toDouble();
        if (!finiteArray<3>(positions.at(index), "instrument_position.positions",
                            &state.positionKilometers, errorMessage)
            || !finiteArray<3>(velocities.at(index), "instrument_position.velocities",
                               &state.velocityKilometersPerSecond, errorMessage)
            || (!parsed._instrumentStates.empty()
                && state.ephemerisTimeSeconds <= parsed._instrumentStates.back().ephemerisTimeSeconds))
        {
            setError(errorMessage, "instrument_position times must be strictly increasing");
            return false;
        }
        parsed._instrumentStates.push_back(state);
    }

    if (std::abs(parsed._centerEphemerisTimeSeconds
                 + parsed._lineRates.front().startTimeRelativeToCenterSeconds
                 - parsed._startingEphemerisTimeSeconds) > 1.0e-5)
    {
        setError(errorMessage, "ISD starting time is inconsistent with the first line-scan-rate record");
        return false;
    }

    parsed._isLoaded = true;
    double first_line_et = 0.0;
    double last_line_et = 0.0;
    if (!parsed.absoluteEtForLine(0.5, PixelConvention::CsmPixelCenter, &first_line_et)
        || !parsed.absoluteEtForLine(static_cast<double>(parsed._imageLines) - 0.5,
                                     PixelConvention::CsmPixelCenter, &last_line_et)
        || first_line_et < parsed._instrumentStates.front().ephemerisTimeSeconds - 1.0e-7
        || last_line_et > parsed._instrumentStates.back().ephemerisTimeSeconds + 1.0e-7
        || first_line_et < parsed._bodyRotations.front().ephemerisTimeSeconds - 1.0e-7
        || last_line_et > parsed._bodyRotations.back().ephemerisTimeSeconds + 1.0e-7
        || first_line_et < parsed._instrumentPointing.front().ephemerisTimeSeconds - 1.0e-7
        || last_line_et > parsed._instrumentPointing.back().ephemerisTimeSeconds + 1.0e-7)
    {
        setError(errorMessage, "ISD trajectory or attitude tables do not cover all image-line exposure times");
        return false;
    }
    *this = std::move(parsed);
    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

} // namespace xjw
