#include "PlanetaryLineScanIsdIO.h"

#include "io/PathIO.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QString>

#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

namespace xjw::camera_models::linescan
{
    namespace
    {

        constexpr xjw::coordinate_system::TimeScale kTimeScale = xjw::coordinate_system::TimeScale::Tdb;

        void setError(std::string* error, const std::string& message)
        {
            if (error)
            {
                *error = message;
            }
        }

        bool finiteNumber(const QJsonObject& object, const char* key, double* value, std::string* error)
        {
            const QJsonValue item = object.value(QLatin1String(key));
            if (!item.isDouble() || !std::isfinite(item.toDouble()))
            {
                setError(error, std::string("ISD field '") + key + "' must be a finite number");
                return false;
            }
            *value = item.toDouble();
            return true;
        }

        bool positiveInteger(const QJsonObject& object, const char* key, int* value, std::string* error)
        {
            double parsed = 0.0;
            if (!finiteNumber(object, key, &parsed, error) || parsed < 1.0 ||
                parsed > static_cast<double>(std::numeric_limits<int>::max()) || std::floor(parsed) != parsed)
            {
                setError(error, std::string("ISD field '") + key + "' must be a positive integer");
                return false;
            }
            *value = static_cast<int>(parsed);
            return true;
        }

        bool nonEmptyString(const QJsonObject& object, const char* key, std::string* value, std::string* error)
        {
            const QJsonValue item = object.value(QLatin1String(key));
            if (!item.isString() || item.toString().trimmed().isEmpty())
            {
                setError(error, std::string("ISD field '") + key + "' must be a non-empty string");
                return false;
            }
            *value = item.toString().toUtf8().toStdString();
            return true;
        }

        template <std::size_t Size>
        bool finiteArray(const QJsonValue& value,
                         const std::string& path,
                         std::array<double, Size>* output,
                         std::string* error)
        {
            if (!value.isArray() || value.toArray().size() != static_cast<int>(Size))
            {
                setError(error, path + " must contain exactly " + std::to_string(Size) + " numbers");
                return false;
            }
            const QJsonArray values = value.toArray();
            for (std::size_t index = 0; index < Size; ++index)
            {
                const QJsonValue item = values.at(static_cast<int>(index));
                if (!item.isDouble() || !std::isfinite(item.toDouble()))
                {
                    setError(error, path + " contains a non-finite or non-numeric value");
                    return false;
                }
                (*output)[index] = item.toDouble();
            }
            return true;
        }

        bool parseConstantRotation(const QJsonObject& orientation,
                                   const std::string& path,
                                   camera_core::Rotation* rotation,
                                   std::string* error)
        {
            const QJsonValue value = orientation.value(QStringLiteral("constant_rotation"));
            if (!value.isUndefined() && !finiteArray<9>(value, path + ".constant_rotation", rotation, error))
            {
                return false;
            }
            try
            {
                (void)camera_core::Pose::create(xjw::coordinate_system::CoordinateFrameId("isd-rotation-validation"),
                                                {0.0, 0.0, 0.0},
                                                *rotation);
            }
            catch (const std::exception&)
            {
                setError(error, path + ".constant_rotation must be a proper orthonormal rotation");
                return false;
            }
            return true;
        }

        bool parseOrientation(const QJsonObject& orientation,
                              const std::string& name,
                              FrameRotationTrajectory* trajectory,
                              std::string* error)
        {
            if (!parseConstantRotation(orientation, name, &trajectory->constantRotation, error))
            {
                return false;
            }
            const QJsonArray times = orientation.value(QStringLiteral("ephemeris_times")).toArray();
            const QJsonArray quaternions = orientation.value(QStringLiteral("quaternions")).toArray();
            if (times.size() < 2 || times.size() != quaternions.size())
            {
                setError(error, name + " requires matching ephemeris_times and quaternions");
                return false;
            }
            for (int index = 0; index < times.size(); ++index)
            {
                std::array<double, 4> quaternion{};
                const double seconds = times.at(index).toDouble(std::numeric_limits<double>::quiet_NaN());
                if (!std::isfinite(seconds) ||
                    !finiteArray<4>(quaternions.at(index), name + ".quaternions", &quaternion, error))
                {
                    setError(error, name + " contains an invalid time or quaternion");
                    return false;
                }
                trajectory->samples.push_back(
                    {xjw::coordinate_system::TimeReference::create(kTimeScale, seconds), quaternion});
            }
            return true;
        }

        bool parseStates(const QJsonObject& position,
                         std::vector<TranslationalStateSample>* states,
                         std::string* error)
        {
            const QJsonArray times = position.value(QStringLiteral("ephemeris_times")).toArray();
            const QJsonArray positions = position.value(QStringLiteral("positions")).toArray();
            const QJsonArray velocities = position.value(QStringLiteral("velocities")).toArray();
            if (times.size() < 2 || times.size() != positions.size() || times.size() != velocities.size())
            {
                setError(error, "instrument_position requires matching times, positions and velocities");
                return false;
            }
            for (int index = 0; index < times.size(); ++index)
            {
                const double seconds = times.at(index).toDouble(std::numeric_limits<double>::quiet_NaN());
                std::array<double, 3> positionKilometers{};
                std::array<double, 3> velocityKilometersPerSecond{};
                if (!std::isfinite(seconds) ||
                    !finiteArray<3>(positions.at(index), "instrument_position.positions", &positionKilometers, error) ||
                    !finiteArray<3>(velocities.at(index),
                                    "instrument_position.velocities",
                                    &velocityKilometersPerSecond,
                                    error))
                {
                    return false;
                }
                for (int axis = 0; axis < 3; ++axis)
                {
                    positionKilometers[static_cast<std::size_t>(axis)] *= 1000.0;
                    velocityKilometersPerSecond[static_cast<std::size_t>(axis)] *= 1000.0;
                }
                states->push_back({xjw::coordinate_system::TimeReference::create(kTimeScale, seconds),
                                   positionKilometers,
                                   velocityKilometersPerSecond});
            }
            return true;
        }

        bool parseTiming(const QJsonObject& root,
                         double centerTime,
                         LineTiming* timing,
                         std::string* error)
        {
            const QJsonArray rates = root.value(QStringLiteral("line_scan_rate")).toArray();
            if (rates.isEmpty())
            {
                setError(error, "ISD line_scan_rate must contain at least one record");
                return false;
            }
            timing->timeScale = kTimeScale;
            for (int index = 0; index < rates.size(); ++index)
            {
                std::array<double, 3> rate{};
                if (!finiteArray<3>(rates.at(index),
                                    "line_scan_rate[" + std::to_string(index) + "]",
                                    &rate,
                                    error) ||
                    !(rate[2] > 0.0))
                {
                    setError(error, "ISD line_scan_rate records require positive line periods");
                    return false;
                }
                timing->segments.push_back({rate[0], centerTime + rate[1] + 0.5 * rate[2], rate[2]});
            }
            timing->lineZero = timing->segments.front().startLine;
            timing->startTimeSeconds = timing->segments.front().startTimeSeconds;
            timing->secondsPerLine = timing->segments.front().secondsPerLine;
            return true;
        }

        bool coversImage(const LineScanInstance& instance)
        {
            double firstTime = 0.0;
            double lastTime = 0.0;
            if (!instance.timeForLine(0.5, &firstTime) ||
                !instance.timeForLine(static_cast<double>(instance.imageSize().lines) - 0.5, &lastTime))
            {
                return false;
            }
            try
            {
                (void)instance.trajectory().poseAt(xjw::coordinate_system::TimeReference::create(kTimeScale, firstTime),
                                                   instance.definition().worldFrame());
                (void)instance.trajectory().poseAt(xjw::coordinate_system::TimeReference::create(kTimeScale, lastTime),
                                                   instance.definition().worldFrame());
            }
            catch (const std::exception&)
            {
                return false;
            }
            return true;
        }

    } // namespace

    bool importPlanetaryLineScanIsd(const std::string& path,
                                    camera_core::CameraDefinitionId definitionId,
                                    camera_core::CameraInstanceId instanceId,
                                    camera_core::ImageId imageId,
                                    PlanetaryLineScanIsdImport* imported,
                                    std::string* error)
    {
        if (!imported || path.empty())
        {
            setError(error, imported ? "line-scan ISD path is empty" : "line-scan ISD output is null");
            return false;
        }
        QString readError;
        const QByteArray bytes = common::io::readFileBytes(path, &readError);
        if (!readError.isEmpty())
        {
            setError(error, readError.toUtf8().toStdString());
            return false;
        }
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject())
        {
            setError(error,
                     "Unable to parse USGSCSM ISD JSON object: " + parseError.errorString().toUtf8().toStdString());
            return false;
        }

        const QJsonObject root = document.object();
        PlanetaryLineScanIsdMetadata metadata;
        int imageLines = 0;
        int imageSamples = 0;
        if (!positiveInteger(root, "image_lines", &imageLines, error) ||
            !positiveInteger(root, "image_samples", &imageSamples, error) ||
            !nonEmptyString(root, "name_model", &metadata.modelName, error) ||
            !nonEmptyString(root, "name_platform", &metadata.platformName, error) ||
            !nonEmptyString(root, "name_sensor", &metadata.sensorName, error) ||
            !nonEmptyString(root, "interpolation_method", &metadata.interpolationMethod, error) ||
            !finiteNumber(root, "starting_ephemeris_time", &metadata.startingEphemerisTimeSeconds, error) ||
            !finiteNumber(root, "center_ephemeris_time", &metadata.centerEphemerisTimeSeconds, error))
        {
            return false;
        }
        if (metadata.modelName != "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL")
        {
            setError(error, "ISD camera model is not USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL");
            return false;
        }
        if (metadata.interpolationMethod != "lagrange")
        {
            setError(error,
                     "line-scan import only accepts ISD interpolation_method='lagrange'; "
                     "raw-table Hermite/SLERP evaluation is an explicit approximation");
            return false;
        }

        const QJsonObject naif = root.value(QStringLiteral("naif_keywords")).toObject();
        double bodyFrameCode = 0.0;
        if (naif.isEmpty() || !finiteNumber(naif, "BODY_FRAME_CODE", &bodyFrameCode, error) ||
            bodyFrameCode != 31001.0)
        {
            setError(error, "line-scan ISD requires BODY_FRAME_CODE=31001 (MOON_ME)");
            return false;
        }
        metadata.bodyFixedFrameCode = 31001;
        metadata.targetName = "MOON";
        metadata.bodyFixedFrameName = "MOON_ME";

        LineScanOptics optics;
        LineScanDetectorGeometry detector;
        const QJsonObject focalModel = root.value(QStringLiteral("focal_length_model")).toObject();
        const QJsonObject detectorCenter = root.value(QStringLiteral("detector_center")).toObject();
        if (focalModel.isEmpty() || detectorCenter.isEmpty() ||
            !finiteNumber(focalModel, "focal_length", &optics.focalLengthMillimeters, error) ||
            !finiteNumber(detectorCenter, "sample", &detector.detectorSampleOrigin, error) ||
            !finiteNumber(detectorCenter, "line", &detector.detectorLineOrigin, error) ||
            !finiteNumber(root, "detector_sample_summing", &detector.detectorSampleSumming, error) ||
            !finiteNumber(root, "detector_line_summing", &detector.detectorLineSumming, error) ||
            !finiteNumber(root, "starting_detector_sample", &detector.startingDetectorSample, error) ||
            !finiteNumber(root, "starting_detector_line", &detector.startingDetectorLine, error) ||
            !finiteArray<3>(root.value(QStringLiteral("focal2pixel_samples")),
                            "focal2pixel_samples",
                            &detector.focalToPixelSamples,
                            error) ||
            !finiteArray<3>(root.value(QStringLiteral("focal2pixel_lines")),
                            "focal2pixel_lines",
                            &detector.focalToPixelLines,
                            error))
        {
            return false;
        }
        metadata.focalLengthMillimeters = optics.focalLengthMillimeters;
        metadata.detectorSampleSumming = detector.detectorSampleSumming;
        metadata.detectorLineSumming = detector.detectorLineSumming;
        optics.detectorGeometry = detector;
        optics.distortionModel = LineScanDistortionModel::LroNacFocalPlane;
        const QJsonObject distortion = root.value(QStringLiteral("optical_distortion")).toObject();
        const QJsonObject lroNac = distortion.value(QStringLiteral("lrolrocnac")).toObject();
        std::array<double, 1> distortionCoefficients{};
        if (lroNac.isEmpty() ||
            !finiteArray<1>(lroNac.value(QStringLiteral("coefficients")),
                            "optical_distortion.lrolrocnac.coefficients",
                            &distortionCoefficients,
                            error))
        {
            setError(error, "line-scan ISD requires the LRO LROC NAC distortion model");
            return false;
        }
        optics.distortionK1 = distortionCoefficients[0];

        LineTiming timing;
        FrameComposedTrajectory trajectory;
        const QJsonObject body = root.value(QStringLiteral("body_rotation")).toObject();
        const QJsonObject pointing = root.value(QStringLiteral("instrument_pointing")).toObject();
        const QJsonObject position = root.value(QStringLiteral("instrument_position")).toObject();
        if (!parseTiming(root, metadata.centerEphemerisTimeSeconds, &timing, error) || body.isEmpty() ||
            pointing.isEmpty() || position.isEmpty() ||
            !parseOrientation(body, "body_rotation", &trajectory.inertialToWorld, error) ||
            !parseOrientation(pointing, "instrument_pointing", &trajectory.inertialToSensor, error) ||
            !parseStates(position, &trajectory.inertialStates, error))
        {
            return false;
        }
        if (std::abs(timing.segments.front().startTimeSeconds - 0.5 * timing.segments.front().secondsPerLine -
                     metadata.startingEphemerisTimeSeconds) > 1.0e-5)
        {
            setError(error, "ISD starting time is inconsistent with the first line-scan-rate record");
            return false;
        }

        try
        {
            auto definition = LineScanDefinition::create(std::move(definitionId),
                                                         xjw::coordinate_system::CoordinateFrameId(metadata.bodyFixedFrameName),
                                                         optics,
                                                         PixelConvention::PixelCenter);
            std::unique_ptr<LineScanInstance> instance = LineScanInstance::createUnique(
                std::move(instanceId),
                std::move(imageId),
                std::move(definition),
                camera_core::ImageSize{imageSamples, imageLines},
                LineScanTrajectory::createFrameComposed(std::move(trajectory)),
                std::move(timing),
                xjw::coordinate_system::TimeReference::create(kTimeScale, metadata.centerEphemerisTimeSeconds));
            if (!coversImage(*instance))
            {
                setError(error, "ISD trajectory or attitude tables do not cover all image-line exposure times");
                return false;
            }
            imported->instance = std::shared_ptr<const LineScanInstance>(std::move(instance));
            imported->metadata = std::move(metadata);
        }
        catch (const std::exception& exception)
        {
            setError(error, exception.what());
            return false;
        }
        if (error)
        {
            error->clear();
        }
        return true;
    }

} // namespace xjw::camera_models::linescan
