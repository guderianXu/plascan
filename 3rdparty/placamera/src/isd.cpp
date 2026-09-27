#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "placamera/isd.h"

namespace placamera
{
    namespace
    {
        using Json = nlohmann::json;
        constexpr TimeScale kTimeScale = TimeScale::Tdb;

        void setError(std::string* error, const std::string& message)
        {
            if (error)
            {
                *error = message;
            }
        }

        std::string pathText(const std::filesystem::path& path)
        {
            const auto utf8 = path.u8string();
            return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
        }

        const Json& member(const Json& object, const char* key)
        {
            static const Json missing;
            if (!object.is_object())
            {
                return missing;
            }
            const auto found = object.find(key);
            return found == object.end() ? missing : found.value();
        }

        bool finiteNumber(const Json& object, const char* key, double* value, std::string* error)
        {
            const Json& item = member(object, key);
            if (!item.is_number() || !std::isfinite(item.get<double>()))
            {
                setError(error, std::string("ISD field '") + key + "' must be a finite number");
                return false;
            }
            *value = item.get<double>();
            return true;
        }

        bool optionalFiniteNumber(const Json& object, const char* key, double* value, std::string* error)
        {
            if (!object.is_object() || object.find(key) == object.end())
            {
                return true;
            }
            return finiteNumber(object, key, value, error);
        }

        bool positiveInteger(const Json& object, const char* key, int* value, std::string* error)
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

        bool nonEmptyString(const Json& object, const char* key, std::string* value, std::string* error)
        {
            const Json& item = member(object, key);
            if (!item.is_string())
            {
                setError(error, std::string("ISD field '") + key + "' must be a non-empty string");
                return false;
            }
            *value = item.get<std::string>();
            if (std::all_of(value->begin(), value->end(), [](unsigned char ch) { return std::isspace(ch) != 0; }))
            {
                setError(error, std::string("ISD field '") + key + "' must be a non-empty string");
                return false;
            }
            return true;
        }

        template <std::size_t Size>
        bool
        finiteArray(const Json& value, const std::string& path, std::array<double, Size>* output, std::string* error)
        {
            if (!value.is_array() || value.size() != Size)
            {
                setError(error, path + " must contain exactly " + std::to_string(Size) + " numbers");
                return false;
            }
            for (std::size_t index = 0; index < Size; ++index)
            {
                if (!value[index].is_number() || !std::isfinite(value[index].get<double>()))
                {
                    setError(error, path + " contains a non-finite or non-numeric value");
                    return false;
                }
                (*output)[index] = value[index].get<double>();
            }
            return true;
        }

        bool parseOrientation(const Json& orientation,
                              const std::string& name,
                              FrameRotationTrajectory* trajectory,
                              std::string* error)
        {
            const Json& constant = member(orientation, "constant_rotation");
            if (!constant.is_null() &&
                !finiteArray<9>(constant, name + ".constant_rotation", &trajectory->constantRotation, error))
            {
                return false;
            }
            try
            {
                (void)Pose::create(FrameId("isd-rotation-validation"), {0.0, 0.0, 0.0}, trajectory->constantRotation);
            }
            catch (const std::exception&)
            {
                setError(error, name + ".constant_rotation must be a proper orthonormal rotation");
                return false;
            }

            const Json& times = member(orientation, "ephemeris_times");
            const Json& quaternions = member(orientation, "quaternions");
            if (!times.is_array() || !quaternions.is_array() || times.size() < 2 || times.size() != quaternions.size())
            {
                setError(error, name + " requires matching ephemeris_times and quaternions");
                return false;
            }
            for (std::size_t index = 0; index < times.size(); ++index)
            {
                std::array<double, 4> quaternion{};
                if (!times[index].is_number() || !std::isfinite(times[index].get<double>()) ||
                    !finiteArray<4>(quaternions[index], name + ".quaternions", &quaternion, error))
                {
                    setError(error, name + " contains an invalid time or quaternion");
                    return false;
                }
                trajectory->samples.push_back(
                    {TimeReference::create(kTimeScale, times[index].get<double>()), quaternion});
            }
            return true;
        }

        bool parseStates(const Json& position, std::vector<TranslationalStateSample>* states, std::string* error)
        {
            const Json& times = member(position, "ephemeris_times");
            const Json& positions = member(position, "positions");
            const Json& velocities = member(position, "velocities");
            if (!times.is_array() || !positions.is_array() || !velocities.is_array() || times.size() < 2 ||
                times.size() != positions.size() || times.size() != velocities.size())
            {
                setError(error, "instrument_position requires matching times, positions and velocities");
                return false;
            }
            for (std::size_t index = 0; index < times.size(); ++index)
            {
                std::array<double, 3> position_kilometers{};
                std::array<double, 3> velocity_kilometers_per_second{};
                if (!times[index].is_number() || !std::isfinite(times[index].get<double>()) ||
                    !finiteArray<3>(positions[index], "instrument_position.positions", &position_kilometers, error) ||
                    !finiteArray<3>(
                        velocities[index], "instrument_position.velocities", &velocity_kilometers_per_second, error))
                {
                    return false;
                }
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    position_kilometers[axis] *= 1000.0;
                    velocity_kilometers_per_second[axis] *= 1000.0;
                }
                states->push_back({TimeReference::create(kTimeScale, times[index].get<double>()),
                                   position_kilometers,
                                   velocity_kilometers_per_second});
            }
            return true;
        }

        bool parseTiming(const Json& root, double center_time, LineTiming* timing, std::string* error)
        {
            const Json& rates = member(root, "line_scan_rate");
            if (!rates.is_array() || rates.empty())
            {
                setError(error, "ISD line_scan_rate must contain at least one record");
                return false;
            }
            timing->timeScale = kTimeScale;
            for (std::size_t index = 0; index < rates.size(); ++index)
            {
                std::array<double, 3> rate{};
                if (!finiteArray<3>(rates[index], "line_scan_rate[" + std::to_string(index) + "]", &rate, error) ||
                    !(rate[2] > 0.0))
                {
                    setError(error, "ISD line_scan_rate records require positive line periods");
                    return false;
                }
                timing->segments.push_back({rate[0], center_time + rate[1] + 0.5 * rate[2], rate[2]});
            }
            timing->lineZero = timing->segments.front().startLine;
            timing->startTimeSeconds = timing->segments.front().startTimeSeconds;
            timing->secondsPerLine = timing->segments.front().secondsPerLine;
            return true;
        }

        bool coversImage(const LineScanModel& instance)
        {
            const auto first_time = instance.timeForLine(0.5);
            const auto last_time = instance.timeForLine(static_cast<double>(instance.imageSize().lines) - 0.5);
            return first_time && last_time &&
                   instance.trajectory().poseAt(first_time.value(), instance.groundFrame()).ok() &&
                   instance.trajectory().poseAt(last_time.value(), instance.groundFrame()).ok();
        }

        bool parseCompleteCalibration(const Json& root, LineScanOptics* optics, std::string* error)
        {
            const Json& preferred = member(root, "complete_calibration");
            const Json& legacy_name = member(root, "metashape_calibration");
            const Json& encoded = !preferred.is_null() ? preferred : legacy_name;
            if (encoded.is_null())
            {
                return true;
            }
            if (!encoded.is_object())
            {
                setError(error, "ISD complete_calibration must be an object");
                return false;
            }

            MetashapeCalibration calibration;
            if (!finiteNumber(encoded, "f", &calibration.f, error) ||
                !optionalFiniteNumber(encoded, "sample_pitch_mm", &optics->samplePitchMillimeters, error) ||
                !optionalFiniteNumber(encoded, "principal_sample", &optics->principalSample, error) ||
                !optionalFiniteNumber(encoded, "cx", &calibration.cx, error) ||
                !optionalFiniteNumber(encoded, "cy", &calibration.cy, error) ||
                !optionalFiniteNumber(encoded, "b1", &calibration.b1, error) ||
                !optionalFiniteNumber(encoded, "b2", &calibration.b2, error) ||
                !optionalFiniteNumber(encoded, "k1", &calibration.k1, error) ||
                !optionalFiniteNumber(encoded, "k2", &calibration.k2, error) ||
                !optionalFiniteNumber(encoded, "k3", &calibration.k3, error) ||
                !optionalFiniteNumber(encoded, "k4", &calibration.k4, error) ||
                !optionalFiniteNumber(encoded, "p1", &calibration.p1, error) ||
                !optionalFiniteNumber(encoded, "p2", &calibration.p2, error) ||
                !optionalFiniteNumber(encoded, "p3", &calibration.p3, error) ||
                !optionalFiniteNumber(encoded, "p4", &calibration.p4, error))
            {
                return false;
            }

            const Json& center = member(encoded, "image_center");
            const bool has_cx_offset = encoded.find("cx_offset") != encoded.end();
            const bool has_cy_offset = encoded.find("cy_offset") != encoded.end();
            if (!center.is_null() || has_cx_offset || has_cy_offset)
            {
                std::array<double, 2> image_center{};
                PrincipalPointDecomposition principal;
                if (!finiteArray<2>(center, "complete_calibration.image_center", &image_center, error) ||
                    !finiteNumber(encoded, "cx_offset", &principal.cxOffset, error) ||
                    !finiteNumber(encoded, "cy_offset", &principal.cyOffset, error))
                {
                    setError(error, "ISD complete_calibration requires image_center, cx_offset and cy_offset together");
                    return false;
                }
                principal.imageCenterX = image_center[0];
                principal.imageCenterY = image_center[1];
                calibration.principalPointDecomposition = principal;
            }
            optics->completeCalibration = calibration;
            return true;
        }
    } // namespace

    static bool parsePlanetaryLineScanIsdImpl(std::string_view json,
                                              CameraDefinitionId definitionId,
                                              CameraInstanceId instanceId,
                                              ImageId imageId,
                                              PlanetaryLineScanIsdImport* imported,
                                              std::string* error)
    {
        if (!imported)
        {
            setError(error, "line-scan ISD output is null");
            return false;
        }
        Json root;
        try
        {
            root = Json::parse(json.begin(), json.end());
        }
        catch (const Json::parse_error& exception)
        {
            setError(error, "Unable to parse USGSCSM ISD JSON object: " + std::string(exception.what()));
            return false;
        }
        if (!root.is_object())
        {
            setError(error, "Unable to parse USGSCSM ISD JSON object: root is not an object");
            return false;
        }

        PlanetaryLineScanIsdMetadata metadata;
        int image_lines = 0;
        int image_samples = 0;
        if (!positiveInteger(root, "image_lines", &image_lines, error) ||
            !positiveInteger(root, "image_samples", &image_samples, error) ||
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

        double body_frame_code = 0.0;
        if (!finiteNumber(member(root, "naif_keywords"), "BODY_FRAME_CODE", &body_frame_code, error) ||
            body_frame_code != 31001.0)
        {
            setError(error, "line-scan ISD requires BODY_FRAME_CODE=31001 (MOON_ME)");
            return false;
        }
        metadata.bodyFixedFrameCode = 31001;
        metadata.targetName = "MOON";
        metadata.bodyFixedFrameName = "MOON_ME";

        LineScanOptics optics;
        LineScanDetectorGeometry detector;
        const Json& focal_model = member(root, "focal_length_model");
        const Json& detector_center = member(root, "detector_center");
        if (!finiteNumber(focal_model, "focal_length", &optics.focalLengthMillimeters, error) ||
            !finiteNumber(detector_center, "sample", &detector.detectorSampleOrigin, error) ||
            !finiteNumber(detector_center, "line", &detector.detectorLineOrigin, error) ||
            !finiteNumber(root, "detector_sample_summing", &detector.detectorSampleSumming, error) ||
            !finiteNumber(root, "detector_line_summing", &detector.detectorLineSumming, error) ||
            !finiteNumber(root, "starting_detector_sample", &detector.startingDetectorSample, error) ||
            !finiteNumber(root, "starting_detector_line", &detector.startingDetectorLine, error) ||
            !finiteArray<3>(
                member(root, "focal2pixel_samples"), "focal2pixel_samples", &detector.focalToPixelSamples, error) ||
            !finiteArray<3>(member(root, "focal2pixel_lines"), "focal2pixel_lines", &detector.focalToPixelLines, error))
        {
            return false;
        }
        metadata.focalLengthMillimeters = optics.focalLengthMillimeters;
        metadata.detectorSampleSumming = detector.detectorSampleSumming;
        metadata.detectorLineSumming = detector.detectorLineSumming;
        optics.detectorGeometry = detector;
        optics.distortionModel = LineScanDistortionModel::LroNacFocalPlane;
        std::array<double, 1> distortion_coefficients{};
        if (!finiteArray<1>(member(member(member(root, "optical_distortion"), "lrolrocnac"), "coefficients"),
                            "optical_distortion.lrolrocnac.coefficients",
                            &distortion_coefficients,
                            error))
        {
            setError(error, "line-scan ISD requires the LRO LROC NAC distortion model");
            return false;
        }
        optics.distortionK1 = distortion_coefficients[0];
        if (!parseCompleteCalibration(root, &optics, error))
        {
            return false;
        }

        LineTiming timing;
        FrameComposedTrajectory trajectory;
        const Json& body = member(root, "body_rotation");
        const Json& pointing = member(root, "instrument_pointing");
        const Json& position = member(root, "instrument_position");
        if (!parseTiming(root, metadata.centerEphemerisTimeSeconds, &timing, error) || !body.is_object() ||
            !pointing.is_object() || !position.is_object() ||
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
                                                         FrameId(metadata.bodyFixedFrameName),
                                                         optics,
                                                         LineScanPixelConvention::PixelCenter);
            auto instance =
                LineScanModel::create(std::move(instanceId),
                                      std::move(imageId),
                                      std::move(definition),
                                      ImageSize{image_samples, image_lines},
                                      LineScanTrajectory::createFrameComposed(std::move(trajectory)),
                                      std::move(timing),
                                      {},
                                      TimeReference::create(kTimeScale, metadata.centerEphemerisTimeSeconds));
            if (!coversImage(instance))
            {
                setError(error, "ISD trajectory or attitude tables do not cover all image-line exposure times");
                return false;
            }
            imported->instance = std::make_shared<const LineScanModel>(std::move(instance));
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

    Result<PlanetaryLineScanIsdImport> parsePlanetaryLineScanIsd(std::string_view json,
                                                                 CameraDefinitionId definitionId,
                                                                 CameraInstanceId instanceId,
                                                                 ImageId imageId)
    {
        PlanetaryLineScanIsdImport imported;
        std::string error;
        if (!parsePlanetaryLineScanIsdImpl(
                json, std::move(definitionId), std::move(instanceId), std::move(imageId), &imported, &error))
        {
            return Result<PlanetaryLineScanIsdImport>::failure(
                CameraErrorCode::ParseFailure, std::move(error), "USGSCSM ISD JSON");
        }
        return Result<PlanetaryLineScanIsdImport>::success(std::move(imported));
    }

    Result<PlanetaryLineScanIsdImport> importPlanetaryLineScanIsd(const std::filesystem::path& path,
                                                                  CameraDefinitionId definitionId,
                                                                  CameraInstanceId instanceId,
                                                                  ImageId imageId)
    {
        if (path.empty())
        {
            return Result<PlanetaryLineScanIsdImport>::failure(CameraErrorCode::InvalidArgument,
                                                               "line-scan ISD path is empty");
        }
        std::ifstream input(path);
        if (!input)
        {
            return Result<PlanetaryLineScanIsdImport>::failure(
                CameraErrorCode::IoFailure, "cannot open line-scan ISD file", pathText(path));
        }
        std::ostringstream bytes;
        bytes << input.rdbuf();
        if (input.bad())
        {
            return Result<PlanetaryLineScanIsdImport>::failure(
                CameraErrorCode::IoFailure, "cannot read line-scan ISD file", pathText(path));
        }
        auto imported =
            parsePlanetaryLineScanIsd(bytes.str(), std::move(definitionId), std::move(instanceId), std::move(imageId));
        if (!imported)
        {
            CameraError error = imported.error();
            error.source = pathText(path);
            return Result<PlanetaryLineScanIsdImport>::failure(std::move(error));
        }
        return imported;
    }
} // namespace placamera
