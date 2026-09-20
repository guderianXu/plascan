#include "LineScanModelJson.h"

#include "camera/core/types/CameraErrors.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QJsonValue>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace xjw::camera_models
{
    namespace
    {

        std::optional<QJsonObject> parseObject(std::string_view serialized)
        {
            if (serialized.empty())
            {
                return std::nullopt;
            }
            QJsonParseError error;
            const QJsonDocument document = QJsonDocument::fromJson(
                QByteArray(serialized.data(), static_cast<qsizetype>(serialized.size())), &error);
            if (error.error != QJsonParseError::NoError || !document.isObject())
            {
                return std::nullopt;
            }
            return document.object();
        }

        bool number(const QJsonObject& object, const QString& key, double* output)
        {
            const QJsonValue value = object.value(key);
            if (!output || !value.isDouble() || !std::isfinite(value.toDouble()))
            {
                return false;
            }
            *output = value.toDouble();
            return true;
        }

        template <std::size_t Size> bool numberArray(const QJsonValue& value, std::array<double, Size>* output)
        {
            if (!output || !value.isArray() || value.toArray().size() != static_cast<int>(Size))
            {
                return false;
            }
            const QJsonArray values = value.toArray();
            for (std::size_t index = 0; index < Size; ++index)
            {
                const QJsonValue item = values.at(static_cast<int>(index));
                if (!item.isDouble() || !std::isfinite(item.toDouble()))
                {
                    return false;
                }
                (*output)[index] = item.toDouble();
            }
            return true;
        }

        std::optional<xjw::coordinate_system::TimeScale> timeScale(const QJsonValue& value)
        {
            if (!value.isString())
            {
                return std::nullopt;
            }
            const QString name = value.toString();
            if (name == QStringLiteral("utc"))
            {
                return xjw::coordinate_system::TimeScale::Utc;
            }
            if (name == QStringLiteral("tai"))
            {
                return xjw::coordinate_system::TimeScale::Tai;
            }
            if (name == QStringLiteral("tdb"))
            {
                return xjw::coordinate_system::TimeScale::Tdb;
            }
            if (name == QStringLiteral("relative"))
            {
                return xjw::coordinate_system::TimeScale::Relative;
            }
            return std::nullopt;
        }

        bool imageSize(const QJsonObject& state, camera_core::ImageSize* output)
        {
            if (!output || !state.value(QStringLiteral("image_size")).isObject())
            {
                return false;
            }
            const QJsonObject size = state.value(QStringLiteral("image_size")).toObject();
            double samples = 0.0;
            double lines = 0.0;
            if (!number(size, QStringLiteral("samples"), &samples) || !number(size, QStringLiteral("lines"), &lines) ||
                samples < 1.0 || lines < 1.0 || samples > static_cast<double>(std::numeric_limits<int>::max()) ||
                lines > static_cast<double>(std::numeric_limits<int>::max()) || std::floor(samples) != samples ||
                std::floor(lines) != lines)
            {
                return false;
            }
            *output = {static_cast<int>(samples), static_cast<int>(lines)};
            return true;
        }

        bool captureTime(const QJsonObject& state, std::optional<xjw::coordinate_system::TimeReference>* output)
        {
            if (!output)
            {
                return false;
            }
            *output = std::nullopt;
            if (!state.contains(QStringLiteral("capture_time")))
            {
                return true;
            }
            if (!state.value(QStringLiteral("capture_time")).isObject())
            {
                return false;
            }
            const QJsonObject value = state.value(QStringLiteral("capture_time")).toObject();
            const auto scale = timeScale(value.value(QStringLiteral("time_scale")));
            double seconds = 0.0;
            if (!scale || !number(value, QStringLiteral("seconds"), &seconds))
            {
                return false;
            }
            *output = xjw::coordinate_system::TimeReference::create(*scale, seconds);
            return true;
        }

        bool directTrajectory(const QJsonObject& object,
                              xjw::coordinate_system::TimeScale scale,
                              std::optional<linescan::LineScanTrajectory>* output)
        {
            const QJsonArray values = object.value(QStringLiteral("samples")).toArray();
            if (values.size() < 2)
            {
                return false;
            }
            std::vector<linescan::TrajectorySample> samples;
            samples.reserve(values.size());
            for (const QJsonValue& value : values)
            {
                if (!value.isObject())
                {
                    return false;
                }
                const QJsonObject item = value.toObject();
                linescan::TrajectorySample sample;
                double seconds = 0.0;
                if (!number(item, QStringLiteral("time_seconds"), &seconds) ||
                    !numberArray(item.value(QStringLiteral("center_m")), &sample.center) ||
                    !numberArray(item.value(QStringLiteral("camera_to_world_rotation")), &sample.cameraToWorldRotation))
                {
                    return false;
                }
                sample.time = xjw::coordinate_system::TimeReference::create(scale, seconds);
                samples.push_back(sample);
            }
            output->emplace(linescan::LineScanTrajectory::create(std::move(samples)));
            return true;
        }

        bool rotationTrajectory(const QJsonObject& object,
                                xjw::coordinate_system::TimeScale scale,
                                linescan::FrameRotationTrajectory* output)
        {
            if (!output || !numberArray(object.value(QStringLiteral("constant_rotation")), &output->constantRotation))
            {
                return false;
            }
            const QJsonArray values = object.value(QStringLiteral("samples")).toArray();
            if (values.size() < 2)
            {
                return false;
            }
            output->samples.reserve(values.size());
            for (const QJsonValue& value : values)
            {
                if (!value.isObject())
                {
                    return false;
                }
                const QJsonObject item = value.toObject();
                linescan::QuaternionTrajectorySample sample;
                double seconds = 0.0;
                if (!number(item, QStringLiteral("time_seconds"), &seconds) ||
                    !numberArray(item.value(QStringLiteral("quaternion_scalar_first")), &sample.scalarFirst))
                {
                    return false;
                }
                sample.time = xjw::coordinate_system::TimeReference::create(scale, seconds);
                output->samples.push_back(sample);
            }
            return true;
        }

        bool frameComposedTrajectory(const QJsonObject& object,
                                     xjw::coordinate_system::TimeScale scale,
                                     std::optional<linescan::LineScanTrajectory>* output)
        {
            const QJsonArray values = object.value(QStringLiteral("inertial_states")).toArray();
            const QJsonValue worldValue = object.value(QStringLiteral("inertial_to_world"));
            const QJsonValue sensorValue = object.value(QStringLiteral("inertial_to_sensor"));
            if (values.size() < 2 || !worldValue.isObject() || !sensorValue.isObject())
            {
                return false;
            }
            linescan::FrameComposedTrajectory trajectory;
            trajectory.inertialStates.reserve(values.size());
            for (const QJsonValue& value : values)
            {
                if (!value.isObject())
                {
                    return false;
                }
                const QJsonObject item = value.toObject();
                linescan::TranslationalStateSample state;
                double seconds = 0.0;
                if (!number(item, QStringLiteral("time_seconds"), &seconds) ||
                    !numberArray(item.value(QStringLiteral("position_m")), &state.positionMeters) ||
                    !numberArray(item.value(QStringLiteral("velocity_m_per_s")), &state.velocityMetersPerSecond))
                {
                    return false;
                }
                state.time = xjw::coordinate_system::TimeReference::create(scale, seconds);
                trajectory.inertialStates.push_back(state);
            }
            if (!rotationTrajectory(worldValue.toObject(), scale, &trajectory.inertialToWorld) ||
                !rotationTrajectory(sensorValue.toObject(), scale, &trajectory.inertialToSensor))
            {
                return false;
            }
            output->emplace(linescan::LineScanTrajectory::createFrameComposed(std::move(trajectory)));
            return true;
        }

        bool trajectory(const QJsonObject& state, std::optional<linescan::LineScanTrajectory>* output)
        {
            if (!output || !state.value(QStringLiteral("trajectory")).isObject())
            {
                return false;
            }
            const QJsonObject object = state.value(QStringLiteral("trajectory")).toObject();
            const auto scale = timeScale(object.value(QStringLiteral("time_scale")));
            const QString representation = object.value(QStringLiteral("representation")).toString();
            if (!scale)
            {
                return false;
            }
            if (representation == QStringLiteral("direct_pose_samples"))
            {
                return directTrajectory(object, *scale, output);
            }
            if (representation == QStringLiteral("frame_composed"))
            {
                return frameComposedTrajectory(object, *scale, output);
            }
            return false;
        }

        bool lineTiming(const QJsonObject& state, linescan::LineTiming* output)
        {
            if (!output || !state.value(QStringLiteral("line_timing")).isObject())
            {
                return false;
            }
            const QJsonObject object = state.value(QStringLiteral("line_timing")).toObject();
            const auto scale = timeScale(object.value(QStringLiteral("time_scale")));
            const QJsonArray values = object.value(QStringLiteral("segments")).toArray();
            if (!scale || values.isEmpty())
            {
                return false;
            }
            output->timeScale = *scale;
            output->segments.reserve(values.size());
            for (const QJsonValue& value : values)
            {
                if (!value.isObject())
                {
                    return false;
                }
                const QJsonObject item = value.toObject();
                linescan::LineRateSegment segment;
                if (!number(item, QStringLiteral("start_line"), &segment.startLine) ||
                    !number(item, QStringLiteral("start_time_seconds"), &segment.startTimeSeconds) ||
                    !number(item, QStringLiteral("seconds_per_line"), &segment.secondsPerLine))
                {
                    return false;
                }
                output->segments.push_back(segment);
            }
            output->lineZero = output->segments.front().startLine;
            output->startTimeSeconds = output->segments.front().startTimeSeconds;
            output->secondsPerLine = output->segments.front().secondsPerLine;
            return true;
        }

        bool optics(const QJsonObject& parameters,
                    linescan::LineScanOptics* output,
                    linescan::PixelConvention* pixelConvention)
        {
            if (!output || !pixelConvention || !parameters.value(QStringLiteral("optics")).isObject())
            {
                return false;
            }
            const QJsonObject object = parameters.value(QStringLiteral("optics")).toObject();
            const QString distortion = object.value(QStringLiteral("distortion_model")).toString();
            if (!number(object, QStringLiteral("focal_length_mm"), &output->focalLengthMillimeters) ||
                !number(object, QStringLiteral("distortion_k1"), &output->distortionK1))
            {
                return false;
            }
            if (distortion == QStringLiteral("radial_normalized"))
            {
                output->distortionModel = linescan::LineScanDistortionModel::RadialNormalized;
            }
            else if (distortion == QStringLiteral("lro_nac_focal_plane"))
            {
                output->distortionModel = linescan::LineScanDistortionModel::LroNacFocalPlane;
            }
            else
            {
                return false;
            }

            if (!object.value(QStringLiteral("sample_geometry")).isObject())
            {
                return false;
            }
            const QJsonObject geometry = object.value(QStringLiteral("sample_geometry")).toObject();
            const QString geometryType = geometry.value(QStringLiteral("type")).toString();
            if (geometryType == QStringLiteral("uniform_pitch"))
            {
                if (!number(geometry, QStringLiteral("sample_pitch_mm"), &output->samplePitchMillimeters) ||
                    !number(geometry, QStringLiteral("principal_sample"), &output->principalSample))
                {
                    return false;
                }
            }
            else if (geometryType == QStringLiteral("detector_affine"))
            {
                linescan::LineScanDetectorGeometry detector;
                if (!number(geometry, QStringLiteral("detector_sample_summing"), &detector.detectorSampleSumming) ||
                    !number(geometry, QStringLiteral("detector_line_summing"), &detector.detectorLineSumming) ||
                    !number(geometry, QStringLiteral("detector_sample_origin"), &detector.detectorSampleOrigin) ||
                    !number(geometry, QStringLiteral("detector_line_origin"), &detector.detectorLineOrigin) ||
                    !number(geometry, QStringLiteral("starting_detector_sample"), &detector.startingDetectorSample) ||
                    !number(geometry, QStringLiteral("starting_detector_line"), &detector.startingDetectorLine) ||
                    !numberArray(geometry.value(QStringLiteral("focal_to_pixel_samples")),
                                 &detector.focalToPixelSamples) ||
                    !numberArray(geometry.value(QStringLiteral("focal_to_pixel_lines")), &detector.focalToPixelLines))
                {
                    return false;
                }
                output->detectorGeometry = detector;
            }
            else
            {
                return false;
            }

            const QString convention = parameters.value(QStringLiteral("pixel_convention")).toString();
            if (convention == QStringLiteral("pixel_center"))
            {
                *pixelConvention = linescan::PixelConvention::PixelCenter;
                return true;
            }
            if (convention == QStringLiteral("zero_based"))
            {
                *pixelConvention = linescan::PixelConvention::ZeroBased;
                return true;
            }
            return false;
        }

    } // namespace

    std::unique_ptr<camera_core::CameraDefinition>
    createLineScanDefinitionFromJson(const camera_core::CameraDefinitionId& id,
                                     const xjw::coordinate_system::CoordinateFrameId& frame,
                                     int parameterSchemaVersion,
                                     std::string_view serializedParameters)
    {
        if (parameterSchemaVersion != linescan::LineScanDefinition::ParameterSchemaVersion)
        {
            return {};
        }
        const auto parsed = parseObject(serializedParameters);
        if (!parsed)
        {
            return {};
        }
        linescan::LineScanOptics decodedOptics;
        linescan::PixelConvention convention;
        if (!optics(*parsed, &decodedOptics, &convention))
        {
            return {};
        }
        try
        {
            return linescan::LineScanDefinition::createUnique(id, frame, decodedOptics, convention);
        }
        catch (const std::exception&)
        {
            return {};
        }
    }

    std::unique_ptr<camera_core::CameraInstance>
    createLineScanInstanceFromJson(const camera_core::CameraInstanceId& id,
                                   const camera_core::ImageId& image,
                                   std::shared_ptr<const camera_core::CameraDefinition> definition,
                                   std::string_view serializedState)
    {
        auto lineDefinition = std::dynamic_pointer_cast<const linescan::LineScanDefinition>(definition);
        const auto parsed = parseObject(serializedState);
        if (!lineDefinition || !parsed)
        {
            return {};
        }
        camera_core::ImageSize size;
        std::optional<xjw::coordinate_system::TimeReference> acquisitionTime;
        std::optional<linescan::LineScanTrajectory> decodedTrajectory;
        linescan::LineTiming timing;
        try
        {
            if (!imageSize(*parsed, &size) || !captureTime(*parsed, &acquisitionTime) ||
                !trajectory(*parsed, &decodedTrajectory) || !decodedTrajectory || !lineTiming(*parsed, &timing))
            {
                return {};
            }
            return linescan::LineScanInstance::createUnique(id,
                                                            image,
                                                            std::move(lineDefinition),
                                                            size,
                                                            std::move(*decodedTrajectory),
                                                            std::move(timing),
                                                            acquisitionTime);
        }
        catch (const std::exception&)
        {
            return {};
        }
    }

} // namespace xjw::camera_models
