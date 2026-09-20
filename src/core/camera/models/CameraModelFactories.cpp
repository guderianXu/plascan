#include "CameraModelFactories.h"

#include "LineScanModelJson.h"

#include "frame_pinhole/FramePinholeDefinition.h"
#include "frame_pinhole/FramePinholeInstance.h"
#include "linescan/LineScanDefinition.h"
#include "linescan/LineScanInstance.h"
#include "rpc/RpcDefinition.h"
#include "rpc/RpcInstance.h"

#include <QJsonArray>
#include <QJsonValue>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
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
                return QJsonObject{};
            }
            const QByteArray bytes(serialized.data(), static_cast<qsizetype>(serialized.size()));
            QJsonParseError error;
            const QJsonDocument document = QJsonDocument::fromJson(bytes, &error);
            if (error.error != QJsonParseError::NoError || !document.isObject())
            {
                return std::nullopt;
            }
            return document.object();
        }

        bool requiredNumber(const QJsonObject& object, const QString& key, double* output)
        {
            if (!output)
            {
                return false;
            }
            const QJsonValue value = object.value(key);
            if (!value.isDouble() || !std::isfinite(value.toDouble()))
            {
                return false;
            }
            *output = value.toDouble();
            return true;
        }

        bool coefficients(const QJsonObject& object, const QString& key, std::array<double, 20>* output)
        {
            if (!output)
            {
                return false;
            }
            const QJsonArray values = object.value(key).toArray();
            if (values.size() != static_cast<int>(output->size()))
            {
                return false;
            }
            for (int index = 0; index < values.size(); ++index)
            {
                if (!values.at(index).isDouble() || !std::isfinite(values.at(index).toDouble()))
                {
                    return false;
                }
                (*output)[static_cast<std::size_t>(index)] = values.at(index).toDouble();
            }
            return true;
        }

        bool imageSize(const QJsonObject& object, camera_core::ImageSize* output)
        {
            if (!output)
            {
                return false;
            }
            const QJsonValue value = object.value(QStringLiteral("image_size"));
            if (!value.isObject())
            {
                return false;
            }
            const QJsonObject size = value.toObject();
            const QJsonValue samplesValue = size.value(QStringLiteral("samples"));
            const QJsonValue linesValue = size.value(QStringLiteral("lines"));
            if (!samplesValue.isDouble() || !linesValue.isDouble())
            {
                return false;
            }
            const double samples = samplesValue.toDouble();
            const double lines = linesValue.toDouble();
            if (!std::isfinite(samples) || !std::isfinite(lines) || samples <= 0.0 || lines <= 0.0 ||
                samples > static_cast<double>(std::numeric_limits<int>::max()) ||
                lines > static_cast<double>(std::numeric_limits<int>::max()) || std::floor(samples) != samples ||
                std::floor(lines) != lines)
            {
                return false;
            }
            *output = {static_cast<int>(samples), static_cast<int>(lines)};
            return true;
        }

        bool pose(const QJsonObject& object, camera_core::Pose* output)
        {
            if (!output)
            {
                return false;
            }
            const QJsonValue poseValue = object.value(QStringLiteral("pose"));
            if (!poseValue.isObject())
            {
                return false;
            }
            const QJsonObject poseObject = poseValue.toObject();
            const QJsonArray centerValues = poseObject.value(QStringLiteral("center_m")).toArray();
            const QJsonArray rotationValues = poseObject.value(QStringLiteral("camera_to_world_rotation")).toArray();
            if (centerValues.size() != 3 || rotationValues.size() != 9)
            {
                return false;
            }
            std::array<double, 3> center{};
            camera_core::Rotation rotation{};
            for (int index = 0; index < 3; ++index)
            {
                if (!centerValues.at(index).isDouble() || !std::isfinite(centerValues.at(index).toDouble()))
                {
                    return false;
                }
                center[static_cast<std::size_t>(index)] = centerValues.at(index).toDouble();
            }
            for (int index = 0; index < 9; ++index)
            {
                if (!rotationValues.at(index).isDouble() || !std::isfinite(rotationValues.at(index).toDouble()))
                {
                    return false;
                }
                rotation[static_cast<std::size_t>(index)] = rotationValues.at(index).toDouble();
            }
            const QJsonValue poseFrameValue = poseObject.value(QStringLiteral("frame"));
            if (!poseFrameValue.isString() || poseFrameValue.toString().trimmed().isEmpty())
            {
                return false;
            }
            const QString poseFrame = poseFrameValue.toString();
            try
            {
                *output = camera_core::Pose::create(
                    xjw::coordinate_system::CoordinateFrameId(poseFrame.toStdString()), center, rotation);
            }
            catch (const camera_core::CameraValidationError&)
            {
                return false;
            }
            return true;
        }

        std::optional<xjw::coordinate_system::TimeScale> timeScale(const QJsonValue& value)
        {
            if (!value.isString())
            {
                return std::nullopt;
            }
            const QString scale = value.toString().trimmed().toLower();
            if (scale == QStringLiteral("utc"))
            {
                return xjw::coordinate_system::TimeScale::Utc;
            }
            if (scale == QStringLiteral("tai"))
            {
                return xjw::coordinate_system::TimeScale::Tai;
            }
            if (scale == QStringLiteral("tdb"))
            {
                return xjw::coordinate_system::TimeScale::Tdb;
            }
            if (scale == QStringLiteral("relative"))
            {
                return xjw::coordinate_system::TimeScale::Relative;
            }
            return std::nullopt;
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
            const QJsonValue value = state.value(QStringLiteral("capture_time"));
            if (!value.isObject())
            {
                return false;
            }
            const QJsonObject object = value.toObject();
            const auto scale = timeScale(object.value(QStringLiteral("time_scale")));
            double seconds = 0.0;
            if (!scale || !requiredNumber(object, QStringLiteral("seconds"), &seconds))
            {
                return false;
            }
            try
            {
                *output = xjw::coordinate_system::TimeReference::create(*scale, seconds);
            }
            catch (const camera_core::CameraValidationError&)
            {
                return false;
            }
            return true;
        }

        bool pinholeIntrinsics(const QJsonObject& object, frame_pinhole::Intrinsics* output)
        {
            if (!output)
            {
                return false;
            }
            const QJsonValue value = object.value(QStringLiteral("intrinsics"));
            if (!value.isObject())
            {
                return false;
            }
            const QJsonObject source = value.toObject();
            double focalX = 0.0;
            double focalY = 0.0;
            double principalX = 0.0;
            double principalY = 0.0;
            double pixelPitch = 0.0;
            double uAxisSign = 0.0;
            double vAxisSign = 0.0;
            if (!requiredNumber(source, QStringLiteral("fx_px"), &focalX) ||
                !requiredNumber(source, QStringLiteral("fy_px"), &focalY) ||
                !requiredNumber(source, QStringLiteral("cx_px"), &principalX) ||
                !requiredNumber(source, QStringLiteral("cy_px"), &principalY) ||
                !requiredNumber(source, QStringLiteral("pixel_pitch_mm"), &pixelPitch) ||
                !requiredNumber(source, QStringLiteral("u_axis_sign"), &uAxisSign) ||
                !requiredNumber(source, QStringLiteral("v_axis_sign"), &vAxisSign) ||
                (uAxisSign != 1.0 && uAxisSign != -1.0) || (vAxisSign != 1.0 && vAxisSign != -1.0))
            {
                return false;
            }
            output->focalX = focalX;
            output->focalY = focalY;
            output->principalX = principalX;
            output->principalY = principalY;
            output->pixelPitch = pixelPitch;
            output->uAxisSign = static_cast<int>(uAxisSign);
            output->vAxisSign = static_cast<int>(vAxisSign);
            return true;
        }

        bool pinholeDistortion(const QJsonObject& object, frame_pinhole::Distortion* output)
        {
            if (!output)
            {
                return false;
            }
            const QJsonValue value = object.value(QStringLiteral("distortion"));
            if (!value.isObject())
            {
                return false;
            }
            const QJsonObject source = value.toObject();
            return requiredNumber(source, QStringLiteral("k1"), &output->radialK1) &&
                   requiredNumber(source, QStringLiteral("k2"), &output->radialK2) &&
                   requiredNumber(source, QStringLiteral("k3"), &output->radialK3) &&
                   requiredNumber(source, QStringLiteral("p1"), &output->tangentialP1) &&
                   requiredNumber(source, QStringLiteral("p2"), &output->tangentialP2);
        }

        bool pinholePixelConvention(const QJsonObject& object, frame_pinhole::PixelConvention* output)
        {
            if (!output || !object.value(QStringLiteral("pixel_convention")).isString())
            {
                return false;
            }
            const QString value = object.value(QStringLiteral("pixel_convention")).toString();
            if (value == QStringLiteral("center"))
            {
                *output = frame_pinhole::PixelConvention::PixelCenter;
                return true;
            }
            if (value == QStringLiteral("corner"))
            {
                *output = frame_pinhole::PixelConvention::PixelCorner;
                return true;
            }
            return false;
        }

    } // namespace

    std::optional<int> builtinCameraParameterSchemaVersion(std::string_view modelType) noexcept
    {
        if (modelType == "frame_pinhole")
        {
            return frame_pinhole::FramePinholeDefinition::ParameterSchemaVersion;
        }
        if (modelType == "rpc00b")
        {
            return rpc::RpcDefinition::ParameterSchemaVersion;
        }
        if (modelType == "planetary_linescan")
        {
            return linescan::LineScanDefinition::ParameterSchemaVersion;
        }
        return std::nullopt;
    }

    camera_core::CameraModelRegistry makeBuiltinCameraModelRegistry()
    {
        camera_core::CameraModelRegistry registry;

        registry.registerFactory(
            "frame_pinhole",
            camera_core::CameraModelFactory{
                [](const camera_core::CameraDefinitionId& id,
                   const xjw::coordinate_system::CoordinateFrameId& frame,
                   int parameterSchemaVersion,
                   std::string_view serializedParameters) -> std::unique_ptr<camera_core::CameraDefinition>
                {
                    if (parameterSchemaVersion != frame_pinhole::FramePinholeDefinition::ParameterSchemaVersion)
                    {
                        return std::unique_ptr<camera_core::CameraDefinition>{};
                    }
                    const auto parsedParameters = parseObject(serializedParameters);
                    if (!parsedParameters)
                    {
                        return std::unique_ptr<camera_core::CameraDefinition>{};
                    }
                    const QJsonObject& parameters = *parsedParameters;
                    frame_pinhole::Intrinsics intrinsics;
                    frame_pinhole::Distortion distortion;
                    frame_pinhole::PixelConvention pixelConvention;
                    const QJsonValue depthAxisFlipped = parameters.value(QStringLiteral("depth_axis_flipped"));
                    if (!pinholeIntrinsics(parameters, &intrinsics) || !pinholeDistortion(parameters, &distortion) ||
                        !pinholePixelConvention(parameters, &pixelConvention) || !depthAxisFlipped.isBool())
                    {
                        return std::unique_ptr<camera_core::CameraDefinition>{};
                    }
                    return frame_pinhole::FramePinholeDefinition::createUnique(
                        id, intrinsics, distortion, pixelConvention, frame, depthAxisFlipped.toBool());
                },
                [](const camera_core::CameraInstanceId& id,
                   const camera_core::ImageId& image,
                   std::shared_ptr<const camera_core::CameraDefinition> definition,
                   std::string_view serializedState) -> std::unique_ptr<camera_core::CameraInstance>
                {
                    auto pinhole = std::dynamic_pointer_cast<const frame_pinhole::FramePinholeDefinition>(definition);
                    if (!pinhole)
                    {
                        return std::unique_ptr<camera_core::CameraInstance>{};
                    }
                    const auto parsedState = parseObject(serializedState);
                    if (!parsedState)
                    {
                        return std::unique_ptr<camera_core::CameraInstance>{};
                    }
                    const QJsonObject& state = *parsedState;
                    camera_core::ImageSize size;
                    camera_core::Pose instancePose = camera_core::Pose::create(
                        definition->worldFrame(), {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
                    std::optional<xjw::coordinate_system::TimeReference> acquisitionTime;
                    if (!imageSize(state, &size) || !pose(state, &instancePose) ||
                        !captureTime(state, &acquisitionTime))
                    {
                        return std::unique_ptr<camera_core::CameraInstance>{};
                    }
                    return frame_pinhole::FramePinholeInstance::createUnique(
                        id, image, std::move(pinhole), size, instancePose, acquisitionTime);
                }});

        registry.registerFactory(
            "rpc00b",
            camera_core::CameraModelFactory{
                [](const camera_core::CameraDefinitionId& id,
                   const xjw::coordinate_system::CoordinateFrameId& frame,
                   int parameterSchemaVersion,
                   std::string_view serializedParameters) -> std::unique_ptr<camera_core::CameraDefinition>
                {
                    if (parameterSchemaVersion != rpc::RpcDefinition::ParameterSchemaVersion)
                    {
                        return std::unique_ptr<camera_core::CameraDefinition>{};
                    }
                    const auto parsedObject = parseObject(serializedParameters);
                    if (!parsedObject)
                    {
                        return std::unique_ptr<camera_core::CameraDefinition>{};
                    }
                    const QJsonObject& object = *parsedObject;
                    if (!object.value(QStringLiteral("rpc_spec")).isString() ||
                        object.value(QStringLiteral("rpc_spec")).toString() != QStringLiteral("RPC00B"))
                    {
                        return std::unique_ptr<camera_core::CameraDefinition>{};
                    }
                    rpc::RpcDefinition::Parameters parameters;
                    const auto read = [&object](const QString& key, double* target)
                    { return requiredNumber(object, key, target); };
                    if (!read(QStringLiteral("line_offset"), &parameters.lineOffset) ||
                        !read(QStringLiteral("sample_offset"), &parameters.sampleOffset) ||
                        !read(QStringLiteral("latitude_offset"), &parameters.latitudeOffset) ||
                        !read(QStringLiteral("longitude_offset"), &parameters.longitudeOffset) ||
                        !read(QStringLiteral("height_offset"), &parameters.heightOffset) ||
                        !read(QStringLiteral("line_scale"), &parameters.lineScale) ||
                        !read(QStringLiteral("sample_scale"), &parameters.sampleScale) ||
                        !read(QStringLiteral("latitude_scale"), &parameters.latitudeScale) ||
                        !read(QStringLiteral("longitude_scale"), &parameters.longitudeScale) ||
                        !read(QStringLiteral("height_scale"), &parameters.heightScale) ||
                        !coefficients(object, QStringLiteral("line_numerator"), &parameters.lineNumerator) ||
                        !coefficients(object, QStringLiteral("line_denominator"), &parameters.lineDenominator) ||
                        !coefficients(object, QStringLiteral("sample_numerator"), &parameters.sampleNumerator) ||
                        !coefficients(object, QStringLiteral("sample_denominator"), &parameters.sampleDenominator))
                    {
                        return std::unique_ptr<camera_core::CameraDefinition>{};
                    }
                    if (object.contains(QStringLiteral("error_bias_m")))
                    {
                        double value = 0.0;
                        if (!requiredNumber(object, QStringLiteral("error_bias_m"), &value))
                        {
                            return std::unique_ptr<camera_core::CameraDefinition>{};
                        }
                        parameters.errorBiasMeters = value;
                    }
                    if (object.contains(QStringLiteral("error_random_m")))
                    {
                        double value = 0.0;
                        if (!requiredNumber(object, QStringLiteral("error_random_m"), &value))
                        {
                            return std::unique_ptr<camera_core::CameraDefinition>{};
                        }
                        parameters.errorRandomMeters = value;
                    }
                    return rpc::RpcDefinition::createUnique(id, frame, parameters);
                },
                [](const camera_core::CameraInstanceId& id,
                   const camera_core::ImageId& image,
                   std::shared_ptr<const camera_core::CameraDefinition> definition,
                   std::string_view serializedState) -> std::unique_ptr<camera_core::CameraInstance>
                {
                    auto rpcDefinition = std::dynamic_pointer_cast<const rpc::RpcDefinition>(definition);
                    if (!rpcDefinition)
                    {
                        return std::unique_ptr<camera_core::CameraInstance>{};
                    }
                    const auto parsedState = parseObject(serializedState);
                    if (!parsedState)
                    {
                        return std::unique_ptr<camera_core::CameraInstance>{};
                    }
                    const QJsonObject& state = *parsedState;
                    camera_core::ImageSize size;
                    std::optional<xjw::coordinate_system::TimeReference> acquisitionTime;
                    if (!imageSize(state, &size) || !captureTime(state, &acquisitionTime))
                    {
                        return std::unique_ptr<camera_core::CameraInstance>{};
                    }
                    rpc::ImageCorrection correction;
                    const QJsonObject correctionObject = state.value(QStringLiteral("image_correction")).toObject();
                    if (state.contains(QStringLiteral("image_correction")) && correctionObject.isEmpty())
                    {
                        return std::unique_ptr<camera_core::CameraInstance>{};
                    }
                    if (!correctionObject.isEmpty())
                    {
                        const auto readCorrection = [&correctionObject](const QString& key, double* target)
                        {
                            const QJsonValue value = correctionObject.value(key);
                            if (!value.isDouble() || !std::isfinite(value.toDouble()))
                            {
                                return false;
                            }
                            *target = value.toDouble();
                            return true;
                        };
                        if (!readCorrection(QStringLiteral("sample_offset_px"), &correction.sampleOffsetPixels) ||
                            !readCorrection(QStringLiteral("sample_sample_px"), &correction.sampleSamplePixels) ||
                            !readCorrection(QStringLiteral("sample_line_px"), &correction.sampleLinePixels) ||
                            !readCorrection(QStringLiteral("line_offset_px"), &correction.lineOffsetPixels) ||
                            !readCorrection(QStringLiteral("line_sample_px"), &correction.lineSamplePixels) ||
                            !readCorrection(QStringLiteral("line_line_px"), &correction.lineLinePixels))
                        {
                            return std::unique_ptr<camera_core::CameraInstance>{};
                        }
                    }
                    return rpc::RpcInstance::createUnique(
                        id, image, std::move(rpcDefinition), size, correction, acquisitionTime);
                }});

        registry.registerFactory(
            "planetary_linescan",
            camera_core::CameraModelFactory{createLineScanDefinitionFromJson, createLineScanInstanceFromJson});

        return registry;
    }

} // namespace xjw::camera_models
