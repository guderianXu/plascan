#include "internal/state_json_common.h"

#include "placamera/frame_camera.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace placamera::state_detail
{
    namespace
    {

        const char* pixelConventionName(PixelConvention convention) noexcept
        {
            return convention == PixelConvention::PixelCenter ? "pixel_center" : "pixel_corner";
        }

        bool readPixelConvention(const Json& object, PixelConvention* output)
        {
            std::string value;
            if (!output || !readString(object, "pixel_convention", &value))
            {
                return false;
            }
            if (value == "pixel_center")
            {
                *output = PixelConvention::PixelCenter;
                return true;
            }
            if (value == "pixel_corner")
            {
                *output = PixelConvention::PixelCorner;
                return true;
            }
            return false;
        }

        const char* tangentialConventionName(BrownTangentialConvention convention) noexcept
        {
            return convention == BrownTangentialConvention::Metashape ? "metashape" : "opencv";
        }

        bool readTangentialConvention(const Json& object, BrownTangentialConvention* output)
        {
            std::string value;
            if (!output || !readString(object, "tangential_convention", &value))
            {
                return false;
            }
            if (value == "opencv")
            {
                *output = BrownTangentialConvention::OpenCv;
                return true;
            }
            if (value == "metashape")
            {
                *output = BrownTangentialConvention::Metashape;
                return true;
            }
            return false;
        }

        bool readProjectionModel(const Json& object, FrameProjectionModel* output)
        {
            std::string value;
            if (!output || !readString(object, "projection_model", &value))
            {
                return false;
            }
            if (value == "frame_pinhole")
            {
                *output = FrameProjectionModel::Perspective;
            }
            else if (value == "frame_fisheye")
            {
                *output = FrameProjectionModel::Fisheye;
            }
            else if (value == "frame_equidistant_fisheye")
            {
                *output = FrameProjectionModel::EquidistantFisheye;
            }
            else if (value == "frame_equisolid_fisheye")
            {
                *output = FrameProjectionModel::EquisolidFisheye;
            }
            else if (value == "frame_spherical")
            {
                *output = FrameProjectionModel::Spherical;
            }
            else if (value == "frame_cylindrical")
            {
                *output = FrameProjectionModel::Cylindrical;
            }
            else
            {
                return false;
            }
            return true;
        }

        const char* cameraRoleName(CameraRole role) noexcept
        {
            return role == CameraRole::Keyframe ? "keyframe" : "regular";
        }

        bool readCameraRole(const Json& object, CameraRole* output)
        {
            std::string value;
            if (!output || !readString(object, "role", &value))
            {
                return false;
            }
            if (value == "regular")
            {
                *output = CameraRole::Regular;
                return true;
            }
            if (value == "keyframe")
            {
                *output = CameraRole::Keyframe;
                return true;
            }
            return false;
        }

        const char* rollingShutterModeName(RollingShutterMode mode) noexcept
        {
            switch (mode)
            {
            case RollingShutterMode::Disabled:
                return "disabled";
            case RollingShutterMode::Regularized:
                return "regularized";
            case RollingShutterMode::Full:
                return "full";
            }
            return "invalid";
        }

        bool readRollingShutterMode(const Json& object, RollingShutterMode* output)
        {
            std::string value;
            if (!output || !readString(object, "rolling_shutter_mode", &value))
            {
                return false;
            }
            if (value == "disabled")
            {
                *output = RollingShutterMode::Disabled;
                return true;
            }
            if (value == "regularized")
            {
                *output = RollingShutterMode::Regularized;
                return true;
            }
            if (value == "full")
            {
                *output = RollingShutterMode::Full;
                return true;
            }
            return false;
        }

        template <typename Id> Json optionalId(const std::optional<Id>& value)
        {
            return value ? Json(value->value()) : Json(nullptr);
        }

        template <typename Id> bool readOptionalId(const Json& object, std::string_view key, std::optional<Id>* output)
        {
            if (!output || !object.contains(std::string(key)))
            {
                return false;
            }
            const Json& value = object.at(std::string(key));
            if (value.is_null())
            {
                *output = std::nullopt;
                return true;
            }
            if (!value.is_string())
            {
                return false;
            }
            const std::string encoded = value.get<std::string>();
            if (encoded.empty())
            {
                return false;
            }
            *output = Id(encoded);
            return true;
        }

        bool readSize(const Json& object, std::string_view key, std::size_t* output)
        {
            if (!output || !object.contains(std::string(key)))
            {
                return false;
            }
            const Json& value = object.at(std::string(key));
            std::uint64_t encoded = 0;
            if (value.is_number_unsigned())
            {
                encoded = value.get<std::uint64_t>();
            }
            else if (value.is_number_integer())
            {
                const std::int64_t signed_value = value.get<std::int64_t>();
                if (signed_value < 0)
                {
                    return false;
                }
                encoded = static_cast<std::uint64_t>(signed_value);
            }
            else
            {
                return false;
            }
            if (encoded > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                return false;
            }
            *output = static_cast<std::size_t>(encoded);
            return true;
        }

        Json sensorMount(const SensorMountState& mount)
        {
            return Json{{"master_sensor_id", optionalId(mount.masterSensorId)},
                        {"translation", finiteArray(mount.translation)},
                        {"rotation", finiteArray(mount.rotation)},
                        {"fixed_translation", mount.fixedTranslation},
                        {"fixed_rotation", mount.fixedRotation}};
        }

        bool readSensorMount(const Json& root, SensorMountState* output)
        {
            if (!output ||
                !hasObjectKeys(root,
                               {"master_sensor_id", "translation", "rotation", "fixed_translation", "fixed_rotation"}))
            {
                return false;
            }
            return readOptionalId(root, "master_sensor_id", &output->masterSensorId) &&
                   readFiniteArray(root.at("translation"), &output->translation) &&
                   readFiniteArray(root.at("rotation"), &output->rotation) &&
                   readBool(root, "fixed_translation", &output->fixedTranslation) &&
                   readBool(root, "fixed_rotation", &output->fixedRotation);
        }

        Json acquisition(const CameraAcquisitionState& value)
        {
            return Json{{"role", cameraRoleName(value.role)},
                        {"capture_group_id", optionalId(value.captureGroupId)},
                        {"master_camera_id", optionalId(value.masterCameraId)},
                        {"layer_index", value.layerIndex},
                        {"rolling_shutter_mode", rollingShutterModeName(value.rollingShutterMode)},
                        {"rolling_shutter",
                         Json{{"translation", finiteArray(value.rollingShutter.translation)},
                              {"rotation_vector", finiteArray(value.rollingShutter.rotationVector)}}},
                        {"rolling_shutter_initialized", value.rollingShutterInitialized}};
        }

        bool readAcquisition(const Json& root, CameraAcquisitionState* output)
        {
            if (!output ||
                !hasObjectKeys(root,
                               {"role",
                                "capture_group_id",
                                "master_camera_id",
                                "layer_index",
                                "rolling_shutter_mode",
                                "rolling_shutter",
                                "rolling_shutter_initialized"}) ||
                !hasObjectKeys(root.at("rolling_shutter"), {"translation", "rotation_vector"}))
            {
                return false;
            }
            const Json& rolling = root.at("rolling_shutter");
            return readCameraRole(root, &output->role) &&
                   readOptionalId(root, "capture_group_id", &output->captureGroupId) &&
                   readOptionalId(root, "master_camera_id", &output->masterCameraId) &&
                   readSize(root, "layer_index", &output->layerIndex) &&
                   readRollingShutterMode(root, &output->rollingShutterMode) &&
                   readFiniteArray(rolling.at("translation"), &output->rollingShutter.translation) &&
                   readFiniteArray(rolling.at("rotation_vector"), &output->rollingShutter.rotationVector) &&
                   readBool(root, "rolling_shutter_initialized", &output->rollingShutterInitialized);
        }

    } // namespace

    Json frameDefinitionParameters(const CameraDefinition& definition)
    {
        const auto* frame = dynamic_cast<const FramePinholeDefinition*>(&definition);
        if (!frame)
        {
            throw CameraValidationError(CameraErrorCode::UnsupportedModel,
                                        "central-camera codec received another definition type");
        }
        const FrameIntrinsics& intrinsics = frame->intrinsics();
        const BrownConradyDistortion& distortion = frame->distortion();
        return Json{
            {"intrinsics",
             Json{{"focal_x", intrinsics.focalX},
                  {"focal_y", intrinsics.focalY},
                  {"principal_x", intrinsics.principalX},
                  {"principal_y", intrinsics.principalY},
                  {"pixel_pitch", intrinsics.pixelPitch},
                  {"u_axis_sign", intrinsics.uAxisSign},
                  {"v_axis_sign", intrinsics.vAxisSign},
                  {"skew", intrinsics.skew}}},
            {"distortion",
             Json{{"radial_k1", distortion.radialK1},
                  {"radial_k2", distortion.radialK2},
                  {"radial_k3", distortion.radialK3},
                  {"radial_k4", distortion.radialK4},
                  {"tangential_p1", distortion.tangentialP1},
                  {"tangential_p2", distortion.tangentialP2},
                  {"tangential_p3", distortion.tangentialP3},
                  {"tangential_p4", distortion.tangentialP4},
                  {"tangential_convention", tangentialConventionName(distortion.tangentialConvention)}}},
            {"pixel_convention", pixelConventionName(frame->pixelConvention())},
            {"depth_axis_flipped", frame->depthAxisFlipped()},
            {"projection_model", frameProjectionModelName(frame->projectionModel())},
            {"sensor_mount", sensorMount(frame->sensorMount())},
            {"principal_point_decomposition", principalPointDecomposition(frame->principalPointDecomposition())}};
    }

    Json frameInstanceState(const RasterModel& model)
    {
        const auto* frame = dynamic_cast<const FramePinholeModel*>(&model);
        if (!frame)
        {
            throw CameraValidationError(CameraErrorCode::UnsupportedModel,
                                        "central-camera codec received another instance type");
        }
        return Json{{"pose",
                     Json{{"center", finiteArray(frame->pose().center)},
                          {"camera_to_world_rotation", finiteArray(frame->pose().cameraToWorldRotation)}}},
                    {"acquisition", acquisition(frame->acquisition())}};
    }

    Result<CameraModelFactory::DefinitionPointer> createFrameDefinition(const CameraDefinitionState& state)
    {
        const int schema = state.parameters.schemaVersion;
        if (schema < 1 || schema > FramePinholeDefinition::ParameterSchemaVersion)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "unsupported central-camera parameter schema");
        }
        if (schema < 3 && state.parameters.modelType != "frame_pinhole")
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "legacy frame schema can only describe a perspective camera");
        }
        const auto parsed = parseStateObject(state.parameters, DefinitionParametersJsonMediaType);
        if (!parsed)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(parsed.error());
        }
        const Json& root = parsed.value();
        const bool current_schema = schema == FramePinholeDefinition::ParameterSchemaVersion;
        const bool projection_schema = schema >= 3;
        const bool valid_root =
            current_schema ? hasObjectKeys(root,
                                           {"intrinsics",
                                            "distortion",
                                            "pixel_convention",
                                            "depth_axis_flipped",
                                            "projection_model",
                                            "sensor_mount",
                                            "principal_point_decomposition"})
            : projection_schema
                ? hasObjectKeys(root,
                                {"intrinsics",
                                 "distortion",
                                 "pixel_convention",
                                 "depth_axis_flipped",
                                 "projection_model",
                                 "sensor_mount"})
                : hasObjectKeys(root, {"intrinsics", "distortion", "pixel_convention", "depth_axis_flipped"});
        if (!valid_root || !root.at("intrinsics").is_object() || !root.at("distortion").is_object() ||
            (projection_schema && !root.at("sensor_mount").is_object()))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "central-camera parameters have missing or unknown fields");
        }

        const Json& intrinsics_json = root.at("intrinsics");
        const Json& distortion_json = root.at("distortion");
        const bool legacy_brown = schema == 1;
        const bool valid_intrinsics =
            legacy_brown
                ? hasObjectKeys(
                      intrinsics_json,
                      {"focal_x", "focal_y", "principal_x", "principal_y", "pixel_pitch", "u_axis_sign", "v_axis_sign"})
                : hasObjectKeys(intrinsics_json,
                                {"focal_x",
                                 "focal_y",
                                 "principal_x",
                                 "principal_y",
                                 "pixel_pitch",
                                 "u_axis_sign",
                                 "v_axis_sign",
                                 "skew"});
        const bool valid_distortion =
            legacy_brown ? hasObjectKeys(distortion_json,
                                         {"radial_k1", "radial_k2", "radial_k3", "tangential_p1", "tangential_p2"})
                         : hasObjectKeys(distortion_json,
                                         {"radial_k1",
                                          "radial_k2",
                                          "radial_k3",
                                          "radial_k4",
                                          "tangential_p1",
                                          "tangential_p2",
                                          "tangential_p3",
                                          "tangential_p4",
                                          "tangential_convention"});
        if (!valid_intrinsics || !valid_distortion)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "central-camera nested parameters are malformed");
        }

        FrameIntrinsics intrinsics;
        BrownConradyDistortion distortion;
        PixelConvention convention = PixelConvention::PixelCenter;
        bool depth_axis_flipped = false;
        if (!readFinite(intrinsics_json, "focal_x", &intrinsics.focalX) ||
            !readFinite(intrinsics_json, "focal_y", &intrinsics.focalY) ||
            !readFinite(intrinsics_json, "principal_x", &intrinsics.principalX) ||
            !readFinite(intrinsics_json, "principal_y", &intrinsics.principalY) ||
            !readFinite(intrinsics_json, "pixel_pitch", &intrinsics.pixelPitch) ||
            !readInt(intrinsics_json, "u_axis_sign", &intrinsics.uAxisSign) ||
            !readInt(intrinsics_json, "v_axis_sign", &intrinsics.vAxisSign) ||
            !readFinite(distortion_json, "radial_k1", &distortion.radialK1) ||
            !readFinite(distortion_json, "radial_k2", &distortion.radialK2) ||
            !readFinite(distortion_json, "radial_k3", &distortion.radialK3) ||
            !readFinite(distortion_json, "tangential_p1", &distortion.tangentialP1) ||
            !readFinite(distortion_json, "tangential_p2", &distortion.tangentialP2) ||
            !readPixelConvention(root, &convention) || !readBool(root, "depth_axis_flipped", &depth_axis_flipped))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "central-camera parameter values are invalid");
        }

        if (!legacy_brown && (!readFinite(intrinsics_json, "skew", &intrinsics.skew) ||
                              !readFinite(distortion_json, "radial_k4", &distortion.radialK4) ||
                              !readFinite(distortion_json, "tangential_p3", &distortion.tangentialP3) ||
                              !readFinite(distortion_json, "tangential_p4", &distortion.tangentialP4) ||
                              !readTangentialConvention(distortion_json, &distortion.tangentialConvention)))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "extended central-camera parameter values are invalid");
        }

        FrameProjectionModel projection_model = FrameProjectionModel::Perspective;
        SensorMountState sensor_mount;
        if (projection_schema && (!readProjectionModel(root, &projection_model) ||
                                  frameProjectionModelName(projection_model) != state.parameters.modelType ||
                                  !readSensorMount(root.at("sensor_mount"), &sensor_mount)))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "central-camera projection or sensor mount is invalid");
        }

        std::optional<PrincipalPointDecomposition> principal;
        if (current_schema && !readPrincipalPointDecomposition(root.at("principal_point_decomposition"), &principal))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "central-camera principal-point decomposition is invalid");
        }

        auto frame_definition = FramePinholeDefinition::create(state.definitionId,
                                                               intrinsics,
                                                               distortion,
                                                               convention,
                                                               state.groundFrame,
                                                               depth_axis_flipped,
                                                               projection_model,
                                                               sensor_mount);
        if (principal)
        {
            FrameCalibration calibration = frame_definition->calibration();
            calibration.principalPointDecomposition = principal;
            frame_definition = FramePinholeDefinition::create(state.definitionId,
                                                              calibration,
                                                              convention,
                                                              state.groundFrame,
                                                              depth_axis_flipped,
                                                              intrinsics.pixelPitch,
                                                              intrinsics.uAxisSign,
                                                              intrinsics.vAxisSign,
                                                              projection_model,
                                                              sensor_mount);
        }
        CameraModelFactory::DefinitionPointer definition = std::move(frame_definition);
        return Result<CameraModelFactory::DefinitionPointer>::success(std::move(definition));
    }

    Result<ModelRegistry::ModelPointer> createFrameInstance(const CameraInstanceState& state,
                                                            CameraModelFactory::DefinitionPointer definition)
    {
        if (state.state.schemaVersion != 1 && state.state.schemaVersion != FramePinholeModel::InstanceSchemaVersion)
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "unsupported central-camera instance schema");
        }
        const auto frame_definition = std::dynamic_pointer_cast<const FramePinholeDefinition>(definition);
        if (!frame_definition)
        {
            return Result<ModelRegistry::ModelPointer>::failure(
                CameraErrorCode::InvalidModelState, "central-camera instance requires a central-camera definition");
        }
        const auto parsed = parseStateObject(state.state, InstanceStateJsonMediaType);
        if (!parsed)
        {
            return Result<ModelRegistry::ModelPointer>::failure(parsed.error());
        }
        const Json& root = parsed.value();
        const bool current_schema = state.state.schemaVersion == FramePinholeModel::InstanceSchemaVersion;
        const bool valid_root =
            current_schema ? hasObjectKeys(root, {"pose", "acquisition"}) : hasObjectKeys(root, {"pose"});
        if (!valid_root || !hasObjectKeys(root.at("pose"), {"center", "camera_to_world_rotation"}) ||
            (current_schema && !root.at("acquisition").is_object()))
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "central-camera instance state is malformed");
        }
        Vector3 center{};
        RotationMatrix rotation{};
        CameraAcquisitionState camera_acquisition;
        if (!readFiniteArray(root.at("pose").at("center"), &center) ||
            !readFiniteArray(root.at("pose").at("camera_to_world_rotation"), &rotation) ||
            (current_schema && !readAcquisition(root.at("acquisition"), &camera_acquisition)))
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "central-camera instance values are invalid");
        }

        auto model = FramePinholeModel::create(state.instanceId,
                                               state.imageId,
                                               frame_definition,
                                               state.imageSize,
                                               Pose::create(frame_definition->groundFrame(), center, rotation),
                                               state.captureTime,
                                               camera_acquisition);
        return Result<ModelRegistry::ModelPointer>::success(
            std::make_shared<const FramePinholeModel>(std::move(model)));
    }

} // namespace placamera::state_detail
