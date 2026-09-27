#include "internal/state_json_common.h"

#include "placamera/linescan_camera.h"

#include <memory>
#include <string>
#include <utility>

namespace placamera::state_detail
{
    namespace
    {

        const char* pixelConventionName(LineScanPixelConvention convention) noexcept
        {
            return convention == LineScanPixelConvention::PixelCenter ? "pixel_center" : "zero_based";
        }

        const char* distortionModelName(LineScanDistortionModel model) noexcept
        {
            return model == LineScanDistortionModel::RadialNormalized ? "radial_normalized" : "lro_nac_focal_plane";
        }

        bool readPixelConvention(const Json& root, LineScanPixelConvention* output)
        {
            std::string value;
            if (!output || !readString(root, "pixel_convention", &value))
            {
                return false;
            }
            if (value == "pixel_center")
            {
                *output = LineScanPixelConvention::PixelCenter;
                return true;
            }
            if (value == "zero_based")
            {
                *output = LineScanPixelConvention::ZeroBased;
                return true;
            }
            return false;
        }

        bool readDistortionModel(const Json& optics, LineScanDistortionModel* output)
        {
            std::string value;
            if (!output || !readString(optics, "distortion_model", &value))
            {
                return false;
            }
            if (value == "radial_normalized")
            {
                *output = LineScanDistortionModel::RadialNormalized;
                return true;
            }
            if (value == "lro_nac_focal_plane")
            {
                *output = LineScanDistortionModel::LroNacFocalPlane;
                return true;
            }
            return false;
        }

        Json detectorGeometry(const std::optional<LineScanDetectorGeometry>& detector)
        {
            if (!detector)
            {
                return nullptr;
            }
            return Json{{"sample_summing", detector->detectorSampleSumming},
                        {"line_summing", detector->detectorLineSumming},
                        {"sample_origin", detector->detectorSampleOrigin},
                        {"line_origin", detector->detectorLineOrigin},
                        {"starting_sample", detector->startingDetectorSample},
                        {"starting_line", detector->startingDetectorLine},
                        {"focal_to_pixel_samples", finiteArray(detector->focalToPixelSamples)},
                        {"focal_to_pixel_lines", finiteArray(detector->focalToPixelLines)}};
        }

        bool readDetectorGeometry(const Json& value, std::optional<LineScanDetectorGeometry>* output)
        {
            if (!output)
            {
                return false;
            }
            if (value.is_null())
            {
                *output = std::nullopt;
                return true;
            }
            if (!hasObjectKeys(value,
                               {"sample_summing",
                                "line_summing",
                                "sample_origin",
                                "line_origin",
                                "starting_sample",
                                "starting_line",
                                "focal_to_pixel_samples",
                                "focal_to_pixel_lines"}))
            {
                return false;
            }
            LineScanDetectorGeometry detector;
            if (!readFinite(value, "sample_summing", &detector.detectorSampleSumming) ||
                !readFinite(value, "line_summing", &detector.detectorLineSumming) ||
                !readFinite(value, "sample_origin", &detector.detectorSampleOrigin) ||
                !readFinite(value, "line_origin", &detector.detectorLineOrigin) ||
                !readFinite(value, "starting_sample", &detector.startingDetectorSample) ||
                !readFinite(value, "starting_line", &detector.startingDetectorLine) ||
                !readFiniteArray(value.at("focal_to_pixel_samples"), &detector.focalToPixelSamples) ||
                !readFiniteArray(value.at("focal_to_pixel_lines"), &detector.focalToPixelLines))
            {
                return false;
            }
            *output = detector;
            return true;
        }

    } // namespace

    Json lineScanDefinitionParameters(const CameraDefinition& definition)
    {
        const auto* line_scan = dynamic_cast<const LineScanDefinition*>(&definition);
        if (!line_scan)
        {
            throw CameraValidationError(CameraErrorCode::UnsupportedModel,
                                        "line-scan codec received another definition type");
        }
        const LineScanOptics& optics = line_scan->optics();
        return Json{{"optics",
                     Json{{"focal_length_mm", optics.focalLengthMillimeters},
                          {"sample_pitch_mm", optics.samplePitchMillimeters},
                          {"principal_sample", optics.principalSample},
                          {"distortion_k1", optics.distortionK1},
                          {"distortion_model", distortionModelName(optics.distortionModel)},
                          {"detector_geometry", detectorGeometry(optics.detectorGeometry)},
                          {"complete_calibration", metashapeCalibration(optics.completeCalibration)}}},
                    {"pixel_convention", pixelConventionName(line_scan->pixelConvention())}};
    }

    Result<CameraModelFactory::DefinitionPointer> createLineScanDefinition(const CameraDefinitionState& state)
    {
        if (state.parameters.schemaVersion < 1 ||
            state.parameters.schemaVersion > LineScanDefinition::ParameterSchemaVersion)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                          "unsupported line-scan parameter schema");
        }
        const auto parsed = parseStateObject(state.parameters, DefinitionParametersJsonMediaType);
        if (!parsed)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(parsed.error());
        }
        const Json& root = parsed.value();
        if (!hasObjectKeys(root, {"optics", "pixel_convention"}) || !root.at("optics").is_object())
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "line-scan parameters have missing or unknown fields");
        }

        const Json& optics_json = root.at("optics");
        const bool current_schema = state.parameters.schemaVersion == LineScanDefinition::ParameterSchemaVersion;
        const bool detector_schema = state.parameters.schemaVersion >= 2;
        const bool optics_keys_valid =
            current_schema ? hasObjectKeys(optics_json,
                                           {"focal_length_mm",
                                            "sample_pitch_mm",
                                            "principal_sample",
                                            "distortion_k1",
                                            "distortion_model",
                                            "detector_geometry",
                                            "complete_calibration"})
            : detector_schema
                ? hasObjectKeys(optics_json,
                                {"focal_length_mm",
                                 "sample_pitch_mm",
                                 "principal_sample",
                                 "distortion_k1",
                                 "distortion_model",
                                 "detector_geometry"})
                : hasObjectKeys(optics_json,
                                {"focal_length_mm", "sample_pitch_mm", "principal_sample", "distortion_k1"});
        if (!optics_keys_valid)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "line-scan optics fields do not match their schema");
        }

        LineScanOptics optics;
        LineScanPixelConvention convention = LineScanPixelConvention::PixelCenter;
        if (!readFinite(optics_json, "focal_length_mm", &optics.focalLengthMillimeters) ||
            !readFinite(optics_json, "sample_pitch_mm", &optics.samplePitchMillimeters) ||
            !readFinite(optics_json, "principal_sample", &optics.principalSample) ||
            !readFinite(optics_json, "distortion_k1", &optics.distortionK1) || !readPixelConvention(root, &convention))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                          "line-scan optics values are invalid");
        }
        if (detector_schema && (!readDistortionModel(optics_json, &optics.distortionModel) ||
                                !readDetectorGeometry(optics_json.at("detector_geometry"), &optics.detectorGeometry)))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                          "line-scan detector values are invalid");
        }
        if (current_schema &&
            !readMetashapeCalibration(optics_json.at("complete_calibration"), &optics.completeCalibration))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                          "line-scan complete calibration is invalid");
        }

        CameraModelFactory::DefinitionPointer definition =
            LineScanDefinition::create(state.definitionId, state.groundFrame, optics, convention);
        return Result<CameraModelFactory::DefinitionPointer>::success(std::move(definition));
    }

} // namespace placamera::state_detail
