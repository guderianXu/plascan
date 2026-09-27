#include <array>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "placamera/tsai.h"

namespace placamera
{
    namespace
    {

        std::string trimAsciiWhitespace(std::string_view text)
        {
            const auto first = text.find_first_not_of(" \t\r\n\f\v");
            if (first == std::string_view::npos)
            {
                return {};
            }
            const auto last = text.find_last_not_of(" \t\r\n\f\v");
            return std::string(text.substr(first, last - first + 1));
        }

        std::string asciiLowerCopy(std::string text)
        {
            for (char& value : text)
            {
                if (value >= 'A' && value <= 'Z')
                {
                    value = static_cast<char>(value + ('a' - 'A'));
                }
            }
            return text;
        }

        std::string pathText(const std::filesystem::path& path)
        {
            const auto utf8 = path.u8string();
            return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
        }

        bool extractDoublesFromText(const std::string& text, std::vector<double>& values)
        {
            values.clear();
            static const std::regex number_pattern(R"([+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?)");
            for (auto it = std::sregex_iterator(text.begin(), text.end(), number_pattern); it != std::sregex_iterator();
                 ++it)
            {
                try
                {
                    values.push_back(std::stod(it->str()));
                }
                catch (const std::exception&)
                {
                }
            }
            return !values.empty();
        }

        struct TsaiParameters
        {
            double focalXMillimeters = 0.0;
            double focalYMillimeters = 0.0;
            double principalXMillimeters = 0.0;
            double principalYMillimeters = 0.0;
            double pixelPitchMillimeters = 1.0;
            std::array<double, 3> center{{0.0, 0.0, 0.0}};
            std::array<double, 9> rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
            double radialK1 = 0.0;
            double radialK2 = 0.0;
            double radialK3 = 0.0;
            double tangentialP1 = 0.0;
            double tangentialP2 = 0.0;
            int uAxisSign = 1;
            int vAxisSign = 1;
            bool depthAxisFlipped = false;
        };

        bool startsWithKey(const std::string& textLower, const std::string& keyLower)
        {
            if (textLower.rfind(keyLower, 0) != 0)
            {
                return false;
            }
            if (textLower.size() == keyLower.size())
            {
                return true;
            }
            const char next = textLower[keyLower.size()];
            return std::isspace(static_cast<unsigned char>(next)) || next == '=' || next == ':';
        }

        bool firstNumberAfterSeparator(const std::string& line, double* value)
        {
            if (!value)
            {
                return false;
            }
            const std::size_t separator = line.find_first_of("=:");
            if (separator == std::string::npos)
            {
                return false;
            }
            std::vector<double> values;
            if (!extractDoublesFromText(line.substr(separator + 1), values) || values.empty())
            {
                return false;
            }
            *value = values.front();
            return true;
        }

        void setError(std::string* error, const std::string& message)
        {
            if (error)
            {
                *error = message;
            }
        }

        std::string formatInterchangeNumber(double value)
        {
            if (std::abs(value) < 1e-14)
            {
                value = 0.0;
            }
            std::ostringstream output;
            output << std::setprecision(12) << value;
            return output.str();
        }

        std::optional<RotationMatrix> normalizedTsaiRotation(const std::array<double, 9>& input)
        {
            constexpr double maximum_correction = 1.0e-4;
            for (const double value : input)
            {
                if (!std::isfinite(value))
                {
                    return std::nullopt;
                }
            }

            std::array<double, 3> first{input[0], input[3], input[6]};
            std::array<double, 3> second{input[1], input[4], input[7]};
            const auto length = [](const std::array<double, 3>& vector)
            { return std::hypot(vector[0], vector[1], vector[2]); };
            const double first_length = length(first);
            if (!(first_length > 0.0) || !std::isfinite(first_length))
            {
                return std::nullopt;
            }
            for (double& value : first)
            {
                value /= first_length;
            }
            const double projection = first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
            for (std::size_t index = 0; index < second.size(); ++index)
            {
                second[index] -= projection * first[index];
            }
            const double second_length = length(second);
            if (!(second_length > 0.0) || !std::isfinite(second_length))
            {
                return std::nullopt;
            }
            for (double& value : second)
            {
                value /= second_length;
            }
            const std::array<double, 3> third{first[1] * second[2] - first[2] * second[1],
                                              first[2] * second[0] - first[0] * second[2],
                                              first[0] * second[1] - first[1] * second[0]};
            const RotationMatrix normalized{
                first[0], second[0], third[0], first[1], second[1], third[1], first[2], second[2], third[2]};
            for (std::size_t index = 0; index < input.size(); ++index)
            {
                if (std::abs(normalized[index] - input[index]) > maximum_correction)
                {
                    return std::nullopt;
                }
            }
            return normalized;
        }

        std::optional<TsaiParameters> parseTsaiParameters(std::istream& input, std::string* error)
        {

            TsaiParameters parameters;
            double focal_x_mm = 0.0;
            double focal_y_mm = 0.0;
            double principal_x_mm = 0.0;
            double principal_y_mm = 0.0;
            double pixel_pitch_mm = 1.0;
            std::array<double, 3> center{{0.0, 0.0, 0.0}};
            std::array<double, 9> rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
            double radial_k1 = 0.0;
            double radial_k2 = 0.0;
            double radial_k3 = 0.0;
            double tangential_p1 = 0.0;
            double tangential_p2 = 0.0;
            int u_axis_sign = 1;
            int v_axis_sign = 1;
            bool depth_axis_flipped = false;
            bool has_focal_x = false;
            bool has_focal_y = false;
            bool has_principal_x = false;
            bool has_principal_y = false;
            bool has_center = false;
            bool has_rotation = false;

            std::string line;
            while (std::getline(input, line))
            {
                const std::string trimmed = trimAsciiWhitespace(line);
                if (trimmed.empty())
                {
                    continue;
                }
                const std::string lower = asciiLowerCopy(trimmed);
                std::vector<double> values;
                if (startsWithKey(lower, "fu"))
                {
                    if (extractDoublesFromText(trimmed, values) && !values.empty())
                    {
                        focal_x_mm = values.front();
                        has_focal_x = true;
                    }
                }
                else if (startsWithKey(lower, "fv"))
                {
                    if (extractDoublesFromText(trimmed, values) && !values.empty())
                    {
                        focal_y_mm = values.front();
                        has_focal_y = true;
                    }
                }
                else if (startsWithKey(lower, "cu"))
                {
                    if (extractDoublesFromText(trimmed, values) && !values.empty())
                    {
                        principal_x_mm = values.front();
                        has_principal_x = true;
                    }
                }
                else if (startsWithKey(lower, "cv"))
                {
                    if (extractDoublesFromText(trimmed, values) && !values.empty())
                    {
                        principal_y_mm = values.front();
                        has_principal_y = true;
                    }
                }
                else if (startsWithKey(lower, "c"))
                {
                    if (extractDoublesFromText(trimmed, values) && values.size() >= 3)
                    {
                        center = {values[0], values[1], values[2]};
                        has_center = true;
                    }
                }
                else if (startsWithKey(lower, "r"))
                {
                    if (extractDoublesFromText(trimmed, values) && values.size() >= 9)
                    {
                        for (std::size_t index = 0; index < rotation.size(); ++index)
                        {
                            rotation[index] = values[index];
                        }
                        has_rotation = true;
                    }
                }
                else if (startsWithKey(lower, "k1"))
                {
                    firstNumberAfterSeparator(trimmed, &radial_k1);
                }
                else if (startsWithKey(lower, "k2"))
                {
                    firstNumberAfterSeparator(trimmed, &radial_k2);
                }
                else if (startsWithKey(lower, "k3"))
                {
                    firstNumberAfterSeparator(trimmed, &radial_k3);
                }
                else if (startsWithKey(lower, "p1"))
                {
                    firstNumberAfterSeparator(trimmed, &tangential_p1);
                }
                else if (startsWithKey(lower, "p2"))
                {
                    firstNumberAfterSeparator(trimmed, &tangential_p2);
                }
                else if (startsWithKey(lower, "pitch"))
                {
                    if (extractDoublesFromText(trimmed, values) && !values.empty())
                    {
                        pixel_pitch_mm = values.front();
                    }
                }
                else if (startsWithKey(lower, "u_direction"))
                {
                    if (extractDoublesFromText(trimmed, values) && !values.empty())
                    {
                        const double dominant =
                            values.size() >= 3 ? (values[0] == 0.0 ? values[1] : values[0]) : values.front();
                        u_axis_sign = dominant < 0.0 ? -1 : 1;
                    }
                }
                else if (startsWithKey(lower, "v_direction"))
                {
                    if (extractDoublesFromText(trimmed, values) && !values.empty())
                    {
                        const double dominant =
                            values.size() >= 3 ? (values[1] == 0.0 ? values[0] : values[1]) : values.front();
                        v_axis_sign = dominant < 0.0 ? -1 : 1;
                    }
                }
                else if (startsWithKey(lower, "w_direction"))
                {
                    if (extractDoublesFromText(trimmed, values) && !values.empty())
                    {
                        const double depth = values.size() >= 3 ? values[2] : values.front();
                        depth_axis_flipped = depth < 0.0;
                    }
                }
            }

            if (!(has_focal_x && has_focal_y && has_principal_x && has_principal_y && has_center && has_rotation))
            {
                setError(error, "Tsai camera is missing fu/fv/cu/cv/c/r fields: ");
                return std::nullopt;
            }
            if (!std::isfinite(pixel_pitch_mm) || pixel_pitch_mm <= 0.0)
            {
                setError(error, "Tsai camera pixel pitch must be positive: ");
                return std::nullopt;
            }
            if (!std::isfinite(focal_x_mm) || !std::isfinite(focal_y_mm) || focal_x_mm <= 0.0 || focal_y_mm <= 0.0)
            {
                setError(error, "Tsai camera focal lengths must be positive: ");
                return std::nullopt;
            }

            parameters.focalXMillimeters = focal_x_mm;
            parameters.focalYMillimeters = focal_y_mm;
            parameters.principalXMillimeters = principal_x_mm;
            parameters.principalYMillimeters = principal_y_mm;
            parameters.pixelPitchMillimeters = pixel_pitch_mm;
            parameters.center = center;
            parameters.rotation = rotation;
            parameters.radialK1 = radial_k1;
            parameters.radialK2 = radial_k2;
            parameters.radialK3 = radial_k3;
            parameters.tangentialP1 = tangential_p1;
            parameters.tangentialP2 = tangential_p2;
            parameters.uAxisSign = u_axis_sign;
            parameters.vAxisSign = v_axis_sign;
            parameters.depthAxisFlipped = depth_axis_flipped;
            return parameters;
        }

    } // namespace

    Result<TsaiFramePinhole>
    readTsaiFramePinhole(std::istream& input, CameraDefinitionId definition_id, FrameId world_frame)
    {
        std::string parse_error;
        const auto parameters = parseTsaiParameters(input, &parse_error);
        if (!parameters)
        {
            return Result<TsaiFramePinhole>::failure(
                CameraErrorCode::ParseFailure, std::move(parse_error), "Tsai stream");
        }

        try
        {
            const auto rotation = normalizedTsaiRotation(parameters->rotation);
            if (!rotation)
            {
                return Result<TsaiFramePinhole>::failure(CameraErrorCode::InvalidPose,
                                                         "Tsai camera rotation is not a proper near-orthonormal matrix",
                                                         "Tsai stream");
            }
            const double inverse_pitch = 1.0 / parameters->pixelPitchMillimeters;
            const FrameIntrinsics intrinsics{parameters->focalXMillimeters * inverse_pitch,
                                             parameters->focalYMillimeters * inverse_pitch,
                                             parameters->principalXMillimeters * inverse_pitch,
                                             parameters->principalYMillimeters * inverse_pitch,
                                             parameters->pixelPitchMillimeters,
                                             parameters->uAxisSign,
                                             parameters->vAxisSign};
            const BrownConradyDistortion distortion{parameters->radialK1,
                                                    parameters->radialK2,
                                                    parameters->radialK3,
                                                    parameters->tangentialP1,
                                                    parameters->tangentialP2};
            auto definition = FramePinholeDefinition::create(std::move(definition_id),
                                                             intrinsics,
                                                             distortion,
                                                             PixelConvention::PixelCenter,
                                                             world_frame,
                                                             parameters->depthAxisFlipped);
            auto pose = Pose::create(world_frame, parameters->center, *rotation);
            return Result<TsaiFramePinhole>::success({std::move(definition), std::move(pose)});
        }
        catch (const CameraValidationError& error)
        {
            return Result<TsaiFramePinhole>::failure(
                error.code(), "invalid Tsai camera: " + std::string(error.what()), "Tsai stream");
        }
        catch (const std::exception& exception)
        {
            return Result<TsaiFramePinhole>::failure(CameraErrorCode::InvalidModelState,
                                                     "invalid Tsai camera: " + std::string(exception.what()),
                                                     "Tsai stream");
        }
    }

    Result<TsaiFramePinhole>
    loadTsaiFramePinhole(const std::filesystem::path& path, CameraDefinitionId definitionId, FrameId worldFrame)
    {
        if (path.empty())
        {
            return Result<TsaiFramePinhole>::failure(CameraErrorCode::InvalidArgument, "Tsai camera path is empty");
        }
        std::ifstream input(path);
        if (!input)
        {
            return Result<TsaiFramePinhole>::failure(
                CameraErrorCode::IoFailure, "cannot open Tsai camera file", pathText(path));
        }
        auto parsed = readTsaiFramePinhole(input, std::move(definitionId), std::move(worldFrame));
        if (!parsed)
        {
            CameraError error = parsed.error();
            error.source = pathText(path);
            return Result<TsaiFramePinhole>::failure(std::move(error));
        }
        return parsed;
    }

    Result<void> writeTsaiFramePinhole(std::ostream& output, const TsaiFramePinhole& camera)
    {
        const auto fail = [](CameraErrorCode code, std::string message)
        { return Result<void>::failure(code, std::move(message), "Tsai stream"); };
        if (!camera.definition || camera.definition->groundFrame() != camera.pose.frame)
        {
            return fail(CameraErrorCode::FrameMismatch, "Tsai camera definition and pose must share a world frame");
        }
        try
        {
            Pose::create(camera.pose.frame, camera.pose.center, camera.pose.cameraToWorldRotation);
        }
        catch (const std::exception& exception)
        {
            return fail(CameraErrorCode::InvalidPose, "invalid Tsai camera pose: " + std::string(exception.what()));
        }

        const auto& intrinsics = camera.definition->intrinsics();
        const auto& distortion = camera.definition->distortion();
        output << std::setprecision(std::numeric_limits<double>::max_digits10);
        output << "fu = " << intrinsics.focalX * intrinsics.pixelPitch << "\n";
        output << "fv = " << intrinsics.focalY * intrinsics.pixelPitch << "\n";
        output << "cu = " << intrinsics.principalX * intrinsics.pixelPitch << "\n";
        output << "cv = " << intrinsics.principalY * intrinsics.pixelPitch << "\n";
        output << "c = " << camera.pose.center[0] << " " << camera.pose.center[1] << " " << camera.pose.center[2]
               << "\n";
        output << "r = ";
        for (std::size_t index = 0; index < camera.pose.cameraToWorldRotation.size(); ++index)
        {
            if (index != 0)
            {
                output << ' ';
            }
            output << camera.pose.cameraToWorldRotation[index];
        }
        output << "\n";
        output << "k1 = " << distortion.radialK1 << "\n";
        output << "k2 = " << distortion.radialK2 << "\n";
        output << "k3 = " << distortion.radialK3 << "\n";
        output << "p1 = " << distortion.tangentialP1 << "\n";
        output << "p2 = " << distortion.tangentialP2 << "\n";
        output << "pitch = " << intrinsics.pixelPitch << "\n";
        output << "u_direction = " << intrinsics.uAxisSign << "\n";
        output << "v_direction = " << intrinsics.vAxisSign << "\n";
        output << "w_direction = " << (camera.definition->depthAxisFlipped() ? -1 : 1) << "\n";
        if (!output.good())
        {
            return fail(CameraErrorCode::IoFailure, "cannot write Tsai output");
        }
        return Result<void>::success();
    }

    Result<void> saveTsaiFramePinhole(const TsaiFramePinhole& camera, const std::filesystem::path& path)
    {
        if (path.empty())
        {
            return Result<void>::failure(CameraErrorCode::InvalidArgument, "Tsai output path is empty");
        }
        std::ostringstream serialized;
        const auto serialized_result = writeTsaiFramePinhole(serialized, camera);
        if (!serialized_result)
        {
            return serialized_result;
        }
        std::ofstream output(path);
        if (!output)
        {
            return Result<void>::failure(CameraErrorCode::IoFailure, "cannot open Tsai output file", pathText(path));
        }
        output << serialized.str();
        if (!output.good())
        {
            return Result<void>::failure(CameraErrorCode::IoFailure, "cannot write Tsai output file", pathText(path));
        }
        return Result<void>::success();
    }

    Result<void> writeTsaiPixelCamera(std::ostream& output, const TsaiPixelCamera& camera)
    {
        output << "VERSION_3\nPINHOLE\nTSAI\n";
        output << "fu = " << formatInterchangeNumber(camera.focalX) << "\n";
        output << "fv = " << formatInterchangeNumber(camera.focalY) << "\n";
        output << "cu = " << formatInterchangeNumber(camera.principalX) << "\n";
        output << "cv = " << formatInterchangeNumber(camera.principalY) << "\n";
        output << "u_direction = 1 0 0\nv_direction = 0 1 0\nw_direction = 0 0 1\npitch = 1\n";
        for (std::size_t index = 0; index < camera.distortion.size(); ++index)
        {
            constexpr std::array<const char*, 5> keys{"k1", "k2", "k3", "p1", "p2"};
            output << keys[index] << " = " << formatInterchangeNumber(camera.distortion[index]) << "\n";
        }
        output << "C = " << formatInterchangeNumber(camera.center[0]) << " "
               << formatInterchangeNumber(camera.center[1]) << " " << formatInterchangeNumber(camera.center[2])
               << "\nR = ";
        for (std::size_t index = 0; index < camera.cameraToWorldRotation.size(); ++index)
        {
            if (index != 0)
            {
                output << " ";
            }
            output << formatInterchangeNumber(camera.cameraToWorldRotation[index]);
        }
        output << "\n";
        if (!output.good())
        {
            return Result<void>::failure(CameraErrorCode::IoFailure, "cannot write Tsai output", "Tsai stream");
        }
        return Result<void>::success();
    }

} // namespace placamera
