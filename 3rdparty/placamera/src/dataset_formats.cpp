#include "placamera/dataset_formats.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace placamera
{
    namespace
    {

        std::string trim(std::string text)
        {
            const auto first = text.find_first_not_of(" \t\r\n\f\v");
            if (first == std::string::npos)
            {
                return {};
            }
            const auto last = text.find_last_not_of(" \t\r\n\f\v");
            return text.substr(first, last - first + 1);
        }

        std::string lower(std::string text)
        {
            std::transform(text.begin(),
                           text.end(),
                           text.begin(),
                           [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
            return text;
        }

        std::optional<std::string> xmlAttribute(const std::string& xml, const std::string& name);

        FrameProjectionModel metashapeProjection(const std::string& sensor)
        {
            const std::string value =
                lower(xmlAttribute(sensor, "type").value_or(xmlAttribute(sensor, "projection").value_or("frame")));
            if (value == "frame" || value == "perspective" || value == "pinhole")
            {
                return FrameProjectionModel::Perspective;
            }
            if (value == "fisheye")
            {
                return FrameProjectionModel::Fisheye;
            }
            if (value == "equidistant" || value == "equidistant_fisheye")
            {
                return FrameProjectionModel::EquidistantFisheye;
            }
            if (value == "equisolid" || value == "equisolid_fisheye")
            {
                return FrameProjectionModel::EquisolidFisheye;
            }
            if (value == "spherical")
            {
                return FrameProjectionModel::Spherical;
            }
            if (value == "cylindrical")
            {
                return FrameProjectionModel::Cylindrical;
            }
            throw std::runtime_error("Metashape sensor has unsupported projection type: " + value);
        }

        RollingShutterMode metashapeRollingShutter(const std::string& sensor)
        {
            RollingShutterMode mode = RollingShutterMode::Disabled;
            const std::regex property_pattern("<property\\b[^>]*/?>");
            for (auto it = std::sregex_iterator(sensor.begin(), sensor.end(), property_pattern);
                 it != std::sregex_iterator();
                 ++it)
            {
                const std::string property = it->str();
                const auto name = xmlAttribute(property, "name");
                const auto value = xmlAttribute(property, "value");
                if (!name || !value)
                {
                    continue;
                }
                if (*name == "rolling_shutter")
                {
                    const std::string enabled = lower(*value);
                    if (enabled == "true" || enabled == "1" || enabled == "yes")
                    {
                        mode = RollingShutterMode::Full;
                    }
                    else if (enabled != "false" && enabled != "0" && enabled != "no")
                    {
                        throw std::runtime_error("Metashape rolling_shutter property is invalid: " + *value);
                    }
                }
                else if (*name == "rolling_shutter_flags")
                {
                    mode = std::stoi(*value) == 3 ? RollingShutterMode::Regularized : RollingShutterMode::Full;
                }
            }
            return mode;
        }

        std::vector<std::string> dataLines(std::istream& input)
        {
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(input, line))
            {
                line = trim(std::move(line));
                if (!line.empty() && line.front() != '#')
                {
                    lines.push_back(std::move(line));
                }
            }
            return lines;
        }

        std::vector<double> numericLine(const std::string& line, std::size_t count)
        {
            std::istringstream input(line);
            std::vector<double> values;
            double value = 0.0;
            while (input >> value)
            {
                values.push_back(value);
            }
            if (values.size() != count)
            {
                throw std::runtime_error("numeric line has the wrong field count: " + line);
            }
            return values;
        }

        std::array<double, 9> transpose(const std::array<double, 9>& matrix)
        {
            return {matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5], matrix[8]};
        }

        ImportedCamera middleburyLine(const std::string& line)
        {
            std::istringstream input(line);
            ImportedCamera camera;
            if (!(input >> camera.imageName))
            {
                throw std::runtime_error("Middlebury camera line has no image name: " + line);
            }
            std::vector<double> values;
            double value = 0.0;
            while (input >> value)
            {
                values.push_back(value);
            }
            if (values.size() != 21)
            {
                throw std::runtime_error("Middlebury camera line requires 21 numeric fields: " + line);
            }
            std::copy_n(values.begin(), 9, camera.calibration.intrinsicMatrix.begin());
            std::array<double, 9> world_to_camera{};
            std::copy_n(values.begin() + 9, 9, world_to_camera.begin());
            camera.cameraToWorldRotation = transpose(world_to_camera);
            for (std::size_t row = 0; row < 3; ++row)
            {
                camera.center[row] = -(camera.cameraToWorldRotation[row * 3] * values[18] +
                                       camera.cameraToWorldRotation[row * 3 + 1] * values[19] +
                                       camera.cameraToWorldRotation[row * 3 + 2] * values[20]);
            }
            return camera;
        }

        std::vector<std::string> xmlBlocks(const std::string& xml, const std::string& tag)
        {
            std::vector<std::string> blocks;
            const std::regex pattern("<" + tag + "\\b[^>]*>[\\s\\S]*?</" + tag + ">");
            for (auto it = std::sregex_iterator(xml.begin(), xml.end(), pattern); it != std::sregex_iterator(); ++it)
            {
                blocks.push_back(it->str());
            }
            return blocks;
        }

        std::optional<std::string> xmlElementText(const std::string& xml, const std::string& tag)
        {
            const std::regex pattern("<" + tag + "\\b[^>]*>([\\s\\S]*?)</" + tag + ">");
            std::smatch match;
            if (!std::regex_search(xml, match, pattern) || match.size() < 2)
            {
                return std::nullopt;
            }
            return trim(match[1].str());
        }

        std::optional<std::string> xmlAttribute(const std::string& xml, const std::string& name)
        {
            const std::regex pattern("(?:^|[\\s<])" + name + "\\s*=\\s*[\"']([^\"']*)[\"']");
            std::smatch match;
            if (!std::regex_search(xml, match, pattern) || match.size() < 2)
            {
                return std::nullopt;
            }
            return match[1].str();
        }

        double xmlDoubleOr(const std::string& xml, const std::string& tag, double fallback)
        {
            const auto value = xmlElementText(xml, tag);
            return value.has_value() && !value->empty() ? std::stod(*value) : fallback;
        }

        struct MetashapeSensor
        {
            ImportedPixelCalibration calibration;
            RollingShutterMode rollingShutterMode = RollingShutterMode::Disabled;
        };

        bool matchesMetashapeInspectionView(const ImportedPixelCalibration& imported, double tolerance) noexcept
        {
            if (!imported.metashapeCalibration)
            {
                return true;
            }
            const MetashapeCalibration& exact = *imported.metashapeCalibration;
            const auto& matrix = imported.intrinsicMatrix;
            const BrownConradyDistortion& distortion = imported.distortion;
            const auto matches = [tolerance](double left, double right)
            { return std::isfinite(left) && std::isfinite(right) && std::abs(left - right) <= tolerance; };
            return matches(matrix[0], exact.f + exact.b1) && matches(matrix[1], exact.b2) &&
                   matches(matrix[2], exact.cx) && matches(matrix[4], exact.f) && matches(matrix[5], exact.cy) &&
                   matches(distortion.radialK1, exact.k1) && matches(distortion.radialK2, exact.k2) &&
                   matches(distortion.radialK3, exact.k3) && matches(distortion.radialK4, exact.k4) &&
                   matches(distortion.tangentialP1, exact.p1) && matches(distortion.tangentialP2, exact.p2) &&
                   matches(distortion.tangentialP3, exact.p3) && matches(distortion.tangentialP4, exact.p4) &&
                   distortion.tangentialConvention == BrownTangentialConvention::Metashape;
        }

        std::unordered_map<int, MetashapeSensor> metashapeSensors(const std::string& xml)
        {
            std::unordered_map<int, MetashapeSensor> sensors;
            for (const std::string& sensor_block : xmlBlocks(xml, "sensor"))
            {
                const auto id = xmlAttribute(sensor_block, "id");
                if (!id.has_value())
                {
                    continue;
                }
                const std::regex resolution_pattern("<resolution\\b[^>]*/?>");
                std::smatch resolution;
                if (!std::regex_search(sensor_block, resolution, resolution_pattern))
                {
                    throw std::runtime_error("Metashape sensor has no resolution");
                }
                const auto width_text = xmlAttribute(resolution.str(), "width");
                const auto height_text = xmlAttribute(resolution.str(), "height");
                if (!width_text.has_value() || !height_text.has_value())
                {
                    throw std::runtime_error("Metashape resolution has no width or height");
                }
                const double width = std::stod(*width_text);
                const double height = std::stod(*height_text);

                const auto calibrations = xmlBlocks(sensor_block, "calibration");
                if (calibrations.empty())
                {
                    throw std::runtime_error("Metashape sensor has no calibration");
                }
                std::string calibration = calibrations.front();
                for (const auto& block : calibrations)
                {
                    if (xmlAttribute(block, "class") == "adjusted")
                    {
                        calibration = block;
                        break;
                    }
                }
                const double focal = xmlDoubleOr(calibration, "f", 0.0);
                const double b1 = xmlDoubleOr(calibration, "b1", 0.0);
                const double b2 = xmlDoubleOr(calibration, "b2", 0.0);
                const double fx = xmlDoubleOr(calibration, "fx", focal + b1);
                const double fy = xmlDoubleOr(calibration, "fy", focal);
                if (fx <= 0.0 || fy <= 0.0)
                {
                    throw std::runtime_error("Metashape calibration has no valid focal length");
                }
                const double cx = xmlDoubleOr(calibration, "cx", 0.0);
                const double cy = xmlDoubleOr(calibration, "cy", 0.0);
                MetashapeSensor sensor;
                sensor.calibration.projectionModel = metashapeProjection(sensor_block);
                sensor.rollingShutterMode = metashapeRollingShutter(sensor_block);
                sensor.calibration.intrinsicMatrix = {
                    fx, b2, width * 0.5 + cx, 0.0, fy, height * 0.5 + cy, 0.0, 0.0, 1.0};
                auto& distortion = sensor.calibration.distortion;
                distortion.radialK1 = xmlDoubleOr(calibration, "k1", 0.0);
                distortion.radialK2 = xmlDoubleOr(calibration, "k2", 0.0);
                distortion.radialK3 = xmlDoubleOr(calibration, "k3", 0.0);
                distortion.radialK4 = xmlDoubleOr(calibration, "k4", 0.0);
                distortion.tangentialP1 = xmlDoubleOr(calibration, "p1", 0.0);
                distortion.tangentialP2 = xmlDoubleOr(calibration, "p2", 0.0);
                distortion.tangentialP3 = xmlDoubleOr(calibration, "p3", 0.0);
                distortion.tangentialP4 = xmlDoubleOr(calibration, "p4", 0.0);
                distortion.tangentialConvention = BrownTangentialConvention::Metashape;
                MetashapeCalibration exact;
                exact.f = fy;
                exact.cx = width * 0.5 + cx;
                exact.cy = height * 0.5 + cy;
                exact.b1 = fx - fy;
                exact.b2 = b2;
                exact.k1 = distortion.radialK1;
                exact.k2 = distortion.radialK2;
                exact.k3 = distortion.radialK3;
                exact.k4 = distortion.radialK4;
                exact.p1 = distortion.tangentialP1;
                exact.p2 = distortion.tangentialP2;
                exact.p3 = distortion.tangentialP3;
                exact.p4 = distortion.tangentialP4;
                exact.principalPointDecomposition = PrincipalPointDecomposition{width * 0.5, height * 0.5, cx, cy};
                sensor.calibration.metashapeCalibration = exact;
                sensors.emplace(std::stoi(*id), sensor);
            }
            if (sensors.empty())
            {
                throw std::runtime_error("Metashape document has no usable sensor");
            }
            return sensors;
        }

    } // namespace

    bool CameraImportCompatibility::isExactlyRepresentable(double tolerance) const noexcept
    {
        if (!std::isfinite(tolerance) || tolerance < 0.0 || unsupportedReason.has_value())
        {
            return false;
        }
        return std::all_of(sourceOnlyTerms.begin(),
                           sourceOnlyTerms.end(),
                           [tolerance](const CameraImportCalibrationTerm& term)
                           { return std::isfinite(term.value) && std::abs(term.value) <= tolerance; });
    }

    Result<std::vector<ImportedCamera>> readMiddleburyPar(std::istream& input)
    {
        try
        {
            auto lines = dataLines(input);
            if (!lines.empty())
            {
                std::istringstream first(lines.front());
                int count = 0;
                std::string trailing;
                if ((first >> count) && !(first >> trailing))
                {
                    lines.erase(lines.begin());
                }
            }
            if (lines.empty())
            {
                throw std::runtime_error("Middlebury par file has no camera records");
            }
            std::vector<ImportedCamera> cameras;
            cameras.reserve(lines.size());
            for (const auto& line : lines)
            {
                cameras.push_back(middleburyLine(line));
            }
            return Result<std::vector<ImportedCamera>>::success(std::move(cameras));
        }
        catch (const std::exception& error)
        {
            return Result<std::vector<ImportedCamera>>::failure(
                CameraErrorCode::ParseFailure, error.what(), "Middlebury par");
        }
    }

    Result<ImportedCamera> readEpflCamera(std::istream& input)
    {
        try
        {
            const auto lines = dataLines(input);
            if (lines.size() < 8)
            {
                throw std::runtime_error("EPFL .camera file requires at least eight numeric lines");
            }
            ImportedCamera camera;
            for (std::size_t row = 0; row < 3; ++row)
            {
                const auto values = numericLine(lines[row], 3);
                std::copy(values.begin(), values.end(), camera.calibration.intrinsicMatrix.begin() + row * 3);
            }
            const auto radial = numericLine(lines[3], 3);
            for (std::size_t index = 0; index < radial.size(); ++index)
            {
                camera.compatibility.sourceOnlyTerms.push_back(
                    {"epfl_radial_" + std::to_string(index + 1), radial[index]});
            }
            for (std::size_t row = 0; row < 3; ++row)
            {
                const auto values = numericLine(lines[row + 4], 3);
                std::copy(values.begin(), values.end(), camera.cameraToWorldRotation.begin() + row * 3);
            }
            const auto center = numericLine(lines[7], 3);
            std::copy(center.begin(), center.end(), camera.center.begin());
            return Result<ImportedCamera>::success(std::move(camera));
        }
        catch (const std::exception& error)
        {
            return Result<ImportedCamera>::failure(CameraErrorCode::ParseFailure, error.what(), "EPFL .camera");
        }
    }

    Result<std::vector<ImportedCamera>> parseMetashapeDocument(std::string_view xml)
    {
        try
        {
            const std::string document(xml);
            const auto sensors = metashapeSensors(document);
            std::vector<ImportedCamera> cameras;
            for (const std::string& block : xmlBlocks(document, "camera"))
            {
                const auto label = xmlAttribute(block, "label");
                const auto sensor_id = xmlAttribute(block, "sensor_id");
                const auto transform = xmlElementText(block, "transform");
                if (!label.has_value() || !sensor_id.has_value() || !transform.has_value())
                {
                    continue;
                }
                const auto sensor = sensors.find(std::stoi(*sensor_id));
                if (sensor == sensors.end())
                {
                    throw std::runtime_error("Metashape camera references unknown sensor_id: " + *sensor_id);
                }
                const auto values = numericLine(*transform, 16);
                ImportedCamera camera;
                camera.imageName = *label;
                camera.calibration = sensor->second.calibration;
                camera.acquisition.rollingShutterMode = sensor->second.rollingShutterMode;
                camera.cameraToWorldRotation = {
                    values[0], values[1], values[2], values[4], values[5], values[6], values[8], values[9], values[10]};
                camera.center = {values[3], values[7], values[11]};
                cameras.push_back(std::move(camera));
            }
            if (cameras.empty())
            {
                throw std::runtime_error("Metashape document has no camera records with transforms");
            }
            return Result<std::vector<ImportedCamera>>::success(std::move(cameras));
        }
        catch (const std::exception& error)
        {
            return Result<std::vector<ImportedCamera>>::failure(
                CameraErrorCode::ParseFailure, error.what(), "Metashape XML");
        }
    }

    Result<CentralCameraGeometry> makeCentralCameraGeometry(const ImportedCamera& camera,
                                                            CameraDefinitionId definitionId,
                                                            FrameId worldFrame,
                                                            double tolerance)
    {
        if (!std::isfinite(tolerance) || tolerance < 0.0)
        {
            return Result<FramePinholeGeometry>::failure(CameraErrorCode::InvalidArgument,
                                                         "dataset camera tolerance must be finite and non-negative");
        }
        if (!camera.compatibility.isExactlyRepresentable(tolerance))
        {
            const std::string reason = camera.compatibility.unsupportedReason.value_or(
                "dataset camera contains source calibration terms that the central-camera model cannot express");
            return Result<FramePinholeGeometry>::failure(CameraErrorCode::UnsupportedModel, reason);
        }
        if (!std::all_of(camera.calibration.intrinsicMatrix.begin(),
                         camera.calibration.intrinsicMatrix.end(),
                         [](double value) { return std::isfinite(value); }))
        {
            return Result<FramePinholeGeometry>::failure(CameraErrorCode::InvalidIntrinsics,
                                                         "dataset camera intrinsics must be finite");
        }
        if (!matchesMetashapeInspectionView(camera.calibration, tolerance))
        {
            return Result<FramePinholeGeometry>::failure(
                CameraErrorCode::InvalidModelState,
                "dataset camera's lossless Metashape calibration disagrees with its normalized inspection view");
        }

        const auto& intrinsics = camera.calibration.intrinsicMatrix;
        if (std::abs(intrinsics[3]) > tolerance || std::abs(intrinsics[6]) > tolerance ||
            std::abs(intrinsics[7]) > tolerance || std::abs(intrinsics[8] - 1.0) > tolerance)
        {
            return Result<FramePinholeGeometry>::failure(
                CameraErrorCode::UnsupportedModel, "dataset camera has a nonstandard homogeneous calibration row");
        }

        try
        {
            std::shared_ptr<const FramePinholeDefinition> definition;
            if (camera.calibration.metashapeCalibration)
            {
                definition = FramePinholeDefinition::create(std::move(definitionId),
                                                            *camera.calibration.metashapeCalibration,
                                                            PixelConvention::PixelCenter,
                                                            worldFrame,
                                                            false,
                                                            1.0,
                                                            1,
                                                            1,
                                                            camera.calibration.projectionModel);
            }
            else
            {
                definition = FramePinholeDefinition::create(
                    std::move(definitionId),
                    {intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5], 1.0, 1, 1, intrinsics[1]},
                    camera.calibration.distortion,
                    PixelConvention::PixelCenter,
                    worldFrame,
                    false,
                    camera.calibration.projectionModel);
            }
            auto pose = Pose::create(std::move(worldFrame), camera.center, camera.cameraToWorldRotation);
            return Result<FramePinholeGeometry>::success({std::move(definition), std::move(pose)});
        }
        catch (const CameraValidationError& error)
        {
            return Result<FramePinholeGeometry>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<FramePinholeGeometry>::failure(CameraErrorCode::InvalidModelState, error.what());
        }
    }

    Result<CentralCameraGeometry> makeDatasetCentralCamera(const DatasetFrameCamera& camera,
                                                           CameraDefinitionId definitionId,
                                                           FrameId worldFrame,
                                                           double tolerance)
    {
        return makeCentralCameraGeometry(camera, std::move(definitionId), std::move(worldFrame), tolerance);
    }

    Result<FramePinholeGeometry> makeDatasetFramePinhole(const ImportedCamera& camera,
                                                         CameraDefinitionId definitionId,
                                                         FrameId worldFrame,
                                                         double tolerance)
    {
        return makeCentralCameraGeometry(camera, std::move(definitionId), std::move(worldFrame), tolerance);
    }

} // namespace placamera
