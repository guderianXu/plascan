#include "placamera/colmap.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace placamera
{
    namespace
    {
        int expectedParameterCount(std::string_view model)
        {
            if (model == "SIMPLE_PINHOLE")
            {
                return 3;
            }
            if (model == "PINHOLE")
            {
                return 4;
            }
            if (model == "SIMPLE_RADIAL" || model == "SIMPLE_RADIAL_FISHEYE")
            {
                return 4;
            }
            if (model == "RADIAL" || model == "RADIAL_FISHEYE")
            {
                return 5;
            }
            if (model == "OPENCV" || model == "OPENCV_FISHEYE")
            {
                return 8;
            }
            if (model == "FULL_OPENCV" || model == "THIN_PRISM_FISHEYE")
            {
                return 12;
            }
            return -1;
        }

        std::size_t distortionStart(const ColmapCamera& camera)
        {
            if (camera.model == "SIMPLE_PINHOLE" || camera.model == "PINHOLE")
            {
                return camera.parameters.size();
            }
            if (camera.model == "SIMPLE_RADIAL" || camera.model == "RADIAL" ||
                camera.model == "SIMPLE_RADIAL_FISHEYE" || camera.model == "RADIAL_FISHEYE")
            {
                return 3;
            }
            return 4;
        }

        bool hasSharedFocal(std::string_view model)
        {
            return model == "SIMPLE_PINHOLE" || model == "SIMPLE_RADIAL" || model == "RADIAL" ||
                   model == "SIMPLE_RADIAL_FISHEYE" || model == "RADIAL_FISHEYE";
        }

        RotationMatrix transpose(const RotationMatrix& matrix)
        {
            return {matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5], matrix[8]};
        }

        RotationMatrix rotationFromColmapQuaternion(double qw, double qx, double qy, double qz)
        {
            const double norm = std::hypot(std::hypot(qw, qx), std::hypot(qy, qz));
            if (!(norm > 0.0) || !std::isfinite(norm))
            {
                throw std::invalid_argument("COLMAP images.txt contains an invalid quaternion");
            }
            qw /= norm;
            qx /= norm;
            qy /= norm;
            qz /= norm;
            return {1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy)};
        }
    } // namespace

    bool isSupportedColmapCamera(const ColmapCamera& camera) noexcept
    {
        return camera.width > 0 && camera.height > 0 &&
               expectedParameterCount(camera.model) == static_cast<int>(camera.parameters.size()) &&
               std::all_of(camera.parameters.begin(),
                           camera.parameters.end(),
                           [](double value) { return std::isfinite(value); });
    }

    Result<ColmapCamera> parseColmapCameraLine(std::string_view line)
    {
        std::istringstream input{std::string(line)};
        ColmapCamera camera;
        if (!(input >> camera.cameraId >> camera.model >> camera.width >> camera.height))
        {
            return Result<ColmapCamera>::failure(
                CameraErrorCode::ParseFailure, "malformed COLMAP cameras.txt row", "COLMAP cameras.txt");
        }
        double parameter = 0.0;
        while (input >> parameter)
        {
            camera.parameters.push_back(parameter);
        }
        if (!input.eof() || !isSupportedColmapCamera(camera))
        {
            return Result<ColmapCamera>::failure(CameraErrorCode::UnsupportedModel,
                                                 "unsupported or malformed COLMAP camera model: " + camera.model,
                                                 "COLMAP cameras.txt");
        }
        return Result<ColmapCamera>::success(std::move(camera));
    }

    Result<ColmapImage> parseColmapImageLine(std::string_view line, FrameId worldFrame)
    {
        std::istringstream input{std::string(line)};
        int image_id = 0;
        int camera_id = 0;
        double qw = 0.0, qx = 0.0, qy = 0.0, qz = 0.0;
        Vector3 translation{};
        if (!(input >> image_id >> qw >> qx >> qy >> qz >> translation[0] >> translation[1] >> translation[2] >>
              camera_id))
        {
            return Result<ColmapImage>::failure(
                CameraErrorCode::ParseFailure, "malformed COLMAP images.txt header", "COLMAP images.txt");
        }
        std::string image_name;
        std::getline(input >> std::ws, image_name);
        while (!image_name.empty() && std::isspace(static_cast<unsigned char>(image_name.back())) != 0)
        {
            image_name.pop_back();
        }
        if (image_name.empty())
        {
            return Result<ColmapImage>::failure(CameraErrorCode::ParseFailure,
                                                "COLMAP images.txt header is missing an image name",
                                                "COLMAP images.txt");
        }

        try
        {
            const auto camera_to_world = transpose(rotationFromColmapQuaternion(qw, qx, qy, qz));
            Vector3 center{};
            for (int row = 0; row < 3; ++row)
            {
                center[static_cast<std::size_t>(row)] =
                    -(camera_to_world[static_cast<std::size_t>(3 * row)] * translation[0] +
                      camera_to_world[static_cast<std::size_t>(3 * row + 1)] * translation[1] +
                      camera_to_world[static_cast<std::size_t>(3 * row + 2)] * translation[2]);
            }
            return Result<ColmapImage>::success({image_id,
                                                 camera_id,
                                                 std::move(image_name),
                                                 Pose::create(std::move(worldFrame), center, camera_to_world)});
        }
        catch (const CameraValidationError& error)
        {
            return Result<ColmapImage>::failure(error.code(), error.what(), "COLMAP images.txt");
        }
        catch (const std::exception& error)
        {
            return Result<ColmapImage>::failure(CameraErrorCode::ParseFailure, error.what(), "COLMAP images.txt");
        }
    }

    Result<bool> colmapHasDistortion(const ColmapCamera& camera, double tolerance)
    {
        if (!isSupportedColmapCamera(camera))
        {
            return Result<bool>::failure(CameraErrorCode::UnsupportedModel,
                                         "unsupported or malformed COLMAP camera model: " + camera.model);
        }
        if (!std::isfinite(tolerance) || tolerance < 0.0)
        {
            return Result<bool>::failure(CameraErrorCode::InvalidArgument,
                                         "COLMAP distortion tolerance must be finite and non-negative");
        }
        return Result<bool>::success(
            std::any_of(camera.parameters.begin() + static_cast<std::ptrdiff_t>(distortionStart(camera)),
                        camera.parameters.end(),
                        [tolerance](double value) { return std::abs(value) > tolerance; }));
    }

    Result<BrownConradyDistortion> colmapBrownConradyDistortion(const ColmapCamera& camera, double tolerance)
    {
        if (!isSupportedColmapCamera(camera))
        {
            return Result<BrownConradyDistortion>::failure(
                CameraErrorCode::UnsupportedModel, "unsupported or malformed COLMAP camera model: " + camera.model);
        }
        if (!std::isfinite(tolerance) || tolerance < 0.0)
        {
            return Result<BrownConradyDistortion>::failure(
                CameraErrorCode::InvalidArgument, "COLMAP conversion tolerance must be finite and non-negative");
        }

        const auto& params = camera.parameters;
        BrownConradyDistortion distortion;
        if (camera.model == "SIMPLE_PINHOLE" || camera.model == "PINHOLE")
        {
            return Result<BrownConradyDistortion>::success(distortion);
        }
        if (camera.model == "SIMPLE_RADIAL")
        {
            distortion.radialK1 = params[3];
            return Result<BrownConradyDistortion>::success(distortion);
        }
        if (camera.model == "RADIAL")
        {
            distortion.radialK1 = params[3];
            distortion.radialK2 = params[4];
            return Result<BrownConradyDistortion>::success(distortion);
        }
        if (camera.model == "OPENCV")
        {
            distortion.radialK1 = params[4];
            distortion.radialK2 = params[5];
            distortion.tangentialP1 = params[6];
            distortion.tangentialP2 = params[7];
            return Result<BrownConradyDistortion>::success(distortion);
        }
        if (camera.model == "FULL_OPENCV")
        {
            if (std::abs(params[9]) > tolerance || std::abs(params[10]) > tolerance || std::abs(params[11]) > tolerance)
            {
                return Result<BrownConradyDistortion>::failure(
                    CameraErrorCode::UnsupportedModel,
                    "COLMAP FULL_OPENCV rational denominator cannot be represented exactly by Brown-Conrady");
            }
            distortion.radialK1 = params[4];
            distortion.radialK2 = params[5];
            distortion.radialK3 = params[8];
            distortion.tangentialP1 = params[6];
            distortion.tangentialP2 = params[7];
            return Result<BrownConradyDistortion>::success(distortion);
        }
        return Result<BrownConradyDistortion>::failure(CameraErrorCode::UnsupportedModel,
                                                       "COLMAP model " + camera.model +
                                                           " cannot be represented exactly by Brown-Conrady");
    }

    Result<FrameIntrinsics> colmapRasterIntrinsics(const ColmapCamera& camera)
    {
        if (!isSupportedColmapCamera(camera))
        {
            return Result<FrameIntrinsics>::failure(CameraErrorCode::UnsupportedModel,
                                                    "unsupported or malformed COLMAP camera model: " + camera.model);
        }
        const auto& params = camera.parameters;
        if (hasSharedFocal(camera.model))
        {
            return Result<FrameIntrinsics>::success({params[0], params[0], params[1] - 0.5, params[2] - 0.5});
        }
        return Result<FrameIntrinsics>::success({params[0], params[1], params[2] - 0.5, params[3] - 0.5});
    }

    Result<ColmapProjector> ColmapProjector::create(ColmapCamera camera)
    {
        auto intrinsics = colmapRasterIntrinsics(camera);
        if (!intrinsics)
        {
            return Result<ColmapProjector>::failure(intrinsics.error());
        }
        return Result<ColmapProjector>::success(ColmapProjector(std::move(camera), intrinsics.takeValue()));
    }

    ColmapProjector::ColmapProjector(ColmapCamera camera, FrameIntrinsics intrinsics)
        : _camera(std::move(camera)), _intrinsics(std::move(intrinsics))
    {
    }

    const FrameIntrinsics& ColmapProjector::rasterIntrinsics() const noexcept
    {
        return _intrinsics;
    }

    std::array<double, 2> ColmapProjector::project(double x, double y) const noexcept
    {
        const auto& params = _camera.parameters;
        double u = x;
        double v = y;
        const double r2_pinhole = x * x + y * y;
        if (_camera.model == "SIMPLE_RADIAL")
        {
            const double radial = 1.0 + params[3] * r2_pinhole;
            u = radial * x;
            v = radial * y;
        }
        else if (_camera.model == "RADIAL")
        {
            const double radial = 1.0 + params[3] * r2_pinhole + params[4] * r2_pinhole * r2_pinhole;
            u = radial * x;
            v = radial * y;
        }
        else if (_camera.model == "OPENCV" || _camera.model == "FULL_OPENCV")
        {
            const double r4 = r2_pinhole * r2_pinhole;
            const double r6 = r4 * r2_pinhole;
            double radial = 1.0 + params[4] * r2_pinhole + params[5] * r4;
            if (_camera.model == "FULL_OPENCV")
            {
                radial = (radial + params[8] * r6) / (1.0 + params[9] * r2_pinhole + params[10] * r4 + params[11] * r6);
            }
            u = radial * x + 2.0 * params[6] * x * y + params[7] * (r2_pinhole + 2.0 * x * x);
            v = radial * y + params[6] * (r2_pinhole + 2.0 * y * y) + 2.0 * params[7] * x * y;
        }
        else if (_camera.model == "SIMPLE_RADIAL_FISHEYE" || _camera.model == "RADIAL_FISHEYE" ||
                 _camera.model == "OPENCV_FISHEYE" || _camera.model == "THIN_PRISM_FISHEYE")
        {
            const double radius = std::sqrt(r2_pinhole);
            const double theta = std::atan(radius);
            const double scale = radius > 1.0e-12 ? theta / radius : 1.0;
            const double theta_x = scale * x;
            const double theta_y = scale * y;
            const double distortion_r2 = theta_x * theta_x + theta_y * theta_y;
            const double r4 = distortion_r2 * distortion_r2;
            const double r6 = r4 * distortion_r2;
            const double r8 = r4 * r4;
            if (_camera.model == "SIMPLE_RADIAL_FISHEYE" || _camera.model == "RADIAL_FISHEYE")
            {
                const double k2 = _camera.model == "RADIAL_FISHEYE" ? params[4] : 0.0;
                const double radial = 1.0 + params[3] * distortion_r2 + k2 * r4;
                u = radial * theta_x;
                v = radial * theta_y;
            }
            else if (_camera.model == "OPENCV_FISHEYE")
            {
                const double radial =
                    1.0 + params[4] * distortion_r2 + params[5] * r4 + params[6] * r6 + params[7] * r8;
                u = radial * theta_x;
                v = radial * theta_y;
            }
            else
            {
                const double radial =
                    1.0 + params[4] * distortion_r2 + params[5] * r4 + params[8] * r6 + params[9] * r8;
                const double xy = theta_x * theta_y;
                u = radial * theta_x + 2.0 * params[6] * xy + params[7] * (distortion_r2 + 2.0 * theta_x * theta_x) +
                    params[10] * distortion_r2;
                v = radial * theta_y + params[6] * (distortion_r2 + 2.0 * theta_y * theta_y) + 2.0 * params[7] * xy +
                    params[11] * distortion_r2;
            }
        }
        return {_intrinsics.focalX * u + _intrinsics.principalX, _intrinsics.focalY * v + _intrinsics.principalY};
    }

    Result<std::array<double, 2>> projectColmapNormalized(const ColmapCamera& camera, double x, double y)
    {
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            return Result<std::array<double, 2>>::failure(CameraErrorCode::InvalidArgument,
                                                          "COLMAP normalized coordinates must be finite");
        }
        auto projector = ColmapProjector::create(camera);
        if (!projector)
        {
            return Result<std::array<double, 2>>::failure(projector.error());
        }
        return Result<std::array<double, 2>>::success(projector.value().project(x, y));
    }

} // namespace placamera
