#include <placamera/placamera.h>

#include <cmath>
#include <iostream>
#include <optional>
#include <utility>

int main()
{
    placamera::FrameCalibration calibration;
    calibration.f = 1000.0;
    calibration.cx = 500.0;
    calibration.cy = 400.0;
    calibration.k1 = 0.001;
    calibration.principalPointDecomposition = placamera::PrincipalPointDecomposition{500.0, 400.0, 0.0, 0.0};

    const auto definition =
        placamera::CentralCameraDefinition::create(placamera::CameraDefinitionId("example-definition"),
                                                   calibration,
                                                   placamera::PixelConvention::PixelCenter,
                                                   placamera::FrameId("world"),
                                                   false,
                                                   1.0,
                                                   1,
                                                   1,
                                                   placamera::FrameProjectionModel::EquidistantFisheye);
    placamera::CentralCameraGeometry geometry{definition,
                                              placamera::Pose::create(placamera::FrameId("world"),
                                                                      {0.0, 0.0, 0.0},
                                                                      {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}})};
    placamera::CameraAcquisitionState acquisition;
    acquisition.role = placamera::CameraRole::Keyframe;
    acquisition.captureGroupId = placamera::CaptureGroupId("example-capture");
    acquisition.rollingShutterMode = placamera::RollingShutterMode::Full;
    acquisition.rollingShutter.translation = {0.001, 0.0, 0.0};
    const auto model = placamera::bindCentralCamera(std::move(geometry),
                                                    {placamera::CameraInstanceId("example-model"),
                                                     placamera::ImageId("example-image"),
                                                     placamera::ImageSize{1000, 800},
                                                     std::nullopt,
                                                     acquisition});
    if (!model)
    {
        std::cerr << "bind failed (code " << static_cast<int>(model.errorCode()) << "): " << model.message() << '\n';
        return 1;
    }

    const auto result =
        model.value()->groundToImage(placamera::GroundCoordinate{placamera::FrameId("world"), {1.0, 2.0, 10.0}});
    if (!result)
    {
        std::cerr << "projection failed (code " << static_cast<int>(result.errorCode()) << "): " << result.message()
                  << '\n';
        return 1;
    }

    std::cout << "PlaCamera " << placamera::kVersionString << " (" << model.value()->modelType()
              << "): sample=" << result.value().image.sample << ", line=" << result.value().image.line << '\n';
    return std::isfinite(result.value().image.sample) && std::isfinite(result.value().image.line) ? 0 : 1;
}
