#pragma once

#include "RpcCameraModel.h"

#include <array>
#include <string>

namespace xjw
{

    struct RpcStereoIntersectionOptions
    {
        double pixelTolerance = 1.0e-5;
        double positionToleranceMeters = 1.0e-3;
        int maximumIterations = 30;
    };

    struct RpcStereoIntersectionResult
    {
        std::array<double, 3> ecefMeters{{0.0, 0.0, 0.0}};
        RpcCameraModel::GeodeticCoordinate geodetic{{0.0, 0.0, 0.0}};
        double reprojectionRmsPixels = 0.0;
        int iterations = 0;
    };

    bool intersectRpcObservations(const RpcCameraModel& firstCamera,
                                  const CameraImageCoordinate& firstObservation,
                                  const RpcCameraModel& secondCamera,
                                  const CameraImageCoordinate& secondObservation,
                                  RpcStereoIntersectionResult* result,
                                  std::string* errorMessage = nullptr);

    bool intersectRpcObservations(const RpcCameraModel& firstCamera,
                                  const CameraImageCoordinate& firstObservation,
                                  const RpcCameraModel& secondCamera,
                                  const CameraImageCoordinate& secondObservation,
                                  RpcStereoIntersectionResult* result,
                                  const RpcStereoIntersectionOptions& options,
                                  std::string* errorMessage = nullptr);

} // namespace xjw
