#pragma once

#include "RpcProjection.h"

#include <string>

namespace xjw::camera_models::rpc
{

    struct RpcIntersectionOptions
    {
        double pixelTolerance = 1.0e-5;
        double positionToleranceMeters = 1.0e-3;
        int maximumIterations = 30;
    };

    struct RpcIntersectionResult
    {
        EcefCoordinate ecefMeters{{0.0, 0.0, 0.0}};
        RpcDefinition::GeodeticCoordinate geodetic{{0.0, 0.0, 0.0}};
        double reprojectionRmsPixels = 0.0;
        int iterations = 0;
    };

    class RpcIntersectionService
    {
    public:
        static bool intersect(const RpcInstance& first,
                              const ImagePoint& firstImage,
                              const RpcInstance& second,
                              const ImagePoint& secondImage,
                              RpcIntersectionResult* result,
                              std::string* error = nullptr);

        static bool intersect(const RpcInstance& first,
                              const ImagePoint& firstImage,
                              const RpcInstance& second,
                              const ImagePoint& secondImage,
                              RpcIntersectionResult* result,
                              const RpcIntersectionOptions& options,
                              std::string* error = nullptr);
    };

} // namespace xjw::camera_models::rpc
