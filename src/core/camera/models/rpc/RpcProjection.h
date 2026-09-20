#pragma once

#include "RpcInstance.h"

#include <array>

namespace xjw::camera_models::rpc
{

    using EcefCoordinate = std::array<double, 3>;

    struct ImagePoint
    {
        double sample = 0.0;
        double line = 0.0;
    };

    struct RpcRay
    {
        EcefCoordinate origin{{0.0, 0.0, 0.0}};
        EcefCoordinate direction{{0.0, 0.0, 1.0}};
    };

    class RpcProjection
    {
    public:
        struct InverseOptions
        {
            double pixelTolerance = 1.0e-7;
            int maximumIterations = 30;
        };

        static bool
        groundToImage(const RpcInstance& instance, const RpcDefinition::GeodeticCoordinate& ground, ImagePoint* image);

        static bool groundToImageUncorrected(const RpcInstance& instance,
                                             const RpcDefinition::GeodeticCoordinate& ground,
                                             ImagePoint* image);

        static bool groundToImageEcef(const RpcInstance& instance,
                                      const EcefCoordinate& groundMeters,
                                      ImagePoint* image);

        static bool imageToGroundAtHeight(const RpcInstance& instance,
                                          const ImagePoint& image,
                                          double ellipsoidalHeightMeters,
                                          RpcDefinition::GeodeticCoordinate* ground);

        static bool imageToGroundAtHeight(const RpcInstance& instance,
                                          const ImagePoint& image,
                                          double ellipsoidalHeightMeters,
                                          RpcDefinition::GeodeticCoordinate* ground,
                                          InverseOptions options);

        static bool geodeticToEcef(const RpcDefinition::GeodeticCoordinate& geodetic, EcefCoordinate* ecef);
        static bool ecefToGeodetic(const EcefCoordinate& ecef, RpcDefinition::GeodeticCoordinate* geodetic);

        static bool ray(const RpcInstance& instance, const ImagePoint& image, RpcRay* ray);

    private:
        static bool evaluate(const RpcInstance& instance,
                             double normalizedLongitude,
                             double normalizedLatitude,
                             double normalizedHeight,
                             ImagePoint* image,
                             bool applyCorrection);
    };

} // namespace xjw::camera_models::rpc
