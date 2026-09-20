#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "camera/models/rpc/RpcInstance.h"
#include "camera/models/rpc/RpcProjection.h"
#include "engine/TiePointGraph.h"

namespace xjw::aerial_triangulation::engine
{
    using RpcCamera = std::shared_ptr<const camera_models::rpc::RpcInstance>;

    struct RpcObservation
    {
        ImageId imageId = kInvalidImageId;
        FeatureIdx featureIdx = kInvalidFeatureIdx;
        bool operator<(const RpcObservation& other) const
        {
            return imageId < other.imageId || (imageId == other.imageId && featureIdx < other.featureIdx);
        }
    };
    struct RpcPoint
    {
        camera_models::rpc::EcefCoordinate ecef{};
        camera_models::rpc::RpcDefinition::GeodeticCoordinate geodetic{};
        std::array<float, 3> localEnu{};
        std::array<std::uint8_t, 3> color{{128, 128, 128}};
        std::vector<RpcObservation> observations;
        double rmsPixels = 0.0;
        double maximumResidualPixels = 0.0;
    };
    struct CameraResidualAccumulator
    {
        int observationCount = 0;
        double squaredErrorSum = 0.0;
        double maximumResidual = 0.0;
    };
    struct RpcInput
    {
        std::shared_ptr<const TiePointGraph> graph;
        std::map<ImageId, RpcCamera> cameras;
        double maximumRmsPixels = 2.0;
        std::shared_ptr<std::atomic<bool>> cancelFlag;
        std::function<void(const std::string& stage, int percent)> progressFn;
    };
    struct RpcResult
    {
        bool success = false;
        std::string error;
        std::size_t inputTrackCount = 0;
        std::vector<RpcPoint> points;
        std::map<ImageId, CameraResidualAccumulator> cameraResiduals;
        camera_models::rpc::RpcDefinition::GeodeticCoordinate origin{};
    };
    RpcResult runRpc(const RpcInput& input);
    camera_models::rpc::ImagePoint rpcImageCoordinate(const TiePointGraph& graph, const RpcObservation& observation);
} // namespace xjw::aerial_triangulation::engine
