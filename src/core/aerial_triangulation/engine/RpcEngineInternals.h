#pragma once

#include "engine/RpcEngine.h"

namespace xjw::aerial_triangulation::engine::detail
{
    std::vector<std::vector<RpcObservation>> buildRpcTracks(const TiePointGraph& graph);
    bool intersectRpcTrack(const TiePointGraph& graph,
                           const std::map<ImageId, RpcCamera>& cameras,
                           const std::vector<RpcObservation>& observations,
                           double maximumRmsPixels,
                           RpcPoint* point,
                           std::map<ImageId, CameraResidualAccumulator>* cameraResiduals);
    RpcGeodeticCoordinate assignRpcLocalEnu(std::vector<RpcPoint>* points);
} // namespace xjw::aerial_triangulation::engine::detail
