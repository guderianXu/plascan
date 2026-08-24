#pragma once

/**
 * @file RpcAerialTriangulationRunner.h
 * @brief RPC00B 影像的固定传感器模型空三与稀疏点写出。
 */

#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"

#include "RpcCameraModel.h"

#include <QMap>

namespace xjw::aerial_triangulation
{

    enum class RpcCameraInputStatus
    {
        None,
        Complete,
        Mixed
    };

    struct RpcCameraInput
    {
        RpcCameraInputStatus status = RpcCameraInputStatus::None;
        QMap<ImageId, RpcCameraModel> cameras;
        QString errorMessage;
    };

    /**
     * @brief Runs RPC aerial triangulation without converting RPC cameras to pinhole cameras.
     *
     * Vendor RPC models remain fixed when no ground control is available. Tie-point ground
     * coordinates are optimized by nonlinear RPC forward intersection and exported in a
     * local WGS84 ENU frame with the geodetic origin recorded in the result metadata.
     */
    class RpcAerialTriangulationRunner
    {
    public:
        static RpcCameraInput inspectInput(const PreparedAerialTriangulationInput& input);

        AerialTriangulationReconstructionResult run(const PreparedAerialTriangulationInput& input,
                                                    const QMap<ImageId, RpcCameraModel>& cameras) const;
    };

} // namespace xjw::aerial_triangulation
