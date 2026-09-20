#pragma once

#include "camera/models/rpc/RpcInstance.h"

#include <map>
#include <memory>
#include <string>

namespace xjw::camera_models::rpc
{

    using RpcMetadata = std::map<std::string, std::string>;

    struct RpcRasterData
    {
        RpcDefinition::Parameters parameters;
        camera_core::ImageSize imageSize;
    };

    /** Parse standard GDAL RPC-domain keys into validated RPC00B parameters. */
    bool rpcParametersFromMetadata(const RpcMetadata& metadata,
                                   RpcDefinition::Parameters* parameters,
                                   std::string* error = nullptr);

    /** Read RPC00B parameters and raster dimensions from an image or its GDAL sidecar. */
    bool readRpcRasterData(const std::string& rasterPath, RpcRasterData* data, std::string* error = nullptr);

    /** Import one raster as a fully identified immutable RPC instance. */
    std::shared_ptr<const RpcInstance> importRpcRasterInstance(const std::string& rasterPath,
                                                               camera_core::CameraDefinitionId definitionId,
                                                               camera_core::CameraInstanceId instanceId,
                                                               camera_core::ImageId imageId,
                                                               xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                               std::string* error = nullptr);

} // namespace xjw::camera_models::rpc
