#pragma once

#include <filesystem>
#include "placamera/rpc_camera.h"
#include "placamera/rpc_metadata.h"

namespace placamera
{

    struct RpcRasterData
    {
        RpcParameters parameters;
        ImageSize imageSize;
    };

    /** Read RPC00B parameters and image dimensions from a raster or sidecar. */
    Result<RpcRasterData> readRpcRasterData(const std::filesystem::path& rasterPath);

    /** Import one raster as a fully identified immutable RPC instance. */
    Result<CameraModelPtr<RpcModel>> importRpcRasterModel(const std::filesystem::path& rasterPath,
                                                          CameraDefinitionId definitionId,
                                                          CameraInstanceId instanceId,
                                                          ImageId imageId,
                                                          FrameId worldFrame);

} // namespace placamera
