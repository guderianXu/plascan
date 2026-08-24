#pragma once

#include "RpcCameraModel.h"

#include <map>
#include <string>

namespace xjw
{

    using RpcMetadata = std::map<std::string, std::string>;

    /** Parse the standard GDAL RPC metadata-domain keys into an RPC00B model. */
    bool
    rpcCameraFromMetadata(const RpcMetadata& metadata, RpcCameraModel* camera, std::string* errorMessage = nullptr);

    /**
     * Load embedded RPC metadata, or an RPC/RPB sidecar discovered by GDAL, from a raster.
     */
    bool
    loadRpcCameraFromRaster(const std::string& rasterPath, RpcCameraModel* camera, std::string* errorMessage = nullptr);

} // namespace xjw
