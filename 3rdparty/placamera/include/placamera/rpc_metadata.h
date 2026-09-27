#pragma once

#include "placamera/rpc_camera.h"

#include <map>
#include <string>

namespace placamera
{

    using RpcMetadata = std::map<std::string, std::string>;

    /** Decode standard RPC-domain key/value metadata. */
    Result<RpcParameters> rpcParametersFromMetadata(const RpcMetadata& metadata);

} // namespace placamera
