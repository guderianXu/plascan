#pragma once

#include "placamera/rpc_camera.h"

namespace placamera::internal
{

    EvaluationResult<ImageCoordinate> projectRpc(const RpcDefinition& definition,
                                                 const RpcCorrection& correction,
                                                 const GeodeticCoordinate& ground,
                                                 bool applyCorrection);

    EvaluationResult<GeodeticCoordinate> invertRpcAtHeight(const RpcDefinition& definition,
                                                           const RpcCorrection& correction,
                                                           const ImageCoordinate& image,
                                                           double ellipsoidalHeightMeters,
                                                           const EvaluationOptions& options);

} // namespace placamera::internal
