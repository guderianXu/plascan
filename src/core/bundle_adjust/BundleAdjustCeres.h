#pragma once

#include "BundleAdjust.h"

namespace xjw::detail
{

bool isCeresBackendCompiled();
bool isCeresCudaBackendCompiled();

BAResult optimizePointsWithCeres(const std::vector<Camera> &cameras,
                                 const std::vector<BATrack> &tracks,
                                 const BAOptions &options,
                                 bool requestGpu);

} // namespace xjw::detail
