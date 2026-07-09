#pragma once

#include "BundleAdjust.h"

namespace xjw::detail
{

bool isNativeCudaBackendCompiled();
bool isNativeCudaRuntimeAvailable(int deviceId, std::string *message);

BAResult optimizePointsWithNativeCuda(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks,
                                      const BAOptions &options);

} // namespace xjw::detail
