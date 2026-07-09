#pragma once

#include "BundleAdjust.h"
#include "BundleAdjustNativeCudaTypes.h"

#include <string>
#include <vector>

namespace xjw::detail::native_cuda
{

struct WorksetBuildResult
{
    bool ok = false;
    std::string message;
    Workset workset;
};

WorksetBuildResult buildWorkset(const std::vector<Camera> &cameras,
                                const std::vector<BATrack> &tracks,
                                const BAOptions &options);

bool hasUnsupportedConstraints(const std::vector<BATrack> &tracks,
                               const BAOptions &options,
                               std::string *message);

} // namespace xjw::detail::native_cuda
