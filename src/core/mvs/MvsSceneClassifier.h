#pragma once

#include "MvsTypes.h"

namespace xjw
{
namespace mvs
{

MvsSceneClassification classifyMvsScene(const std::vector<CameraView> &views,
                                        const SparseCloud &sparse_cloud);

/// Return the source-image candidate pool for the effective scene and depth quality.
/// configured_count is treated as a user-requested lower bound, while view_count caps the result.
int recommendedMvsSourceViewCount(MvsSceneProfile scene_profile,
                                  int downsample_factor,
                                  int configured_count,
                                  int view_count);

float recommendedMvsSourceMaximumAngleDeg(MvsSceneProfile scene_profile);

} // namespace mvs
} // namespace xjw
