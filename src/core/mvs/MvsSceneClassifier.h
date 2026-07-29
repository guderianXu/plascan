#pragma once

#include "MvsTypes.h"

namespace xjw
{
namespace mvs
{

MvsSceneClassification classifyMvsScene(const std::vector<CameraView> &views,
                                        const SparseCloud &sparse_cloud);

/// Return the source-image candidate pool for the effective scene and depth quality.
/// Aerial configured_count is treated as a user-requested lower bound. Orbital
/// rings use a scene-derived cap so generic quality presets cannot force
/// distant, weakly verified cameras into sparse object sequences.
int recommendedMvsSourceViewCount(MvsSceneProfile scene_profile,
                                  int downsample_factor,
                                  int configured_count,
                                  int view_count);

float recommendedMvsSourceMaximumAngleDeg(MvsSceneProfile scene_profile,
                                          int requested_source_count = 0);
float adaptiveMvsSourceMaximumAngleDeg(
    MvsSceneProfile scene_profile,
    int requested_source_count,
    const std::vector<float> &candidate_angles_degrees);

} // namespace mvs
} // namespace xjw
