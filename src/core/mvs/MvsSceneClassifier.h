#pragma once

#include "MvsTypes.h"

namespace xjw
{
namespace mvs
{

/// Classify only geometrically verified camera rings as OrbitalObject. Missing,
/// degenerate, and free-form captures fail closed to Custom.
MvsSceneClassification classifyMvsScene(const std::vector<CameraView> &views,
                                        const SparseCloud &sparse_cloud);

/// Automatic filtering is intentionally mild only for a proven orbital ring.
/// Aerial and general/custom captures use the conservative moderate profile.
DepthFilterMode recommendedMvsDepthFilterMode(MvsSceneProfile scene_profile);

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

/// Apply an optional experiment cap without ever widening the scene-derived
/// maximum. A non-positive or non-finite cap keeps the scene value unchanged.
float constrainMvsSourceMaximumAngleDeg(float scene_maximum_degrees,
                                        float configured_cap_degrees);

} // namespace mvs
} // namespace xjw
