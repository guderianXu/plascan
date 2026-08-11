#pragma once

#include "TextureMappingV4Internal.h"

#include <opencv2/core.hpp>

#include <array>
#include <vector>

namespace xjw::mesh::texture_v4
{

struct WeightedColor
{
    cv::Vec3f color{};
    float weight = 0.0f;
};

enum class TextureSampleStatus
{
    Sampled,
    MissingDepthEvidence,
    Rejected
};

TextureSampleStatus sampleTextureView(const PreparedView &view,
                                      const std::array<double, 3> &world,
                                      const FaceCandidate &candidate,
                                      const TextureMappingConfig &config,
                                      double medianEdgeLength,
                                      int padding,
                                      WeightedColor *sample);

cv::Vec3b blendTextureSamples(std::vector<WeightedColor> samples,
                              const TextureMappingConfig &config,
                              TextureMappingResult *result);

float atlasCoordinateToNormalizedUv(double coordinate, int atlasSize);

} // namespace xjw::mesh::texture_v4
