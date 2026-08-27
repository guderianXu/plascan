#pragma once

#include <opencv2/core.hpp>

namespace xjw::mesh::texture_v4
{

    struct TextureNaturalBlendStats
    {
        int pyramidLevels = 0;
        int correctedPixelCount = 0;
        double meanAbsoluteLinearCorrection = 0.0;
    };

    TextureNaturalBlendStats applyTextureNaturalBlend(cv::Mat* atlas,
                                                      const cv::Mat& primaryAtlas,
                                                      const cv::Mat& filledMask,
                                                      const cv::Mat& primaryMask,
                                                      const cv::Rect& region,
                                                      int requestedLevels = 5,
                                                      float lowFrequencyStrength = 1.0f);

} // namespace xjw::mesh::texture_v4
