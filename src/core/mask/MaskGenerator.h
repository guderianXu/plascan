#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace xjw::mask
{

enum class MaskGenerationMethod
{
    BlackBackground,
    Threshold
};

enum class MaskOperation
{
    Replace,
    Union,
    Intersection,
    Difference
};

struct MaskGenerationOptions
{
    MaskGenerationMethod method = MaskGenerationMethod::BlackBackground;
    double threshold = -1.0;
    int minComponentArea = 64;
    int morphologyRadius = 2;
    bool keepLargestComponent = true;
};

cv::Mat generateMask(const cv::Mat &image, const MaskGenerationOptions &options);
cv::Mat composeMasks(const cv::Mat &existingMask, const cv::Mat &generatedMask, MaskOperation operation);
std::vector<std::vector<cv::Point>> extractMaskContours(const cv::Mat &mask, bool foregroundBoundary = true);

} // namespace xjw::mask
