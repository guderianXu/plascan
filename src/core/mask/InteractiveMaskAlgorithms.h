#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace xjw::mask
{

    cv::Mat rectangleSelection(const cv::Size& imageSize, const cv::Rect& rectangle);
    cv::Mat magicWandSelection(const cv::Mat& image, const cv::Point& seed, int colorTolerance);
    cv::Mat
    smartBrushSelection(const cv::Mat& image, const std::vector<cv::Point>& stroke, int radius, int colorTolerance);
    std::vector<cv::Point>
    edgeSnappedPath(const cv::Mat& image, const cv::Point& start, const cv::Point& end, int searchRadius);
    void applySelectionToMask(cv::Mat* mask, const cv::Mat& selection, bool exclude);

} // namespace xjw::mask
