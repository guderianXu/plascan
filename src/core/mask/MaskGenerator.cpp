#include "MaskGenerator.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace xjw::mask
{
namespace
{

cv::Mat toGray8(const cv::Mat &image)
{
    if (image.empty())
    {
        return cv::Mat();
    }

    cv::Mat gray;
    if (image.channels() == 1)
    {
        gray = image;
    }
    else
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    if (gray.depth() == CV_8U)
    {
        return gray.clone();
    }

    cv::Mat normalized;
    cv::normalize(gray, normalized, 0, 255, cv::NORM_MINMAX);
    cv::Mat gray8;
    normalized.convertTo(gray8, CV_8U);
    return gray8;
}

cv::Mat cleanForeground(const cv::Mat &foreground, const MaskGenerationOptions &options)
{
    cv::Mat clean = foreground.clone();
    if (options.morphologyRadius > 0)
    {
        const int radius = options.morphologyRadius;
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                         cv::Size(radius * 2 + 1, radius * 2 + 1));
        cv::morphologyEx(clean, clean, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(clean, clean, cv::MORPH_CLOSE, kernel);
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int components = cv::connectedComponentsWithStats(clean, labels, stats, centroids, 8, CV_32S);
    if (components <= 1)
    {
        return clean;
    }

    cv::Mat filtered = cv::Mat::zeros(clean.size(), CV_8UC1);
    int largestLabel = -1;
    int largestArea = 0;
    for (int label = 1; label < components; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > largestArea)
        {
            largestArea = area;
            largestLabel = label;
        }
    }

    for (int label = 1; label < components; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const bool keep = options.keepLargestComponent
            ? (label == largestLabel)
            : (area >= std::max(1, options.minComponentArea));
        if (keep && area >= std::max(1, options.minComponentArea))
        {
            filtered.setTo(255, labels == label);
        }
    }

    return filtered;
}

cv::Mat normalizeMask(const cv::Mat &mask)
{
    if (mask.empty())
    {
        return cv::Mat();
    }

    cv::Mat gray = toGray8(mask);
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY);
    return binary;
}

} // namespace

cv::Mat generateMask(const cv::Mat &image, const MaskGenerationOptions &options)
{
    const cv::Mat gray = toGray8(image);
    if (gray.empty())
    {
        return cv::Mat();
    }

    cv::Mat foreground;
    if (options.threshold >= 0.0)
    {
        cv::threshold(gray, foreground, options.threshold, 255, cv::THRESH_BINARY);
    }
    else
    {
        cv::threshold(gray, foreground, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    if (options.method == MaskGenerationMethod::BlackBackground)
    {
        foreground = cleanForeground(foreground, options);
    }

    cv::Mat mask;
    cv::bitwise_not(foreground, mask);
    return mask;
}

cv::Mat composeMasks(const cv::Mat &existingMask, const cv::Mat &generatedMask, MaskOperation operation)
{
    const cv::Mat generated = normalizeMask(generatedMask);
    if (generated.empty())
    {
        return cv::Mat();
    }

    const cv::Mat existing = existingMask.empty()
        ? cv::Mat::zeros(generated.size(), CV_8UC1)
        : normalizeMask(existingMask);

    if (existing.size() != generated.size())
    {
        return generated.clone();
    }

    cv::Mat out;
    switch (operation)
    {
    case MaskOperation::Replace:
        return generated.clone();
    case MaskOperation::Union:
        cv::bitwise_or(existing, generated, out);
        return out;
    case MaskOperation::Intersection:
        cv::bitwise_and(existing, generated, out);
        return out;
    case MaskOperation::Difference:
        cv::bitwise_and(existing, ~generated, out);
        return out;
    }

    return generated.clone();
}

std::vector<std::vector<cv::Point>> extractMaskContours(const cv::Mat &mask, bool foregroundBoundary)
{
    const cv::Mat binaryMask = normalizeMask(mask);
    if (binaryMask.empty())
    {
        return {};
    }

    cv::Mat source;
    if (foregroundBoundary)
    {
        cv::bitwise_not(binaryMask, source);
    }
    else
    {
        source = binaryMask;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(source, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &a,
                                                   const std::vector<cv::Point> &b)
    {
        return cv::contourArea(a) > cv::contourArea(b);
    });
    return contours;
}

} // namespace xjw::mask
