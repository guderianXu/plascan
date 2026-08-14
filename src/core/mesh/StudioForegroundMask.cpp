#include "StudioForegroundMask.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace xjw::mesh
{

bool StudioForegroundMask::isUsable() const
{
    return !mask.empty() && borderLuminance <= 55.0f &&
           coverage >= 0.03f && coverage <= 0.80f &&
           borderCoverage <= 0.30f;
}

bool StudioForegroundMask::isUsableForColorSampling() const
{
    return !mask.empty() && borderLuminance <= 55.0f &&
           coverage >= 0.03f && coverage <= 0.95f &&
           borderCoverage <= 0.45f;
}

StudioForegroundMask buildStudioForegroundMask(const cv::Mat &color_image)
{
    StudioForegroundMask result;
    if (color_image.empty())
    {
        return result;
    }

    cv::Mat gray;
    if (color_image.channels() == 3)
    {
        cv::cvtColor(color_image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (color_image.channels() == 1)
    {
        gray = color_image;
    }
    else
    {
        return result;
    }
    if (gray.depth() != CV_8U)
    {
        return result;
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);
    cv::Mat otsu_mask;
    const double otsu_threshold = cv::threshold(
        blurred, otsu_mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const double conservative_threshold = std::clamp(
        otsu_threshold, 24.0, 0.19 * 255.0);
    cv::Mat mask;
    cv::threshold(
        blurred, mask, conservative_threshold, 255.0, cv::THRESH_BINARY);

    const int border_width = std::max(2, std::min(mask.cols, mask.rows) / 80);
    cv::Mat border = cv::Mat::zeros(mask.size(), CV_8UC1);
    border.rowRange(0, border_width).setTo(255);
    border.rowRange(mask.rows - border_width, mask.rows).setTo(255);
    border.colRange(0, border_width).setTo(255);
    border.colRange(mask.cols - border_width, mask.cols).setTo(255);
    const int border_pixels = cv::countNonZero(border);
    const int white_border = cv::countNonZero(mask & border);
    result.borderLuminance = static_cast<float>(cv::mean(gray, border)[0]);
    if (white_border > border_pixels / 2)
    {
        cv::bitwise_not(mask, mask);
    }

    cv::Mat labels;
    cv::Mat statistics;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        mask, labels, statistics, centroids, 8, CV_32S);
    if (component_count <= 1)
    {
        return result;
    }
    int largest_label = 1;
    int largest_area = statistics.at<int>(1, cv::CC_STAT_AREA);
    for (int label = 2; label < component_count; ++label)
    {
        const int area = statistics.at<int>(label, cv::CC_STAT_AREA);
        if (area > largest_area)
        {
            largest_label = label;
            largest_area = area;
        }
    }
    cv::compare(labels, largest_label, mask, cv::CMP_EQ);

    const double image_scale = std::min(mask.cols / 640.0, mask.rows / 480.0);
    const int dilate_radius = std::max(
        2, static_cast<int>(std::lround(10.0 * image_scale)));
    const int erode_radius = std::max(
        1, static_cast<int>(std::lround(7.0 * image_scale)));
    const cv::Mat dilate_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(dilate_radius * 2 + 1, dilate_radius * 2 + 1));
    const cv::Mat erode_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(erode_radius * 2 + 1, erode_radius * 2 + 1));
    cv::dilate(mask, mask, dilate_kernel);
    cv::erode(mask, mask, erode_kernel);

    cv::Mat background;
    cv::bitwise_not(mask, background);
    cv::Mat background_labels;
    cv::Mat background_statistics;
    cv::Mat background_centroids;
    const int background_component_count = cv::connectedComponentsWithStats(
        background,
        background_labels,
        background_statistics,
        background_centroids,
        8,
        CV_32S);
    const int maximum_speckle_hole_area = std::max(
        32, mask.rows * mask.cols / 240);
    for (int label = 1; label < background_component_count; ++label)
    {
        const int left = background_statistics.at<int>(label, cv::CC_STAT_LEFT);
        const int top = background_statistics.at<int>(label, cv::CC_STAT_TOP);
        const int width = background_statistics.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = background_statistics.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = background_statistics.at<int>(label, cv::CC_STAT_AREA);
        const bool touches_border = left == 0 || top == 0 ||
            left + width >= mask.cols || top + height >= mask.rows;
        if (!touches_border && area <= maximum_speckle_hole_area)
        {
            mask.setTo(255, background_labels == label);
        }
    }

    result.coverage = static_cast<float>(cv::countNonZero(mask)) /
        std::max(1, mask.rows * mask.cols);
    result.borderCoverage = static_cast<float>(cv::countNonZero(mask & border)) /
        std::max(1, border_pixels);
    result.mask = std::move(mask);
    return result;
}

} // namespace xjw::mesh
