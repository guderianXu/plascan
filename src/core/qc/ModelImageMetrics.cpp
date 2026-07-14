#include "ModelImageMetrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw::qc
{
namespace
{

cv::Mat binaryMask(const cv::Mat &mask)
{
    if (mask.empty())
    {
        return {};
    }
    cv::Mat gray;
    if (mask.channels() == 1)
    {
        gray = mask;
    }
    else
    {
        cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat binary;
    cv::compare(gray, 0, binary, cv::CMP_GT);
    return binary;
}

double percentile(std::vector<float> values, double fraction)
{
    if (values.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    std::sort(values.begin(), values.end());
    const double position = std::clamp(fraction, 0.0, 1.0) *
                            static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double blend = position - static_cast<double>(lower);
    return static_cast<double>(values[lower]) * (1.0 - blend) +
           static_cast<double>(values[upper]) * blend;
}

cv::Mat maskEdge(const cv::Mat &mask)
{
    cv::Mat edge_image;
    cv::morphologyEx(mask, edge_image, cv::MORPH_GRADIENT,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    return edge_image;
}

void appendEdgeDistances(const cv::Mat &source_edge,
                         const cv::Mat &target_edge,
                         std::vector<float> *distances)
{
    cv::Mat distance_input;
    cv::bitwise_not(target_edge, distance_input);
    cv::Mat distance;
    cv::distanceTransform(distance_input, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    for (int y = 0; y < source_edge.rows; ++y)
    {
        const std::uint8_t *edge_row = source_edge.ptr<std::uint8_t>(y);
        const float *distance_row = distance.ptr<float>(y);
        for (int x = 0; x < source_edge.cols; ++x)
        {
            if (edge_row[x] != 0)
            {
                distances->push_back(distance_row[x]);
            }
        }
    }
}

cv::Mat grayscaleFloat(const cv::Mat &image)
{
    cv::Mat gray;
    if (image.channels() == 1)
    {
        gray = image;
    }
    else
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat value;
    gray.convertTo(value, CV_32FC1);
    return value;
}

} // namespace

ModelViewQuality evaluateModelViewMasks(const cv::Mat &referenceMask,
                                        const cv::Mat &renderedMask)
{
    ModelViewQuality quality;
    const cv::Mat reference = binaryMask(referenceMask);
    const cv::Mat rendered = binaryMask(renderedMask);
    if (reference.empty() || rendered.empty() || reference.size() != rendered.size())
    {
        return quality;
    }

    cv::Mat intersection;
    cv::Mat union_mask;
    cv::bitwise_and(reference, rendered, intersection);
    cv::bitwise_or(reference, rendered, union_mask);
    const double reference_pixels = static_cast<double>(cv::countNonZero(reference));
    const double rendered_pixels = static_cast<double>(cv::countNonZero(rendered));
    const double intersection_pixels = static_cast<double>(cv::countNonZero(intersection));
    const double union_pixels = static_cast<double>(cv::countNonZero(union_mask));
    quality.referenceCoverage = reference_pixels > 0.0
        ? intersection_pixels / reference_pixels : 0.0;
    quality.silhouetteIou = union_pixels > 0.0
        ? intersection_pixels / union_pixels : 0.0;
    quality.floatingPixelRate = rendered_pixels > 0.0
        ? (rendered_pixels - intersection_pixels) / rendered_pixels : 1.0;

    const cv::Mat reference_edge = maskEdge(reference);
    const cv::Mat rendered_edge = maskEdge(rendered);
    if (cv::countNonZero(reference_edge) > 0 && cv::countNonZero(rendered_edge) > 0)
    {
        std::vector<float> distances;
        distances.reserve(static_cast<std::size_t>(
            cv::countNonZero(reference_edge) + cv::countNonZero(rendered_edge)));
        appendEdgeDistances(reference_edge, rendered_edge, &distances);
        appendEdgeDistances(rendered_edge, reference_edge, &distances);
        quality.edgeP50Pixels = percentile(distances, 0.50);
        quality.edgeP90Pixels = percentile(distances, 0.90);
    }
    return quality;
}

void evaluateModelViewAppearance(const cv::Mat &sourceBgr,
                                 const cv::Mat &renderedBgr,
                                 const cv::Mat &validMask,
                                 ModelViewQuality *quality)
{
    if (!quality || sourceBgr.empty() || renderedBgr.empty() ||
        sourceBgr.size() != renderedBgr.size())
    {
        return;
    }
    const cv::Mat mask = binaryMask(validMask);
    if (mask.empty() || cv::countNonZero(mask) < 64)
    {
        return;
    }

    const cv::Mat source = grayscaleFloat(sourceBgr);
    const cv::Mat rendered = grayscaleFloat(renderedBgr);
    cv::Scalar source_mean;
    cv::Scalar source_stddev;
    cv::Scalar rendered_mean;
    cv::Scalar rendered_stddev;
    cv::meanStdDev(source, source_mean, source_stddev, mask);
    cv::meanStdDev(rendered, rendered_mean, rendered_stddev, mask);
    const double scale = rendered_stddev[0] > 1.0e-6
        ? source_stddev[0] / rendered_stddev[0] : 1.0;
    cv::Mat adjusted = (rendered - rendered_mean[0]) * scale + source_mean[0];

    cv::Mat source_squared = source.mul(source);
    cv::Mat adjusted_squared = adjusted.mul(adjusted);
    cv::Mat source_adjusted = source.mul(adjusted);
    cv::Mat source_mu;
    cv::Mat adjusted_mu;
    cv::GaussianBlur(source, source_mu, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(adjusted, adjusted_mu, cv::Size(11, 11), 1.5);
    cv::Mat source_variance;
    cv::Mat adjusted_variance;
    cv::Mat covariance;
    cv::GaussianBlur(source_squared, source_variance, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(adjusted_squared, adjusted_variance, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(source_adjusted, covariance, cv::Size(11, 11), 1.5);
    source_variance -= source_mu.mul(source_mu);
    adjusted_variance -= adjusted_mu.mul(adjusted_mu);
    covariance -= source_mu.mul(adjusted_mu);

    constexpr double c1 = 6.5025;
    constexpr double c2 = 58.5225;
    cv::Mat numerator = (2.0 * source_mu.mul(adjusted_mu) + c1)
                            .mul(2.0 * covariance + c2);
    cv::Mat denominator = (source_mu.mul(source_mu) +
                           adjusted_mu.mul(adjusted_mu) + c1)
                              .mul(source_variance + adjusted_variance + c2);
    cv::Mat ssim_map;
    cv::divide(numerator, denominator, ssim_map);
    quality->foregroundSsim = std::clamp(cv::mean(ssim_map, mask)[0], -1.0, 1.0);

    const cv::Mat difference = source - adjusted;
    const double mse = cv::mean(difference.mul(difference), mask)[0];
    quality->foregroundPsnr = mse <= 1.0e-9
        ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    quality->appearanceAvailable = true;
}

void evaluateModelViewStructure(const cv::Mat &sourceBgr,
                                const cv::Mat &renderedBgr,
                                const cv::Mat &validMask,
                                ModelViewQuality *quality)
{
    if (!quality || sourceBgr.empty() || renderedBgr.empty() ||
        sourceBgr.size() != renderedBgr.size())
    {
        return;
    }
    const cv::Mat mask = binaryMask(validMask);
    if (mask.empty() || cv::countNonZero(mask) < 64)
    {
        return;
    }

    cv::Mat source_gray;
    cv::Mat rendered_gray;
    cv::cvtColor(sourceBgr, source_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(renderedBgr, rendered_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(source_gray, source_gray, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(rendered_gray, rendered_gray, cv::Size(5, 5), 1.0);
    cv::Mat source_edge;
    cv::Mat rendered_edge;
    cv::Canny(source_gray, source_edge, 40.0, 120.0);
    cv::Canny(rendered_gray, rendered_edge, 40.0, 120.0);
    cv::bitwise_and(source_edge, mask, source_edge);
    cv::bitwise_and(rendered_edge, mask, rendered_edge);
    if (cv::countNonZero(source_edge) == 0 || cv::countNonZero(rendered_edge) == 0)
    {
        return;
    }

    std::vector<float> distances;
    distances.reserve(static_cast<std::size_t>(
        cv::countNonZero(source_edge) + cv::countNonZero(rendered_edge)));
    appendEdgeDistances(source_edge, rendered_edge, &distances);
    appendEdgeDistances(rendered_edge, source_edge, &distances);
    quality->edgeP50Pixels = percentile(distances, 0.50);
    quality->edgeP90Pixels = percentile(distances, 0.90);
}

cv::Mat buildDinoForegroundMask(const cv::Mat &sourceBgr)
{
    if (sourceBgr.empty())
    {
        return {};
    }
    cv::Mat gray;
    cv::cvtColor(sourceBgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask;
    cv::threshold(gray, mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
                     cv::Point(-1, -1), 2);

    cv::Mat labels;
    cv::Mat statistics;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, statistics,
                                                        centroids, 8, CV_32S);
    if (count <= 1)
    {
        return mask;
    }
    int largest_label = 1;
    int largest_area = statistics.at<int>(1, cv::CC_STAT_AREA);
    for (int label = 2; label < count; ++label)
    {
        const int area = statistics.at<int>(label, cv::CC_STAT_AREA);
        if (area > largest_area)
        {
            largest_area = area;
            largest_label = label;
        }
    }
    cv::compare(labels, largest_label, mask, cv::CMP_EQ);

    cv::Mat flood = mask.clone();
    cv::copyMakeBorder(flood, flood, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));
    flood = flood(cv::Rect(1, 1, mask.cols, mask.rows));
    cv::bitwise_not(flood, flood);
    cv::bitwise_or(mask, flood, mask);
    return mask;
}

} // namespace xjw::qc
