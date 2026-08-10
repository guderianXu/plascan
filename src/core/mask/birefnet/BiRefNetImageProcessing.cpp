#include "BiRefNetImageProcessing.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace xjw::mask
{
namespace
{

cv::Mat toBgr8(const cv::Mat& image)
{
    if (image.empty())
    {
        return {};
    }

    cv::Mat image8;
    if (image.depth() == CV_8U)
    {
        image8 = image;
    }
    else
    {
        cv::normalize(image, image8, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    cv::Mat bgr;
    if (image8.channels() == 1)
    {
        cv::cvtColor(image8, bgr, cv::COLOR_GRAY2BGR);
    }
    else if (image8.channels() == 3)
    {
        bgr = image8.clone();
    }
    else if (image8.channels() == 4)
    {
        cv::cvtColor(image8, bgr, cv::COLOR_BGRA2BGR);
    }
    else
    {
        throw std::runtime_error("BiRefNet only supports 1, 3, or 4 channel images.");
    }
    return bgr;
}

cv::Mat firstOutputPlane(const cv::Mat& output)
{
    if (output.empty())
    {
        return {};
    }
    if (output.type() != CV_32F)
    {
        throw std::runtime_error("BiRefNet inference output must use a single-channel float32 tensor.");
    }
    const cv::Mat contiguous = output.isContinuous() ? output : output.clone();
    if (output.dims == 4 && output.size[0] == 1 && output.size[1] == 1)
    {
        return cv::Mat(output.size[2], output.size[3], CV_32F,
                       const_cast<float*>(contiguous.ptr<float>())).clone();
    }
    if (output.dims == 3 && output.size[0] == 1)
    {
        return cv::Mat(output.size[1], output.size[2], CV_32F,
                       const_cast<float*>(contiguous.ptr<float>())).clone();
    }
    if (output.dims == 2)
    {
        return output.clone();
    }
    return {};
}

float sigmoid(float value)
{
    if (value >= 0.0f)
    {
        const float exponent = std::exp(-value);
        return 1.0f / (1.0f + exponent);
    }
    const float exponent = std::exp(value);
    return exponent / (1.0f + exponent);
}

cv::Mat filterForeground(const cv::Mat& foreground,
                         int morphologyRadius,
                         int minComponentArea,
                         bool keepLargestComponent)
{
    cv::Mat clean = foreground.clone();
    if (morphologyRadius > 0)
    {
        const int radius = std::clamp(morphologyRadius, 1, 64);
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
        cv::morphologyEx(clean, clean, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(clean, clean, cv::MORPH_CLOSE, kernel);
    }

    const int minimumArea = std::max(1, minComponentArea);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int components = cv::connectedComponentsWithStats(clean, labels, stats, centroids, 8, CV_32S);
    if (components <= 1)
    {
        return clean;
    }

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

    cv::Mat filtered = cv::Mat::zeros(clean.size(), CV_8UC1);
    for (int label = 1; label < components; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const bool keep = keepLargestComponent ? label == largestLabel : area >= minimumArea;
        if (keep && area >= minimumArea)
        {
            filtered.setTo(255, labels == label);
        }
    }
    return filtered;
}

} // namespace

bool BiRefNetLetterbox::isValid() const
{
    return sourceSize.width > 0 && sourceSize.height > 0 && resizedSize.width > 0 &&
           resizedSize.height > 0 && inputSize > 0 && left >= 0 && top >= 0 &&
           left + resizedSize.width <= inputSize && top + resizedSize.height <= inputSize;
}

cv::Mat makeBiRefNetBlob(const cv::Mat& image, int inputSize, BiRefNetLetterbox* letterbox)
{
    if (inputSize <= 0 || inputSize % 32 != 0)
    {
        throw std::invalid_argument("BiRefNet input size must be a positive multiple of 32.");
    }

    const cv::Mat bgr = toBgr8(image);
    if (bgr.empty())
    {
        return {};
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    const double scale = std::min(static_cast<double>(inputSize) / rgb.cols,
                                  static_cast<double>(inputSize) / rgb.rows);
    const cv::Size resizedSize(
        std::clamp(static_cast<int>(std::lround(rgb.cols * scale)), 1, inputSize),
        std::clamp(static_cast<int>(std::lround(rgb.rows * scale)), 1, inputSize));
    cv::Mat resized;
    cv::resize(rgb, resized, resizedSize, 0.0, 0.0, cv::INTER_LINEAR);

    const int left = (inputSize - resizedSize.width) / 2;
    const int top = (inputSize - resizedSize.height) / 2;
    cv::Mat canvas(inputSize, inputSize, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(canvas(cv::Rect(left, top, resizedSize.width, resizedSize.height)));

    cv::Mat normalized;
    canvas.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    const double mean[3] = {0.485, 0.456, 0.406};
    const double standardDeviation[3] = {0.229, 0.224, 0.225};
    for (int index = 0; index < 3; ++index)
    {
        channels[index] = (channels[index] - mean[index]) / standardDeviation[index];
    }
    cv::merge(channels, normalized);

    if (letterbox)
    {
        *letterbox = BiRefNetLetterbox{image.size(), resizedSize, inputSize, left, top};
    }
    return cv::dnn::blobFromImage(normalized, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
}

cv::Mat biRefNetProbabilityFromOutput(const cv::Mat& output)
{
    cv::Mat logits = firstOutputPlane(output);
    if (logits.empty())
    {
        throw std::runtime_error("BiRefNet output must be float32 NCHW [1,1,H,W] logits.");
    }
    if (!logits.isContinuous())
    {
        logits = logits.clone();
    }
    float* values = logits.ptr<float>();
    for (std::size_t index = 0; index < logits.total(); ++index)
    {
        values[index] = sigmoid(values[index]);
    }
    return logits;
}

cv::Mat makeBiRefNetMask(const cv::Mat& probability,
                         const BiRefNetLetterbox& letterbox,
                         float foregroundThreshold,
                         int morphologyRadius,
                         int minComponentArea,
                         bool keepLargestComponent)
{
    if (probability.type() != CV_32F || probability.dims != 2)
    {
        throw std::invalid_argument("BiRefNet probability map must be a single-channel float32 image.");
    }
    if (!letterbox.isValid() ||
        probability.size() != cv::Size(letterbox.inputSize, letterbox.inputSize))
    {
        throw std::invalid_argument("BiRefNet probability map and letterbox metadata are incompatible.");
    }

    const cv::Rect contentRect(letterbox.left,
                               letterbox.top,
                               letterbox.resizedSize.width,
                               letterbox.resizedSize.height);
    cv::Mat fullResolution;
    cv::resize(probability(contentRect), fullResolution, letterbox.sourceSize, 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat foregroundFloat;
    cv::threshold(fullResolution,
                  foregroundFloat,
                  std::clamp(foregroundThreshold, 0.01f, 0.99f),
                  255.0,
                  cv::THRESH_BINARY);
    cv::Mat foreground;
    foregroundFloat.convertTo(foreground, CV_8U);
    foreground = filterForeground(
        foreground, morphologyRadius, minComponentArea, keepLargestComponent);

    cv::Mat mask;
    cv::bitwise_not(foreground, mask);
    return mask;
}

} // namespace xjw::mask
