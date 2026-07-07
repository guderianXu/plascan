#include "U2NetMaskGenerator.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace xjw::mask
{
namespace
{

cv::Mat toBgr8(const cv::Mat &image)
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
        throw std::runtime_error("U2Net only supports 1, 3, or 4 channel images.");
    }
    return bgr;
}

cv::Mat makeU2NetBlob(const cv::Mat &image, int inputSize)
{
    const int size = std::clamp(inputSize, 128, 2048);
    cv::Mat bgr = toBgr8(image);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(size, size), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    const double mean[3] = {0.485, 0.456, 0.406};
    const double stddev[3] = {0.229, 0.224, 0.225};
    for (int i = 0; i < 3; ++i)
    {
        channels[i] = (channels[i] - mean[i]) / stddev[i];
    }
    cv::merge(channels, normalized);

    return cv::dnn::blobFromImage(normalized, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
}

cv::Mat firstOutputPlane(const cv::Mat &output)
{
    if (output.empty())
    {
        return {};
    }

    if (output.dims == 4)
    {
        const int height = output.size[2];
        const int width = output.size[3];
        return cv::Mat(height, width, CV_32F, const_cast<float *>(output.ptr<float>())).clone();
    }

    if (output.dims == 3)
    {
        const int height = output.size[1];
        const int width = output.size[2];
        return cv::Mat(height, width, CV_32F, const_cast<float *>(output.ptr<float>())).clone();
    }

    if (output.dims == 2)
    {
        return output.clone();
    }

    return {};
}

cv::Mat normalizeProbability(const cv::Mat &scores)
{
    if (scores.empty())
    {
        return {};
    }

    cv::Mat floatScores;
    scores.convertTo(floatScores, CV_32F);

    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(floatScores, &minValue, &maxValue);
    if (maxValue - minValue > 1e-6)
    {
        floatScores = (floatScores - minValue) / (maxValue - minValue);
    }
    else
    {
        floatScores.setTo(0.0f);
    }
    return floatScores;
}

cv::Mat filterForeground(const cv::Mat &foreground,
                         int morphologyRadius,
                         int minComponentArea,
                         bool keepLargestComponent)
{
    cv::Mat clean = foreground.clone();
    if (morphologyRadius > 0)
    {
        const int radius = std::clamp(morphologyRadius, 1, 64);
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

    const int minArea = std::max(1, minComponentArea);
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
        const bool keep = keepLargestComponent ? (label == largestLabel) : (area >= minArea);
        if (keep && area >= minArea)
        {
            filtered.setTo(255, labels == label);
        }
    }
    return filtered;
}

int cudaDeviceCount()
{
    try
    {
        return cv::cuda::getCudaEnabledDeviceCount();
    }
    catch (const cv::Exception &)
    {
        return 0;
    }
}

bool buildInfoFlagEnabled(const std::string &buildInfo, const std::string &label)
{
    const std::size_t labelPos = buildInfo.find(label);
    if (labelPos == std::string::npos)
    {
        return false;
    }

    const std::size_t lineEnd = buildInfo.find('\n', labelPos);
    const std::string line = buildInfo.substr(labelPos, lineEnd == std::string::npos
                                                            ? std::string::npos
                                                            : lineEnd - labelPos);
    return line.find("YES") != std::string::npos;
}

bool dnnCudaTargetAvailable()
{
    try
    {
        const std::vector<cv::dnn::Target> targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
        return std::find(targets.begin(), targets.end(), cv::dnn::DNN_TARGET_CUDA) != targets.end() ||
               std::find(targets.begin(), targets.end(), cv::dnn::DNN_TARGET_CUDA_FP16) != targets.end();
    }
    catch (const cv::Exception &)
    {
        return false;
    }
}

const char *yesNo(bool value)
{
    return value ? "yes" : "no";
}

} // namespace

std::string u2netDefaultModelFileName()
{
    return "U2Net_v1.onnx";
}

U2NetDnnCapabilities detectU2NetDnnCapabilities()
{
    U2NetDnnCapabilities capabilities;
    const std::string buildInfo = cv::getBuildInformation();
    capabilities.opencvBuiltWithCuda = buildInfoFlagEnabled(buildInfo, "NVIDIA CUDA");
    capabilities.opencvBuiltWithCudnn = buildInfoFlagEnabled(buildInfo, "cuDNN");
    capabilities.cudaDeviceCount = cudaDeviceCount();
    capabilities.hasCudaDevice = capabilities.cudaDeviceCount > 0;
    const bool hasDnnCudaTarget = dnnCudaTargetAvailable();
    capabilities.hasDnnCudaBackend = hasDnnCudaTarget && capabilities.hasCudaDevice;

    std::ostringstream out;
    out << "OpenCV DNN CPU available"
        << "; OpenCV CUDA build=" << yesNo(capabilities.opencvBuiltWithCuda)
        << "; OpenCV cuDNN build=" << yesNo(capabilities.opencvBuiltWithCudnn)
        << "; CUDA devices=" << capabilities.cudaDeviceCount
        << "; DNN CUDA target=" << yesNo(hasDnnCudaTarget)
        << "; DNN CUDA backend=" << (capabilities.hasDnnCudaBackend ? "available" : "unavailable");
    capabilities.summary = out.str();
    return capabilities;
}

U2NetMaskGenerator::U2NetMaskGenerator(const U2NetMaskGeneratorConfig &config)
    : _config(config)
{
    if (_config.modelPath.empty())
    {
        throw std::runtime_error("U2Net ONNX model path is empty.");
    }
    if (!std::filesystem::exists(std::filesystem::u8path(_config.modelPath)))
    {
        throw std::runtime_error("U2Net ONNX model does not exist: " + _config.modelPath);
    }

    try
    {
        loadNet(_config.useCuda);
    }
    catch (const std::exception &)
    {
        if (!_config.useCuda || !_config.allowDeviceFallback)
        {
            throw;
        }
        loadNet(false);
    }
}

void U2NetMaskGenerator::loadNet(bool useCuda)
{
    if (useCuda)
    {
        const U2NetDnnCapabilities capabilities = detectU2NetDnnCapabilities();
        if (!capabilities.opencvBuiltWithCuda)
        {
            throw std::runtime_error(
                "OpenCV was not built with CUDA support. Rebuild with vcpkg feature opencv-dnn-cuda.");
        }
        if (!capabilities.hasDnnCudaBackend && capabilities.hasCudaDevice)
        {
            throw std::runtime_error(
                "OpenCV DNN CUDA target is not available. Rebuild with vcpkg feature opencv-dnn-cuda. " +
                capabilities.summary);
        }
        if (capabilities.cudaDeviceCount <= 0)
        {
            throw std::runtime_error("OpenCV CUDA device is not available. " + capabilities.summary);
        }
        if (_config.cudaDevice < 0 || _config.cudaDevice >= capabilities.cudaDeviceCount)
        {
            throw std::runtime_error("Requested U2Net CUDA device is out of range.");
        }
        cv::cuda::setDevice(_config.cudaDevice);
    }

    _net = cv::dnn::readNetFromONNX(_config.modelPath);
    if (_net.empty())
    {
        throw std::runtime_error("Failed to load U2Net ONNX model: " + _config.modelPath);
    }

    if (useCuda)
    {
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        _usedCuda = true;
    }
    else
    {
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        _usedCuda = false;
    }
}

U2NetMaskResult U2NetMaskGenerator::generate(const cv::Mat &image)
{
    try
    {
        return runForward(image);
    }
    catch (const cv::Exception &)
    {
        if (!_usedCuda || !_config.allowDeviceFallback)
        {
            throw;
        }
        loadNet(false);
        return runForward(image);
    }
}

U2NetMaskResult U2NetMaskGenerator::runForward(const cv::Mat &image)
{
    if (image.empty())
    {
        return {};
    }

    const cv::Mat blob = makeU2NetBlob(image, _config.inputSize);
    _net.setInput(blob);

    cv::Mat output = _net.forward();
    cv::Mat probability = normalizeProbability(firstOutputPlane(output));
    if (probability.empty())
    {
        throw std::runtime_error("U2Net ONNX output is empty or has an unsupported shape.");
    }

    cv::Mat fullResolution;
    cv::resize(probability, fullResolution, image.size(), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat foregroundFloat;
    cv::threshold(fullResolution,
                  foregroundFloat,
                  std::clamp(_config.foregroundThreshold, 0.01f, 0.99f),
                  255.0,
                  cv::THRESH_BINARY);
    cv::Mat foreground;
    foregroundFloat.convertTo(foreground, CV_8U);
    foreground = filterForeground(foreground,
                                  _config.morphologyRadius,
                                  _config.minComponentArea,
                                  _config.keepLargestComponent);

    U2NetMaskResult result;
    cv::bitwise_not(foreground, result.mask);
    result.usedCuda = _usedCuda;
    result.deviceLabel = deviceLabel();
    return result;
}

bool U2NetMaskGenerator::usedCuda() const
{
    return _usedCuda;
}

std::string U2NetMaskGenerator::deviceLabel() const
{
    return _usedCuda ? "CUDA" : "CPU";
}

} // namespace xjw::mask
