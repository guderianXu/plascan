#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>

namespace xjw::mask
{

struct U2NetDnnCapabilities
{
    bool hasCpu = true;
    bool opencvBuiltWithCuda = false;
    bool opencvBuiltWithCudnn = false;
    bool hasDnnCudaBackend = false;
    bool hasCudaDevice = false;
    int cudaDeviceCount = 0;
    std::string summary;
};

struct U2NetMaskGeneratorConfig
{
    std::string modelPath;
    bool useCuda = false;
    bool allowDeviceFallback = true;
    int cudaDevice = 0;
    int inputSize = 320;
    float foregroundThreshold = 0.5f;
    int morphologyRadius = 1;
    int minComponentArea = 64;
    bool keepLargestComponent = true;
};

struct U2NetMaskResult
{
    cv::Mat mask;
    bool usedCuda = false;
    std::string deviceLabel;
};

std::string u2netDefaultModelFileName();
U2NetDnnCapabilities detectU2NetDnnCapabilities();

class U2NetMaskGenerator
{
public:
    explicit U2NetMaskGenerator(const U2NetMaskGeneratorConfig &config);

    U2NetMaskResult generate(const cv::Mat &image);
    bool usedCuda() const;
    std::string deviceLabel() const;

private:
    U2NetMaskGeneratorConfig _config;
    cv::dnn::Net _net;
    bool _usedCuda = false;

    void loadNet(bool useCuda);
    U2NetMaskResult runForward(const cv::Mat &image);
};

} // namespace xjw::mask
