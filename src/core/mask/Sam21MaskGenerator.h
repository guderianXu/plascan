#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace xjw::mask
{

enum class Sam21ModelVariant
{
    Tiny,
    Small,
    BasePlus,
    Large
};

struct Sam21TorchScriptModelNames
{
    std::string encoder;
    std::string decoder;
};

struct Sam21PointPrompt
{
    float x = 0.0f;
    float y = 0.0f;
    int label = 1; // 1=foreground, 0=background.
};

struct Sam21Prompt
{
    std::vector<Sam21PointPrompt> points;
    bool hasBox = false;
    cv::Rect2f box;
    cv::Mat maskInput;

    bool empty() const;
    static Sam21Prompt fullImageBox(const cv::Size &imageSize);
    static Sam21Prompt autoBox(const cv::Mat &image, float paddingRatio = 0.03f);
};

struct Sam21MaskGeneratorConfig
{
    std::string encoderModelPath;
    std::string decoderModelPath;
    std::string cpuEncoderModelPath;
    std::string cpuDecoderModelPath;
    bool useCuda = true;
    int cudaDevice = 0;
    bool allowDeviceFallback = true;
    int inputSize = 1024;
    double maskThreshold = 0.0;
    bool multimaskOutput = true;
};

struct Sam21MaskResult
{
    cv::Mat mask;
    bool usedCuda = false;
    std::string deviceLabel;
    std::string warning;
    double score = 0.0;
};

std::string sam21VariantToken(Sam21ModelVariant variant);
Sam21ModelVariant sam21VariantFromToken(const std::string &token);
Sam21TorchScriptModelNames sam21TorchScriptModelNames(Sam21ModelVariant variant, bool useCuda);

class Sam21MaskGenerator
{
public:
    explicit Sam21MaskGenerator(const Sam21MaskGeneratorConfig &config);
    ~Sam21MaskGenerator();

    Sam21MaskGenerator(const Sam21MaskGenerator &) = delete;
    Sam21MaskGenerator &operator=(const Sam21MaskGenerator &) = delete;

    Sam21MaskResult generate(const cv::Mat &image, const Sam21Prompt &prompt);
    bool usedCuda() const;
    std::string deviceLabel() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace xjw::mask
