#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace xjw::mask
{

inline constexpr int kBiRefNetDynamicInputSize = 1024;

enum class BiRefNetBackendType
{
    Auto,
    TensorRt
};

enum class BiRefNetInferencePrecision
{
    Unknown,
    Fp32,
    Fp16
};

struct BiRefNetInferenceCapabilities
{
    bool tensorRtCompiled = false;
    bool tensorRtAvailable = false;
    bool hasCudaDevice = false;
    int cudaDeviceCount = 0;
    bool supportsFp16 = false;
    std::string tensorRtVersion;
    std::string gpuName;
    std::string errorMessage;
    std::string summary;
};

struct BiRefNetMaskGeneratorConfig
{
    std::string modelPath;
    BiRefNetBackendType backend = BiRefNetBackendType::Auto;
    int cudaDevice = 0;
    int inputSize = kBiRefNetDynamicInputSize;
    float foregroundThreshold = 0.5f;
    int morphologyRadius = 0;
    int minComponentArea = 64;
    bool keepLargestComponent = false;

    std::string engineCacheDirectory;
    bool preferFp16 = true;
    std::uint64_t tensorRtWorkspaceBytes = 0;
    int tensorRtBuilderOptimizationLevel = 3;
    int tensorRtMaximumAuxiliaryStreams = 0;
    std::function<void(const std::string&)> statusCallback;
};

struct BiRefNetMaskResult
{
    cv::Mat mask;
    BiRefNetBackendType requestedBackend = BiRefNetBackendType::Auto;
    BiRefNetBackendType actualBackend = BiRefNetBackendType::TensorRt;
    BiRefNetInferencePrecision precision = BiRefNetInferencePrecision::Unknown;
    bool usedCuda = false;
    bool engineReused = false;
    std::string deviceLabel;
    std::string enginePath;
    std::string outputName;
    std::string environmentSummary;
    std::string modelSha256;
};

std::string biRefNetDynamicDefaultModelFileName();
std::string biRefNetBackendTypeToken(BiRefNetBackendType backend);
std::optional<BiRefNetBackendType> parseBiRefNetBackendType(const std::string& token);
std::string biRefNetInferencePrecisionToken(BiRefNetInferencePrecision precision);
BiRefNetInferenceCapabilities detectBiRefNetInferenceCapabilities(int cudaDevice = 0);

class BiRefNetMaskGenerator
{
public:
    explicit BiRefNetMaskGenerator(const BiRefNetMaskGeneratorConfig& config);
    ~BiRefNetMaskGenerator();

    BiRefNetMaskGenerator(const BiRefNetMaskGenerator&) = delete;
    BiRefNetMaskGenerator& operator=(const BiRefNetMaskGenerator&) = delete;
    BiRefNetMaskGenerator(BiRefNetMaskGenerator&&) noexcept;
    BiRefNetMaskGenerator& operator=(BiRefNetMaskGenerator&&) noexcept;

    BiRefNetMaskResult generate(const cv::Mat& image);
    BiRefNetBackendType actualBackend() const;
    BiRefNetInferencePrecision precision() const;
    std::string deviceLabel() const;
    bool engineReused() const;
    std::string enginePath() const;
    std::string environmentSummary() const;
    std::string modelSha256() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace xjw::mask
