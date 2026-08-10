#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace xjw::mask
{

    inline constexpr int kU2NetModelInputSize = 320;

    enum class U2NetBackendType
    {
        Auto,
        TensorRt,
        OpenCvCpu
    };

    enum class U2NetInferencePrecision
    {
        Unknown,
        Fp32,
        Fp16
    };

    struct U2NetInferenceCapabilities
    {
        bool hasOpenCvCpu = true;
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

    struct U2NetMaskGeneratorConfig
    {
        std::string modelPath;
        U2NetBackendType backend = U2NetBackendType::Auto;
        bool allowDeviceFallback = false;
        int cudaDevice = 0;
        int inputSize = kU2NetModelInputSize;
        float foregroundThreshold = 0.5f;
        int morphologyRadius = 1;
        int minComponentArea = 64;
        bool keepLargestComponent = true;

        std::string engineCacheDirectory;
        bool preferFp16 = true;
        std::uint64_t tensorRtWorkspaceBytes = 0;
        int tensorRtBuilderOptimizationLevel = 3;
        int tensorRtMaximumAuxiliaryStreams = 0;
        std::function<void(const std::string&)> statusCallback;
    };

    struct U2NetMaskResult
    {
        cv::Mat mask;
        U2NetBackendType requestedBackend = U2NetBackendType::Auto;
        U2NetBackendType actualBackend = U2NetBackendType::OpenCvCpu;
        U2NetInferencePrecision precision = U2NetInferencePrecision::Unknown;
        bool usedCuda = false;
        bool deviceFallback = false;
        bool engineReused = false;
        std::string deviceLabel;
        std::string fallbackReason;
        std::string enginePath;
        std::string fusedOutputName;
        std::string environmentSummary;
        std::string modelSha256;
    };

    std::string u2netDefaultModelFileName();
    std::string u2netBackendTypeToken(U2NetBackendType backend);
    std::string u2netBackendTypeLabel(U2NetBackendType backend);
    std::optional<U2NetBackendType> parseU2NetBackendType(const std::string& token);
    std::string u2netInferencePrecisionToken(U2NetInferencePrecision precision);
    U2NetInferenceCapabilities detectU2NetInferenceCapabilities(int cudaDevice = 0);

    class U2NetMaskGenerator
    {
    public:
        explicit U2NetMaskGenerator(const U2NetMaskGeneratorConfig& config);
        ~U2NetMaskGenerator();

        U2NetMaskGenerator(const U2NetMaskGenerator&) = delete;
        U2NetMaskGenerator& operator=(const U2NetMaskGenerator&) = delete;
        U2NetMaskGenerator(U2NetMaskGenerator&&) noexcept;
        U2NetMaskGenerator& operator=(U2NetMaskGenerator&&) noexcept;

        U2NetMaskResult generate(const cv::Mat& image);
        bool usedCuda() const;
        U2NetBackendType actualBackend() const;
        U2NetInferencePrecision precision() const;
        std::string deviceLabel() const;
        bool engineReused() const;
        std::string enginePath() const;
        std::string environmentSummary() const;
        std::string modelSha256() const;
        std::string fallbackReason() const;

    private:
        class Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace xjw::mask
