#pragma once

#include "U2NetMaskGenerator.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace xjw::mask
{

    struct U2NetBackendMetadata
    {
        U2NetBackendType backend = U2NetBackendType::OpenCvCpu;
        U2NetInferencePrecision precision = U2NetInferencePrecision::Unknown;
        std::string deviceLabel;
        bool engineReused = false;
        std::string enginePath;
        std::string fusedOutputName;
        std::string environmentSummary;
    };

    class U2NetInferenceBackend
    {
    public:
        virtual ~U2NetInferenceBackend() = default;

        virtual cv::Mat forward(const cv::Mat& inputBlob) = 0;
        virtual const U2NetBackendMetadata& metadata() const = 0;
    };

    std::unique_ptr<U2NetInferenceBackend> createU2NetOpenCvCpuBackend(const U2NetMaskGeneratorConfig& config);
    std::unique_ptr<U2NetInferenceBackend> createU2NetTensorRtBackend(const U2NetMaskGeneratorConfig& config);

} // namespace xjw::mask
