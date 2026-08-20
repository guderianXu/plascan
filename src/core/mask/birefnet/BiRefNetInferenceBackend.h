#pragma once

#include "BiRefNetMaskGenerator.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace xjw::mask
{

struct BiRefNetBackendMetadata
{
    BiRefNetBackendType backend = BiRefNetBackendType::OnnxRuntimeCpu;
    BiRefNetInferencePrecision precision = BiRefNetInferencePrecision::Unknown;
    std::string deviceLabel;
    bool engineReused = false;
    std::string enginePath;
    std::string outputName;
    std::string environmentSummary;
};

class BiRefNetInferenceBackend
{
public:
    virtual ~BiRefNetInferenceBackend() = default;

    virtual cv::Mat forward(const cv::Mat& inputBlob) = 0;
    virtual const BiRefNetBackendMetadata& metadata() const = 0;
};

std::unique_ptr<BiRefNetInferenceBackend>
createBiRefNetTensorRtBackend(const BiRefNetMaskGeneratorConfig& config);
std::unique_ptr<BiRefNetInferenceBackend>
createBiRefNetOnnxRuntimeCpuBackend(const BiRefNetMaskGeneratorConfig& config);

} // namespace xjw::mask
