#pragma once

/**
 * @file LoMaRTensorRtBackend.h
 * @brief DaD + DeDoDe-G 与 LoMa-R 的纯 TensorRT 运行后端。
 */

#include "ImageMatchingAlgorithm.h"

#include <memory>

namespace xjw::image_matching
{

struct LoMaRTensorRtConfig
{
    QString featureEnginePath;
    QString matcherEnginePath;
    int cudaDevice = 0;
    int inputWidth = 784;
    int inputHeight = 784;
    int keypointCount = 2048;
    int descriptorDimension = 256;
    int maxKeypoints = 2048;
    float matchThreshold = 0.1f;
};

class LoMaRTensorRtBackend final
{
public:
    explicit LoMaRTensorRtBackend(LoMaRTensorRtConfig config);
    ~LoMaRTensorRtBackend();

    FeatureSet extract(const ImageFeatureInput &input) const;
    MatchResult match(const FeatureSet &features0, const FeatureSet &features1);

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace xjw::image_matching
