#pragma once

/**
 * @file TensorRtEngineBuilder.h
 * @brief 旧影像匹配 include 路径的兼容转发。
 */

#include "inference/tensorrt/TensorRtEngineBuilder.h"

namespace xjw::image_matching
{

    using TensorRtBuildPrecision = inference::TensorRtBuildPrecision;
    using TensorRtInputShape = inference::TensorRtInputShape;
    using TensorRtEngineBuildRequest = inference::TensorRtEngineBuildRequest;
    using TensorRtEngineBuildResult = inference::TensorRtEngineBuildResult;
    using TensorRtTensorMode = inference::TensorRtTensorMode;
    using TensorRtTensorDataType = inference::TensorRtTensorDataType;
    using TensorRtTensorInfo = inference::TensorRtTensorInfo;

    inline TensorRtEngineBuildResult ensureTensorRtEngine(const TensorRtEngineBuildRequest& request)
    {
        return inference::ensureTensorRtEngine(request);
    }

} // namespace xjw::image_matching
