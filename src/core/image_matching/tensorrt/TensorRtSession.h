#pragma once

/**
 * @file TensorRtSession.h
 * @brief 旧影像匹配 include 路径的兼容转发。
 */

#include "inference/tensorrt/TensorRtSession.h"

namespace xjw::image_matching
{

    using TensorRtHostBinding = inference::TensorRtHostBinding;
    using TensorRtSession = inference::TensorRtSession;
    using TensorRtTensorMode = inference::TensorRtTensorMode;
    using TensorRtTensorDataType = inference::TensorRtTensorDataType;
    using TensorRtTensorInfo = inference::TensorRtTensorInfo;

} // namespace xjw::image_matching
