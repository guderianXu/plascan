#pragma once

#include "TensorRtTensorInfo.h"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#include <cstdint>
#include <vector>

namespace xjw::inference::detail
{

    inline TensorRtTensorMode fromNativeMode(nvinfer1::TensorIOMode mode)
    {
        return mode == nvinfer1::TensorIOMode::kOUTPUT ? TensorRtTensorMode::Output : TensorRtTensorMode::Input;
    }

    inline TensorRtTensorDataType fromNativeDataType(nvinfer1::DataType type)
    {
        switch (type)
        {
        case nvinfer1::DataType::kFLOAT:
            return TensorRtTensorDataType::Float32;
        case nvinfer1::DataType::kHALF:
            return TensorRtTensorDataType::Float16;
        case nvinfer1::DataType::kINT8:
            return TensorRtTensorDataType::Int8;
        case nvinfer1::DataType::kINT32:
            return TensorRtTensorDataType::Int32;
        case nvinfer1::DataType::kBOOL:
            return TensorRtTensorDataType::Bool;
        case nvinfer1::DataType::kUINT8:
            return TensorRtTensorDataType::UInt8;
        case nvinfer1::DataType::kFP8:
            return TensorRtTensorDataType::Float8;
        case nvinfer1::DataType::kBF16:
            return TensorRtTensorDataType::BFloat16;
        case nvinfer1::DataType::kINT64:
            return TensorRtTensorDataType::Int64;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT4:
            return TensorRtTensorDataType::Int4;
        case nvinfer1::DataType::kFP4:
            return TensorRtTensorDataType::Float4E2M1;
#endif
        }
        return TensorRtTensorDataType::Unknown;
    }

    inline std::vector<std::int64_t> fromNativeDims(const nvinfer1::Dims& dims)
    {
        std::vector<std::int64_t> result;
        if (dims.nbDims < 0)
        {
            return result;
        }
        result.reserve(static_cast<std::size_t>(dims.nbDims));
        for (int index = 0; index < dims.nbDims; ++index)
        {
            result.push_back(static_cast<std::int64_t>(dims.d[index]));
        }
        return result;
    }

    inline bool toNativeDims(const std::vector<std::int64_t>& dimensions, nvinfer1::Dims* dims)
    {
        if (!dims || dimensions.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS))
        {
            return false;
        }
        dims->nbDims = static_cast<int>(dimensions.size());
        for (std::size_t index = 0; index < dimensions.size(); ++index)
        {
            if (dimensions[index] <= 0)
            {
                return false;
            }
            dims->d[index] = dimensions[index];
        }
        return true;
    }

} // namespace xjw::inference::detail
