#pragma once

/**
 * @file TensorRtSession.h
 * @brief 可枚举、校验并执行固定形状 TensorRT engine 的通用会话。
 */

#include "TensorRtTensorInfo.h"

#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

namespace xjw::inference
{

    struct TensorRtHostBinding
    {
        const char* name = nullptr;
        void* data = nullptr;
        std::size_t bytes = 0;
        bool input = true;
    };

    class TensorRtSession final
    {
    public:
        TensorRtSession(const std::string& enginePath, int cudaDevice);
        TensorRtSession(const QString& enginePath, int cudaDevice);
        ~TensorRtSession();

        TensorRtSession(const TensorRtSession&) = delete;
        TensorRtSession& operator=(const TensorRtSession&) = delete;

        int cudaDevice() const;
        const std::vector<TensorRtTensorInfo>& tensors() const;
        TensorRtTensorInfo tensorInfo(const char* name) const;
        bool hasTensor(const char* name) const;

        void validateTensor(const char* name, TensorRtTensorMode mode, TensorRtTensorDataType type) const;
        void validateTensor(const char* name, nvinfer1::TensorIOMode mode, nvinfer1::DataType type) const;
        nvinfer1::Dims tensorShape(const char* name) const;
        void setInputShape(const char* name, const nvinfer1::Dims& shape);

        /**
         * 输入必须全部显式绑定。输出可以省略；省略的输出仍分配 GPU 缓冲区以满足
         * TensorRT 执行要求，但不会复制回主机。
         */
        void execute(const std::vector<TensorRtHostBinding>& bindings);

    private:
        class Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace xjw::inference
