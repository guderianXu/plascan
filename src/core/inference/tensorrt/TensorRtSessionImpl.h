#pragma once

#include "TensorRtSession.h"
#include "TensorRtSessionResources.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace xjw::inference
{

    class TensorRtSession::Impl
    {
    public:
        Impl(const std::string& enginePath, int cudaDevice);
        ~Impl();

        int cudaDevice() const
        {
            return _cudaDevice;
        }
        const std::vector<TensorRtTensorInfo>& tensors() const
        {
            return _tensors;
        }
        TensorRtTensorInfo tensorInfo(const char* name) const;
        bool hasTensor(const char* name) const;
        void validate(const char* name, TensorRtTensorMode mode, TensorRtTensorDataType type) const;
        void validateNative(const char* name, nvinfer1::TensorIOMode mode, nvinfer1::DataType type) const;
        nvinfer1::Dims shape(const char* name) const;
        void setInputShape(const char* name, const nvinfer1::Dims& shape);
        void execute(const std::vector<TensorRtHostBinding>& bindings);

    private:
        const TensorRtTensorInfo* findTensor(const char* name) const;
        void refreshTensorInfo();
        [[noreturn]] void fail(const std::string& message) const;

        int _cudaDevice = 0;
        detail::TensorRtSessionLogger _logger;
        detail::TensorRtSessionPtr<nvinfer1::IRuntime> _runtime;
        detail::TensorRtSessionPtr<nvinfer1::ICudaEngine> _engine;
        detail::TensorRtSessionPtr<nvinfer1::IExecutionContext> _context;
        std::unique_ptr<detail::CudaStream> _stream;
        std::vector<TensorRtTensorInfo> _tensors;
        std::unordered_map<std::string, detail::DeviceBuffer> _buffers;
    };

} // namespace xjw::inference
