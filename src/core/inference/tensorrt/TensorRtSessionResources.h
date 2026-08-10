#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace xjw::inference::detail
{

    inline void checkCuda(cudaError_t status, const char* operation)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string("[TensorRtSession] ") + operation + ": " + cudaGetErrorString(status));
        }
    }

    class TensorRtSessionLogger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* message) noexcept override
        {
            if (severity <= Severity::kERROR && message)
            {
                try
                {
                    _error = message;
                }
                catch (...)
                {
                }
            }
        }

        const std::string& error() const
        {
            return _error;
        }

    private:
        std::string _error;
    };

    template <typename T> struct TensorRtSessionDeleter
    {
        void operator()(T* value) const noexcept
        {
            delete value;
        }
    };

    template <typename T> using TensorRtSessionPtr = std::unique_ptr<T, TensorRtSessionDeleter<T>>;

    class CudaStream final
    {
    public:
        CudaStream()
        {
            checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        }

        ~CudaStream()
        {
            if (_stream)
            {
                cudaStreamDestroy(_stream);
            }
        }

        CudaStream(const CudaStream&) = delete;
        CudaStream& operator=(const CudaStream&) = delete;

        cudaStream_t get() const
        {
            return _stream;
        }

    private:
        cudaStream_t _stream = nullptr;
    };

    class DeviceBuffer final
    {
    public:
        ~DeviceBuffer()
        {
            if (_data)
            {
                cudaFree(_data);
            }
        }

        DeviceBuffer() = default;
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        void ensure(std::size_t bytes)
        {
            if (bytes == 0)
            {
                throw std::invalid_argument("[TensorRtSession] zero-sized device buffer");
            }
            if (bytes <= _capacity)
            {
                return;
            }
            if (_data)
            {
                checkCuda(cudaFree(_data), "cudaFree");
                _data = nullptr;
                _capacity = 0;
            }
            checkCuda(cudaMalloc(&_data, bytes), "cudaMalloc");
            _capacity = bytes;
        }

        void* data() const
        {
            return _data;
        }

    private:
        void* _data = nullptr;
        std::size_t _capacity = 0;
    };

} // namespace xjw::inference::detail
