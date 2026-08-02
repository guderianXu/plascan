#include "TensorRtSession.h"

#include "io/PathIO.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xjw::image_matching
{
namespace
{

void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string("[TensorRtSession] ") + operation +
                                 ": " + cudaGetErrorString(status));
    }
}

class Logger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *message) noexcept override
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

    const std::string &error() const { return _error; }

private:
    std::string _error;
};

template<typename T>
struct TensorRtDeleter
{
    void operator()(T *value) const noexcept { delete value; }
};

template<typename T>
using TensorRtPtr = std::unique_ptr<T, TensorRtDeleter<T>>;

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

    void ensure(std::size_t bytes)
    {
        if (bytes <= _capacity)
        {
            return;
        }
        if (_data)
        {
            checkCuda(cudaFree(_data), "cudaFree");
        }
        checkCuda(cudaMalloc(&_data, bytes), "cudaMalloc");
        _capacity = bytes;
    }

    void *data() const { return _data; }

private:
    void *_data = nullptr;
    std::size_t _capacity = 0;
};

} // namespace

class TensorRtSession::Impl
{
public:
    Impl(const std::string &enginePath, int cudaDevice)
        : _cudaDevice(std::max(0, cudaDevice))
    {
        if (enginePath.empty())
        {
            throw std::invalid_argument("[TensorRtSession] engine path is empty");
        }
        checkCuda(cudaSetDevice(_cudaDevice), "cudaSetDevice");
        checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");

        std::ifstream input = common::io::openInputFile(enginePath);
        if (!input)
        {
            throw std::runtime_error("[TensorRtSession] cannot open engine: " + enginePath);
        }
        input.seekg(0, std::ios::end);
        const std::streamoff size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (size <= 0)
        {
            throw std::runtime_error("[TensorRtSession] engine is empty: " + enginePath);
        }
        std::vector<char> bytes(static_cast<std::size_t>(size));
        input.read(bytes.data(), static_cast<std::streamsize>(size));
        if (!input)
        {
            throw std::runtime_error("[TensorRtSession] failed to read engine: " + enginePath);
        }

        _runtime.reset(nvinfer1::createInferRuntime(_logger));
        if (!_runtime)
        {
            fail("cannot create runtime");
        }
        _engine.reset(_runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!_engine)
        {
            fail("cannot deserialize engine");
        }
        _context.reset(_engine->createExecutionContext());
        if (!_context)
        {
            fail("cannot create execution context");
        }
    }

    ~Impl()
    {
        _context.reset();
        _engine.reset();
        _runtime.reset();
        if (_stream)
        {
            cudaStreamDestroy(_stream);
        }
    }

    void validate(const char *name,
                  nvinfer1::TensorIOMode mode,
                  nvinfer1::DataType type) const
    {
        if (!name || _engine->getTensorIOMode(name) != mode)
        {
            throw std::runtime_error(std::string("[TensorRtSession] missing tensor: ") +
                                     (name ? name : "<null>"));
        }
        if (_engine->getTensorDataType(name) != type)
        {
            throw std::runtime_error(std::string("[TensorRtSession] invalid tensor type: ") + name);
        }
    }

    nvinfer1::Dims shape(const char *name) const
    {
        return _engine->getTensorShape(name);
    }

    void execute(const std::vector<TensorRtHostBinding> &bindings)
    {
        checkCuda(cudaSetDevice(_cudaDevice), "cudaSetDevice");
        for (const TensorRtHostBinding &binding : bindings)
        {
            if (!binding.name || !binding.data || binding.bytes == 0)
            {
                throw std::invalid_argument("[TensorRtSession] invalid host binding");
            }
            auto &buffer = _buffers[std::string(binding.name)];
            buffer.ensure(binding.bytes);
            if (binding.input)
            {
                checkCuda(cudaMemcpyAsync(buffer.data(), binding.data, binding.bytes,
                                          cudaMemcpyHostToDevice, _stream),
                          "cudaMemcpyAsync(H2D)");
            }
            if (!_context->setTensorAddress(binding.name, buffer.data()))
            {
                fail(std::string("cannot bind tensor ") + binding.name);
            }
        }
        if (!_context->enqueueV3(_stream))
        {
            fail("inference failed");
        }
        for (const TensorRtHostBinding &binding : bindings)
        {
            if (!binding.input)
            {
                DeviceBuffer &buffer = _buffers.at(std::string(binding.name));
                checkCuda(cudaMemcpyAsync(binding.data, buffer.data(), binding.bytes,
                                          cudaMemcpyDeviceToHost, _stream),
                          "cudaMemcpyAsync(D2H)");
            }
        }
        checkCuda(cudaStreamSynchronize(_stream), "cudaStreamSynchronize");
    }

private:
    [[noreturn]] void fail(const std::string &message) const
    {
        throw std::runtime_error("[TensorRtSession] " + message +
                                 (_logger.error().empty() ? std::string()
                                                          : ": " + _logger.error()));
    }

    int _cudaDevice = 0;
    Logger _logger;
    TensorRtPtr<nvinfer1::IRuntime> _runtime;
    TensorRtPtr<nvinfer1::ICudaEngine> _engine;
    TensorRtPtr<nvinfer1::IExecutionContext> _context;
    cudaStream_t _stream = nullptr;
    std::unordered_map<std::string, DeviceBuffer> _buffers;
};

TensorRtSession::TensorRtSession(const std::string &enginePath, int cudaDevice)
    : _impl(std::make_unique<Impl>(enginePath, cudaDevice))
{
}

TensorRtSession::~TensorRtSession() = default;

void TensorRtSession::validateTensor(const char *name,
                                     nvinfer1::TensorIOMode mode,
                                     nvinfer1::DataType type) const
{
    _impl->validate(name, mode, type);
}

nvinfer1::Dims TensorRtSession::tensorShape(const char *name) const
{
    return _impl->shape(name);
}

void TensorRtSession::execute(const std::vector<TensorRtHostBinding> &bindings)
{
    _impl->execute(bindings);
}

} // namespace xjw::image_matching
