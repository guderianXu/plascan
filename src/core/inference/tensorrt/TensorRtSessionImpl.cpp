#include "TensorRtSessionImpl.h"

#include "TensorRtTypeConversions.h"
#include "io/PathIO.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xjw::inference
{

    TensorRtSession::Impl::Impl(const std::string& enginePath, int cudaDevice) : _cudaDevice(std::max(0, cudaDevice))
    {
        if (enginePath.empty())
        {
            throw std::invalid_argument("[TensorRtSession] engine path is empty");
        }
        detail::checkCuda(cudaSetDevice(_cudaDevice), "cudaSetDevice");
        _stream = std::make_unique<detail::CudaStream>();

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
        refreshTensorInfo();
    }

    TensorRtSession::Impl::~Impl() = default;

    TensorRtTensorInfo TensorRtSession::Impl::tensorInfo(const char* name) const
    {
        const TensorRtTensorInfo* tensor = findTensor(name);
        if (!tensor)
        {
            throw std::runtime_error(std::string("[TensorRtSession] missing tensor: ") + (name ? name : "<null>"));
        }
        return *tensor;
    }

    bool TensorRtSession::Impl::hasTensor(const char* name) const
    {
        return findTensor(name) != nullptr;
    }

    void TensorRtSession::Impl::validate(const char* name, TensorRtTensorMode mode, TensorRtTensorDataType type) const
    {
        const TensorRtTensorInfo tensor = tensorInfo(name);
        if (tensor.mode != mode)
        {
            throw std::runtime_error(std::string("[TensorRtSession] invalid tensor mode: ") + name);
        }
        if (tensor.dataType != type)
        {
            throw std::runtime_error(std::string("[TensorRtSession] invalid tensor type: ") + name);
        }
    }

    void
    TensorRtSession::Impl::validateNative(const char* name, nvinfer1::TensorIOMode mode, nvinfer1::DataType type) const
    {
        if (!name || _engine->getTensorIOMode(name) != mode)
        {
            throw std::runtime_error(std::string("[TensorRtSession] missing tensor: ") + (name ? name : "<null>"));
        }
        if (_engine->getTensorDataType(name) != type)
        {
            throw std::runtime_error(std::string("[TensorRtSession] invalid tensor type: ") + name);
        }
    }

    nvinfer1::Dims TensorRtSession::Impl::shape(const char* name) const
    {
        if (!hasTensor(name))
        {
            throw std::runtime_error(std::string("[TensorRtSession] missing tensor: ") + (name ? name : "<null>"));
        }
        return _context->getTensorShape(name);
    }

    void TensorRtSession::Impl::setInputShape(const char* name, const nvinfer1::Dims& shape)
    {
        if (!name || _engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT ||
            !_context->setInputShape(name, shape))
        {
            fail(std::string("cannot set input shape for ") + (name ? name : "<null>"));
        }
        refreshTensorInfo();
    }

    const TensorRtTensorInfo* TensorRtSession::Impl::findTensor(const char* name) const
    {
        if (!name)
        {
            return nullptr;
        }
        const QString candidate = QString::fromUtf8(name);
        const auto iterator =
            std::find_if(_tensors.begin(),
                         _tensors.end(),
                         [&candidate](const TensorRtTensorInfo& tensor) { return tensor.name == candidate; });
        return iterator == _tensors.end() ? nullptr : &*iterator;
    }

    void TensorRtSession::Impl::refreshTensorInfo()
    {
        _tensors.clear();
        _tensors.reserve(static_cast<std::size_t>(_engine->getNbIOTensors()));
        for (int index = 0; index < _engine->getNbIOTensors(); ++index)
        {
            const char* name = _engine->getIOTensorName(index);
            if (!name)
            {
                continue;
            }
            TensorRtTensorInfo tensor;
            tensor.name = QString::fromUtf8(name);
            tensor.mode = detail::fromNativeMode(_engine->getTensorIOMode(name));
            tensor.dataType = detail::fromNativeDataType(_engine->getTensorDataType(name));
            tensor.dimensions = detail::fromNativeDims(_context->getTensorShape(name));
            _tensors.push_back(std::move(tensor));
        }
    }

    [[noreturn]] void TensorRtSession::Impl::fail(const std::string& message) const
    {
        throw std::runtime_error("[TensorRtSession] " + message +
                                 (_logger.error().empty() ? std::string() : ": " + _logger.error()));
    }

} // namespace xjw::inference
