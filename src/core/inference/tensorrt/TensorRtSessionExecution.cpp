#include "TensorRtSessionImpl.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace xjw::inference
{

    void TensorRtSession::Impl::execute(const std::vector<TensorRtHostBinding>& bindings)
    {
        detail::checkCuda(cudaSetDevice(_cudaDevice), "cudaSetDevice");
        refreshTensorInfo();

        std::unordered_map<std::string, const TensorRtHostBinding*> host_bindings;
        host_bindings.reserve(bindings.size());
        for (const TensorRtHostBinding& binding : bindings)
        {
            if (!binding.name || !binding.data || binding.bytes == 0)
            {
                throw std::invalid_argument("[TensorRtSession] invalid host binding");
            }
            if (!host_bindings.emplace(binding.name, &binding).second)
            {
                throw std::invalid_argument(std::string("[TensorRtSession] duplicate binding: ") + binding.name);
            }
            const TensorRtTensorInfo tensor = tensorInfo(binding.name);
            const bool is_input = tensor.mode == TensorRtTensorMode::Input;
            if (binding.input != is_input)
            {
                throw std::invalid_argument(std::string("[TensorRtSession] binding mode mismatch: ") + binding.name);
            }
            const std::uint64_t expected_bytes = tensor.byteCount();
            if (expected_bytes == 0 || expected_bytes != binding.bytes)
            {
                throw std::invalid_argument(std::string("[TensorRtSession] binding byte size mismatch: ") +
                                            binding.name + ", expected " + std::to_string(expected_bytes) + ", got " +
                                            std::to_string(binding.bytes));
            }
        }

        for (const TensorRtTensorInfo& tensor : _tensors)
        {
            const std::string name = tensor.name.toStdString();
            const auto host_iterator = host_bindings.find(name);
            if (tensor.mode == TensorRtTensorMode::Input && host_iterator == host_bindings.end())
            {
                throw std::invalid_argument("[TensorRtSession] missing input binding: " + name);
            }
            if (_engine->isShapeInferenceIO(name.c_str()))
            {
                throw std::runtime_error("[TensorRtSession] shape inference I/O is not supported: " + name);
            }
            const std::uint64_t byte_count = tensor.byteCount();
            if (byte_count == 0 || byte_count > std::numeric_limits<std::size_t>::max())
            {
                throw std::runtime_error("[TensorRtSession] unresolved or unsupported tensor shape/type: " + name);
            }
            detail::DeviceBuffer& buffer = _buffers[name];
            buffer.ensure(static_cast<std::size_t>(byte_count));
            if (!_context->setTensorAddress(name.c_str(), buffer.data()))
            {
                fail("cannot bind tensor " + name);
            }
            if (tensor.mode == TensorRtTensorMode::Input)
            {
                const TensorRtHostBinding& binding = *host_iterator->second;
                detail::checkCuda(
                    cudaMemcpyAsync(buffer.data(), binding.data, binding.bytes, cudaMemcpyHostToDevice, _stream->get()),
                    "cudaMemcpyAsync(H2D)");
            }
        }

        if (!_context->enqueueV3(_stream->get()))
        {
            fail("inference failed");
        }
        for (const TensorRtHostBinding& binding : bindings)
        {
            if (!binding.input)
            {
                detail::DeviceBuffer& buffer = _buffers.at(std::string(binding.name));
                detail::checkCuda(
                    cudaMemcpyAsync(binding.data, buffer.data(), binding.bytes, cudaMemcpyDeviceToHost, _stream->get()),
                    "cudaMemcpyAsync(D2H)");
            }
        }
        detail::checkCuda(cudaStreamSynchronize(_stream->get()), "cudaStreamSynchronize");
    }

} // namespace xjw::inference
