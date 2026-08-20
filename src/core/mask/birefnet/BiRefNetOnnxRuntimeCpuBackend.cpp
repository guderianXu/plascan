#include "BiRefNetInferenceBackend.h"

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace xjw::mask
{
namespace
{

Ort::Env& onnxRuntimeEnvironment()
{
    static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "PlaScan.BiRefNet");
    return environment;
}

class BiRefNetOnnxRuntimeCpuBackend final : public BiRefNetInferenceBackend
{
public:
    explicit BiRefNetOnnxRuntimeCpuBackend(const BiRefNetMaskGeneratorConfig& config)
    {
        if (config.statusCallback)
        {
            config.statusCallback("正在使用 ONNX Runtime CPU 加载 BiRefNet 模型……");
        }

        Ort::SessionOptions options;
        const unsigned int hardware_threads = std::max(1U, std::thread::hardware_concurrency());
        const int inference_threads = static_cast<int>(
            std::clamp(hardware_threads / 2U, 1U, 8U));
        options.SetIntraOpNumThreads(inference_threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const auto* model_bytes = reinterpret_cast<const char8_t*>(config.modelPath.data());
        const std::u8string model_utf8(model_bytes, model_bytes + config.modelPath.size());
        const std::filesystem::path model_path(model_utf8);
        _session = std::make_unique<Ort::Session>(
            onnxRuntimeEnvironment(), model_path.c_str(), options);
        validateContract();

        _metadata.backend = BiRefNetBackendType::OnnxRuntimeCpu;
        _metadata.precision = BiRefNetInferencePrecision::Fp32;
        _metadata.deviceLabel = "ONNX Runtime CPU";
        _metadata.outputName = _outputName;
        _metadata.environmentSummary =
            "ONNX Runtime " + std::string(OrtGetApiBase()->GetVersionString()) +
            " CPU; threads=" + std::to_string(inference_threads) + "; output=" + _outputName;
    }

    cv::Mat forward(const cv::Mat& inputBlob) override
    {
        if (inputBlob.type() != CV_32F || inputBlob.dims != 4 ||
            inputBlob.size[0] != 1 || inputBlob.size[1] != 3 ||
            inputBlob.size[2] != kBiRefNetDynamicInputSize ||
            inputBlob.size[3] != kBiRefNetDynamicInputSize)
        {
            throw std::invalid_argument(
                "BiRefNet ONNX Runtime input must be float32 NCHW [1,3,1024,1024].");
        }

        const cv::Mat contiguous = inputBlob.isContinuous() ? inputBlob : inputBlob.clone();
        const std::array<std::int64_t, 4> input_shape{
            1, 3, kBiRefNetDynamicInputSize, kBiRefNetDynamicInputSize};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory,
            const_cast<float*>(contiguous.ptr<float>()),
            contiguous.total(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {_inputName.c_str()};
        const char* output_names[] = {_outputName.c_str()};
        std::vector<Ort::Value> outputs = _session->Run(
            Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        if (outputs.size() != 1 || !outputs.front().IsTensor())
        {
            throw std::runtime_error("BiRefNet ONNX Runtime did not return one tensor output.");
        }

        const Ort::TensorTypeAndShapeInfo output_info =
            outputs.front().GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> shape = output_info.GetShape();
        const std::array<std::int64_t, 4> expected_shape{
            1, 1, kBiRefNetDynamicInputSize, kBiRefNetDynamicInputSize};
        if (shape.size() != expected_shape.size() ||
            !std::equal(shape.cbegin(), shape.cend(), expected_shape.cbegin()))
        {
            throw std::runtime_error(
                "BiRefNet ONNX Runtime returned an unexpected output shape.");
        }

        const int dimensions[] = {1, 1, kBiRefNetDynamicInputSize, kBiRefNetDynamicInputSize};
        cv::Mat result(4, dimensions, CV_32F);
        std::memcpy(result.ptr<float>(),
                    outputs.front().GetTensorData<float>(),
                    result.total() * sizeof(float));
        return result;
    }

    const BiRefNetBackendMetadata& metadata() const override
    {
        return _metadata;
    }

private:
    void validateContract()
    {
        if (_session->GetInputCount() != 1 || _session->GetOutputCount() != 1)
        {
            throw std::runtime_error(
                "BiRefNet ONNX model must have exactly one input and one output.");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = _session->GetInputNameAllocated(0, allocator);
        const auto output_name = _session->GetOutputNameAllocated(0, allocator);
        _inputName = input_name.get();
        _outputName = output_name.get();
        if (_inputName != "input_image" || _outputName != "output_image")
        {
            throw std::runtime_error(
                "BiRefNet ONNX model must expose input_image and output_image.");
        }

        const std::vector<std::int64_t> input_shape =
            _session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const std::vector<std::int64_t> output_shape =
            _session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const std::vector<std::int64_t> expected_input{
            1, 3, kBiRefNetDynamicInputSize, kBiRefNetDynamicInputSize};
        const std::vector<std::int64_t> expected_output{
            1, 1, kBiRefNetDynamicInputSize, kBiRefNetDynamicInputSize};
        if (input_shape != expected_input || output_shape != expected_output)
        {
            throw std::runtime_error(
                "BiRefNet ONNX model must use fixed input [1,3,1024,1024] and "
                "output [1,1,1024,1024].");
        }
    }

    std::unique_ptr<Ort::Session> _session;
    std::string _inputName;
    std::string _outputName;
    BiRefNetBackendMetadata _metadata;
};

} // namespace

std::unique_ptr<BiRefNetInferenceBackend>
createBiRefNetOnnxRuntimeCpuBackend(const BiRefNetMaskGeneratorConfig& config)
{
    return std::make_unique<BiRefNetOnnxRuntimeCpuBackend>(config);
}

} // namespace xjw::mask
