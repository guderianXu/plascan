#include "BiRefNetMaskGenerator.h"

#include "BiRefNetImageProcessing.h"
#include "BiRefNetInferenceBackend.h"
#include "inference/tensorrt/TensorRtCapabilities.h"

#include <QCryptographicHash>
#include <QFile>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace xjw::mask
{
namespace
{

std::string toUtf8(const QString& value)
{
    const QByteArray bytes = value.toUtf8();
    return std::string(bytes.constData(), static_cast<std::size_t>(bytes.size()));
}

std::string normalizedToken(std::string token)
{
    std::transform(token.begin(),
                   token.end(),
                   token.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    token.erase(std::remove_if(token.begin(),
                               token.end(),
                               [](char value)
                               {
                                   return value == '-' || value == '_' ||
                                          std::isspace(static_cast<unsigned char>(value));
                               }),
                token.end());
    return token;
}

std::string sha256File(const std::string& path)
{
    QFile file(QString::fromUtf8(path.c_str()));
    if (!file.open(QIODevice::ReadOnly))
    {
        throw std::runtime_error("Cannot read the BiRefNet ONNX model for SHA-256: " + path);
    }

    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        const QByteArray block = file.read(4 * 1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError)
        {
            throw std::runtime_error("Failed to calculate the BiRefNet ONNX SHA-256: " + path);
        }
        hash.addData(block);
    }
    return hash.result().toHex().toStdString();
}

} // namespace

std::string biRefNetDynamicDefaultModelFileName()
{
    return "BiRefNet_dynamic_1024.onnx";
}

std::string biRefNetBackendTypeToken(BiRefNetBackendType backend)
{
    switch (backend)
    {
    case BiRefNetBackendType::Auto:
        return "auto";
    case BiRefNetBackendType::TensorRt:
        return "tensorrt";
    }
    return "auto";
}

std::optional<BiRefNetBackendType> parseBiRefNetBackendType(const std::string& token)
{
    const std::string normalized = normalizedToken(token);
    if (normalized == "auto")
    {
        return BiRefNetBackendType::Auto;
    }
    if (normalized == "tensorrt" || normalized == "cuda" || normalized == "gpu")
    {
        return BiRefNetBackendType::TensorRt;
    }
    return std::nullopt;
}

std::string biRefNetInferencePrecisionToken(BiRefNetInferencePrecision precision)
{
    switch (precision)
    {
    case BiRefNetInferencePrecision::Fp16:
        return "fp16";
    case BiRefNetInferencePrecision::Fp32:
        return "fp32";
    case BiRefNetInferencePrecision::Unknown:
        return "unknown";
    }
    return "unknown";
}

BiRefNetInferenceCapabilities detectBiRefNetInferenceCapabilities(int cudaDevice)
{
    const inference::TensorRtCapabilities detected = inference::queryTensorRtCapabilities(cudaDevice);
    BiRefNetInferenceCapabilities result;
    result.tensorRtCompiled = detected.compiled;
    result.tensorRtAvailable = detected.isAvailable();
    result.hasCudaDevice = detected.cudaAvailable;
    result.cudaDeviceCount = detected.deviceCount;
    result.supportsFp16 = detected.fastFp16;
    result.tensorRtVersion = toUtf8(detected.tensorRtVersion);
    result.gpuName = toUtf8(detected.gpuName);
    result.errorMessage = toUtf8(detected.errorMessage);

    std::ostringstream summary;
    summary << "BiRefNet requires TensorRT; compiled=" << (result.tensorRtCompiled ? "yes" : "no")
            << "; available=" << (result.tensorRtAvailable ? "yes" : "no")
            << "; CUDA devices=" << result.cudaDeviceCount
            << "; FP16=" << (result.supportsFp16 ? "available" : "unavailable");
    if (!result.gpuName.empty())
    {
        summary << "; GPU=" << result.gpuName;
    }
    if (!result.errorMessage.empty())
    {
        summary << "; reason=" << result.errorMessage;
    }
    result.summary = summary.str();
    return result;
}

class BiRefNetMaskGenerator::Impl
{
public:
    explicit Impl(BiRefNetMaskGeneratorConfig config) : _config(std::move(config))
    {
        if (_config.modelPath.empty())
        {
            throw std::runtime_error("BiRefNet ONNX model path is empty.");
        }
        if (!std::filesystem::exists(std::filesystem::u8path(_config.modelPath)))
        {
            throw std::runtime_error("BiRefNet ONNX model does not exist: " + _config.modelPath);
        }
        if (_config.inputSize != kBiRefNetDynamicInputSize)
        {
            throw std::invalid_argument(
                "The bundled BiRefNet Dynamic deployment model requires a fixed 1024x1024 input.");
        }
        _modelSha256 = sha256File(_config.modelPath);
        _backend = createBiRefNetTensorRtBackend(_config);
    }

    BiRefNetMaskResult generate(const cv::Mat& image)
    {
        if (image.empty())
        {
            return makeResult({});
        }

        BiRefNetLetterbox letterbox;
        const cv::Mat blob = makeBiRefNetBlob(image, _config.inputSize, &letterbox);
        const cv::Mat probability = biRefNetProbabilityFromOutput(_backend->forward(blob));
        cv::Mat mask = makeBiRefNetMask(probability,
                                        letterbox,
                                        _config.foregroundThreshold,
                                        _config.morphologyRadius,
                                        _config.minComponentArea,
                                        _config.keepLargestComponent);
        return makeResult(std::move(mask));
    }

    const BiRefNetBackendMetadata& metadata() const
    {
        return _backend->metadata();
    }

    const std::string& modelSha256() const
    {
        return _modelSha256;
    }

private:
    BiRefNetMaskResult makeResult(cv::Mat mask) const
    {
        const BiRefNetBackendMetadata& metadata = _backend->metadata();
        BiRefNetMaskResult result;
        result.mask = std::move(mask);
        result.requestedBackend = _config.backend;
        result.actualBackend = metadata.backend;
        result.precision = metadata.precision;
        result.usedCuda = true;
        result.engineReused = metadata.engineReused;
        result.deviceLabel = metadata.deviceLabel;
        result.enginePath = metadata.enginePath;
        result.outputName = metadata.outputName;
        result.environmentSummary = metadata.environmentSummary;
        result.modelSha256 = _modelSha256;
        return result;
    }

    BiRefNetMaskGeneratorConfig _config;
    std::unique_ptr<BiRefNetInferenceBackend> _backend;
    std::string _modelSha256;
};

BiRefNetMaskGenerator::BiRefNetMaskGenerator(const BiRefNetMaskGeneratorConfig& config)
    : _impl(std::make_unique<Impl>(config))
{
}

BiRefNetMaskGenerator::~BiRefNetMaskGenerator() = default;
BiRefNetMaskGenerator::BiRefNetMaskGenerator(BiRefNetMaskGenerator&&) noexcept = default;
BiRefNetMaskGenerator& BiRefNetMaskGenerator::operator=(BiRefNetMaskGenerator&&) noexcept = default;

BiRefNetMaskResult BiRefNetMaskGenerator::generate(const cv::Mat& image)
{
    return _impl->generate(image);
}

BiRefNetBackendType BiRefNetMaskGenerator::actualBackend() const
{
    return _impl->metadata().backend;
}

BiRefNetInferencePrecision BiRefNetMaskGenerator::precision() const
{
    return _impl->metadata().precision;
}

std::string BiRefNetMaskGenerator::deviceLabel() const
{
    return _impl->metadata().deviceLabel;
}

bool BiRefNetMaskGenerator::engineReused() const
{
    return _impl->metadata().engineReused;
}

std::string BiRefNetMaskGenerator::enginePath() const
{
    return _impl->metadata().enginePath;
}

std::string BiRefNetMaskGenerator::environmentSummary() const
{
    return _impl->metadata().environmentSummary;
}

std::string BiRefNetMaskGenerator::modelSha256() const
{
    return _impl->modelSha256();
}

} // namespace xjw::mask
