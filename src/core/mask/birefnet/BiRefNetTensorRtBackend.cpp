#include "BiRefNetInferenceBackend.h"

#include "inference/tensorrt/TensorRtCapabilities.h"
#include "inference/tensorrt/TensorRtEngineBuilder.h"

#include <QDir>
#include <QJsonObject>
#include <QStandardPaths>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(PLASCAN_HAS_TENSORRT)
#include "inference/tensorrt/TensorRtSession.h"
#endif

namespace xjw::mask
{
namespace
{

std::string toUtf8(const QString& value)
{
    const QByteArray bytes = value.toUtf8();
    return std::string(bytes.constData(), static_cast<std::size_t>(bytes.size()));
}

QString cacheDirectory(const BiRefNetMaskGeneratorConfig& config)
{
    if (!config.engineCacheDirectory.empty())
    {
        return QString::fromUtf8(config.engineCacheDirectory.c_str());
    }

    const QString writableRoot = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    if (writableRoot.isEmpty())
    {
        throw std::runtime_error(
            "Cannot resolve a writable per-user directory for the BiRefNet TensorRT engine cache.");
    }
    return QDir(writableRoot).filePath(QStringLiteral("models/birefnet_dynamic/engines"));
}

void reportStatus(const BiRefNetMaskGeneratorConfig& config, const std::string& message)
{
    if (config.statusCallback)
    {
        config.statusCallback(message);
    }
}

#if defined(PLASCAN_HAS_TENSORRT)

using inference::TensorRtBuildPrecision;
using inference::TensorRtEngineBuildRequest;
using inference::TensorRtEngineBuildResult;
using inference::TensorRtHostBinding;
using inference::TensorRtInputShape;
using inference::TensorRtSession;
using inference::TensorRtTensorDataType;
using inference::TensorRtTensorInfo;
using inference::TensorRtTensorMode;

const TensorRtTensorInfo& requiredTensor(const std::vector<TensorRtTensorInfo>& tensors,
                                         TensorRtTensorMode mode,
                                         const QString& name)
{
    const auto found = std::find_if(tensors.begin(),
                                    tensors.end(),
                                    [mode, &name](const auto& tensor)
                                    { return tensor.mode == mode && tensor.name == name; });
    if (found == tensors.end())
    {
        throw std::runtime_error("BiRefNet TensorRT engine is missing required tensor: " + toUtf8(name));
    }
    return *found;
}

TensorRtEngineBuildRequest buildRequest(const BiRefNetMaskGeneratorConfig& config,
                                        TensorRtBuildPrecision precision)
{
    TensorRtEngineBuildRequest request;
    request.onnxPath = QString::fromUtf8(config.modelPath.c_str());
    request.cacheDirectory = cacheDirectory(config);
    request.engineName = QStringLiteral("birefnet_dynamic_%1_%2.engine")
                             .arg(config.inputSize)
                             .arg(inference::tensorRtBuildPrecisionName(precision));
    request.precision = precision;
    request.cudaDevice = config.cudaDevice;
    request.workspaceBytes = config.tensorRtWorkspaceBytes;
    request.builderOptimizationLevel = config.tensorRtBuilderOptimizationLevel;
    request.maximumAuxiliaryStreams = config.tensorRtMaximumAuxiliaryStreams;
    request.inputShapes.push_back(
        TensorRtInputShape{QStringLiteral("input_image"), {1, 3, config.inputSize, config.inputSize}});
    request.requiredOutputNames.append(QStringLiteral("output_image"));
    request.fingerprintAttributes =
        QJsonObject{{QStringLiteral("consumer"), QStringLiteral("birefnet_dynamic_mask")},
                    {QStringLiteral("contract_version"), 1},
                    {QStringLiteral("input_size"), config.inputSize},
                    {QStringLiteral("preprocessing"), QStringLiteral("imagenet_letterbox")},
                    {QStringLiteral("output_transform"), QStringLiteral("sigmoid")}};
    return request;
}

TensorRtEngineBuildResult buildEngine(const BiRefNetMaskGeneratorConfig& config,
                                      const inference::TensorRtCapabilities& capabilities)
{
    std::string fp16Error;
    if (config.preferFp16 && capabilities.fastFp16)
    {
        reportStatus(config,
                     "正在检查 BiRefNet Dynamic TensorRT FP16 engine 缓存；"
                     "首次运行需要为本机 GPU 构建 engine。");
        TensorRtEngineBuildResult fp16 =
            inference::ensureTensorRtEngine(buildRequest(config, TensorRtBuildPrecision::Fp16));
        if (fp16.isValid())
        {
            return fp16;
        }
        fp16Error = toUtf8(fp16.errorMessage);
        reportStatus(config, "BiRefNet FP16 构建不可用，正在尝试 FP32：" + fp16Error);
    }

    reportStatus(config,
                 "正在检查 BiRefNet Dynamic TensorRT FP32 engine 缓存；"
                 "首次运行需要为本机 GPU 构建 engine。");
    TensorRtEngineBuildResult fp32 =
        inference::ensureTensorRtEngine(buildRequest(config, TensorRtBuildPrecision::Fp32));
    if (!fp32.isValid())
    {
        std::string error = toUtf8(fp32.errorMessage);
        if (!fp16Error.empty())
        {
            error = "FP16: " + fp16Error + "; FP32: " + error;
        }
        throw std::runtime_error("Failed to prepare the BiRefNet TensorRT engine: " + error);
    }
    return fp32;
}

void validateShape(const TensorRtTensorInfo& tensor,
                   const std::vector<std::int64_t>& expected,
                   const char* role)
{
    if (tensor.dimensions != expected)
    {
        throw std::runtime_error(std::string("BiRefNet TensorRT ") + role +
                                 " tensor has an unexpected shape: " + toUtf8(tensor.name));
    }
    if (tensor.dataType != TensorRtTensorDataType::Float32)
    {
        throw std::runtime_error(std::string("BiRefNet TensorRT ") + role +
                                 " tensor must expose float32 host I/O: " + toUtf8(tensor.name));
    }
}

class BiRefNetTensorRtBackend final : public BiRefNetInferenceBackend
{
public:
    BiRefNetTensorRtBackend(const BiRefNetMaskGeneratorConfig& config,
                            const inference::TensorRtCapabilities& capabilities)
        : _inputSize(config.inputSize), _cudaDevice(config.cudaDevice)
    {
        const TensorRtEngineBuildResult build = buildEngine(config, capabilities);
        _session = std::make_unique<TensorRtSession>(build.enginePath, _cudaDevice);

        const TensorRtTensorInfo& input = requiredTensor(
            _session->tensors(), TensorRtTensorMode::Input, QStringLiteral("input_image"));
        const TensorRtTensorInfo& output = requiredTensor(
            _session->tensors(), TensorRtTensorMode::Output, QStringLiteral("output_image"));
        validateShape(input, {1, 3, _inputSize, _inputSize}, "input");
        validateShape(output, {1, 1, _inputSize, _inputSize}, "output");
        _inputName = toUtf8(input.name);
        _outputName = toUtf8(output.name);
        _output.resize(static_cast<std::size_t>(output.elementCount()));

        _metadata.backend = BiRefNetBackendType::TensorRt;
        _metadata.precision = build.precision == TensorRtBuildPrecision::Fp16
                                  ? BiRefNetInferencePrecision::Fp16
                                  : BiRefNetInferencePrecision::Fp32;
        _metadata.deviceLabel =
            "TensorRT GPU (" + biRefNetInferencePrecisionToken(_metadata.precision) + ")";
        _metadata.engineReused = build.reused;
        _metadata.enginePath = toUtf8(build.enginePath);
        _metadata.outputName = _outputName;
        _metadata.environmentSummary = toUtf8(build.environmentSummary) + "; output=" + _outputName;

        reportStatus(config,
                     build.reused ? "正在复用 BiRefNet TensorRT engine 缓存：" + _metadata.enginePath
                                  : "已为本机 GPU 构建 BiRefNet TensorRT engine：" + _metadata.enginePath);
    }

    cv::Mat forward(const cv::Mat& inputBlob) override
    {
        if (inputBlob.type() != CV_32F || inputBlob.dims != 4 || inputBlob.size[0] != 1 ||
            inputBlob.size[1] != 3 || inputBlob.size[2] != _inputSize || inputBlob.size[3] != _inputSize)
        {
            throw std::invalid_argument("BiRefNet TensorRT input must be float32 NCHW [1,3,1024,1024].");
        }

        const cv::Mat contiguous = inputBlob.isContinuous() ? inputBlob : inputBlob.clone();
        const std::uint64_t inputBytes = contiguous.total() * contiguous.elemSize();
        const std::uint64_t outputBytes = _output.size() * sizeof(float);
        if (inputBytes > std::numeric_limits<std::size_t>::max() ||
            outputBytes > std::numeric_limits<std::size_t>::max())
        {
            throw std::overflow_error("BiRefNet TensorRT tensor size exceeds the host address space.");
        }

        std::vector<TensorRtHostBinding> bindings{
            {_inputName.c_str(),
             const_cast<float*>(contiguous.ptr<float>()),
             static_cast<std::size_t>(inputBytes),
             true},
            {_outputName.c_str(), _output.data(), static_cast<std::size_t>(outputBytes), false}};
        _session->execute(bindings);

        const int shape[] = {1, 1, _inputSize, _inputSize};
        return cv::Mat(4, shape, CV_32F, _output.data()).clone();
    }

    const BiRefNetBackendMetadata& metadata() const override
    {
        return _metadata;
    }

private:
    int _inputSize = kBiRefNetDynamicInputSize;
    int _cudaDevice = 0;
    std::string _inputName;
    std::string _outputName;
    std::vector<float> _output;
    std::unique_ptr<TensorRtSession> _session;
    BiRefNetBackendMetadata _metadata;
};

#endif

} // namespace

std::unique_ptr<BiRefNetInferenceBackend>
createBiRefNetTensorRtBackend(const BiRefNetMaskGeneratorConfig& config)
{
    if (config.backend != BiRefNetBackendType::Auto && config.backend != BiRefNetBackendType::TensorRt)
    {
        throw std::invalid_argument("BiRefNet Dynamic supports only the TensorRT backend.");
    }

    const inference::TensorRtCapabilities capabilities = inference::queryTensorRtCapabilities(config.cudaDevice);
    if (!capabilities.isAvailable())
    {
        const std::string reason =
            capabilities.errorMessage.isEmpty()
                ? "TensorRT/CUDA is not available in this PlaScan build or on the selected device."
                : toUtf8(capabilities.errorMessage);
        throw std::runtime_error("BiRefNet Dynamic requires TensorRT/CUDA: " + reason);
    }

#if defined(PLASCAN_HAS_TENSORRT)
    return std::make_unique<BiRefNetTensorRtBackend>(config, capabilities);
#else
    throw std::runtime_error("PlaScan was built without TensorRT support required by BiRefNet Dynamic.");
#endif
}

} // namespace xjw::mask
