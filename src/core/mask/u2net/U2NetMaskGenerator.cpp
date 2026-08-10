#include "U2NetMaskGenerator.h"

#include "U2NetInferenceBackend.h"
#include "U2NetImageProcessing.h"
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
                throw std::runtime_error("Cannot read the U2Net ONNX model for SHA-256: " + path);
            }

            QCryptographicHash hash(QCryptographicHash::Sha256);
            while (!file.atEnd())
            {
                const QByteArray block = file.read(4 * 1024 * 1024);
                if (block.isEmpty() && file.error() != QFileDevice::NoError)
                {
                    throw std::runtime_error("Failed to calculate the U2Net ONNX SHA-256: " + path);
                }
                hash.addData(block);
            }
            return hash.result().toHex().toStdString();
        }

    } // namespace

    std::string u2netDefaultModelFileName()
    {
        return "U2Net_v1.onnx";
    }

    std::string u2netBackendTypeToken(U2NetBackendType backend)
    {
        switch (backend)
        {
        case U2NetBackendType::Auto:
            return "auto";
        case U2NetBackendType::TensorRt:
            return "tensorrt";
        case U2NetBackendType::OpenCvCpu:
            return "opencv_cpu";
        }
        return "auto";
    }

    std::string u2netBackendTypeLabel(U2NetBackendType backend)
    {
        switch (backend)
        {
        case U2NetBackendType::Auto:
            return "Auto";
        case U2NetBackendType::TensorRt:
            return "TensorRT GPU";
        case U2NetBackendType::OpenCvCpu:
            return "OpenCV CPU";
        }
        return "Auto";
    }

    std::optional<U2NetBackendType> parseU2NetBackendType(const std::string& token)
    {
        const std::string normalized = normalizedToken(token);
        if (normalized == "auto")
        {
            return U2NetBackendType::Auto;
        }
        if (normalized == "tensorrt" || normalized == "cuda" || normalized == "gpu")
        {
            return U2NetBackendType::TensorRt;
        }
        if (normalized == "opencvcpu" || normalized == "cpu")
        {
            return U2NetBackendType::OpenCvCpu;
        }
        return std::nullopt;
    }

    std::string u2netInferencePrecisionToken(U2NetInferencePrecision precision)
    {
        switch (precision)
        {
        case U2NetInferencePrecision::Fp16:
            return "fp16";
        case U2NetInferencePrecision::Fp32:
            return "fp32";
        case U2NetInferencePrecision::Unknown:
            return "unknown";
        }
        return "unknown";
    }

    U2NetInferenceCapabilities detectU2NetInferenceCapabilities(int cudaDevice)
    {
        const inference::TensorRtCapabilities detected = inference::queryTensorRtCapabilities(cudaDevice);
        U2NetInferenceCapabilities result;
        result.tensorRtCompiled = detected.compiled;
        result.tensorRtAvailable = detected.isAvailable();
        result.hasCudaDevice = detected.cudaAvailable;
        result.cudaDeviceCount = detected.deviceCount;
        result.supportsFp16 = detected.fastFp16;
        result.tensorRtVersion = toUtf8(detected.tensorRtVersion);
        result.gpuName = toUtf8(detected.gpuName);
        result.errorMessage = toUtf8(detected.errorMessage);

        std::ostringstream summary;
        summary << "OpenCV DNN CPU available; TensorRT compiled=" << (result.tensorRtCompiled ? "yes" : "no")
                << "; TensorRT backend=" << (result.tensorRtAvailable ? "available" : "unavailable")
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

    class U2NetMaskGenerator::Impl
    {
    public:
        explicit Impl(U2NetMaskGeneratorConfig config) : _config(std::move(config))
        {
            if (_config.modelPath.empty())
            {
                throw std::runtime_error("U2Net ONNX model path is empty.");
            }
            if (!std::filesystem::exists(std::filesystem::u8path(_config.modelPath)))
            {
                throw std::runtime_error("U2Net ONNX model does not exist: " + _config.modelPath);
            }
            _modelSha256 = sha256File(_config.modelPath);
            if (_config.inputSize != kU2NetModelInputSize)
            {
                throw std::invalid_argument("The bundled U2Net ONNX model requires a fixed 320x320 input.");
            }
            initializeBackend();
        }

        U2NetMaskResult generate(const cv::Mat& image)
        {
            if (image.empty())
            {
                return makeResult({});
            }

            const cv::Mat blob = makeU2NetBlob(image, _config.inputSize);
            cv::Mat probability;
            try
            {
                probability = inferProbability(blob);
            }
            catch (const std::exception& error)
            {
                if (_backend->metadata().backend != U2NetBackendType::TensorRt || !canFallback())
                {
                    throw;
                }
                activateCpuBackend(error.what());
                probability = inferProbability(blob);
            }

            cv::Mat mask = makeU2NetMask(probability,
                                         image.size(),
                                         _config.foregroundThreshold,
                                         _config.morphologyRadius,
                                         _config.minComponentArea,
                                         _config.keepLargestComponent);
            return makeResult(std::move(mask));
        }

        const U2NetBackendMetadata& metadata() const
        {
            return _backend->metadata();
        }
        const std::string& fallbackReason() const
        {
            return _fallbackReason;
        }
        const std::string& modelSha256() const
        {
            return _modelSha256;
        }

    private:
        void initializeBackend()
        {
            if (_config.backend == U2NetBackendType::OpenCvCpu)
            {
                _backend = createU2NetOpenCvCpuBackend(_config);
                return;
            }

            try
            {
                _backend = createU2NetTensorRtBackend(_config);
            }
            catch (const std::exception& error)
            {
                if (!canFallback())
                {
                    throw std::runtime_error(std::string("U2Net TensorRT backend initialization failed: ") +
                                             error.what());
                }
                activateCpuBackend(error.what());
            }
        }

        bool canFallback() const
        {
            return _config.backend == U2NetBackendType::Auto || _config.allowDeviceFallback;
        }

        void activateCpuBackend(const std::string& reason)
        {
            _fallbackReason = reason;
            if (_config.statusCallback)
            {
                _config.statusCallback("U2Net TensorRT 不可用，正在回退到 OpenCV CPU：" + reason);
            }
            try
            {
                _backend = createU2NetOpenCvCpuBackend(_config);
            }
            catch (const std::exception& cpuError)
            {
                throw std::runtime_error("U2Net TensorRT failed (" + reason +
                                         "); OpenCV CPU fallback also failed: " + cpuError.what());
            }
        }

        cv::Mat inferProbability(const cv::Mat& blob)
        {
            return u2netProbabilityFromOutput(_backend->forward(blob));
        }

        U2NetMaskResult makeResult(cv::Mat mask) const
        {
            const U2NetBackendMetadata& metadata = _backend->metadata();
            U2NetMaskResult result;
            result.mask = std::move(mask);
            result.requestedBackend = _config.backend;
            result.actualBackend = metadata.backend;
            result.precision = metadata.precision;
            result.usedCuda = metadata.backend == U2NetBackendType::TensorRt;
            result.deviceFallback = !_fallbackReason.empty();
            result.engineReused = metadata.engineReused;
            result.deviceLabel = metadata.deviceLabel;
            result.fallbackReason = _fallbackReason;
            result.enginePath = metadata.enginePath;
            result.fusedOutputName = metadata.fusedOutputName;
            result.environmentSummary = metadata.environmentSummary;
            result.modelSha256 = _modelSha256;
            return result;
        }

        U2NetMaskGeneratorConfig _config;
        std::unique_ptr<U2NetInferenceBackend> _backend;
        std::string _fallbackReason;
        std::string _modelSha256;
    };

    U2NetMaskGenerator::U2NetMaskGenerator(const U2NetMaskGeneratorConfig& config)
        : _impl(std::make_unique<Impl>(config))
    {
    }

    U2NetMaskGenerator::~U2NetMaskGenerator() = default;
    U2NetMaskGenerator::U2NetMaskGenerator(U2NetMaskGenerator&&) noexcept = default;
    U2NetMaskGenerator& U2NetMaskGenerator::operator=(U2NetMaskGenerator&&) noexcept = default;

    U2NetMaskResult U2NetMaskGenerator::generate(const cv::Mat& image)
    {
        return _impl->generate(image);
    }

    bool U2NetMaskGenerator::usedCuda() const
    {
        return actualBackend() == U2NetBackendType::TensorRt;
    }

    U2NetBackendType U2NetMaskGenerator::actualBackend() const
    {
        return _impl->metadata().backend;
    }

    U2NetInferencePrecision U2NetMaskGenerator::precision() const
    {
        return _impl->metadata().precision;
    }

    std::string U2NetMaskGenerator::deviceLabel() const
    {
        return _impl->metadata().deviceLabel;
    }

    bool U2NetMaskGenerator::engineReused() const
    {
        return _impl->metadata().engineReused;
    }

    std::string U2NetMaskGenerator::enginePath() const
    {
        return _impl->metadata().enginePath;
    }

    std::string U2NetMaskGenerator::environmentSummary() const
    {
        return _impl->metadata().environmentSummary;
    }

    std::string U2NetMaskGenerator::modelSha256() const
    {
        return _impl->modelSha256();
    }

    std::string U2NetMaskGenerator::fallbackReason() const
    {
        return _impl->fallbackReason();
    }

} // namespace xjw::mask
