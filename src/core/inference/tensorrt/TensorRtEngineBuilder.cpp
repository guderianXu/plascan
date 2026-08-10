#include "TensorRtEngineBuilder.h"

#include "TensorRtBuildSupport.h"
#include "TensorRtEngineCache.h"
#include "TensorRtNetworkBuilder.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QLockFile>
#include <QSaveFile>

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace xjw::inference
{
    namespace
    {

        std::uint64_t automaticWorkspaceBytes(std::size_t totalMemory)
        {
            constexpr std::uint64_t kMinimum = 512ULL * 1024ULL * 1024ULL;
            constexpr std::uint64_t kMaximum = 4ULL * 1024ULL * 1024ULL * 1024ULL;
            const auto quarter = static_cast<std::uint64_t>(totalMemory / 4U);
            return std::clamp(quarter, kMinimum, kMaximum);
        }

        QStringList normalizedOutputNames(const QStringList& names)
        {
            QStringList result;
            for (const QString& name : names)
            {
                if (!name.trimmed().isEmpty())
                {
                    result.append(name.trimmed());
                }
            }
            result.removeDuplicates();
            result.sort();
            return result;
        }

        bool initializeCudaDevice(int cudaDevice, cudaDeviceProp* properties, QString* errorMessage)
        {
            cudaError_t status = cudaSetDevice(cudaDevice);
            if (status == cudaSuccess)
            {
                status = cudaGetDeviceProperties(properties, cudaDevice);
            }
            if (status == cudaSuccess)
            {
                return true;
            }
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法查询 CUDA 设备 %1：%2")
                                    .arg(cudaDevice)
                                    .arg(QString::fromLatin1(cudaGetErrorString(status)));
            }
            return false;
        }

    } // namespace

    QString tensorRtBuildPrecisionName(TensorRtBuildPrecision precision)
    {
        return precision == TensorRtBuildPrecision::Fp16 ? QStringLiteral("fp16") : QStringLiteral("fp32");
    }

    TensorRtEngineBuildResult ensureTensorRtEngine(const TensorRtEngineBuildRequest& request)
    {
        TensorRtEngineBuildResult result;
        result.precision = request.precision;
        const QFileInfo onnx_info(request.onnxPath);
        if (!onnx_info.isFile())
        {
            result.errorMessage = QStringLiteral("ONNX 模型不存在：%1").arg(request.onnxPath);
            return result;
        }

        const std::vector<TensorRtInputShape> input_shapes = detail::normalizeInputShapes(request);
        result.errorMessage = detail::validateInputShapes(input_shapes);
        if (!result.errorMessage.isEmpty())
        {
            return result;
        }

        const int cuda_device = std::max(0, request.cudaDevice);
        cudaDeviceProp properties{};
        if (!initializeCudaDevice(cuda_device, &properties, &result.errorMessage))
        {
            return result;
        }

        QString hash_error;
        const QString onnx_hash = detail::sha256File(onnx_info.absoluteFilePath(), &hash_error);
        if (onnx_hash.isEmpty())
        {
            result.errorMessage = hash_error;
            return result;
        }

        int cuda_runtime_version = 0;
        int cuda_driver_version = 0;
        cudaRuntimeGetVersion(&cuda_runtime_version);
        cudaDriverGetVersion(&cuda_driver_version);
        const std::uint64_t workspace =
            request.workspaceBytes > 0 ? request.workspaceBytes : automaticWorkspaceBytes(properties.totalGlobalMem);
        const QStringList required_outputs = normalizedOutputNames(request.requiredOutputNames);

        detail::TensorRtCacheIdentityOptions identity_options;
        identity_options.onnxHash = onnx_hash;
        identity_options.tensorRtVersion = detail::compiledTensorRtVersion();
        identity_options.cudaRuntimeVersion = cuda_runtime_version;
        identity_options.cudaDriverVersion = cuda_driver_version;
        identity_options.computeCapability = QStringLiteral("%1.%2").arg(properties.major).arg(properties.minor);
        identity_options.workspaceBytes = workspace;
        identity_options.inputShapes = input_shapes;
        identity_options.requiredOutputNames = required_outputs;
        const QJsonObject identity = detail::makeCacheIdentity(request, identity_options);
        result.cacheFingerprint = detail::fingerprintCacheIdentity(identity);
        result.environmentSummary = QStringLiteral("TensorRT %1，GPU %2（SM %3.%4），%5")
                                        .arg(detail::compiledTensorRtVersion(), QString::fromLocal8Bit(properties.name))
                                        .arg(properties.major)
                                        .arg(properties.minor)
                                        .arg(tensorRtBuildPrecisionName(request.precision).toUpper());

        const QString cache_root = request.cacheDirectory.trimmed().isEmpty()
                                       ? QDir(onnx_info.absolutePath()).filePath(QStringLiteral("engines"))
                                       : QDir::cleanPath(request.cacheDirectory);
        const QString fingerprint_directory = QDir(cache_root).filePath(result.cacheFingerprint.left(20));
        if (!QDir().mkpath(fingerprint_directory))
        {
            result.errorMessage = QStringLiteral("无法创建 TensorRT 缓存目录：%1").arg(fingerprint_directory);
            return result;
        }

        const QString requested_engine_name = request.engineName.trimmed().isEmpty()
                                                  ? onnx_info.completeBaseName() + QStringLiteral(".engine")
                                                  : request.engineName.trimmed();
        const QString engine_name = QFileInfo(requested_engine_name).fileName();
        if (engine_name != requested_engine_name || engine_name.isEmpty())
        {
            result.errorMessage =
                QStringLiteral("TensorRT engineName 必须是不含目录的文件名：%1").arg(requested_engine_name);
            return result;
        }
        result.enginePath = QDir(fingerprint_directory).filePath(engine_name);
        result.metadataPath = result.enginePath + QStringLiteral(".json");

        QLockFile lock(result.enginePath + QStringLiteral(".lock"));
        lock.setStaleLockTime(30 * 60 * 1000);
        if (!lock.tryLock(30 * 60 * 1000))
        {
            result.errorMessage = QStringLiteral("等待 TensorRT engine 构建锁超时：%1").arg(result.enginePath);
            result.enginePath.clear();
            return result;
        }
        if (detail::loadMatchingEngineMetadata(
                result.metadataPath, result.enginePath, result.cacheFingerprint, &result))
        {
            result.reused = true;
            return result;
        }

        detail::TensorRtBuildLogger logger;
        detail::TensorRtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder)
        {
            result.errorMessage = QStringLiteral("无法创建 TensorRT Builder：%1").arg(logger.errors());
            result.enginePath.clear();
            return result;
        }
        if (request.precision == TensorRtBuildPrecision::Fp16 && !builder->platformHasFastFp16())
        {
            result.errorMessage = QStringLiteral("当前 GPU 不支持高效 TensorRT FP16 构建");
            result.enginePath.clear();
            return result;
        }

        const auto flags = 1U << static_cast<unsigned int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        detail::TensorRtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
        detail::TensorRtPtr<nvonnxparser::IParser> parser(network ? nvonnxparser::createParser(*network, logger)
                                                                  : nullptr);
        if (!network || !parser ||
            !parser->parseFromFile(QFile::encodeName(onnx_info.absoluteFilePath()).constData(),
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        {
            result.errorMessage =
                QStringLiteral("TensorRT 解析 ONNX 失败：%1\n%2")
                    .arg(onnx_info.absoluteFilePath(), parser ? detail::parserErrors(*parser) : logger.errors());
            result.enginePath.clear();
            return result;
        }

        detail::TensorRtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config)
        {
            result.errorMessage = QStringLiteral("无法创建 TensorRT BuilderConfig");
            result.enginePath.clear();
            return result;
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace);
        config->setBuilderOptimizationLevel(std::clamp(request.builderOptimizationLevel, 0, 5));
        config->setMaxAuxStreams(std::max(0, request.maximumAuxiliaryStreams));
        if (request.precision == TensorRtBuildPrecision::Fp16)
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        result.errorMessage = detail::configureInputProfile(*builder, *network, *config, input_shapes);
        if (result.errorMessage.isEmpty())
        {
            result.errorMessage = detail::validateRequiredOutputs(*network, required_outputs);
        }
        if (!result.errorMessage.isEmpty())
        {
            result.enginePath.clear();
            return result;
        }

        const auto started = std::chrono::steady_clock::now();
        detail::TensorRtPtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
        if (!serialized)
        {
            result.errorMessage = QStringLiteral("TensorRT 构建 engine 失败：%1").arg(logger.errors());
            result.enginePath.clear();
            return result;
        }
        const double build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        result.ioTensors =
            detail::inspectEngine(serialized->data(), serialized->size(), input_shapes, logger, &result.errorMessage);
        if (result.ioTensors.empty())
        {
            if (result.errorMessage.isEmpty())
            {
                result.errorMessage = QStringLiteral("TensorRT engine 不包含可枚举的 I/O tensor");
            }
            result.enginePath.clear();
            return result;
        }

        QSaveFile engine_file(result.enginePath);
        if (!engine_file.open(QIODevice::WriteOnly) ||
            engine_file.write(static_cast<const char*>(serialized->data()), static_cast<qint64>(serialized->size())) !=
                static_cast<qint64>(serialized->size()) ||
            !engine_file.commit())
        {
            result.errorMessage = QStringLiteral("无法原子写入 TensorRT engine：%1").arg(result.enginePath);
            result.enginePath.clear();
            return result;
        }

        if (!detail::saveEngineMetadata(result.metadataPath,
                                        identity,
                                        result,
                                        onnx_info.fileName(),
                                        QString::fromLocal8Bit(properties.name),
                                        build_seconds,
                                        serialized->size(),
                                        &result.errorMessage))
        {
            QFile::remove(result.enginePath);
            result.enginePath.clear();
        }
        return result;
    }

} // namespace xjw::inference
