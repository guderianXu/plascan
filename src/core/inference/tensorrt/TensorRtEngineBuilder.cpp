#include "TensorRtEngineBuilder.h"

#include "TensorRtBuildSupport.h"
#include "TensorRtEngineCache.h"
#include "TensorRtNetworkBuilder.h"
#include "log/Logger.h"

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
#include <cmath>
#include <cstdint>
#include <thread>

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

        void reportBuildProgress(const TensorRtEngineBuildRequest& request,
                                 const QString& message,
                                 int current,
                                 int maximum) noexcept
        {
            if (!request.progressCallback)
            {
                return;
            }
            try
            {
                request.progressCallback({message, current, maximum});
            }
            catch (...)
            {
                LOG_WARN(QStringLiteral("TensorRT engine 进度回调抛出异常，已忽略"));
            }
        }

        QString elapsedBuildText(qint64 seconds)
        {
            if (seconds < 60)
            {
                return QStringLiteral("%1 秒").arg(seconds);
            }
            return QStringLiteral("%1 分 %2 秒").arg(seconds / 60).arg(seconds % 60);
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
        reportBuildProgress(request, QStringLiteral("检查 ONNX 文件和 CUDA 环境"), 0, 6);
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

        reportBuildProgress(
            request,
            QStringLiteral("计算 ONNX SHA-256（%1 MB）")
                .arg(onnx_info.size() / (1024 * 1024)),
            1,
            6);
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

        reportBuildProgress(request, QStringLiteral("检查本机 engine 缓存和构建锁"), 2, 6);
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
            result.cacheDecision = QStringLiteral("命中缓存：%1（指纹 %2）")
                                       .arg(engine_name, result.cacheFingerprint.left(12));
            LOG_INFO(QStringLiteral("TensorRT engine %1；路径=%2")
                         .arg(result.cacheDecision, result.enginePath));
            reportBuildProgress(request, result.cacheDecision, 6, 6);
            return result;
        }

        result.cacheDecision = detail::describeEngineCacheMiss(
            cache_root,
            engine_name,
            result.metadataPath,
            result.enginePath,
            identity);
        LOG_INFO(QStringLiteral("TensorRT engine 需要重新生成：%1；目标=%2")
                     .arg(result.cacheDecision, result.enginePath));
        reportBuildProgress(
            request,
            QStringLiteral("需要重新生成：%1").arg(result.cacheDecision),
            2,
            6);

        reportBuildProgress(request, QStringLiteral("解析 ONNX 并配置 TensorRT 网络"), 3, 6);
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
        const QString building_message = QStringLiteral(
            "TensorRT 正在搜索并编译最优内核（不提供可验证百分比）");
        reportBuildProgress(request, building_message, 0, 0);
        LOG_INFO(QStringLiteral("TensorRT engine 开始构建：%1").arg(result.enginePath));
        std::jthread heartbeat;
        if (request.progressCallback)
        {
            const TensorRtEngineBuildProgressCallback callback = request.progressCallback;
            const QString heartbeat_engine_name = engine_name;
            heartbeat = std::jthread(
                [callback, started, building_message, heartbeat_engine_name](std::stop_token stop_token)
                {
                    while (!stop_token.stop_requested())
                    {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        if (stop_token.stop_requested())
                        {
                            break;
                        }
                        const qint64 elapsed_seconds =
                            std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
                        try
                        {
                            callback({QStringLiteral("%1，已耗时 %2")
                                          .arg(building_message, elapsedBuildText(elapsed_seconds)),
                                      0,
                                      0});
                        }
                        catch (...)
                        {
                        }
                        if (elapsed_seconds > 0 && elapsed_seconds % 30 == 0)
                        {
                            LOG_INFO(QStringLiteral("TensorRT engine 构建中：%1，已耗时 %2")
                                         .arg(heartbeat_engine_name,
                                              elapsedBuildText(elapsed_seconds)));
                        }
                    }
                });
        }
        detail::TensorRtPtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
        heartbeat.request_stop();
        if (heartbeat.joinable())
        {
            heartbeat.join();
        }
        if (!serialized)
        {
            result.errorMessage = QStringLiteral("TensorRT 构建 engine 失败：%1").arg(logger.errors());
            result.enginePath.clear();
            return result;
        }
        const double build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        reportBuildProgress(request, QStringLiteral("校验生成的 engine I/O"), 4, 6);
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

        reportBuildProgress(request, QStringLiteral("写入 engine 和完整性元数据"), 5, 6);
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
        else
        {
            const QString completed = QStringLiteral("engine 已生成并缓存，耗时 %1")
                                          .arg(elapsedBuildText(
                                              static_cast<qint64>(std::llround(build_seconds))));
            LOG_INFO(QStringLiteral("TensorRT %1：%2").arg(completed, result.enginePath));
            reportBuildProgress(request, completed, 6, 6);
        }
        return result;
    }

} // namespace xjw::inference
