#include "TensorRtEngineBuilder.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLockFile>
#include <QSaveFile>
#include <QStringList>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>

namespace xjw::image_matching
{
namespace
{

class TensorRtBuildLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *message) noexcept override
    {
        if (severity <= Severity::kERROR && message)
        {
            try
            {
                if (!_errors.isEmpty())
                {
                    _errors += QLatin1Char('\n');
                }
                _errors += QString::fromUtf8(message);
            }
            catch (...)
            {
            }
        }
    }

    QString errors() const { return _errors; }

private:
    QString _errors;
};

template<typename T>
struct TensorRtDeleter
{
    void operator()(T *value) const noexcept { delete value; }
};

template<typename T>
using TensorRtPtr = std::unique_ptr<T, TensorRtDeleter<T>>;

QString sha256File(const QString &path, QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取 ONNX：%1").arg(path);
        }
        return QString();
    }

    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        const QByteArray block = file.read(4 * 1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("计算 ONNX SHA-256 失败：%1").arg(path);
            }
            return QString();
        }
        hash.addData(block);
    }
    return QString::fromLatin1(hash.result().toHex());
}

QString tensorRtVersion()
{
    return QStringLiteral("%1.%2.%3.%4")
        .arg(NV_TENSORRT_MAJOR)
        .arg(NV_TENSORRT_MINOR)
        .arg(NV_TENSORRT_PATCH)
        .arg(NV_TENSORRT_BUILD);
}

std::uint64_t automaticWorkspaceBytes(std::size_t totalMemory)
{
    constexpr std::uint64_t kMinimum = 512ULL * 1024ULL * 1024ULL;
    constexpr std::uint64_t kMaximum = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    const auto quarter = static_cast<std::uint64_t>(totalMemory / 4U);
    return std::clamp(quarter, kMinimum, kMaximum);
}

bool metadataMatches(const QString &metadataPath,
                     const QString &enginePath,
                     const QString &fingerprint)
{
    QFile file(metadataPath);
    if (!QFileInfo::exists(enginePath) || !file.open(QIODevice::ReadOnly))
    {
        return false;
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    return document.isObject() &&
        document.object().value(QStringLiteral("cache_fingerprint")).toString() == fingerprint;
}

QString parserErrors(nvonnxparser::IParser &parser)
{
    QStringList errors;
    for (int index = 0; index < parser.getNbErrors(); ++index)
    {
        const nvonnxparser::IParserError *error = parser.getError(index);
        if (error)
        {
            errors.append(QString::fromUtf8(error->desc()));
        }
    }
    return errors.join(QLatin1Char('\n'));
}

} // namespace

TensorRtEngineBuildResult ensureTensorRtEngine(
    const TensorRtEngineBuildRequest &request)
{
    TensorRtEngineBuildResult result;
    const QFileInfo onnxInfo(request.onnxPath);
    if (!onnxInfo.isFile())
    {
        result.errorMessage = QStringLiteral("ONNX 模型不存在：%1").arg(request.onnxPath);
        return result;
    }

    const int cudaDevice = std::max(0, request.cudaDevice);
    cudaDeviceProp properties{};
    cudaError_t cudaStatus = cudaSetDevice(cudaDevice);
    if (cudaStatus == cudaSuccess)
    {
        cudaStatus = cudaGetDeviceProperties(&properties, cudaDevice);
    }
    if (cudaStatus != cudaSuccess)
    {
        result.errorMessage = QStringLiteral("无法查询 CUDA 设备 %1：%2")
            .arg(cudaDevice)
            .arg(QString::fromLatin1(cudaGetErrorString(cudaStatus)));
        return result;
    }

    QString hashError;
    const QString onnxHash = sha256File(onnxInfo.absoluteFilePath(), &hashError);
    if (onnxHash.isEmpty())
    {
        result.errorMessage = hashError;
        return result;
    }

    const QString precision = request.precision == TensorRtBuildPrecision::Fp16
        ? QStringLiteral("fp16")
        : QStringLiteral("fp32");
    const std::uint64_t workspace = request.workspaceBytes > 0
        ? request.workspaceBytes
        : automaticWorkspaceBytes(properties.totalGlobalMem);
    const QJsonObject identity{
        {QStringLiteral("schema"), 1},
        {QStringLiteral("onnx_sha256"), onnxHash},
        {QStringLiteral("tensorrt"), tensorRtVersion()},
        {QStringLiteral("compute_capability"),
         QStringLiteral("%1.%2").arg(properties.major).arg(properties.minor)},
        {QStringLiteral("precision"), precision},
        {QStringLiteral("workspace_bytes"), static_cast<double>(workspace)},
        {QStringLiteral("builder_optimization_level"), request.builderOptimizationLevel},
        {QStringLiteral("maximum_auxiliary_streams"), request.maximumAuxiliaryStreams}};
    QJsonObject fingerprintIdentity = identity;
    fingerprintIdentity[QStringLiteral("fixed_keypoint_count")] = request.fixedKeypointCount;
    result.cacheFingerprint = QString::fromLatin1(
        QCryptographicHash::hash(QJsonDocument(fingerprintIdentity).toJson(QJsonDocument::Compact),
                                 QCryptographicHash::Sha256)
            .toHex());
    result.environmentSummary = QStringLiteral("TensorRT %1，GPU %2（SM %3.%4），%5")
        .arg(tensorRtVersion(), QString::fromLocal8Bit(properties.name))
        .arg(properties.major)
        .arg(properties.minor)
        .arg(precision.toUpper());

    const QString cacheRoot = request.cacheDirectory.trimmed().isEmpty()
        ? QDir(onnxInfo.absolutePath()).filePath(QStringLiteral("engines"))
        : QDir::cleanPath(request.cacheDirectory);
    const QString fingerprintDirectory = QDir(cacheRoot).filePath(
        result.cacheFingerprint.left(20));
    if (!QDir().mkpath(fingerprintDirectory))
    {
        result.errorMessage = QStringLiteral("无法创建 TensorRT 缓存目录：%1")
            .arg(fingerprintDirectory);
        return result;
    }

    const QString engineName = request.engineName.trimmed().isEmpty()
        ? onnxInfo.completeBaseName() + QStringLiteral(".engine")
        : request.engineName;
    result.enginePath = QDir(fingerprintDirectory).filePath(engineName);
    result.metadataPath = result.enginePath + QStringLiteral(".json");

    QLockFile lock(result.enginePath + QStringLiteral(".lock"));
    lock.setStaleLockTime(30 * 60 * 1000);
    if (!lock.tryLock(30 * 60 * 1000))
    {
        result.errorMessage = QStringLiteral("等待 TensorRT engine 构建锁超时：%1")
            .arg(result.enginePath);
        result.enginePath.clear();
        return result;
    }
    if (metadataMatches(result.metadataPath, result.enginePath, result.cacheFingerprint))
    {
        result.reused = true;
        return result;
    }

    TensorRtBuildLogger logger;
    TensorRtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
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

    const auto flags = 1U << static_cast<unsigned int>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TensorRtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    TensorRtPtr<nvonnxparser::IParser> parser(
        network ? nvonnxparser::createParser(*network, logger) : nullptr);
    if (!network || !parser ||
        !parser->parseFromFile(QFile::encodeName(onnxInfo.absoluteFilePath()).constData(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        result.errorMessage = QStringLiteral("TensorRT 解析 ONNX 失败：%1\n%2")
            .arg(onnxInfo.absoluteFilePath(), parser ? parserErrors(*parser) : logger.errors());
        result.enginePath.clear();
        return result;
    }

    TensorRtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config)
    {
        result.errorMessage = QStringLiteral("无法创建 TensorRT BuilderConfig");
        result.enginePath.clear();
        return result;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace);
    config->setBuilderOptimizationLevel(
        std::clamp(request.builderOptimizationLevel, 0, 5));
    config->setMaxAuxStreams(std::max(0, request.maximumAuxiliaryStreams));
    if (request.precision == TensorRtBuildPrecision::Fp16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    nvinfer1::IOptimizationProfile *optimizationProfile = nullptr;
    if (request.fixedKeypointCount > 0)
    {
        optimizationProfile = builder->createOptimizationProfile();
        if (!optimizationProfile)
        {
            result.errorMessage = QStringLiteral("无法创建 TensorRT 动态关键点 profile");
            result.enginePath.clear();
            return result;
        }
        const int count = request.fixedKeypointCount;
        const auto setShape = [&](const char *name, const nvinfer1::Dims &shape)
        {
            return optimizationProfile->setDimensions(
                       name, nvinfer1::OptProfileSelector::kMIN, shape) &&
                optimizationProfile->setDimensions(
                    name, nvinfer1::OptProfileSelector::kOPT, shape) &&
                optimizationProfile->setDimensions(
                    name, nvinfer1::OptProfileSelector::kMAX, shape);
        };
        const nvinfer1::Dims3 keypoints{1, count, 2};
        const nvinfer1::Dims3 descriptors{1, count, 256};
        const nvinfer1::Dims2 valid{1, count};
        const bool profileValid =
            setShape("keypoints0", keypoints) && setShape("keypoints1", keypoints) &&
            setShape("descriptors0", descriptors) && setShape("descriptors1", descriptors) &&
            setShape("valid0", valid) && setShape("valid1", valid);
        if (!profileValid || config->addOptimizationProfile(optimizationProfile) < 0)
        {
            result.errorMessage = QStringLiteral("LoMa-R 动态关键点 profile 与 ONNX 输入不兼容");
            result.enginePath.clear();
            return result;
        }
    }

    const auto started = std::chrono::steady_clock::now();
    TensorRtPtr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized)
    {
        result.errorMessage = QStringLiteral("TensorRT 构建 engine 失败：%1").arg(logger.errors());
        result.enginePath.clear();
        return result;
    }
    const double buildSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    QSaveFile engineFile(result.enginePath);
    if (!engineFile.open(QIODevice::WriteOnly) ||
        engineFile.write(static_cast<const char *>(serialized->data()),
                         static_cast<qint64>(serialized->size())) !=
            static_cast<qint64>(serialized->size()) ||
        !engineFile.commit())
    {
        result.errorMessage = QStringLiteral("无法原子写入 TensorRT engine：%1")
            .arg(result.enginePath);
        result.enginePath.clear();
        return result;
    }

    QJsonObject metadata = fingerprintIdentity;
    metadata[QStringLiteral("cache_fingerprint")] = result.cacheFingerprint;
    metadata[QStringLiteral("onnx_file")] = onnxInfo.fileName();
    metadata[QStringLiteral("engine_file")] = QFileInfo(result.enginePath).fileName();
    metadata[QStringLiteral("gpu_name")] = QString::fromLocal8Bit(properties.name);
    metadata[QStringLiteral("build_seconds")] = buildSeconds;
    metadata[QStringLiteral("engine_bytes")] = static_cast<double>(serialized->size());
    QSaveFile metadataFile(result.metadataPath);
    if (!metadataFile.open(QIODevice::WriteOnly) ||
        metadataFile.write(QJsonDocument(metadata).toJson(QJsonDocument::Indented)) < 0 ||
        !metadataFile.commit())
    {
        QFile::remove(result.enginePath);
        result.errorMessage = QStringLiteral("无法写入 TensorRT engine 元数据：%1")
            .arg(result.metadataPath);
        result.enginePath.clear();
        return result;
    }
    return result;
}

} // namespace xjw::image_matching
