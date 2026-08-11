#include "TensorRtEngineCache.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

namespace xjw::inference::detail
{
    namespace
    {

        QJsonArray dimensionsToJson(const std::vector<std::int64_t>& dimensions)
        {
            QJsonArray result;
            for (const std::int64_t dimension : dimensions)
            {
                result.append(static_cast<double>(dimension));
            }
            return result;
        }

        std::vector<std::int64_t> dimensionsFromJson(const QJsonArray& array)
        {
            std::vector<std::int64_t> result;
            result.reserve(static_cast<std::size_t>(array.size()));
            for (const QJsonValue& value : array)
            {
                result.push_back(static_cast<std::int64_t>(value.toDouble(-1.0)));
            }
            return result;
        }

        QJsonArray inputShapesToJson(const std::vector<TensorRtInputShape>& inputShapes)
        {
            QJsonArray result;
            for (const TensorRtInputShape& input : inputShapes)
            {
                result.append(QJsonObject{{QStringLiteral("name"), input.name},
                                          {QStringLiteral("dimensions"), dimensionsToJson(input.dimensions)}});
            }
            return result;
        }

        QJsonArray tensorInfoToJson(const std::vector<TensorRtTensorInfo>& tensors)
        {
            QJsonArray result;
            for (const TensorRtTensorInfo& tensor : tensors)
            {
                result.append(QJsonObject{{QStringLiteral("name"), tensor.name},
                                          {QStringLiteral("mode"), tensorRtTensorModeName(tensor.mode)},
                                          {QStringLiteral("data_type"), tensorRtTensorDataTypeName(tensor.dataType)},
                                          {QStringLiteral("dimensions"), dimensionsToJson(tensor.dimensions)}});
            }
            return result;
        }

        std::vector<TensorRtTensorInfo> tensorInfoFromJson(const QJsonArray& array)
        {
            std::vector<TensorRtTensorInfo> result;
            result.reserve(static_cast<std::size_t>(array.size()));
            for (const QJsonValue& value : array)
            {
                const QJsonObject object = value.toObject();
                TensorRtTensorInfo tensor;
                tensor.name = object.value(QStringLiteral("name")).toString();
                tensor.mode = tensorRtTensorModeFromName(object.value(QStringLiteral("mode")).toString());
                tensor.dataType = tensorRtTensorDataTypeFromName(object.value(QStringLiteral("data_type")).toString());
                tensor.dimensions = dimensionsFromJson(object.value(QStringLiteral("dimensions")).toArray());
                if (!tensor.name.isEmpty())
                {
                    result.push_back(std::move(tensor));
                }
            }
            return result;
        }

        QJsonObject readJsonObject(const QString& path)
        {
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly))
            {
                return {};
            }
            const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
            return document.isObject() ? document.object() : QJsonObject();
        }

        QString identityFieldName(const QString& field)
        {
            static const QHash<QString, QString> names = {
                {QStringLiteral("onnx_sha256"), QStringLiteral("ONNX 模型内容")},
                {QStringLiteral("tensorrt"), QStringLiteral("TensorRT 版本")},
                {QStringLiteral("cuda_runtime"), QStringLiteral("CUDA 运行时版本")},
                {QStringLiteral("cuda_driver"), QStringLiteral("CUDA 驱动版本")},
                {QStringLiteral("compute_capability"), QStringLiteral("GPU 架构")},
                {QStringLiteral("precision"), QStringLiteral("构建精度")},
                {QStringLiteral("workspace_bytes"), QStringLiteral("构建工作区")},
                {QStringLiteral("builder_optimization_level"), QStringLiteral("优化等级")},
                {QStringLiteral("maximum_auxiliary_streams"), QStringLiteral("辅助流数量")},
                {QStringLiteral("input_shapes"), QStringLiteral("输入形状")},
                {QStringLiteral("required_outputs"), QStringLiteral("输出约束")},
                {QStringLiteral("attributes"), QStringLiteral("模型属性")}};
            return names.value(field, field);
        }

        QString describeIdentityDifference(const QJsonObject& previous,
                                           const QJsonObject& current)
        {
            const int previous_schema = previous.value(QStringLiteral("schema")).toInt(-1);
            const int current_schema = current.value(QStringLiteral("schema")).toInt(-1);
            if (previous_schema != current_schema)
            {
                return QStringLiteral(
                    "检测到旧缓存格式 v%1，当前要求 v%2；旧缓存缺少新版完整性和 I/O 元数据")
                    .arg(previous_schema)
                    .arg(current_schema);
            }

            static const QStringList fields = {
                QStringLiteral("onnx_sha256"),
                QStringLiteral("tensorrt"),
                QStringLiteral("cuda_runtime"),
                QStringLiteral("cuda_driver"),
                QStringLiteral("compute_capability"),
                QStringLiteral("precision"),
                QStringLiteral("workspace_bytes"),
                QStringLiteral("builder_optimization_level"),
                QStringLiteral("maximum_auxiliary_streams"),
                QStringLiteral("input_shapes"),
                QStringLiteral("required_outputs"),
                QStringLiteral("attributes")};
            QStringList differences;
            for (const QString& field : fields)
            {
                if (previous.value(field) != current.value(field))
                {
                    differences.append(identityFieldName(field));
                }
            }
            if (differences.isEmpty())
            {
                return QStringLiteral("已有缓存未通过 engine 尺寸、哈希或 I/O 完整性校验");
            }
            return QStringLiteral("缓存身份发生变化：%1").arg(differences.join(QStringLiteral("、")));
        }

    } // namespace

    QString sha256File(const QString& path, QString* errorMessage)
    {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法读取文件：%1").arg(path);
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
                    *errorMessage = QStringLiteral("计算文件 SHA-256 失败：%1").arg(path);
                }
                return QString();
            }
            hash.addData(block);
        }
        return QString::fromLatin1(hash.result().toHex());
    }

    QJsonObject makeCacheIdentity(const TensorRtEngineBuildRequest& request,
                                  const TensorRtCacheIdentityOptions& options)
    {
        QJsonArray required_outputs;
        for (const QString& name : options.requiredOutputNames)
        {
            required_outputs.append(name);
        }
        return QJsonObject{{QStringLiteral("schema"), 2},
                           {QStringLiteral("onnx_sha256"), options.onnxHash},
                           {QStringLiteral("tensorrt"), options.tensorRtVersion},
                           {QStringLiteral("cuda_runtime"), options.cudaRuntimeVersion},
                           {QStringLiteral("cuda_driver"), options.cudaDriverVersion},
                           {QStringLiteral("compute_capability"), options.computeCapability},
                           {QStringLiteral("precision"), tensorRtBuildPrecisionName(request.precision)},
                           {QStringLiteral("workspace_bytes"), static_cast<double>(options.workspaceBytes)},
                           {QStringLiteral("builder_optimization_level"), request.builderOptimizationLevel},
                           {QStringLiteral("maximum_auxiliary_streams"), request.maximumAuxiliaryStreams},
                           {QStringLiteral("input_shapes"), inputShapesToJson(options.inputShapes)},
                           {QStringLiteral("required_outputs"), required_outputs},
                           {QStringLiteral("attributes"), request.fingerprintAttributes}};
    }

    QString fingerprintCacheIdentity(const QJsonObject& identity)
    {
        return QString::fromLatin1(
            QCryptographicHash::hash(QJsonDocument(identity).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256)
                .toHex());
    }

    QString describeEngineCacheMiss(const QString& cacheRoot,
                                    const QString& engineName,
                                    const QString& currentMetadataPath,
                                    const QString& currentEnginePath,
                                    const QJsonObject& currentIdentity)
    {
        if (QFileInfo::exists(currentMetadataPath) || QFileInfo::exists(currentEnginePath))
        {
            return QStringLiteral("当前缓存未通过 engine 尺寸、哈希或 I/O 完整性校验");
        }

        const QDir root(cacheRoot);
        const QFileInfoList directories = root.entryInfoList(
            QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
        QJsonObject best_candidate;
        for (const QFileInfo& directory : directories)
        {
            const QString metadata_path =
                QDir(directory.absoluteFilePath()).filePath(engineName + QStringLiteral(".json"));
            const QString engine_path =
                QDir(directory.absoluteFilePath()).filePath(engineName);
            if (!QFileInfo::exists(engine_path))
            {
                continue;
            }
            const QJsonObject candidate = readJsonObject(metadata_path);
            if (candidate.isEmpty())
            {
                continue;
            }
            if (candidate.value(QStringLiteral("onnx_sha256")) ==
                currentIdentity.value(QStringLiteral("onnx_sha256")))
            {
                return describeIdentityDifference(candidate, currentIdentity);
            }
            if (best_candidate.isEmpty())
            {
                best_candidate = candidate;
            }
        }
        return best_candidate.isEmpty()
                   ? QStringLiteral("未找到可复用的本机 engine 缓存（首次构建）")
                   : describeIdentityDifference(best_candidate, currentIdentity);
    }

    bool loadMatchingEngineMetadata(const QString& metadataPath,
                                    const QString& enginePath,
                                    const QString& fingerprint,
                                    TensorRtEngineBuildResult* result)
    {
        const QFileInfo engine_info(enginePath);
        QFile file(metadataPath);
        if (!result || !engine_info.isFile() || engine_info.size() <= 0 || !file.open(QIODevice::ReadOnly))
        {
            return false;
        }
        const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
        if (!document.isObject())
        {
            return false;
        }
        const QJsonObject object = document.object();
        if (object.value(QStringLiteral("cache_fingerprint")).toString() != fingerprint ||
            object.value(QStringLiteral("engine_bytes")).toDouble(-1.0) != static_cast<double>(engine_info.size()))
        {
            return false;
        }
        const QString expected_engine_hash = object.value(QStringLiteral("engine_sha256")).toString();
        if (expected_engine_hash.size() != 64 || sha256File(enginePath, nullptr) != expected_engine_hash)
        {
            return false;
        }
        result->environmentSummary = object.value(QStringLiteral("environment_summary")).toString();
        result->precision = object.value(QStringLiteral("precision")).toString() == QStringLiteral("fp16")
                                ? TensorRtBuildPrecision::Fp16
                                : TensorRtBuildPrecision::Fp32;
        result->ioTensors = tensorInfoFromJson(object.value(QStringLiteral("io_tensors")).toArray());
        return !result->ioTensors.empty();
    }

    bool saveEngineMetadata(const QString& metadataPath,
                            const QJsonObject& identity,
                            const TensorRtEngineBuildResult& result,
                            const QString& onnxFileName,
                            const QString& gpuName,
                            double buildSeconds,
                            std::size_t engineBytes,
                            QString* errorMessage)
    {
        const QFileInfo engine_info(result.enginePath);
        if (!engine_info.isFile() || engine_info.size() != static_cast<qint64>(engineBytes))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("TensorRT engine 写入后尺寸不一致：%1").arg(result.enginePath);
            }
            return false;
        }
        QString hash_error;
        const QString engine_hash = sha256File(result.enginePath, &hash_error);
        if (engine_hash.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = hash_error;
            }
            return false;
        }

        QJsonObject metadata = identity;
        metadata[QStringLiteral("cache_fingerprint")] = result.cacheFingerprint;
        metadata[QStringLiteral("onnx_file")] = onnxFileName;
        metadata[QStringLiteral("engine_file")] = QFileInfo(result.enginePath).fileName();
        metadata[QStringLiteral("gpu_name")] = gpuName;
        metadata[QStringLiteral("environment_summary")] = result.environmentSummary;
        metadata[QStringLiteral("build_seconds")] = buildSeconds;
        metadata[QStringLiteral("engine_bytes")] = static_cast<double>(engineBytes);
        metadata[QStringLiteral("engine_sha256")] = engine_hash;
        metadata[QStringLiteral("io_tensors")] = tensorInfoToJson(result.ioTensors);

        QSaveFile metadata_file(metadataPath);
        if (!metadata_file.open(QIODevice::WriteOnly) ||
            metadata_file.write(QJsonDocument(metadata).toJson(QJsonDocument::Indented)) < 0 || !metadata_file.commit())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法写入 TensorRT engine 元数据：%1").arg(metadataPath);
            }
            return false;
        }
        return true;
    }

} // namespace xjw::inference::detail
