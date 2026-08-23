#include "MatchPhotosRuntime.h"

#include "MatchPhotosParallelism.h"
#include "inference/tensorrt/TensorRtEngineBuilder.h"

#include "model/ModelAssetCatalog.h"
#include "model/ModelFileResolver.h"
#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSet>

#include <algorithm>
#include <cmath>

namespace xjw::matchphotos
{
namespace
{

constexpr int kTiePointFrontendVersion = 4;

QString cleanPath(const QString &path)
{
    return QDir::cleanPath(QDir::fromNativeSeparators(path.trimmed()));
}

QString imageBaseName(const QString &path)
{
    const QString base = QFileInfo(path).completeBaseName();
    return base.isEmpty() ? QFileInfo(path).fileName() : base;
}

QString resolveImageToken(const QString &token, const QStringList &images)
{
    for (const QString &imagePath : images)
    {
        if (common::project::imageTokensReferToSameImage(token, imagePath))
        {
            return imagePath;
        }
    }
    return QString();
}

void appendUniqueDirectory(QStringList *directories, const QString &path)
{
    if (!directories || path.trimmed().isEmpty())
    {
        return;
    }
    const QString clean = cleanPath(QFileInfo(path).absoluteFilePath());
    if (!directories->contains(clean, Qt::CaseInsensitive))
    {
        directories->append(clean);
    }
}

QStringList tensorRtModelDirectories(const QString &package_directory)
{
    QStringList directories;
    const common::model::ModelFileResolver resolver;
    const QStringList model_roots = resolver.searchDirectories();
    for (const QString &root : model_roots)
    {
        appendUniqueDirectory(&directories, root);
        appendUniqueDirectory(&directories, QDir(root).filePath(package_directory));
    }

#ifdef PLASCAN_SOURCE_DIR
    const QDir source_directory(QStringLiteral(PLASCAN_SOURCE_DIR));
    const QString source_models = cleanPath(
        QFileInfo(source_directory.filePath(QStringLiteral("resources/models")))
            .absoluteFilePath());
    bool source_tree_runtime = false;
    for (const QString &root : model_roots)
    {
        if (cleanPath(QFileInfo(root).absoluteFilePath())
                .compare(source_models, Qt::CaseInsensitive) == 0)
        {
            source_tree_runtime = true;
            break;
        }
    }
    if (source_tree_runtime)
    {
        appendUniqueDirectory(
            &directories,
            source_directory.filePath(
                QStringLiteral("build/model_cache/%1").arg(package_directory)));
    }
#endif

    const QDir executable_dir(QCoreApplication::applicationDirPath());
    appendUniqueDirectory(
        &directories,
        executable_dir.filePath(QStringLiteral("../model_cache/%1").arg(package_directory)));
    appendUniqueDirectory(
        &directories,
        executable_dir.filePath(QStringLiteral("../../model_cache/%1").arg(package_directory)));
    appendUniqueDirectory(&directories, QStringLiteral("models"));
    appendUniqueDirectory(
        &directories,
        QDir(QStringLiteral("models")).filePath(package_directory));
    return directories;
}

QStringList lightGlueTensorRtModelDirectories()
{
    return tensorRtModelDirectories(QStringLiteral("lightglue_tensorrt"));
}

QStringList loMaRTensorRtModelDirectories()
{
    return tensorRtModelDirectories(QStringLiteral("loma_r_tensorrt"));
}

QString fileSha256(const QString &path, QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取 ONNX 文件：%1（%2）")
                                .arg(path, file.errorString());
        }
        return QString();
    }

    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("读取 ONNX 文件失败：%1（%2）")
                                    .arg(path, file.errorString());
            }
            return QString();
        }
        hash.addData(block);
    }
    return QString::fromLatin1(hash.result().toHex());
}

bool validateDeclaredOnnxSha256(const QJsonObject &manifest,
                                const QString &field,
                                const QString &onnxPath,
                                const QString &displayName,
                                bool hashArtifact,
                                QString *errorMessage)
{
    const QString declared = manifest.value(field).toString().trimmed().toLower();
    static const QRegularExpression shaPattern(QStringLiteral("^[0-9a-f]{64}$"));
    if (!shaPattern.match(declared).hasMatch())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "LoMa-R schema 2 manifest 缺少或包含无效的 %1 SHA-256 声明（字段 %2）")
                                .arg(displayName, field);
        }
        return false;
    }
    if (!hashArtifact)
    {
        return true;
    }

    QString hashError;
    const QString actual = fileSha256(onnxPath, &hashError);
    if (actual.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = hashError;
        }
        return false;
    }
    if (actual != declared)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "LoMa-R %1 SHA-256 不匹配：manifest 声明 %2，实际文件 %3 为 %4")
                                .arg(displayName, declared, onnxPath, actual);
        }
        return false;
    }
    return true;
}

inference::TensorRtEngineBuildResult buildOnnxEngine(
    const QString &onnxPath,
    const QString &cacheDirectory,
    const QString &engineName,
    const QString &displayName,
    inference::TensorRtBuildPrecision precision,
    int cudaDevice,
    int fixedKeypointCount,
    const ModelPreparationProgressCallback &progressCallback)
{
    inference::TensorRtEngineBuildRequest request;
    request.onnxPath = onnxPath;
    request.cacheDirectory = cacheDirectory;
    request.engineName = engineName;
    request.precision = precision;
    request.cudaDevice = cudaDevice;
    request.fixedKeypointCount = fixedKeypointCount;
    if (progressCallback)
    {
        request.progressCallback = [progressCallback, displayName](
                                       const inference::TensorRtEngineBuildProgress &progress)
        {
            progressCallback(
                QStringLiteral("%1：%2").arg(displayName, progress.message),
                progress.current,
                progress.maximum);
        };
    }
    return inference::ensureTensorRtEngine(request);
}

ResolvedLoMaRTensorRtPackage parseLoMaRPackage(const QString &manifestPath,
                                               int cudaDevice,
                                               bool prepareEngines,
                                               const ModelPreparationProgressCallback &progressCallback)
{
    ResolvedLoMaRTensorRtPackage resolved;
    const QFileInfo manifestInfo(manifestPath);
    if (!manifestInfo.isFile())
    {
        resolved.errorMessage = QStringLiteral("LoMa-R manifest 不存在: %1")
            .arg(manifestPath);
        return resolved;
    }

    QFile file(manifestInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly))
    {
        resolved.errorMessage = QStringLiteral("无法读取 LoMa-R manifest: %1")
            .arg(manifestInfo.absoluteFilePath());
        return resolved;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        resolved.errorMessage = QStringLiteral("LoMa-R manifest JSON 无效: %1")
            .arg(parseError.errorString());
        return resolved;
    }

    const QJsonObject object = document.object();
    const int schemaVersion = object.value(QStringLiteral("schema_version")).toInt();
    if ((schemaVersion != 1 && schemaVersion != 2) ||
        object.value(QStringLiteral("algorithm_id")).toString() != QLatin1String("loma_r") ||
        object.value(QStringLiteral("algorithm_version")).toInt() != 1)
    {
        resolved.errorMessage = QStringLiteral("LoMa-R manifest 版本或算法标识不兼容");
        return resolved;
    }

    const QDir directory = manifestInfo.absoluteDir();
    const auto resolveFile = [&](const QString &field)
    {
        const QString configured = object.value(field).toString().trimmed();
        const QFileInfo info(QFileInfo(configured).isAbsolute()
                                 ? configured
                                 : directory.filePath(configured));
        return info.isFile() ? cleanPath(info.absoluteFilePath()) : QString();
    };
    if (schemaVersion == 1)
    {
        resolved.featureEnginePath = resolveFile(QStringLiteral("feature_engine"));
        resolved.matcherEnginePath = resolveFile(QStringLiteral("matcher_engine"));
        if (resolved.featureEnginePath.isEmpty())
        {
            resolved.errorMessage = QStringLiteral("LoMa-R feature engine 不存在");
            return resolved;
        }
        if (resolved.matcherEnginePath.isEmpty())
        {
            resolved.errorMessage = QStringLiteral("LoMa-R matcher engine 不存在");
            return resolved;
        }
    }
    else
    {
        resolved.featureOnnxPath = resolveFile(QStringLiteral("feature_onnx"));
        resolved.matcherOnnxPath = resolveFile(QStringLiteral("matcher_onnx"));
        if (resolved.featureOnnxPath.isEmpty() || resolved.matcherOnnxPath.isEmpty())
        {
            resolved.errorMessage = QStringLiteral("LoMa-R ONNX 模型包不完整");
            return resolved;
        }
        const QString precisionText = object.value(QStringLiteral("precision"))
                                          .toString(QStringLiteral("fp16"))
                                          .trimmed()
                                          .toLower();
        if (precisionText != QLatin1String("fp16") &&
            precisionText != QLatin1String("fp32"))
        {
            resolved.errorMessage = QStringLiteral(
                "LoMa-R schema 2 manifest 的 precision 必须是 fp16 或 fp32");
            return resolved;
        }
        if (!validateDeclaredOnnxSha256(
                object,
                QStringLiteral("feature_onnx_sha256"),
                resolved.featureOnnxPath,
                QStringLiteral("feature ONNX"),
                prepareEngines,
                &resolved.errorMessage) ||
            !validateDeclaredOnnxSha256(
                object,
                QStringLiteral("matcher_onnx_sha256"),
                resolved.matcherOnnxPath,
                QStringLiteral("matcher ONNX"),
                prepareEngines,
                &resolved.errorMessage))
        {
            return resolved;
        }
        if (prepareEngines)
        {
            constexpr int kEnginePreparationSteps = 6;
            constexpr int kPackagePreparationSteps = 2 * kEnginePreparationSteps;
            const auto packageProgress = [&progressCallback](int offset)
            {
                if (!progressCallback)
                {
                    return ModelPreparationProgressCallback();
                }
                return ModelPreparationProgressCallback(
                    [progressCallback, offset, total = kPackagePreparationSteps](
                        const QString &message, int current, int maximum)
                    {
                        progressCallback(
                            message,
                            maximum <= 0 ? 0 : offset + current,
                            maximum <= 0 ? 0 : total);
                    });
            };
            const auto precision = precisionText == QLatin1String("fp32")
                ? inference::TensorRtBuildPrecision::Fp32
                : inference::TensorRtBuildPrecision::Fp16;
            const int matcherCount = object.value(QStringLiteral("keypoint_count")).toInt();
            const int featureCount = object.value(QStringLiteral("feature_keypoint_count"))
                                         .toInt(matcherCount);
            const QString matcherSuffix = QStringLiteral("k%1_%2")
                .arg(matcherCount)
                .arg(precisionText);
            const QString featureSuffix = QStringLiteral("k%1_%2")
                .arg(featureCount)
                .arg(precisionText);
            const QString cacheDirectory =
                common::model::modelPackageEngineCacheDirectory(
                    common::model::loMaRTensorRtPackage(matcherCount));
            const auto featureBuild = buildOnnxEngine(
                resolved.featureOnnxPath,
                cacheDirectory,
                QStringLiteral("loma_r_features_%1.engine").arg(featureSuffix),
                QStringLiteral("LoMa-R 特征 engine"),
                precision,
                cudaDevice,
                0,
                packageProgress(0));
            if (!featureBuild.isValid())
            {
                resolved.errorMessage = QStringLiteral("LoMa-R 特征 engine 本机构建失败：%1")
                    .arg(featureBuild.errorMessage);
                return resolved;
            }
            const auto matcherBuild = buildOnnxEngine(
                resolved.matcherOnnxPath,
                cacheDirectory,
                QStringLiteral("loma_r_matcher_%1.engine").arg(matcherSuffix),
                QStringLiteral("LoMa-R 匹配 engine"),
                precision,
                cudaDevice,
                matcherCount,
                packageProgress(kEnginePreparationSteps));
            if (!matcherBuild.isValid())
            {
                resolved.errorMessage = QStringLiteral("LoMa-R 匹配 engine 本机构建失败：%1")
                    .arg(matcherBuild.errorMessage);
                return resolved;
            }
            resolved.featureEnginePath = featureBuild.enginePath;
            resolved.matcherEnginePath = matcherBuild.enginePath;
            resolved.environmentSummary = featureBuild.environmentSummary;
        }
    }

    resolved.inputWidth = object.value(QStringLiteral("input_width")).toInt();
    resolved.inputHeight = object.value(QStringLiteral("input_height")).toInt();
    resolved.keypointCount = object.value(QStringLiteral("keypoint_count")).toInt();
    resolved.featureKeypointCount = object.value(QStringLiteral("feature_keypoint_count"))
                                        .toInt(resolved.keypointCount);
    resolved.descriptorDimension = object.value(QStringLiteral("descriptor_dimension")).toInt();
    if (resolved.inputWidth <= 0 || resolved.inputHeight <= 0 ||
        resolved.keypointCount <= 0 ||
        resolved.featureKeypointCount < resolved.keypointCount ||
        resolved.descriptorDimension != 256)
    {
        resolved.errorMessage = QStringLiteral("LoMa-R manifest 的输入尺寸或特征规格无效");
        resolved.featureEnginePath.clear();
        resolved.matcherEnginePath.clear();
        return resolved;
    }
    resolved.manifestPath = cleanPath(manifestInfo.absoluteFilePath());
    return resolved;
}

int lightGlueEngineBucketFromMetadata(const QString &enginePath)
{
    QFile metadataFile(enginePath + QStringLiteral(".json"));
    if (metadataFile.open(QIODevice::ReadOnly))
    {
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(
            metadataFile.readAll(), &parseError);
        if (parseError.error == QJsonParseError::NoError && document.isObject())
        {
            const int bucket = document.object()
                .value(QStringLiteral("bucket_keypoints")).toInt();
            if (bucket > 0)
            {
                return bucket;
            }
        }
    }

    const QRegularExpression bucketPattern(
        QStringLiteral("(?:^|_)bucket(\\d+)(?:_|\\.)"),
        QRegularExpression::CaseInsensitiveOption);
    const QRegularExpressionMatch match = bucketPattern.match(QFileInfo(enginePath).fileName());
    return match.hasMatch() ? match.captured(1).toInt() : 0;
}

struct LightGlueEngineCandidate
{
    QString path;
    QString name;
    int bucketKeypoints = 0;
    int discoveryOrder = 0;
};

bool engineCandidateIsBetter(const LightGlueEngineCandidate &candidate,
                             const LightGlueEngineCandidate &current,
                             int preferredKeypoints)
{
    if (current.path.isEmpty())
    {
        return true;
    }

    const auto category = [preferredKeypoints](int bucket)
    {
        if (bucket > 0 && (preferredKeypoints <= 0 || bucket <= preferredKeypoints))
        {
            return 0;
        }
        if (bucket == 0)
        {
            return 1;
        }
        return 2;
    };
    const int candidateCategory = category(candidate.bucketKeypoints);
    const int currentCategory = category(current.bucketKeypoints);
    if (candidateCategory != currentCategory)
    {
        return candidateCategory < currentCategory;
    }
    if (candidateCategory == 0 && candidate.bucketKeypoints != current.bucketKeypoints)
    {
        return candidate.bucketKeypoints > current.bucketKeypoints;
    }
    if (candidateCategory == 2 && candidate.bucketKeypoints != current.bucketKeypoints)
    {
        return candidate.bucketKeypoints < current.bucketKeypoints;
    }
    return candidate.discoveryOrder < current.discoveryOrder;
}

bool loMaRPackageIsBetter(const ResolvedLoMaRTensorRtPackage &candidate,
                          const ResolvedLoMaRTensorRtPackage &current,
                          int preferredKeypoints)
{
    if (!current.isValid())
    {
        return true;
    }

    const bool candidateWithinBudget = candidate.keypointCount <= preferredKeypoints;
    const bool currentWithinBudget = current.keypointCount <= preferredKeypoints;
    if (candidateWithinBudget != currentWithinBudget)
    {
        return candidateWithinBudget;
    }
    if (candidate.keypointCount == current.keypointCount)
    {
        // 搜索目录按“显式环境目录 -> 安装资源 -> 兼容缓存”排列；同一档位
        // 必须保留先发现的模型，不能再用绝对路径字典序打乱该优先级。
        return false;
    }
    return candidateWithinBudget
        ? candidate.keypointCount > current.keypointCount
        : candidate.keypointCount < current.keypointCount;
}

} // namespace

QString matchPhotosMatchDirectory(const MatchPhotosContext &context)
{
    if (!context.matchDirectory.trimmed().isEmpty())
    {
        return cleanPath(context.matchDirectory);
    }
    const QString root = context.workingDirectory.trimmed().isEmpty()
        ? common::project::ProjectIO::projectRootFromPlascan(context.projectPath)
        : context.workingDirectory;
    return cleanPath(QDir(root).filePath(QStringLiteral("image_matches")));
}

bool shouldCancelMatchPhotos(const MatchPhotosContext &context)
{
    return context.cancelFlag && context.cancelFlag->load(std::memory_order_relaxed);
}

void advanceMatchPhotosProgress(const MatchPhotosContext &context)
{
    if (context.progressCount)
    {
        context.progressCount->fetch_add(1, std::memory_order_relaxed);
    }
}

void reportMatchPhotosProgress(const MatchPhotosContext &context,
                               const QString &stageId,
                               const QString &message,
                               int current,
                               int maximum)
{
    if (context.progressCallback)
    {
        context.progressCallback(stageId,
                                 message,
                                 std::max(0, current),
                                 std::max(0, maximum));
    }
}

bool resolveMatchPhotosPair(const MatchPhotosContext &context,
                            const PairCandidate &candidate,
                            ResolvedImagePair *resolved,
                            QString *errorMessage)
{
    if (!resolved)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：影像对输出为空");
        }
        return false;
    }

    QString image0;
    QString image1;
    const QStringList &images = context.pairInput.images;
    if (candidate.pair.isValid(images.size()))
    {
        const ImagePair normalized = candidate.pair.normalized();
        image0 = images.at(normalized.indexA);
        image1 = images.at(normalized.indexB);
    }
    else
    {
        const QStringList parts = candidate.pairKey.split(QLatin1Char('\n'));
        if (parts.size() == 2)
        {
            image0 = resolveImageToken(parts.at(0), images);
            image1 = resolveImageToken(parts.at(1), images);
        }
    }

    if (image0.trimmed().isEmpty() || image1.trimmed().isEmpty() ||
        common::project::imageTokensReferToSameImage(image0, image1))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法解析影像对：%1").arg(candidate.pairKey);
        }
        return false;
    }

    resolved->image0Path = cleanPath(QFileInfo(image0).absoluteFilePath());
    resolved->image1Path = cleanPath(QFileInfo(image1).absoluteFilePath());
    resolved->pairName = imageBaseName(image0) + QStringLiteral(" / ") + imageBaseName(image1);
    resolved->pairKey = makePairKey(resolved->image0Path, resolved->image1Path);
    return true;
}

ResolvedLightGlueTensorRtEngine resolveLightGlueTensorRtEngine(
    const MatchPhotosOptions &options,
    int preferredKeypoints,
    bool prepareEngine,
    const ModelPreparationProgressCallback &progressCallback)
{
    ResolvedLightGlueTensorRtEngine resolved;
    QString configured = options.lightGlueTensorRtEnginePath.trimmed();
    if (configured.isEmpty())
    {
        configured = qEnvironmentVariable("PLASCAN_LIGHTGLUE_TENSORRT_ENGINE").trimmed();
    }
    if (!configured.isEmpty())
    {
        const QFileInfo info(configured);
        if (info.isFile() && info.suffix().compare(QStringLiteral("onnx"),
                                                   Qt::CaseInsensitive) == 0)
        {
            if (!prepareEngine)
            {
                resolved.path = cleanPath(info.absoluteFilePath());
                resolved.name = info.fileName();
                resolved.sourceOnnxPath = resolved.path;
                resolved.bucketKeypoints = lightGlueEngineBucketFromMetadata(resolved.path);
                return resolved;
            }
            const auto build = buildOnnxEngine(
                info.absoluteFilePath(),
                common::model::modelPackageEngineCacheDirectory(
                    common::model::lightGlueTensorRtPackage()),
                info.completeBaseName() + QStringLiteral("_fp32.engine"),
                QStringLiteral("LightGlue engine"),
                inference::TensorRtBuildPrecision::Fp32,
                options.cudaDevice,
                0,
                progressCallback);
            if (build.isValid())
            {
                resolved.path = build.enginePath;
                resolved.name = QFileInfo(build.enginePath).fileName();
                resolved.sourceOnnxPath = info.absoluteFilePath();
                resolved.environmentSummary = build.environmentSummary;
                resolved.bucketKeypoints = lightGlueEngineBucketFromMetadata(
                    info.absoluteFilePath());
            }
            else
            {
                resolved.errorMessage = build.errorMessage;
            }
        }
        else if (info.isFile())
        {
            resolved.path = cleanPath(info.absoluteFilePath());
            resolved.name = info.fileName();
            resolved.bucketKeypoints = lightGlueEngineBucketFromMetadata(resolved.path);
        }
        return resolved;
    }

    resolved.searchedDirectories = lightGlueTensorRtModelDirectories();

    LightGlueEngineCandidate bestOnnx;
    int onnxDiscoveryOrder = 0;
    for (const QString &directoryPath : resolved.searchedDirectories)
    {
        const QDir directory(directoryPath);
        const QStringList names = directory.entryList(
            QStringList{QStringLiteral("lightglue_sift*.onnx")}, QDir::Files, QDir::Name);
        for (const QString &name : names)
        {
            LightGlueEngineCandidate candidate;
            candidate.path = cleanPath(QFileInfo(directory.filePath(name)).absoluteFilePath());
            candidate.name = name;
            candidate.bucketKeypoints = lightGlueEngineBucketFromMetadata(candidate.path);
            candidate.discoveryOrder = onnxDiscoveryOrder++;
            if (engineCandidateIsBetter(candidate, bestOnnx, preferredKeypoints))
            {
                bestOnnx = candidate;
            }
        }
    }
    if (!bestOnnx.path.isEmpty())
    {
        if (!prepareEngine)
        {
            resolved.path = bestOnnx.path;
            resolved.name = bestOnnx.name;
            resolved.sourceOnnxPath = bestOnnx.path;
            resolved.bucketKeypoints = bestOnnx.bucketKeypoints;
            return resolved;
        }
        const QFileInfo info(bestOnnx.path);
        const auto build = buildOnnxEngine(
            bestOnnx.path,
            common::model::modelPackageEngineCacheDirectory(
                common::model::lightGlueTensorRtPackage()),
            info.completeBaseName() + QStringLiteral("_fp32.engine"),
            QStringLiteral("LightGlue engine"),
            inference::TensorRtBuildPrecision::Fp32,
            options.cudaDevice,
            0,
            progressCallback);
        if (build.isValid())
        {
            resolved.path = build.enginePath;
            resolved.name = QFileInfo(build.enginePath).fileName();
            resolved.sourceOnnxPath = bestOnnx.path;
            resolved.environmentSummary = build.environmentSummary;
            resolved.bucketKeypoints = bestOnnx.bucketKeypoints;
            return resolved;
        }
        resolved.errorMessage = build.errorMessage;
        return resolved;
    }

    // 自动发现不再加载旧 Release 留在用户目录中的裸 engine。此类 plan 可能绑定
    // 另一块 GPU 或 TensorRT 补丁版本；需要兼容使用时仍可通过显式路径选择。
    resolved.errorMessage = QStringLiteral(
        "未找到便携 LightGlue ONNX；请在工作流程设置中下载 models-v1.1.0 模型");
    return resolved;
}

QString resolveLightGlueTensorRtEnginePath(const MatchPhotosOptions &options,
                                           QString *engineName)
{
    const ResolvedLightGlueTensorRtEngine resolved =
        resolveLightGlueTensorRtEngine(options);
    if (engineName)
    {
        *engineName = resolved.name;
    }
    return resolved.path;
}

ResolvedLoMaRTensorRtPackage resolveLoMaRTensorRtPackage(
    const MatchPhotosOptions &options,
    int preferredKeypoints,
    bool prepareEngines,
    const ModelPreparationProgressCallback &progressCallback)
{
    QString configured = options.lomaRTensorRtPackagePath.trimmed();
    if (configured.isEmpty())
    {
        configured = qEnvironmentVariable("PLASCAN_LOMA_R_TENSORRT_PACKAGE").trimmed();
    }
    if (!configured.isEmpty())
    {
        return parseLoMaRPackage(
            configured, options.cudaDevice, prepareEngines, progressCallback);
    }

    const int targetKeypoints = resolveLoMaRKeypointBudget(
        preferredKeypoints,
        options.lomaRKeypointBudget,
        queryMatchPhotosGpuMemory(options.cudaDevice));

    ResolvedLoMaRTensorRtPackage unresolved;
    unresolved.searchedDirectories = loMaRTensorRtModelDirectories();
    ResolvedLoMaRTensorRtPackage best;
    QString firstPackageError;
    QSet<QString> visitedManifests;
    for (const QString &directory : unresolved.searchedDirectories)
    {
        QStringList names = {
            QStringLiteral("loma_r_tensorrt.json"),
            QStringLiteral("loma_r_fp16.json"),
            QStringLiteral("loma_r_fp32.json")};
        const QStringList discovered = QDir(directory).entryList(
            QStringList{QStringLiteral("loma_r*.json")},
            QDir::Files,
            QDir::Name);
        for (const QString &name : discovered)
        {
            if (name.endsWith(QStringLiteral(".provenance.json"),
                              Qt::CaseInsensitive))
            {
                continue;
            }
            if (!names.contains(name, Qt::CaseInsensitive))
            {
                names.append(name);
            }
        }
        for (const QString &name : names)
        {
            const QString candidate = QDir(directory).filePath(name);
            if (!QFileInfo::exists(candidate))
            {
                continue;
            }
            const QString identity = cleanPath(QFileInfo(candidate).absoluteFilePath()).toLower();
            if (visitedManifests.contains(identity))
            {
                continue;
            }
            visitedManifests.insert(identity);
            ResolvedLoMaRTensorRtPackage resolved = parseLoMaRPackage(
                candidate, options.cudaDevice, false, {});
            if (!resolved.isValid())
            {
                if (firstPackageError.isEmpty())
                {
                    firstPackageError = QStringLiteral("%1: %2")
                        .arg(QFileInfo(candidate).fileName(), resolved.errorMessage);
                }
                continue;
            }
            if (resolved.featureOnnxPath.isEmpty() || resolved.matcherOnnxPath.isEmpty())
            {
                if (firstPackageError.isEmpty())
                {
                    firstPackageError = QStringLiteral(
                        "%1 是旧版 engine 清单，请下载 models-v1.1.0 ONNX 模型包")
                                            .arg(QFileInfo(candidate).fileName());
                }
                continue;
            }
            if (loMaRPackageIsBetter(resolved, best, targetKeypoints))
            {
                best = std::move(resolved);
            }
        }
    }
    if (best.isValid())
    {
        if (prepareEngines)
        {
            ResolvedLoMaRTensorRtPackage prepared = parseLoMaRPackage(
                best.manifestPath, options.cudaDevice, true, progressCallback);
            prepared.searchedDirectories = unresolved.searchedDirectories;
            return prepared;
        }
        best.searchedDirectories = unresolved.searchedDirectories;
        return best;
    }
    unresolved.errorMessage = firstPackageError.isEmpty()
        ? QStringLiteral("未找到 LoMa-R TensorRT manifest")
        : QStringLiteral("未找到完整可用的 LoMa-R TensorRT 模型包；%1")
              .arg(firstPackageError);
    return unresolved;
}

int resolveFeatureKeypointLimit(const MatchPhotosOptions &options,
                                const MatchPhotosAlgorithmPlan &plan,
                                int imageWidth,
                                int imageHeight)
{
    if (plan.enableGuidedMatching && options.keypointLimitPerMegapixel > 0 &&
        imageWidth > 0 && imageHeight > 0)
    {
        const double megapixels = static_cast<double>(imageWidth) *
            static_cast<double>(imageHeight) / 1000000.0;
        return std::max(1,
                        static_cast<int>(std::round(
                            static_cast<double>(options.keypointLimitPerMegapixel) * megapixels)));
    }
    return std::max(0, plan.maxKeypoints);
}

QJsonObject makeFeatureRecordSettings(const MatchPhotosAlgorithmPlan &plan,
                                      const MatchPhotosOptions &options)
{
    QJsonObject settings;
    settings[QStringLiteral("algorithm_id")] = plan.algorithmId;
    settings[QStringLiteral("algorithm_version")] = static_cast<int>(plan.algorithmVersion);
    settings[QStringLiteral("max_keypoints")] = plan.maxKeypoints;
    settings[QStringLiteral("keypoint_limit_per_mpx")] = options.keypointLimitPerMegapixel;
    settings[QStringLiteral("mask_apply_mode")] = options.maskApplyMode.trimmed().toLower();
    settings[QStringLiteral("max_image_dim")] = options.maxImageDim;
    settings[QStringLiteral("low_texture_recovery")] = plan.lowTextureRecovery;
    settings[QStringLiteral("storage")] = QStringLiteral("memory_only");
    return settings;
}

QJsonObject makeMatchRecordSettings(const MatchPhotosAlgorithmPlan &plan,
                                    const MatchPhotosOptions &options,
                                    const ResolvedImagePair &pair,
                                    int matchCount,
                                    const QJsonObject &extraSettings)
{
    QJsonArray images;
    images.append(pair.image0Path);
    images.append(pair.image1Path);

    QJsonObject settings;
    settings[QStringLiteral("image_files")] = images;
    settings[QStringLiteral("image0")] = pair.image0Path;
    settings[QStringLiteral("image1")] = pair.image1Path;
    settings[QStringLiteral("pair_name")] = pair.pairName;
    settings[QStringLiteral("pair_key")] = pair.pairKey;
    settings[QStringLiteral("algorithm_id")] = plan.algorithmId;
    settings[QStringLiteral("algorithm_version")] = static_cast<int>(plan.algorithmVersion);
    settings[QStringLiteral("match_threshold")] = static_cast<double>(options.matchThreshold);
    settings[QStringLiteral("sift_maximum_ratio")] = static_cast<double>(options.siftMaximumRatio);
    settings[QStringLiteral("sift_minimum_adaptive_ratio")] =
        static_cast<double>(options.siftMinimumAdaptiveRatio);
    settings[QStringLiteral("adaptive_sift_ratio")] = options.adaptiveSiftRatio;
    settings[QStringLiteral("low_texture_recovery")] = plan.lowTextureRecovery;
    settings[QStringLiteral("keypoint_limit")] = options.maxKeypoints;
    settings[QStringLiteral("keypoint_limit_per_mpx")] = options.keypointLimitPerMegapixel;
    settings[QStringLiteral("tie_point_frontend_version")] = kTiePointFrontendVersion;
    settings[QStringLiteral("mask_apply_mode")] = options.maskApplyMode.trimmed().toLower();
    settings[QStringLiteral("mask_hard_exclusion_threshold")] =
        static_cast<double>(options.maskHardExclusionThreshold);
    settings[QStringLiteral("mask_minimum_tiepoint_weight")] =
        static_cast<double>(options.maskMinimumTiepointWeight);
    settings[QStringLiteral("mask_relaxation_radius")] = options.maskRelaxationRadius;
    settings[QStringLiteral("tiepoint_limit")] = options.maxTiePointsPerImage;
    settings[QStringLiteral("exclude_stationary_tie_points")] =
        options.excludeStationaryTiePoints;
    settings[QStringLiteral("num_matches")] = matchCount;
    settings[QStringLiteral("storage_format")] = QStringLiteral("pimatch");
    for (auto it = extraSettings.constBegin(); it != extraSettings.constEnd(); ++it)
    {
        settings[it.key()] = it.value();
    }
    return settings;
}

} // namespace xjw::matchphotos
