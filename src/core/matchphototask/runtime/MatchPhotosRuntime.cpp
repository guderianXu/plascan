#include "MatchPhotosRuntime.h"

#include "MatchPhotosParallelism.h"
#include "tensorrt/TensorRtEngineBuilder.h"

#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSet>
#include <QStandardPaths>

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

void keepEnvironmentModelDirectories(QStringList *directories)
{
    const QString configured = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!directories || configured.isEmpty())
    {
        return;
    }
    const QString root = cleanPath(QFileInfo(configured).absoluteFilePath());
    directories->erase(
        std::remove_if(directories->begin(), directories->end(), [&](const QString &path)
        {
            const QString candidate = cleanPath(QFileInfo(path).absoluteFilePath());
            return candidate.compare(root, Qt::CaseInsensitive) != 0 &&
                !candidate.startsWith(root + QLatin1Char('/'), Qt::CaseInsensitive);
        }),
        directories->end());
}

QStringList lightGlueTensorRtModelDirectories()
{
    QStringList directories;
    const QString environmentDirectory = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!environmentDirectory.isEmpty())
    {
        appendUniqueDirectory(&directories, environmentDirectory);
    }

    const QString applicationModels = QDir(
        QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
        .filePath(QStringLiteral("models"));
    appendUniqueDirectory(&directories, applicationModels);

#ifdef PLASCAN_SOURCE_DIR
    const QDir sourceDirectory(QStringLiteral(PLASCAN_SOURCE_DIR));
    appendUniqueDirectory(
        &directories,
        sourceDirectory.filePath(QStringLiteral("build/model_cache/lightglue_tensorrt")));
    appendUniqueDirectory(
        &directories,
        sourceDirectory.filePath(QStringLiteral("resources/models")));
#endif

    const QString executableDirectory = QCoreApplication::applicationDirPath();
    const QDir executableDir(executableDirectory);
    appendUniqueDirectory(&directories, executableDir.filePath(QStringLiteral("models")));
    appendUniqueDirectory(&directories, executableDir.filePath(QStringLiteral("../models")));
    appendUniqueDirectory(&directories, executableDir.filePath(QStringLiteral("../resources/models")));
    appendUniqueDirectory(&directories, executableDir.filePath(QStringLiteral("../../resources/models")));
    appendUniqueDirectory(&directories, executableDir.filePath(QStringLiteral("../model_cache/lightglue_tensorrt")));
    appendUniqueDirectory(&directories, executableDir.filePath(QStringLiteral("../../model_cache/lightglue_tensorrt")));
    appendUniqueDirectory(&directories, QStringLiteral("models"));

    // Release 下载器按算法分目录安装。保留上面的历史根目录搜索，同时显式扩展
    // lightglue_tensorrt 子目录，确保源码运行与安装版使用完全相同的布局。
    const QStringList modelRoots = directories;
    for (const QString &root : modelRoots)
    {
        appendUniqueDirectory(
            &directories,
            QDir(root).filePath(QStringLiteral("lightglue_tensorrt")));
    }
    keepEnvironmentModelDirectories(&directories);
    return directories;
}

QStringList loMaRTensorRtModelDirectories()
{
    QStringList directories = lightGlueTensorRtModelDirectories();
#ifdef PLASCAN_SOURCE_DIR
    appendUniqueDirectory(
        &directories,
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR))
            .filePath(QStringLiteral("build/model_cache/loma_r_tensorrt")));
#endif
    const QDir executableDir(QCoreApplication::applicationDirPath());
    appendUniqueDirectory(
        &directories,
        executableDir.filePath(QStringLiteral("../model_cache/loma_r_tensorrt")));
    appendUniqueDirectory(
        &directories,
        executableDir.filePath(QStringLiteral("../../model_cache/loma_r_tensorrt")));
    const QStringList modelRoots = directories;
    for (const QString &root : modelRoots)
    {
        appendUniqueDirectory(
            &directories,
            QDir(root).filePath(QStringLiteral("loma_r_tensorrt")));
    }
    keepEnvironmentModelDirectories(&directories);
    return directories;
}

image_matching::TensorRtEngineBuildResult buildOnnxEngine(
    const QString &onnxPath,
    const QString &engineName,
    image_matching::TensorRtBuildPrecision precision,
    int cudaDevice,
    int fixedKeypointCount = 0)
{
    image_matching::TensorRtEngineBuildRequest request;
    request.onnxPath = onnxPath;
    request.cacheDirectory = QDir(QFileInfo(onnxPath).absolutePath())
        .filePath(QStringLiteral("engines"));
    request.engineName = engineName;
    request.precision = precision;
    request.cudaDevice = cudaDevice;
    request.fixedKeypointCount = fixedKeypointCount;
    return image_matching::ensureTensorRtEngine(request);
}

ResolvedLoMaRTensorRtPackage parseLoMaRPackage(const QString &manifestPath,
                                               int cudaDevice,
                                               bool prepareEngines)
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
        if (prepareEngines)
        {
            const auto precision = precisionText == QLatin1String("fp32")
                ? image_matching::TensorRtBuildPrecision::Fp32
                : image_matching::TensorRtBuildPrecision::Fp16;
            const int matcherCount = object.value(QStringLiteral("keypoint_count")).toInt();
            const int featureCount = object.value(QStringLiteral("feature_keypoint_count"))
                                         .toInt(matcherCount);
            const QString matcherSuffix = QStringLiteral("k%1_%2")
                .arg(matcherCount)
                .arg(precisionText);
            const QString featureSuffix = QStringLiteral("k%1_%2")
                .arg(featureCount)
                .arg(precisionText);
            const auto featureBuild = buildOnnxEngine(
                resolved.featureOnnxPath,
                QStringLiteral("loma_r_features_%1.engine").arg(featureSuffix),
                precision,
                cudaDevice);
            if (!featureBuild.isValid())
            {
                resolved.errorMessage = QStringLiteral("LoMa-R 特征 engine 本机构建失败：%1")
                    .arg(featureBuild.errorMessage);
                return resolved;
            }
            const auto matcherBuild = buildOnnxEngine(
                resolved.matcherOnnxPath,
                QStringLiteral("loma_r_matcher_%1.engine").arg(matcherSuffix),
                precision,
                cudaDevice,
                matcherCount);
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
        return candidate.manifestPath < current.manifestPath;
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
                                 std::max(1, maximum));
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
    bool prepareEngine)
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
                info.completeBaseName() + QStringLiteral("_fp32.engine"),
                image_matching::TensorRtBuildPrecision::Fp32,
                options.cudaDevice);
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

    const QStringList preferredNames = {
        QStringLiteral("lightglue_sift_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket16384_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket12288_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket8192_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket6144_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket4096_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket3072_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket2048_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket1024_fp32.engine")};

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
            info.completeBaseName() + QStringLiteral("_fp32.engine"),
            image_matching::TensorRtBuildPrecision::Fp32,
            options.cudaDevice);
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

    QSet<QString> visitedPaths;
    LightGlueEngineCandidate best;
    int discoveryOrder = 0;
    for (const QString &directoryPath : resolved.searchedDirectories)
    {
        const QDir directory(directoryPath);
        QStringList names = preferredNames;
        const QStringList discovered = directory.entryList(
            QStringList{QStringLiteral("lightglue_sift*_fp32.engine")},
            QDir::Files,
            QDir::Name);
        for (const QString &name : discovered)
        {
            if (!names.contains(name, Qt::CaseInsensitive))
            {
                names.append(name);
            }
        }
        for (const QString &name : names)
        {
            const QFileInfo info(directory.filePath(name));
            if (!info.isFile())
            {
                continue;
            }
            const QString path = cleanPath(info.absoluteFilePath());
            const QString identity = path.toLower();
            if (visitedPaths.contains(identity))
            {
                continue;
            }
            visitedPaths.insert(identity);

            LightGlueEngineCandidate candidate;
            candidate.path = path;
            candidate.name = info.fileName();
            candidate.bucketKeypoints = lightGlueEngineBucketFromMetadata(path);
            candidate.discoveryOrder = discoveryOrder++;
            if (engineCandidateIsBetter(candidate, best, preferredKeypoints))
            {
                best = candidate;
            }
        }
    }

    resolved.path = best.path;
    resolved.name = best.name;
    resolved.bucketKeypoints = best.bucketKeypoints;
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
    bool prepareEngines)
{
    QString configured = options.lomaRTensorRtPackagePath.trimmed();
    if (configured.isEmpty())
    {
        configured = qEnvironmentVariable("PLASCAN_LOMA_R_TENSORRT_PACKAGE").trimmed();
    }
    if (!configured.isEmpty())
    {
        return parseLoMaRPackage(configured, options.cudaDevice, prepareEngines);
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
                candidate, options.cudaDevice, prepareEngines);
            if (!resolved.isValid())
            {
                if (firstPackageError.isEmpty())
                {
                    firstPackageError = QStringLiteral("%1: %2")
                        .arg(QFileInfo(candidate).fileName(), resolved.errorMessage);
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
    settings[QStringLiteral("keypoint_limit")] = options.maxKeypoints;
    settings[QStringLiteral("keypoint_limit_per_mpx")] = options.keypointLimitPerMegapixel;
    settings[QStringLiteral("tie_point_frontend_version")] = kTiePointFrontendVersion;
    settings[QStringLiteral("mask_apply_mode")] = options.maskApplyMode.trimmed().toLower();
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
