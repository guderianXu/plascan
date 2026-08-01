#include "MatchPhotosRuntime.h"

#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>

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

QString modelPathCandidate(const QString &modelName)
{
    QStringList candidates;
    const QString environmentDirectory = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!environmentDirectory.isEmpty())
    {
        candidates.append(QDir(environmentDirectory).filePath(modelName));
    }

#ifdef PLASCAN_SOURCE_DIR
    candidates.append(QDir(QStringLiteral(PLASCAN_SOURCE_DIR))
                          .filePath(QStringLiteral("resources/models/%1").arg(modelName)));
#endif

    const QString executableDirectory = QCoreApplication::applicationDirPath();
    candidates.append(QDir(executableDirectory).filePath(QStringLiteral("../models/%1").arg(modelName)));
    candidates.append(QDir(executableDirectory).filePath(QStringLiteral("../resources/models/%1").arg(modelName)));
    candidates.append(QDir(executableDirectory).filePath(QStringLiteral("../../resources/models/%1").arg(modelName)));
    candidates.append(QStringLiteral("models/%1").arg(modelName));
    for (const QString &candidate : candidates)
    {
        if (QFileInfo(candidate).isFile())
        {
            return cleanPath(QFileInfo(candidate).absoluteFilePath());
        }
    }
    return QString();
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

QString resolveLightGlueTensorRtEnginePath(const MatchPhotosOptions &options,
                                           QString *engineName)
{
    QString configured = options.lightGlueTensorRtEnginePath.trimmed();
    if (configured.isEmpty())
    {
        configured = qEnvironmentVariable("PLASCAN_LIGHTGLUE_TENSORRT_ENGINE").trimmed();
    }
    if (!configured.isEmpty())
    {
        const QFileInfo info(configured);
        if (info.isFile())
        {
            if (engineName)
            {
                *engineName = info.fileName();
            }
            return cleanPath(info.absoluteFilePath());
        }
        if (engineName)
        {
            engineName->clear();
        }
        return QString();
    }

    const QStringList candidates = {
        QStringLiteral("lightglue_sift_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket16384_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket12288_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket8192_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket4096_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket2048_fp32.engine"),
        QStringLiteral("lightglue_sift_bucket1024_fp32.engine")};
    for (const QString &candidate : candidates)
    {
        const QString path = modelPathCandidate(candidate);
        if (!path.isEmpty())
        {
            if (engineName)
            {
                *engineName = candidate;
            }
            return path;
        }
    }
    if (engineName)
    {
        engineName->clear();
    }
    return QString();
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
