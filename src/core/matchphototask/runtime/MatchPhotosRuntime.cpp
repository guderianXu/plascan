#include "FeatureData.h"
#include "match.h"
#include "MatchPhotosRuntime.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>

#include <algorithm>
#include <cmath>

namespace xjw
{
namespace matchphotos
{
namespace
{

QString cleanPath(const QString &path)
{
    return QDir::cleanPath(QDir::fromNativeSeparators(path.trimmed()));
}

QString imageBaseName(const QString &path)
{
    const QString base = QFileInfo(path).completeBaseName();
    return base.isEmpty() ? QFileInfo(path).fileName() : base;
}

QString normalizedImageToken(const QString &path)
{
    QString normalized = cleanPath(path);
    normalized.replace(QLatin1Char('\\'), QLatin1Char('/'));
    return normalized.toLower();
}

bool imageTokenMatches(const QString &candidate, const QString &imagePath)
{
    if (candidate.trimmed().isEmpty() || imagePath.trimmed().isEmpty())
    {
        return false;
    }

    const QFileInfo candidateInfo(candidate);
    const QFileInfo imageInfo(imagePath);
    return normalizedImageToken(candidate) == normalizedImageToken(imagePath) ||
           candidateInfo.fileName().compare(imageInfo.fileName(), Qt::CaseInsensitive) == 0 ||
           candidateInfo.completeBaseName().compare(imageInfo.completeBaseName(), Qt::CaseInsensitive) == 0;
}

QString resolveImageToken(const QString &token, const QStringList &images)
{
    for (const QString &imagePath : images)
    {
        if (imageTokenMatches(token, imagePath))
        {
            return imagePath;
        }
    }
    return QString();
}

QString modelPathCandidate(const QString &modelName)
{
    QStringList candidates;
    const QString envModelDir = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!envModelDir.isEmpty())
    {
        candidates.append(QDir(envModelDir).filePath(modelName));
    }

#ifdef PLASCAN_SOURCE_DIR
    candidates.append(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("resources/models/%1").arg(modelName)));
#endif

    const QString exeDir = QCoreApplication::applicationDirPath();
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../models/%1").arg(modelName)));
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../resources/models/%1").arg(modelName)));
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../../resources/models/%1").arg(modelName)));
    candidates.append(QStringLiteral("models/%1").arg(modelName));

    for (const QString &candidate : candidates)
    {
        if (QFile::exists(candidate))
        {
            return cleanPath(QFileInfo(candidate).absoluteFilePath());
        }
    }
    return QString();
}

QJsonArray makePointArray(float x, float y)
{
    QJsonArray point;
    point.append(static_cast<double>(x));
    point.append(static_cast<double>(y));
    return point;
}

float scoreForMatch(const xjw::feature_match::MatchResult &matchResult, int index0, float distance)
{
    if (index0 >= 0 && index0 < static_cast<int>(matchResult.matchingScores0.size()))
    {
        return matchResult.matchingScores0[static_cast<std::size_t>(index0)];
    }
    if (std::isfinite(distance))
    {
        return 1.0f / (1.0f + std::max(0.0f, distance));
    }
    return 1.0f;
}

} // namespace

QString matchPhotosFeatureDirectory(const MatchPhotosContext &context)
{
    if (!context.featureDirectory.trimmed().isEmpty())
    {
        return cleanPath(context.featureDirectory);
    }
    const QString root = context.workingDirectory.trimmed().isEmpty()
        ? QFileInfo(context.projectPath).absolutePath()
        : context.workingDirectory;
    return cleanPath(QDir(root).filePath(QStringLiteral("ip")));
}

QString matchPhotosMatchDirectory(const MatchPhotosContext &context)
{
    if (!context.matchDirectory.trimmed().isEmpty())
    {
        return cleanPath(context.matchDirectory);
    }
    const QString root = context.workingDirectory.trimmed().isEmpty()
        ? QFileInfo(context.projectPath).absolutePath()
        : context.workingDirectory;
    return cleanPath(QDir(root).filePath(QStringLiteral("matches")));
}

QString matchPhotosFeaturePath(const MatchPhotosContext &context,
                               const QString &imagePath,
                               const MatchPhotosAlgorithmPlan &plan)
{
    const QString suffix = plan.featureSuffix.isEmpty() ? QStringLiteral(".sift") : plan.featureSuffix;
    return QDir(matchPhotosFeatureDirectory(context)).filePath(imageBaseName(imagePath) + suffix);
}

QString matchPhotosMatchPath(const MatchPhotosContext &context,
                             const QString &image0Path,
                             const QString &image1Path,
                             const MatchPhotosAlgorithmPlan &plan)
{
    const QString pairName = imageBaseName(image0Path) + QStringLiteral("__") + imageBaseName(image1Path);
    const QString matcher = plan.matcherAlgorithm.isEmpty() ? QStringLiteral("lightglue") : plan.matcherAlgorithm;
    return QDir(matchPhotosMatchDirectory(context)).filePath(pairName + QStringLiteral("_") + matcher + QStringLiteral(".match"));
}

bool shouldCancelMatchPhotos(const MatchPhotosContext &context)
{
    return context.cancelFlag && context.cancelFlag->load();
}

void advanceMatchPhotosProgress(const MatchPhotosContext &context)
{
    if (context.progressCount)
    {
        context.progressCount->fetch_add(1);
    }
}

bool resolveMatchPhotosPair(const MatchPhotosContext &context,
                            const PairCandidate &candidate,
                            ResolvedImagePair *resolved,
                            QString *errorMessage)
{
    if (!resolved)
    {
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

    if (image0.trimmed().isEmpty() || image1.trimmed().isEmpty() || imageTokenMatches(image0, image1))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法解析影像对: %1").arg(candidate.pairKey);
        }
        return false;
    }

    resolved->image0Path = cleanPath(image0);
    resolved->image1Path = cleanPath(image1);
    resolved->pairName = imageBaseName(image0) + QStringLiteral("__") + imageBaseName(image1);
    resolved->pairKey = makePairKey(resolved->image0Path, resolved->image1Path);
    return true;
}

QString resolveLightGlueModelPath(const MatchPhotosAlgorithmPlan &plan,
                                  const MatchPhotosOptions &options,
                                  bool *useCuda,
                                  QString *modelName)
{
    const bool requestCuda = options.device == ComputeDevice::Cuda ||
        (options.device == ComputeDevice::Auto && plan.preferCuda);
    const QStringList suffixes = requestCuda
        ? QStringList{QStringLiteral("cuda"), QStringLiteral("cpu")}
        : QStringList{QStringLiteral("cpu")};

    for (const QString &suffix : suffixes)
    {
        const QString name = QStringLiteral("lightglue_sift_%1.torchscript").arg(suffix);
        const QString path = modelPathCandidate(name);
        if (!path.isEmpty())
        {
            if (useCuda)
            {
                *useCuda = suffix == QStringLiteral("cuda");
            }
            if (modelName)
            {
                *modelName = name;
            }
            return path;
        }
    }

    if (useCuda)
    {
        *useCuda = false;
    }
    if (modelName)
    {
        modelName->clear();
    }
    return QString();
}

QJsonObject makeFeatureRecordSettings(const MatchPhotosAlgorithmPlan &plan,
                                      const MatchPhotosOptions &options)
{
    QJsonObject settings;
    settings[QStringLiteral("feature_algorithm")] = plan.featureAlgorithm;
    settings[QStringLiteral("feature_suffix")] = plan.featureSuffix;
    settings[QStringLiteral("max_keypoints")] = plan.maxKeypoints;
    settings[QStringLiteral("max_image_dim")] = options.maxImageDim;
    settings[QStringLiteral("matchphotos_task")] = true;
    return settings;
}

QJsonObject makeMatchRecordSettings(const MatchPhotosAlgorithmPlan &plan,
                                    const MatchPhotosOptions &options,
                                    const ResolvedImagePair &pair,
                                    const QString &feature0Path,
                                    const QString &feature1Path,
                                    const QString &matchPath,
                                    const QString &sidecarPath,
                                    int matchCount)
{
    QJsonObject settings;
    QJsonArray imageFiles;
    imageFiles.append(pair.image0Path);
    imageFiles.append(pair.image1Path);
    settings[QStringLiteral("image_files")] = imageFiles;
    settings[QStringLiteral("image0")] = pair.image0Path;
    settings[QStringLiteral("image1")] = pair.image1Path;
    settings[QStringLiteral("feature0_path")] = feature0Path;
    settings[QStringLiteral("feature1_path")] = feature1Path;
    settings[QStringLiteral("sp0_path")] = feature0Path;
    settings[QStringLiteral("sp1_path")] = feature1Path;
    settings[QStringLiteral("pair_name")] = pair.pairName;
    settings[QStringLiteral("sidecar_json")] = sidecarPath;
    settings[QStringLiteral("feature_algorithm")] = plan.featureAlgorithm;
    settings[QStringLiteral("match_algorithm")] = plan.matcherAlgorithm;
    settings[QStringLiteral("match_threshold")] = static_cast<double>(options.matchThreshold);
    settings[QStringLiteral("num_matches")] = matchCount;
    settings[QStringLiteral("matchphotos_task")] = true;
    return settings;
}

bool writeMatchPhotosSidecar(const QString &sidecarPath,
                             const ResolvedImagePair &pair,
                             const QString &feature0Path,
                             const QString &feature1Path,
                             const QString &matchPath,
                             const xjw::feature_extractors::FeatureData &feature0,
                             const xjw::feature_extractors::FeatureData &feature1,
                             const xjw::feature_match::MatchResult &matchResult,
                             const MatchPhotosAlgorithmPlan &plan,
                             const MatchPhotosOptions &options)
{
    QJsonArray points0;
    QJsonArray points1;
    QJsonArray indices0;
    QJsonArray indices1;
    QJsonArray scores;

    for (const cv::DMatch &match : matchResult.cvMatches)
    {
        const int index0 = match.queryIdx;
        const int index1 = match.trainIdx;
        if (index0 < 0 || index1 < 0 ||
            index0 >= static_cast<int>(feature0.keypoints.size()) ||
            index1 >= static_cast<int>(feature1.keypoints.size()))
        {
            continue;
        }

        const cv::KeyPoint &keypoint0 = feature0.keypoints[static_cast<std::size_t>(index0)];
        const cv::KeyPoint &keypoint1 = feature1.keypoints[static_cast<std::size_t>(index1)];
        indices0.append(index0);
        indices1.append(index1);
        points0.append(makePointArray(keypoint0.pt.x, keypoint0.pt.y));
        points1.append(makePointArray(keypoint1.pt.x, keypoint1.pt.y));
        scores.append(static_cast<double>(
            std::max(0.0f, std::min(1.0f, scoreForMatch(matchResult, index0, match.distance)))));
    }

    QJsonObject sidecar = makeMatchRecordSettings(
        plan, options, pair, feature0Path, feature1Path, matchPath, sidecarPath, indices0.size());
    sidecar[QStringLiteral("match_file")] = matchPath;
    sidecar[QStringLiteral("image0_name")] = QFileInfo(pair.image0Path).completeBaseName();
    sidecar[QStringLiteral("image1_name")] = QFileInfo(pair.image1Path).completeBaseName();
    sidecar[QStringLiteral("image0_path")] = pair.image0Path;
    sidecar[QStringLiteral("image1_path")] = pair.image1Path;
    sidecar[QStringLiteral("feature_format_version")] = 2;
    sidecar[QStringLiteral("lightglue_configured_match_threshold")] =
        static_cast<double>(options.matchThreshold);
    sidecar[QStringLiteral("lightglue_effective_match_threshold")] =
        static_cast<double>(options.matchThreshold);
    sidecar[QStringLiteral("matched_points0")] = points0;
    sidecar[QStringLiteral("matched_points1")] = points1;
    sidecar[QStringLiteral("matched_indices0")] = indices0;
    sidecar[QStringLiteral("matched_indices1")] = indices1;
    sidecar[QStringLiteral("matched_scores")] = scores;

    QFile file(sidecarPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    file.write(QJsonDocument(sidecar).toJson(QJsonDocument::Compact));
    return true;
}

} // namespace matchphotos
} // namespace xjw
