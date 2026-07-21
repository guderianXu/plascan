#include "preparation/MatchResultCatalog.h"

#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QMap>
#include <QStringList>

#include <algorithm>
#include <cstring>

namespace xjw::aerial_triangulation
{
namespace
{
struct SgmtHeader
{
    bool ok = false;
    int matchCount = -1;
    QString imageA;
    QString imageB;
    QString reason;
};
QString normalizedPathKey(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }
    QString key = QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
#if defined(Q_OS_WIN)
    key = key.toLower();
#endif
    return key;
}
QString lowerCleanToken(const QString &token)
{
    return QDir::cleanPath(QDir::fromNativeSeparators(token.trimmed())).toLower();
}
bool tokenHasPathComponent(const QString &token)
{
    const QString clean = QDir::fromNativeSeparators(token.trimmed());
    return QFileInfo(clean).isAbsolute() || clean.contains(QLatin1Char('/'));
}
bool imageTokensReferToSameImage(const QString &lhs, const QString &rhs)
{
    const QString left = lhs.trimmed();
    const QString right = rhs.trimmed();
    if (left.isEmpty() || right.isEmpty())
    {
        return false;
    }
    if (lowerCleanToken(left) == lowerCleanToken(right))
    {
        return true;
    }
    if (tokenHasPathComponent(left) && tokenHasPathComponent(right))
    {
        return false;
    }
    const QFileInfo leftInfo(left);
    const QFileInfo rightInfo(right);
    const QString leftFile = leftInfo.fileName().toLower();
    const QString rightFile = rightInfo.fileName().toLower();
    if (!leftFile.isEmpty() && leftFile == rightFile)
    {
        return true;
    }
    const QString leftBase = leftInfo.completeBaseName().toLower();
    const QString rightBase = rightInfo.completeBaseName().toLower();
    return !leftBase.isEmpty() && leftBase == rightBase;
}
bool unorderedImageTokensMatch(const QString &headerA,
                               const QString &headerB,
                               const QString &sidecarA,
                               const QString &sidecarB)
{
    const bool direct = imageTokensReferToSameImage(headerA, sidecarA) &&
                        imageTokensReferToSameImage(headerB, sidecarB);
    const bool reverse = imageTokensReferToSameImage(headerA, sidecarB) &&
                         imageTokensReferToSameImage(headerB, sidecarA);
    return direct || reverse;
}
bool pairContainsImageToken(const QString &imageA,
                            const QString &imageB,
                            const QString &targetImagePath)
{
    if (targetImagePath.trimmed().isEmpty())
    {
        return true;
    }

    return imageTokensReferToSameImage(imageA, targetImagePath) ||
           imageTokensReferToSameImage(imageB, targetImagePath);
}
bool imageTokenInSet(const QString &imageToken, const QStringList &targetImagePaths)
{
    if (imageToken.trimmed().isEmpty())
    {
        return false;
    }
    for (const QString &targetImagePath : targetImagePaths)
    {
        if (imageTokensReferToSameImage(imageToken, targetImagePath))
        {
            return true;
        }
    }
    return false;
}
bool pairBelongsToImageSet(const QString &imageA,
                           const QString &imageB,
                           const QStringList &targetImagePaths)
{
    if (targetImagePaths.isEmpty())
    {
        return true;
    }

    return imageTokenInSet(imageA, targetImagePaths) &&
           imageTokenInSet(imageB, targetImagePaths);
}
bool readUtf8String(QDataStream &in, QString *value)
{
    if (!value)
    {
        return false;
    }
    quint32 length = 0;
    in >> length;
    if (in.status() != QDataStream::Ok || length > 1024 * 1024)
    {
        return false;
    }
    QByteArray bytes(static_cast<int>(length), 0);
    if (in.readRawData(bytes.data(), static_cast<int>(length)) != static_cast<int>(length))
    {
        return false;
    }
    *value = QString::fromUtf8(bytes);
    return true;
}
SgmtHeader readSgmtHeader(const QString &path)
{
    SgmtHeader header;

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        header.reason = QStringLiteral("match_file_unreadable");
        return header;
    }
    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);

    char magic[4] = {};
    if (in.readRawData(magic, 4) != 4 || std::strncmp(magic, "SGMT", 4) != 0)
    {
        header.reason = QStringLiteral("sgmt_magic_missing");
        return header;
    }
    quint32 version = 0;
    in >> version;
    if (in.status() != QDataStream::Ok || (version != 1 && version != 2))
    {
        header.reason = QStringLiteral("sgmt_version_unsupported");
        return header;
    }
    if (!readUtf8String(in, &header.imageA) || !readUtf8String(in, &header.imageB))
    {
        header.reason = QStringLiteral("sgmt_image_names_unreadable");
        return header;
    }
    qint32 matchCount = -1;
    qint32 ignoredKeypointCountA = 0;
    qint32 ignoredKeypointCountB = 0;
    in >> matchCount >> ignoredKeypointCountA >> ignoredKeypointCountB;
    if (in.status() != QDataStream::Ok || matchCount < 0)
    {
        header.reason = QStringLiteral("sgmt_match_count_unreadable");
        return header;
    }
    header.ok = true;
    header.matchCount = static_cast<int>(matchCount);
    return header;
}
QJsonObject readJsonObject(const QString &path, QString *errorReason)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorReason)
        {
            *errorReason = QStringLiteral("sidecar_json_unreadable");
        }
        return QJsonObject();
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        if (errorReason)
        {
            *errorReason = QStringLiteral("sidecar_json_parse_failed");
        }
        return QJsonObject();
    }
    return document.object();
}
QString firstString(const QJsonObject &object, const QStringList &keys)
{
    for (const QString &key : keys)
    {
        const QString value = object.value(key).toString().trimmed();
        if (!value.isEmpty())
        {
            return value;
        }
    }
    return QString();
}
bool firstInt(const QJsonObject &object, const QStringList &keys, int *valueOut)
{
    for (const QString &key : keys)
    {
        const QJsonValue value = object.value(key);
        if (value.isDouble())
        {
            *valueOut = std::max(0, value.toInt());
            return true;
        }
    }
    return false;
}
QString sidecarAlgorithm(const QJsonObject &sidecar, const QString &key)
{
    QString value = sidecar.value(key).toString().trimmed();
    if (value.isEmpty())
    {
        value = sidecar.value(QStringLiteral("settings")).toObject()
                    .value(key).toString().trimmed();
    }
    return value.toLower();
}

QString featureAlgorithmFromFeaturePath(const QString &path)
{
    const QString suffix = QFileInfo(path.trimmed()).suffix().toLower();
    if (suffix == QStringLiteral("dsk"))
    {
        return QStringLiteral("disk");
    }
    if (suffix == QStringLiteral("alk"))
    {
        return QStringLiteral("aliked");
    }
    if (suffix == QStringLiteral("sift"))
    {
        return QStringLiteral("sift");
    }
    if (suffix == QStringLiteral("orb"))
    {
        return QStringLiteral("orb");
    }
    if (suffix == QStringLiteral("akz"))
    {
        return QStringLiteral("akaze");
    }
    if (suffix == QStringLiteral("dedode"))
    {
        return QStringLiteral("dedode");
    }
    if (suffix == QStringLiteral("sp"))
    {
        return QStringLiteral("superpoint");
    }
    return QString();
}

QString inferFeatureAlgorithmFromSidecar(const QJsonObject &sidecar)
{
    const QString featurePath = firstString(sidecar, {
        QStringLiteral("feature0_path"),
        QStringLiteral("feature1_path"),
        QStringLiteral("sp0_path"),
        QStringLiteral("sp1_path")
    });
    return featureAlgorithmFromFeaturePath(featurePath);
}

QString normalizedAlgorithmToken(QString value)
{
    value = value.trimmed().toLower();
    value.replace(QLatin1Char('_'), QLatin1Char('-'));
    value.replace(QLatin1Char('+'), QLatin1Char('-'));
    value.replace(QLatin1Char(' '), QLatin1Char('-'));
    while (value.contains(QStringLiteral("--")))
    {
        value.replace(QStringLiteral("--"), QStringLiteral("-"));
    }
    return value;
}

QString inferFeatureAlgorithmFromMatchStem(const QString &stem, const QString &matchAlgorithm)
{
    const QString haystack = QStringLiteral("_") + stem.toLower() + QStringLiteral("_");
    const QString normalizedMatch = normalizedAlgorithmToken(matchAlgorithm);
    const QStringList features = {
        QStringLiteral("superpoint"),
        QStringLiteral("disk"),
        QStringLiteral("aliked"),
        QStringLiteral("sift"),
        QStringLiteral("orb"),
        QStringLiteral("akaze"),
        QStringLiteral("dedode")
    };

    for (const QString &feature : features)
    {
        if (haystack.contains(QStringLiteral("_%1_lightglue_").arg(feature)) ||
            haystack.contains(QStringLiteral("_%1_superglue_").arg(feature)) ||
            haystack.contains(QStringLiteral("_%1_bf_").arg(feature)) ||
            haystack.contains(QStringLiteral("_%1_flann_").arg(feature)) ||
            normalizedMatch.startsWith(feature + QLatin1Char('-')))
        {
            return feature;
        }
    }

    return QString();
}

void setIncompatible(MatchVariant *variant, const QString &status, const QString &reason)
{
    variant->compatible = false;
    variant->status = status;
    variant->reason = reason;
}
MatchVariant readVariant(const QFileInfo &matchInfo)
{
    MatchVariant variant;
    variant.matchFilePath = matchInfo.absoluteFilePath();
    variant.sidecarPath = variant.matchFilePath + QStringLiteral(".json");
    variant.modifiedTime = matchInfo.lastModified();
    variant.status = QStringLiteral("compatible");
    const SgmtHeader header = readSgmtHeader(variant.matchFilePath);
    if (header.ok)
    {
        variant.imageA = header.imageA;
        variant.imageB = header.imageB;
        variant.totalMatches = header.matchCount;
    }
    else
    {
        setIncompatible(&variant, QStringLiteral("invalid_match_file"), header.reason);
    }

    if (!QFileInfo::exists(variant.sidecarPath))
    {
        if (variant.status == QStringLiteral("compatible"))
        {
            setIncompatible(&variant,
                            QStringLiteral("missing_sidecar"),
                            QStringLiteral("sidecar_json_missing"));
        }
        return variant;
    }

    QString sidecarError;
    const QJsonObject sidecar = readJsonObject(variant.sidecarPath, &sidecarError);
    if (sidecar.isEmpty() && !sidecarError.isEmpty())
    {
        setIncompatible(&variant, QStringLiteral("invalid_sidecar"), sidecarError);
        return variant;
    }

    const QString sidecarImageA = firstString(sidecar, {
        QStringLiteral("image0_path"),
        QStringLiteral("image0_name"),
        QStringLiteral("image_a")
    });
    const QString sidecarImageB = firstString(sidecar, {
        QStringLiteral("image1_path"),
        QStringLiteral("image1_name"),
        QStringLiteral("image_b")
    });
    variant.featureAlgorithm = sidecarAlgorithm(sidecar, QStringLiteral("feature_algorithm"));
    if (variant.featureAlgorithm.isEmpty())
    {
        variant.featureAlgorithm = inferFeatureAlgorithmFromSidecar(sidecar);
    }
    variant.matchAlgorithm = sidecarAlgorithm(sidecar, QStringLiteral("match_algorithm"));
    int sidecarTotalMatches = std::max(0, variant.totalMatches);
    if (firstInt(sidecar,
                 {QStringLiteral("num_matches"),
                  QStringLiteral("match_count"),
                  QStringLiteral("total_matches")},
                 &sidecarTotalMatches))
    {
        variant.totalMatches = sidecarTotalMatches;
    }
    variant.hasInlierStats = firstInt(sidecar,
                                      {QStringLiteral("geometric_verified_inliers"),
                                       QStringLiteral("geometric_inlier_count"),
                                       QStringLiteral("geometric_inliers"),
                                       QStringLiteral("valid_inlier_count"),
                                       QStringLiteral("valid_inliers"),
                                       QStringLiteral("verified_inliers"),
                                       QStringLiteral("inlier_count"),
                                       QStringLiteral("primary_inlier_count")},
                                      &variant.geometricVerifiedInliers);

    if (sidecarImageA.isEmpty() || sidecarImageB.isEmpty())
    {
        if (variant.status == QStringLiteral("compatible"))
        {
            setIncompatible(&variant,
                            QStringLiteral("missing_image_names"),
                            QStringLiteral("sidecar_image_names_missing"));
        }
        return variant;
    }

    if (header.ok && !unorderedImageTokensMatch(header.imageA, header.imageB, sidecarImageA, sidecarImageB))
    {
        setIncompatible(&variant,
                        QStringLiteral("mismatched_image_names"),
                        QStringLiteral("sidecar_images_do_not_match_sgmt_header"));
        return variant;
    }

    variant.imageA = sidecarImageA;
    variant.imageB = sidecarImageB;

    if (!header.ok)
    {
        return variant;
    }

    variant.compatible = true;
    variant.status = QStringLiteral("compatible");
    variant.reason.clear();
    return variant;
}
bool variantContainsImageToken(const MatchVariant &variant, const QString &targetImagePath)
{
    return pairContainsImageToken(variant.imageA, variant.imageB, targetImagePath);
}
bool variantBelongsToImageSet(const MatchVariant &variant, const QStringList &targetImagePaths)
{
    return pairBelongsToImageSet(variant.imageA, variant.imageB, targetImagePaths);
}
bool variantIsBetter(const MatchVariant &candidate, const MatchVariant &current)
{
    if (candidate.geometricVerifiedInliers != current.geometricVerifiedInliers)
    {
        return candidate.geometricVerifiedInliers > current.geometricVerifiedInliers;
    }
    if (candidate.totalMatches != current.totalMatches)
    {
        return candidate.totalMatches > current.totalMatches;
    }
    if (candidate.modifiedTime != current.modifiedTime)
    {
        return candidate.modifiedTime > current.modifiedTime;
    }
    return candidate.matchFilePath < current.matchFilePath;
}
void updateBestVariant(MatchPairGroup *group)
{
    if (!group)
    {
        return;
    }

    group->bestVariantIndex = -1;
    for (int i = 0; i < group->variants.size(); ++i)
    {
        const MatchVariant &candidate = group->variants.at(i);
        if (!candidate.compatible)
        {
            continue;
        }
        if (group->bestVariantIndex < 0 ||
            variantIsBetter(candidate, group->variants.at(group->bestVariantIndex)))
        {
            group->bestVariantIndex = i;
        }
    }
}
} // namespace
MatchResultCatalog::MatchResultCatalog(const MatchResultCatalogConfig &config)
    : _config(config)
{
}
QString MatchResultCatalog::canonicalPairKey(const QString &imageA, const QString &imageB)
{
    const QString normA = normalizedPathKey(imageA);
    const QString normB = normalizedPathKey(imageB);
    if (normA.isEmpty() || normB.isEmpty() || normA == normB)
    {
        return QString();
    }
    return normA < normB
        ? normA + QStringLiteral("\n") + normB
        : normB + QStringLiteral("\n") + normA;
}
int MatchResultCatalog::readSgmtMatchCount(const QString &path)
{
    const SgmtHeader header = readSgmtHeader(path);
    return header.ok ? header.matchCount : -1;
}
QString MatchResultCatalog::algorithmDisplayLabel(const MatchVariant &variant)
{
    QString featureAlgorithm = normalizedAlgorithmToken(variant.featureAlgorithm);
    QString matchAlgorithm = normalizedAlgorithmToken(variant.matchAlgorithm);

    if (featureAlgorithm.isEmpty())
    {
        featureAlgorithm = inferFeatureAlgorithmFromMatchStem(
            QFileInfo(variant.matchFilePath).completeBaseName(),
            variant.matchAlgorithm);
    }

    if (matchAlgorithm.isEmpty())
    {
        matchAlgorithm = normalizedAlgorithmToken(QFileInfo(variant.matchFilePath).completeBaseName());
    }
    if (matchAlgorithm.isEmpty())
    {
        return QStringLiteral("unknown");
    }

    if (featureAlgorithm.isEmpty())
    {
        return QStringLiteral("unknown-") + matchAlgorithm;
    }

    if (matchAlgorithm == featureAlgorithm ||
        matchAlgorithm.startsWith(featureAlgorithm + QLatin1Char('-')))
    {
        return matchAlgorithm;
    }

    return featureAlgorithm + QLatin1Char('-') + matchAlgorithm;
}
MatchResultCatalogSummary MatchResultCatalog::scan() const
{
    MatchResultCatalogSummary summary;

    auto reportProgress = [this](int processed, int total)
    {
        if (_config.progressCallback)
        {
            _config.progressCallback(processed, total);
        }
    };

    const QDir matchDir(_config.matchDirectory);
    if (!matchDir.exists())
    {
        reportProgress(0, 0);
        return summary;
    }

    const QFileInfoList matchFiles = matchDir.entryInfoList(QStringList{QStringLiteral("*.match")},
                                                            QDir::Files,
                                                            QDir::Name);
    const int totalFileCount = matchFiles.size();
    int processedFileCount = 0;
    reportProgress(processedFileCount, totalFileCount);

    auto finishCurrentFile = [&]()
    {
        ++processedFileCount;
        reportProgress(processedFileCount, totalFileCount);
    };

    QMap<QString, int> groupIndexByKey;
    const QString targetImagePath = _config.targetImagePath.trimmed();
    QStringList targetImagePaths;
    for (const QString &imagePath : _config.targetImagePaths)
    {
        if (!imagePath.trimmed().isEmpty())
        {
            targetImagePaths.append(imagePath);
        }
    }

    for (const QFileInfo &matchInfo : matchFiles)
    {
        // 影像选择器和空三前置检查只关心局部影像集合。
        // 先读轻量 SGMT 头，避免为无关匹配解析巨大的 sidecar JSON。
        if (!targetImagePath.isEmpty() || !targetImagePaths.isEmpty())
        {
            const SgmtHeader header = readSgmtHeader(matchInfo.absoluteFilePath());
            if (header.ok && !pairContainsImageToken(header.imageA, header.imageB, targetImagePath))
            {
                finishCurrentFile();
                continue;
            }
            if (header.ok && !pairBelongsToImageSet(header.imageA, header.imageB, targetImagePaths))
            {
                finishCurrentFile();
                continue;
            }
        }

        MatchVariant variant = readVariant(matchInfo);
        if (!targetImagePath.isEmpty() && !variantContainsImageToken(variant, targetImagePath))
        {
            finishCurrentFile();
            continue;
        }
        if (!targetImagePaths.isEmpty() && !variantBelongsToImageSet(variant, targetImagePaths))
        {
            finishCurrentFile();
            continue;
        }

        ++summary.matchFileCount;
        ++summary.variantCount;
        if (variant.compatible)
        {
            ++summary.compatibleVariantCount;
        }
        else
        {
            ++summary.incompatibleVariantCount;
        }

        QString pairKey = canonicalPairKey(variant.imageA, variant.imageB);
        if (pairKey.isEmpty())
        {
            pairKey = QStringLiteral("unresolved:%1").arg(variant.matchFilePath);
        }

        int groupIndex = groupIndexByKey.value(pairKey, -1);
        if (groupIndex < 0)
        {
            MatchPairGroup group;
            group.pairKey = pairKey;
            group.imageA = variant.imageA;
            group.imageB = variant.imageB;
            summary.pairGroups.append(group);
            groupIndex = summary.pairGroups.size() - 1;
            groupIndexByKey.insert(pairKey, groupIndex);
        }

        summary.pairGroups[groupIndex].variants.append(variant);
        finishCurrentFile();
    }

    for (MatchPairGroup &group : summary.pairGroups)
    {
        updateBestVariant(&group);
    }
    summary.pairGroupCount = summary.pairGroups.size();
    return summary;
}

} // namespace xjw::aerial_triangulation
