#include "MatchResultCatalog.h"

#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QMap>

#include <algorithm>
#include <cstring>

namespace xjw
{
namespace pipeline
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
    return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
}

QString lowerCleanToken(const QString &token)
{
    return QDir::cleanPath(token.trimmed()).toLower();
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

int firstInt(const QJsonObject &object, const QStringList &keys, int fallback)
{
    for (const QString &key : keys)
    {
        const QJsonValue value = object.value(key);
        if (value.isDouble())
        {
            return std::max(0, value.toInt());
        }
    }
    return fallback;
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
    variant.matchAlgorithm = sidecarAlgorithm(sidecar, QStringLiteral("match_algorithm"));
    variant.totalMatches = firstInt(sidecar,
                                    {QStringLiteral("num_matches"),
                                     QStringLiteral("match_count"),
                                     QStringLiteral("total_matches")},
                                    std::max(0, variant.totalMatches));
    variant.geometricVerifiedInliers = firstInt(sidecar,
                                                {QStringLiteral("geometric_verified_inliers"),
                                                 QStringLiteral("geometric_inlier_count"),
                                                 QStringLiteral("geometric_inliers"),
                                                 QStringLiteral("valid_inlier_count"),
                                                 QStringLiteral("valid_inliers"),
                                                 QStringLiteral("verified_inliers"),
                                                 QStringLiteral("inlier_count"),
                                                 QStringLiteral("primary_inlier_count")},
                                                0);

    if (sidecarImageA.isEmpty() || sidecarImageB.isEmpty())
    {
        setIncompatible(&variant,
                        QStringLiteral("missing_image_names"),
                        QStringLiteral("sidecar_image_names_missing"));
        return variant;
    }

    if (header.ok && !unorderedImageTokensMatch(header.imageA, header.imageB, sidecarImageA, sidecarImageB))
    {
        setIncompatible(&variant,
                        QStringLiteral("mismatched_image_names"),
                        QStringLiteral("sidecar_images_do_not_match_sgmt_header"));
        return variant;
    }

    if (!header.ok)
    {
        return variant;
    }

    variant.imageA = sidecarImageA;
    variant.imageB = sidecarImageB;
    variant.compatible = true;
    variant.status = QStringLiteral("compatible");
    variant.reason.clear();
    return variant;
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

MatchResultCatalogSummary MatchResultCatalog::scan() const
{
    MatchResultCatalogSummary summary;

    const QDir matchDir(_config.matchDirectory);
    if (!matchDir.exists())
    {
        return summary;
    }

    const QFileInfoList matchFiles = matchDir.entryInfoList(QStringList{QStringLiteral("*.match")},
                                                            QDir::Files,
                                                            QDir::Name);
    QMap<QString, int> groupIndexByKey;

    for (const QFileInfo &matchInfo : matchFiles)
    {
        ++summary.matchFileCount;
        MatchVariant variant = readVariant(matchInfo);
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
    }

    for (MatchPairGroup &group : summary.pairGroups)
    {
        updateBestVariant(&group);
    }
    summary.pairGroupCount = summary.pairGroups.size();
    return summary;
}

} // namespace pipeline
} // namespace xjw
