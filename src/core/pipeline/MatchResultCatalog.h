#pragma once

#include <QDateTime>
#include <QString>
#include <QVector>

namespace xjw
{
namespace pipeline
{

struct MatchVariant
{
    QString imageA;
    QString imageB;
    QString featureAlgorithm;
    QString matchAlgorithm;
    QString matchFilePath;
    QString sidecarPath;
    int totalMatches = 0;
    int geometricVerifiedInliers = 0;
    bool hasInlierStats = false;
    bool compatible = false;
    QString status;
    QString reason;
    QDateTime modifiedTime;
};

struct MatchPairGroup
{
    QString pairKey;
    QString imageA;
    QString imageB;
    QVector<MatchVariant> variants;
    int bestVariantIndex = -1;
};

struct MatchResultCatalogConfig
{
    QString matchDirectory;
};

struct MatchResultCatalogSummary
{
    int matchFileCount = 0;
    int variantCount = 0;
    int compatibleVariantCount = 0;
    int incompatibleVariantCount = 0;
    int pairGroupCount = 0;
    QVector<MatchPairGroup> pairGroups;
};

class MatchResultCatalog
{
public:
    explicit MatchResultCatalog(const MatchResultCatalogConfig &config = MatchResultCatalogConfig());

    MatchResultCatalogSummary scan() const;

    static QString canonicalPairKey(const QString &imageA, const QString &imageB);
    static QString algorithmDisplayLabel(const MatchVariant &variant);
    static int readSgmtMatchCount(const QString &path);

private:
    MatchResultCatalogConfig _config;
};

} // namespace pipeline
} // namespace xjw
