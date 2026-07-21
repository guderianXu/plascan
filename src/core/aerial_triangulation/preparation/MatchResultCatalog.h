#pragma once

#include <QDateTime>
#include <QString>
#include <QStringList>
#include <QVector>

#include <functional>

namespace xjw::aerial_triangulation
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
    // 仅构建包含该影像的匹配目录索引；为空时扫描全部匹配结果。
    QString targetImagePath;
    // 仅构建两端都属于当前影像集合的匹配目录索引；空三前置检查用于跳过历史工程残留匹配。
    QStringList targetImagePaths;
    // 扫描 .match 文件时上报 processed/total，用于 GUI 展示真实百分比。
    std::function<void(int processed, int total)> progressCallback;
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

} // namespace xjw::aerial_triangulation
