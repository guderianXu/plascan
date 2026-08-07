#pragma once

/**
 * @file MatchResultCatalog.h
 * @brief 通过轻量 `.pidx` 对逐影像 `.pimatch` 建立像对/算法变体目录。
 *
 * 文件系统层面一幅影像只有一个分片；逻辑层面同一像对仍可在分片内部保存多个
 * algorithmId + version + configFingerprint 变体。目录器只依赖统一二进制契约，
 * 不读取旧 JSON sidecar，也不从文件名推断影像身份或算法。缺少 `.pidx` 时会
 * 从权威分片自动重建一次，之后仅校验固定大小的 payload 签名。
 */

#include <QByteArray>
#include <QDateTime>
#include <QString>
#include <QStringList>
#include <QVector>

#include <cstdint>
#include <functional>

namespace xjw::aerial_triangulation
{

/// 同一无向影像对在一个算法版本和配置下的持久化结果。
struct MatchVariant
{
    QString imageA;
    QString imageB;
    QString algorithmId;
    std::uint32_t algorithmVersion = 0;
    QByteArray configFingerprint;
    QString matchFilePath; ///< 可直接读取该像对的 owner `.pimatch` 分片。
    QString peerMatchFilePath; ///< 对称的 peer 分片，诊断缺失分片时使用。
    int totalMatches = 0;
    int geometricVerifiedInliers = 0;
    int tiePointMatches = 0;
    double geometricCoverage = 0.0; ///< 几何内点在两幅影像 4x4 网格中的平均覆盖率。
    bool hasInlierStats = true;
    bool geometryPassed = false;
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
    QString targetImagePath;
    QStringList targetImagePaths;
    /// 0 表示按硬件线程自动选择，但目录读取最多并行 8 路，避免机械盘随机争用。
    int maxConcurrency = 0;
    std::function<void(int processed, int total)> progressCallback;
};

struct MatchResultCatalogSummary
{
    int matchFileCount = 0; ///< 实际访问的逐影像分片数。
    int variantCount = 0;
    int compatibleVariantCount = 0;
    int incompatibleVariantCount = 0; ///< 文件损坏或版本不支持的分片数。
    int memoryIndexHitCount = 0;
    int persistentIndexHitCount = 0;
    int rebuiltIndexCount = 0;
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

private:
    MatchResultCatalogConfig _config;
};

} // namespace xjw::aerial_triangulation
