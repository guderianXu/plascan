#include "MvsSourcePairQualityLoader.h"

#include "preparation/MatchResultCatalog.h"
#include "io/PathIO.h"

#include <algorithm>
#include <utility>

namespace xjw::core::project
{

MvsSourcePairQualityLoadResult loadMvsSourcePairQualities(
    const QString &matchDirectory,
    const QStringList &targetImagePaths)
{
    MvsSourcePairQualityLoadResult result;
    if (matchDirectory.trimmed().isEmpty())
    {
        return result;
    }

    // `.pimatch` 是匹配结果的唯一权威来源。直接扫描轻量索引可避免旧 JSON
    // 报告落后于当前缓存，也不会把对称分片中的同一逻辑像对重复加入规划器。
    xjw::aerial_triangulation::MatchResultCatalogConfig config;
    config.matchDirectory = matchDirectory;
    config.targetImagePaths = targetImagePaths;
    const xjw::aerial_triangulation::MatchResultCatalogSummary summary =
        xjw::aerial_triangulation::MatchResultCatalog(config).scan();

    result.matchFileCount = summary.matchFileCount;
    result.catalogPairCount = summary.pairGroupCount;
    result.incompatibleVariantCount = summary.incompatibleVariantCount;
    result.qualities.reserve(static_cast<std::size_t>(summary.pairGroups.size()));
    for (const xjw::aerial_triangulation::MatchPairGroup &group :
         summary.pairGroups)
    {
        if (group.bestVariantIndex < 0 ||
            group.bestVariantIndex >= group.variants.size())
        {
            continue;
        }

        const xjw::aerial_triangulation::MatchVariant &variant =
            group.variants.at(group.bestVariantIndex);
        if (!variant.compatible)
        {
            continue;
        }

        xjw::mvs::MvsSourcePairQuality quality;
        quality.imageA = xjw::common::io::toUtf8Path(variant.imageA);
        quality.imageB = xjw::common::io::toUtf8Path(variant.imageB);
        quality.totalMatches = std::max(0, variant.totalMatches);
        quality.geometricInliers = std::max(0, variant.geometricVerifiedInliers);
        quality.hasVerificationStatistics = variant.hasInlierStats;
        quality.verified = variant.hasInlierStats &&
            variant.geometryPassed && quality.geometricInliers > 0;
        quality.geometricCoverage = static_cast<float>(variant.geometricCoverage);
        if (quality.verified)
        {
            quality.verificationReason = "verified_from_pimatch";
            ++result.verifiedPairCount;
        }
        else if (!quality.hasVerificationStatistics)
        {
            quality.verificationReason = "missing_geometric_inlier_statistics";
            ++result.missingStatisticsPairCount;
        }
        else
        {
            quality.verificationReason = "pimatch_geometry_gate_failed";
            ++result.failedPairCount;
        }
        result.qualities.push_back(std::move(quality));
    }
    return result;
}

void applyMvsSourcePairQualities(
    xjw::mvs::DepthGenConfig *config,
    MvsSourcePairQualityLoadResult result)
{
    if (!config)
    {
        return;
    }

    config->requireVerifiedSourcePairs = result.verifiedPairCount > 0;
    config->sourcePairQualities = std::move(result.qualities);
}

} // namespace xjw::core::project
