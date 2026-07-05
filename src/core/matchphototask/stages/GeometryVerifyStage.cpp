#include "GeometryVerifyStage.h"

// Avoid Qt keyword macros rewriting LibTorch's slots() member name.
#ifdef slots
#undef slots
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#endif
#ifdef signals
#undef signals
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#endif
#ifdef emit
#undef emit
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#endif

#include "FeatureData.h"

#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#define slots Q_SLOTS
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#endif
#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#define signals Q_SIGNALS
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#endif
#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#define emit Q_EMIT
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#endif

#include "FeatureFileIO.h"
#include "MatchFileIO.h"
#include "MatchGeometryFilter.h"
#include "MatchPhotosRuntime.h"

namespace xjw
{
namespace matchphotos
{
namespace
{

MatchPhotosStageReport makeGeometryReport(MatchPhotosStageStatus status,
                                          const QString &message,
                                          int itemCount = 0)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("geometry_verify");
    report.displayName = QStringLiteral("几何验证");
    report.status = status;
    report.message = message;
    report.itemCount = itemCount;
    return report;
}

void normalizeMatchResult(xjw::feature_match::MatchResult *matchResult,
                          int keypointCount0,
                          int keypointCount1)
{
    if (!matchResult)
    {
        return;
    }
    if (matchResult->matches0.empty() && !matchResult->cvMatches.empty())
    {
        matchResult->buildIndicesFromCvMatches(keypointCount0, keypointCount1);
    }
    if (matchResult->cvMatches.empty() && !matchResult->matches0.empty())
    {
        matchResult->buildCvMatchesFromIndices();
    }
    matchResult->numMatches = static_cast<int>(matchResult->cvMatches.size());
}

} // namespace

MatchPhotosStageReport GeometryVerifyStage::run(const MatchPhotosContext &context,
                                                const MatchPhotosOptions &options,
                                                std::vector<MatchPhotosMatchRecord> *matchRecords) const
{
    if (options.planOnly)
    {
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("plan-only 模式，跳过几何验证"));
    }
    if (!options.enableGeometryVerification)
    {
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("几何验证已禁用"));
    }
    if (!matchRecords || matchRecords->empty())
    {
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("没有可用于几何验证的匹配结果"));
    }

    xjw::feature_match::OutlierFilterConfig filterConfig;
    filterConfig.method = xjw::feature_match::OutlierMethod::FundamentalUsacMagsac;
    filterConfig.reprojThreshold = options.geometryReprojThreshold;
    filterConfig.minInliers = options.geometryMinInliers;

    int passedPairs = 0;
    int failedPairs = 0;
    int totalInliers = 0;
    for (MatchPhotosMatchRecord &record : *matchRecords)
    {
        if (shouldCancelMatchPhotos(context))
        {
            return makeGeometryReport(MatchPhotosStageStatus::Failed,
                                      QStringLiteral("用户取消几何验证"),
                                      passedPairs);
        }

        const QString feature0Path = record.settings.value(QStringLiteral("feature0_path")).toString();
        const QString feature1Path = record.settings.value(QStringLiteral("feature1_path")).toString();
        QString image0Name;
        QString image1Name;
        xjw::feature_extractors::FeatureData feature0;
        xjw::feature_extractors::FeatureData feature1;
        xjw::feature_match::MatchResult rawMatch;
        if (!FeatureFileIO::readData(feature0Path, image0Name, feature0) ||
            !FeatureFileIO::readData(feature1Path, image1Name, feature1) ||
            !xjw::feature_match::readIndexedMatchFile(record.matchPath, image0Name, image1Name, rawMatch))
        {
            ++failedPairs;
            continue;
        }

        normalizeMatchResult(&rawMatch, feature0.size(), feature1.size());
        int inlierCount = 0;
        xjw::feature_match::MatchResult filteredMatch = xjw::feature_match::MatchGeometryFilter::filter(
            rawMatch,
            feature0.keypoints,
            feature1.keypoints,
            filterConfig,
            &inlierCount);
        normalizeMatchResult(&filteredMatch, feature0.size(), feature1.size());

        record.geometricInlierCount = inlierCount;
        record.passedGeometry = inlierCount >= options.geometryMinInliers &&
            filteredMatch.numMatches >= options.geometryMinInliers;
        record.inlierIndexPairs.clear();
        if (record.passedGeometry)
        {
            record.inlierIndexPairs.reserve(filteredMatch.cvMatches.size());
            for (const cv::DMatch &match : filteredMatch.cvMatches)
            {
                record.inlierIndexPairs.push_back({match.queryIdx, match.trainIdx});
            }
            ++passedPairs;
            totalInliers += static_cast<int>(record.inlierIndexPairs.size());
        }
        else
        {
            ++failedPairs;
        }

        record.settings[QStringLiteral("geometry_verified")] = record.passedGeometry;
        record.settings[QStringLiteral("geometric_inliers")] = record.geometricInlierCount;
        record.settings[QStringLiteral("geometry_reproj_threshold")] = options.geometryReprojThreshold;
    }

    return makeGeometryReport(MatchPhotosStageStatus::Completed,
                              QStringLiteral("几何验证完成：通过 %1 对，失败 %2 对，内点 %3")
                                  .arg(passedPairs)
                                  .arg(failedPairs)
                                  .arg(totalInliers),
                              passedPairs);
}

} // namespace matchphotos
} // namespace xjw
