#include "SiftLightGlueRecovery.h"

// 避免 Qt 关键字宏改写 LibTorch 头文件中的 slots()/signals() 成员。
#ifdef slots
#undef slots
#define PLASCAN_SIFT_RECOVERY_RESTORE_QT_SLOTS
#endif
#ifdef signals
#undef signals
#define PLASCAN_SIFT_RECOVERY_RESTORE_QT_SIGNALS
#endif
#ifdef emit
#undef emit
#define PLASCAN_SIFT_RECOVERY_RESTORE_QT_EMIT
#endif

#include "FeatureData.h"

#ifdef PLASCAN_SIFT_RECOVERY_RESTORE_QT_SLOTS
#define slots Q_SLOTS
#undef PLASCAN_SIFT_RECOVERY_RESTORE_QT_SLOTS
#endif
#ifdef PLASCAN_SIFT_RECOVERY_RESTORE_QT_SIGNALS
#define signals Q_SIGNALS
#undef PLASCAN_SIFT_RECOVERY_RESTORE_QT_SIGNALS
#endif
#ifdef PLASCAN_SIFT_RECOVERY_RESTORE_QT_EMIT
#define emit Q_EMIT
#undef PLASCAN_SIFT_RECOVERY_RESTORE_QT_EMIT
#endif

#include "MatchFileIO.h"
#include "MatchPhotosRuntime.h"
#include "TraditionalFeatureMatcher.h"

#include <QFileInfo>
#include <QHash>

#include <algorithm>
#include <exception>

namespace xjw::matchphotos
{
namespace
{

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

bool isSiftLightGlueRecord(const MatchPhotosMatchRecord &record)
{
    return record.settings.value(QStringLiteral("feature_algorithm")).toString().trimmed().toLower() ==
            QStringLiteral("sift") &&
        record.settings.value(QStringLiteral("match_algorithm")).toString().trimmed().toLower() ==
            QStringLiteral("lightglue");
}

xjw::feature_match::MatchResult filterGeometry(
    const xjw::feature_match::MatchResult &rawMatch,
    const xjw::feature_extractors::FeatureData &feature0,
    const xjw::feature_extractors::FeatureData &feature1,
    const xjw::feature_match::OutlierFilterConfig &filterConfig)
{
    int inlierCount = 0;
    xjw::feature_match::MatchResult filtered = xjw::feature_match::MatchGeometryFilter::filter(
        rawMatch, feature0.keypoints, feature1.keypoints, filterConfig, &inlierCount);
    normalizeMatchResult(&filtered, feature0.size(), feature1.size());
    return filtered;
}

class ImageDisjointSet
{
public:
    void add(const QString &image)
    {
        if (!image.isEmpty() && !_parent.contains(image))
        {
            _parent.insert(image, image);
        }
    }

    QString root(const QString &image)
    {
        add(image);
        QString current = image;
        while (_parent.value(current) != current)
        {
            current = _parent.value(current);
        }
        QString node = image;
        while (_parent.value(node) != node)
        {
            const QString next = _parent.value(node);
            _parent[node] = current;
            node = next;
        }
        return current;
    }

    void unite(const QString &left, const QString &right)
    {
        const QString leftRoot = root(left);
        const QString rightRoot = root(right);
        if (leftRoot != rightRoot)
        {
            _parent[rightRoot] = leftRoot;
        }
    }

private:
    QHash<QString, QString> _parent;
};

} // namespace

bool shouldAugmentDenseSiftPair(const MatchPhotosOptions &options,
                                const MatchPhotosMatchRecord &record,
                                int rawMatchCount)
{
    if ((options.profile != MatchPhotosProfile::HighAccuracy &&
         options.profile != MatchPhotosProfile::DifficultTexture) ||
        !isSiftLightGlueRecord(record))
    {
        return false;
    }

    const bool limited = record.settings.value(QStringLiteral("lightglue_limited_keypoints0")).toBool() ||
        record.settings.value(QStringLiteral("lightglue_limited_keypoints1")).toBool() ||
        record.settings.value(QStringLiteral("lightglue_limited_keypoints")).toBool();
    const int budget = record.settings.value(QStringLiteral("lightglue_keypoint_budget")).toInt();
    if (!limited || budget <= 0)
    {
        return false;
    }

    // 只增强 LightGlue 已证明存在强重叠、但有效输入被显存预算截断的像对。
    const int strongOverlapThreshold = std::max(options.geometryMinInliers * 4, budget / 4);
    return rawMatchCount >= strongOverlapThreshold;
}

bool runFullSiftRecovery(const MatchPhotosOptions &options,
                         const xjw::feature_extractors::FeatureData &feature0,
                         const xjw::feature_extractors::FeatureData &feature1,
                         const xjw::feature_match::OutlierFilterConfig &filterConfig,
                         xjw::feature_match::MatchResult *rawRecovered,
                         xjw::feature_match::MatchResult *filteredRecovered,
                         bool *usedCuda)
{
    if (!rawRecovered || !filteredRecovered)
    {
        return false;
    }

    xjw::feature_match::tradition::TraditionalMatchConfig matchConfig;
    matchConfig.algorithmName = "sift_bf_l2";
    matchConfig.ratioTestThreshold = 0.75f;
    matchConfig.requireMutualConsistency = true;
    matchConfig.cudaDevice = options.cudaDevice;
    matchConfig.useCuda = options.device != ComputeDevice::Cpu;

    auto execute = [&]()
    {
        *rawRecovered = xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
            feature0.toCvDescriptors(matchConfig.algorithmName),
            feature1.toCvDescriptors(matchConfig.algorithmName),
            feature0.size(),
            feature1.size(),
            matchConfig);
    };

    try
    {
        execute();
    }
    catch (const std::exception &)
    {
        if (!matchConfig.useCuda)
        {
            return false;
        }
        // CUDA 全量描述子匹配内存不足时，仅该恢复像对改用 CPU。
        matchConfig.useCuda = false;
        try
        {
            execute();
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    normalizeMatchResult(rawRecovered, feature0.size(), feature1.size());
    *filteredRecovered = filterGeometry(*rawRecovered, feature0, feature1, filterConfig);
    if (usedCuda)
    {
        *usedCuda = matchConfig.useCuda;
    }
    return true;
}

bool persistRecoveredMatch(const MatchPhotosOptions &options,
                           const MatchPhotosAlgorithmPlan &plan,
                           const xjw::feature_extractors::FeatureData &feature0,
                           const xjw::feature_extractors::FeatureData &feature1,
                           const xjw::feature_match::MatchResult &rawRecovered,
                           const QString &reason,
                           int primaryRawCount,
                           int primaryInlierCount,
                           int recoveredInlierCount,
                           bool usedCuda,
                           MatchPhotosMatchRecord *record)
{
    if (!record)
    {
        return false;
    }

    const QString feature0Path = record->settings.value(QStringLiteral("feature0_path")).toString();
    const QString feature1Path = record->settings.value(QStringLiteral("feature1_path")).toString();
    ResolvedImagePair pair;
    pair.image0Path = record->image0Path;
    pair.image1Path = record->image1Path;
    pair.pairName = record->settings.value(QStringLiteral("pair_name")).toString();
    pair.pairKey = pair.image0Path + QLatin1Char('\n') + pair.image1Path;

    QJsonObject settings = record->settings;
    settings[QStringLiteral("traditional_sift_recovery")] = true;
    settings[QStringLiteral("traditional_sift_recovery_reason")] = reason;
    settings[QStringLiteral("traditional_sift_recovery_cuda")] = usedCuda;
    settings[QStringLiteral("primary_lightglue_raw_matches")] = primaryRawCount;
    settings[QStringLiteral("primary_lightglue_geometric_inliers")] = primaryInlierCount;
    settings[QStringLiteral("fallback_raw_match_count")] = rawRecovered.numMatches;
    settings[QStringLiteral("num_matches")] = rawRecovered.numMatches;
    settings[QStringLiteral("geometry_verified")] = true;
    settings[QStringLiteral("geometric_inliers")] = recoveredInlierCount;

    const QString sidecarPath = record->sidecarPath.isEmpty()
        ? record->matchPath + QStringLiteral(".json")
        : record->sidecarPath;
    if (!xjw::feature_match::writeIndexedMatchFile(record->matchPath,
                                                   QFileInfo(pair.image0Path).completeBaseName(),
                                                   QFileInfo(pair.image1Path).completeBaseName(),
                                                   rawRecovered) ||
        !writeMatchPhotosSidecar(sidecarPath,
                                 pair,
                                 feature0Path,
                                 feature1Path,
                                 record->matchPath,
                                 feature0,
                                 feature1,
                                 rawRecovered,
                                 plan,
                                 options,
                                 settings))
    {
        return false;
    }

    record->sidecarPath = sidecarPath;
    record->matchCount = rawRecovered.numMatches;
    record->settings = settings;
    return true;
}

std::vector<int> disconnectedRecoveryCandidates(
    const std::vector<MatchPhotosMatchRecord> &records,
    const QSet<int> &attempted)
{
    ImageDisjointSet components;
    for (const MatchPhotosMatchRecord &record : records)
    {
        components.add(record.image0Path);
        components.add(record.image1Path);
        if (record.passedGeometry)
        {
            components.unite(record.image0Path, record.image1Path);
        }
    }

    std::vector<int> candidates;
    for (int index = 0; index < static_cast<int>(records.size()); ++index)
    {
        const MatchPhotosMatchRecord &record = records[static_cast<std::size_t>(index)];
        if (record.passedGeometry || attempted.contains(index) || !isSiftLightGlueRecord(record))
        {
            continue;
        }
        if (components.root(record.image0Path) != components.root(record.image1Path))
        {
            candidates.push_back(index);
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [&](int left, int right)
    {
        return records[static_cast<std::size_t>(left)].matchCount >
            records[static_cast<std::size_t>(right)].matchCount;
    });
    return candidates;
}

} // namespace xjw::matchphotos
