#include "StoredPairGeometryAudit.h"

// Avoid Qt keyword macros rewriting LibTorch's slots() member name.
#ifdef slots
#undef slots
#define PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_SLOTS
#endif
#ifdef signals
#undef signals
#define PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_SIGNALS
#endif
#ifdef emit
#undef emit
#define PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_EMIT
#endif

#include "FeatureData.h"

#ifdef PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_SLOTS
#define slots Q_SLOTS
#undef PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_SLOTS
#endif
#ifdef PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_SIGNALS
#define signals Q_SIGNALS
#undef PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_SIGNALS
#endif
#ifdef PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_EMIT
#define emit Q_EMIT
#undef PLASCAN_STORED_PAIR_AUDIT_RESTORE_QT_EMIT
#endif

#include "FeatureFileIO.h"
#include "GeometryVerifyStage.h"
#include "MatchFileIO.h"
#include "MatchGeometryFilter.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>

#include <algorithm>
#include <array>
#include <cmath>

namespace xjw::matchphotos
{
namespace
{

QString firstPath(const QJsonObject &sidecar, const QStringList &keys)
{
    const QJsonObject settings = sidecar.value(QStringLiteral("settings")).toObject();
    for (const QString &key : keys)
    {
        QString value = sidecar.value(key).toString().trimmed();
        if (value.isEmpty())
        {
            value = settings.value(key).toString().trimmed();
        }
        if (!value.isEmpty())
        {
            return value;
        }
    }
    return QString();
}

QString resolveSidecarPath(const QString &path, const QString &sidecarPath)
{
    const QFileInfo info(path);
    if (info.isAbsolute())
    {
        return QDir::cleanPath(info.absoluteFilePath());
    }
    return QDir::cleanPath(
        QFileInfo(sidecarPath).absoluteDir().absoluteFilePath(path));
}

double gridCoverage(const std::vector<cv::DMatch> &matches,
                    const std::vector<cv::KeyPoint> &keypoints,
                    bool querySide,
                    int imageWidth,
                    int imageHeight)
{
    constexpr int gridColumns = 4;
    constexpr int gridRows = 4;
    if (matches.empty() || keypoints.empty())
    {
        return 0.0;
    }

    float maximum_x = static_cast<float>(std::max(1, imageWidth));
    float maximum_y = static_cast<float>(std::max(1, imageHeight));
    if (imageWidth <= 0 || imageHeight <= 0)
    {
        for (const cv::KeyPoint &keypoint : keypoints)
        {
            maximum_x = std::max(maximum_x, keypoint.pt.x + 1.0f);
            maximum_y = std::max(maximum_y, keypoint.pt.y + 1.0f);
        }
    }

    std::array<bool, gridColumns * gridRows> occupied{};
    for (const cv::DMatch &match : matches)
    {
        const int index = querySide ? match.queryIdx : match.trainIdx;
        if (index < 0 || index >= static_cast<int>(keypoints.size()))
        {
            continue;
        }
        const cv::Point2f point = keypoints[static_cast<std::size_t>(index)].pt;
        const int column = std::clamp(
            static_cast<int>(std::floor(point.x / maximum_x * gridColumns)),
            0,
            gridColumns - 1);
        const int row = std::clamp(
            static_cast<int>(std::floor(point.y / maximum_y * gridRows)),
            0,
            gridRows - 1);
        occupied[static_cast<std::size_t>(row * gridColumns + column)] = true;
    }
    return static_cast<double>(std::count(
               occupied.cbegin(), occupied.cend(), true)) /
        static_cast<double>(occupied.size());
}

} // namespace

StoredPairGeometryAuditResult auditStoredPairGeometry(
    const QString &matchPath,
    const QString &sidecarPath,
    int minimumInliers,
    double reprojectionThreshold)
{
    StoredPairGeometryAuditResult result;
    QFile sidecar_file(sidecarPath);
    if (!sidecar_file.open(QIODevice::ReadOnly))
    {
        result.reason = QStringLiteral("sidecar_unreadable");
        return result;
    }
    const QJsonDocument document = QJsonDocument::fromJson(sidecar_file.readAll());
    if (!document.isObject())
    {
        result.reason = QStringLiteral("sidecar_parse_failed");
        return result;
    }
    const QJsonObject sidecar = document.object();
    const QString feature0_path = resolveSidecarPath(
        firstPath(sidecar, {
            QStringLiteral("feature0_path"),
            QStringLiteral("sp0_path")
        }),
        sidecarPath);
    const QString feature1_path = resolveSidecarPath(
        firstPath(sidecar, {
            QStringLiteral("feature1_path"),
            QStringLiteral("sp1_path")
        }),
        sidecarPath);
    if (feature0_path.isEmpty() || feature1_path.isEmpty())
    {
        result.reason = QStringLiteral("feature_paths_missing");
        return result;
    }

    QString feature_image0;
    QString feature_image1;
    QString match_image0;
    QString match_image1;
    xjw::feature_extractors::FeatureData feature0;
    xjw::feature_extractors::FeatureData feature1;
    xjw::feature_match::MatchResult raw_match;
    if (!FeatureFileIO::readData(feature0_path, feature_image0, feature0) ||
        !FeatureFileIO::readData(feature1_path, feature_image1, feature1))
    {
        result.reason = QStringLiteral("feature_data_unreadable");
        return result;
    }
    if (!xjw::feature_match::readIndexedMatchFile(
            matchPath, match_image0, match_image1, raw_match))
    {
        result.reason = QStringLiteral("indexed_match_unreadable");
        return result;
    }

    if (raw_match.cvMatches.empty() && !raw_match.matches0.empty())
    {
        raw_match.buildCvMatchesFromIndices();
    }
    raw_match.numMatches = static_cast<int>(raw_match.cvMatches.size());
    result.totalMatches = raw_match.numMatches;
    if (result.totalMatches <= 0)
    {
        // An empty stored pair contains no direct two-view evidence.  It is
        // not a failed geometric model: downstream MVS may still have strong
        // shared-track evidence from the global SfM graph.  Keep statistics
        // unavailable so verified-first planning can use that independent
        // evidence as a low-trust backfill, while non-empty pairs that fail
        // USAC remain hard failures.
        result.reason = QStringLiteral("stored_match_evidence_absent");
        return result;
    }
    if (result.totalMatches < std::max(1, minimumInliers))
    {
        // Fewer correspondences than the production gate requires cannot
        // establish either a verified model or a reliable negative result.
        // Treat this as insufficient direct evidence, not as proof that a
        // source view is geometrically bad.
        result.reason = QStringLiteral("stored_match_evidence_insufficient");
        return result;
    }

    xjw::feature_match::OutlierFilterConfig filter_config;
    filter_config.method =
        xjw::feature_match::OutlierMethod::FundamentalUsacMagsac;
    filter_config.reprojThreshold = std::max(0.1, reprojectionThreshold);
    filter_config.minInliers = std::max(1, minimumInliers);
    filter_config.randomSeed = 0;
    int inlier_count = 0;
    const xjw::feature_match::MatchResult filtered =
        xjw::feature_match::MatchGeometryFilter::filter(
            raw_match,
            feature0.keypoints,
            feature1.keypoints,
            filter_config,
            &inlier_count);
    result.statisticsAvailable = true;
    result.geometricInliers = std::max(
        0, std::max(inlier_count, filtered.numMatches));
    result.inlierRatio = result.totalMatches > 0
        ? static_cast<double>(result.geometricInliers) /
              static_cast<double>(result.totalMatches)
        : 0.0;
    result.coverageScore = 0.5 * (
        gridCoverage(filtered.cvMatches,
                     feature0.keypoints,
                     true,
                     feature0.imageWidth,
                     feature0.imageHeight) +
        gridCoverage(filtered.cvMatches,
                     feature1.keypoints,
                     false,
                     feature1.imageWidth,
                     feature1.imageHeight));
    result.verified = passesGeometryQualityGate(
        result.totalMatches,
        result.geometricInliers,
        filter_config.minInliers);
    result.reason = result.verified
        ? QStringLiteral("verified_from_stored_matches")
        : QStringLiteral("stored_match_geometry_gate_failed");
    return result;
}

} // namespace xjw::matchphotos
