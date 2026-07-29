#include "cli_common.h"

#include "preparation/MatchResultCatalog.h"
#include "GeometryVerifyStage.h"
#include "StoredPairGeometryAudit.h"

#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>

#include <algorithm>
#include <cstdio>
#include <string>

namespace
{

QStringList manifestImages(const QString &manifestPath, QString *error)
{
    QFile file(manifestPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (error)
        {
            *error = QStringLiteral("无法读取 MVS manifest：%1").arg(manifestPath);
        }
        return {};
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject())
    {
        if (error)
        {
            *error = QStringLiteral("MVS manifest 不是有效 JSON 对象：%1")
                         .arg(manifestPath);
        }
        return {};
    }

    QStringList images;
    for (const QJsonValue &value :
         document.object().value(QStringLiteral("frames")).toArray())
    {
        const QString image =
            value.toObject().value(QStringLiteral("ref_image")).toString().trimmed();
        if (!image.isEmpty() && !images.contains(image, Qt::CaseInsensitive))
        {
            images.append(image);
        }
    }
    if (images.size() < 2 && error)
    {
        *error = QStringLiteral("MVS manifest 中当前参考影像少于 2 张：%1")
                     .arg(manifestPath);
    }
    return images;
}

QString verificationStatus(bool statisticsAvailable, bool verified)
{
    if (!statisticsAvailable)
    {
        return QStringLiteral("missing_statistics");
    }
    return verified ? QStringLiteral("verified") : QStringLiteral("failed");
}

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication qt_app(argc, argv);
    CLI::App app{"PlaScan MVS 匹配对几何审计"};
    std::string match_dir;
    std::string manifest_path;
    std::string output_path;
    int minimum_inliers = 20;
    double reprojection_threshold = 1.5;
    app.add_option("--match-dir", match_dir, "assets/matches 目录")->required();
    app.add_option("--mvs-manifest", manifest_path, "用于限定当前影像集合的 mvs_manifest.json")
        ->required();
    app.add_option("-o,--output", output_path, "可选 JSON 输出路径");
    app.add_option("--minimum-inliers", minimum_inliers, "几何验证最少内点")
        ->check(CLI::Range(8, 1000));
    app.add_option("--reprojection-threshold", reprojection_threshold,
                   "USAC/MAGSAC 重投影阈值（像素）")
        ->check(CLI::Range(0.1, 10.0));
    CLI11_PARSE(app, argc, argv);

    QString error;
    const QStringList images = manifestImages(
        QString::fromUtf8(manifest_path.c_str()), &error);
    if (!error.isEmpty())
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    xjw::aerial_triangulation::MatchResultCatalogConfig config;
    config.matchDirectory = QString::fromUtf8(match_dir.c_str());
    config.targetImagePaths = images;
    const xjw::aerial_triangulation::MatchResultCatalogSummary catalog =
        xjw::aerial_triangulation::MatchResultCatalog(config).scan();

    int verified_count = 0;
    int failed_count = 0;
    int missing_count = 0;
    QJsonArray pairs;
    for (const xjw::aerial_triangulation::MatchPairGroup &group :
         catalog.pairGroups)
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

        int total_matches = std::max(0, variant.totalMatches);
        int geometric_inliers =
            std::max(0, variant.geometricVerifiedInliers);
        bool statistics_available =
            variant.hasInlierStats && total_matches >= minimum_inliers;
        bool verified = statistics_available &&
            xjw::matchphotos::passesGeometryQualityGate(
                total_matches,
                geometric_inliers,
                minimum_inliers);
        double coverage_score = 0.0;
        QString reason = verified
            ? QStringLiteral("verified_from_sidecar")
            : (statistics_available
                   ? QStringLiteral("sidecar_geometry_gate_failed")
                   : QStringLiteral("missing_geometric_inlier_statistics"));
        if (!statistics_available)
        {
            const xjw::matchphotos::StoredPairGeometryAuditResult audit =
                xjw::matchphotos::auditStoredPairGeometry(
                    variant.matchFilePath,
                    variant.sidecarPath,
                    minimum_inliers,
                    reprojection_threshold);
            statistics_available = audit.statisticsAvailable;
            verified = audit.verified;
            total_matches = audit.statisticsAvailable
                ? audit.totalMatches
                : total_matches;
            geometric_inliers = audit.geometricInliers;
            coverage_score = audit.coverageScore;
            reason = audit.reason;
        }

        if (verified)
        {
            ++verified_count;
        }
        else if (statistics_available)
        {
            ++failed_count;
        }
        else
        {
            ++missing_count;
        }
        pairs.append(QJsonObject{
            {QStringLiteral("image_a"), variant.imageA},
            {QStringLiteral("image_b"), variant.imageB},
            {QStringLiteral("match_path"), variant.matchFilePath},
            {QStringLiteral("status"),
             verificationStatus(statistics_available, verified)},
            {QStringLiteral("total_matches"), total_matches},
            {QStringLiteral("geometric_inliers"), geometric_inliers},
            {QStringLiteral("inlier_ratio"),
             total_matches > 0
                 ? static_cast<double>(geometric_inliers) /
                       static_cast<double>(total_matches)
                 : 0.0},
            {QStringLiteral("coverage_score"), coverage_score},
            {QStringLiteral("reason"), reason}
        });
    }

    const QJsonObject root{
        {QStringLiteral("schema"), QStringLiteral("plascan_mvs_pair_audit_v1")},
        {QStringLiteral("current_image_count"), images.size()},
        {QStringLiteral("catalog_pair_count"), catalog.pairGroupCount},
        {QStringLiteral("audited_pair_count"), pairs.size()},
        {QStringLiteral("verified_pair_count"), verified_count},
        {QStringLiteral("failed_pair_count"), failed_count},
        {QStringLiteral("missing_statistics_pair_count"), missing_count},
        {QStringLiteral("pairs"), pairs}
    };
    const QByteArray json = QJsonDocument(root).toJson(QJsonDocument::Indented);
    if (!output_path.empty())
    {
        QSaveFile output(QString::fromUtf8(output_path.c_str()));
        if (!output.open(QIODevice::WriteOnly) ||
            output.write(json) != json.size() ||
            !output.commit())
        {
            std::fprintf(stderr, "无法写入审计报告：%s\n", output_path.c_str());
            return cli::EXIT_IO_ERR;
        }
    }
    std::fwrite(json.constData(), 1, static_cast<std::size_t>(json.size()), stdout);
    return cli::EXIT_OK;
}
