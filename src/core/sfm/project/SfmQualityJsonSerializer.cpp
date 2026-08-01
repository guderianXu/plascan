#include "project/SfmQualityJsonSerializer.h"

#include <QJsonArray>
#include <QString>

#include <cmath>

namespace xjw
{

namespace
{

double roundMetric(double value)
{
    // 报告保留 6 位小数，减少重复保存造成的 JSON 噪声；核心指标仍保留 double。
    return std::round(value * 1000000.0) / 1000000.0;
}

QJsonObject serializeNumericSummary(const SfmNumericSummary &summary, const QString &unit)
{
    QJsonObject object;
    object.insert(QStringLiteral("count"), summary.count);
    object.insert(QStringLiteral("min"), roundMetric(summary.min));
    object.insert(QStringLiteral("max"), roundMetric(summary.max));
    object.insert(QStringLiteral("mean"), roundMetric(summary.mean));
    object.insert(QStringLiteral("p50"), roundMetric(summary.p50));
    object.insert(QStringLiteral("p84"), roundMetric(summary.p84));
    object.insert(QStringLiteral("p95"), roundMetric(summary.p95));
    object.insert(QStringLiteral("unit"), unit);
    return object;
}

QJsonArray serializeStrings(const std::vector<std::string> &values)
{
    QJsonArray array;
    for (const std::string &value : values)
    {
        array.append(QString::fromStdString(value));
    }
    return array;
}

} // namespace

QJsonObject serializeSfmQualityMetrics(const SfmQualityMetrics &metrics)
{
    // 字段分为原始计数、数值分布、空间覆盖和质量门控四组。机器状态与人类可读
    // 建议分别保存，后续版本可新增建议而不破坏 acceptable_for_mvs。
    QJsonObject histogram;
    for (const auto &[length, count] : metrics.trackLengthHistogram)
    {
        histogram.insert(QString::number(length), count);
    }

    QJsonObject coverage;
    coverage.insert(QStringLiteral("mean"), roundMetric(metrics.observationGridCoverageMean));
    coverage.insert(QStringLiteral("grid_columns"), metrics.coverageGridColumns);
    coverage.insert(QStringLiteral("grid_rows"), metrics.coverageGridRows);

    QJsonObject qualityGate;
    qualityGate.insert(QStringLiteral("acceptable_for_mvs"), metrics.acceptableForMvs);
    qualityGate.insert(QStringLiteral("status"), QString::fromStdString(metrics.qualityStatus));
    qualityGate.insert(QStringLiteral("advisories"), serializeStrings(metrics.qualityAdvisories));
    qualityGate.insert(QStringLiteral("warnings"), serializeStrings(metrics.qualityWarnings));

    QJsonObject object;
    object.insert(QStringLiteral("registered_image_count"), metrics.registeredImageCount);
    object.insert(QStringLiteral("total_image_count"), metrics.totalImageCount);
    object.insert(QStringLiteral("point_count"), metrics.pointCount);
    object.insert(QStringLiteral("two_view_track_count"), metrics.twoViewTrackCount);
    object.insert(QStringLiteral("multi_view_track_count"), metrics.multiViewTrackCount);
    object.insert(QStringLiteral("weak_track_count"), metrics.weakTrackCount);
    object.insert(QStringLiteral("weak_triangulation_angle_count"), metrics.weakTriangulationAngleCount);
    object.insert(QStringLiteral("high_reprojection_error_count"), metrics.highReprojectionErrorCount);
    object.insert(QStringLiteral("track_length"),
                  serializeNumericSummary(metrics.trackLength, QStringLiteral("observations")));
    object.insert(QStringLiteral("track_length_histogram"), histogram);
    object.insert(QStringLiteral("reprojection_error"),
                  serializeNumericSummary(metrics.reprojectionError, QStringLiteral("px")));
    object.insert(QStringLiteral("triangulation_angle"),
                  serializeNumericSummary(metrics.triangulationAngle, QStringLiteral("deg")));
    object.insert(QStringLiteral("observation_grid_coverage"), coverage);
    object.insert(QStringLiteral("quality_gate"), qualityGate);
    return object;
}

} // namespace xjw
