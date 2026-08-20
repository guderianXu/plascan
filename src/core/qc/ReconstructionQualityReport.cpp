#include "ReconstructionQualityReport.h"
#include "SurveyControlReport.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonValue>
#include <QSaveFile>
#include <QTextStream>

#include <algorithm>
#include <cmath>
#include <optional>

namespace xjw::qc
{

namespace
{

QJsonObject lastObject(const QJsonObject &root, const QString &arrayKey)
{
    const QJsonArray array = root.value(arrayKey).toArray();
    if (array.isEmpty())
    {
        return QJsonObject();
    }
    return array.at(array.size() - 1).toObject();
}

QJsonObject latestSparseArtifact(const QJsonObject &projectMeta)
{
    return lastObject(
        projectMeta, QStringLiteral("aerial_triangulation_results"));
}

QJsonObject qualityFromSparseArtifactRecord(const QJsonObject &record)
{
    const QJsonObject diagnostics = record.value(QStringLiteral("sfm_diagnostics")).toObject();
    QJsonObject quality = diagnostics.value(QStringLiteral("sparse_quality")).toObject();
    const int sparsePointCount = record.value(
        QStringLiteral("sparse_point_count")).toInt(-1);
    if (sparsePointCount >= 0 && !quality.contains(QStringLiteral("point_count")))
    {
        quality[QStringLiteral("point_count")] = sparsePointCount;
    }

    const int registeredImageCount = record.value(QStringLiteral("registered_image_count")).toInt(-1);
    if (registeredImageCount >= 0 && !quality.contains(QStringLiteral("registered_image_count")))
    {
        quality[QStringLiteral("registered_image_count")] = registeredImageCount;
    }

    const int totalImageCount = record.value(
        QStringLiteral("total_image_count")).toInt(-1);
    if (totalImageCount >= 0 && !quality.contains(QStringLiteral("total_image_count")))
    {
        quality[QStringLiteral("total_image_count")] = totalImageCount;
    }

    if (record.contains(QStringLiteral("track_length_histogram")) &&
        !quality.contains(QStringLiteral("track_length_histogram")))
    {
        quality[QStringLiteral("track_length_histogram")] = record.value(QStringLiteral("track_length_histogram"));
    }
    return quality;
}

QJsonObject qualityFromLatestSparseResult(const QJsonObject &projectMeta)
{
    QJsonObject quality = qualityFromSparseArtifactRecord(latestSparseArtifact(projectMeta));
    if (!quality.isEmpty())
    {
        return quality;
    }

    const QJsonObject baResult = lastObject(projectMeta, QStringLiteral("bundle_adjust_results"));
    quality = baResult.value(QStringLiteral("quality")).toObject();
    if (!quality.isEmpty())
    {
        return quality;
    }

    return QJsonObject();
}

QJsonObject baSummaryFromLatestSparseResult(const QJsonObject &projectMeta)
{
    const QJsonObject sparseArtifact = latestSparseArtifact(projectMeta);
    const QJsonObject diagnostics = sparseArtifact.value(QStringLiteral("sfm_diagnostics")).toObject();
    return diagnostics.value(QStringLiteral("ba_summary")).toObject();
}

int imageCount(const QJsonObject &projectMeta)
{
    return projectMeta.value(QStringLiteral("images")).toArray().size();
}

int registeredImageCountFromImages(const QJsonObject &projectMeta)
{
    int registeredCount = 0;
    const QJsonArray images = projectMeta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        if (image.value(QStringLiteral("registered")).toBool(false))
        {
            ++registeredCount;
        }
    }
    return registeredCount;
}

QJsonArray unregisteredImages(const QJsonObject &projectMeta)
{
    QJsonArray result;
    const QJsonArray images = projectMeta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        if (!image.value(QStringLiteral("registered")).toBool(false))
        {
            const QString path = image.value(QStringLiteral("path")).toString(
                image.value(QStringLiteral("file")).toString());
            if (!path.isEmpty())
            {
                result.append(path);
            }
        }
    }
    return result;
}

std::optional<double> depthCoverage(const QJsonObject &record)
{
    const auto finiteRatio = [](const QJsonValue &value) -> std::optional<double>
    {
        if (!value.isDouble())
        {
            return std::nullopt;
        }
        const double ratio = value.toDouble();
        if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0)
        {
            return std::nullopt;
        }
        return ratio;
    };

    if (const auto value = finiteRatio(record.value(QStringLiteral("valid_coverage"))))
    {
        return value;
    }
    const int width = record.value(QStringLiteral("grid_width")).toInt(0);
    const int height = record.value(QStringLiteral("grid_height")).toInt(0);
    const int valid = record.value(QStringLiteral("valid_pixel_count")).toInt(-1);
    if (width > 0 && height > 0 && valid >= 0)
    {
        return std::clamp(static_cast<double>(valid) /
                              static_cast<double>(width) / static_cast<double>(height),
                          0.0,
                          1.0);
    }
    return std::nullopt;
}

std::optional<double> averageCompletedDepthCoverage(const QJsonObject &projectMeta)
{
    const QJsonArray depthResults = projectMeta.value(QStringLiteral("depth_map_results")).toArray();
    double sum = 0.0;
    int count = 0;
    for (const QJsonValue &value : depthResults)
    {
        const QJsonObject record = value.toObject();
        const QString status = record.value(QStringLiteral("status")).toString();
        if (!status.isEmpty() && status != QStringLiteral("completed"))
        {
            continue;
        }

        const std::optional<double> coverage = depthCoverage(record);
        if (!coverage)
        {
            continue;
        }
        sum += *coverage;
        ++count;
    }
    return count > 0
        ? std::optional<double>(sum / static_cast<double>(count))
        : std::nullopt;
}

int completedDepthFrameCount(const QJsonObject &projectMeta)
{
    int count = 0;
    const QJsonArray depthResults = projectMeta.value(QStringLiteral("depth_map_results")).toArray();
    for (const QJsonValue &value : depthResults)
    {
        const QJsonObject record = value.toObject();
        const QString status = record.value(QStringLiteral("status")).toString();
        if (status.isEmpty() || status == QStringLiteral("completed"))
        {
            ++count;
        }
    }
    return count;
}

double demCoverage(const QJsonObject &projectMeta)
{
    const QJsonObject demResult = lastObject(projectMeta, QStringLiteral("dem_results"));
    return demResult.value(QStringLiteral("coverage_ratio")).toDouble(0.0);
}

int densePointCount(const QJsonObject &projectMeta)
{
    const QJsonObject denseResult = lastObject(projectMeta, QStringLiteral("dense_cloud_results"));
    return denseResult.value(QStringLiteral("point_count")).toInt(0);
}

QString csvEscape(const QString &value)
{
    if (!value.contains(QLatin1Char(',')) && !value.contains(QLatin1Char('"')) && !value.contains(QLatin1Char('\n')))
    {
        return value;
    }

    QString escaped = value;
    escaped.replace(QStringLiteral("\""), QStringLiteral("\"\""));
    return QStringLiteral("\"%1\"").arg(escaped);
}

void appendCsvMetric(QStringList *lines, const QString &name, const QJsonValue &value)
{
    if (!lines)
    {
        return;
    }

    if (value.isDouble())
    {
        lines->append(QStringLiteral("%1,%2").arg(name, QString::number(value.toDouble(), 'g', 12)));
    }
    else if (value.isBool())
    {
        lines->append(QStringLiteral("%1,%2").arg(name, value.toBool() ? QStringLiteral("true") : QStringLiteral("false")));
    }
    else if (value.isString())
    {
        lines->append(QStringLiteral("%1,%2").arg(name, csvEscape(value.toString())));
    }
}

bool writeTextAtomically(const QString &path, const QByteArray &data, QString *error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (error)
        {
            *error = file.errorString();
        }
        return false;
    }

    file.write(data);
    if (!file.commit())
    {
        if (error)
        {
            *error = file.errorString();
        }
        return false;
    }
    return true;
}

} // namespace

QJsonObject ReconstructionQualityReport::buildFromProjectMeta(const QJsonObject &projectMeta)
{
    const QJsonObject &normalizedMeta = projectMeta;
    const QJsonObject sparseQuality = qualityFromLatestSparseResult(normalizedMeta);
    const QJsonObject baSummary = baSummaryFromLatestSparseResult(normalizedMeta);
    const int totalImages = sparseQuality.value(QStringLiteral("total_image_count")).toInt(imageCount(normalizedMeta));
    const int registeredImages = sparseQuality.value(QStringLiteral("registered_image_count")).toInt(
        registeredImageCountFromImages(normalizedMeta));

    QJsonObject report;
    report[QStringLiteral("type")] = QStringLiteral("reconstruction_quality");
    report[QStringLiteral("total_image_count")] = totalImages;
    report[QStringLiteral("registered_image_count")] = registeredImages;
    report[QStringLiteral("unregistered_image_count")] = std::max(0, totalImages - registeredImages);
    report[QStringLiteral("unregistered_images")] = unregisteredImages(normalizedMeta);
    report[QStringLiteral("sparse_point_count")] = sparseQuality.value(QStringLiteral("point_count")).toInt(0);
    report[QStringLiteral("dense_point_count")] = densePointCount(normalizedMeta);
    report[QStringLiteral("mean_reprojection_error_px")] =
        sparseQuality.value(QStringLiteral("mean_reprojection_error_px")).toDouble(0.0);
    report[QStringLiteral("track_length_histogram")] =
        sparseQuality.value(QStringLiteral("track_length_histogram")).toObject();
    report[QStringLiteral("track_length")] = sparseQuality.value(QStringLiteral("track_length")).toObject();
    report[QStringLiteral("reprojection_error")] = sparseQuality.value(QStringLiteral("reprojection_error")).toObject();
    report[QStringLiteral("triangulation_angle")] = sparseQuality.value(QStringLiteral("triangulation_angle")).toObject();
    report[QStringLiteral("ba_summary")] = baSummary;
    report[QStringLiteral("mvs_depth_frame_count")] =
        normalizedMeta.value(QStringLiteral("depth_map_results")).toArray().size();
    report[QStringLiteral("mvs_completed_depth_frame_count")] = completedDepthFrameCount(normalizedMeta);
    if (const auto coverage = averageCompletedDepthCoverage(normalizedMeta))
    {
        report[QStringLiteral("mvs_valid_coverage")] = *coverage;
    }
    report[QStringLiteral("dem_coverage")] = demCoverage(normalizedMeta);

    const QJsonObject surveySummary = buildSurveyControlSummary(normalizedMeta);
    report[QStringLiteral("survey_control")] = surveySummary;
    report[QStringLiteral("control_point_count")] = surveySummary.value(QStringLiteral("control_point_count")).toInt();
    report[QStringLiteral("check_point_count")] = surveySummary.value(QStringLiteral("check_point_count")).toInt();
    report[QStringLiteral("scale_bar_count")] = surveySummary.value(QStringLiteral("scale_bar_count")).toInt();
    report[QStringLiteral("control_point_rmse_m")] =
        surveySummary.value(QStringLiteral("control_point_rmse_m")).toDouble();
    report[QStringLiteral("check_point_rmse_m")] =
        surveySummary.value(QStringLiteral("check_point_rmse_m")).toDouble();
    report[QStringLiteral("scale_bar_rmse_m")] =
        surveySummary.value(QStringLiteral("scale_bar_rmse_m")).toDouble();
    return report;
}

ReconstructionQualityReportWriteResult ReconstructionQualityReport::writeFromProjectMeta(const QJsonObject &projectMeta,
                                                                                         const QString &outputDir,
                                                                                         const QString &baseName)
{
    ReconstructionQualityReportWriteResult result;
    result.report = buildFromProjectMeta(projectMeta);

    if (outputDir.isEmpty())
    {
        result.error = QStringLiteral("输出目录为空");
        return result;
    }

    if (!QDir().mkpath(outputDir))
    {
        result.error = QStringLiteral("无法创建输出目录: %1").arg(outputDir);
        return result;
    }

    const QString safeBaseName = baseName.isEmpty()
        ? QStringLiteral("reconstruction_quality_report")
        : baseName;
    const QDir dir(outputDir);
    result.jsonPath = dir.filePath(safeBaseName + QStringLiteral(".json"));
    result.csvPath = dir.filePath(safeBaseName + QStringLiteral(".csv"));

    QString error;
    if (!writeTextAtomically(result.jsonPath,
                             QJsonDocument(result.report).toJson(QJsonDocument::Indented),
                             &error))
    {
        result.error = QStringLiteral("写入质量报告 JSON 失败: %1").arg(error);
        return result;
    }

    QStringList lines;
    lines.append(QStringLiteral("metric,value"));
    for (auto it = result.report.begin(); it != result.report.end(); ++it)
    {
        appendCsvMetric(&lines, it.key(), it.value());
    }

    if (!writeTextAtomically(result.csvPath, lines.join(QLatin1Char('\n')).toUtf8(), &error))
    {
        result.error = QStringLiteral("写入质量报告 CSV 失败: %1").arg(error);
        return result;
    }

    result.ok = true;
    return result;
}

} // namespace xjw::qc
