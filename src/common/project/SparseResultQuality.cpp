#include "project/SparseResultQuality.h"

#include <QJsonArray>
#include <QJsonValue>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace xjw::common::project
{

namespace
{

constexpr double kProductionMaxTwoViewRatio = 0.95;
constexpr double kProductionMinRegisteredImageRatio = 0.5;
constexpr double kWarningTwoViewRatio = 0.8;

int pointTrackLen(const QJsonObject &point)
{
    return std::max(0, point.value(QStringLiteral("track_len")).toInt(0));
}

double pointRms(const QJsonObject &point)
{
    return point.value(QStringLiteral("rms_reproj_px")).toDouble(
        point.value(QStringLiteral("rms_after")).toDouble(0.0));
}

double medianDouble(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if ((values.size() % 2) == 1)
    {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

int medianInt(std::vector<int> values)
{
    if (values.empty())
    {
        return 0;
    }

    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

QJsonObject qualityObjectFromRecord(const QJsonObject &record)
{
    const QJsonObject quality = record.value(QStringLiteral("quality")).toObject();
    if (!quality.isEmpty())
    {
        return quality;
    }
    return record;
}

int registeredImageCount(const QJsonObject &quality)
{
    return quality.value(QStringLiteral("registered_image_count")).toInt(
        quality.value(QStringLiteral("camera_count")).toInt(0));
}

int inputImageCount(const QJsonObject &quality)
{
    return quality.value(QStringLiteral("input_image_count")).toInt(
        quality.value(QStringLiteral("selected_image_count")).toInt(0));
}

bool hasInsufficientRegisteredImageCoverage(const QJsonObject &quality)
{
    const int inputCount = inputImageCount(quality);
    if (inputCount < 3)
    {
        return false;
    }

    const int registeredCount = registeredImageCount(quality);
    if (registeredCount < 2)
    {
        return true;
    }

    return static_cast<double>(registeredCount) / static_cast<double>(inputCount)
        < kProductionMinRegisteredImageRatio;
}

QJsonObject qualityGateObjectFromRecord(const QJsonObject &record)
{
    const QJsonObject quality = qualityObjectFromRecord(record);
    QJsonObject gate = quality.value(QStringLiteral("quality_gate")).toObject();
    if (!gate.isEmpty())
    {
        return gate;
    }

    gate = record.value(QStringLiteral("quality_gate")).toObject();
    if (!gate.isEmpty())
    {
        return gate;
    }

    const QJsonObject sparseQuality = record.value(QStringLiteral("sfm_diagnostics"))
        .toObject()
        .value(QStringLiteral("sparse_quality"))
        .toObject();
    return sparseQuality.value(QStringLiteral("quality_gate")).toObject();
}

bool qualityGateBlocksMvs(const QJsonObject &record)
{
    const QJsonObject gate = qualityGateObjectFromRecord(record);
    return gate.contains(QStringLiteral("acceptable_for_mvs"))
        && !gate.value(QStringLiteral("acceptable_for_mvs")).toBool(true);
}

QString qualityGateWarningText(const QString &warning)
{
    if (warning == QLatin1String("high_reprojection_error"))
    {
        return QStringLiteral("重投影误差过高");
    }
    if (warning == QLatin1String("weak_triangulation_angle"))
    {
        return QStringLiteral("三角角过小");
    }
    if (warning == QLatin1String("too_many_two_view_tracks"))
    {
        return QStringLiteral("两视 track 占比过高");
    }
    if (warning == QLatin1String("low_registered_image_coverage"))
    {
        return QStringLiteral("注册影像覆盖率过低");
    }
    if (warning == QLatin1String("poor_observation_spatial_coverage"))
    {
        return QStringLiteral("观测空间覆盖不足");
    }
    return warning;
}

QString qualityGateBlockingReason(const QJsonObject &record)
{
    const QJsonObject gate = qualityGateObjectFromRecord(record);
    if (gate.isEmpty())
    {
        return QString();
    }

    QStringList warnings;
    for (const QJsonValue &value : gate.value(QStringLiteral("warnings")).toArray())
    {
        const QString text = qualityGateWarningText(value.toString());
        if (!text.isEmpty())
        {
            warnings.append(text);
        }
    }

    const QString warningText = warnings.isEmpty()
        ? QStringLiteral("稀疏质量指标未达到 MVS 阈值")
        : warnings.join(QStringLiteral("、"));
    return QStringLiteral("当前 SfM 稀疏点云未通过 MVS 质量门控：%1。"
                          "请检查匹配/SfM/BA 质量后再生成深度图。")
        .arg(warningText);
}

} // namespace

QJsonObject buildSparseQualityMetadata(const QJsonArray &points,
                                       int cameraCount,
                                       bool baApplied,
                                       const QString &resultKind,
                                       const QString &sourceResultKind,
                                       const QString &sourceResultRef)
{
    return buildSparseQualityMetadata(points,
                                      cameraCount,
                                      baApplied,
                                      resultKind,
                                      sourceResultKind,
                                      sourceResultRef,
                                      0);
}

QJsonObject buildSparseQualityMetadata(const QJsonArray &points,
                                       int cameraCount,
                                       bool baApplied,
                                       const QString &resultKind,
                                       const QString &sourceResultKind,
                                       const QString &sourceResultRef,
                                       int inputImageCount)
{
    QJsonObject histogram;
    std::vector<int> trackLens;
    std::vector<double> rmsValues;
    int twoViewCount = 0;

    trackLens.reserve(static_cast<std::size_t>(points.size()));
    rmsValues.reserve(static_cast<std::size_t>(points.size()));

    for (const QJsonValue &value : points)
    {
        const QJsonObject point = value.toObject();
        const int trackLen = pointTrackLen(point);
        if (trackLen <= 0)
        {
            continue;
        }

        const QString key = QString::number(trackLen);
        histogram[key] = histogram.value(key).toInt(0) + 1;
        trackLens.push_back(trackLen);
        if (trackLen == 2)
        {
            ++twoViewCount;
        }

        const double rms = pointRms(point);
        if (std::isfinite(rms) && rms >= 0.0)
        {
            rmsValues.push_back(rms);
        }
    }

    const int pointCount = static_cast<int>(trackLens.size());
    const double twoViewRatio = pointCount > 0
        ? static_cast<double>(twoViewCount) / static_cast<double>(pointCount)
        : 0.0;
    const double meanRms = !rmsValues.empty()
        ? std::accumulate(rmsValues.begin(), rmsValues.end(), 0.0) / rmsValues.size()
        : 0.0;

    QJsonObject quality;
    quality[QStringLiteral("result_kind")] =
        resultKind.isEmpty() ? kSparseResultKindUnknown : resultKind;
    quality[QStringLiteral("camera_count")] = std::max(0, cameraCount);
    quality[QStringLiteral("registered_image_count")] = std::max(0, cameraCount);
    if (inputImageCount > 0)
    {
        quality[QStringLiteral("input_image_count")] = inputImageCount;
    }
    quality[QStringLiteral("point_count")] = pointCount;
    quality[QStringLiteral("track_len_histogram")] = histogram;
    quality[QStringLiteral("two_view_ratio")] = twoViewRatio;
    quality[QStringLiteral("median_track_len")] = medianInt(trackLens);
    quality[QStringLiteral("mean_reproj_px")] = meanRms;
    quality[QStringLiteral("median_reproj_px")] = medianDouble(rmsValues);
    quality[QStringLiteral("ba_applied")] = baApplied;
    if (!sourceResultKind.isEmpty())
    {
        quality[QStringLiteral("source_result_kind")] = sourceResultKind;
    }
    if (!sourceResultRef.isEmpty())
    {
        quality[QStringLiteral("source_result_ref")] = sourceResultRef;
    }
    return quality;
}

QJsonObject mergeSparseQualityIntoRecord(const QJsonObject &record,
                                         const QJsonObject &quality)
{
    QJsonObject merged = record;
    for (auto it = quality.begin(); it != quality.end(); ++it)
    {
        merged[it.key()] = it.value();
    }
    merged[QStringLiteral("quality")] = quality;
    return merged;
}

QString sparseResultKind(const QJsonObject &record)
{
    const QJsonObject quality = qualityObjectFromRecord(record);
    const QString direct = quality.value(QStringLiteral("result_kind")).toString();
    if (!direct.isEmpty())
    {
        return direct;
    }

    const QString operation = record.value(QStringLiteral("operation")).toString();
    if (operation == QLatin1String("triangulation"))
    {
        return kSparseResultKindPairwisePreview;
    }
    if (operation == QLatin1String("bundle_adjust")
        || operation == QLatin1String("workflow_aerial_triangulation"))
    {
        return kSparseResultKindSfmSparseReconstruction;
    }
    return kSparseResultKindUnknown;
}

QString sparseResultKindDisplayName(const QString &resultKind)
{
    if (resultKind == kSparseResultKindPairwisePreview)
    {
        return QStringLiteral("两视预览云");
    }
    if (resultKind == kSparseResultKindSfmSparseReconstruction)
    {
        return QStringLiteral("正式 SfM 稀疏云");
    }
    if (resultKind == kSparseResultKindSparsePostprocess)
    {
        return QStringLiteral("稀疏云后处理");
    }
    return QStringLiteral("未知稀疏云");
}

bool isPairwisePreviewSparseResult(const QJsonObject &record)
{
    return sparseResultKind(record) == kSparseResultKindPairwisePreview;
}

bool isProductionSparseResult(const QJsonObject &record)
{
    const QJsonObject quality = qualityObjectFromRecord(record);
    const QString kind = sparseResultKind(record);
    const QString sourceKind = quality.value(QStringLiteral("source_result_kind")).toString();
    const bool formalKind = kind == kSparseResultKindSfmSparseReconstruction
        || (kind == kSparseResultKindSparsePostprocess
            && sourceKind == kSparseResultKindSfmSparseReconstruction);
    if (!formalKind)
    {
        return false;
    }
    if (!quality.value(QStringLiteral("ba_applied")).toBool(false))
    {
        return false;
    }
    if (registeredImageCount(quality) < 2)
    {
        return false;
    }
    if (qualityGateBlocksMvs(record))
    {
        return false;
    }
    if (hasInsufficientRegisteredImageCoverage(quality))
    {
        return false;
    }
    if (quality.value(QStringLiteral("point_count")).toInt(0) <= 0)
    {
        return false;
    }
    if (quality.value(QStringLiteral("two_view_ratio")).toDouble(1.0) >= kProductionMaxTwoViewRatio)
    {
        return false;
    }
    return true;
}

QString sparseResultBlockingReason(const QJsonObject &record)
{
    const QJsonObject quality = qualityObjectFromRecord(record);
    const QString kind = sparseResultKind(record);
    if (kind == kSparseResultKindPairwisePreview)
    {
        return QStringLiteral("当前结果是两视初始三角化预览云，不能作为正式航测稀疏点云。请先运行空中三角测量。");
    }
    if (!quality.value(QStringLiteral("ba_applied")).toBool(false))
    {
        return QStringLiteral("当前稀疏点云没有光束法平差质量标记，请先运行空中三角测量。");
    }
    if (registeredImageCount(quality) < 2)
    {
        return QStringLiteral("当前稀疏点云注册影像少于 2 张，不能作为正式航测稀疏点云。");
    }
    if (qualityGateBlocksMvs(record))
    {
        return qualityGateBlockingReason(record);
    }
    if (hasInsufficientRegisteredImageCoverage(quality))
    {
        const int registeredCount = registeredImageCount(quality);
        const int inputCount = inputImageCount(quality);
        const double ratio = inputCount > 0
            ? static_cast<double>(registeredCount) / static_cast<double>(inputCount)
            : 0.0;
        return QStringLiteral("当前稀疏点云注册影像覆盖率过低（%1/%2，%3%，生产阈值 %4%）。"
                              "请检查匹配连通性/相机位姿，重新运行正式 SfM/空三。")
            .arg(registeredCount)
            .arg(inputCount)
            .arg(ratio * 100.0, 0, 'f', 1)
            .arg(kProductionMinRegisteredImageRatio * 100.0, 0, 'f', 0);
    }
    const double twoViewRatio = quality.value(QStringLiteral("two_view_ratio")).toDouble(1.0);
    if (twoViewRatio >= kProductionMaxTwoViewRatio)
    {
        return QStringLiteral("当前稀疏点云两视 track 占比过高（%1%，生产阈值 %2%），多视约束不足。"
                              "请检查相机位姿/匹配结果，建议使用调整后的相机或重新运行正式 SfM/BA。")
            .arg(twoViewRatio * 100.0, 0, 'f', 1)
            .arg(kProductionMaxTwoViewRatio * 100.0, 0, 'f', 0);
    }
    if (quality.value(QStringLiteral("point_count")).toInt(0) <= 0)
    {
        return QStringLiteral("当前稀疏点云没有有效三维点。");
    }
    return QString();
}

QString sparseResultWarningText(const QJsonObject &record)
{
    const QJsonObject quality = qualityObjectFromRecord(record);
    const double twoViewRatio = quality.value(QStringLiteral("two_view_ratio")).toDouble(0.0);
    if (twoViewRatio >= kWarningTwoViewRatio)
    {
        return QStringLiteral("两视 track 占比过高：%1%，结果可能不稳定。")
            .arg(twoViewRatio * 100.0, 0, 'f', 1);
    }
    return QString();
}

} // namespace xjw::common::project
