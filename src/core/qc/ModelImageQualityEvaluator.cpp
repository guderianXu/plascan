#include "ModelImageQualityEvaluator.h"

#include "ModelImageMetrics.h"
#include "ModelMeshRenderer.h"
#include "DepthMapMeshBuilder.h"
#include "io/PathIO.h"
#include "result/OperationResult.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw::core::project
{
xjw::common::OperationResult loadDepthMatStorage(const QString &path, cv::Mat *matrix);
}

namespace xjw::qc
{
namespace
{

double median(std::vector<double> values)
{
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](double value) { return !std::isfinite(value); }),
                 values.end());
    if (values.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

QString safeViewId(QString value, int index)
{
    if (value.trimmed().isEmpty())
    {
        value = QStringLiteral("view_%1").arg(index, 3, 10, QLatin1Char('0'));
    }
    for (QChar &character : value)
    {
        if (!character.isLetterOrNumber() && character != QLatin1Char('-') &&
            character != QLatin1Char('_'))
        {
            character = QLatin1Char('_');
        }
    }
    return value;
}

QString csvEscaped(QString value)
{
    value.replace(QLatin1Char('"'), QStringLiteral("\"\""));
    return value;
}

xjw::Camera scaledCamera(
    const xjw::Camera &camera,
    double scale_x,
    double scale_y)
{
    return camera.scaledIntrinsics(scale_x, scale_y);
}

cv::Mat buildOverlay(const cv::Mat &source,
                     const cv::Mat &rendered,
                     const cv::Mat &valid_mask)
{
    cv::Mat overlay = source.clone();
    cv::Mat blended;
    cv::addWeighted(source, 0.45, rendered, 0.55, 0.0, blended);
    blended.copyTo(overlay, valid_mask);
    return overlay;
}

cv::Mat buildErrorHeatmap(const cv::Mat &source,
                          const cv::Mat &rendered,
                          const cv::Mat &valid_mask)
{
    cv::Mat difference;
    cv::absdiff(source, rendered, difference);
    cv::cvtColor(difference, difference, cv::COLOR_BGR2GRAY);
    cv::Mat heatmap;
    cv::applyColorMap(difference, heatmap, cv::COLORMAP_TURBO);
    cv::Mat result = cv::Mat::zeros(source.size(), CV_8UC3);
    heatmap.copyTo(result, valid_mask);
    return result;
}

bool loadStoredMatrix(const QString &path, cv::Mat *matrix)
{
    if (!matrix || path.trimmed().isEmpty() || !QFileInfo::exists(path))
    {
        return false;
    }
    const xjw::common::OperationResult status =
        xjw::core::project::loadDepthMatStorage(path, matrix);
    return status.ok && !matrix->empty();
}

cv::Mat resizedNearest(const cv::Mat &source, const cv::Size &size)
{
    if (source.empty())
    {
        return {};
    }
    if (source.size() == size)
    {
        return source;
    }
    cv::Mat resized;
    cv::resize(source, resized, size, 0.0, 0.0, cv::INTER_NEAREST);
    return resized;
}

struct EdgeTailArtifacts
{
    ModelViewQuality::EdgeTailDiagnostics diagnostics;
    cv::Mat referenceEdge;
    cv::Mat renderedEdge;
    cv::Mat distanceHeatmap;
    cv::Mat tailMask;
    cv::Mat geometrySupport;
    cv::Mat geometrySourceMask;
    cv::Mat geometrySourceCount;
    cv::Mat inverseDepthSpread;
    cv::Mat crossViewRepairedMask;
};

int sourceBitCount(std::uint16_t mask)
{
    int count = 0;
    while (mask != 0)
    {
        mask = static_cast<std::uint16_t>(mask & (mask - 1));
        ++count;
    }
    return count;
}

cv::Mat maskEdge(const cv::Mat &mask)
{
    cv::Mat binary;
    cv::compare(mask, 0, binary, cv::CMP_GT);
    cv::Mat edge;
    cv::morphologyEx(binary,
                     edge,
                     cv::MORPH_GRADIENT,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    return edge;
}

cv::Mat distanceToEdge(const cv::Mat &edge)
{
    cv::Mat input;
    cv::bitwise_not(edge, input);
    cv::Mat distance;
    cv::distanceTransform(input, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    return distance;
}

cv::Mat colorizeFloat(const cv::Mat &values, const cv::Mat &mask, double maximum)
{
    if (values.empty())
    {
        return {};
    }
    cv::Mat scaled;
    values.convertTo(scaled, CV_8UC1, 255.0 / std::max(1.0e-6, maximum));
    cv::Mat color;
    cv::applyColorMap(scaled, color, cv::COLORMAP_TURBO);
    cv::Mat output = cv::Mat::zeros(values.size(), CV_8UC3);
    color.copyTo(output, mask);
    return output;
}

EdgeTailArtifacts buildEdgeTailArtifacts(
    const ModelValidationView &validation,
    const cv::Mat &reference_mask,
    const cv::Mat &rendered_mask,
    double p90_threshold)
{
    EdgeTailArtifacts result;
    if (reference_mask.empty() || rendered_mask.empty() ||
        reference_mask.size() != rendered_mask.size() ||
        !std::isfinite(p90_threshold))
    {
        return result;
    }
    result.referenceEdge = maskEdge(reference_mask);
    result.renderedEdge = maskEdge(rendered_mask);
    if (cv::countNonZero(result.referenceEdge) == 0 ||
        cv::countNonZero(result.renderedEdge) == 0)
    {
        return result;
    }

    const cv::Mat reference_to_render = distanceToEdge(result.renderedEdge);
    const cv::Mat render_to_reference = distanceToEdge(result.referenceEdge);
    cv::Mat distance_values(reference_mask.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat reference_tail(reference_mask.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat rendered_tail(reference_mask.size(), CV_8UC1, cv::Scalar(0));
    float maximum_distance = 0.0f;
    const float threshold = static_cast<float>(std::max(0.0, p90_threshold));
    for (int row = 0; row < reference_mask.rows; ++row)
    {
        for (int column = 0; column < reference_mask.cols; ++column)
        {
            if (result.referenceEdge.at<std::uint8_t>(row, column) != 0)
            {
                const float distance = reference_to_render.at<float>(row, column);
                distance_values.at<float>(row, column) = distance;
                maximum_distance = std::max(maximum_distance, distance);
                if (distance + 1.0e-4f >= threshold)
                {
                    reference_tail.at<std::uint8_t>(row, column) = 255;
                }
            }
            if (result.renderedEdge.at<std::uint8_t>(row, column) != 0)
            {
                const float distance = render_to_reference.at<float>(row, column);
                distance_values.at<float>(row, column) = std::max(
                    distance_values.at<float>(row, column), distance);
                maximum_distance = std::max(maximum_distance, distance);
                if (distance + 1.0e-4f >= threshold)
                {
                    rendered_tail.at<std::uint8_t>(row, column) = 255;
                }
            }
        }
    }
    cv::Mat tail_union;
    cv::bitwise_or(reference_tail, rendered_tail, tail_union);
    result.tailMask = cv::Mat::zeros(reference_mask.size(), CV_8UC3);
    result.tailMask.setTo(cv::Scalar(0, 0, 255), reference_tail);
    result.tailMask.setTo(cv::Scalar(255, 255, 0), rendered_tail);
    cv::Mat overlapping_tail;
    cv::bitwise_and(reference_tail, rendered_tail, overlapping_tail);
    result.tailMask.setTo(cv::Scalar(255, 0, 255), overlapping_tail);
    result.distanceHeatmap = colorizeFloat(
        distance_values,
        result.referenceEdge | result.renderedEdge,
        std::max(1.0f, maximum_distance));

    cv::Mat geometry_support;
    cv::Mat geometry_source_mask;
    cv::Mat inverse_depth_mean;
    cv::Mat inverse_depth_spread;
    (void)loadStoredMatrix(validation.geometrySupportPath, &geometry_support);
    (void)loadStoredMatrix(validation.geometrySourceMaskPath, &geometry_source_mask);
    (void)loadStoredMatrix(validation.inverseDepthMeanPath, &inverse_depth_mean);
    (void)loadStoredMatrix(validation.inverseDepthSpreadPath, &inverse_depth_spread);
    geometry_support = resizedNearest(geometry_support, reference_mask.size());
    geometry_source_mask = resizedNearest(geometry_source_mask, reference_mask.size());
    inverse_depth_mean = resizedNearest(inverse_depth_mean, reference_mask.size());
    inverse_depth_spread = resizedNearest(inverse_depth_spread, reference_mask.size());

    cv::Mat support_visual(reference_mask.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat source_count_visual(reference_mask.size(), CV_8UC1, cv::Scalar(0));
    result.geometrySourceMask = cv::Mat(
        reference_mask.size(), CV_16UC1, cv::Scalar(0));
    double support_sum = 0.0;
    double source_count_sum = 0.0;
    double spread_sum = 0.0;
    int spread_count = 0;
    const int tail_count = cv::countNonZero(tail_union);
    for (int row = 0; row < reference_mask.rows; ++row)
    {
        for (int column = 0; column < reference_mask.cols; ++column)
        {
            if (tail_union.at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }
            const std::uint16_t support = geometry_support.type() == CV_16UC1
                ? geometry_support.at<std::uint16_t>(row, column) : 0;
            const std::uint16_t source_mask = geometry_source_mask.type() == CV_16UC1
                ? geometry_source_mask.at<std::uint16_t>(row, column) : 0;
            const int source_count = sourceBitCount(source_mask);
            support_visual.at<std::uint8_t>(row, column) =
                static_cast<std::uint8_t>(std::min<int>(255, support * 32));
            source_count_visual.at<std::uint8_t>(row, column) =
                static_cast<std::uint8_t>(std::min(255, source_count * 32));
            result.geometrySourceMask.at<std::uint16_t>(row, column) = source_mask;
            support_sum += support;
            source_count_sum += source_count;
            if (inverse_depth_mean.type() == CV_32FC1 &&
                inverse_depth_spread.type() == CV_32FC1 &&
                inverse_depth_mean.at<float>(row, column) > 0.0f)
            {
                const float spread = inverse_depth_spread.at<float>(row, column);
                if (std::isfinite(spread) && spread >= 0.0f)
                {
                    spread_sum += spread;
                    ++spread_count;
                }
            }
        }
    }
    cv::applyColorMap(support_visual, result.geometrySupport, cv::COLORMAP_TURBO);
    result.geometrySupport.setTo(cv::Scalar(0, 0, 0), tail_union == 0);
    cv::applyColorMap(source_count_visual,
                      result.geometrySourceCount,
                      cv::COLORMAP_TURBO);
    result.geometrySourceCount.setTo(cv::Scalar(0, 0, 0), tail_union == 0);
    if (inverse_depth_spread.type() == CV_32FC1)
    {
        result.inverseDepthSpread = colorizeFloat(
            inverse_depth_spread, tail_union, 0.03);
    }
    if (!validation.crossViewRepairedMaskPath.isEmpty())
    {
        cv::Mat repaired = xjw::common::io::readImage(
            validation.crossViewRepairedMaskPath, cv::IMREAD_GRAYSCALE);
        repaired = resizedNearest(repaired, reference_mask.size());
        if (!repaired.empty())
        {
            cv::bitwise_and(repaired, tail_union, result.crossViewRepairedMask);
        }
    }

    result.diagnostics.available = true;
    result.diagnostics.thresholdPixels = p90_threshold;
    result.diagnostics.referenceEdgePixelCount = cv::countNonZero(result.referenceEdge);
    result.diagnostics.renderedEdgePixelCount = cv::countNonZero(result.renderedEdge);
    result.diagnostics.referenceTailPixelCount = cv::countNonZero(reference_tail);
    result.diagnostics.renderedTailPixelCount = cv::countNonZero(rendered_tail);
    result.diagnostics.tailGeometrySupportMean = tail_count > 0
        ? support_sum / tail_count : 0.0;
    result.diagnostics.tailGeometrySourceCountMean = tail_count > 0
        ? source_count_sum / tail_count : 0.0;
    result.diagnostics.tailInverseDepthSpreadMean = spread_count > 0
        ? spread_sum / spread_count : 0.0;
    result.diagnostics.tailCrossViewRepairedPixelCount =
        result.crossViewRepairedMask.empty()
            ? 0 : cv::countNonZero(result.crossViewRepairedMask);
    return result;
}

ModelViewQuality::DepthCoverageAttribution attributeMissingDepthCoverage(
    const ModelValidationView &validation,
    const cv::Mat &reference_mask,
    const cv::Mat &rendered_mask,
    cv::Mat *diagnostic)
{
    ModelViewQuality::DepthCoverageAttribution result;
    result.fusionEligible = validation.fusionEligible;
    result.frameAcceptance = validation.frameAcceptance;

    cv::Mat depth;
    if (!loadStoredMatrix(validation.depthPath, &depth) || depth.type() != CV_32FC1)
    {
        return result;
    }
    depth = resizedNearest(depth, reference_mask.size());

    cv::Mat support;
    if (!validation.supportMaskPath.isEmpty())
    {
        support = xjw::common::io::readImage(validation.supportMaskPath, cv::IMREAD_GRAYSCALE);
    }
    support = resizedNearest(support, reference_mask.size());
    if (support.empty())
    {
        support = cv::Mat(reference_mask.size(), CV_8UC1, cv::Scalar(255));
    }

    cv::Mat valid;
    if (!validation.validMaskPath.isEmpty())
    {
        valid = xjw::common::io::readImage(validation.validMaskPath, cv::IMREAD_GRAYSCALE);
    }
    valid = resizedNearest(valid, reference_mask.size());
    if (valid.empty())
    {
        valid = depth > 0.0f;
    }

    cv::Mat geometry_support;
    if (loadStoredMatrix(validation.geometrySupportPath, &geometry_support) &&
        geometry_support.type() == CV_16UC1)
    {
        geometry_support = resizedNearest(geometry_support, reference_mask.size());
        result.geometrySupportThreshold = 4;
    }
    else
    {
        geometry_support = cv::Mat(reference_mask.size(), CV_16UC1, cv::Scalar(0));
    }

    cv::Mat missing;
    cv::bitwise_not(rendered_mask, missing);
    cv::bitwise_and(missing, reference_mask, missing);
    result.foregroundPixelCount = cv::countNonZero(reference_mask);
    result.meshMissingPixelCount = cv::countNonZero(missing);
    cv::Mat stage = cv::Mat::zeros(reference_mask.size(), CV_8UC3);
    int verified_foreground = 0;
    for (int row = 0; row < reference_mask.rows; ++row)
    {
        for (int column = 0; column < reference_mask.cols; ++column)
        {
            if (reference_mask.at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }
            const float depth_value = depth.at<float>(row, column);
            const bool supported = support.at<std::uint8_t>(row, column) != 0;
            const bool depth_valid = valid.at<std::uint8_t>(row, column) != 0 &&
                std::isfinite(depth_value) && depth_value > 0.0f;
            const bool geometry_verified = result.geometrySupportThreshold == 0 ||
                geometry_support.at<std::uint16_t>(row, column) >=
                    result.geometrySupportThreshold;
            if (supported && depth_valid && geometry_verified)
            {
                ++verified_foreground;
            }
            if (missing.at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }
            if (!supported)
            {
                ++result.outsideSupportMissingPixelCount;
                stage.at<cv::Vec3b>(row, column) = cv::Vec3b(0, 0, 255);
            }
            else if (!depth_valid)
            {
                ++result.depthInvalidMissingPixelCount;
                stage.at<cv::Vec3b>(row, column) = cv::Vec3b(0, 128, 255);
            }
            else if (!geometry_verified)
            {
                ++result.geometryUnverifiedMissingPixelCount;
                stage.at<cv::Vec3b>(row, column) = cv::Vec3b(0, 255, 255);
            }
            else
            {
                ++result.verifiedDepthButMeshMissingPixelCount;
                stage.at<cv::Vec3b>(row, column) = cv::Vec3b(255, 0, 255);
            }
        }
    }
    result.available = true;
    result.verifiedDepthForegroundCoverage = result.foregroundPixelCount > 0
        ? static_cast<double>(verified_foreground) / result.foregroundPixelCount : 0.0;
    const int missing_without_verified = result.outsideSupportMissingPixelCount +
        result.depthInvalidMissingPixelCount + result.geometryUnverifiedMissingPixelCount;
    result.missingWithoutVerifiedDepthRate = result.meshMissingPixelCount > 0
        ? static_cast<double>(missing_without_verified) / result.meshMissingPixelCount : 0.0;
    result.verifiedDepthButMeshMissingRate = result.meshMissingPixelCount > 0
        ? static_cast<double>(result.verifiedDepthButMeshMissingPixelCount) /
            result.meshMissingPixelCount : 0.0;
    if (diagnostic)
    {
        *diagnostic = stage;
    }
    return result;
}

QJsonObject attributionToJson(
    const ModelViewQuality::DepthCoverageAttribution &attribution)
{
    QJsonObject object;
    object[QStringLiteral("available")] = attribution.available;
    object[QStringLiteral("fusion_eligible")] = attribution.fusionEligible;
    object[QStringLiteral("frame_acceptance")] = attribution.frameAcceptance;
    object[QStringLiteral("geometry_support_threshold")] =
        attribution.geometrySupportThreshold;
    object[QStringLiteral("foreground_pixel_count")] = attribution.foregroundPixelCount;
    object[QStringLiteral("mesh_missing_pixel_count")] = attribution.meshMissingPixelCount;
    object[QStringLiteral("outside_support_missing_pixel_count")] =
        attribution.outsideSupportMissingPixelCount;
    object[QStringLiteral("depth_invalid_missing_pixel_count")] =
        attribution.depthInvalidMissingPixelCount;
    object[QStringLiteral("geometry_unverified_missing_pixel_count")] =
        attribution.geometryUnverifiedMissingPixelCount;
    object[QStringLiteral("verified_depth_but_mesh_missing_pixel_count")] =
        attribution.verifiedDepthButMeshMissingPixelCount;
    object[QStringLiteral("verified_depth_foreground_coverage")] =
        attribution.verifiedDepthForegroundCoverage;
    object[QStringLiteral("missing_without_verified_depth_rate")] =
        attribution.missingWithoutVerifiedDepthRate;
    object[QStringLiteral("verified_depth_but_mesh_missing_rate")] =
        attribution.verifiedDepthButMeshMissingRate;
    return object;
}

QJsonObject edgeTailToJson(
    const ModelViewQuality::EdgeTailDiagnostics &diagnostics)
{
    QJsonObject object;
    object[QStringLiteral("available")] = diagnostics.available;
    object[QStringLiteral("threshold_pixels")] = diagnostics.thresholdPixels;
    object[QStringLiteral("reference_edge_pixel_count")] =
        diagnostics.referenceEdgePixelCount;
    object[QStringLiteral("rendered_edge_pixel_count")] =
        diagnostics.renderedEdgePixelCount;
    object[QStringLiteral("reference_tail_pixel_count")] =
        diagnostics.referenceTailPixelCount;
    object[QStringLiteral("rendered_tail_pixel_count")] =
        diagnostics.renderedTailPixelCount;
    object[QStringLiteral("tail_geometry_support_mean")] =
        diagnostics.tailGeometrySupportMean;
    object[QStringLiteral("tail_geometry_source_count_mean")] =
        diagnostics.tailGeometrySourceCountMean;
    object[QStringLiteral("tail_inverse_depth_spread_mean")] =
        diagnostics.tailInverseDepthSpreadMean;
    object[QStringLiteral("tail_cross_view_repaired_pixel_count")] =
        diagnostics.tailCrossViewRepairedPixelCount;
    return object;
}

QJsonObject viewToJson(const ModelViewQuality &view)
{
    QJsonObject object;
    object[QStringLiteral("view_id")] = view.viewId;
    object[QStringLiteral("image_path")] = view.imagePath;
    object[QStringLiteral("render_ok")] = view.renderOk;
    object[QStringLiteral("error")] = view.error;
    object[QStringLiteral("width")] = view.width;
    object[QStringLiteral("height")] = view.height;
    object[QStringLiteral("render_elapsed_ms")] = view.renderElapsedMs;
    object[QStringLiteral("reference_coverage")] = view.referenceCoverage;
    object[QStringLiteral("silhouette_iou")] = view.silhouetteIou;
    object[QStringLiteral("floating_pixel_rate")] = view.floatingPixelRate;
    object[QStringLiteral("edge_p50_pixels")] = std::isfinite(view.edgeP50Pixels)
        ? QJsonValue(view.edgeP50Pixels) : QJsonValue();
    object[QStringLiteral("edge_p90_pixels")] = std::isfinite(view.edgeP90Pixels)
        ? QJsonValue(view.edgeP90Pixels) : QJsonValue();
    object[QStringLiteral("appearance_available")] = view.appearanceAvailable;
    object[QStringLiteral("foreground_ssim")] = view.foregroundSsim;
    object[QStringLiteral("foreground_psnr")] = view.foregroundPsnr;
    object[QStringLiteral("depth_coverage_attribution")] =
        attributionToJson(view.depthAttribution);
    object[QStringLiteral("edge_tail_diagnostics")] = edgeTailToJson(view.edgeTail);
    return object;
}

QJsonObject geometryToJson(const ModelGeometryQuality &geometry)
{
    QJsonObject object;
    object[QStringLiteral("component_count")] = geometry.componentCount;
    object[QStringLiteral("largest_component_face_ratio")] =
        geometry.largestComponentFaceRatio;
    object[QStringLiteral("largest_floating_diagonal_ratio")] =
        geometry.largestFloatingDiagonalRatio;
    return object;
}

QJsonObject referenceToJson(const ReferenceGeometryQuality &reference)
{
    QJsonObject object;
    object[QStringLiteral("available")] = reference.available;
    object[QStringLiteral("error")] = reference.error;
    object[QStringLiteral("source_point_count")] =
        static_cast<qint64>(reference.sourcePointCount);
    object[QStringLiteral("reference_point_count")] =
        static_cast<qint64>(reference.referencePointCount);
    object[QStringLiteral("rmse")] = reference.rmse;
    object[QStringLiteral("p50")] = reference.p50;
    object[QStringLiteral("p84")] = reference.p84;
    object[QStringLiteral("p95")] = reference.p95;
    object[QStringLiteral("distance_threshold")] = reference.distanceThreshold;
    object[QStringLiteral("source_coverage")] = reference.sourceCoverage;
    object[QStringLiteral("reference_coverage")] = reference.referenceCoverage;
    return object;
}

bool writeReportFiles(const ModelImageQualityOptions &options,
                      const ModelImageQualityResult &result,
                      const cv::Mat &contact_sheet,
                      QString *error)
{
    QJsonObject report = result.summary;
    QJsonArray views;
    for (const ModelViewQuality &view : result.views)
    {
        views.append(viewToJson(view));
    }
    report[QStringLiteral("views")] = views;
    report[QStringLiteral("geometry")] = geometryToJson(result.geometry);
    report[QStringLiteral("reference_geometry")] = referenceToJson(result.referenceGeometry);
    report[QStringLiteral("ok")] = result.ok;
    QJsonArray failures;
    for (const QString &failure : result.failureReasons)
    {
        failures.append(failure);
    }
    report[QStringLiteral("failure_reasons")] = failures;

    const QString json_path = QDir(options.outputDirectory).filePath(
        QStringLiteral("model_quality_report.json"));
    if (!xjw::common::io::writeFileBytesAtomic(
            json_path, QJsonDocument(report).toJson(QJsonDocument::Indented), error))
    {
        return false;
    }

    QByteArray csv("view_id,image_path,render_ok,coverage,iou,floating_rate,edge_p50,edge_p90,ssim,psnr,error\n");
    for (const ModelViewQuality &view : result.views)
    {
        QString line = QStringLiteral("\"%1\",\"%2\",%3,%4,%5,%6,%7,%8,%9,%10,\"%11\"\n")
            .arg(csvEscaped(view.viewId))
            .arg(csvEscaped(view.imagePath))
            .arg(view.renderOk ? 1 : 0)
            .arg(view.referenceCoverage, 0, 'g', 10)
            .arg(view.silhouetteIou, 0, 'g', 10)
            .arg(view.floatingPixelRate, 0, 'g', 10)
            .arg(view.edgeP50Pixels, 0, 'g', 10)
            .arg(view.edgeP90Pixels, 0, 'g', 10)
            .arg(view.foregroundSsim, 0, 'g', 10)
            .arg(view.foregroundPsnr, 0, 'g', 10)
            .arg(csvEscaped(view.error));
        csv.append(line.toUtf8());
    }
    if (!xjw::common::io::writeFileBytesAtomic(
            QDir(options.outputDirectory).filePath(QStringLiteral("model_quality_views.csv")),
            csv, error))
    {
        return false;
    }
    if (!contact_sheet.empty() && !xjw::common::io::writeImage(
            QDir(options.outputDirectory).filePath(QStringLiteral("contact_sheet.png")),
            contact_sheet))
    {
        if (error)
        {
            *error = QStringLiteral("无法写入 contact_sheet.png");
        }
        return false;
    }
    return true;
}

} // namespace

QVector<ModelValidationView> ModelImageQualityEvaluator::validationViewsFromMvsWorkspace(
    const QString &workspacePath,
    QString *error)
{
    QVector<ModelValidationView> views;
    const QVector<xjw::mesh::DepthFrameArtifact> frames =
        xjw::mesh::DepthMapMeshBuilder::discoverDepthFrames(workspacePath);
    for (const xjw::mesh::DepthFrameArtifact &frame : frames)
    {
        if (!frame.hasCameraModel || frame.refImage.isEmpty() ||
            !QFileInfo::exists(frame.refImage))
        {
            continue;
        }
        ModelValidationView view;
        view.id = QFileInfo(frame.refImage).completeBaseName();
        view.imagePath = frame.refImage;
        view.camera = frame.cameraModel;
        view.cameraWidth = frame.gridWidth;
        view.cameraHeight = frame.gridHeight;
        view.depthPath = frame.depthPath;
        view.geometrySupportPath = frame.geometrySupportPath;
        view.geometrySourceMaskPath = frame.geometrySourceMaskPath;
        view.inverseDepthMeanPath = frame.inverseDepthMeanPath;
        view.inverseDepthSpreadPath = frame.inverseDepthSpreadPath;
        view.crossViewRepairedMaskPath = frame.crossViewRepairedMaskPath;
        view.validMaskPath = frame.validMaskPath;
        view.supportMaskPath = frame.supportMaskPath;
        view.frameAcceptance = frame.acceptance;
        view.fusionEligible = frame.fusionEligible;
        views.push_back(std::move(view));
    }
    if (views.isEmpty() && error)
    {
        *error = QStringLiteral("MVS 工作区没有包含有效相机模型的已完成深度帧: %1")
                     .arg(workspacePath);
    }
    else if (error)
    {
        error->clear();
    }
    return views;
}

QStringList ModelImageQualityEvaluator::qualityFailures(
    ModelSceneType sceneType,
    const QVector<ModelViewQuality> &views,
    const ModelGeometryQuality &geometry)
{
    QStringList failures;
    if (views.isEmpty())
    {
        failures.append(QStringLiteral("没有可用验收视角"));
        return failures;
    }
    for (const ModelViewQuality &view : views)
    {
        if (!view.renderOk)
        {
            failures.append(QStringLiteral("视角 %1 渲染失败: %2")
                                .arg(view.viewId, view.error));
        }
    }
    std::vector<double> coverage;
    std::vector<double> iou;
    std::vector<double> edge_p90;
    std::vector<double> ssim;
    for (const ModelViewQuality &view : views)
    {
        if (!view.renderOk)
        {
            continue;
        }
        coverage.push_back(view.referenceCoverage);
        iou.push_back(view.silhouetteIou);
        edge_p90.push_back(view.edgeP90Pixels);
        if (view.appearanceAvailable)
        {
            ssim.push_back(view.foregroundSsim);
        }
    }
    const double median_coverage = median(coverage);
    const double median_iou = median(iou);
    const double median_edge = median(edge_p90);
    const double median_ssim = median(ssim);
    if (!std::isfinite(median_coverage) || median_coverage < 0.90)
    {
        failures.append(QStringLiteral("有效覆盖率中位数低于 90%"));
    }
    if (sceneType == ModelSceneType::Dino &&
        (!std::isfinite(median_iou) || median_iou < 0.90))
    {
        failures.append(QStringLiteral("轮廓 IoU 中位数低于 90%"));
    }
    if (!std::isfinite(median_edge) || median_edge > 3.0)
    {
        failures.append(QStringLiteral("结构/轮廓边缘 P90 中位数超过 3 像素"));
    }
    const double minimum_ssim = sceneType == ModelSceneType::Dino ? 0.75 : 0.70;
    if (!std::isfinite(median_ssim) || median_ssim < minimum_ssim)
    {
        failures.append(QStringLiteral("前景/有效区域 SSIM 中位数低于 %1")
                            .arg(minimum_ssim, 0, 'f', 2));
    }
    if (geometry.largestComponentFaceRatio < 0.85)
    {
        failures.append(QStringLiteral("主连通分量面占比低于 85%"));
    }
    if (geometry.largestFloatingDiagonalRatio > 0.05)
    {
        failures.append(QStringLiteral("存在包围盒对角线超过模型 5% 的漂浮分量"));
    }
    return failures;
}

ModelImageQualityResult ModelImageQualityEvaluator::evaluate(
    const ModelImageQualityOptions &options) const
{
    ModelImageQualityResult result;
    if (options.outputDirectory.trimmed().isEmpty() || options.meshPath.trimmed().isEmpty())
    {
        result.error = QStringLiteral("模型路径和输出目录不能为空");
        return result;
    }
    if (options.validationViews.isEmpty())
    {
        result.error = QStringLiteral("没有验收视角");
        return result;
    }
    if (!QDir().mkpath(options.outputDirectory))
    {
        result.error = QStringLiteral("无法创建输出目录: %1").arg(options.outputDirectory);
        return result;
    }

    xjw::mesh::TriMesh mesh;
    std::string mesh_error;
    if (!xjw::mesh::TriMesh::loadPLY(
            xjw::common::io::toUtf8Path(options.meshPath), &mesh, &mesh_error))
    {
        result.error = QStringLiteral("无法加载模型: %1")
                           .arg(QString::fromUtf8(mesh_error));
        return result;
    }
    result.geometry = ModelGeometryComparator::analyzeMesh(mesh);
    if (!options.referenceCloudPath.trimmed().isEmpty())
    {
        result.referenceGeometry = ModelGeometryComparator::compareReferenceCloud(
            mesh,
            options.referenceCloudPath,
            options.alignReferenceCloud,
            options.hasReferenceTransform ? &options.referenceTransform : nullptr,
            options.cropReferenceToModelBounds);
    }

    ModelMeshRenderer renderer;
    std::vector<cv::Mat> contact_rows;
    for (int index = 0; index < options.validationViews.size(); ++index)
    {
        const ModelValidationView &validation = options.validationViews[index];
        ModelViewQuality quality;
        quality.viewId = safeViewId(validation.id, index);
        quality.imagePath = validation.imagePath;
        cv::Mat source = xjw::common::io::readImage(validation.imagePath, cv::IMREAD_COLOR);
        if (source.empty())
        {
            quality.error = QStringLiteral("无法读取原始影像");
            result.views.append(quality);
            continue;
        }

        const cv::Size original_size = source.size();
        const int maximum_dimension = std::max(1, options.maximumRenderDimension);
        const double scale = std::min(
            1.0, static_cast<double>(maximum_dimension) /
                     static_cast<double>(std::max(source.cols, source.rows)));
        const cv::Size render_size(std::max(1, static_cast<int>(std::lround(source.cols * scale))),
                                   std::max(1, static_cast<int>(std::lround(source.rows * scale))));
        if (render_size != source.size())
        {
            cv::resize(source, source, render_size, 0.0, 0.0, cv::INTER_AREA);
        }
        const int camera_width = validation.cameraWidth > 0
            ? validation.cameraWidth : original_size.width;
        const int camera_height = validation.cameraHeight > 0
            ? validation.cameraHeight : original_size.height;
        const xjw::Camera camera = scaledCamera(
            validation.camera,
            static_cast<double>(render_size.width) / static_cast<double>(camera_width),
            static_cast<double>(render_size.height) / static_cast<double>(camera_height));
        const ModelRenderResult render = renderer.render(mesh, camera, render_size);
        quality.width = render_size.width;
        quality.height = render_size.height;
        quality.renderElapsedMs = render.elapsedMs;
        quality.renderOk = render.ok;
        quality.error = render.error;
        if (!render.ok)
        {
            result.views.append(quality);
            continue;
        }

        cv::Mat reference_mask;
        if (options.sceneType == ModelSceneType::Dino)
        {
            reference_mask = buildDinoForegroundMask(source);
        }
        else
        {
            reference_mask = cv::Mat(source.size(), CV_8UC1, cv::Scalar(255));
        }
        quality = [&]()
        {
            ModelViewQuality measured = evaluateModelViewMasks(reference_mask, render.validMask);
            measured.viewId = safeViewId(validation.id, index);
            measured.imagePath = validation.imagePath;
            measured.renderOk = true;
            measured.width = render_size.width;
            measured.height = render_size.height;
            measured.renderElapsedMs = render.elapsedMs;
            return measured;
        }();
        cv::Mat comparison_mask;
        cv::bitwise_and(reference_mask, render.validMask, comparison_mask);
        evaluateModelViewAppearance(source, render.color, comparison_mask, &quality);
        if (options.sceneType == ModelSceneType::Aerial)
        {
            evaluateModelViewStructure(source, render.color, comparison_mask, &quality);
        }

        cv::Mat missing_stage;
        quality.depthAttribution = attributeMissingDepthCoverage(
            validation, reference_mask, render.validMask, &missing_stage);
        const EdgeTailArtifacts edge_tail = buildEdgeTailArtifacts(
            validation, reference_mask, render.validMask, quality.edgeP90Pixels);
        quality.edgeTail = edge_tail.diagnostics;

        const QString view_directory = QDir(options.outputDirectory).filePath(
            QStringLiteral("comparisons/%1").arg(quality.viewId));
        QDir().mkpath(view_directory);
        const cv::Mat overlay = buildOverlay(source, render.color, render.validMask);
        const cv::Mat heatmap = buildErrorHeatmap(source, render.color, comparison_mask);
        xjw::common::io::writeImage(QDir(view_directory).filePath(
                                        QStringLiteral("source.png")), source);
        xjw::common::io::writeImage(QDir(view_directory).filePath(
                                        QStringLiteral("render.png")), render.color);
        xjw::common::io::writeImage(QDir(view_directory).filePath(
                                        QStringLiteral("overlay.png")), overlay);
        xjw::common::io::writeImage(QDir(view_directory).filePath(
                                        QStringLiteral("error_heatmap.png")), heatmap);
        xjw::common::io::writeImage(QDir(view_directory).filePath(
                                        QStringLiteral("valid_mask.png")), render.validMask);
        if (!missing_stage.empty())
        {
            xjw::common::io::writeImage(QDir(view_directory).filePath(
                                            QStringLiteral("missing_stage.png")), missing_stage);
        }
        if (edge_tail.diagnostics.available)
        {
            const QDir directory(view_directory);
            xjw::common::io::writeImage(
                directory.filePath(QStringLiteral("reference_edge.png")),
                edge_tail.referenceEdge);
            xjw::common::io::writeImage(
                directory.filePath(QStringLiteral("rendered_edge.png")),
                edge_tail.renderedEdge);
            xjw::common::io::writeImage(
                directory.filePath(QStringLiteral("edge_distance_bidirectional.png")),
                edge_tail.distanceHeatmap);
            xjw::common::io::writeImage(
                directory.filePath(QStringLiteral("edge_p90_tail_mask.png")),
                edge_tail.tailMask);
            if (!edge_tail.geometrySupport.empty())
            {
                xjw::common::io::writeImage(
                    directory.filePath(QStringLiteral("edge_tail_geometry_support.png")),
                    edge_tail.geometrySupport);
            }
            if (!edge_tail.geometrySourceMask.empty())
            {
                xjw::common::io::writeImage(
                    directory.filePath(QStringLiteral("edge_tail_geometry_source_mask.png")),
                    edge_tail.geometrySourceMask);
            }
            if (!edge_tail.geometrySourceCount.empty())
            {
                xjw::common::io::writeImage(
                    directory.filePath(QStringLiteral("edge_tail_geometry_source_count.png")),
                    edge_tail.geometrySourceCount);
            }
            if (!edge_tail.inverseDepthSpread.empty())
            {
                xjw::common::io::writeImage(
                    directory.filePath(QStringLiteral("edge_tail_inverse_depth_spread.png")),
                    edge_tail.inverseDepthSpread);
            }
            if (!edge_tail.crossViewRepairedMask.empty())
            {
                xjw::common::io::writeImage(
                    directory.filePath(QStringLiteral("edge_tail_cross_view_repaired.png")),
                    edge_tail.crossViewRepairedMask);
            }
            if (!missing_stage.empty())
            {
                cv::Mat tail_gray;
                cv::cvtColor(edge_tail.tailMask, tail_gray, cv::COLOR_BGR2GRAY);
                cv::Mat tail_stage = cv::Mat::zeros(
                    missing_stage.size(), missing_stage.type());
                missing_stage.copyTo(tail_stage, tail_gray);
                xjw::common::io::writeImage(
                    directory.filePath(QStringLiteral("edge_tail_missing_stage.png")),
                    tail_stage);
            }
        }

        cv::Mat row;
        cv::hconcat(std::vector<cv::Mat>{source, render.color, overlay, heatmap}, row);
        contact_rows.push_back(row);
        result.views.append(quality);
    }

    result.failureReasons = qualityFailures(options.sceneType, result.views, result.geometry);
    result.ok = result.error.isEmpty() && result.failureReasons.isEmpty();
    std::vector<double> coverage;
    std::vector<double> iou;
    std::vector<double> edge;
    std::vector<double> ssim;
    for (const ModelViewQuality &view : result.views)
    {
        if (view.renderOk)
        {
            coverage.push_back(view.referenceCoverage);
            iou.push_back(view.silhouetteIou);
            edge.push_back(view.edgeP90Pixels);
            if (view.appearanceAvailable)
            {
                ssim.push_back(view.foregroundSsim);
            }
        }
    }
    result.summary[QStringLiteral("scene_type")] =
        options.sceneType == ModelSceneType::Dino ? QStringLiteral("dino") : QStringLiteral("aerial");
    result.summary[QStringLiteral("mesh_path")] = options.meshPath;
    result.summary[QStringLiteral("validation_view_count")] = result.views.size();
    result.summary[QStringLiteral("median_coverage")] = median(coverage);
    result.summary[QStringLiteral("median_iou")] = median(iou);
    result.summary[QStringLiteral("median_edge_p90_pixels")] = median(edge);
    result.summary[QStringLiteral("median_ssim")] = median(ssim);
    std::vector<double> verified_depth_coverage;
    std::vector<double> missing_without_verified_depth;
    std::vector<double> verified_depth_but_mesh_missing;
    for (const ModelViewQuality &view : result.views)
    {
        if (!view.depthAttribution.available)
        {
            continue;
        }
        verified_depth_coverage.push_back(
            view.depthAttribution.verifiedDepthForegroundCoverage);
        missing_without_verified_depth.push_back(
            view.depthAttribution.missingWithoutVerifiedDepthRate);
        verified_depth_but_mesh_missing.push_back(
            view.depthAttribution.verifiedDepthButMeshMissingRate);
    }
    QJsonObject attribution_summary;
    attribution_summary[QStringLiteral("view_count")] =
        static_cast<int>(verified_depth_coverage.size());
    attribution_summary[QStringLiteral("median_verified_depth_foreground_coverage")] =
        median(verified_depth_coverage);
    attribution_summary[QStringLiteral("median_missing_without_verified_depth_rate")] =
        median(missing_without_verified_depth);
    attribution_summary[QStringLiteral("median_verified_depth_but_mesh_missing_rate")] =
        median(verified_depth_but_mesh_missing);
    result.summary[QStringLiteral("depth_coverage_attribution")] = attribution_summary;

    cv::Mat contact_sheet;
    if (!contact_rows.empty())
    {
        cv::vconcat(contact_rows, contact_sheet);
    }
    QString write_error;
    if (!writeReportFiles(options, result, contact_sheet, &write_error))
    {
        result.ok = false;
        result.error = write_error;
    }
    return result;
}

} // namespace xjw::qc
