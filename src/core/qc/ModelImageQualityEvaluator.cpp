#include "ModelImageQualityEvaluator.h"

#include "ModelImageMetrics.h"
#include "ModelMeshRenderer.h"
#include "DepthMapMeshBuilder.h"
#include "io/PathIO.h"

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
