#include "DepthMapMeshBuilder.h"

#include "Camera.h"
#include "DepthFrameUtils.h"
#include "VisualHullReconstructor.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QStringList>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace xjw::mesh
{
namespace
{

QDir sourceDirectory(const QString &source_path)
{
    const QFileInfo info(source_path);
    return QDir(info.isDir() ? info.absoluteFilePath() : info.absolutePath());
}

QString resolveArtifactPath(const QDir &directory, const QString &path)
{
    if (path.isEmpty())
    {
        return QString();
    }
    const QFileInfo info(path);
    return QDir::cleanPath(info.isAbsolute() ? path : directory.filePath(path));
}

bool readJsonObject(const QString &path, QJsonObject *object)
{
    QFile file(path);
    if (!object || !file.open(QIODevice::ReadOnly))
    {
        return false;
    }
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject())
    {
        return false;
    }
    *object = document.object();
    return true;
}

bool parseDoubleArray(const QJsonValue &value, double *output, int count)
{
    const QJsonArray array = value.toArray();
    if (!output || array.size() != count)
    {
        return false;
    }
    for (int index = 0; index < count; ++index)
    {
        output[index] = array[index].toDouble();
    }
    return true;
}

bool parseCameraModel(const QJsonObject &object, Camera *camera)
{
    if (!camera || object.isEmpty())
    {
        return false;
    }
    std::array<double, 9> worldToCamera{};
    std::array<double, 3> translation{};
    std::array<double, 3> center{};
    const bool arrays_ok =
        parseDoubleArray(object.value(QStringLiteral("rotation_world_to_camera")),
                         worldToCamera.data(),
                         9) &&
        parseDoubleArray(object.value(QStringLiteral("translation_world_to_camera")),
                         translation.data(),
                         3) &&
        parseDoubleArray(object.value(QStringLiteral("camera_center")), center.data(), 3);
    const double focalX = object.value(QStringLiteral("fx")).toDouble();
    const double focalY = object.value(QStringLiteral("fy")).toDouble();
    if (!arrays_ok || !std::isfinite(focalX) || !std::isfinite(focalY) ||
        std::fabs(focalX) <= 1.0e-12 || std::fabs(focalY) <= 1.0e-12)
    {
        return false;
    }

    std::array<double, 9> cameraToWorld{{
        worldToCamera[0], worldToCamera[3], worldToCamera[6],
        worldToCamera[1], worldToCamera[4], worldToCamera[7],
        worldToCamera[2], worldToCamera[5], worldToCamera[8]
    }};
    Camera parsed;
    parsed.setIntrinsics(focalX,
                         focalY,
                         object.value(QStringLiteral("cx")).toDouble(),
                         object.value(QStringLiteral("cy")).toDouble());
    parsed.setPose(cameraToWorld, center);
    parsed.setDistortion(Camera::Distortion{});
    *camera = parsed;
    return true;
}

bool selectExistingPyramidArtifact(const QDir &directory,
                                   const QJsonObject &frame_object,
                                   QString *depth_path,
                                   QString *confidence_path,
                                   QString *preview_path,
                                   QString *valid_mask_path,
                                   int *grid_width,
                                   int *grid_height)
{
    if (!depth_path || !confidence_path || !preview_path || !valid_mask_path ||
        !grid_width || !grid_height)
    {
        return false;
    }

    QJsonObject selected;
    qint64 selected_area = -1;
    for (const QJsonValue &level_value : frame_object.value(QStringLiteral("pyramid_levels")).toArray())
    {
        const QJsonObject level = level_value.toObject();
        if (!level.value(QStringLiteral("success")).toBool(false))
        {
            continue;
        }
        const QString candidate = resolveArtifactPath(
            directory, level.value(QStringLiteral("raw_depth_path")).toString());
        const int width = level.value(QStringLiteral("artifact_width")).toInt(0);
        const int height = level.value(QStringLiteral("artifact_height")).toInt(0);
        const qint64 area = static_cast<qint64>(width) * height;
        if (width <= 0 || height <= 0 || !QFileInfo::exists(candidate) || area <= selected_area)
        {
            continue;
        }
        selected = level;
        selected_area = area;
    }
    if (selected.isEmpty())
    {
        return false;
    }

    *depth_path = resolveArtifactPath(
        directory, selected.value(QStringLiteral("raw_depth_path")).toString());
    *confidence_path = resolveArtifactPath(
        directory, selected.value(QStringLiteral("raw_confidence_path")).toString());
    *preview_path = resolveArtifactPath(
        directory, selected.value(QStringLiteral("preview_path")).toString());
    *valid_mask_path = resolveArtifactPath(
        directory, selected.value(QStringLiteral("valid_mask_path")).toString());
    *grid_width = selected.value(QStringLiteral("artifact_width")).toInt();
    *grid_height = selected.value(QStringLiteral("artifact_height")).toInt();
    return true;
}

void attachLegacyReportCameras(const QDir &directory, QVector<DepthFrameArtifact> *frames)
{
    if (!frames)
    {
        return;
    }
    QJsonObject report;
    const QStringList candidates = {
        directory.filePath(QStringLiteral("report.json")),
        QDir(directory.absolutePath() + QStringLiteral("/..")).filePath(QStringLiteral("report.json"))
    };
    for (const QString &candidate : candidates)
    {
        if (readJsonObject(candidate, &report))
        {
            break;
        }
    }
    if (report.isEmpty())
    {
        return;
    }

    std::unordered_map<std::string, QString> cameras_by_image;
    for (const QJsonValue &value : report.value(QStringLiteral("inputs")).toArray())
    {
        const QJsonObject input = value.toObject();
        const QString image = input.value(QStringLiteral("image")).toString();
        const QString camera = input.value(QStringLiteral("camera")).toString();
        cameras_by_image[QFileInfo(image).fileName().toCaseFolded().toStdString()] = camera;
    }

    for (DepthFrameArtifact &frame : *frames)
    {
        if (frame.hasCameraModel || frame.refImage.isEmpty())
        {
            continue;
        }
        const auto it = cameras_by_image.find(
            QFileInfo(frame.refImage).fileName().toCaseFolded().toStdString());
        if (it == cameras_by_image.end())
        {
            continue;
        }
        Camera camera;
        if (!camera.loadFromFile(xjw::common::io::toUtf8Path(it->second)))
        {
            continue;
        }
        Camera model = camera.normalizedForPositiveDepth();
        model.setDistortion(Camera::Distortion{});
        const cv::Mat image = xjw::common::io::readImage(
            xjw::common::io::toUtf8Path(frame.refImage), cv::IMREAD_GRAYSCALE);
        if (!image.empty() && frame.gridWidth > 0 && frame.gridHeight > 0 &&
            (image.cols != frame.gridWidth || image.rows != frame.gridHeight))
        {
            model = model.scaledIntrinsics(
                static_cast<double>(frame.gridWidth) / image.cols,
                static_cast<double>(frame.gridHeight) / image.rows);
        }
        frame.cameraModel = model;
        frame.hasCameraModel = model.isValid();
    }
}

cv::Mat buildSilhouetteMask(const cv::Mat &color_image,
                            float *coverage,
                            float *border_coverage,
                            float *border_luminance)
{
    cv::Mat gray;
    if (color_image.channels() == 3)
    {
        cv::cvtColor(color_image, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = color_image;
    }
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);
    cv::Mat otsu_mask;
    const double otsu_threshold = cv::threshold(
        blurred, otsu_mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const double conservative_threshold = std::clamp(otsu_threshold, 24.0, 0.19 * 255.0);
    cv::Mat mask;
    cv::threshold(blurred, mask, conservative_threshold, 255.0, cv::THRESH_BINARY);

    const int border_width = std::max(2, std::min(mask.cols, mask.rows) / 80);
    cv::Mat border = cv::Mat::zeros(mask.size(), CV_8UC1);
    border.rowRange(0, border_width).setTo(255);
    border.rowRange(mask.rows - border_width, mask.rows).setTo(255);
    border.colRange(0, border_width).setTo(255);
    border.colRange(mask.cols - border_width, mask.cols).setTo(255);
    const int border_pixels = cv::countNonZero(border);
    const int white_border = cv::countNonZero(mask & border);
    if (border_luminance)
    {
        *border_luminance = static_cast<float>(cv::mean(gray, border)[0]);
    }
    if (white_border > border_pixels / 2)
    {
        cv::bitwise_not(mask, mask);
    }

    cv::Mat labels;
    cv::Mat statistics;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        mask,
        labels,
        statistics,
        centroids,
        8,
        CV_32S);
    if (component_count <= 1)
    {
        return {};
    }
    int largest_label = 1;
    int largest_area = statistics.at<int>(1, cv::CC_STAT_AREA);
    for (int label = 2; label < component_count; ++label)
    {
        const int area = statistics.at<int>(label, cv::CC_STAT_AREA);
        if (area > largest_area)
        {
            largest_label = label;
            largest_area = area;
        }
    }
    cv::compare(labels, largest_label, mask, cv::CMP_EQ);
    const double image_scale = std::min(mask.cols / 640.0, mask.rows / 480.0);
    const int dilate_radius = std::max(2, static_cast<int>(std::lround(10.0 * image_scale)));
    const int erode_radius = std::max(1, static_cast<int>(std::lround(7.0 * image_scale)));
    const cv::Mat dilate_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(dilate_radius * 2 + 1, dilate_radius * 2 + 1));
    const cv::Mat erode_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(erode_radius * 2 + 1, erode_radius * 2 + 1));
    cv::dilate(mask, mask, dilate_kernel);
    cv::erode(mask, mask, erode_kernel);

    cv::Mat background;
    cv::bitwise_not(mask, background);
    cv::Mat background_labels;
    cv::Mat background_statistics;
    cv::Mat background_centroids;
    const int background_component_count = cv::connectedComponentsWithStats(
        background,
        background_labels,
        background_statistics,
        background_centroids,
        8,
        CV_32S);
    const int maximum_speckle_hole_area =
        std::max(32, mask.rows * mask.cols / 240);
    for (int label = 1; label < background_component_count; ++label)
    {
        const int left =
            background_statistics.at<int>(label, cv::CC_STAT_LEFT);
        const int top =
            background_statistics.at<int>(label, cv::CC_STAT_TOP);
        const int width =
            background_statistics.at<int>(label, cv::CC_STAT_WIDTH);
        const int height =
            background_statistics.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area =
            background_statistics.at<int>(label, cv::CC_STAT_AREA);
        const bool touches_border =
            left == 0 || top == 0 || left + width >= mask.cols ||
            top + height >= mask.rows;
        if (!touches_border && area <= maximum_speckle_hole_area)
        {
            mask.setTo(255, background_labels == label);
        }
    }

    if (coverage)
    {
        *coverage = static_cast<float>(cv::countNonZero(mask)) /
                    std::max(1, mask.rows * mask.cols);
    }
    if (border_coverage)
    {
        *border_coverage = static_cast<float>(cv::countNonZero(mask & border)) /
                           std::max(1, border_pixels);
    }
    return mask;
}

int countEnclosedMaskHoles(const cv::Mat &mask)
{
    if (mask.empty() || mask.type() != CV_8UC1)
    {
        return 0;
    }
    cv::Mat background;
    cv::bitwise_not(mask, background);
    cv::Mat labels;
    cv::Mat statistics;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        background,
        labels,
        statistics,
        centroids,
        8,
        CV_32S);
    int hole_count = 0;
    for (int label = 1; label < component_count; ++label)
    {
        const int left =
            statistics.at<int>(label, cv::CC_STAT_LEFT);
        const int top =
            statistics.at<int>(label, cv::CC_STAT_TOP);
        const int width =
            statistics.at<int>(label, cv::CC_STAT_WIDTH);
        const int height =
            statistics.at<int>(label, cv::CC_STAT_HEIGHT);
        const bool touches_border =
            left == 0 || top == 0 ||
            left + width >= mask.cols ||
            top + height >= mask.rows;
        if (!touches_border)
        {
            ++hole_count;
        }
    }
    return hole_count;
}

bool estimateBounds(const QVector<DepthFrameArtifact> &frames,
                    std::array<float, 3> *minimum,
                    std::array<float, 3> *maximum)
{
    std::array<std::vector<float>, 3> coordinates;
    for (const DepthFrameArtifact &frame : frames)
    {
        if (!frame.hasCameraModel || frame.depthPath.isEmpty())
        {
            continue;
        }
        cv::Mat depth;
        if (!xjw::core::project::loadDepthMatStorage(frame.depthPath, &depth).ok || depth.empty())
        {
            continue;
        }
        const int stride = std::max(1, static_cast<int>(std::sqrt(
            static_cast<double>(depth.total()) / 6000.0)));
        for (int row = 0; row < depth.rows; row += stride)
        {
            for (int column = 0; column < depth.cols; column += stride)
            {
                const float value = depth.at<float>(row, column);
                if (!std::isfinite(value) || value <= 0.0f)
                {
                    continue;
                }
                const double pixel[2] = {column + 0.5, row + 0.5};
                double world[3] = {};
                if (frame.cameraModel.unprojectPixel(pixel, value, world) &&
                    std::isfinite(world[0]) && std::isfinite(world[1]) &&
                    std::isfinite(world[2]))
                {
                    coordinates[0].push_back(static_cast<float>(world[0]));
                    coordinates[1].push_back(static_cast<float>(world[1]));
                    coordinates[2].push_back(static_cast<float>(world[2]));
                }
            }
        }
    }
    if (coordinates[0].size() < 500)
    {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        std::sort(coordinates[axis].begin(), coordinates[axis].end());
        const std::size_t last = coordinates[axis].size() - 1;
        const float low = coordinates[axis][static_cast<std::size_t>(last * 0.01)];
        const float high = coordinates[axis][static_cast<std::size_t>(last * 0.99)];
        const float padding = std::max((high - low) * 0.08f, 1.0e-5f);
        (*minimum)[axis] = low - padding;
        (*maximum)[axis] = high + padding;
    }
    return true;
}

} // namespace

QVector<DepthFrameArtifact> DepthMapMeshBuilder::discoverDepthFrames(const QString &source_path)
{
    const QDir directory = sourceDirectory(source_path);
    QVector<DepthFrameArtifact> frames;
    QJsonObject manifest;
    if (readJsonObject(directory.filePath(QStringLiteral("mvs_manifest.json")), &manifest))
    {
        const int manifest_algorithm_revision =
            manifest.value(QStringLiteral("algorithm_revision")).toInt(0);
        for (const QJsonValue &value : manifest.value(QStringLiteral("frames")).toArray())
        {
            const QJsonObject object = value.toObject();
            const QString status = object.value(QStringLiteral("status")).toString();
            if (!status.isEmpty() && status != QStringLiteral("completed"))
            {
                continue;
            }
            DepthFrameArtifact frame;
            frame.refIndex = object.value(QStringLiteral("ref_index")).toInt(-1);
            frame.refImage = resolveArtifactPath(directory, object.value(QStringLiteral("ref_image")).toString());
            frame.depthPath = resolveArtifactPath(directory, object.value(QStringLiteral("raw_depth_path")).toString());
            frame.confidencePath = resolveArtifactPath(
                directory, object.value(QStringLiteral("raw_confidence_path")).toString());
            frame.geometrySupportPath = resolveArtifactPath(
                directory, object.value(QStringLiteral("raw_geometry_support_path")).toString());
            frame.geometrySourceMaskPath = resolveArtifactPath(
                directory, object.value(QStringLiteral("raw_geometry_source_mask_path")).toString());
            frame.adaptiveGeometrySupportWeightPath = resolveArtifactPath(
                directory,
                object.value(QStringLiteral(
                    "raw_adaptive_geometry_support_weight_path")).toString());
            frame.adaptiveGeometryEffectiveViewCountPath = resolveArtifactPath(
                directory,
                object.value(QStringLiteral(
                    "raw_adaptive_geometry_effective_view_count_path")).toString());
            frame.adaptiveGeometryConflictRatioPath = resolveArtifactPath(
                directory,
                object.value(QStringLiteral(
                    "raw_adaptive_geometry_conflict_ratio_path")).toString());
            frame.adaptiveGeometryConflictWeightPath = resolveArtifactPath(
                directory,
                object.value(QStringLiteral(
                    "raw_adaptive_geometry_conflict_weight_path")).toString());
            frame.inverseDepthMeanPath = resolveArtifactPath(
                directory, object.value(QStringLiteral("raw_inverse_depth_mean_path")).toString());
            frame.inverseDepthSpreadPath = resolveArtifactPath(
                directory, object.value(QStringLiteral("raw_inverse_depth_spread_path")).toString());
            frame.crossViewRepairedMaskPath = resolveArtifactPath(
                directory, object.value(QStringLiteral("cross_view_repaired_mask_path")).toString());
            frame.depthProvenancePath = resolveArtifactPath(
                directory,
                object.value(QStringLiteral("depth_provenance_path")).toString());
            for (const QJsonValue &source_value :
                 object.value(QStringLiteral("source_indices")).toArray())
            {
                frame.sourceIndices.push_back(source_value.toInt(-1));
            }
            frame.previewPath = resolveArtifactPath(directory, object.value(QStringLiteral("depth_png")).toString());
            frame.validMaskPath = resolveArtifactPath(
                directory, object.value(QStringLiteral("valid_mask_path")).toString());
            frame.supportMaskPath = resolveArtifactPath(
                directory, object.value(QStringLiteral("support_mask_path")).toString());
            frame.status = status;
            frame.sceneProfile = object.value(QStringLiteral("scene_profile")).toString();
            frame.algorithmRevision = object.value(
                QStringLiteral("algorithm_revision")).toInt(
                    manifest_algorithm_revision);
            const QJsonObject depthQuality = object.value(QStringLiteral("depth_quality")).toObject();
            const QJsonObject qualityDecision = object.value(QStringLiteral("quality_decision")).toObject();
            frame.acceptance = object.value(QStringLiteral("acceptance")).toString(
                qualityDecision.value(QStringLiteral("acceptance")).toString());
            frame.fusionEligible = object.contains(QStringLiteral("fusion_eligible"))
                ? object.value(QStringLiteral("fusion_eligible")).toBool()
                : frame.acceptance != QStringLiteral("rejected");
            frame.validCoverage = object.value(QStringLiteral("valid_coverage")).toDouble(
                depthQuality.value(QStringLiteral("valid_coverage")).toDouble(-1.0));
            const QJsonObject depthCompleteness =
                object.value(QStringLiteral("depth_completeness")).toObject();
            frame.validWithinMaskRatio =
                object.value(QStringLiteral("valid_within_mask_ratio")).toDouble(
                    depthCompleteness.value(
                        QStringLiteral("valid_within_mask_ratio")).toDouble(-1.0));
            frame.consistencyRetentionRatio =
                depthCompleteness.value(
                    QStringLiteral("consistency_retention_ratio")).toDouble(-1.0);
            frame.largestComponentRatio =
                depthQuality.value(
                    QStringLiteral("largest_component_ratio")).toDouble(-1.0);
            frame.meanConfidence = object.value(QStringLiteral("depth_confidence_mean")).toDouble(
                depthQuality.value(QStringLiteral("mean_confidence")).toDouble(-1.0));
            frame.sourceViewCount = object.value(QStringLiteral("source_view_count")).toInt(
                depthQuality.value(QStringLiteral("source_view_count")).toInt(0));
            for (const QJsonValue &reason :
                 qualityDecision.value(QStringLiteral("reasons")).toArray())
            {
                const QString text = reason.toString().trimmed();
                if (!text.isEmpty())
                {
                    frame.qualityReasons.append(text);
                }
            }
            frame.gridWidth = object.value(QStringLiteral("grid_width")).toInt();
            frame.gridHeight = object.value(QStringLiteral("grid_height")).toInt();
            frame.hasCameraModel = parseCameraModel(
                object.value(QStringLiteral("camera_model")).toObject(), &frame.cameraModel);
            const int full_grid_width = frame.gridWidth;
            const int full_grid_height = frame.gridHeight;
            if (!QFileInfo::exists(frame.depthPath) && selectExistingPyramidArtifact(
                    directory,
                    object,
                    &frame.depthPath,
                    &frame.confidencePath,
                    &frame.previewPath,
                    &frame.validMaskPath,
                    &frame.gridWidth,
                    &frame.gridHeight))
            {
                if (frame.hasCameraModel && full_grid_width > 0 && full_grid_height > 0)
                {
                    frame.cameraModel = frame.cameraModel.scaledIntrinsics(
                        static_cast<double>(frame.gridWidth) / full_grid_width,
                        static_cast<double>(frame.gridHeight) / full_grid_height);
                }
                frame.pyramidFallback = true;
            }
            if (QFileInfo::exists(frame.depthPath))
            {
                frames.push_back(frame);
            }
        }
    }

    if (frames.isEmpty())
    {
        const QStringList depth_files = directory.entryList(
            QStringList() << QStringLiteral("depth_*.bin"), QDir::Files, QDir::Name);
        for (const QString &file_name : depth_files)
        {
            static const QRegularExpression pyramid_artifact_pattern(
                QStringLiteral("^depth_\\d+_level_\\d+(?:_.*)?\\.bin$"),
                QRegularExpression::CaseInsensitiveOption);
            if (pyramid_artifact_pattern.match(file_name).hasMatch() ||
                file_name.endsWith(QStringLiteral("_conf.bin")) ||
                file_name.endsWith(QStringLiteral("_geometry_support.bin")) ||
                file_name.endsWith(QStringLiteral("_geometry_source_mask.bin")) ||
                file_name.endsWith(QStringLiteral("_adaptive_geometry_support_weight.bin")) ||
                file_name.endsWith(QStringLiteral("_adaptive_geometry_effective_view_count.bin")) ||
                file_name.endsWith(QStringLiteral("_adaptive_geometry_conflict_ratio.bin")) ||
                file_name.endsWith(QStringLiteral("_adaptive_geometry_conflict_weight.bin")) ||
                file_name.endsWith(QStringLiteral("_inverse_depth_mean.bin")) ||
                file_name.endsWith(QStringLiteral("_inverse_depth_spread.bin")) ||
                file_name.endsWith(QStringLiteral("_support.bin")) ||
                file_name.endsWith(QStringLiteral("_uncertainty.bin")))
            {
                continue;
            }
            DepthFrameArtifact frame;
            frame.depthPath = directory.filePath(file_name);
            const QString base_name = QFileInfo(file_name).completeBaseName();
            frame.confidencePath = directory.filePath(base_name + QStringLiteral("_conf.bin"));
            frame.previewPath = directory.filePath(base_name + QStringLiteral(".png"));
            frame.validMaskPath = directory.filePath(base_name + QStringLiteral("_mask.png"));
            frames.push_back(frame);
        }
    }
    attachLegacyReportCameras(directory, &frames);
    std::sort(frames.begin(), frames.end(), [](const auto &left, const auto &right)
    {
        return left.refIndex < right.refIndex;
    });
    return frames;
}

DepthMapVisualHullResult DepthMapMeshBuilder::buildVisualHull(
    const QString &source_path,
    int resolution,
    const std::function<void(const QString &, int)> &progress)
{
    return buildVisualHull(source_path, resolution, DepthMapVisualHullOptions{}, progress);
}

DepthMapVisualHullResult DepthMapMeshBuilder::buildVisualHull(
    const QString &source_path,
    int resolution,
    const DepthMapVisualHullOptions &options,
    const std::function<void(const QString &, int)> &progress)
{
    DepthMapVisualHullResult result;
    const QVector<DepthFrameArtifact> frames = discoverDepthFrames(source_path);
    std::vector<VisualHullView> views;
    for (const DepthFrameArtifact &frame : frames)
    {
        if (!frame.hasCameraModel || frame.refImage.isEmpty())
        {
            continue;
        }
        cv::Mat color = xjw::common::io::readImage(
            xjw::common::io::toUtf8Path(frame.refImage), cv::IMREAD_COLOR);
        if (color.empty())
        {
            continue;
        }
        if (frame.gridWidth > 0 && frame.gridHeight > 0 &&
            (color.cols != frame.gridWidth || color.rows != frame.gridHeight))
        {
            cv::resize(color, color, cv::Size(frame.gridWidth, frame.gridHeight));
        }
        float coverage = 0.0f;
        float border_coverage = 1.0f;
        float border_luminance = 255.0f;
        cv::Mat silhouette = buildSilhouetteMask(
            color, &coverage, &border_coverage, &border_luminance);
        const bool has_dark_studio_background = border_luminance <= 55.0f;
        if (silhouette.empty() || !has_dark_studio_background ||
            coverage < 0.03f || coverage > 0.80f || border_coverage > 0.30f)
        {
            continue;
        }
        if (countEnclosedMaskHoles(silhouette) > 0)
        {
            ++result.preservedSilhouetteHoleViewCount;
        }
        VisualHullView view;
        view.camera = frame.cameraModel;
        view.silhouetteMask = silhouette;
        view.colorImage = color;
        view.imagePath = xjw::common::io::toUtf8Path(frame.refImage);
        cv::Mat depth;
        if (!frame.depthPath.isEmpty() &&
            xjw::core::project::loadDepthMatStorage(frame.depthPath, &depth).ok &&
            !depth.empty())
        {
            if (depth.size() != silhouette.size())
            {
                cv::resize(depth, depth, silhouette.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            depth.setTo(0.0f, silhouette == 0);
            view.depthMap = depth;
            ++result.depthViewCount;
        }
        views.push_back(std::move(view));
    }

    result.usableViewCount = static_cast<int>(views.size());
    result.applicable = views.size() >= 6;
    if (!result.applicable)
    {
        result.message = QStringLiteral("清晰前景轮廓视图不足，继续使用通用点云网格路径");
        return result;
    }

    VisualHullConfig config;
    if (!estimateBounds(frames, &config.boundsMin, &config.boundsMax))
    {
        result.message = QStringLiteral("无法从深度图估计视觉外壳范围");
        return result;
    }
    config.resolution = std::clamp(resolution, 96, 384);
    config.minimumVisibleViews = std::max(4, result.usableViewCount / 3);
    config.allowedSilhouetteViolations = std::max(1, result.usableViewCount / 10);
    config.useContinuousSilhouetteField =
        options.useContinuousSilhouetteField;
    config.topologyClosingIterations =
        options.topologyClosingIterations >= 0
        ? std::clamp(options.topologyClosingIterations, 0, 3)
        : (config.useContinuousSilhouetteField ? 0 : 2);
    result.topologyClosingIterations =
        config.topologyClosingIterations;
    config.smoothingIterations =
        std::clamp(options.smoothingIterations, 0, 20);
    config.smoothingLambda =
        std::clamp(options.smoothingLambda, 0.0f, 0.49f);
    const int minimum_depth_views = std::max(4, result.usableViewCount / 3);
    config.enableDepthFreeSpaceCarving =
        options.strictVolumetricMasks && result.depthViewCount >= minimum_depth_views;
    config.minimumDepthFreeSpaceViolations = std::max(4, result.usableViewCount / 3);
    config.relativeDepthTolerance = 0.03f;
    if (progress)
    {
        config.progressFn = [progress](const std::string &stage, float fraction)
        {
            progress(QString::fromStdString(stage), static_cast<int>(fraction * 100.0f));
        };
        progress(QStringLiteral("正在按多视轮廓重建物体表面..."), 5);
    }
    std::string error;
    result.ok = VisualHullReconstructor::reconstruct(views, config, &result.mesh, &error);
    result.usedDepthFreeSpaceCarving = config.enableDepthFreeSpaceCarving;
    if (result.ok)
    {
        result.connectivity = VisualHullReconstructor::analyzeConnectivity(result.mesh);
    }

    if (result.ok && config.enableDepthFreeSpaceCarving &&
        VisualHullReconstructor::requiresSilhouetteOnlyRetry(
            result.connectivity,
            options.minimumLargestComponentFaceRatio,
            options.maximumConnectedComponents))
    {
        result.fallbackReason = QStringLiteral(
            "深度雕刻结果碎片化：连通分量 %1，最大分量占比 %2%")
                                    .arg(result.connectivity.componentCount)
                                    .arg(result.connectivity.largestComponentFaceRatio * 100.0,
                                         0,
                                         'f',
                                         1);
        if (progress)
        {
            progress(QStringLiteral("深度雕刻结果碎片化，正在回退到轮廓视觉外壳..."), 5);
        }
        VisualHullConfig fallback_config = config;
        fallback_config.enableDepthFreeSpaceCarving = false;
        TriMesh fallback_mesh;
        std::string fallback_error;
        if (VisualHullReconstructor::reconstruct(
                views, fallback_config, &fallback_mesh, &fallback_error))
        {
            const MeshConnectivityStats fallback_connectivity =
                VisualHullReconstructor::analyzeConnectivity(fallback_mesh);
            if (fallback_connectivity.largestComponentFaceRatio >=
                result.connectivity.largestComponentFaceRatio)
            {
                result.mesh = std::move(fallback_mesh);
                result.connectivity = fallback_connectivity;
                result.usedDepthFreeSpaceCarving = false;
                result.retriedWithoutDepthCarving = true;
            }
        }
    }

    if (result.ok && result.connectivity.componentCount > 1 &&
        result.connectivity.largestComponentFaceRatio >=
            options.minimumLargestComponentFaceRatio)
    {
        result.removedSatelliteComponentCount = result.connectivity.componentCount - 1;
        if (VisualHullReconstructor::retainLargestConnectedComponent(&result.mesh))
        {
            result.connectivity = VisualHullReconstructor::analyzeConnectivity(result.mesh);
        }
    }

    if (result.ok && VisualHullReconstructor::requiresSilhouetteOnlyRetry(
            result.connectivity,
            options.minimumLargestComponentFaceRatio,
            options.maximumConnectedComponents))
    {
        result.qualityRejected = true;
        result.ok = false;
        const QString quality_reason = QStringLiteral(
            "多视轮廓与相机姿态不一致：连通分量 %1，最大分量占比仅 %2%（要求至少 %3%）")
                                           .arg(result.connectivity.componentCount)
                                           .arg(result.connectivity.largestComponentFaceRatio * 100.0,
                                                0,
                                                'f',
                                                1)
                                           .arg(options.minimumLargestComponentFaceRatio * 100.0,
                                                0,
                                                'f',
                                                0);
        result.fallbackReason = result.fallbackReason.isEmpty()
            ? quality_reason
            : result.fallbackReason + QStringLiteral("；") + quality_reason;
    }

    result.actualAlgorithm = result.usedDepthFreeSpaceCarving
        ? QStringLiteral("depth_constrained_visual_hull")
        : QStringLiteral("silhouette_visual_hull");
    result.message = result.qualityRejected
        ? result.fallbackReason
        : (result.ok
        ? QStringLiteral("已使用 %1 个轮廓视图重建视觉外壳（最大连通分量 %2%）")
              .arg(result.usableViewCount)
              .arg(result.connectivity.largestComponentFaceRatio * 100.0, 0, 'f', 1)
        : QString::fromStdString(error));
    return result;
}

QString DepthMapMeshBuilder::resolveReusableDenseCloud(const QString &source_path, QString *error_message)
{
    const QDir directory = sourceDirectory(source_path);
    const QString dense_path = directory.filePath(QStringLiteral("dense_cloud.ply"));
    if (QFileInfo::exists(dense_path))
    {
        return QDir::cleanPath(dense_path);
    }
    if (error_message)
    {
        *error_message = QStringLiteral("未找到可复用的深度图融合点云: %1").arg(dense_path);
    }
    return QString();
}

} // namespace xjw::mesh
