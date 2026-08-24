#include "DepthMapMeshBuilder.h"

#include "FramePinholeCamera.h"
#include "DepthFrameUtils.h"
#include "StudioForegroundMask.h"
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
#include <cstdint>
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

bool hasConsistentUsableSceneProfile(
    const QVector<DepthFrameArtifact> &frames)
{
    QString canonical_scene_profile;
    for (const DepthFrameArtifact &frame : frames)
    {
        if (frame.role == xjw::mvs::DepthFrameRole::Excluded)
        {
            continue;
        }
        if (!xjw::mvs::extendCanonicalDepthSceneProfileBatch(
                frame.sceneProfile, &canonical_scene_profile))
        {
            return false;
        }
    }
    return !canonical_scene_profile.isEmpty();
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

bool parseCameraModel(const QJsonObject &object, FramePinholeCamera *camera)
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
    FramePinholeCamera parsed;
    parsed.setIntrinsics(focalX,
                         focalY,
                         object.value(QStringLiteral("cx")).toDouble(),
                         object.value(QStringLiteral("cy")).toDouble());
    parsed.setPose(cameraToWorld, center);
    parsed.setDistortion(FramePinholeCamera::Distortion{});
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
        FramePinholeCamera camera;
        if (!camera.loadFromFile(xjw::common::io::toUtf8Path(it->second)))
        {
            continue;
        }
        FramePinholeCamera model = camera.normalizedForPositiveDepth();
        model.setDistortion(FramePinholeCamera::Distortion{});
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

bool isUsableStudioSilhouette(const cv::Mat &color_image)
{
    return buildStudioForegroundMask(color_image).isUsable();
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
        if (!xjw::mvs::isPrimaryFusionFrame(frame.role) ||
            !frame.hasCameraModel || frame.depthPath.isEmpty())
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
                // OpenCV's calibrated projection uses integer coordinates for
                // pixel centres.  Adding half a pixel here shifts every visual-
                // hull ray away from the depth/TSDF projection convention.
                const double pixel[2] = {
                    static_cast<double>(column),
                    static_cast<double>(row)};
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
            DepthFrameArtifact frame;
            frame.refIndex = object.value(QStringLiteral("ref_index")).toInt(-1);
            frame.sourceImage = resolveArtifactPath(
                directory,
                object.value(QStringLiteral("ref_image")).toString());
            const QString prepared_image = object.value(
                QStringLiteral("prepared_image")).toString();
            frame.refImage = resolveArtifactPath(
                directory,
                prepared_image.isEmpty()
                    ? object.value(QStringLiteral("ref_image")).toString()
                    : prepared_image);
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
            for (const QJsonValue &source_value :
                 object.value(QStringLiteral(
                     "geometry_source_indices")).toArray())
            {
                frame.geometrySourceIndices.push_back(
                    source_value.toInt(-1));
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
            const xjw::mvs::MvsDepthFrameQualification qualification =
                xjw::mvs::qualifyMvsDepthFrameArtifact(object);
            frame.acceptance = qualification.acceptance;
            frame.fusionEligible = qualification.fusionEligible;
            frame.fusionEligibilityKnown =
                qualification.fusionEligibilityKnown;
            frame.role = qualification.role;
            frame.useDiscreteGeometryFallback =
                qualification.useDiscreteGeometryFallback;
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
            frame.sparseAbsoluteDepthMedianLogError =
                qualityDecision.value(
                    QStringLiteral("sparse_absolute_depth_residual"))
                    .toObject()
                    .value(QStringLiteral("median_absolute_log_error"))
                    .toDouble(-1.0);
            frame.sourceViewCount = object.value(QStringLiteral("source_view_count")).toInt(
                depthQuality.value(QStringLiteral("source_view_count")).toInt(0));
            const QJsonObject geometryEvidence = object.value(
                QStringLiteral("geometry_evidence_diagnostics")).toObject();
            const double geometry_valid_pixel_count = geometryEvidence.value(
                QStringLiteral("valid_pixel_count")).toDouble(0.0);
            const double discrete_core_ratio = geometryEvidence.value(
                QStringLiteral("discrete_geometry_core_ratio")).toDouble(-1.0);
            if (std::isfinite(geometry_valid_pixel_count) &&
                geometry_valid_pixel_count > 0.0 &&
                std::isfinite(discrete_core_ratio) &&
                discrete_core_ratio >= 0.0)
            {
                frame.trustedGeometryCorePixelCount =
                    static_cast<std::uint64_t>(std::llround(
                        geometry_valid_pixel_count *
                        std::clamp(discrete_core_ratio, 0.0, 1.0)));
            }
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

DepthMapVisualHullPreflightResult DepthMapMeshBuilder::inspectVisualHullApplicability(
    const QString &source_path,
    int maximum_inspected_frames,
    int minimum_usable_views)
{
    DepthMapVisualHullPreflightResult result;
    const QVector<DepthFrameArtifact> frames = discoverDepthFrames(source_path);
    if (!hasConsistentUsableSceneProfile(frames))
    {
        return result;
    }
    QVector<const DepthFrameArtifact *> candidates;
    candidates.reserve(frames.size());
    for (const DepthFrameArtifact &frame : frames)
    {
        if (xjw::mvs::isPrimaryFusionFrame(frame.role) &&
            frame.hasCameraModel && !frame.refImage.isEmpty())
        {
            candidates.push_back(&frame);
        }
    }

    result.candidateFrameCount = candidates.size();
    const int inspection_limit = std::min(
        std::max(1, maximum_inspected_frames),
        result.candidateFrameCount);
    const int required_views = std::max(1, minimum_usable_views);
    for (int sample_index = 0; sample_index < inspection_limit; ++sample_index)
    {
        const int candidate_index = static_cast<int>(
            static_cast<std::int64_t>(sample_index) * result.candidateFrameCount /
            inspection_limit);
        const DepthFrameArtifact &frame = *candidates[candidate_index];
        const cv::Mat color = xjw::common::io::readImage(
            xjw::common::io::toUtf8Path(frame.refImage),
            cv::IMREAD_REDUCED_COLOR_8);
        ++result.inspectedFrameCount;
        if (!color.empty() && isUsableStudioSilhouette(color))
        {
            ++result.usableViewCount;
            if (result.usableViewCount >= required_views)
            {
                result.applicable = true;
                break;
            }
        }
    }
    return result;
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
    if (!hasConsistentUsableSceneProfile(frames))
    {
        result.message = QStringLiteral(
            "深度图批次的 scene_profile 缺失、无法识别或不一致，"
            "不能安全构建视觉外壳");
        return result;
    }
    std::vector<VisualHullView> views;
    for (const DepthFrameArtifact &frame : frames)
    {
        if (!xjw::mvs::isPrimaryFusionFrame(frame.role) ||
            !frame.hasCameraModel || frame.refImage.isEmpty())
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
        StudioForegroundMask foreground = buildStudioForegroundMask(color);
        if (!foreground.isUsable())
        {
            continue;
        }
        cv::Mat silhouette = std::move(foreground.mask);
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
    config.computeBackend = options.computeBackend;
    config.computeDeviceIndex = options.computeDeviceIndex;
    config.gpuSlabDepth = options.gpuSlabDepth;
    if (progress)
    {
        config.progressFn = [progress](const std::string &stage, float fraction)
        {
            progress(QString::fromStdString(stage), static_cast<int>(fraction * 100.0f));
        };
        progress(QStringLiteral("正在按多视轮廓重建物体表面..."), 5);
    }
    std::string error;
    result.ok = VisualHullReconstructor::reconstruct(
        views, config, &result.mesh, &error, &result.executionInfo);
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
        VisualHullExecutionInfo fallback_execution_info;
        if (VisualHullReconstructor::reconstruct(
                views, fallback_config, &fallback_mesh, &fallback_error, &fallback_execution_info))
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
                result.executionInfo = std::move(fallback_execution_info);
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
