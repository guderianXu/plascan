#include "ModelWorkflowService.h"

#include "DepthMapMeshBuilder.h"
#include "DepthConstrainedSurfaceRefiner.h"
#include "DepthMeshCompleteness.h"
#include "DepthTsdfSurfaceBuilder.h"
#include "Mc33IsoSurfaceExtractor.h"
#include "MeshColorizer.h"
#include "MeshIsotropicRemesher.h"
#include "MeshQuadricSimplifier.h"
#include "MeshTopologyQuality.h"
#include "OpenMeshSimplifier.h"
#include "SurfaceReconstructor.h"
#include "SurfaceReconstructorPostprocess.h"
#include "VisualHullDepthRefiner.h"
#include "VisibilityOccupancyCarrierFairer.h"
#include "VisibilityOccupancyCarrierFieldProjector.h"
#include "VisibilityOccupancyCarrierSubdivider.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QTextStream>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace xjw::mesh::workflow
{

namespace
{

struct RefinementQualityGuardResult
{
    bool applied = false;
    bool limited = false;
    float acceptedBlend = 1.0f;
    double areaBefore = 0.0;
    double areaAfter = 0.0;
    double normalVariationBefore = 0.0;
    double normalVariationAfter = 0.0;
};

struct FinalSurfaceDenoisingResult
{
    bool attempted = false;
    bool accepted = false;
    int movedVertexCount = 0;
    double areaBefore = 0.0;
    double areaAfter = 0.0;
    MeshTopologyQualityStatistics qualityBefore;
    MeshTopologyQualityStatistics qualityAfter;
};

bool hasSameFaceIndexBuffer(const TriMesh &first, const TriMesh &second)
{
    if (first.faces.size() != second.faces.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < first.faces.size(); ++index)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            if (first.faces[index].v[corner] !=
                second.faces[index].v[corner])
            {
                return false;
            }
        }
    }
    return true;
}

QJsonArray eulerCharacteristicsToJson(const std::vector<int> &values)
{
    QJsonArray json;
    for (const int value : values)
    {
        json.append(value);
    }
    return json;
}

double meshSurfaceArea(const TriMesh &mesh)
{
    double area = 0.0;
    for (const Triangle &face : mesh.faces)
    {
        const MeshVertex &first = mesh.vertices[
            static_cast<std::size_t>(face.v[0])];
        const MeshVertex &second = mesh.vertices[
            static_cast<std::size_t>(face.v[1])];
        const MeshVertex &third = mesh.vertices[
            static_cast<std::size_t>(face.v[2])];
        const double ab_x = second.x - first.x;
        const double ab_y = second.y - first.y;
        const double ab_z = second.z - first.z;
        const double ac_x = third.x - first.x;
        const double ac_y = third.y - first.y;
        const double ac_z = third.z - first.z;
        const double cross_x = ab_y * ac_z - ab_z * ac_y;
        const double cross_y = ab_z * ac_x - ab_x * ac_z;
        const double cross_z = ab_x * ac_y - ab_y * ac_x;
        area += 0.5 * std::sqrt(
            cross_x * cross_x +
            cross_y * cross_y +
            cross_z * cross_z);
    }
    return area;
}

double meshMeanNormalVariation(const TriMesh &mesh)
{
    double variation_sum = 0.0;
    std::uint64_t sample_count = 0;
    for (const Triangle &face : mesh.faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const MeshVertex &first = mesh.vertices[
                static_cast<std::size_t>(face.v[corner])];
            const MeshVertex &second = mesh.vertices[
                static_cast<std::size_t>(face.v[(corner + 1) % 3])];
            const double dot = std::clamp(
                static_cast<double>(
                    first.nx * second.nx +
                    first.ny * second.ny +
                    first.nz * second.nz),
                -1.0,
                1.0);
            variation_sum += 1.0 - dot;
            ++sample_count;
        }
    }
    return sample_count > 0
        ? variation_sum / static_cast<double>(sample_count)
        : 0.0;
}

RefinementQualityGuardResult limitRefinementRoughness(
    const TriMesh &original,
    TriMesh *refined,
    double maximumAreaGrowth,
    double maximumNormalVariationGrowth)
{
    RefinementQualityGuardResult result;
    if (refined == nullptr ||
        original.vertices.size() != refined->vertices.size() ||
        original.faces.size() != refined->faces.size())
    {
        return result;
    }

    result.applied = true;
    result.areaBefore = meshSurfaceArea(original);
    result.normalVariationBefore =
        meshMeanNormalVariation(original);
    const auto acceptable = [&](const TriMesh &candidate)
    {
        const double candidate_area = meshSurfaceArea(candidate);
        const double candidate_variation =
            meshMeanNormalVariation(candidate);
        const double area_limit =
            result.areaBefore * maximumAreaGrowth;
        const double variation_limit =
            std::max(
                result.normalVariationBefore *
                    maximumNormalVariationGrowth,
                result.normalVariationBefore + 1.0e-6);
        return candidate_area <= area_limit &&
               candidate_variation <= variation_limit;
    };
    if (!acceptable(*refined))
    {
        const TriMesh full_refinement = *refined;
        bool accepted = false;
        for (const float blend : {0.75f, 0.50f, 0.25f})
        {
            TriMesh candidate = original;
            for (std::size_t index = 0;
                 index < candidate.vertices.size();
                 ++index)
            {
                MeshVertex &vertex = candidate.vertices[index];
                const MeshVertex &target =
                    full_refinement.vertices[index];
                vertex.x += blend * (target.x - vertex.x);
                vertex.y += blend * (target.y - vertex.y);
                vertex.z += blend * (target.z - vertex.z);
            }
            detail::recomputeNormals(&candidate);
            if (acceptable(candidate))
            {
                *refined = std::move(candidate);
                result.acceptedBlend = blend;
                accepted = true;
                break;
            }
        }
        if (!accepted)
        {
            *refined = original;
            result.acceptedBlend = 0.0f;
        }
        result.limited = true;
    }
    result.areaAfter = meshSurfaceArea(*refined);
    result.normalVariationAfter =
        meshMeanNormalVariation(*refined);
    return result;
}

QJsonObject textureResultToJson(
    const xjw::mesh::TextureMappingResult &result,
    const xjw::mesh::TextureMappingConfig *config = nullptr)
{
    QJsonObject object;
    object[QStringLiteral("model_obj")] = xjw::common::io::fromUtf8Path(result.modelObjPath);
    object[QStringLiteral("model_mtl")] = xjw::common::io::fromUtf8Path(result.modelMtlPath);
    object[QStringLiteral("texture_png")] = xjw::common::io::fromUtf8Path(result.texturePngPath);
    object[QStringLiteral("texture_image")] = xjw::common::io::fromUtf8Path(result.texturePngPath);
    object[QStringLiteral("textured")] = !result.modelObjPath.empty()
        && !result.texturePngPath.empty();
    object[QStringLiteral("texture_size")] = result.textureSize;
    object[QStringLiteral("texture_algorithm")] = QString::fromStdString(result.textureAlgorithm);
    object[QStringLiteral("uv_method")] = QString::fromStdString(result.uvMethod);
    object[QStringLiteral("blend_method")] = QString::fromStdString(result.blendMethod);
    object[QStringLiteral("texture_source_view_count")] = result.sourceViewCount;
    object[QStringLiteral("texture_mapped_face_count")] = result.mappedFaceCount;
    object[QStringLiteral("texture_fallback_mapped_face_count")] =
        result.fallbackMappedFaceCount;
    object[QStringLiteral("texture_coherence_adjusted_face_count")] =
        result.coherenceAdjustedFaceCount;
    object[QStringLiteral("texture_unmapped_face_count")] = result.unmappedFaceCount;
    object[QStringLiteral("texture_strict_mapped_face_count")] =
        result.strictMappedFaceCount;
    object[QStringLiteral("texture_chart_count")] = result.chartCount;
    object[QStringLiteral("texture_used_view_count")] = result.usedViewCount;
    object[QStringLiteral("texture_candidate_evaluation_count")] =
        static_cast<qint64>(result.candidateEvaluationCount);
    object[QStringLiteral("texture_rejected_projection_count")] =
        static_cast<qint64>(result.rejectedProjectionCount);
    object[QStringLiteral("texture_rejected_mask_count")] =
        static_cast<qint64>(result.rejectedMaskCount);
    object[QStringLiteral("texture_rejected_depth_count")] =
        static_cast<qint64>(result.rejectedDepthCount);
    object[QStringLiteral("texture_rejected_angle_count")] =
        static_cast<qint64>(result.rejectedAngleCount);
    object[QStringLiteral("texture_rejected_resolution_count")] =
        static_cast<qint64>(result.rejectedResolutionCount);
    object[QStringLiteral("texture_rejected_color_outlier_count")] =
        static_cast<qint64>(result.rejectedColorOutlierCount);
    object[QStringLiteral("texture_atlas_occupancy")] = result.atlasOccupancy;
    object[QStringLiteral("texture_median_texel_density")] =
        result.medianTexelDensity;
    object[QStringLiteral("texture_seam_color_difference")] =
        result.seamColorDifference;
    object[QStringLiteral("texture_peak_memory_estimate_mib")] =
        result.peakMemoryEstimateMiB;
    if (config)
    {
        object[QStringLiteral("effective_texture_image_downscale")] =
            std::clamp(config->imageDownscale, 1, 8);
        object[QStringLiteral("effective_texture_padding")] =
            std::clamp(config->padding, 2, 64);
        object[QStringLiteral("effective_texture_ghost_filter")] =
            config->enableGhostFilter;
        object[QStringLiteral("effective_texture_out_of_focus_filter")] =
            config->enableOutOfFocusFilter;
        object[QStringLiteral("effective_texture_color_correction")] =
            config->enableColorCorrection;
        object[QStringLiteral("effective_texture_sharpening_strength")] =
            std::clamp(config->sharpeningStrength, 0.0f, 2.0f);
        object[QStringLiteral("effective_texture_hole_fill")] =
            config->holeFillMode != xjw::mesh::TextureHoleFillMode::Disabled;
    }
    return object;
}

MeshColorView textureViewFromFrame(const DepthTsdfFrame &frame)
{
    MeshColorView view;
    view.camera = frame.camera;
    view.depth = frame.depth;
    view.confidence = frame.confidence;
    view.depthValidMask = frame.depthValidMask;
    view.supportMask = frame.supportMask;
    view.qualityWeight = frame.frameQualityWeight;

    if (!frame.refImage.isEmpty() && QFileInfo::exists(frame.refImage))
    {
        view.colorBgr = xjw::common::io::readImage(
            xjw::common::io::toUtf8Path(frame.refImage), cv::IMREAD_COLOR);
    }
    if (view.colorBgr.empty())
    {
        view.colorBgr = frame.colorBgr;
    }
    if (!view.colorBgr.empty() && frame.depth.cols > 0 && frame.depth.rows > 0)
    {
        view.colorCamera = frame.camera.scaledIntrinsics(
            static_cast<double>(view.colorBgr.cols) / frame.depth.cols,
            static_cast<double>(view.colorBgr.rows) / frame.depth.rows);
    }
    return view;
}

void assignFinalModelFields(QJsonObject *result, bool exportObj)
{
    if (!result)
    {
        return;
    }

    (*result)[QStringLiteral("requested_export_format")] = exportObj ? QStringLiteral("OBJ") : QStringLiteral("PLY");

    if (exportObj)
    {
        const QString objPath = result->value(QStringLiteral("model_obj")).toString();
        if (!objPath.isEmpty())
        {
            (*result)[QStringLiteral("final_model_format")] = QStringLiteral("OBJ");
            (*result)[QStringLiteral("final_model_path")] = objPath;
            return;
        }
    }

    const QString plyPath = result->value(QStringLiteral("model_ply")).toString();
    (*result)[QStringLiteral("final_model_format")] = QStringLiteral("PLY");
    (*result)[QStringLiteral("final_model_path")] = plyPath;
}

WorkflowResult saveMeshAndOptionalTexture(const xjw::mesh::TriMesh &mesh,
                                          const std::string &mesh_algorithm,
                                          const QString &output_root,
                                          bool export_obj,
                                          const xjw::mesh::TextureMappingConfig &texture,
                                          const std::function<void(const QString &, int)> &progress,
                                          const QVector<MeshColorView> *camera_views = nullptr)
{
    WorkflowResult result;
    const QString products_dir = QDir(output_root).filePath(QStringLiteral("products"));
    QDir().mkpath(products_dir);
    const QString mesh_ply_path = QDir(products_dir).filePath(QStringLiteral("model_from_mesh.ply"));
    std::string mesh_error;
    if (!mesh.savePLY(xjw::common::io::toUtf8Path(mesh_ply_path), &mesh_error))
    {
        result.errorMessage = QStringLiteral("网格保存失败: %1").arg(QString::fromStdString(mesh_error));
        return result;
    }

    result.payload[QStringLiteral("mesh_ply")] = mesh_ply_path;
    result.payload[QStringLiteral("model_ply")] = mesh_ply_path;
    result.payload[QStringLiteral("vertex_count")] = mesh.vertexCount();
    result.payload[QStringLiteral("face_count")] = mesh.faceCount();
    result.payload[QStringLiteral("has_vertex_colors")] = mesh.hasVertexColors;
    if (mesh.hasVertexColors)
    {
        result.payload[QStringLiteral("vertex_color_format")] =
            QStringLiteral("3波段, uint8");
    }
    result.payload[QStringLiteral("mesh_algorithm")] =
        QString::fromStdString(mesh_algorithm.empty() ? "unknown" : mesh_algorithm);

    if (export_obj)
    {
        std::string texture_error;
        xjw::mesh::TextureMappingConfig texture_config = texture;
        if (progress)
        {
            texture_config.progressFn = [progress](const std::string &stage, int percent)
            {
                progress(QString::fromStdString(stage), percent);
            };
        }
        xjw::mesh::TextureMappingResult texture_result;
        const bool texture_ok = camera_views && !camera_views->empty()
            ? xjw::mesh::TextureMapper::generateCameraTexturedModelFromMeshFile(
                  xjw::common::io::toUtf8Path(mesh_ply_path),
                  xjw::common::io::toUtf8Path(products_dir),
                  texture_config,
                  *camera_views,
                  &texture_result,
                  &texture_error)
            : xjw::mesh::TextureMapper::generateTexturedModelFromMeshFile(
                  xjw::common::io::toUtf8Path(mesh_ply_path),
                  xjw::common::io::toUtf8Path(products_dir),
                  texture_config,
                  &texture_result,
                  &texture_error);
        if (texture_ok)
        {
            const QJsonObject texture_json =
                textureResultToJson(texture_result, &texture_config);
            for (auto it = texture_json.begin(); it != texture_json.end(); ++it)
            {
                result.payload[it.key()] = it.value();
            }
        }
        else if (!texture_error.empty())
        {
            result.payload[QStringLiteral("texture_warning")] = QString::fromStdString(texture_error);
        }
    }

    assignFinalModelFields(&result.payload, export_obj);
    result.ok = true;
    return result;
}

bool interpolationIsDisabled(const QJsonObject &settings)
{
    return settings.value(QStringLiteral("interpolation"))
               .toString(QStringLiteral("enabled")) ==
        QStringLiteral("disabled");
}

void enforceObservationOnlySurfacePolicy(
    const QJsonObject &settings,
    xjw::mesh::DepthTsdfOptions *options)
{
    if (!options || !interpolationIsDisabled(settings))
    {
        return;
    }

    // “插值禁用” must dominate automatic orbital completion defaults.  Keep
    // only surfaces supported by integrated depth evidence; otherwise the GUI
    // says interpolation is off while occupancy/visual-hull recovery can still
    // bridge unobserved regions with large triangle sheets.
    options->fillSmallBoundaryHoles = false;
    options->enableSilhouetteAwareFinalHoleFill = false;
    options->enableVisibilityConstrainedFinalHoleFill = false;
    options->enableTinyBoundaryLoopCollapse = false;
    options->enableVisibilityOccupancyCompletion = false;
    options->enableVisualHullSignedDistanceCompletion = false;
    options->enableOrbitalGapBoundaryRecovery = false;
    options->enableOrbitalGapAdaptiveTruncation = false;
    options->enableGeometryZeroCrossingRecovery = false;
    options->enableCrossViewAnchoredSurfaceRecovery = false;
    options->enableGeometryZeroCrossingCellSheets = false;
    options->enableContourBandZeroCrossingSupport = false;
    options->enableSurfacePatchSupport = false;
    options->enableGeometryVerifiedBoundaryRecovery = false;
    options->allowInvalidNearestPixelRecovery = false;
    options->excludeAnchoredInterpolationObservations = true;
    options->adaptiveTgvRecoverUnsupportedSamples = false;
    options->implicitRegularizationRecoverAxialGaps = false;
    // Observation-only output is expected to retain genuine gaps in the
    // measured surface. Keep completeness diagnostics for the report, but do
    // not reject an otherwise usable mesh against thresholds designed for a
    // completed/interpolated surface.
    options->enforceDepthCompletenessGate = false;
    // "No interpolation" disables every source of synthetic geometry above,
    // but it must still allow the iso-surface extractor to interpolate a zero
    // crossing inside the already fused TSDF cell. Requiring both signs to
    // carry direct sample support deleted most MC33 faces at narrow-band
    // boundaries and turned valid measured surfaces into holes.
    options->mc33RequireSupportedSignChange = false;
}

xjw::mesh::DepthTsdfOptions makeDepthTsdfOptions(const QJsonObject &settings,
                                                 int requested_resolution)
{
    xjw::mesh::DepthTsdfOptions options;
    options.resolution = settings.contains(QStringLiteral("meshResolution"))
        ? meshResolutionFromSettings(settings)
        : requested_resolution;
    options.calculateVertexColors =
        settings.value(QStringLiteral("calculateVertexColors")).toBool(true);
    options.compensateColorExposure = settings.value(
        QStringLiteral("tsdfCompensateColorExposure")).toBool(false);
    options.coherentFacePrimaryViewColors = settings.value(
        QStringLiteral("tsdfCoherentFacePrimaryViewColors")).toBool(false);
    options.simplifyTargetFaces = qBound(
        0,
        settings.value(QStringLiteral("simplifyTargetFaces"))
            .toInt(settings.value(QStringLiteral("targetFaces")).toInt(0)),
        2000000);
    options.enableQuadricSimplification = settings.contains(
        QStringLiteral("tsdfQuadricSimplification"))
        ? settings.value(QStringLiteral("tsdfQuadricSimplification")).toBool()
        : options.simplifyTargetFaces > 0;
    const bool arbitrary_surface =
        settings.value(QStringLiteral("surface_type"))
                .toString(QStringLiteral("arbitrary_3d")) ==
            QStringLiteral("arbitrary_3d");
    const bool automatic_openmesh_simplification =
        openMeshSimplifierAvailable() &&
        options.resolution >= 320 &&
        options.simplifyTargetFaces > 0 &&
        options.simplifyTargetFaces <= 240000 &&
        arbitrary_surface;
    options.enableOpenMeshSimplification = settings.value(
        QStringLiteral("tsdfOpenMeshSimplification")).toBool(
            automatic_openmesh_simplification);
    options.openMeshMaximumNormalDeviationDegrees = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOpenMeshMaximumNormalDeviationDegrees")).toDouble(180.0)),
        1.0f,
        180.0f);
    options.openMeshMaximumNormalFlippingDegrees = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOpenMeshMaximumNormalFlippingDegrees")).toDouble(75.0)),
        1.0f,
        180.0f);
    options.openMeshSmoothingIterations = qBound(
        0,
        settings.value(QStringLiteral("tsdfOpenMeshSmoothingIterations"))
            .toInt(options.enableOpenMeshSimplification ? 12 : 2),
        20);
    options.openMeshSmoothingMaximumDisplacementVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOpenMeshSmoothingMaximumDisplacementVoxels"))
                               .toDouble(
                                   options.enableOpenMeshSimplification
                                       ? 1.60
                                       : 0.40)),
        0.0f,
        2.0f);
    options.openMeshSmoothingFeatureAngleDegrees = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOpenMeshSmoothingFeatureAngleDegrees"))
                               .toDouble(
                                   options.enableOpenMeshSimplification
                                       ? 175.0
                                       : 120.0)),
        1.0f,
        180.0f);
    options.openMeshNotificationInterval = qBound(
        1,
        settings.value(QStringLiteral("tsdfOpenMeshNotificationInterval"))
            .toInt(4096),
        1000000);
    options.enableVoxelFallbackSimplification = settings.value(
        QStringLiteral("tsdfVoxelFallbackSimplification")).toBool(true);
    const bool automatic_voxel_fallback_qem_polish =
        options.resolution >= 384 &&
        options.simplifyTargetFaces > 0 &&
        options.simplifyTargetFaces <= 240000 &&
        settings.value(QStringLiteral("surface_type"))
                .toString(QStringLiteral("arbitrary_3d")) ==
            QStringLiteral("arbitrary_3d");
    options.enableVoxelFallbackQemPolish = settings.value(
        QStringLiteral("tsdfVoxelFallbackQemPolish")).toBool(
            automatic_voxel_fallback_qem_polish);
    options.voxelFallbackMinimumProtectedBoundaryVertices = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVoxelFallbackMinimumProtectedBoundaryVertices")).toInt(1),
        256);
    options.voxelFallbackMaximumCollapsibleBoundaryDiameterVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVoxelFallbackMaximumCollapsibleBoundaryDiameterVoxels"))
                               .toDouble(0.0)),
        0.0f,
        16.0f);
    options.voxelFallbackMaximumNormalClusterAngleDegrees = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVoxelFallbackMaximumNormalClusterAngleDegrees"))
                               .toDouble(180.0)),
        5.0f,
        180.0f);
    options.voxelFallbackInitialClusterFactor = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVoxelFallbackInitialClusterFactor"))
                               .toDouble(1.0)),
        1.0f,
        4.0f);
    options.enableVoxelFallbackMultiViewSilhouetteProtection = settings.value(
        QStringLiteral("tsdfVoxelFallbackMultiViewSilhouetteProtection"))
            .toBool(true);
    options.voxelFallbackMinimumSilhouetteViews = qBound(
        1,
        settings.value(QStringLiteral("tsdfVoxelFallbackMinimumSilhouetteViews"))
            .toInt(2),
        8);
    options.voxelFallbackSilhouetteBandPixels = qBound(
        1,
        settings.value(QStringLiteral("tsdfVoxelFallbackSilhouetteBandPixels"))
            .toInt(2),
        8);
    options.voxelFallbackSilhouetteDepthToleranceVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVoxelFallbackSilhouetteDepthToleranceVoxels"))
                               .toDouble(8.0)),
        1.0f,
        32.0f);
    options.voxelFallbackMaximumSliverRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVoxelFallbackMaximumSliverRatio"))
                               .toDouble(0.08)),
        0.0f,
        0.50f);
    const bool automatic_triangle_quality_optimization =
        options.resolution >= 384 && options.simplifyTargetFaces > 0;
    options.enableTriangleQualityOptimization = settings.value(
        QStringLiteral("tsdfTriangleQualityOptimization")).toBool(
            automatic_triangle_quality_optimization);
    options.triangleQualityOptimizationMaximumPasses = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfTriangleQualityOptimizationMaximumPasses"))
            .toInt(options.triangleQualityOptimizationMaximumPasses),
        12);
    options.triangleQualityMinimumAspectImprovementRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTriangleQualityMinimumAspectImprovementRatio"))
                               .toDouble(
                                   options.triangleQualityMinimumAspectImprovementRatio)),
        0.0f,
        0.25f);
    options.triangleQualityMaximumFeatureAngleDegrees = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTriangleQualityMaximumFeatureAngleDegrees"))
                               .toDouble(
                                   options.triangleQualityMaximumFeatureAngleDegrees)),
        5.0f,
        90.0f);
    options.triangleQualityMaximumNormalDeviationDegrees = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTriangleQualityMaximumNormalDeviationDegrees"))
                               .toDouble(
                                   options.triangleQualityMaximumNormalDeviationDegrees)),
        5.0f,
        90.0f);
    options.enableTriangleQualityTangentialRelaxation = settings.value(
        QStringLiteral("tsdfTriangleQualityTangentialRelaxation")).toBool(
            options.enableTriangleQualityTangentialRelaxation);
    options.triangleQualityTangentialRelaxationPasses = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfTriangleQualityTangentialRelaxationPasses"))
            .toInt(options.triangleQualityTangentialRelaxationPasses),
        8);
    options.triangleQualityTangentialRelaxationLambda = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTriangleQualityTangentialRelaxationLambda"))
                               .toDouble(
                                   options.triangleQualityTangentialRelaxationLambda)),
        0.0f,
        1.0f);
    options.triangleQualityTangentialMaximumDisplacementEdgeRatio =
        std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfTriangleQualityTangentialMaximumDisplacementEdgeRatio"))
                                   .toDouble(
                                       options
                                           .triangleQualityTangentialMaximumDisplacementEdgeRatio)),
            0.0f,
            0.50f);
    const bool automatic_isotropic_remeshing =
        automatic_triangle_quality_optimization &&
        options.simplifyTargetFaces <= 120000;
    options.enableTriangleQualityIsotropicRemeshing = settings.value(
        QStringLiteral("tsdfTriangleQualityIsotropicRemeshing")).toBool(
            automatic_isotropic_remeshing);
    options.triangleQualityIsotropicRemeshingPasses = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfTriangleQualityIsotropicRemeshingPasses"))
            .toInt(options.triangleQualityIsotropicRemeshingPasses),
        4);
    options.triangleQualityIsotropicShortEdgeRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTriangleQualityIsotropicShortEdgeRatio"))
                               .toDouble(
                                   options.triangleQualityIsotropicShortEdgeRatio)),
        0.05f,
        0.45f);
    options.triangleQualityIsotropicLongEdgeRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTriangleQualityIsotropicLongEdgeRatio"))
                               .toDouble(
                                   options.triangleQualityIsotropicLongEdgeRatio)),
        1.25f,
        4.0f);
    options.triangleQualityIsotropicMaximumFaceGrowthRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTriangleQualityIsotropicMaximumFaceGrowthRatio"))
                               .toDouble(
                                   options
                                       .triangleQualityIsotropicMaximumFaceGrowthRatio)),
        0.0f,
        0.20f);
    options.topologyQualityMaximumBoundaryEdgeRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTopologyQualityMaximumBoundaryEdgeRatio"))
                               .toDouble(
                                   options.topologyQualityMaximumBoundaryEdgeRatio)),
        0.0f,
        1.0f);
    options.topologyQualityMaximumHighAspectFaceRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTopologyQualityMaximumHighAspectFaceRatio"))
                               .toDouble(
                                   options.topologyQualityMaximumHighAspectFaceRatio)),
        0.0f,
        1.0f);
    options.topologyQualityMaximumExtremeAspectFaceRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTopologyQualityMaximumExtremeAspectFaceRatio"))
                               .toDouble(
                                   options.topologyQualityMaximumExtremeAspectFaceRatio)),
        0.0f,
        1.0f);
    options.topologyQualityMaximumClosedGenus = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTopologyQualityMaximumClosedGenus"))
                               .toDouble(
                                   options.topologyQualityMaximumClosedGenus)),
        0.0f,
        100000.0f);
    options.topologyQualityMaximumTopologicalComplexity = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfTopologyQualityMaximumTopologicalComplexity"))
            .toInt(options.topologyQualityMaximumTopologicalComplexity),
        200000);
    options.enableDepthCompletenessDiagnostics = settings.value(
        QStringLiteral("tsdfDepthCompletenessDiagnostics")).toBool(false);
    options.enforceDepthCompletenessGate = settings.value(
        QStringLiteral("tsdfEnforceDepthCompletenessGate")).toBool(false);
    options.depthCompletenessMaximumSamplesPerFrame = qBound(
        500,
        settings.value(QStringLiteral("tsdfDepthCompletenessMaximumSamplesPerFrame"))
            .toInt(options.depthCompletenessMaximumSamplesPerFrame),
        50000);
    options.depthCompletenessToleranceVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfDepthCompletenessToleranceVoxels"))
                               .toDouble(options.depthCompletenessToleranceVoxels)),
        1.0f,
        16.0f);
    options.minimumDepthCompletenessP10Recall = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMinimumDepthCompletenessP10Recall"))
                               .toDouble(options.minimumDepthCompletenessP10Recall)),
        0.0f,
        1.0f);
    options.minimumDepthCompletenessMedianRecall = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMinimumDepthCompletenessMedianRecall"))
                               .toDouble(options.minimumDepthCompletenessMedianRecall)),
        0.0f,
        1.0f);
    options.maximumSimplificationBoundaryEdgeGrowthRatio = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumSimplificationBoundaryEdgeGrowthRatio"))
                               .toDouble(options.maximumSimplificationBoundaryEdgeGrowthRatio)),
        0.0f,
        1.0f);
    options.simplificationMaximumPasses = qBound(
        1,
        settings.value(QStringLiteral("tsdfSimplificationMaximumPasses"))
            .toInt(options.simplificationMaximumPasses),
        96);
    options.simplificationFeatureAngleDegrees = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfSimplificationFeatureAngleDegrees"))
                               .toDouble(options.simplificationFeatureAngleDegrees)),
        5.0f,
        85.0f);
    options.simplificationMaximumNormalDeviationDegrees = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfSimplificationMaximumNormalDeviationDegrees"))
                               .toDouble(options.simplificationMaximumNormalDeviationDegrees)),
        5.0f,
        85.0f);
    options.simplificationMinimumSharpEdgeEndpointDegree = qBound(
        1,
        settings.value(QStringLiteral("tsdfSimplificationMinimumSharpEdgeEndpointDegree"))
            .toInt(options.simplificationMinimumSharpEdgeEndpointDegree),
        4);
    options.simplifySimpleOpenBoundaries = settings.value(
        QStringLiteral("tsdfSimplifySimpleOpenBoundaries"))
            .toBool(options.simplifySimpleOpenBoundaries);
    options.minimumInputFrames = 3;
    const int automatic_camera_support = 2;
    options.minimumDistinctCameraSupport = qBound(
        1,
        settings.value(QStringLiteral("tsdfMinimumDistinctCameraSupport"))
            .toInt(automatic_camera_support),
        16);
    options.minimumComponentFaces = qBound(
        0,
        settings.value(QStringLiteral("minFaces")).toInt(options.minimumComponentFaces),
        100000);
    options.minimumComponentFaceRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfMinimumComponentFaceRatio"))
                               .toDouble(options.minimumComponentFaceRatio)),
        0.0f,
        1.0f);
    options.workerCount = qBound(
        0,
        settings.value(QStringLiteral("threads")).toInt(options.workerCount),
        128);

    const QString filtering = settings.value(QStringLiteral("depthFiltering"))
                                  .toString(QStringLiteral("moderate"));
    if (filtering == QStringLiteral("disabled"))
    {
        options.minimumConfidence = 0.10f;
        options.minimumSingleObservationWeight = 0.55f;
    }
    else if (filtering == QStringLiteral("mild"))
    {
        options.minimumConfidence = 0.20f;
        options.minimumSingleObservationWeight = 0.60f;
    }
    else if (filtering == QStringLiteral("aggressive"))
    {
        options.minimumConfidence = 0.40f;
        options.minimumSingleObservationWeight = 0.80f;
    }
    else
    {
        options.minimumConfidence = 0.25f;
        options.minimumSingleObservationWeight = 0.70f;
    }

    options.truncationVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfTruncationVoxels"))
                               .toDouble(options.truncationVoxels)),
        1.0f,
        12.0f);
    options.surfaceSupportBandVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfSurfaceSupportBandVoxels"))
                               .toDouble(options.surfaceSupportBandVoxels)),
        0.0f,
        12.0f);
    options.enableUncertaintyAdaptiveTruncation = settings.value(
        QStringLiteral("tsdfUncertaintyAdaptiveTruncation")).toBool(false);
    options.uncertaintyAdaptiveScale = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfUncertaintyAdaptiveScale"))
                               .toDouble(options.uncertaintyAdaptiveScale)),
        0.0f,
        2.0f);
    options.uncertaintyAdaptiveActivationRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfUncertaintyAdaptiveActivationRatio"))
                               .toDouble(
                                   options.uncertaintyAdaptiveActivationRatio)),
        1.0f,
        4.0f);
    options.uncertaintyAdaptiveMaximumTruncationVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfUncertaintyAdaptiveMaximumTruncationVoxels"))
                               .toDouble(
                                   options.uncertaintyAdaptiveMaximumTruncationVoxels)),
        1.0f,
        16.0f);
    options.uncertaintyAdaptiveMaximumSamplesPerFrame = qBound(
        256,
        settings.value(QStringLiteral(
            "tsdfUncertaintyAdaptiveMaximumSamplesPerFrame"))
            .toInt(options.uncertaintyAdaptiveMaximumSamplesPerFrame),
        100000);
    options.uncertaintyAdaptiveMinimumSampleCount = qBound(
        64,
        settings.value(QStringLiteral(
            "tsdfUncertaintyAdaptiveMinimumSampleCount"))
            .toInt(options.uncertaintyAdaptiveMinimumSampleCount),
        100000);
    options.minimumVoxelWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfMinimumVoxelWeight"))
                               .toDouble(options.minimumVoxelWeight)),
        0.05f,
        20.0f);
    options.enablePixelEvidenceWeighting = settings.value(
        QStringLiteral("tsdfPixelEvidenceWeighting")).toBool(false);
    options.unconfirmedNativeObservationMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfUnconfirmedNativeObservationMultiplier"))
                               .toDouble(
                                   options.unconfirmedNativeObservationMultiplier)),
        0.05f,
        1.0f);
    options.weakNativeObservationMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfWeakNativeObservationMultiplier"))
                               .toDouble(options.weakNativeObservationMultiplier)),
        0.05f,
        1.0f);
    options.repairedObservationMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfRepairedObservationMultiplier"))
                               .toDouble(options.repairedObservationMultiplier)),
        0.05f,
        1.0f);
    options.adaptiveGeometryMinimumObservationMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveGeometryMinimumObservationMultiplier"))
                               .toDouble(
                                   options.adaptiveGeometryMinimumObservationMultiplier)),
        0.05f,
        1.0f);
    options.enableAdaptiveConflictRobustWeighting = settings.value(
        QStringLiteral("tsdfAdaptiveConflictRobustWeighting"))
        .toBool(options.enableAdaptiveConflictRobustWeighting);
    options.adaptiveConflictWeightKnee = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveConflictWeightKnee"))
                               .toDouble(options.adaptiveConflictWeightKnee)),
        0.0f,
        0.99f);
    options.adaptiveConflictWeightZero = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveConflictWeightZero"))
                               .toDouble(options.adaptiveConflictWeightZero)),
        options.adaptiveConflictWeightKnee + 1.0e-6f,
        1.0f);
    options.minimumAdaptiveConflictWeightMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMinimumAdaptiveConflictWeightMultiplier"))
                               .toDouble(
                                   options.minimumAdaptiveConflictWeightMultiplier)),
        0.0f,
        1.0f);
    options.adaptiveGeometryFullIntegrationMinimumSupportWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveGeometryFullIntegrationMinimumSupportWeight"))
                               .toDouble(
                                   options
                                       .adaptiveGeometryFullIntegrationMinimumSupportWeight)),
        0.0f,
        1.0f);
    options.adaptiveGeometryFullIntegrationMinimumEffectiveViewCount = std::max(
        1.0f,
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveGeometryFullIntegrationMinimumEffectiveViewCount"))
                               .toDouble(
                                   options
                                       .adaptiveGeometryFullIntegrationMinimumEffectiveViewCount)));
    options.adaptiveGeometryFullIntegrationMaximumConflictRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveGeometryFullIntegrationMaximumConflictRatio"))
                               .toDouble(
                                   options
                                       .adaptiveGeometryFullIntegrationMaximumConflictRatio)),
        0.0f,
        1.0f);
    options.enableInverseDepthSpreadWeighting = settings.value(
        QStringLiteral("tsdfInverseDepthSpreadWeighting")).toBool(false);
    options.inverseDepthSpreadWeightKnee = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfInverseDepthSpreadWeightKnee"))
                               .toDouble(options.inverseDepthSpreadWeightKnee)),
        0.0f,
        0.099f);
    options.inverseDepthSpreadWeightZero = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfInverseDepthSpreadWeightZero"))
                               .toDouble(options.inverseDepthSpreadWeightZero)),
        options.inverseDepthSpreadWeightKnee + 1.0e-6f,
        0.10f);
    options.minimumInverseDepthSpreadWeightMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMinimumInverseDepthSpreadWeightMultiplier"))
                               .toDouble(
                                   options.minimumInverseDepthSpreadWeightMultiplier)),
        0.0f,
        1.0f);
    options.enableInverseDepthSpreadSupportWeightDecoupling = settings.value(
        QStringLiteral("tsdfInverseDepthSpreadSupportWeightDecoupling"))
        .toBool(false);
    options.inverseDepthSpreadSupportWeightExponent = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfInverseDepthSpreadSupportWeightExponent"))
                               .toDouble(
                                   options.inverseDepthSpreadSupportWeightExponent)),
        0.05f,
        1.0f);
    options.enableEvidenceSupportWeightDecoupling = settings.value(
        QStringLiteral("tsdfEvidenceSupportWeightDecoupling")).toBool(false);
    options.evidenceSupportWeightExponent = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfEvidenceSupportWeightExponent"))
                               .toDouble(options.evidenceSupportWeightExponent)),
        0.0f,
        1.0f);
    options.enableWeakEvidenceSurfaceOnlyIntegration = settings.value(
        QStringLiteral("tsdfWeakEvidenceSurfaceOnlyIntegration")).toBool(false);
    options.weakEvidenceSurfaceBandVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfWeakEvidenceSurfaceBandVoxels"))
                               .toDouble(options.weakEvidenceSurfaceBandVoxels)),
        0.0f,
        16.0f);
    options.minimumSingleObservationWeight = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMinimumSingleObservationWeight"))
                               .toDouble(options.minimumSingleObservationWeight)),
        0.05f,
        1.0f);
    options.allowGeometryVerifiedSingleObservation = settings.value(
        QStringLiteral("tsdfAllowGeometryVerifiedSingleObservation")).toBool(
            options.resolution >= 384);
    options.minimumGeometryVerifiedObservationWeight = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMinimumGeometryVerifiedObservationWeight"))
                               .toDouble(options.minimumGeometryVerifiedObservationWeight)),
        0.05f,
        1.0f);
    const int automatic_minimum_geometry_support = options.resolution >= 384 ? 3 : 4;
    options.minimumGeometrySupportCount = qBound(
        2,
        settings.value(QStringLiteral("tsdfMinimumGeometrySupportCount"))
            .toInt(automatic_minimum_geometry_support),
        16);
    options.enableGeometrySingleViewNeighborhoodGuard = settings.value(
        QStringLiteral("tsdfGeometrySingleViewNeighborhoodGuard")).toBool(false);
    options.minimumGeometrySingleViewNeighborCount = qBound(
        1,
        settings.value(QStringLiteral("tsdfMinimumGeometrySingleViewNeighborCount"))
            .toInt(options.minimumGeometrySingleViewNeighborCount),
        26);
    options.geometrySingleViewGrowthPasses = qBound(
        1,
        settings.value(QStringLiteral("tsdfGeometrySingleViewGrowthPasses"))
            .toInt(options.geometrySingleViewGrowthPasses),
        6);
    options.maximumGeometrySingleViewNeighborTsdfDelta = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMaximumGeometrySingleViewNeighborTsdfDelta"))
                               .toDouble(options.maximumGeometrySingleViewNeighborTsdfDelta)),
        0.01f,
        2.0f);
    options.enableDiscontinuityAwareSampling = settings.value(
        QStringLiteral("tsdfDiscontinuityAwareSampling")).toBool(
            options.resolution >= 384);
    options.maximumInterpolationRelativeDepthSpread = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumInterpolationRelativeDepthSpread"))
                               .toDouble(options.maximumInterpolationRelativeDepthSpread)),
        0.001f,
        0.10f);
    options.maximumObservationInverseDepthSpread = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumObservationInverseDepthSpread"))
                               .toDouble(options.maximumObservationInverseDepthSpread)),
        0.0f,
        0.10f);
    options.allowInvalidNearestPixelRecovery = settings.value(
        QStringLiteral("tsdfAllowInvalidNearestPixelRecovery")).toBool(
            options.resolution < 384);
    options.maximumInvalidNearestPixelRecoveryInverseDepthSpread = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMaximumInvalidNearestPixelRecoveryInverseDepthSpread"))
                               .toDouble(options.maximumInvalidNearestPixelRecoveryInverseDepthSpread)),
        0.0f,
        0.10f);
    options.enableCrossViewConsensusDepth = settings.value(
        QStringLiteral("tsdfCrossViewConsensusDepth")).toBool(false);
    options.maximumCrossViewConsensusInverseDepthSpread = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMaximumCrossViewConsensusInverseDepthSpread"))
                               .toDouble(options.maximumCrossViewConsensusInverseDepthSpread)),
        0.001f,
        0.10f);
    options.crossViewConsensusContourBandOnly = settings.value(
        QStringLiteral("tsdfCrossViewConsensusContourBandOnly")).toBool(false);
    options.enableRobustFrameQualityWeighting = settings.value(
        QStringLiteral("tsdfRobustFrameQualityWeighting")).toBool(false);
    options.robustFrameQualityMinimumMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfRobustFrameQualityMinimumMultiplier"))
                               .toDouble(
                                   options.robustFrameQualityMinimumMultiplier)),
        0.05f,
        1.0f);
    options.robustFrameQualityMadFloor = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfRobustFrameQualityMadFloor"))
                               .toDouble(options.robustFrameQualityMadFloor)),
        0.001f,
        0.10f);
    options.robustFrameQualityPenaltyOnset = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfRobustFrameQualityPenaltyOnset"))
                               .toDouble(options.robustFrameQualityPenaltyOnset)),
        0.0f,
        4.0f);
    options.robustFrameQualityPenaltyStrength = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfRobustFrameQualityPenaltyStrength"))
                               .toDouble(options.robustFrameQualityPenaltyStrength)),
        0.0f,
        4.0f);
    options.enableRobustFrameQualityRejection = settings.value(
        QStringLiteral("tsdfRobustFrameQualityRejection")).toBool(false);
    options.robustFrameQualityRejectionSigma = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfRobustFrameQualityRejectionSigma"))
                               .toDouble(options.robustFrameQualityRejectionSigma)),
        0.5f,
        6.0f);
    options.robustFrameQualityMaximumRejectedRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfRobustFrameQualityMaximumRejectedRatio"))
                               .toDouble(
                                   options.robustFrameQualityMaximumRejectedRatio)),
        0.0f,
        0.50f);
    options.robustFrameQualityMinimumRetainedFrames = qBound(
        2,
        settings.value(QStringLiteral(
            "tsdfRobustFrameQualityMinimumRetainedFrames"))
            .toInt(options.robustFrameQualityMinimumRetainedFrames),
        64);
    options.enableOrbitalFrameCoverageProtection = settings.value(
        QStringLiteral("tsdfOrbitalFrameCoverageProtection")).toBool(false);
    options.maximumOrbitalAngularGapRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMaximumOrbitalAngularGapRatio"))
                               .toDouble(options.maximumOrbitalAngularGapRatio)),
        1.0f,
        6.0f);
    options.validationOnlyFrameWeightMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfValidationOnlyFrameWeightMultiplier"))
                               .toDouble(options.validationOnlyFrameWeightMultiplier)),
        0.05f,
        1.0f);
    options.coverageProtectedFrameMinimumMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfCoverageProtectedFrameMinimumMultiplier"))
                               .toDouble(options.coverageProtectedFrameMinimumMultiplier)),
        0.05f,
        1.0f);
    options.enableOrbitalGapBoundaryRecovery = settings.value(
        QStringLiteral("tsdfOrbitalGapBoundaryRecovery")).toBool(false);
    options.orbitalGapBoundaryMinimumQualityMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOrbitalGapBoundaryMinimumQualityMultiplier"))
                               .toDouble(
                                   options.orbitalGapBoundaryMinimumQualityMultiplier)),
        0.05f,
        1.0f);
    options.orbitalGapOppositeMinimumQualityMultiplier = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOrbitalGapOppositeMinimumQualityMultiplier"))
                               .toDouble(
                                   options.orbitalGapOppositeMinimumQualityMultiplier)),
        0.05f,
        1.0f);
    options.orbitalGapBoundaryMinimumObservationWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOrbitalGapBoundaryMinimumObservationWeight"))
                               .toDouble(
                                   options.orbitalGapBoundaryMinimumObservationWeight)),
        0.05f,
        1.0f);
    options.enableOrbitalGapAdaptiveTruncation = settings.value(
        QStringLiteral("tsdfOrbitalGapAdaptiveTruncation")).toBool(false);
    options.orbitalGapAdaptiveTruncationScale = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOrbitalGapAdaptiveTruncationScale"))
                               .toDouble(
                                   options.orbitalGapAdaptiveTruncationScale)),
        0.0f,
        2.0f);
    options.orbitalGapAdaptiveMaximumTruncationVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfOrbitalGapAdaptiveMaximumTruncationVoxels"))
                               .toDouble(
                                   options.orbitalGapAdaptiveMaximumTruncationVoxels)),
        1.0f,
        16.0f);
    const bool automatic_surface_patch_support = options.resolution >= 384 &&
        options.simplifyTargetFaces > 0 &&
        (options.simplifyTargetFaces <= 120000 ||
         (options.enableOpenMeshSimplification &&
          options.simplifyTargetFaces <= 240000));
    options.enableSurfacePatchSupport = settings.value(
        QStringLiteral("tsdfSurfacePatchSupport")).toBool(
            automatic_surface_patch_support);
    options.enableContourBandZeroCrossingSupport = settings.value(
        QStringLiteral("tsdfContourBandZeroCrossingSupport")).toBool(false);
    options.collectZeroCrossingDiagnostics = settings.value(
        QStringLiteral("tsdfCollectZeroCrossingDiagnostics")).toBool(false);
    options.collectAcquisitionGapReport = settings.value(
        QStringLiteral("tsdfAcquisitionGapReport")).toBool(false);
    options.enableMeasuredSupportConnectivity = settings.value(
        QStringLiteral("tsdfMeasuredSupportConnectivity")).toBool(false);
    options.measuredSupportMinimumObservationWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMeasuredSupportMinimumObservationWeight"))
                               .toDouble(
                                   options.measuredSupportMinimumObservationWeight)),
        0.01f,
        1.0f);
    options.measuredSupportMinimumSourceCount = qBound(
        2,
        settings.value(QStringLiteral("tsdfMeasuredSupportMinimumSourceCount"))
            .toInt(options.measuredSupportMinimumSourceCount),
        16);
    options.measuredSupportMinimumGeometrySupport = qBound(
        2,
        settings.value(QStringLiteral("tsdfMeasuredSupportMinimumGeometrySupport"))
            .toInt(options.measuredSupportMinimumGeometrySupport),
        16);
    options.measuredSupportMaximumInverseDepthSpread = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMeasuredSupportMaximumInverseDepthSpread"))
                               .toDouble(
                                   options.measuredSupportMaximumInverseDepthSpread)),
        0.001f,
        0.05f);
    options.measuredSupportMinimumSurfaceWeightRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMeasuredSupportMinimumSurfaceWeightRatio"))
                               .toDouble(
                                   options.measuredSupportMinimumSurfaceWeightRatio)),
        0.01f,
        1.0f);
    options.measuredSupportMaximumAbsoluteTsdf = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMeasuredSupportMaximumAbsoluteTsdf"))
                               .toDouble(options.measuredSupportMaximumAbsoluteTsdf)),
        0.05f,
        0.95f);
    options.measuredSupportMinimumSupportedCellCorners = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfMeasuredSupportMinimumSupportedCellCorners"))
            .toInt(options.measuredSupportMinimumSupportedCellCorners),
        7);
    options.measuredSupportMinimumComponentCells = qBound(
        1,
        settings.value(QStringLiteral("tsdfMeasuredSupportMinimumComponentCells"))
            .toInt(options.measuredSupportMinimumComponentCells),
        4096);
    options.measuredSupportMinimumAnchorCells = qBound(
        1,
        settings.value(QStringLiteral("tsdfMeasuredSupportMinimumAnchorCells"))
            .toInt(options.measuredSupportMinimumAnchorCells),
        4096);
    options.measuredSupportMaximumSingleVoteAbsoluteTsdf = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMeasuredSupportMaximumSingleVoteAbsoluteTsdf"))
                               .toDouble(
                                   options.measuredSupportMaximumSingleVoteAbsoluteTsdf)),
        0.0f,
        1.0f);
    options.enableConsistentIsoSurfaceExtraction = settings.value(
        QStringLiteral("tsdfConsistentIsoSurfaceExtraction")).toBool(false);
    options.enableMc33IsoSurfaceExtraction = settings.value(
        QStringLiteral("tsdfMc33IsoSurfaceExtraction")).toBool(
            options.enableOpenMeshSimplification &&
            Mc33IsoSurfaceExtractor::isAvailable());
    options.mc33RequireSupportedSignChange = settings.value(
        QStringLiteral("tsdfMc33RequireSupportedSignChange")).toBool(true);
    options.enableGeometryZeroCrossingRecovery = settings.value(
        QStringLiteral("tsdfGeometryZeroCrossingRecovery")).toBool(false);
    options.geometryZeroCrossingMinimumSupportedCorners = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfGeometryZeroCrossingMinimumSupportedCorners"))
            .toInt(options.geometryZeroCrossingMinimumSupportedCorners),
        7);
    options.geometryZeroCrossingMinimumCellVotes = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfGeometryZeroCrossingMinimumCellVotes"))
            .toInt(options.geometryZeroCrossingMinimumCellVotes),
        8);
    options.enableCrossViewAnchoredSurfaceRecovery = settings.value(
        QStringLiteral("tsdfCrossViewAnchoredSurfaceRecovery")).toBool(false);
    options.crossViewAnchoredMinimumObservationWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfCrossViewAnchoredMinimumObservationWeight"))
                               .toDouble(options
                                   .crossViewAnchoredMinimumObservationWeight)),
        0.05f,
        1.0f);
    options.crossViewAnchoredMinimumSupportedCorners = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfCrossViewAnchoredMinimumSupportedCorners"))
            .toInt(options.crossViewAnchoredMinimumSupportedCorners),
        7);
    options.crossViewAnchoredMinimumCellVotes = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfCrossViewAnchoredMinimumCellVotes"))
            .toInt(options.crossViewAnchoredMinimumCellVotes),
        8);
    options.crossViewAnchoredGrowthPasses = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfCrossViewAnchoredGrowthPasses"))
            .toInt(options.crossViewAnchoredGrowthPasses),
        4);
    options.enableGeometryZeroCrossingCellSheets = settings.value(
        QStringLiteral("tsdfGeometryZeroCrossingCellSheets")).toBool(false);
    options.minimumGeometryZeroCrossingSheetCells = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfMinimumGeometryZeroCrossingSheetCells"))
            .toInt(options.minimumGeometryZeroCrossingSheetCells),
        4096);
    options.minimumGeometryZeroCrossingSheetAnchorCells = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfMinimumGeometryZeroCrossingSheetAnchorCells"))
            .toInt(options.minimumGeometryZeroCrossingSheetAnchorCells),
        4096);
    options.maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf =
        std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral(
                    "tsdfMaximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf"))
                    .toDouble(options
                        .maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf)),
            0.0f,
            1.0f);
    options.enableGlobalImplicitRegularization = settings.value(
        QStringLiteral("tsdfGlobalImplicitRegularization")).toBool(false);
    options.implicitRegularizationLevels = qBound(
        1,
        settings.value(QStringLiteral("tsdfImplicitRegularizationLevels"))
            .toInt(options.implicitRegularizationLevels),
        3);
    options.implicitRegularizationPassesPerLevel = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfImplicitRegularizationPassesPerLevel"))
            .toInt(options.implicitRegularizationPassesPerLevel),
        4);
    options.implicitRegularizationSmoothness = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfImplicitRegularizationSmoothness"))
                               .toDouble(
                                   options.implicitRegularizationSmoothness)),
        0.0f,
        2.0f);
    options.implicitRegularizationDataFidelity = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfImplicitRegularizationDataFidelity"))
                               .toDouble(
                                   options.implicitRegularizationDataFidelity)),
        0.01f,
        8.0f);
    options.implicitRegularizationMaximumUpdate = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfImplicitRegularizationMaximumUpdate"))
                               .toDouble(
                                   options.implicitRegularizationMaximumUpdate)),
        0.0f,
        0.5f);
    options.implicitRegularizationEdgeThreshold = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfImplicitRegularizationEdgeThreshold"))
                               .toDouble(
                                   options.implicitRegularizationEdgeThreshold)),
        0.02f,
        1.0f);
    options.implicitRegularizationRecoverAxialGaps = settings.value(
        QStringLiteral("tsdfImplicitRegularizationRecoverAxialGaps")).toBool(
            options.implicitRegularizationRecoverAxialGaps);
    options.implicitRegularizationMinimumBridgeAxes = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfImplicitRegularizationMinimumBridgeAxes"))
            .toInt(options.implicitRegularizationMinimumBridgeAxes),
        3);
    options.implicitRegularizationMaximumBridgePredictionDelta = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfImplicitRegularizationMaximumBridgePredictionDelta"))
                               .toDouble(options
                                             .implicitRegularizationMaximumBridgePredictionDelta)),
        0.01f,
        0.5f);
    options.enableAdaptiveTgvRegularization = settings.value(
        QStringLiteral("tsdfAdaptiveTgvRegularization")).toBool(false);
    options.adaptiveTgvMaximumMergeLevel = qBound(
        0,
        settings.value(QStringLiteral("tsdfAdaptiveTgvMaximumMergeLevel"))
            .toInt(options.adaptiveTgvMaximumMergeLevel),
        10);
    options.adaptiveTgvMinimumMergeAbsoluteField = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvMinimumMergeAbsoluteField"))
                               .toDouble(
                                   options.adaptiveTgvMinimumMergeAbsoluteField)),
        0.1f,
        1.0f);
    options.adaptiveTgvMaximumMergeFieldRange = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvMaximumMergeFieldRange"))
                               .toDouble(
                                   options.adaptiveTgvMaximumMergeFieldRange)),
        0.01f,
        1.0f);
    options.adaptiveTgvMaximumActiveAbsoluteField = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvMaximumActiveAbsoluteField"))
                               .toDouble(
                                   options.adaptiveTgvMaximumActiveAbsoluteField)),
        0.10f,
        1.0f);
    options.adaptiveTgvMaximumIterations = qBound(
        10,
        settings.value(QStringLiteral("tsdfAdaptiveTgvMaximumIterations"))
            .toInt(options.adaptiveTgvMaximumIterations),
        400);
    options.adaptiveTgvMinimumIterations = qBound(
        5,
        settings.value(QStringLiteral("tsdfAdaptiveTgvMinimumIterations"))
            .toInt(options.adaptiveTgvMinimumIterations),
        options.adaptiveTgvMaximumIterations);
    options.adaptiveTgvFirstOrderWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvFirstOrderWeight"))
                               .toDouble(options.adaptiveTgvFirstOrderWeight)),
        0.001f,
        2.0f);
    options.adaptiveTgvSecondOrderWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvSecondOrderWeight"))
                               .toDouble(options.adaptiveTgvSecondOrderWeight)),
        0.001f,
        2.0f);
    options.adaptiveTgvDataFidelity = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvDataFidelity"))
                               .toDouble(options.adaptiveTgvDataFidelity)),
        0.001f,
        2.0f);
    options.adaptiveTgvPrimalStep = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvPrimalStep"))
                               .toDouble(options.adaptiveTgvPrimalStep)),
        0.001f,
        0.24f);
    options.adaptiveTgvDualStep = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvDualStep"))
                               .toDouble(options.adaptiveTgvDualStep)),
        0.001f,
        0.24f);
    options.adaptiveTgvConvergenceTolerance = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvConvergenceTolerance"))
                               .toDouble(
                                   options.adaptiveTgvConvergenceTolerance)),
        0.0f,
        0.01f);
    options.adaptiveTgvUseGlobalVisibilityField = settings.value(
        QStringLiteral("tsdfAdaptiveTgvUseGlobalVisibilityField")).toBool(
            options.adaptiveTgvUseGlobalVisibilityField);
    options.adaptiveTgvRecoverUnsupportedSamples = settings.value(
        QStringLiteral("tsdfAdaptiveTgvRecoverUnsupportedSamples")).toBool(
            options.adaptiveTgvRecoverUnsupportedSamples);
    options.adaptiveTgvRecoveryPasses = qBound(
        1,
        settings.value(QStringLiteral("tsdfAdaptiveTgvRecoveryPasses"))
            .toInt(options.adaptiveTgvRecoveryPasses),
        6);
    options.adaptiveTgvMinimumRecoveryNeighbors = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfAdaptiveTgvMinimumRecoveryNeighbors"))
            .toInt(options.adaptiveTgvMinimumRecoveryNeighbors),
        6);
    options.adaptiveTgvMaximumRecoveryConflictRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfAdaptiveTgvMaximumRecoveryConflictRatio"))
                               .toDouble(options
                                             .adaptiveTgvMaximumRecoveryConflictRatio)),
        0.0f,
        0.5f);
    options.enableVisualHullSignedDistanceCompletion = settings.value(
        QStringLiteral("tsdfVisualHullSignedDistanceCompletion")).toBool(
            options.enableVisualHullSignedDistanceCompletion);
    options.visualHullCompletionMinimumVisibleViews = qBound(
        2,
        settings.value(QStringLiteral(
            "tsdfVisualHullCompletionMinimumVisibleViews"))
            .toInt(options.visualHullCompletionMinimumVisibleViews),
        64);
    options.visualHullCompletionAllowedSilhouetteViolations = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisualHullCompletionAllowedSilhouetteViolations"))
            .toInt(options.visualHullCompletionAllowedSilhouetteViolations),
        16);
    options.visualHullCompletionBandVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisualHullCompletionBandVoxels"))
                               .toDouble(
                                   options.visualHullCompletionBandVoxels)),
        1.0f,
        24.0f);
    options.visualHullCompletionPreserveObservedTsdf = settings.value(
        QStringLiteral("tsdfVisualHullCompletionPreserveObservedTsdf"))
        .toBool(options.visualHullCompletionPreserveObservedTsdf);
    options.visualHullCompletionMaximumObservedAbsoluteTsdf = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisualHullCompletionMaximumObservedAbsoluteTsdf"))
                               .toDouble(options
                                             .visualHullCompletionMaximumObservedAbsoluteTsdf)),
        0.05f,
        1.0f);
    options.visualHullCompletionMinimumGeometrySupport = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisualHullCompletionMinimumGeometrySupport"))
            .toInt(options.visualHullCompletionMinimumGeometrySupport),
        16);
    options.visualHullCompletionRelaxationIterations = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisualHullCompletionRelaxationIterations"))
            .toInt(options.visualHullCompletionRelaxationIterations),
        24);
    options.visualHullCompletionRelaxationLambda = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisualHullCompletionRelaxationLambda"))
                               .toDouble(options
                                             .visualHullCompletionRelaxationLambda)),
        0.0f,
        0.49f);
    options.visualHullCompletionMaximumUpdate = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisualHullCompletionMaximumUpdate"))
                               .toDouble(options
                                             .visualHullCompletionMaximumUpdate)),
        0.01f,
        1.0f);
    options.enableVisibilityOccupancyCompletion = settings.value(
        QStringLiteral("tsdfVisibilityOccupancyCompletion")).toBool(false);
    options.visibilityOccupancyResolution = qBound(
        24,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyResolution"))
            .toInt(options.visibilityOccupancyResolution),
        128);
    options.visibilityOccupancyAlignCarrierGrid = settings.value(
        QStringLiteral("tsdfVisibilityOccupancyAlignCarrierGrid"))
        .toBool(options.visibilityOccupancyAlignCarrierGrid);
    options.visibilityOccupancyNativeCarrierExtraction = settings.value(
        QStringLiteral("tsdfVisibilityOccupancyNativeCarrierExtraction"))
        .toBool(options.visibilityOccupancyNativeCarrierExtraction);
    options.visibilityOccupancyCellBoundaryExtraction = settings.value(
        QStringLiteral("tsdfVisibilityOccupancyCellBoundaryExtraction"))
        .toBool(options.visibilityOccupancyCellBoundaryExtraction);
    options.visibilityOccupancyMinimumVisibleViews = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyMinimumVisibleViews"))
            .toInt(options.visibilityOccupancyMinimumVisibleViews),
        16);
    options.visibilityOccupancyMinimumSilhouetteViews = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyMinimumSilhouetteViews"))
            .toInt(options.visibilityOccupancyMinimumSilhouetteViews),
        16);
    options.visibilityOccupancyMinimumDepthFullViewsForSilhouettePrior = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyMinimumDepthFullViewsForSilhouettePrior"))
            .toInt(options
                       .visibilityOccupancyMinimumDepthFullViewsForSilhouettePrior),
        16);
    options.visibilityOccupancyAdaptiveDepthSupportMinimumFullFraction =
        std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyAdaptiveDepthSupportMinimumFullFraction"))
                                   .toDouble(options
                                       .visibilityOccupancyAdaptiveDepthSupportMinimumFullFraction)),
            0.0f,
            0.25f);
    options.visibilityOccupancyAllowedSilhouetteViolations = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyAllowedSilhouetteViolations"))
            .toInt(options.visibilityOccupancyAllowedSilhouetteViolations),
        8);
    options.visibilityOccupancyFrontTolerancePixelFootprints = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyFrontTolerancePixelFootprints"))
                               .toDouble(options
                                             .visibilityOccupancyFrontTolerancePixelFootprints)),
        0.5f,
        12.0f);
    options.visibilityOccupancyBehindSurfaceBandPixelFootprints = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyBehindSurfaceBandPixelFootprints"))
                               .toDouble(options
                                             .visibilityOccupancyBehindSurfaceBandPixelFootprints)),
        1.0f,
        16.0f);
    options.visibilityOccupancyDepthEmptyCapacity = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthEmptyCapacity"))
            .toInt(options.visibilityOccupancyDepthEmptyCapacity),
        1000);
    options.visibilityOccupancyDepthFullCapacity = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthFullCapacity"))
            .toInt(options.visibilityOccupancyDepthFullCapacity),
        1000);
    options.visibilityOccupancySilhouetteEmptyCapacity = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancySilhouetteEmptyCapacity"))
            .toInt(options.visibilityOccupancySilhouetteEmptyCapacity),
        1000);
    options.visibilityOccupancySilhouetteFullPriorCapacity = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancySilhouetteFullPriorCapacity"))
            .toInt(options.visibilityOccupancySilhouetteFullPriorCapacity),
        1000);
    options.visibilityOccupancyPairwiseCapacity = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyPairwiseCapacity"))
            .toInt(options.visibilityOccupancyPairwiseCapacity),
        1000);
    options.visibilityOccupancyClosingIterations = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyClosingIterations"))
            .toInt(options.visibilityOccupancyClosingIterations),
        8);
    options.visibilityOccupancyMaximumHandleRepairPasses = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumHandleRepairPasses"))
            .toInt(options.visibilityOccupancyMaximumHandleRepairPasses),
        16);
    options.visibilityOccupancyMaximumHandleRepairAcceptedCandidateCount =
        qBound(
            0,
            settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyMaximumHandleRepairAcceptedCandidateCount"))
                .toInt(options
                    .visibilityOccupancyMaximumHandleRepairAcceptedCandidateCount),
            512);
    options.visibilityOccupancyMaximumHandleRepairCandidateSampleCount =
        static_cast<std::size_t>(qBound(
            1,
            settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyMaximumHandleRepairCandidateSampleCount"))
                .toInt(static_cast<int>(std::min<std::size_t>(
                    options
                        .visibilityOccupancyMaximumHandleRepairCandidateSampleCount,
                    static_cast<std::size_t>(65536)))),
            65536));
    options.visibilityOccupancyMaximumHandleRepairSubsetSampleCount =
        static_cast<std::size_t>(qBound(
            1,
            settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyMaximumHandleRepairSubsetSampleCount"))
                .toInt(static_cast<int>(std::min<std::size_t>(
                    options
                        .visibilityOccupancyMaximumHandleRepairSubsetSampleCount,
                    static_cast<std::size_t>(1024)))),
            1024));
    options.visibilityOccupancyMaximumHandleRepairSubsetSeedCount = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumHandleRepairSubsetSeedCount"))
            .toInt(options
                .visibilityOccupancyMaximumHandleRepairSubsetSeedCount),
        4096);
    options.visibilityOccupancyClosingMinimumDepthEmptyViewsToProtect = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyClosingMinimumDepthEmptyViewsToProtect"))
            .toInt(options
                .visibilityOccupancyClosingMinimumDepthEmptyViewsToProtect),
        16);
    options.visibilityOccupancyClosingMinimumSilhouetteOutsideViewsToProtect =
        qBound(
            1,
            settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyClosingMinimumSilhouetteOutsideViewsToProtect"))
                .toInt(options
                    .visibilityOccupancyClosingMinimumSilhouetteOutsideViewsToProtect),
            16);
    options.visibilityOccupancyTopologyLockedResidualBlend = settings.value(
        QStringLiteral(
            "tsdfVisibilityOccupancyTopologyLockedResidualBlend"))
        .toBool(options.visibilityOccupancyTopologyLockedResidualBlend);
    options.visibilityOccupancyObservedBand = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyObservedBand"))
                               .toDouble(
                                   options.visibilityOccupancyObservedBand)),
        0.01f,
        1.0f);
    options.visibilityOccupancyCarrierBand = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyCarrierBand"))
                               .toDouble(
                                   options.visibilityOccupancyCarrierBand)),
        0.01f,
        1.0f);
    options.visibilityOccupancyMaximumResidual = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumResidual"))
                               .toDouble(
                                   options.visibilityOccupancyMaximumResidual)),
        0.0f,
        1.0f);
    options.visibilityOccupancyDetailBlend = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDetailBlend"))
                               .toDouble(
                                   options.visibilityOccupancyDetailBlend)),
        0.0f,
        1.0f);
    options.visibilityOccupancyPreserveAllObservedSamples = settings.value(
        QStringLiteral(
            "tsdfVisibilityOccupancyPreserveAllObservedSamples"))
        .toBool(options.visibilityOccupancyPreserveAllObservedSamples);
    options.visibilityOccupancyPreserveObservedNearSurface = settings.value(
        QStringLiteral(
            "tsdfVisibilityOccupancyPreserveObservedNearSurface"))
        .toBool(options.visibilityOccupancyPreserveObservedNearSurface);
    options.visibilityOccupancyRequireSignAgreement = settings.value(
        QStringLiteral(
            "tsdfVisibilityOccupancyRequireSignAgreement"))
        .toBool(options.visibilityOccupancyRequireSignAgreement);
    options.visibilityOccupancyMaximumPreservedAbsoluteTsdf = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumPreservedAbsoluteTsdf"))
                               .toDouble(options
                                             .visibilityOccupancyMaximumPreservedAbsoluteTsdf)),
        0.0f,
        1.0f);
    options.visibilityOccupancySignedDistanceNormalizationSamples =
        std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancySignedDistanceNormalizationSamples"))
                                   .toDouble(options
                                                 .visibilityOccupancySignedDistanceNormalizationSamples)),
            0.5f,
            16.0f);
    const float automatic_minimum_surface_patch_observation_weight = 0.60f;
    options.minimumSurfacePatchObservationWeight = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMinimumSurfacePatchObservationWeight"))
                               .toDouble(automatic_minimum_surface_patch_observation_weight)),
        0.05f,
        1.0f);
    options.minimumSurfacePatchSourceCount = qBound(
        2,
        settings.value(QStringLiteral("tsdfMinimumSurfacePatchSourceCount"))
            .toInt(options.minimumSurfacePatchSourceCount),
        8);
    options.minimumSurfacePatchCoreNeighborCount = qBound(
        1,
        settings.value(QStringLiteral("tsdfMinimumSurfacePatchCoreNeighborCount"))
            .toInt(options.minimumSurfacePatchCoreNeighborCount),
        26);
    options.surfacePatchGrowthPasses = qBound(
        1,
        settings.value(QStringLiteral("tsdfSurfacePatchGrowthPasses"))
            .toInt(1),
        6);
    options.maximumSurfacePatchInverseDepthSpread = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumSurfacePatchInverseDepthSpread"))
                               .toDouble(options.maximumSurfacePatchInverseDepthSpread)),
        0.001f,
        0.05f);
    options.maximumSurfacePatchNormalAngleDegrees = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumSurfacePatchNormalAngleDegrees"))
                               .toDouble(options.maximumSurfacePatchNormalAngleDegrees)),
        5.0f,
        45.0f);
    options.maximumSurfacePatchAbsoluteTsdf = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumSurfacePatchAbsoluteTsdf"))
                               .toDouble(options.maximumSurfacePatchAbsoluteTsdf)),
        0.05f,
        0.95f);
    options.maximumContourBandAbsoluteTsdf = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumContourBandAbsoluteTsdf"))
                               .toDouble(options.maximumSurfacePatchAbsoluteTsdf)),
        options.maximumSurfacePatchAbsoluteTsdf,
        0.95f);
    options.minimumSurfacePatchWeightRatio = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMinimumSurfacePatchWeightRatio"))
                               .toDouble(options.minimumSurfacePatchWeightRatio)),
        0.01f,
        1.0f);
    options.enableSupportMaskFreeSpaceCarving = settings.value(
        QStringLiteral("tsdfSupportMaskFreeSpaceCarving")).toBool(false);
    options.enableSurfaceEvidenceFreeSpaceVeto = settings.value(
        QStringLiteral("tsdfSurfaceEvidenceFreeSpaceVeto")).toBool(true);
    options.maximumFreeSpaceVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfMaximumFreeSpaceVoxels"))
                               .toDouble(options.maximumFreeSpaceVoxels)),
        0.0f,
        64.0f);
    const int automatic_boundary_erosion = options.resolution >= 384 ? 2 : 1;
    options.depthValidBoundaryErosionPixels = qBound(
        0,
        settings.value(QStringLiteral("tsdfDepthValidBoundaryErosionPixels"))
            .toInt(automatic_boundary_erosion),
        4);
    const bool automatic_boundary_recovery = options.resolution >= 384 &&
        options.simplifyTargetFaces > 0 &&
        (options.simplifyTargetFaces <= 120000 ||
         (options.enableOpenMeshSimplification &&
          options.simplifyTargetFaces <= 240000));
    options.enableGeometryVerifiedBoundaryRecovery = settings.value(
        QStringLiteral("tsdfGeometryVerifiedBoundaryRecovery")).toBool(
            automatic_boundary_recovery);
    options.minimumBoundaryRecoveryGeometrySupport = qBound(
        2,
        settings.value(QStringLiteral("tsdfMinimumBoundaryRecoveryGeometrySupport"))
            .toInt(options.minimumBoundaryRecoveryGeometrySupport),
        16);
    options.maximumBoundaryRecoveryInverseDepthSpread = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfMaximumBoundaryRecoveryInverseDepthSpread"))
                               .toDouble(
                                   options.maximumBoundaryRecoveryInverseDepthSpread)),
        0.001f,
        0.05f);
    options.supportMaskFreeSpaceWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfSupportMaskFreeSpaceWeight"))
                               .toDouble(options.supportMaskFreeSpaceWeight)),
        0.0f,
        1.0f);
    options.minimumSupportMaskFreeSpaceViews = qBound(
        1,
        settings.value(QStringLiteral("tsdfMinimumSupportMaskFreeSpaceViews"))
            .toInt(options.minimumSupportMaskFreeSpaceViews),
        16);
    options.enableNarrowBandActivation = settings.value(
        QStringLiteral("tsdfNarrowBandActivation")).toBool(false);
    options.narrowBandActivationBlockSizeSamples = qBound(
        2,
        settings.value(QStringLiteral(
            "tsdfNarrowBandActivationBlockSizeSamples"))
            .toInt(options.narrowBandActivationBlockSizeSamples),
        32);
    options.narrowBandActivationDepthStride = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfNarrowBandActivationDepthStride"))
            .toInt(options.narrowBandActivationDepthStride),
        16);
    options.narrowBandActivationRayStepVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfNarrowBandActivationRayStepVoxels"))
                               .toDouble(
                                   options.narrowBandActivationRayStepVoxels)),
        0.25f,
        4.0f);
    options.narrowBandActivationHaloBlocks = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfNarrowBandActivationHaloBlocks"))
            .toInt(options.narrowBandActivationHaloBlocks),
        3);

    const QString interpolation = settings.value(
        QStringLiteral("interpolation")).toString(QStringLiteral("enabled"));
    options.fillSmallBoundaryHoles = interpolation != QStringLiteral("disabled")
        && settings.value(QStringLiteral("holeFill")).toBool(true);
    options.splitPinchedBoundaryVertices = settings.value(
        QStringLiteral("tsdfSplitPinchedBoundaryVertices"))
            .toBool(options.splitPinchedBoundaryVertices);
    const double maximum_hole_area = std::max(
        1.0, settings.value(QStringLiteral("maxHoleSize")).toDouble(100.0));
    const int conservative_edges = std::clamp(
        static_cast<int>(std::lround(std::sqrt(maximum_hole_area) * 1.6)),
        8,
        32);
    options.maximumHoleBoundaryEdges = interpolation == QStringLiteral("extrapolated")
        ? std::clamp(std::max(64, conservative_edges * 4), 64, 128)
        : std::clamp(std::max(48, conservative_edges * 3), 32, 64);
    options.maximumHoleDiameterVoxels = interpolation == QStringLiteral("extrapolated")
        ? 16.0f
        : 10.0f;
    options.maximumHoleBoundaryEdges = qBound(
        3,
        settings.value(QStringLiteral("tsdfMaximumHoleBoundaryEdges"))
            .toInt(options.maximumHoleBoundaryEdges),
        256);
    options.maximumHoleDiameterVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfMaximumHoleDiameterVoxels"))
                               .toDouble(options.maximumHoleDiameterVoxels)),
        0.0f,
        64.0f);
    const bool automatic_final_hole_fill =
        options.fillSmallBoundaryHoles &&
        options.resolution >= 384 &&
        options.simplifyTargetFaces > 0 &&
        options.simplifyTargetFaces <= 120000;
    options.enableSilhouetteAwareFinalHoleFill = settings.value(
        QStringLiteral("tsdfSilhouetteAwareFinalHoleFill")).toBool(
            automatic_final_hole_fill);
    if (automatic_final_hole_fill)
    {
        // This stage is a residual micro-hole repair.  Large boundary loops
        // represent missing implicit geometry and must remain visible instead
        // of being hidden behind a triangle fan.
        options.finalHoleFillMaximumBoundaryEdges = 24;
        options.finalHoleFillMaximumDiameterVoxels = 4.0f;
        options.finalHoleFillMaximumFaceGrowthRatio = 0.03f;
    }
    options.finalHoleFillMaximumBoundaryEdges = qBound(
        3,
        settings.value(QStringLiteral("tsdfFinalHoleFillMaximumBoundaryEdges"))
            .toInt(options.finalHoleFillMaximumBoundaryEdges),
        512);
    options.finalHoleFillMaximumDiameterVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfFinalHoleFillMaximumDiameterVoxels"))
                               .toDouble(options.finalHoleFillMaximumDiameterVoxels)),
        0.0f,
        64.0f);
    options.finalHoleFillMaximumFaceGrowthRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfFinalHoleFillMaximumFaceGrowthRatio"))
                               .toDouble(options.finalHoleFillMaximumFaceGrowthRatio)),
        0.0f,
        0.50f);
    options.finalHoleFillMaximumSliverRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfFinalHoleFillMaximumSliverRatio"))
                               .toDouble(options.finalHoleFillMaximumSliverRatio)),
        0.0f,
        0.50f);
    options.enableVisibilityConstrainedFinalHoleFill = settings.value(
        QStringLiteral("tsdfVisibilityConstrainedFinalHoleFill")).toBool(
            automatic_final_hole_fill &&
            options.enableSilhouetteAwareFinalHoleFill);
    options.visibilityHoleFillMinimumSupportingViews = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisibilityHoleFillMinimumSupportingViews"))
            .toInt(options.visibilityHoleFillMinimumSupportingViews),
        8);
    options.visibilityHoleFillMaximumConflictViews = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityHoleFillMaximumConflictViews"))
            .toInt(
                automatic_final_hole_fill &&
                        settings.value(QStringLiteral("surface_type"))
                                .toString(QStringLiteral("arbitrary_3d")) ==
                            QStringLiteral("arbitrary_3d")
                    ? 2
                    : options.visibilityHoleFillMaximumConflictViews),
        16);
    options.visibilityHoleFillDepthToleranceVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityHoleFillDepthToleranceVoxels"))
                               .toDouble(
                                   options.visibilityHoleFillDepthToleranceVoxels)),
        1.0f,
        32.0f);
    options.visibilityHoleFillStrongSilhouetteRatio = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityHoleFillStrongSilhouetteRatio"))
                               .toDouble(
                                   options.visibilityHoleFillStrongSilhouetteRatio)),
        0.0f,
        1.0f);
    options.enableTinyBoundaryLoopCollapse = settings.value(
        QStringLiteral("tsdfTinyBoundaryLoopCollapse")).toBool(
            automatic_final_hole_fill &&
            options.enableVisibilityConstrainedFinalHoleFill);
    options.tinyBoundaryLoopCollapseMaximumEdges = qBound(
        3,
        settings.value(QStringLiteral(
            "tsdfTinyBoundaryLoopCollapseMaximumEdges"))
            .toInt(options.tinyBoundaryLoopCollapseMaximumEdges),
        16);
    options.tinyBoundaryLoopCollapseMaximumDiameterVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTinyBoundaryLoopCollapseMaximumDiameterVoxels"))
                               .toDouble(
                                   options.tinyBoundaryLoopCollapseMaximumDiameterVoxels)),
        0.0f,
        8.0f);
    options.tinyBoundaryLoopCollapseMaximumEdgeVoxels = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfTinyBoundaryLoopCollapseMaximumEdgeVoxels"))
                               .toDouble(
                                   options.tinyBoundaryLoopCollapseMaximumEdgeVoxels)),
        0.0f,
        1.0f);
    options.tinyBoundaryLoopCollapseMaximumPasses = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfTinyBoundaryLoopCollapseMaximumPasses"))
            .toInt(options.tinyBoundaryLoopCollapseMaximumPasses),
        8);
    const int automatic_boundary_smoothing_iterations = options.resolution >= 384 ? 2 : 1;
    options.boundarySmoothingIterations = qBound(
        0,
        settings.value(QStringLiteral("tsdfBoundarySmoothingIterations"))
            .toInt(automatic_boundary_smoothing_iterations),
        4);
    options.boundarySmoothingLambda = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfBoundarySmoothingLambda"))
                               .toDouble(options.boundarySmoothingLambda)),
        0.0f,
        0.5f);
    options.maximumBoundarySmoothingDisplacementVoxels = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumBoundarySmoothingDisplacementVoxels"))
                               .toDouble(options.maximumBoundarySmoothingDisplacementVoxels)),
        0.0f,
        1.0f);
    const bool automatic_surface_denoising =
        options.resolution >= 256 &&
        options.simplifyTargetFaces > 0 &&
        options.simplifyTargetFaces <= 240000 &&
        settings.value(QStringLiteral("surface_type"))
                .toString(QStringLiteral("arbitrary_3d")) ==
            QStringLiteral("arbitrary_3d");
    const bool high_face_budget_surface_denoising =
        automatic_surface_denoising &&
        options.simplifyTargetFaces > 120000;
    const int automatic_surface_denoising_iterations =
        automatic_surface_denoising
            ? (high_face_budget_surface_denoising ? 8 : 1)
            : options.surfaceDenoisingIterations;
    if (automatic_surface_denoising)
    {
        options.surfaceDenoisingLambda =
            high_face_budget_surface_denoising ? 0.50f : 0.30f;
        options.surfaceDenoisingMu =
            high_face_budget_surface_denoising ? -0.53f : 0.0f;
        options.maximumSurfaceDenoisingDisplacementVoxels =
            high_face_budget_surface_denoising ? 0.75f : 0.12f;
        options.maximumSurfaceDenoisingNormalAngleDegrees =
            high_face_budget_surface_denoising ? 120.0f : 25.0f;
        options.surfaceDenoisingBoundaryProtectionRings =
            high_face_budget_surface_denoising ? 0 : 1;
        options.enableProtectedTaubinSurfaceDenoising =
            high_face_budget_surface_denoising;
    }
    options.surfaceDenoisingIterations = qBound(
        0,
        settings.value(QStringLiteral("tsdfSurfaceDenoisingIterations"))
            .toInt(automatic_surface_denoising_iterations),
        8);
    options.surfaceDenoisingLambda = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfSurfaceDenoisingLambda"))
                               .toDouble(options.surfaceDenoisingLambda)),
        0.0f,
        0.75f);
    options.surfaceDenoisingMu = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfSurfaceDenoisingMu"))
                               .toDouble(options.surfaceDenoisingMu)),
        -1.0f,
        0.0f);
    options.maximumSurfaceDenoisingDisplacementVoxels = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumSurfaceDenoisingDisplacementVoxels"))
                               .toDouble(options.maximumSurfaceDenoisingDisplacementVoxels)),
        0.0f,
        1.0f);
    options.maximumSurfaceDenoisingNormalAngleDegrees = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMaximumSurfaceDenoisingNormalAngleDegrees"))
                               .toDouble(options.maximumSurfaceDenoisingNormalAngleDegrees)),
        5.0f,
        170.0f);
    options.surfaceDenoisingBoundaryProtectionRings = qBound(
        0,
        settings.value(QStringLiteral("tsdfSurfaceDenoisingBoundaryProtectionRings"))
            .toInt(options.surfaceDenoisingBoundaryProtectionRings),
        2);
    options.enableProtectedTaubinSurfaceDenoising = settings.value(
        QStringLiteral("tsdfProtectedTaubinSurfaceDenoising"))
        .toBool(options.enableProtectedTaubinSurfaceDenoising);
    options.enablePostSimplificationSurfaceDenoising = settings.value(
        QStringLiteral("tsdfPostSimplificationSurfaceDenoising"))
        .toBool(options.enablePostSimplificationSurfaceDenoising);
    const bool automatic_trim_weak_boundary_tips = options.resolution >= 384;
    options.trimWeakBoundaryTips = settings.value(
        QStringLiteral("tsdfTrimWeakBoundaryTips")).toBool(
            automatic_trim_weak_boundary_tips);
    const int automatic_weak_boundary_trim_passes = options.resolution >= 384 ? 2 : 1;
    options.weakBoundaryTipTrimPasses = qBound(
        1,
        settings.value(QStringLiteral("tsdfWeakBoundaryTipTrimPasses"))
            .toInt(automatic_weak_boundary_trim_passes),
        4);
    enforceObservationOnlySurfacePolicy(settings, &options);
    return options;
}

float visibilityOccupancyMedianVoxelStep(
    const xjw::mesh::DepthTsdfLayout &layout,
    int occupancy_resolution)
{
    std::array<float, 3> occupancy_steps{};
    for (int axis = 0; axis < 3; ++axis)
    {
        occupancy_steps[axis] =
            (layout.boundsMax[axis] - layout.boundsMin[axis]) /
            static_cast<float>(std::max(2, occupancy_resolution - 1));
    }
    std::sort(occupancy_steps.begin(), occupancy_steps.end());
    return std::max(occupancy_steps[1], 1.0e-6f);
}

xjw::mesh::DepthConstrainedSurfaceRefineOptions
makeVisibilityOccupancyDepthRefineOptions(
    const QJsonObject &settings,
    const xjw::mesh::DepthTsdfLayout &layout,
    int occupancy_resolution,
    bool orbital_workspace)
{
    xjw::mesh::DepthConstrainedSurfaceRefineOptions options;
    const float occupancy_step = visibilityOccupancyMedianVoxelStep(
        layout, occupancy_resolution);
    options.passes = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthRefinementPasses"))
            .toInt(orbital_workspace ? 4 : 1),
        4);
    options.depthRefine.maximumEvidenceDistance =
        occupancy_step * std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthMaximumEvidenceVoxels"))
                                   .toDouble(3.0)),
            0.5f,
            12.0f);
    options.depthRefine.maximumDisplacement =
        occupancy_step * std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthMaximumDisplacementVoxels"))
                                   .toDouble(orbital_workspace ? 0.60 : 0.20)),
            0.02f,
            1.0f);
    options.depthRefine.minimumViewCount = qBound(
        2,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumViews"))
            .toInt(2),
        8);
    options.depthRefine.minimumNativeViewCount = qBound(
        0,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumNativeViews"))
            .toInt(1),
        options.depthRefine.minimumViewCount);
    options.depthRefine.minimumDepthConfidence = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumConfidence"))
                               .toDouble(0.20)),
        0.0f,
        1.0f);
    options.depthRefine.repairedObservationWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthRepairedWeight"))
                               .toDouble(0.25)),
        0.0f,
        1.0f);
    options.depthRefine.enableInverseDepthSpreadWeighting =
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthSpreadWeighting"))
            .toBool(true);
    options.depthRefine.inverseDepthSpreadWeightKnee = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthSpreadWeightKnee"))
                               .toDouble(0.005)),
        0.0f,
        0.099f);
    options.depthRefine.inverseDepthSpreadWeightZero = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthSpreadWeightZero"))
                               .toDouble(0.015)),
        options.depthRefine.inverseDepthSpreadWeightKnee + 1.0e-6f,
        0.10f);
    options.depthRefine.minimumInverseDepthSpreadWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumSpreadWeight"))
                               .toDouble(0.05)),
        0.0f,
        1.0f);
    options.depthRefine.maximumViewMedianAbsoluteDeviation =
        occupancy_step * std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthMaximumMadVoxels"))
                                   .toDouble(1.5)),
            0.25f,
            6.0f);
    options.depthRefine.enableCrossViewBiasCompensation =
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthBiasCompensation"))
            .toBool(true);
    options.depthRefine.minimumCrossViewBiasPairSamples = qBound(
        8,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthBiasMinimumPairSamples"))
            .toInt(64),
        4096);
    options.depthRefine.maximumCrossViewBias =
        occupancy_step * std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthMaximumBiasVoxels"))
                                   .toDouble(0.75)),
            0.05f,
            3.0f);
    options.depthRefine.minimumAnchorWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumAnchorWeight"))
                               .toDouble(0.05)),
        0.0f,
        1.0f);
    options.depthRefine.regularizationMaximumNormalAngleDegrees =
        std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthMaximumNormalAngleDegrees"))
                                   .toDouble(55.0)),
            5.0f,
            89.0f);
    options.depthRefine.enableGlobalRobustSolver =
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthGlobalRobustSolver"))
            .toBool(true);
    options.depthRefine.globalSolverIrlsIterations = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthGlobalSolverIrlsIterations"))
            .toInt(4),
        10);
    options.depthRefine.globalSolverMaximumPcgIterations = qBound(
        1,
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthGlobalSolverMaximumPcgIterations"))
            .toInt(120),
        500);
    options.depthRefine.globalSolverConvergenceTolerance =
        std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthGlobalSolverTolerance"))
                                   .toDouble(1.0e-5)),
            1.0e-9f,
            1.0e-2f);
    options.depthRefine.globalSolverRobustScaleMultiplier =
        std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthGlobalSolverRobustScale"))
                                   .toDouble(0.5)),
            0.01f,
            4.0f);
    options.depthRefine.globalSolverLaplacianWeight =
        std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthGlobalSolverLaplacian"))
                                   .toDouble(orbital_workspace ? 2.50 : 0.60)),
            0.0f,
            20.0f);
    options.depthRefine.globalSolverHullPriorWeight =
        std::clamp(
            static_cast<float>(settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyDepthGlobalSolverCarrierPrior"))
                                   .toDouble(orbital_workspace ? 0.04 : 0.05)),
            1.0e-6f,
            1.0f);
    options.minimumAreaRatio = std::clamp(
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumAreaRatio"))
            .toDouble(orbital_workspace ? 0.80 : 0.97),
        0.80,
        1.0);
    options.maximumAreaRatio = std::clamp(
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMaximumAreaRatio"))
            .toDouble(1.03),
        1.0,
        1.20);
    options.minimumVolumeRatio = std::clamp(
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumVolumeRatio"))
            .toDouble(orbital_workspace ? 0.85 : 0.98),
        0.80,
        1.0);
    options.maximumVolumeRatio = std::clamp(
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMaximumVolumeRatio"))
            .toDouble(1.02),
        1.0,
        1.20);
    options.minimumFaceNormalDot = std::clamp(
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumFaceNormalDot"))
            .toDouble(0.25),
        -1.0,
        1.0);
    options.minimumFaceAreaRatio = std::clamp(
        settings.value(QStringLiteral(
            "tsdfVisibilityOccupancyDepthMinimumFaceAreaRatio"))
            .toDouble(0.10),
        0.01,
        1.0);
    options.removeMedianNormalBias = settings.value(QStringLiteral(
        "tsdfVisibilityOccupancyDepthRemoveMedianNormalBias"))
        .toBool(true);
    return options;
}

void mergePayload(const QJsonObject &source, QJsonObject *target)
{
    if (!target)
    {
        return;
    }
    for (auto it = source.constBegin(); it != source.constEnd(); ++it)
    {
        (*target)[it.key()] = it.value();
    }
}

int maximumReliableOrbitalResolution(
    const QVector<DepthTsdfFrame> &frames,
    int requestedResolution)
{
    std::vector<int> image_dimensions;
    image_dimensions.reserve(static_cast<std::size_t>(frames.size()));
    for (const DepthTsdfFrame &frame : frames)
    {
        if (!frame.depth.empty())
        {
            image_dimensions.push_back(
                std::min(frame.depth.cols, frame.depth.rows));
        }
    }
    if (image_dimensions.empty())
    {
        return requestedResolution;
    }
    const auto middle =
        image_dimensions.begin() + image_dimensions.size() / 2;
    std::nth_element(
        image_dimensions.begin(), middle, image_dimensions.end());
    const int median_dimension = *middle;
    const int sampled_resolution =
        std::max(192, median_dimension * 2 / 5);
    const int aligned_resolution =
        std::max(192, sampled_resolution / 64 * 64);
    return std::min(
        requestedResolution,
        std::clamp(aligned_resolution, 192, 384));
}

} // namespace

xjw::mesh::DepthTsdfOptions depthTsdfOptionsFromSettings(const QJsonObject &settings,
                                                         int requestedResolution)
{
    return makeDepthTsdfOptions(settings, requestedResolution);
}

void applyOrbitalDepthTsdfDefaults(const QJsonObject &settings,
                                   xjw::mesh::DepthTsdfOptions *options,
                                   int maximumReliableResolution)
{
    if (!options)
    {
        return;
    }

    if (!settings.contains(QStringLiteral("tsdfSupportMaskFreeSpaceCarving")))
    {
        options->enableSupportMaskFreeSpaceCarving = false;
    }
    if (!settings.contains(QStringLiteral("tsdfNarrowBandActivation")))
    {
        options->enableNarrowBandActivation = true;
    }
    if (!settings.contains(QStringLiteral("tsdfPixelEvidenceWeighting")))
    {
        options->enablePixelEvidenceWeighting = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfEvidenceSupportWeightDecoupling")))
    {
        options->enableEvidenceSupportWeightDecoupling = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfAdaptiveConflictRobustWeighting")))
    {
        options->enableAdaptiveConflictRobustWeighting = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfAdaptiveConflictWeightKnee")))
    {
        options->adaptiveConflictWeightKnee = 0.20f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfAdaptiveConflictWeightZero")))
    {
        options->adaptiveConflictWeightZero = 0.75f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfMinimumAdaptiveConflictWeightMultiplier")))
    {
        options->minimumAdaptiveConflictWeightMultiplier = 0.02f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfWeakEvidenceSurfaceOnlyIntegration")))
    {
        options->enableWeakEvidenceSurfaceOnlyIntegration = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfGeometrySingleViewNeighborhoodGuard")))
    {
        options->enableGeometrySingleViewNeighborhoodGuard = true;
    }
    if (!settings.contains(QStringLiteral("tsdfMinimumSupportMaskFreeSpaceViews")))
    {
        options->minimumSupportMaskFreeSpaceViews = 5;
    }
    if (!settings.contains(QStringLiteral("tsdfRobustFrameQualityWeighting")))
    {
        options->enableRobustFrameQualityWeighting = true;
    }
    if (!settings.contains(QStringLiteral("tsdfRobustFrameQualityRejection")))
    {
        options->enableRobustFrameQualityRejection = false;
    }
    if (!settings.contains(QStringLiteral("tsdfOrbitalFrameCoverageProtection")))
    {
        options->enableOrbitalFrameCoverageProtection = true;
    }
    if (!settings.contains(QStringLiteral("tsdfOrbitalGapBoundaryRecovery")))
    {
        options->enableOrbitalGapBoundaryRecovery = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfOrbitalGapAdaptiveTruncation")))
    {
        options->enableOrbitalGapAdaptiveTruncation = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfOrbitalGapAdaptiveTruncationScale")))
    {
        options->orbitalGapAdaptiveTruncationScale = 1.50f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfOrbitalGapAdaptiveMaximumTruncationVoxels")))
    {
        options->orbitalGapAdaptiveMaximumTruncationVoxels = 16.0f;
    }
    if (!settings.contains(QStringLiteral("tsdfSurfaceEvidenceFreeSpaceVeto")))
    {
        options->enableSurfaceEvidenceFreeSpaceVeto = true;
    }
    if (!settings.contains(QStringLiteral("tsdfDepthCompletenessDiagnostics")))
    {
        options->enableDepthCompletenessDiagnostics = true;
    }
    if (!settings.contains(QStringLiteral("tsdfEnforceDepthCompletenessGate")))
    {
        options->enforceDepthCompletenessGate = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfDepthCompletenessToleranceVoxels")))
    {
        // Orbital MVS depth is fused across a complete viewing ring and then
        // simplified/denoised.  Judge observation support within one TSDF
        // truncation band instead of the generic four-voxel tolerance.
        options->depthCompletenessToleranceVoxels = std::max(
            6.0f, options->truncationVoxels);
    }
    if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyCompletion")))
    {
        options->enableVisibilityOccupancyCompletion = true;
    }
    if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyResolution")))
    {
        options->visibilityOccupancyResolution = 72;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfVisibilityOccupancyCellBoundaryExtraction")))
    {
        // Keep the low-resolution visibility graph cut as a topology/sign
        // prior, but extract the surface from the completed high-resolution
        // TSDF.  Emitting the occupancy-cell boundary here quantizes the
        // geometry to the carrier grid and discards the sub-voxel zero
        // crossing retained by MC33 (the same distinction made by Open3D's
        // TSDF integration/extraction path).
        options->visibilityOccupancyCellBoundaryExtraction = false;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfMc33RequireSupportedSignChange")))
    {
        // Visibility completion deliberately supplies a topology-consistent
        // sign for samples without direct depth support. Requiring both edge
        // endpoints to be observed would cut those recovered regions away.
        options->mc33RequireSupportedSignChange = false;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfVisibilityOccupancyClosingIterations")))
    {
        options->visibilityOccupancyClosingIterations = 1;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumHandleRepairPasses")))
    {
        options->visibilityOccupancyMaximumHandleRepairPasses = 8;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumHandleRepairAcceptedCandidateCount")))
    {
        options->visibilityOccupancyMaximumHandleRepairAcceptedCandidateCount = 128;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumHandleRepairCandidateSampleCount")))
    {
        options->visibilityOccupancyMaximumHandleRepairCandidateSampleCount = 4096;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumHandleRepairSubsetSampleCount")))
    {
        options->visibilityOccupancyMaximumHandleRepairSubsetSampleCount = 256;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfVisibilityOccupancyMaximumHandleRepairSubsetSeedCount")))
    {
        options->visibilityOccupancyMaximumHandleRepairSubsetSeedCount = 4096;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfVisibilityOccupancyTopologyLockedResidualBlend")))
    {
        options->visibilityOccupancyTopologyLockedResidualBlend = true;
    }
    if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyObservedBand")))
    {
        options->visibilityOccupancyObservedBand = 1.0f;
    }
    if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyCarrierBand")))
    {
        options->visibilityOccupancyCarrierBand = 1.0f;
    }
    if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyMaximumResidual")))
    {
        options->visibilityOccupancyMaximumResidual = 1.0f;
    }
    if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyDetailBlend")))
    {
        options->visibilityOccupancyDetailBlend = 1.0f;
    }
    const bool high_detail_model = options->resolution >= 384 &&
        options->simplifyTargetFaces > 0 &&
        (options->simplifyTargetFaces <= 120000 ||
         (options->enableOpenMeshSimplification &&
          options->simplifyTargetFaces <= 240000));
    if (!high_detail_model)
    {
        // Low-resolution and large-face orbital presets return before the
        // detail-only defaults below.  Re-apply the user-facing interpolation
        // contract here as well, otherwise the orbital defaults above leave
        // occupancy completion enabled even when the dialog says disabled.
        enforceObservationOnlySurfacePolicy(settings, options);
        return;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfInverseDepthSpreadWeighting")))
    {
        options->enableInverseDepthSpreadWeighting = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfInverseDepthSpreadWeightKnee")))
    {
        options->inverseDepthSpreadWeightKnee = 0.005f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfInverseDepthSpreadWeightZero")))
    {
        options->inverseDepthSpreadWeightZero = 0.015f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfMinimumInverseDepthSpreadWeightMultiplier")))
    {
        options->minimumInverseDepthSpreadWeightMultiplier = 0.05f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfInverseDepthSpreadSupportWeightDecoupling")))
    {
        options->enableInverseDepthSpreadSupportWeightDecoupling = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfInverseDepthSpreadSupportWeightExponent")))
    {
        options->inverseDepthSpreadSupportWeightExponent = 0.25f;
    }
    if (settings.value(QStringLiteral(
            "tsdfOrbitalAdaptiveResolution")).toBool(true) &&
        maximumReliableResolution > 0)
    {
        const int requested_resolution = options->resolution;
        options->resolution = std::min(
            options->resolution, maximumReliableResolution);
        if (options->resolution < requested_resolution &&
            options->simplifyTargetFaces > 0)
        {
            const double resolution_ratio =
                static_cast<double>(options->resolution) /
                static_cast<double>(requested_resolution);
            const int uncertainty_matched_face_budget =
                std::max(
                    60000,
                    static_cast<int>(std::lround(
                        options->simplifyTargetFaces *
                        resolution_ratio * resolution_ratio)));
            options->simplifyTargetFaces = std::min(
                options->simplifyTargetFaces,
                uncertainty_matched_face_budget);
        }
    }
    if (!settings.contains(QStringLiteral("tsdfTruncationVoxels")))
    {
        options->truncationVoxels = 7.5f;
    }
    if (!settings.contains(QStringLiteral("tsdfSurfaceSupportBandVoxels")))
    {
        options->surfaceSupportBandVoxels = 7.5f;
    }
    if (!settings.contains(QStringLiteral("tsdfUncertaintyAdaptiveTruncation")))
    {
        options->enableUncertaintyAdaptiveTruncation = true;
    }
    if (!settings.contains(QStringLiteral("tsdfUncertaintyAdaptiveScale")))
    {
        options->uncertaintyAdaptiveScale = 0.40f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfUncertaintyAdaptiveActivationRatio")))
    {
        options->uncertaintyAdaptiveActivationRatio = 1.20f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfUncertaintyAdaptiveMaximumTruncationVoxels")))
    {
        options->uncertaintyAdaptiveMaximumTruncationVoxels = 12.0f;
    }
    if (!settings.contains(QStringLiteral("tsdfAllowInvalidNearestPixelRecovery")))
    {
        options->allowInvalidNearestPixelRecovery = false;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfMaximumInvalidNearestPixelRecoveryInverseDepthSpread")))
    {
        options->maximumInvalidNearestPixelRecoveryInverseDepthSpread = 0.01f;
    }
    if (!settings.contains(QStringLiteral("tsdfGeometryZeroCrossingRecovery")))
    {
        options->enableGeometryZeroCrossingRecovery = false;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfCrossViewAnchoredSurfaceRecovery")))
    {
        options->enableCrossViewAnchoredSurfaceRecovery = false;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfCrossViewAnchoredMinimumObservationWeight")))
    {
        options->crossViewAnchoredMinimumObservationWeight = 0.25f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfCrossViewAnchoredMinimumSupportedCorners")))
    {
        options->crossViewAnchoredMinimumSupportedCorners = 2;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfCrossViewAnchoredMinimumCellVotes")))
    {
        options->crossViewAnchoredMinimumCellVotes = 1;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfCrossViewAnchoredGrowthPasses")))
    {
        options->crossViewAnchoredGrowthPasses = 2;
    }
    if (!settings.contains(QStringLiteral("tsdfGeometryZeroCrossingCellSheets")))
    {
        options->enableGeometryZeroCrossingCellSheets = false;
    }
    if (!settings.contains(QStringLiteral("tsdfContourBandZeroCrossingSupport")))
    {
        options->enableContourBandZeroCrossingSupport = false;
    }
    if (!settings.contains(QStringLiteral("tsdfAdaptiveTgvRegularization")))
    {
        options->enableAdaptiveTgvRegularization = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfAdaptiveTgvRecoverUnsupportedSamples")))
    {
        options->adaptiveTgvRecoverUnsupportedSamples = false;
    }
    if (!settings.contains(QStringLiteral("tsdfSurfacePatchSupport")))
    {
        options->enableSurfacePatchSupport = false;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfGeometryVerifiedBoundaryRecovery")))
    {
        options->enableGeometryVerifiedBoundaryRecovery = false;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfAllowGeometryVerifiedSingleObservation")))
    {
        options->allowGeometryVerifiedSingleObservation = false;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfMinimumGeometrySupportCount")))
    {
        options->minimumGeometrySupportCount = 4;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfDiscontinuityAwareSampling")))
    {
        options->enableDiscontinuityAwareSampling = false;
    }
    if (!settings.contains(QStringLiteral("tsdfCrossViewConsensusDepth")))
    {
        options->enableCrossViewConsensusDepth = true;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfMaximumCrossViewConsensusInverseDepthSpread")))
    {
        options->maximumCrossViewConsensusInverseDepthSpread = 0.008f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfMaximumObservationInverseDepthSpread")))
    {
        options->maximumObservationInverseDepthSpread = 0.015f;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfDepthValidBoundaryErosionPixels")))
    {
        options->depthValidBoundaryErosionPixels = 1;
    }
    if (!settings.contains(QStringLiteral(
            "tsdfBoundarySmoothingIterations")))
    {
        options->boundarySmoothingIterations = 1;
    }
    if (!settings.contains(QStringLiteral("tsdfTrimWeakBoundaryTips")))
    {
        options->trimWeakBoundaryTips = false;
    }
    if (options->fillSmallBoundaryHoles)
    {
        if (!settings.contains(QStringLiteral(
                "tsdfSilhouetteAwareFinalHoleFill")))
        {
            options->enableSilhouetteAwareFinalHoleFill = true;
        }
        if (!settings.contains(QStringLiteral(
                "tsdfVisibilityConstrainedFinalHoleFill")))
        {
            options->enableVisibilityConstrainedFinalHoleFill = true;
        }
        if (!settings.contains(QStringLiteral(
                "tsdfFinalHoleFillMaximumBoundaryEdges")))
        {
            options->finalHoleFillMaximumBoundaryEdges = 24;
        }
        if (!settings.contains(QStringLiteral(
                "tsdfFinalHoleFillMaximumDiameterVoxels")))
        {
            options->finalHoleFillMaximumDiameterVoxels = 4.0f;
        }
        if (!settings.contains(QStringLiteral(
                "tsdfFinalHoleFillMaximumFaceGrowthRatio")))
        {
            options->finalHoleFillMaximumFaceGrowthRatio = 0.03f;
        }
        if (!settings.contains(QStringLiteral(
                "tsdfVisibilityHoleFillMinimumSupportingViews")))
        {
            options->visibilityHoleFillMinimumSupportingViews = 2;
        }
        if (!settings.contains(QStringLiteral(
                "tsdfVisibilityHoleFillMaximumConflictViews")))
        {
            options->visibilityHoleFillMaximumConflictViews = 0;
        }
    }
    enforceObservationOnlySurfacePolicy(settings, options);
}

FinalSurfaceDenoisingResult applyTopologyGuardedFinalSurfaceDenoising(
    TriMesh *mesh,
    int iterations,
    float lambda,
    float mu,
    float maximumDisplacement,
    float featureAngleDegrees,
    int boundaryProtectionRings)
{
    FinalSurfaceDenoisingResult result;
    if (mesh == nullptr || mesh->empty() || iterations <= 0 ||
        lambda <= 0.0f || maximumDisplacement <= 0.0f)
    {
        return result;
    }

    result.attempted = true;
    result.areaBefore = meshSurfaceArea(*mesh);
    result.qualityBefore = evaluateMeshTopologyQuality(*mesh);
    const MeshTopologySignature topology_before =
        meshTopologySignature(result.qualityBefore);

    TriMesh candidate = *mesh;
    result.movedVertexCount =
        detail::smoothSurfaceVerticesTaubinProtected(
            &candidate,
            iterations,
            lambda,
            mu,
            maximumDisplacement,
            featureAngleDegrees,
            boundaryProtectionRings);
    detail::recomputeNormals(&candidate);
    result.areaAfter = meshSurfaceArea(candidate);
    result.qualityAfter = evaluateMeshTopologyQuality(candidate);
    const MeshTopologySignature topology_after =
        meshTopologySignature(result.qualityAfter);

    const double area_ratio = result.areaBefore > 1.0e-12
        ? result.areaAfter / result.areaBefore
        : 0.0;
    const bool geometry_preserved =
        result.movedVertexCount > 0 &&
        candidate.vertexCount() == mesh->vertexCount() &&
        candidate.faceCount() == mesh->faceCount() &&
        hasSameFaceIndexBuffer(*mesh, candidate) &&
        topology_after == topology_before &&
        result.qualityAfter.validFaceCount ==
            result.qualityBefore.validFaceCount &&
        area_ratio >= 0.96 && area_ratio <= 1.01 &&
        result.qualityAfter.highAspectFaceRatio <=
            result.qualityBefore.highAspectFaceRatio + 1.0e-6 &&
        result.qualityAfter.extremeAspectFaceRatio <=
            result.qualityBefore.extremeAspectFaceRatio + 1.0e-6;
    const bool normal_quality_not_worse =
        result.qualityAfter.adjacentNormalAngleP90Degrees <=
            result.qualityBefore.adjacentNormalAngleP90Degrees + 1.0e-6 &&
        result.qualityAfter.adjacentNormalAngleOver30Ratio <=
            result.qualityBefore.adjacentNormalAngleOver30Ratio + 1.0e-6;
    const bool normal_quality_improved =
        result.qualityAfter.adjacentNormalAngleP90Degrees <=
            result.qualityBefore.adjacentNormalAngleP90Degrees - 0.25 ||
        result.qualityAfter.adjacentNormalAngleOver30Ratio <=
            result.qualityBefore.adjacentNormalAngleOver30Ratio - 0.001 ||
        result.qualityAfter.adjacentNormalAngleMedianDegrees <=
            result.qualityBefore.adjacentNormalAngleMedianDegrees - 0.10;

    result.accepted = geometry_preserved &&
        normal_quality_not_worse && normal_quality_improved;
    if (result.accepted)
    {
        *mesh = std::move(candidate);
    }
    return result;
}

bool visibilityOccupancyDepthRefinementEnabled(const QJsonObject &settings,
                                               bool orbitalWorkspace)
{
    return settings.value(QStringLiteral(
        "tsdfVisibilityOccupancyDepthRefinement")).toBool(orbitalWorkspace);
}

QString orbitalRoleForDepthFrame(const QJsonArray &roles, int refIndex)
{
    for (const QJsonValue &value : roles)
    {
        const QJsonObject role = value.toObject();
        if (role.value(QStringLiteral("ref_index")).toInt(-1) == refIndex)
        {
            return role.value(QStringLiteral("role")).toString();
        }
    }
    return {};
}

QStringList worstDepthCompletenessLabels(
    const DepthMeshCompletenessStatistics &completeness,
    const QJsonArray &orbitalRoles,
    int maximumCount = 3)
{
    std::vector<DepthMeshFrameCompleteness> frames(
        completeness.frames.cbegin(), completeness.frames.cend());
    std::sort(
        frames.begin(),
        frames.end(),
        [](const DepthMeshFrameCompleteness &lhs,
           const DepthMeshFrameCompleteness &rhs)
        {
            return lhs.recall < rhs.recall;
        });
    QStringList labels;
    const int count = std::min(maximumCount, static_cast<int>(frames.size()));
    for (int index = 0; index < count; ++index)
    {
        const DepthMeshFrameCompleteness &frame =
            frames[static_cast<std::size_t>(index)];
        const QString role = orbitalRoleForDepthFrame(
            orbitalRoles, frame.refIndex);
        labels.push_back(
            role.isEmpty()
                ? QStringLiteral("%1=%2%")
                      .arg(frame.refIndex)
                      .arg(100.0 * frame.recall, 0, 'f', 1)
                : QStringLiteral("%1=%2%[%3]")
                      .arg(frame.refIndex)
                      .arg(100.0 * frame.recall, 0, 'f', 1)
                      .arg(role));
    }
    return labels;
}

void addDepthCompletenessPayload(
    const DepthMeshCompletenessStatistics &completeness,
    const QString &prefix,
    QJsonObject *payload)
{
    if (payload == nullptr)
    {
        return;
    }
    (*payload)[prefix + QStringLiteral("available")] = completeness.available;
    (*payload)[prefix + QStringLiteral("gate_passed")] = completeness.gatePassed;
    (*payload)[prefix + QStringLiteral("distance_method")] =
        QStringLiteral("exact_point_to_triangle_bvh");
    (*payload)[prefix + QStringLiteral("tolerance")] = completeness.tolerance;
    (*payload)[prefix + QStringLiteral("sampled_point_count")] =
        static_cast<double>(completeness.sampledDepthPointCount);
    (*payload)[prefix + QStringLiteral("explained_point_count")] =
        static_cast<double>(completeness.explainedDepthPointCount);
    (*payload)[prefix + QStringLiteral("aggregate_recall")] =
        completeness.aggregateRecall;
    (*payload)[prefix + QStringLiteral("minimum_frame_recall")] =
        completeness.minimumFrameRecall;
    (*payload)[prefix + QStringLiteral("p10_frame_recall")] =
        completeness.p10FrameRecall;
    (*payload)[prefix + QStringLiteral("median_frame_recall")] =
        completeness.medianFrameRecall;
    QJsonArray frames;
    for (const DepthMeshFrameCompleteness &frame : completeness.frames)
    {
        frames.push_back(QJsonObject{
            {QStringLiteral("ref_index"), frame.refIndex},
            {QStringLiteral("auxiliary_surface_only"),
             frame.auxiliarySurfaceOnly},
            {QStringLiteral("sampled_point_count"),
             static_cast<double>(frame.sampledDepthPointCount)},
            {QStringLiteral("explained_point_count"),
             static_cast<double>(frame.explainedDepthPointCount)},
            {QStringLiteral("recall"), frame.recall}
        });
    }
    (*payload)[prefix + QStringLiteral("frames")] = frames;
}

DepthMeshCompletenessStatistics evaluateDepthCompleteness(
    const TriMesh &mesh,
    const QVector<DepthTsdfFrame> &frames,
    const DepthTsdfOptions &options,
    const DepthTsdfLayout &layout)
{
    DepthMeshCompletenessOptions completeness_options;
    completeness_options.maximumDepthSamplesPerFrame =
        options.depthCompletenessMaximumSamplesPerFrame;
    completeness_options.tolerance = std::max({
        layout.voxelSize[0],
        layout.voxelSize[1],
        layout.voxelSize[2]}) *
        std::max(1.0f, options.depthCompletenessToleranceVoxels);
    completeness_options.minimumP10FrameRecall =
        options.minimumDepthCompletenessP10Recall;
    completeness_options.minimumMedianFrameRecall =
        options.minimumDepthCompletenessMedianRecall;
    return DepthMeshCompleteness::evaluate(
        mesh, frames, completeness_options);
}

bool shouldUseOrbitalVisualHullCompletion(bool orbitalWorkspace,
                                          bool enabled,
                                          bool observationOnlySurface,
                                          double aggregateProjectionRecall,
                                          int boundaryEdgeCount,
                                          int faceCount)
{
    if (!orbitalWorkspace || !enabled || observationOnlySurface ||
        faceCount <= 0)
    {
        return false;
    }

    const double boundary_ratio =
        static_cast<double>(std::max(0, boundaryEdgeCount)) /
        static_cast<double>(std::max(1, faceCount));
    const bool projection_recall_failed =
        aggregateProjectionRecall > 0.0 &&
        aggregateProjectionRecall < 0.82;
    const bool unresolved_boundary_loops =
        boundaryEdgeCount > 128;
    return projection_recall_failed || unresolved_boundary_loops ||
           boundary_ratio > 0.025;
}

int meshResolutionFromSettings(const QJsonObject &settings)
{
    const double requestedResolution = settings.value(QStringLiteral("meshResolution")).toDouble(0.0);
    if (requestedResolution > 0.0)
    {
        return qBound(64, static_cast<int>(std::lround(requestedResolution)), 1024);
    }

    const int octreeDepth = settings.value(QStringLiteral("octreeDepth")).toInt(10);
    const int depth = qBound(4, octreeDepth, 14);
    return qBound(64, 1 << (depth - 2), 1024);
}

QString depthReconstructionModeFromSettings(const QJsonObject &settings)
{
    const QString requested = settings.value(QStringLiteral("reconstruction_mode"))
                                  .toString()
                                  .trimmed()
                                  .toLower();
    if (!requested.isEmpty())
    {
        return requested;
    }

    return settings.value(QStringLiteral("surface_type")).toString()
               == QStringLiteral("height_field")
        ? QStringLiteral("poisson_legacy")
        : QStringLiteral("depth_tsdf");
}

xjw::mesh::ReconstructionConfig reconstructionConfigFromModelSettings(const QJsonObject &settings)
{
    xjw::mesh::ReconstructionConfig config;

    const QString surfaceType =
        settings.value(QStringLiteral("surface_type")).toString(QStringLiteral("arbitrary_3d"));
    config.resolution = meshResolutionFromSettings(settings);
    config.smoothIterations = qBound(0, settings.value(QStringLiteral("smoothIter")).toInt(3), 50);
    config.smoothLambda = 0.5f;
    config.padding = 0.05f;
    config.forcePoisson =
        surfaceType != QStringLiteral("height_field") &&
        settings.value(QStringLiteral("method"))
            .toString(QStringLiteral("Poisson Surface"))
            .contains(QStringLiteral("Poisson"), Qt::CaseInsensitive);
    config.allowHeightGridFallback = surfaceType == QStringLiteral("height_field");
    config.orientNormalsForClosedSurface = surfaceType == QStringLiteral("arbitrary_3d");
    config.poissonDepth = qBound(7, settings.value(QStringLiteral("octreeDepth")).toInt(10), 12);
    config.poissonThreads = qBound(1, settings.value(QStringLiteral("threads")).toInt(8), 128);

    const double pointWeight =
        settings.value(QStringLiteral("poissonPointWeight")).toDouble(config.poissonPointWeight);
    config.poissonPointWeight = std::clamp(static_cast<float>(pointWeight), 0.0f, 8.0f);

    const double poissonTrim = settings.value(QStringLiteral("poissonTrim")).toDouble(config.poissonTrim);
    config.poissonTrim = std::clamp(static_cast<float>(poissonTrim), 0.0f, 12.0f);

    const QString interpolation =
        settings.value(QStringLiteral("interpolation")).toString(QStringLiteral("enabled"));
    config.fillHoles = interpolation != QStringLiteral("disabled") &&
                       settings.value(QStringLiteral("holeFill")).toBool(true);
    if (interpolation == QStringLiteral("extrapolated"))
    {
        config.holeFillPasses = std::max(16, holeFillPassesFromArea(
            settings.value(QStringLiteral("maxHoleSize")).toDouble(400.0)));
    }
    else
    {
        config.holeFillPasses =
            holeFillPassesFromArea(settings.value(QStringLiteral("maxHoleSize")).toDouble(100.0));
    }

    config.cleanSmallComponents = settings.value(QStringLiteral("cleanSmall")).toBool(true);
    config.minComponentFaces = qBound(2, settings.value(QStringLiteral("minFaces")).toInt(100), 100000);

    const QString qualityProfile = settings.contains(QStringLiteral("qualityProfile"))
        ? settings.value(QStringLiteral("qualityProfile")).toString()
        : QStringLiteral("balanced");
    const QString quality = settings.value(QStringLiteral("quality")).toString();
    const QString voxelDensity = settings.value(QStringLiteral("voxelDensity")).toString(QStringLiteral("medium"));

    if (qualityProfile == QStringLiteral("detail"))
    {
        config.resolution = std::max(config.resolution, quality == QStringLiteral("ultra") ? 384 : 320);
        config.poissonDepth = std::max(config.poissonDepth, quality == QStringLiteral("ultra") ? 11 : 10);
        config.poissonPointWeight = std::max(config.poissonPointWeight, 4.8f);
        config.poissonTrim = std::max(config.poissonTrim, 9.0f);
        config.simplifyTargetFaces = std::max(config.simplifyTargetFaces, 65000);
        config.enableDownsample = false;
        config.voxelSimplifyFactor = 1.15f;
        config.kNormals = std::max(config.kNormals, 18);
        config.smoothIterations = std::max(config.smoothIterations, 4);
        config.smoothLambda = 0.36f;
    }
    else if (qualityProfile == QStringLiteral("lite"))
    {
        config.resolution = std::min(config.resolution, 224);
        config.poissonDepth = std::min(config.poissonDepth, 9);
        config.poissonPointWeight = std::min(config.poissonPointWeight, 3.2f);
        config.poissonTrim = std::min(config.poissonTrim, 8.4f);
        config.simplifyTargetFaces = 16000;
        config.enableDownsample = true;
        config.downsampleVoxelScale = 1.0f;
        config.voxelSimplifyFactor = 2.5f;
        config.smoothIterations = std::min(3, config.smoothIterations + 1);
        config.smoothLambda = 0.55f;
    }
    else if (voxelDensity == QStringLiteral("coarse"))
    {
        config.voxelSimplifyFactor = 2.6f;
        config.enableDownsample = true;
        config.downsampleVoxelScale = 1.0f;
        config.poissonPointWeight = std::min(config.poissonPointWeight, 3.6f);
        config.simplifyTargetFaces = 14000;
    }
    else if (voxelDensity == QStringLiteral("fine"))
    {
        config.voxelSimplifyFactor = 1.25f;
        config.enableDownsample = false;
        config.denoiseStdMul = 1.8f;
        config.kNormals = 18;
        config.poissonPointWeight = std::max(config.poissonPointWeight, 4.6f);
        config.poissonTrim = std::max(8.0f, config.poissonTrim);
        config.simplifyTargetFaces = 60000;
    }
    else
    {
        config.voxelSimplifyFactor = 1.8f;
        config.enableDownsample = true;
        config.downsampleVoxelScale = 0.8f;
        config.simplifyTargetFaces = 28000;
    }

    const int targetFaces =
        settings.value(QStringLiteral("simplifyTargetFaces"))
            .toInt(settings.value(QStringLiteral("targetFaces")).toInt(0));
    if (targetFaces > 0)
    {
        config.simplifyTargetFaces = qBound(1000, targetFaces, 2000000);
    }
    else if (settings.contains(QStringLiteral("targetFaces")))
    {
        config.simplifyTargetFaces = 0;
    }

    if (config.forcePoisson)
    {
        // PlaPoint's current octree solver is capped at depth 8. Feeding tens of
        // millions of nearly redundant samples only increases solve time and
        // memory pressure; it cannot raise the octree resolution. Keep enough
        // samples to support the requested face budget while preserving a
        // bounded production runtime.
        if (config.simplifyTargetFaces > 0)
        {
            const long long requested_samples =
                static_cast<long long>(config.simplifyTargetFaces) * 5LL / 2LL;
            config.maxInputPointsForMeshing = static_cast<int>(
                std::clamp(requested_samples, 250000LL, 400000LL));
        }
        else
        {
            config.maxInputPointsForMeshing = 400000;
        }
    }

    const QString depthFiltering =
        settings.value(QStringLiteral("depthFiltering")).toString(QStringLiteral("moderate"));
    if (depthFiltering == QStringLiteral("disabled"))
    {
        config.enableDenoise = false;
    }
    else if (depthFiltering == QStringLiteral("mild"))
    {
        config.enableDenoise = true;
        config.denoiseK = 16;
        config.denoiseStdMul = std::max(config.denoiseStdMul, 1.6f);
    }
    else if (depthFiltering == QStringLiteral("aggressive"))
    {
        config.enableDenoise = true;
        config.denoiseK = std::max(config.denoiseK, 28);
        config.denoiseStdMul = std::min(config.denoiseStdMul, 0.95f);
    }
    else
    {
        config.enableDenoise = true;
        config.denoiseK = std::max(config.denoiseK, 20);
        config.denoiseStdMul = std::min(config.denoiseStdMul, 1.25f);
    }

    const bool decimate = settings.value(QStringLiteral("decimate")).toBool(false);
    if (decimate && config.simplifyTargetFaces > 0)
    {
        const double decimateRatio =
            std::clamp(settings.value(QStringLiteral("decimateRatio")).toDouble(0.5), 0.05, 1.0);
        config.simplifyTargetFaces =
            std::max(1000, static_cast<int>(std::lround(config.simplifyTargetFaces * decimateRatio)));
        config.enableDownsample = true;
        config.voxelSimplifyFactor =
            std::max(config.voxelSimplifyFactor, static_cast<float>(1.0 / decimateRatio));
    }

    config.verbose = false;
    return config;
}

xjw::mesh::ReconstructionConfig reconstructionConfigForDenseScene(int requestedResolution,
                                                                   bool aerialTerrain,
                                                                   bool preserveDetail)
{
    xjw::mesh::ReconstructionConfig config;
    config.resolution = qBound(64, requestedResolution, 1024);
    config.poissonDepth = 9;
    config.simplifyTargetFaces = 28000;
    config.forcePoisson = !aerialTerrain;
    config.allowHeightGridFallback = aerialTerrain;
    config.orientNormalsForClosedSurface = !aerialTerrain;

    if (aerialTerrain && preserveDetail)
    {
        config.resolution = std::max(config.resolution, 320);
        config.enableDownsample = false;
        config.simplifyTargetFaces = 65000;
        config.smoothIterations = std::max(config.smoothIterations, 4);
        config.smoothLambda = 0.36f;
    }

    return config;
}

int holeFillPassesFromArea(double maxHoleArea)
{
    if (maxHoleArea <= 0.0)
    {
        return 0;
    }

    return qBound(1, static_cast<int>(std::ceil(std::sqrt(maxHoleArea * 0.5))), 64);
}

xjw::mesh::TextureMappingConfig defaultTextureConfig()
{
    xjw::mesh::TextureMappingConfig config;
    return config;
}

xjw::mesh::TextureMappingConfig textureConfigFromSettings(const QJsonObject &settings)
{
    xjw::mesh::TextureMappingConfig config = defaultTextureConfig();
    if (!settings.isEmpty())
    {
        config.textureSize = settings.value(QStringLiteral("textureSize")).toInt(config.textureSize);
        config.imageDownscale =
            settings.value(QStringLiteral("imageDownscale")).toInt(config.imageDownscale);
        config.padding = settings.value(QStringLiteral("padding")).toInt(config.padding);
        config.keepUnmapped = settings.value(QStringLiteral("keepUnmapped")).toBool(config.keepUnmapped);
        config.enableGhostFilter =
            settings.value(QStringLiteral("ghostFilter")).toBool(config.enableGhostFilter);
        config.enableOutOfFocusFilter = settings.value(
            QStringLiteral("outOfFocusFilter")).toBool(config.enableOutOfFocusFilter);
        config.enableColorCorrection = settings.value(
            QStringLiteral("colorCorrection")).toBool(config.enableColorCorrection);
        config.sharpeningStrength = static_cast<float>(
            settings.value(QStringLiteral("sharpeningStrength"))
                .toDouble(config.sharpeningStrength));
        const QString hole_fill_mode =
            settings.value(QStringLiteral("holeFillMode")).toString();
        if (hole_fill_mode == QStringLiteral("disabled"))
        {
            config.holeFillMode = xjw::mesh::TextureHoleFillMode::Disabled;
        }
        else if (hole_fill_mode == QStringLiteral("neighbor_view_recovery"))
        {
            config.holeFillMode =
                xjw::mesh::TextureHoleFillMode::NeighborViewRecovery;
        }
        else
        {
            config.holeFillMode =
                settings.value(QStringLiteral("holeFill")).toBool(true)
                ? xjw::mesh::TextureHoleFillMode::TextureSpaceSmallHoles
                : xjw::mesh::TextureHoleFillMode::Disabled;
        }

        const QString blendMethod = settings
            .value(QStringLiteral("blendMode"))
            .toString(settings.value(QStringLiteral("blendMethod")).toString())
            .trimmed()
            .toLower();
        if (!blendMethod.isEmpty())
        {
            config.blendMethod = blendMethod.toStdString();
            if (blendMethod == QStringLiteral("best_view") ||
                blendMethod.contains(QStringLiteral("最佳")))
            {
                config.blendMode = xjw::mesh::TextureBlendMode::BestView;
            }
            else if (blendMethod == QStringLiteral("weighted_average") ||
                     blendMethod.contains(QStringLiteral("加权")))
            {
                config.blendMode = xjw::mesh::TextureBlendMode::WeightedAverage;
            }
            else
            {
                config.blendMode = xjw::mesh::TextureBlendMode::Natural;
            }
        }

        const QString uvMethod = settings
            .value(QStringLiteral("mappingMode"))
            .toString(settings.value(QStringLiteral("uvMethod")).toString())
            .trimmed()
            .toLower();
        if (!uvMethod.isEmpty())
        {
            config.uvMethod = uvMethod.toStdString();
            config.mappingMode =
                uvMethod == QStringLiteral("keep_existing_uv")
                ? xjw::mesh::TextureMappingMode::KeepExistingUv
                : xjw::mesh::TextureMappingMode::AutoProjective;
        }
    }

    return config;
}

PointCloudQualityReport evaluatePointCloudQuality(const QString &pointCloudPath,
                                                  qint64 recommendedMinimum)
{
    PointCloudQualityReport report;

    const QString extension = QFileInfo(pointCloudPath).suffix().toLower();
    if (extension == QStringLiteral("ply"))
    {
        QFile file(pointCloudPath);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            QTextStream stream(&file);
            while (!stream.atEnd())
            {
                const QString line = stream.readLine().trimmed();
                if (line.startsWith(QStringLiteral("element vertex")))
                {
                    report.pointCount = line.split(QLatin1Char(' ')).last().toLongLong();
                    report.hasCount = true;
                }
                if (line == QStringLiteral("end_header"))
                {
                    break;
                }
            }
            file.close();
        }
    }
    else
    {
        QFile file(pointCloudPath);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            report.hasCount = true;
            while (!file.atEnd())
            {
                file.readLine();
                ++report.pointCount;
            }
            file.close();
        }
    }

    report.belowRecommended = report.hasCount && report.pointCount < recommendedMinimum;
    return report;
}

WorkflowResult buildMeshAndOptionalTexture(const MeshBuildRequest &request)
{
    WorkflowResult result;

    if (request.pointCloudPath.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("点云路径为空");
        return result;
    }

    xjw::mesh::ReconstructionConfig reconstruction = request.reconstruction;
    if (request.progress)
    {
        reconstruction.progressFn = [cb = request.progress](const std::string &stage, float fraction) {
            cb(QString::fromStdString(stage), static_cast<int>(fraction * 100.0f));
        };
    }

    xjw::mesh::TriMesh mesh;
    std::string meshError;
    std::string meshAlgorithm;
    if (!xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(
            xjw::common::io::toUtf8Path(request.pointCloudPath),
            reconstruction,
            mesh,
            &meshError,
            &meshAlgorithm))
    {
        result.errorMessage = QStringLiteral("网格重建失败: %1").arg(QString::fromStdString(meshError));
        return result;
    }

    return saveMeshAndOptionalTexture(mesh,
                                      meshAlgorithm,
                                      request.outputRoot,
                                      request.exportObj,
                                      request.texture,
                                      request.progress);
}

WorkflowResult buildMeshFromDepthMaps(const DepthMapMeshBuildRequest &request)
{
    WorkflowResult result;
    const QString mode = depthReconstructionModeFromSettings(request.settings);
    result.payload[QStringLiteral("actual_mesh_algorithm")] = mode;
    result.payload[QStringLiteral("reconstruction_mode")] = mode;
    result.payload[QStringLiteral("depth_map_source_path")] = request.depthMapSourcePath;
    result.payload[QStringLiteral("source_data")] = QStringLiteral("depth_maps");

    if (request.depthMapSourcePath.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("深度图源路径为空");
        return result;
    }

    const QFileInfo depth_source_info(request.depthMapSourcePath);
    const QString output_root = request.outputRoot.isEmpty()
        ? (depth_source_info.isDir()
               ? depth_source_info.absoluteFilePath()
               : depth_source_info.absolutePath())
        : request.outputRoot;

    if (mode == QStringLiteral("depth_tsdf"))
    {
        const QVector<DepthFrameArtifact> artifacts =
            DepthMapMeshBuilder::discoverDepthFrames(request.depthMapSourcePath);
        const DepthTsdfFrameLoadResult loaded = DepthTsdfSurfaceBuilder::loadFrames(artifacts);
        if (!loaded.ok)
        {
            result.errorMessage = loaded.errorMessage;
            return result;
        }

        DepthTsdfOptions options = depthTsdfOptionsFromSettings(
            request.settings,
            request.reconstruction.resolution);
        const int orbital_frame_count = std::count_if(
            artifacts.cbegin(),
            artifacts.cend(),
            [](const DepthFrameArtifact &artifact)
            {
                return artifact.sceneProfile == QStringLiteral("orbital_object");
            });
        const bool orbital_workspace = orbital_frame_count >=
            std::max(3, static_cast<int>(artifacts.size() / 2));
        const int requested_tsdf_resolution = options.resolution;
        const int requested_target_faces = options.simplifyTargetFaces;
        if (orbital_workspace)
        {
            applyOrbitalDepthTsdfDefaults(
                request.settings,
                &options,
                maximumReliableOrbitalResolution(
                    loaded.frames, options.resolution));
        }
        const int tsdf_progress_end = request.exportObj ? 84 : 96;
        if (request.progress)
        {
            options.progress = [progress = request.progress,
                                tsdf_progress_end](const QString &stage, int percent)
            {
                const int bounded_percent = std::clamp(percent, 0, 100);
                progress(stage, bounded_percent * tsdf_progress_end / 100);
            };
        }
        options.isCancelled = request.isCancelled;
        result.payload[QStringLiteral(
            "configured_robust_frame_quality_weighting")] =
            options.enableRobustFrameQualityWeighting;
        result.payload[QStringLiteral(
            "configured_robust_frame_quality_rejection")] =
            options.enableRobustFrameQualityRejection;
        result.payload[QStringLiteral(
            "configured_orbital_frame_coverage_protection")] =
            options.enableOrbitalFrameCoverageProtection;
        result.payload[QStringLiteral(
            "configured_orbital_gap_boundary_recovery")] =
            options.enableOrbitalGapBoundaryRecovery;
        result.payload[QStringLiteral(
            "configured_orbital_gap_adaptive_truncation")] =
            options.enableOrbitalGapAdaptiveTruncation;
        result.payload[QStringLiteral(
            "configured_cross_view_anchored_surface_recovery")] =
            options.enableCrossViewAnchoredSurfaceRecovery;
        result.payload[QStringLiteral(
            "configured_uncertainty_adaptive_truncation")] =
            options.enableUncertaintyAdaptiveTruncation;
        result.payload[QStringLiteral(
            "configured_support_mask_free_space_carving")] =
            options.enableSupportMaskFreeSpaceCarving;
        result.payload[QStringLiteral(
            "configured_surface_evidence_free_space_veto")] =
            options.enableSurfaceEvidenceFreeSpaceVeto;
        result.payload[QStringLiteral(
            "configured_narrow_band_activation")] =
            options.enableNarrowBandActivation;
        result.payload[QStringLiteral(
            "configured_visibility_occupancy_completion")] =
            options.enableVisibilityOccupancyCompletion;
        result.payload[QStringLiteral("configured_interpolation")] =
            request.settings.value(QStringLiteral("interpolation"))
                .toString(QStringLiteral("enabled"));
        result.payload[QStringLiteral(
            "configured_fill_small_boundary_holes")] =
            options.fillSmallBoundaryHoles;
        result.payload[QStringLiteral(
            "configured_mc33_require_supported_sign_change")] =
            options.mc33RequireSupportedSignChange;
        result.payload[QStringLiteral(
            "configured_depth_completeness_diagnostics")] =
            options.enableDepthCompletenessDiagnostics;
        result.payload[QStringLiteral(
            "configured_depth_completeness_gate_enforcement")] =
            options.enforceDepthCompletenessGate;
        result.payload[QStringLiteral(
            "configured_triangle_quality_optimization")] =
            options.enableTriangleQualityOptimization;
        result.payload[QStringLiteral(
            "configured_triangle_quality_isotropic_remeshing")] =
            options.enableTriangleQualityIsotropicRemeshing;
        result.payload[QStringLiteral("configured_acquisition_gap_report")] =
            options.collectAcquisitionGapReport;
        result.payload[QStringLiteral("requested_tsdf_resolution")] =
            requested_tsdf_resolution;
        result.payload[QStringLiteral("configured_tsdf_resolution")] =
            options.resolution;
        result.payload[QStringLiteral("requested_target_faces")] =
            requested_target_faces;
        result.payload[QStringLiteral("configured_target_faces")] =
            options.simplifyTargetFaces;
        result.payload[QStringLiteral(
            "orbital_adaptive_resolution_applied")] =
            orbital_workspace &&
            options.resolution < requested_tsdf_resolution;
        const bool defer_depth_completeness_gate =
            orbital_workspace &&
            options.enableDepthCompletenessDiagnostics &&
            options.enforceDepthCompletenessGate;
        result.payload[QStringLiteral(
            "depth_completeness_gate_deferred_until_final_surface")] =
            defer_depth_completeness_gate;
        DepthTsdfOptions build_options = options;
        if (defer_depth_completeness_gate)
        {
            build_options.enforceDepthCompletenessGate = false;
        }
        if (request.progress)
        {
            request.progress(
                QStringLiteral(
                    "模型配置：深度帧=%1，模式=%2，TSDF=%3，目标面数=%4，插值=%5")
                    .arg(loaded.frames.size())
                    .arg(orbital_workspace
                             ? QStringLiteral("环拍目标")
                             : QStringLiteral("常规场景"))
                    .arg(options.resolution)
                    .arg(options.simplifyTargetFaces)
                    .arg(request.settings.value(QStringLiteral("interpolation"))
                             .toString(QStringLiteral("enabled"))),
                1);
        }
        DepthTsdfResult tsdf = DepthTsdfSurfaceBuilder::build(
            loaded.frames, build_options);
        mergePayload(DepthTsdfSurfaceBuilder::statisticsToJson(tsdf), &result.payload);
        result.payload[QStringLiteral(
            "base_depth_completeness_available")] =
            tsdf.statistics.depthCompletenessAvailable;
        result.payload[QStringLiteral(
            "base_depth_completeness_gate_passed")] =
            tsdf.statistics.depthCompletenessGatePassed;
        result.payload[QStringLiteral(
            "base_depth_completeness_aggregate_recall")] =
            tsdf.statistics.depthCompletenessAggregateRecall;
        result.payload[QStringLiteral(
            "base_depth_completeness_minimum_frame_recall")] =
            tsdf.statistics.depthCompletenessMinimumFrameRecall;
        result.payload[QStringLiteral(
            "base_depth_completeness_p10_frame_recall")] =
            tsdf.statistics.depthCompletenessP10FrameRecall;
        result.payload[QStringLiteral(
            "base_depth_completeness_median_frame_recall")] =
            tsdf.statistics.depthCompletenessMedianFrameRecall;
        result.payload[QStringLiteral("tsdf_resolution_x")] = tsdf.layout.cells[0];
        result.payload[QStringLiteral("tsdf_resolution_y")] = tsdf.layout.cells[1];
        result.payload[QStringLiteral("tsdf_resolution_z")] = tsdf.layout.cells[2];
        result.payload[QStringLiteral("tsdf_required_bytes")] =
            static_cast<double>(tsdf.layout.requiredBytes);
        if (!tsdf.ok)
        {
            result.errorMessage = tsdf.errorMessage;
            return result;
        }
        if (request.progress)
        {
            request.progress(
                QStringLiteral(
                    "基础 TSDF 完成：顶点=%1，面=%2，边界边=%3，"
                    "深度召回率(中位/P10)=%4/%5")
                    .arg(tsdf.mesh.vertexCount())
                    .arg(tsdf.mesh.faceCount())
                    .arg(tsdf.statistics.boundaryEdgeCountAfter)
                    .arg(tsdf.statistics.depthCompletenessMedianFrameRecall,
                         0,
                         'f',
                         4)
                    .arg(tsdf.statistics.depthCompletenessP10FrameRecall,
                         0,
                         'f',
                         4),
                tsdf_progress_end);
        }

        const bool orbital_visual_hull_completion_configured =
            request.settings.value(QStringLiteral(
                "tsdfOrbitalVisualHullCompletion")).toBool(true);
        const bool observation_only_surface =
            interpolationIsDisabled(request.settings);
        const bool orbital_visual_hull_completion_enabled =
            orbital_visual_hull_completion_configured &&
            !observation_only_surface;
        const bool orbital_visual_hull_completion_requested =
            shouldUseOrbitalVisualHullCompletion(
                orbital_workspace,
                orbital_visual_hull_completion_configured,
                observation_only_surface,
                tsdf.statistics.depthCompletenessAggregateRecall,
                tsdf.statistics.boundaryEdgeCountAfter,
                tsdf.mesh.faceCount());
        result.payload[QStringLiteral(
            "requested_orbital_visual_hull_completion")] =
            orbital_visual_hull_completion_configured;
        result.payload[QStringLiteral(
            "configured_orbital_visual_hull_completion")] =
            orbital_visual_hull_completion_enabled;
        result.payload[QStringLiteral(
            "orbital_visual_hull_completion_suppressed_by_disabled_interpolation")] =
            orbital_visual_hull_completion_configured &&
            observation_only_surface;
        result.payload[QStringLiteral(
            "orbital_visual_hull_completion_requested")] =
            orbital_visual_hull_completion_requested;
        if (request.progress)
        {
            request.progress(
                observation_only_surface &&
                        orbital_visual_hull_completion_configured
                    ? QStringLiteral(
                          "插值已禁用，跳过环拍视觉外壳补全，仅输出深度观测支持的表面")
                    : orbital_visual_hull_completion_requested
                    ? QStringLiteral(
                          "基础表面不完整，正在启动环拍视觉外壳补全...")
                    : QStringLiteral(
                          "基础表面完整性满足补全策略，跳过环拍外壳补全"),
                tsdf_progress_end);
        }

        DepthMapVisualHullResult orbital_completion;
        TriMesh *output_mesh = &tsdf.mesh;
        QString output_algorithm = QStringLiteral("depth_tsdf");
        if (orbital_visual_hull_completion_requested)
        {
            DepthMapVisualHullOptions completion_options;
            completion_options.strictVolumetricMasks = false;
            completion_options.useContinuousSilhouetteField =
                request.settings.value(QStringLiteral(
                    "tsdfOrbitalVisualHullContinuousSilhouetteField"))
                    .toBool(false);
            completion_options.smoothingIterations = qBound(
                0,
                request.settings.value(QStringLiteral(
                    "tsdfOrbitalVisualHullSmoothingIterations")).toInt(12),
                20);
            completion_options.smoothingLambda = std::clamp(
                static_cast<float>(request.settings.value(QStringLiteral(
                    "tsdfOrbitalVisualHullSmoothingLambda")).toDouble(0.18)),
                0.0f,
                0.49f);
            completion_options.topologyClosingIterations = qBound(
                -1,
                request.settings.value(QStringLiteral(
                    "tsdfOrbitalVisualHullTopologyClosingIterations"))
                    .toInt(-1),
                3);
            std::function<void(const QString &, int)> completion_progress;
            if (request.progress)
            {
                completion_progress = [progress = request.progress](
                                          const QString &stage, int percent)
                {
                    progress(
                        stage,
                        96 + std::clamp(percent, 0, 100) / 100);
                };
            }
            orbital_completion = DepthMapMeshBuilder::buildVisualHull(
                request.depthMapSourcePath,
                request.reconstruction.resolution,
                completion_options,
                completion_progress);
            result.payload[QStringLiteral(
                "orbital_visual_hull_completion_views")] =
                orbital_completion.usableViewCount;
            result.payload[QStringLiteral(
                "orbital_visual_hull_preserved_silhouette_hole_views")] =
                orbital_completion.preservedSilhouetteHoleViewCount;
            result.payload[QStringLiteral(
                "orbital_visual_hull_topology_closing_iterations")] =
                orbital_completion.topologyClosingIterations;
            result.payload[QStringLiteral(
                "orbital_visual_hull_continuous_silhouette_field")] =
                completion_options.useContinuousSilhouetteField;
            result.payload[QStringLiteral(
                "orbital_visual_hull_completion_component_count")] =
                orbital_completion.connectivity.componentCount;
            result.payload[QStringLiteral(
                "orbital_visual_hull_completion_largest_component_ratio")] =
                orbital_completion.connectivity.largestComponentFaceRatio;
            if (orbital_completion.applicable && orbital_completion.ok &&
                orbital_completion.connectivity.componentCount == 1)
            {
                const bool refine_before_simplification =
                    request.settings.value(QStringLiteral(
                        "tsdfOrbitalVisualHullRefineBeforeSimplification"))
                        .toBool(false);
                result.payload[QStringLiteral(
                    "configured_orbital_visual_hull_refine_before_simplification")] =
                    refine_before_simplification;
                const auto simplify_orbital_visual_hull = [&]()
                {
                    if (options.simplifyTargetFaces <= 0 ||
                        orbital_completion.mesh.faceCount() <=
                            options.simplifyTargetFaces)
                    {
                        return;
                    }
                    const TriMesh unsimplified_mesh = orbital_completion.mesh;
                    const MeshTopologyQualityStatistics quality_before =
                        evaluateMeshTopologyQuality(unsimplified_mesh);
                    const auto preserves_topology =
                        [&quality_before](
                            const MeshTopologyQualityStatistics &quality_after)
                    {
                        const int boundary_edge_limit = std::max(
                            quality_before.boundaryEdgeCount + 128,
                            static_cast<int>(std::ceil(
                                quality_before.boundaryEdgeCount * 1.5)));
                        const int complexity_limit = std::max(
                            quality_before.topologicalComplexity + 32,
                            static_cast<int>(std::ceil(
                                quality_before.topologicalComplexity * 1.5)));
                        return quality_after.componentCount <=
                                   std::max(1, quality_before.componentCount) &&
                               quality_after.largestComponentFaceRatio + 0.005 >=
                                   quality_before.largestComponentFaceRatio &&
                               quality_after.nonManifoldEdgeCount <=
                                   quality_before.nonManifoldEdgeCount &&
                               quality_after.boundaryEdgeCount <=
                                   boundary_edge_limit &&
                               quality_after.topologicalComplexity <=
                                   complexity_limit;
                    };
                    OpenMeshSimplifyOptions simplify_options;
                    simplify_options.targetFaceCount =
                        options.simplifyTargetFaces;
                    simplify_options.maximumNormalDeviationDegrees = 35.0f;
                    simplify_options.maximumNormalFlippingDegrees = 75.0f;
                    simplify_options.notificationInterval = 4096;
                    simplify_options.isCancelled = request.isCancelled;
                    const bool prefer_quadric_simplification =
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullPreferQuadricSimplification"))
                            .toBool(false);
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_prefer_quadric_simplification")] =
                        prefer_quadric_simplification;
                    OpenMeshSimplifyStatistics simplify;
                    if (!prefer_quadric_simplification)
                    {
                        simplify = simplifyMeshWithOpenMesh(
                            &orbital_completion.mesh,
                            simplify_options);
                    }
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_available")] =
                        simplify.available;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_succeeded")] =
                        simplify.succeeded;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_reached_target")] =
                        simplify.reachedTarget;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_input_faces")] =
                        simplify.inputFaceCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_output_faces")] =
                        simplify.outputFaceCount;
                    const MeshTopologyQualityStatistics quality_after =
                        evaluateMeshTopologyQuality(
                            orbital_completion.mesh);
                    const bool topology_preserved =
                        !prefer_quadric_simplification &&
                        simplify.succeeded &&
                        preserves_topology(quality_after);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_topology_preserved")] =
                        topology_preserved;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_component_count_before")] =
                        quality_before.componentCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_component_count_after")] =
                        quality_after.componentCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_boundary_edges_before")] =
                        quality_before.boundaryEdgeCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_boundary_edges_after")] =
                        quality_after.boundaryEdgeCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_complexity_before")] =
                        quality_before.topologicalComplexity;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_complexity_after")] =
                        quality_after.topologicalComplexity;
                    if (!topology_preserved)
                    {
                        orbital_completion.mesh = unsimplified_mesh;
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_simplification_rejected")] =
                            !prefer_quadric_simplification;
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_openmesh_skipped_for_quadric")] =
                            prefer_quadric_simplification;
                        QuadricSimplifyOptions quadric_options;
                        quadric_options.targetFaceCount =
                            options.simplifyTargetFaces;
                        quadric_options.maximumPasses = 12;
                        quadric_options.workerCount = options.workerCount;
                        quadric_options.featureAngleDegrees = 35.0f;
                        quadric_options.maximumNormalDeviationDegrees = 45.0f;
                        quadric_options.preserveOpenBoundaries = true;
                        quadric_options.simplifySimpleOpenBoundaries = false;
                        quadric_options.isCancelled = request.isCancelled;
                        if (request.progress)
                        {
                            quadric_options.progress =
                                [progress = request.progress,
                                 prefer_quadric_simplification](
                                    int,
                                    int current_faces)
                                {
                                    progress(
                                        (prefer_quadric_simplification
                                             ? QStringLiteral(
                                                   "正在进行保拓扑 QEM 简化（约 %1 面）...")
                                             : QStringLiteral(
                                                   "OpenMesh 破坏拓扑，正在进行保拓扑 QEM 简化（约 %1 面）..."))
                                            .arg(current_faces),
                                        98);
                                };
                        }
                        TriMesh quadric_candidate = unsimplified_mesh;
                        const QuadricSimplifyStatistics quadric =
                            simplifyMeshQuadric(
                                &quadric_candidate,
                                quadric_options);
                        const MeshTopologyQualityStatistics quadric_quality =
                            evaluateMeshTopologyQuality(
                                quadric_candidate);
                        const bool quadric_accepted =
                            quadric_candidate.faceCount() <
                                unsimplified_mesh.faceCount() &&
                            preserves_topology(quadric_quality);
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_quadric_fallback_attempted")] =
                            true;
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_quadric_fallback_accepted")] =
                            quadric_accepted;
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_quadric_fallback_output_faces")] =
                            quadric_candidate.faceCount();
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_quadric_fallback_collapsed_edges")] =
                            quadric.collapsedEdgeCount;
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_quadric_fallback_reached_target")] =
                            quadric.reachedTarget;
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_quadric_fallback_component_count")] =
                            quadric_quality.componentCount;
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_quadric_fallback_boundary_edges")] =
                            quadric_quality.boundaryEdgeCount;
                        result.payload[QStringLiteral(
                            "orbital_visual_hull_quadric_fallback_complexity")] =
                            quadric_quality.topologicalComplexity;
                        if (quadric_accepted)
                        {
                            orbital_completion.mesh =
                                std::move(quadric_candidate);
                        }
                    }
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_simplification_effective_faces")] =
                        orbital_completion.mesh.faceCount();
                };
                if (!refine_before_simplification)
                {
                    simplify_orbital_visual_hull();
                }
                const bool depth_refinement_enabled =
                    request.settings.value(QStringLiteral(
                        "tsdfOrbitalVisualHullDepthRefinement"))
                        .toBool(true);
                result.payload[QStringLiteral(
                    "configured_orbital_visual_hull_depth_refinement")] =
                    depth_refinement_enabled;
                if (depth_refinement_enabled)
                {
                    const TriMesh refinement_input_mesh =
                        orbital_completion.mesh;
                    const float maximum_voxel_size =
                        *std::max_element(
                            tsdf.layout.voxelSize.cbegin(),
                            tsdf.layout.voxelSize.cend());
                    VisualHullDepthRefineOptions refine_options;
                    refine_options.maximumEvidenceDistance =
                        maximum_voxel_size * std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthMaximumEvidenceVoxels"))
                                                   .toDouble(10.0)),
                            1.0f,
                            24.0f);
                    refine_options.maximumDisplacement =
                        maximum_voxel_size * std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthMaximumDisplacementVoxels"))
                                                   .toDouble(3.5)),
                            0.25f,
                            8.0f);
                    refine_options.minimumViewCount = qBound(
                        2,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthMinimumViews"))
                            .toInt(2),
                        8);
                    refine_options.minimumNativeViewCount = qBound(
                        0,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthMinimumNativeViews"))
                            .toInt(1),
                        refine_options.minimumViewCount);
                    refine_options.minimumDepthConfidence = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullDepthMinimumConfidence"))
                                               .toDouble(0.25)),
                        0.0f,
                        1.0f);
                    refine_options.repairedObservationWeight = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullDepthRepairedWeight"))
                                               .toDouble(0.35)),
                        0.0f,
                        1.0f);
                    refine_options.enableInverseDepthSpreadWeighting =
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthSpreadWeighting"))
                            .toBool(true);
                    refine_options.inverseDepthSpreadWeightKnee = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullDepthSpreadWeightKnee"))
                                               .toDouble(0.005)),
                        0.0f,
                        0.099f);
                    refine_options.inverseDepthSpreadWeightZero = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullDepthSpreadWeightZero"))
                                               .toDouble(0.015)),
                        refine_options.inverseDepthSpreadWeightKnee +
                            1.0e-6f,
                        0.10f);
                    refine_options.minimumInverseDepthSpreadWeight =
                        std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthMinimumSpreadWeight"))
                                                   .toDouble(0.05)),
                            0.0f,
                            1.0f);
                    refine_options
                        .maximumViewMedianAbsoluteDeviation =
                        maximum_voxel_size * std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthMaximumMadVoxels"))
                                                   .toDouble(2.5)),
                            0.25f,
                            8.0f);
                    refine_options.enableCrossViewBiasCompensation =
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthBiasCompensation"))
                            .toBool(true);
                    refine_options.minimumCrossViewBiasPairSamples =
                        qBound(
                            8,
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullDepthBiasMinimumPairSamples"))
                                .toInt(64),
                            4096);
                    refine_options.maximumCrossViewBias =
                        maximum_voxel_size * std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthMaximumBiasVoxels"))
                                                   .toDouble(2.0)),
                            0.25f,
                            6.0f);
                    refine_options.minimumAnchorWeight = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullDepthMinimumAnchorWeight"))
                                               .toDouble(0.05)),
                        0.0f,
                        1.0f);
                    refine_options.regularizationIterations = qBound(
                        0,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthRegularizationIterations"))
                            .toInt(30),
                        100);
                    refine_options.regularizationWeight = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullDepthRegularizationWeight"))
                                               .toDouble(10.0)),
                        0.0f,
                        20.0f);
                    refine_options.propagationDecay = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullDepthPropagationDecay"))
                                               .toDouble(0.90)),
                        0.0f,
                        1.0f);
                    refine_options
                        .regularizationMaximumNormalAngleDegrees =
                        std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthRegularizationMaximumNormalAngleDegrees"))
                                                   .toDouble(50.0)),
                            5.0f,
                            89.0f);
                    refine_options.enableGlobalRobustSolver =
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthGlobalRobustSolver"))
                            .toBool(false);
                    refine_options.globalSolverIrlsIterations = qBound(
                        1,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthGlobalSolverIrlsIterations"))
                            .toInt(4),
                        10);
                    refine_options.globalSolverMaximumPcgIterations =
                        qBound(
                            1,
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullDepthGlobalSolverMaximumPcgIterations"))
                                .toInt(120),
                            500);
                    refine_options
                        .globalSolverConvergenceTolerance =
                        std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthGlobalSolverConvergenceTolerance"))
                                                   .toDouble(1.0e-5)),
                            1.0e-9f,
                            1.0e-2f);
                    refine_options.globalSolverRobustScaleMultiplier =
                        std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthGlobalSolverRobustScaleMultiplier"))
                                                   .toDouble(0.5)),
                            0.01f,
                            4.0f);
                    refine_options.globalSolverLaplacianWeight =
                        std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthGlobalSolverLaplacianWeight"))
                                                   .toDouble(0.45)),
                            0.0f,
                            20.0f);
                    refine_options.globalSolverHullPriorWeight =
                        std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthGlobalSolverHullPriorWeight"))
                                                   .toDouble(0.02)),
                            1.0e-6f,
                            1.0f);
                    if (request.progress)
                    {
                        request.progress(
                            QStringLiteral(
                                "正在用多视深度细化闭合表面..."),
                            99);
                    }
                    const int refinement_pass_count = qBound(
                        1,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthRefinementPasses"))
                            .toInt(2),
                        4);
                    VisualHullDepthRefineStatistics refine;
                    int applied_refinement_pass_count = 0;
                    for (int pass = 0;
                         pass < refinement_pass_count;
                         ++pass)
                    {
                        if (request.isCancelled &&
                            request.isCancelled())
                        {
                            break;
                        }
                        const VisualHullDepthRefineStatistics pass_refine =
                            VisualHullDepthRefiner::refine(
                                &orbital_completion.mesh,
                                loaded.frames,
                                refine_options,
                                request.isCancelled);
                        refine.projectedObservationCount +=
                            pass_refine.projectedObservationCount;
                        refine.acceptedObservationCount +=
                            pass_refine.acceptedObservationCount;
                        refine.spreadDownweightedObservationCount +=
                            pass_refine.spreadDownweightedObservationCount;
                        refine.spreadVeryWeakObservationCount +=
                            pass_refine.spreadVeryWeakObservationCount;
                        refine.anchoredVertexCount +=
                            pass_refine.anchoredVertexCount;
                        refine.blendedConsensusVertexCount +=
                            pass_refine.blendedConsensusVertexCount;
                        refine.biasCalibratedFrameCount = std::max(
                            refine.biasCalibratedFrameCount,
                            pass_refine.biasCalibratedFrameCount);
                        refine.biasCalibrationPairCount = std::max(
                            refine.biasCalibrationPairCount,
                            pass_refine.biasCalibrationPairCount);
                        refine.maximumAbsoluteFrameBias = std::max(
                            refine.maximumAbsoluteFrameBias,
                            pass_refine.maximumAbsoluteFrameBias);
                        refine.displacedVertexCount +=
                            pass_refine.displacedVertexCount;
                        refine.medianSupportingViewCount =
                            pass_refine.medianSupportingViewCount;
                        refine.p90SupportingViewCount =
                            pass_refine.p90SupportingViewCount;
                        refine.maximumAppliedDisplacement = std::max(
                            refine.maximumAppliedDisplacement,
                            pass_refine.maximumAppliedDisplacement);
                        refine.medianAppliedDisplacement =
                            pass_refine.medianAppliedDisplacement;
                        refine.p90AppliedDisplacement =
                            pass_refine.p90AppliedDisplacement;
                        const bool first_global_solver_pass =
                            refine.globalSolverAttemptCount == 0 &&
                            pass_refine.globalSolverAttemptCount > 0;
                        refine.globalSolverAttemptCount +=
                            pass_refine.globalSolverAttemptCount;
                        refine.globalSolverSolvedPassCount +=
                            pass_refine.globalSolverSolvedPassCount;
                        refine.globalSolverAppliedPassCount +=
                            pass_refine.globalSolverAppliedPassCount;
                        refine.globalSolverConvergedPassCount +=
                            pass_refine.globalSolverConvergedPassCount;
                        refine.globalSolverFallbackPassCount +=
                            pass_refine.globalSolverFallbackPassCount;
                        refine.globalSolverCancelled =
                            refine.globalSolverCancelled ||
                            pass_refine.globalSolverCancelled;
                        refine.globalSolverIrlsIterationCount +=
                            pass_refine.globalSolverIrlsIterationCount;
                        refine.globalSolverPcgIterationCount +=
                            pass_refine.globalSolverPcgIterationCount;
                        refine.globalSolverObservationCount +=
                            pass_refine.globalSolverObservationCount;
                        refine.globalSolverRegularizationEdgeCount =
                            pass_refine
                                .globalSolverRegularizationEdgeCount;
                        refine.globalSolverAnchoredVertexCount =
                            pass_refine.globalSolverAnchoredVertexCount;
                        refine.globalSolverPriorOnlyVertexCount =
                            pass_refine.globalSolverPriorOnlyVertexCount;
                        refine.globalSolverEffectiveRobustScale =
                            pass_refine
                                .globalSolverEffectiveRobustScale;
                        if (first_global_solver_pass)
                        {
                            refine.globalSolverInitialEnergy =
                                pass_refine.globalSolverInitialEnergy;
                        }
                        if (pass_refine.globalSolverAttemptCount > 0)
                        {
                            refine.globalSolverFinalEnergy =
                                pass_refine.globalSolverFinalEnergy;
                            refine.globalSolverFinalRelativeResidual =
                                pass_refine
                                    .globalSolverFinalRelativeResidual;
                        }
                        if (pass_refine.globalSolverCancelled)
                        {
                            break;
                        }
                        if (!pass_refine.applied)
                        {
                            break;
                        }
                        refine.applied = true;
                        ++applied_refinement_pass_count;
                    }
                    const bool refinement_quality_guard_enabled =
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullDepthRefinementQualityGuard"))
                            .toBool(true);
                    RefinementQualityGuardResult refinement_guard;
                    if (refine.applied &&
                        refinement_quality_guard_enabled)
                    {
                        refinement_guard = limitRefinementRoughness(
                            refinement_input_mesh,
                            &orbital_completion.mesh,
                            std::clamp(
                                request.settings.value(QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthMaximumAreaGrowth"))
                                    .toDouble(1.03),
                                1.0,
                                1.25),
                            std::clamp(
                                request.settings.value(QStringLiteral(
                                    "tsdfOrbitalVisualHullDepthMaximumNormalVariationGrowth"))
                                    .toDouble(3.5),
                                1.0,
                                5.0));
                    }
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_refinement_quality_guard")] =
                        refinement_quality_guard_enabled;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_quality_guard_applied")] =
                        refinement_guard.applied;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_quality_guard_limited")] =
                        refinement_guard.limited;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_accepted_blend")] =
                        refinement_guard.acceptedBlend;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_area_before")] =
                        refinement_guard.areaBefore;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_area_after")] =
                        refinement_guard.areaAfter;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_normal_variation_before")] =
                        refinement_guard.normalVariationBefore;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_normal_variation_after")] =
                        refinement_guard.normalVariationAfter;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_applied")] =
                        refine.applied;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_passes")] =
                        applied_refinement_pass_count;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_projected_observations")] =
                        static_cast<double>(
                            refine.projectedObservationCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_accepted_observations")] =
                        static_cast<double>(
                            refine.acceptedObservationCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_spread_downweighted_observations")] =
                        static_cast<double>(
                            refine.spreadDownweightedObservationCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_spread_very_weak_observations")] =
                        static_cast<double>(
                            refine.spreadVeryWeakObservationCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_anchored_vertices")] =
                        static_cast<double>(refine.anchoredVertexCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_blended_consensus_vertices")] =
                        static_cast<double>(
                            refine.blendedConsensusVertexCount);
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_bias_compensation")] =
                        refine_options.enableCrossViewBiasCompensation;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_bias_calibrated_frames")] =
                        refine.biasCalibratedFrameCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_bias_calibration_pairs")] =
                        static_cast<double>(
                            refine.biasCalibrationPairCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_bias_maximum_absolute_offset")] =
                        refine.maximumAbsoluteFrameBias;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_displaced_vertices")] =
                        static_cast<double>(refine.displacedVertexCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_median_views")] =
                        refine.medianSupportingViewCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_p90_views")] =
                        refine.p90SupportingViewCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_maximum_displacement")] =
                        refine.maximumAppliedDisplacement;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_median_displacement")] =
                        refine.medianAppliedDisplacement;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_p90_displacement")] =
                        refine.p90AppliedDisplacement;
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_global_robust_solver")] =
                        refine_options.enableGlobalRobustSolver;
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_global_solver_irls_iterations")] =
                        refine_options.globalSolverIrlsIterations;
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_global_solver_maximum_pcg_iterations")] =
                        refine_options.globalSolverMaximumPcgIterations;
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_global_solver_convergence_tolerance")] =
                        refine_options.globalSolverConvergenceTolerance;
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_global_solver_robust_scale_multiplier")] =
                        refine_options.globalSolverRobustScaleMultiplier;
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_global_solver_laplacian_weight")] =
                        refine_options.globalSolverLaplacianWeight;
                    result.payload[QStringLiteral(
                        "configured_orbital_visual_hull_depth_global_solver_hull_prior_weight")] =
                        refine_options.globalSolverHullPriorWeight;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_effective_robust_scale")] =
                        refine.globalSolverEffectiveRobustScale;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_attempts")] =
                        refine.globalSolverAttemptCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_solved_passes")] =
                        refine.globalSolverSolvedPassCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_applied_passes")] =
                        refine.globalSolverAppliedPassCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_converged_passes")] =
                        refine.globalSolverConvergedPassCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_fallback_passes")] =
                        refine.globalSolverFallbackPassCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_cancelled")] =
                        refine.globalSolverCancelled;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_irls_iterations")] =
                        refine.globalSolverIrlsIterationCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_pcg_iterations")] =
                        refine.globalSolverPcgIterationCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_observations")] =
                        static_cast<double>(
                            refine.globalSolverObservationCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_regularization_edges")] =
                        static_cast<double>(
                            refine.globalSolverRegularizationEdgeCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_anchored_vertices")] =
                        static_cast<double>(
                            refine.globalSolverAnchoredVertexCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_prior_only_vertices")] =
                        static_cast<double>(
                            refine.globalSolverPriorOnlyVertexCount);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_initial_energy")] =
                        refine.globalSolverInitialEnergy;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_final_energy")] =
                        refine.globalSolverFinalEnergy;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_global_solver_final_relative_residual")] =
                        refine.globalSolverFinalRelativeResidual;
                }
                else
                {
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_depth_refinement_applied")] =
                        false;
                }
                if (refine_before_simplification)
                {
                    simplify_orbital_visual_hull();
                }
                const bool post_smoothing_enabled =
                    request.settings.value(QStringLiteral(
                        "tsdfOrbitalVisualHullPostSmoothing"))
                        .toBool(false);
                result.payload[QStringLiteral(
                    "configured_orbital_visual_hull_post_smoothing")] =
                    post_smoothing_enabled;
                if (post_smoothing_enabled)
                {
                    const float maximum_voxel_size =
                        *std::max_element(
                            tsdf.layout.voxelSize.cbegin(),
                            tsdf.layout.voxelSize.cend());
                    const int smoothing_iterations = qBound(
                        1,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullPostSmoothingIterations"))
                            .toInt(3),
                        12);
                    const float smoothing_lambda = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullPostSmoothingLambda"))
                                               .toDouble(0.20)),
                        0.01f,
                        0.75f);
                    const float maximum_displacement =
                        maximum_voxel_size * std::clamp(
                            static_cast<float>(request.settings.value(
                                QStringLiteral(
                                    "tsdfOrbitalVisualHullPostSmoothingMaximumDisplacementVoxels"))
                                                   .toDouble(0.75)),
                            0.05f,
                            3.0f);
                    const float maximum_normal_angle = std::clamp(
                        static_cast<float>(request.settings.value(
                            QStringLiteral(
                                "tsdfOrbitalVisualHullPostSmoothingMaximumNormalAngleDegrees"))
                                               .toDouble(45.0)),
                        5.0f,
                        85.0f);
                    const int boundary_protection_rings = qBound(
                        0,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullPostSmoothingBoundaryProtectionRings"))
                            .toInt(1),
                        4);
                    const int moved_vertex_count =
                        detail::smoothSurfaceVerticesNormalAware(
                            &orbital_completion.mesh,
                            smoothing_iterations,
                            smoothing_lambda,
                            maximum_displacement,
                            maximum_normal_angle,
                            boundary_protection_rings);
                    detail::recomputeNormals(&orbital_completion.mesh);
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_smoothing_moved_vertices")] =
                        moved_vertex_count;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_smoothing_iterations")] =
                        smoothing_iterations;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_smoothing_maximum_displacement")] =
                        maximum_displacement;
                }
                const bool post_refinement_remeshing_enabled =
                    request.settings.value(QStringLiteral(
                        "tsdfOrbitalVisualHullPostRefinementRemeshing"))
                        .toBool(true);
                result.payload[QStringLiteral(
                    "configured_orbital_visual_hull_post_refinement_remeshing")] =
                    post_refinement_remeshing_enabled;
                if (post_refinement_remeshing_enabled)
                {
                    const TriMesh remeshing_input =
                        orbital_completion.mesh;
                    const MeshTopologyQualityStatistics topology_before =
                        evaluateMeshTopologyQuality(remeshing_input);
                    MeshIsotropicRemeshOptions remesh_options;
                    remesh_options.maximumPasses = qBound(
                        1,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullPostRefinementRemeshingPasses"))
                            .toInt(2),
                        4);
                    remesh_options.minimumAffectedAspectRatio =
                        std::clamp(
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullPostRefinementRemeshingMinimumAspect"))
                                .toDouble(8.0),
                            5.0,
                            30.0);
                    remesh_options.maximumFeatureAngleDegrees =
                        std::clamp(
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullPostRefinementRemeshingFeatureAngleDegrees"))
                                .toDouble(50.0),
                            20.0,
                            80.0);
                    remesh_options.maximumNormalDeviationDegrees = 35.0;
                    remesh_options.maximumFaceGrowthRatio = 0.05;
                    remesh_options.isCancelled = request.isCancelled;
                    const MeshIsotropicRemeshStatistics remesh =
                        remeshInteriorHighAspectTriangles(
                            &orbital_completion.mesh,
                            remesh_options);
                    const MeshTopologyQualityStatistics topology_after =
                        evaluateMeshTopologyQuality(
                            orbital_completion.mesh);
                    const bool remesh_topology_preserved =
                        topology_after.componentCount ==
                            topology_before.componentCount &&
                        topology_after.nonManifoldEdgeCount <=
                            topology_before.nonManifoldEdgeCount &&
                        topology_after.boundaryEdgeCount <=
                            topology_before.boundaryEdgeCount &&
                        topology_after.topologicalComplexity <=
                            topology_before.topologicalComplexity;
                    if (!remesh_topology_preserved)
                    {
                        orbital_completion.mesh = remeshing_input;
                    }
                    else
                    {
                        detail::recomputeNormals(
                            &orbital_completion.mesh);
                    }
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_remeshing_topology_preserved")] =
                        remesh_topology_preserved;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_remeshing_passes")] =
                        remesh.passCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_remeshing_collapsed_edges")] =
                        remesh.collapsedEdgeCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_remeshing_split_edges")] =
                        remesh.splitEdgeCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_remeshing_faces_before")] =
                        remeshing_input.faceCount();
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_remeshing_faces_after")] =
                        orbital_completion.mesh.faceCount();
                }
                const bool post_refinement_edge_flips_enabled =
                    request.settings.value(QStringLiteral(
                        "tsdfOrbitalVisualHullPostRefinementEdgeFlips"))
                        .toBool(true);
                result.payload[QStringLiteral(
                    "configured_orbital_visual_hull_post_refinement_edge_flips")] =
                    post_refinement_edge_flips_enabled;
                if (post_refinement_edge_flips_enabled)
                {
                    const TriMesh optimization_input =
                        orbital_completion.mesh;
                    const MeshTopologyQualityStatistics topology_before =
                        evaluateMeshTopologyQuality(optimization_input);
                    MeshTriangleOptimizationOptions optimization_options;
                    optimization_options.maximumPasses = qBound(
                        1,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullPostRefinementEdgeFlipPasses"))
                            .toInt(4),
                        8);
                    optimization_options.minimumWorstAspectImprovementRatio =
                        std::clamp(
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullPostRefinementEdgeFlipMinimumImprovement"))
                                .toDouble(0.01),
                            0.0,
                            0.25);
                    optimization_options.maximumFeatureAngleDegrees =
                        std::clamp(
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullPostRefinementEdgeFlipFeatureAngleDegrees"))
                                .toDouble(50.0),
                            20.0,
                            80.0);
                    optimization_options.maximumNormalDeviationDegrees =
                        std::clamp(
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullPostRefinementEdgeFlipNormalDeviationDegrees"))
                                .toDouble(30.0),
                            10.0,
                            60.0);
                    optimization_options.enableTangentialRelaxation =
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullPostRefinementTangentialRelaxation"))
                            .toBool(true);
                    optimization_options.tangentialRelaxationPasses = qBound(
                        0,
                        request.settings.value(QStringLiteral(
                            "tsdfOrbitalVisualHullPostRefinementTangentialRelaxationPasses"))
                            .toInt(2),
                        6);
                    optimization_options.tangentialRelaxationLambda =
                        std::clamp(
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullPostRefinementTangentialRelaxationLambda"))
                                .toDouble(0.35),
                            0.0,
                            1.0);
                    optimization_options.tangentialMaximumDisplacementEdgeRatio =
                        std::clamp(
                            request.settings.value(QStringLiteral(
                                "tsdfOrbitalVisualHullPostRefinementTangentialMaximumDisplacementEdgeRatio"))
                                .toDouble(0.15),
                            0.0,
                            0.35);
                    optimization_options.enableIsotropicRemeshing = false;
                    optimization_options.isCancelled = request.isCancelled;
                    const MeshTriangleOptimizationStatistics optimization =
                        optimizeTriangleQuality(
                            &orbital_completion.mesh,
                            optimization_options);
                    const MeshTopologyQualityStatistics topology_after =
                        evaluateMeshTopologyQuality(
                            orbital_completion.mesh);
                    const bool topology_preserved =
                        topology_after.componentCount ==
                            topology_before.componentCount &&
                        topology_after.nonManifoldEdgeCount <=
                            topology_before.nonManifoldEdgeCount &&
                        topology_after.boundaryEdgeCount <=
                            topology_before.boundaryEdgeCount &&
                        topology_after.topologicalComplexity <=
                            topology_before.topologicalComplexity;
                    if (!topology_preserved)
                    {
                        orbital_completion.mesh = optimization_input;
                    }
                    else
                    {
                        detail::recomputeNormals(
                            &orbital_completion.mesh);
                    }
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_edge_flips_topology_preserved")] =
                        topology_preserved;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_edge_flip_passes")] =
                        optimization.passCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_flipped_edges")] =
                        optimization.flippedEdgeCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_tangential_relaxed_vertices")] =
                        optimization.tangentialRelaxedVertexCount;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_high_aspect_ratio_before")] =
                        topology_before.highAspectFaceRatio;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_high_aspect_ratio_after")] =
                        topology_after.highAspectFaceRatio;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_extreme_aspect_ratio_before")] =
                        topology_before.extremeAspectFaceRatio;
                    result.payload[QStringLiteral(
                        "orbital_visual_hull_post_refinement_extreme_aspect_ratio_after")] =
                        topology_after.extremeAspectFaceRatio;
                }
                const QString products_path =
                    QDir(output_root).filePath(QStringLiteral("products"));
                QDir().mkpath(products_path);
                const QString tsdf_candidate_path =
                    QDir(products_path).filePath(
                        QStringLiteral("depth_tsdf_detail_candidate.ply"));
                std::string candidate_error;
                if (tsdf.mesh.savePLY(
                        xjw::common::io::toUtf8Path(tsdf_candidate_path),
                        &candidate_error))
                {
                    result.payload[QStringLiteral(
                        "depth_tsdf_detail_candidate_path")] =
                        tsdf_candidate_path;
                }
                else
                {
                    result.payload[QStringLiteral(
                        "depth_tsdf_detail_candidate_error")] =
                        QString::fromUtf8(candidate_error);
                }
                output_mesh = &orbital_completion.mesh;
                output_algorithm =
                    QStringLiteral("orbital_visual_hull_completion");
                result.payload[QStringLiteral(
                    "orbital_visual_hull_completion_applied")] = true;
                result.payload[QStringLiteral("actual_mesh_algorithm")] =
                    output_algorithm;
            }
            else
            {
                result.payload[QStringLiteral(
                    "orbital_visual_hull_completion_applied")] = false;
                result.payload[QStringLiteral(
                    "orbital_visual_hull_completion_error")] =
                    orbital_completion.message;
                if (request.progress)
                {
                    request.progress(
                        QStringLiteral(
                            "环拍外壳补全未采用：%1；继续使用基础 TSDF 表面")
                            .arg(orbital_completion.message),
                        97);
                }
            }
        }
        else
        {
            result.payload[QStringLiteral(
                "orbital_visual_hull_completion_applied")] = false;
        }

        const bool direct_visibility_occupancy_output =
            output_mesh == &tsdf.mesh &&
            tsdf.statistics
                .effectiveVisibilityOccupancyCellBoundaryExtraction;
        const int carrier_subdivision_passes =
            direct_visibility_occupancy_output
            ? qBound(
                  0,
                  request.settings.value(QStringLiteral(
                      "tsdfVisibilityOccupancyCarrierSubdivisionPasses"))
                      .toInt(orbital_workspace ? 1 : 0),
                  1)
            : 0;
        result.payload[QStringLiteral(
            "configured_visibility_occupancy_carrier_subdivision_passes")] =
            carrier_subdivision_passes;
        if (carrier_subdivision_passes > 0)
        {
            if (request.progress)
            {
                request.progress(
                    QStringLiteral("正在细分闭合拓扑载体..."),
                    96);
            }
            const MeshTopologySignature topology_before_subdivision =
                evaluateMeshTopologySignature(tsdf.mesh);
            VisibilityOccupancyCarrierSubdivisionResult subdivision =
                VisibilityOccupancyCarrierSubdivider::subdivide(
                    tsdf.mesh,
                    {request.isCancelled});
            if (!subdivision.ok)
            {
                result.errorMessage = subdivision.cancelled
                    ? QStringLiteral("模型载体细分已取消。")
                    : QStringLiteral("模型载体细分失败：%1")
                          .arg(QString::fromStdString(
                              subdivision.errorMessage));
                return result;
            }
            const MeshTopologySignature topology_after_subdivision =
                evaluateMeshTopologySignature(subdivision.mesh);
            if (!topology_before_subdivision.closedTwoManifold ||
                topology_after_subdivision != topology_before_subdivision)
            {
                result.errorMessage = QStringLiteral(
                    "模型载体细分未保持闭合二流形拓扑，已停止写入模型。");
                return result;
            }
            tsdf.mesh = std::move(subdivision.mesh);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_subdivision_applied")] = true;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_subdivision_input_vertices")] =
                static_cast<double>(
                    subdivision.statistics.inputVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_subdivision_input_faces")] =
                static_cast<double>(subdivision.statistics.inputFaceCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_subdivision_midpoint_vertices")] =
                static_cast<double>(
                    subdivision.statistics.createdMidpointVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_subdivision_output_vertices")] =
                static_cast<double>(
                    subdivision.statistics.outputVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_subdivision_output_faces")] =
                static_cast<double>(subdivision.statistics.outputFaceCount);
        }
        else
        {
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_subdivision_applied")] = false;
        }

        const int carrier_fairing_passes =
            direct_visibility_occupancy_output
            ? qBound(
                  0,
                  request.settings.value(QStringLiteral(
                      "tsdfVisibilityOccupancyCarrierFairingPasses"))
                      .toInt(orbital_workspace ? 8 : 0),
                  12)
            : 0;
        result.payload[QStringLiteral(
            "configured_visibility_occupancy_carrier_fairing_passes")] =
            carrier_fairing_passes;
        if (carrier_fairing_passes > 0)
        {
            if (request.progress)
            {
                request.progress(
                    QStringLiteral("正在保拓扑平滑闭合载体..."),
                    96);
            }
            const float occupancy_step =
                visibilityOccupancyMedianVoxelStep(
                    tsdf.layout,
                    options.visibilityOccupancyResolution);
            VisibilityOccupancyCarrierFairingOptions fairing_options;
            fairing_options.iterations = carrier_fairing_passes;
            fairing_options.lambda = std::clamp(
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyCarrierFairingLambda"))
                    .toDouble(0.20),
                0.01,
                0.60);
            fairing_options.mu = std::clamp(
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyCarrierFairingMu"))
                    .toDouble(-0.21),
                -0.80,
                -0.01);
            fairing_options.absoluteMaximumDisplacement =
                static_cast<double>(occupancy_step) * std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFairingMaximumDisplacementVoxels"))
                        .toDouble(0.35),
                    0.02,
                    1.0);
            fairing_options.minimumNormalDot = std::clamp(
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyCarrierFairingMinimumNormalDot"))
                    .toDouble(0.50),
                -1.0,
                1.0);
            fairing_options.minimumFaceAreaRatio = std::clamp(
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyCarrierFairingMinimumFaceAreaRatio"))
                    .toDouble(0.25),
                0.01,
                1.0);
            fairing_options.minimumSurfaceAreaRatio = std::clamp(
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyCarrierFairingMinimumAreaRatio"))
                    .toDouble(0.90),
                0.50,
                1.0);
            fairing_options.maximumSurfaceAreaRatio = std::clamp(
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyCarrierFairingMaximumAreaRatio"))
                    .toDouble(1.05),
                1.0,
                1.50);
            fairing_options.minimumAbsoluteVolumeRatio = std::clamp(
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyCarrierFairingMinimumVolumeRatio"))
                    .toDouble(0.90),
                0.50,
                1.0);
            fairing_options.maximumAbsoluteVolumeRatio = std::clamp(
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyCarrierFairingMaximumVolumeRatio"))
                    .toDouble(1.05),
                1.0,
                1.50);
            fairing_options.isCancelled = request.isCancelled;

            const TriMesh fairing_baseline = tsdf.mesh;
            const MeshTopologySignature topology_before_fairing =
                evaluateMeshTopologySignature(fairing_baseline);
            VisibilityOccupancyCarrierFairingResult fairing =
                VisibilityOccupancyCarrierFairer::fair(
                    fairing_baseline,
                    fairing_options);
            if (fairing.cancelled)
            {
                result.errorMessage = QStringLiteral(
                    "模型载体平滑已取消。");
                return result;
            }
            bool fairing_topology_preserved = false;
            if (fairing.ok)
            {
                const MeshTopologySignature topology_after_fairing =
                    evaluateMeshTopologySignature(fairing.mesh);
                fairing_topology_preserved =
                    topology_before_fairing.closedTwoManifold &&
                    topology_after_fairing == topology_before_fairing &&
                    hasSameFaceIndexBuffer(
                        fairing_baseline,
                        fairing.mesh);
                if (fairing_topology_preserved)
                {
                    tsdf.mesh = std::move(fairing.mesh);
                    detail::recomputeNormals(&tsdf.mesh);
                }
            }
            const bool fairing_applied =
                fairing.ok && fairing_topology_preserved;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_applied")] =
                fairing_applied;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_reverted")] =
                !fairing_applied;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_topology_preserved")] =
                fairing_topology_preserved;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_error")] =
                QString::fromStdString(fairing.errorMessage);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_completed_passes")] =
                fairing.statistics.completedIterationCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_accepted_half_steps")] =
                fairing.statistics.acceptedHalfStepCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_frozen_vertices")] =
                static_cast<double>(
                    fairing.statistics.locallyFrozenVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_rejected_faces")] =
                static_cast<double>(
                    fairing.statistics.locallyRejectedFaceCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_clamped_vertices")] =
                static_cast<double>(
                    fairing.statistics.displacementClampedVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_mean_edge_length")] =
                fairing.statistics.meanEdgeLength;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_displacement_limit")] =
                fairing.statistics.resolvedMaximumDisplacement;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_maximum_displacement")] =
                fairing.statistics.maximumAppliedDisplacement;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_area_ratio")] =
                fairing.statistics.finalSurfaceAreaRatio;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_volume_ratio")] =
                fairing.statistics.finalAbsoluteVolumeRatio;
        }
        else
        {
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_fairing_applied")] = false;
        }

        const bool carrier_field_projection_requested =
            request.settings.value(QStringLiteral(
                "tsdfVisibilityOccupancyCarrierFieldProjection"))
                .toBool(orbital_workspace);
        const bool carrier_field_projection_enabled =
            direct_visibility_occupancy_output &&
            carrier_field_projection_requested;
        result.payload[QStringLiteral(
            "configured_visibility_occupancy_carrier_field_projection")] =
            carrier_field_projection_enabled;
        if (carrier_field_projection_enabled)
        {
            const VisibilityOccupancyCarrierFieldGrid &projection_field =
                tsdf.depthImplicitField.valid()
                ? tsdf.depthImplicitField
                : tsdf.visibilityOccupancyCarrierField;
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_field_projection_source")] =
                tsdf.depthImplicitField.valid()
                ? QStringLiteral("depth_implicit_field")
                : QStringLiteral("occupancy_distance_field");
            const bool well_composed_carrier =
                tsdf.statistics
                    .visibilityOccupancyWellComposedRepairRemainingEdgeCheckerboardCount == 0 &&
                tsdf.statistics
                    .visibilityOccupancyWellComposedRepairRemainingVertexOccupiedDefectCount == 0 &&
                tsdf.statistics
                    .visibilityOccupancyWellComposedRepairRemainingVertexEmptyDefectCount == 0;
            const bool projection_prerequisites_met =
                well_composed_carrier &&
                tsdf.statistics
                    .visibilityOccupancyBoundaryTopologyConsistent &&
                projection_field.valid();
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_field_projection_prerequisites_met")] =
                projection_prerequisites_met;
            if (projection_prerequisites_met)
            {
                if (request.progress)
                {
                    request.progress(
                        QStringLiteral(
                            "正在将闭合载体弱投影到可见性距离场..."),
                        97);
                }
                VisibilityOccupancyCarrierFieldProjectionOptions
                    projection_options;
                const float occupancy_step =
                    visibilityOccupancyMedianVoxelStep(
                        tsdf.layout,
                        options.visibilityOccupancyResolution);
                std::array<double, 3> projection_spacing{};
                for (int axis = 0; axis < 3; ++axis)
                {
                    projection_spacing[axis] =
                        (static_cast<double>(projection_field.boundsMax[axis]) -
                         static_cast<double>(projection_field.boundsMin[axis])) /
                        static_cast<double>(std::max(
                            1,
                            projection_field.sampleDimensions[axis] - 1));
                }
                const double minimum_projection_spacing = std::max(
                    1.0e-12,
                    std::min({
                        projection_spacing[0],
                        projection_spacing[1],
                        projection_spacing[2]}));
                const double carrier_voxel_to_projection_spacing =
                    std::max(
                        1.0,
                        static_cast<double>(occupancy_step) /
                            minimum_projection_spacing);
                projection_options.iterations = qBound(
                    1,
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionIterations"))
                        .toInt(orbital_workspace ? 4 : 3),
                    8);
                projection_options.maximumBacktrackingSteps = qBound(
                    0,
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionMaximumBacktrackingSteps"))
                        .toInt(6),
                    10);
                projection_options.relaxation = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionRelaxation"))
                        .toDouble(orbital_workspace ? 0.45 : 0.40),
                    0.05,
                    1.0);
                const double maximum_step_carrier_voxels = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionMaximumStepVoxels"))
                        .toDouble(orbital_workspace ? 0.25 : 0.15),
                    0.01,
                    0.50);
                projection_options.maximumStepSpacingRatio =
                    maximum_step_carrier_voxels *
                    carrier_voxel_to_projection_spacing;
                const double maximum_displacement_carrier_voxels =
                    std::clamp(
                        request.settings.value(QStringLiteral(
                            "tsdfVisibilityOccupancyCarrierFieldProjectionMaximumDisplacementVoxels"))
                            .toDouble(orbital_workspace ? 0.75 : 0.30),
                        0.01,
                        1.0);
                projection_options.maximumCumulativeDisplacementSpacingRatio =
                    maximum_displacement_carrier_voxels *
                    carrier_voxel_to_projection_spacing;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_carrier_voxel_to_field_spacing_ratio")] =
                    carrier_voxel_to_projection_spacing;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_maximum_step_carrier_voxels")] =
                    maximum_step_carrier_voxels;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_maximum_displacement_carrier_voxels")] =
                    maximum_displacement_carrier_voxels;
                projection_options.smoothNarrowBand =
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionSmoothNarrowBand"))
                        .toBool(true);
                projection_options.narrowBandWidthSpacingRatio = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionNarrowBandVoxels"))
                        .toDouble(2.0),
                    0.25,
                    8.0);
                projection_options.scalarSmoothingRelaxation = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionSmoothingLambda"))
                        .toDouble(0.12),
                    0.0,
                    0.50);
                projection_options.minimumNormalDot = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionMinimumNormalDot"))
                        .toDouble(0.50),
                    -1.0,
                    1.0);
                projection_options.minimumFaceAreaRatio = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionMinimumFaceAreaRatio"))
                        .toDouble(0.25),
                    0.01,
                    1.0);
                projection_options.minimumSurfaceAreaRatio = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionMinimumAreaRatio"))
                        .toDouble(0.90),
                    0.50,
                    1.0);
                projection_options.maximumSurfaceAreaRatio = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionMaximumAreaRatio"))
                        .toDouble(1.05),
                    1.0,
                    1.50);
                projection_options.minimumAbsoluteVolumeRatio = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionMinimumVolumeRatio"))
                        .toDouble(0.92),
                    0.50,
                    1.0);
                projection_options.maximumAbsoluteVolumeRatio = std::clamp(
                    request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyCarrierFieldProjectionMaximumVolumeRatio"))
                        .toDouble(1.05),
                    1.0,
                    1.50);
                projection_options.isCancelled = request.isCancelled;

                const TriMesh projection_baseline = tsdf.mesh;
                const MeshTopologySignature topology_before_projection =
                    evaluateMeshTopologySignature(projection_baseline);
                VisibilityOccupancyCarrierFieldProjectionResult projection =
                    VisibilityOccupancyCarrierFieldProjector::project(
                        projection_baseline,
                        projection_field.sampleDimensions,
                        projection_field.boundsMin,
                        projection_field.boundsMax,
                        projection_field.signedWorldDistance,
                        projection_options);
                if (projection.cancelled)
                {
                    result.errorMessage = QStringLiteral(
                        "模型载体距离场投影已取消。");
                    return result;
                }
                bool projection_topology_preserved = false;
                if (projection.ok)
                {
                    const MeshTopologySignature topology_after_projection =
                        evaluateMeshTopologySignature(projection.mesh);
                    projection_topology_preserved =
                        topology_before_projection.closedTwoManifold &&
                        topology_after_projection ==
                            topology_before_projection &&
                        hasSameFaceIndexBuffer(
                            projection_baseline,
                            projection.mesh);
                    if (projection_topology_preserved)
                    {
                        tsdf.mesh = std::move(projection.mesh);
                        detail::recomputeNormals(&tsdf.mesh);
                    }
                }
                const bool projection_applied =
                    projection.ok && projection_topology_preserved;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_applied")] =
                    projection_applied;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_reverted")] =
                    projection.rolledBack ||
                    (projection.ok && !projection_topology_preserved);
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_topology_preserved")] =
                    projection_topology_preserved;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_error")] =
                    QString::fromStdString(projection.errorMessage);
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_completed_iterations")] =
                    projection.statistics.completedIterationCount;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_attempted_blends")] =
                    projection.statistics.attemptedBlendCount;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_accepted_full_steps")] =
                    projection.statistics.acceptedFullStepCount;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_accepted_half_steps")] =
                    projection.statistics.acceptedHalfStepCount;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_rejected_blends")] =
                    projection.statistics.rejectedBlendCount;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_minimum_accepted_blend")] =
                    projection.statistics.minimumAcceptedBlend;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_projected_vertices")] =
                    static_cast<double>(
                        projection.statistics.projectedVertexCount);
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_frozen_vertices")] =
                    static_cast<double>(
                        projection.statistics.locallyFrozenVertexCount);
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_step_clamped_vertices")] =
                    static_cast<double>(
                        projection.statistics.stepClampedVertexCount);
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_cumulative_clamped_vertices")] =
                    static_cast<double>(
                        projection.statistics.cumulativeClampedVertexCount);
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_mean_residual_before")] =
                    projection.statistics.meanAbsoluteFieldResidualBefore;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_mean_residual_after")] =
                    projection.statistics.meanAbsoluteFieldResidualAfter;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_p90_residual_before")] =
                    projection.statistics.p90AbsoluteFieldResidualBefore;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_p90_residual_after")] =
                    projection.statistics.p90AbsoluteFieldResidualAfter;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_maximum_displacement")] =
                    projection.statistics
                        .resolvedMaximumCumulativeDisplacement;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_area_ratio")] =
                    projection.statistics.finalSurfaceAreaRatio;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_volume_ratio")] =
                    projection.statistics.finalAbsoluteVolumeRatio;
            }
            else
            {
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_applied")] =
                    false;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_reverted")] =
                    false;
                result.payload[QStringLiteral(
                    "visibility_occupancy_carrier_field_projection_error")] =
                    QStringLiteral(
                        "占据场或良构闭合拓扑前置条件不满足，已保留原载体。");
            }
        }
        else
        {
            result.payload[QStringLiteral(
                "visibility_occupancy_carrier_field_projection_applied")] =
                false;
        }

        const bool visibility_occupancy_depth_refinement_enabled =
            output_mesh == &tsdf.mesh &&
            options.enableVisibilityOccupancyCompletion &&
            visibilityOccupancyDepthRefinementEnabled(
                request.settings, orbital_workspace);
        result.payload[QStringLiteral(
            "configured_visibility_occupancy_depth_refinement")] =
            visibility_occupancy_depth_refinement_enabled;
        if (visibility_occupancy_depth_refinement_enabled)
        {
            TriMesh direct_refinement_baseline;
            MeshTopologySignature direct_refinement_topology;
            if (direct_visibility_occupancy_output)
            {
                direct_refinement_baseline = tsdf.mesh;
                direct_refinement_topology =
                    evaluateMeshTopologySignature(tsdf.mesh);
            }
            if (request.progress)
            {
                request.progress(
                    QStringLiteral(
                        "正在用鲁棒多视深度约束细化闭合表面..."),
                    97);
            }
            const DepthConstrainedSurfaceRefineOptions refine_options =
                makeVisibilityOccupancyDepthRefineOptions(
                    request.settings,
                    tsdf.layout,
                    options.visibilityOccupancyResolution,
                    orbital_workspace);
            const DepthConstrainedSurfaceRefineStatistics refinement =
                DepthConstrainedSurfaceRefiner::refine(
                    &tsdf.mesh,
                    loaded.frames,
                    refine_options,
                    request.isCancelled);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_applied")] =
                refinement.applied;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_reverted")] =
                refinement.reverted;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_attempted_passes")] =
                refinement.attemptedPassCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_applied_passes")] =
                refinement.appliedPassCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_reverted_passes")] =
                refinement.revertedPassCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_accepted_blend")] =
                refinement.acceptedBlend;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_area_before")] =
                refinement.areaBefore;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_area_after")] =
                refinement.areaAfter;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_volume_before")] =
                refinement.absoluteVolumeBefore;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_volume_after")] =
                refinement.absoluteVolumeAfter;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_flipped_faces")] =
                static_cast<double>(refinement.flippedFaceCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_degenerate_faces")] =
                static_cast<double>(refinement.degenerateFaceCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_locally_projected_candidates")] =
                refinement.locallyProjectedCandidateCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_local_projection_iterations")] =
                refinement.localSafetyProjectionIterationCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_locally_rejected_faces")] =
                static_cast<double>(refinement.locallyRejectedFaceCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_locally_frozen_vertices")] =
                static_cast<double>(refinement.locallyFrozenVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_accepted_rejected_faces")] =
                static_cast<double>(
                    refinement.acceptedLocallyRejectedFaceCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_accepted_frozen_vertices")] =
                static_cast<double>(
                    refinement.acceptedLocallyFrozenVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_removed_median_normal_bias")] =
                refinement.removedMedianNormalBias;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_projected_observations")] =
                static_cast<double>(
                    refinement.refiner.projectedObservationCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_accepted_observations")] =
                static_cast<double>(
                    refinement.refiner.acceptedObservationCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_anchored_vertices")] =
                static_cast<double>(
                    refinement.refiner.anchoredVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_displaced_vertices")] =
                static_cast<double>(
                    refinement.refiner.displacedVertexCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_bias_frames")] =
                refinement.refiner.biasCalibratedFrameCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_bias_pairs")] =
                static_cast<double>(
                    refinement.refiner.biasCalibrationPairCount);
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_maximum_bias")] =
                refinement.refiner.maximumAbsoluteFrameBias;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_median_views")] =
                refinement.refiner.medianSupportingViewCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_p90_views")] =
                refinement.refiner.p90SupportingViewCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_maximum_displacement")] =
                refinement.refiner.maximumAppliedDisplacement;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_global_solver_applied")] =
                refinement.refiner.globalSolverAppliedPassCount > 0;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_global_solver_energy_before")] =
                refinement.refiner.globalSolverInitialEnergy;
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_global_solver_energy_after")] =
                refinement.refiner.globalSolverFinalEnergy;
            bool topology_guard_reverted = false;
            if (direct_visibility_occupancy_output && refinement.applied)
            {
                const MeshTopologySignature refined_topology =
                    evaluateMeshTopologySignature(tsdf.mesh);
                const bool topology_preserved =
                    direct_refinement_topology.closedTwoManifold &&
                    refined_topology == direct_refinement_topology &&
                    hasSameFaceIndexBuffer(
                        direct_refinement_baseline,
                        tsdf.mesh);
                if (!topology_preserved)
                {
                    tsdf.mesh = std::move(direct_refinement_baseline);
                    topology_guard_reverted = true;
                    result.payload[QStringLiteral(
                        "visibility_occupancy_depth_refinement_applied")] =
                        false;
                    result.payload[QStringLiteral(
                        "visibility_occupancy_depth_refinement_reverted")] =
                        true;
                }
                result.payload[QStringLiteral(
                    "visibility_occupancy_depth_refinement_topology_preserved")] =
                    topology_preserved;
            }
            result.payload[QStringLiteral(
                "visibility_occupancy_depth_refinement_topology_guard_reverted")] =
                topology_guard_reverted;
        }

        const int post_refinement_target_faces =
            direct_visibility_occupancy_output
            ? qBound(
                  0,
                  request.settings.value(QStringLiteral(
                      "tsdfVisibilityOccupancyPostRefinementTargetFaces"))
                      .toInt(0),
                  5000000)
            : 0;
        result.payload[QStringLiteral(
            "configured_visibility_occupancy_post_refinement_target_faces")] =
            post_refinement_target_faces;
        if (post_refinement_target_faces > 0 &&
            post_refinement_target_faces < output_mesh->faceCount())
        {
            if (request.progress)
            {
                request.progress(
                    QStringLiteral("正在保拓扑简化细化表面..."),
                    98);
            }
            const MeshTopologySignature topology_before_simplification =
                evaluateMeshTopologySignature(*output_mesh);
            TriMesh simplified_candidate = *output_mesh;
            QuadricSimplifyOptions simplify_options;
            simplify_options.targetFaceCount = post_refinement_target_faces;
            simplify_options.maximumPasses = 24;
            simplify_options.workerCount = options.workerCount;
            simplify_options.maximumResultFaceAspectRatio =
                std::clamp(
                    static_cast<float>(request.settings.value(QStringLiteral(
                        "tsdfVisibilityOccupancyPostRefinementMaximumFaceAspectRatio"))
                        .toDouble(10.0)),
                    1.2f,
                    1000.0f);
            simplify_options.featureAngleDegrees = 45.0f;
            simplify_options.maximumNormalDeviationDegrees = 45.0f;
            simplify_options.preserveOpenBoundaries = true;
            simplify_options.simplifySimpleOpenBoundaries = false;
            simplify_options.isCancelled = request.isCancelled;
            const QuadricSimplifyStatistics simplification =
                simplifyMeshQuadric(
                    &simplified_candidate,
                    simplify_options);
            const MeshTopologySignature topology_after_simplification =
                evaluateMeshTopologySignature(simplified_candidate);
            const bool simplification_accepted =
                !simplification.cancelled &&
                simplified_candidate.faceCount() <
                    output_mesh->faceCount() &&
                topology_before_simplification.closedTwoManifold &&
                topology_after_simplification ==
                    topology_before_simplification;
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_simplification_attempted")] =
                true;
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_simplification_accepted")] =
                simplification_accepted;
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_simplification_input_faces")] =
                simplification.inputFaceCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_simplification_output_faces")] =
                simplification.outputFaceCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_simplification_collapsed_edges")] =
                simplification.collapsedEdgeCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_simplification_rejected_triangle_quality_edges")] =
                simplification.rejectedTriangleQualityEdgeCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_maximum_face_aspect_ratio")] =
                simplify_options.maximumResultFaceAspectRatio;
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_simplification_reached_target")] =
                simplification.reachedTarget;
            if (simplification_accepted)
            {
                tsdf.mesh = std::move(simplified_candidate);
            }
        }
        else
        {
            result.payload[QStringLiteral(
                "visibility_occupancy_post_refinement_simplification_attempted")] =
                false;
        }

        const bool final_surface_denoising_enabled =
            request.settings.contains(QStringLiteral(
                "tsdfFinalSurfaceDenoising"))
            ? request.settings.value(QStringLiteral(
                  "tsdfFinalSurfaceDenoising")).toBool(true)
            : request.settings.value(QStringLiteral(
                  "tsdfVisibilityOccupancyFinalSurfaceDenoising"))
                  .toBool(true);
        result.payload[QStringLiteral(
            "configured_final_surface_denoising")] =
            final_surface_denoising_enabled;
        result.payload[QStringLiteral(
            "configured_visibility_occupancy_final_surface_denoising")] =
            final_surface_denoising_enabled;
        if (final_surface_denoising_enabled)
        {
            if (request.progress)
            {
                request.progress(
                    QStringLiteral("正在进行保存前的曲率保护表面降噪..."),
                    98);
            }
            const int final_denoising_iterations = qBound(
                0,
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyFinalSurfaceDenoisingIterations"))
                    .toInt(std::max(8, options.surfaceDenoisingIterations)),
                12);
            const float final_denoising_lambda = std::clamp(
                static_cast<float>(request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyFinalSurfaceDenoisingLambda"))
                    .toDouble(std::max(0.50f, options.surfaceDenoisingLambda))),
                0.0f,
                0.75f);
            const float final_denoising_mu = std::clamp(
                static_cast<float>(request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyFinalSurfaceDenoisingMu"))
                    .toDouble(options.surfaceDenoisingMu)),
                -1.0f,
                0.0f);
            const float final_denoising_displacement_voxels = std::clamp(
                static_cast<float>(request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyFinalSurfaceDenoisingMaximumDisplacementVoxels"))
                    .toDouble(std::max(
                        0.75f,
                        options.maximumSurfaceDenoisingDisplacementVoxels))),
                0.0f,
                1.5f);
            const float final_denoising_feature_angle = std::clamp(
                static_cast<float>(request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyFinalSurfaceDenoisingFeatureAngleDegrees"))
                    .toDouble(std::max(
                        120.0f,
                        options.maximumSurfaceDenoisingNormalAngleDegrees))),
                5.0f,
                170.0f);
            const int final_denoising_boundary_rings = qBound(
                0,
                request.settings.value(QStringLiteral(
                    "tsdfVisibilityOccupancyFinalSurfaceDenoisingBoundaryProtectionRings"))
                    .toInt(0),
                2);
            const float maximum_voxel_size = std::max({
                tsdf.layout.voxelSize[0],
                tsdf.layout.voxelSize[1],
                tsdf.layout.voxelSize[2]});
            const FinalSurfaceDenoisingResult denoising =
                applyTopologyGuardedFinalSurfaceDenoising(
                    output_mesh,
                    final_denoising_iterations,
                    final_denoising_lambda,
                    final_denoising_mu,
                    final_denoising_displacement_voxels * maximum_voxel_size,
                    final_denoising_feature_angle,
                    final_denoising_boundary_rings);
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_attempted")] =
                denoising.attempted;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_accepted")] =
                denoising.accepted;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_moved_vertices")] =
                denoising.movedVertexCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_area_before")] =
                denoising.areaBefore;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_area_after")] =
                denoising.areaAfter;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_normal_median_before")] =
                denoising.qualityBefore.adjacentNormalAngleMedianDegrees;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_normal_median_after")] =
                denoising.qualityAfter.adjacentNormalAngleMedianDegrees;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_normal_p90_before")] =
                denoising.qualityBefore.adjacentNormalAngleP90Degrees;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_normal_p90_after")] =
                denoising.qualityAfter.adjacentNormalAngleP90Degrees;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_normal_over_30_before")] =
                denoising.qualityBefore.adjacentNormalAngleOver30Ratio;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_denoising_normal_over_30_after")] =
                denoising.qualityAfter.adjacentNormalAngleOver30Ratio;
            result.payload[QStringLiteral(
                "final_surface_denoising_accepted")] =
                denoising.accepted;
            result.payload[QStringLiteral(
                "final_surface_denoising_moved_vertices")] =
                denoising.movedVertexCount;
            result.payload[QStringLiteral(
                "final_surface_denoising_normal_p90_before")] =
                denoising.qualityBefore.adjacentNormalAngleP90Degrees;
            result.payload[QStringLiteral(
                "final_surface_denoising_normal_p90_after")] =
                denoising.qualityAfter.adjacentNormalAngleP90Degrees;
            result.payload[QStringLiteral(
                "final_surface_denoising_normal_over_30_before")] =
                denoising.qualityBefore.adjacentNormalAngleOver30Ratio;
            result.payload[QStringLiteral(
                "final_surface_denoising_normal_over_30_after")] =
                denoising.qualityAfter.adjacentNormalAngleOver30Ratio;
        }

        if (options.enableDepthCompletenessDiagnostics)
        {
            if (request.progress)
            {
                request.progress(
                    QStringLiteral(
                        "正在以精确点到三角形距离检查最终模型完整性..."),
                    99);
            }
            DepthMeshCompletenessStatistics final_completeness =
                evaluateDepthCompleteness(
                    *output_mesh, loaded.frames, options, tsdf.layout);
            bool gap_boundary_available = false;
            double gap_boundary_minimum_recall = 1.0;
            for (const DepthMeshFrameCompleteness &frame :
                 final_completeness.frames)
            {
                if (orbitalRoleForDepthFrame(
                        tsdf.statistics.orbitalFrameRoles,
                        frame.refIndex) != QStringLiteral("gap_boundary"))
                {
                    continue;
                }
                gap_boundary_available = true;
                gap_boundary_minimum_recall = std::min(
                    gap_boundary_minimum_recall, frame.recall);
            }
            const bool gap_boundary_gate_passed =
                !gap_boundary_available ||
                gap_boundary_minimum_recall + 1.0e-9 >=
                    options.minimumDepthCompletenessP10Recall;
            final_completeness.gatePassed =
                final_completeness.gatePassed &&
                gap_boundary_gate_passed;
            addDepthCompletenessPayload(
                final_completeness,
                QStringLiteral("final_depth_completeness_"),
                &result.payload);
            addDepthCompletenessPayload(
                final_completeness,
                QStringLiteral("depth_completeness_"),
                &result.payload);
            result.payload[QStringLiteral(
                "final_depth_completeness_gap_boundary_available")] =
                gap_boundary_available;
            result.payload[QStringLiteral(
                "final_depth_completeness_gap_boundary_gate_passed")] =
                gap_boundary_gate_passed;
            result.payload[QStringLiteral(
                "final_depth_completeness_gap_boundary_minimum_recall")] =
                gap_boundary_available
                    ? gap_boundary_minimum_recall
                    : 0.0;
            result.payload[QStringLiteral(
                "final_depth_completeness_minimum_p10_threshold")] =
                options.minimumDepthCompletenessP10Recall;
            result.payload[QStringLiteral(
                "final_depth_completeness_minimum_median_threshold")] =
                options.minimumDepthCompletenessMedianRecall;
            const QStringList worst_labels = worstDepthCompletenessLabels(
                final_completeness,
                tsdf.statistics.orbitalFrameRoles);
            result.payload[QStringLiteral(
                "final_depth_completeness_worst_frames")] =
                worst_labels.join(QStringLiteral(", "));
            if (request.progress)
            {
                request.progress(
                    QStringLiteral(
                        "最终完整性：中位=%1(阈值 %2)，P10=%3(阈值 %4)，"
                        "最低=%5，最差视角=%6")
                        .arg(final_completeness.medianFrameRecall, 0, 'f', 4)
                        .arg(options.minimumDepthCompletenessMedianRecall,
                             0,
                             'f',
                             4)
                        .arg(final_completeness.p10FrameRecall, 0, 'f', 4)
                        .arg(options.minimumDepthCompletenessP10Recall,
                             0,
                             'f',
                             4)
                        .arg(final_completeness.minimumFrameRecall,
                             0,
                             'f',
                             4)
                        .arg(worst_labels.join(QStringLiteral(", "))),
                    99);
            }
            if (options.enforceDepthCompletenessGate &&
                (!final_completeness.available ||
                 !final_completeness.gatePassed))
            {
                QStringList failed_conditions;
                if (!final_completeness.available)
                {
                    failed_conditions.push_back(
                        QStringLiteral("没有足够的有效深度观测"));
                }
                if (final_completeness.available &&
                    final_completeness.medianFrameRecall + 1.0e-9 <
                        options.minimumDepthCompletenessMedianRecall)
                {
                    failed_conditions.push_back(
                        QStringLiteral("中位召回率 %1 < %2")
                            .arg(final_completeness.medianFrameRecall,
                                 0,
                                 'f',
                                 4)
                            .arg(options.minimumDepthCompletenessMedianRecall,
                                 0,
                                 'f',
                                 4));
                }
                if (final_completeness.available &&
                    final_completeness.p10FrameRecall + 1.0e-9 <
                        options.minimumDepthCompletenessP10Recall)
                {
                    failed_conditions.push_back(
                        QStringLiteral("P10 召回率 %1 < %2")
                            .arg(final_completeness.p10FrameRecall,
                                 0,
                                 'f',
                                 4)
                            .arg(options.minimumDepthCompletenessP10Recall,
                                 0,
                                 'f',
                                 4));
                }
                if (!gap_boundary_gate_passed)
                {
                    failed_conditions.push_back(
                        QStringLiteral("缺口边界视角最低召回率 %1 < %2")
                            .arg(gap_boundary_minimum_recall, 0, 'f', 4)
                            .arg(options.minimumDepthCompletenessP10Recall,
                                 0,
                                 'f',
                                 4));
                }
                result.errorMessage = QStringLiteral(
                    "最终模型深度观测完整性质量门未通过：%1；"
                    "最差视角=%2。基础 TSDF 与补全阶段均已执行，"
                    "已停止写入不完整模型。")
                    .arg(failed_conditions.join(QStringLiteral("；")),
                         worst_labels.join(QStringLiteral(", ")));
                return result;
            }
        }

        if (direct_visibility_occupancy_output)
        {
            const MeshTopologyQualityStatistics final_topology =
                evaluateMeshTopologyQuality(*output_mesh);
            const int expected_surface_euler =
                2 * tsdf.statistics
                        .visibilityOccupancyBoundaryBodyEulerCharacteristic;
            const bool final_topology_preserved =
                final_topology.closedTwoManifold &&
                final_topology.eulerCharacteristic ==
                    tsdf.statistics
                        .visibilityOccupancyBoundarySurfaceEulerCharacteristic &&
                final_topology.eulerCharacteristic == expected_surface_euler;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_closed_two_manifold")] =
                final_topology.closedTwoManifold;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_boundary_edge_count")] =
                final_topology.boundaryEdgeCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_non_manifold_edge_count")] =
                final_topology.nonManifoldEdgeCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_non_manifold_vertex_count")] =
                final_topology.nonManifoldVertexCount;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_euler_characteristic")] =
                final_topology.eulerCharacteristic;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_high_aspect_face_ratio")] =
                final_topology.highAspectFaceRatio;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_extreme_aspect_face_ratio")] =
                final_topology.extremeAspectFaceRatio;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_adjacent_normal_median_degrees")] =
                final_topology.adjacentNormalAngleMedianDegrees;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_adjacent_normal_p90_degrees")] =
                final_topology.adjacentNormalAngleP90Degrees;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_adjacent_normal_over_30_ratio")] =
                final_topology.adjacentNormalAngleOver30Ratio;
            result.payload[QStringLiteral(
                "visibility_occupancy_final_surface_area")] =
                meshSurfaceArea(*output_mesh);
            result.payload[QStringLiteral(
                "visibility_occupancy_final_component_eulers")] =
                eulerCharacteristicsToJson(
                    final_topology.componentEulerCharacteristics);
            result.payload[QStringLiteral(
                "visibility_occupancy_final_topology_preserved")] =
                final_topology_preserved;
            if (!final_topology_preserved)
            {
                result.errorMessage = QStringLiteral(
                    "可见性占据场最终拓扑质量门未通过：边界边=%1，"
                    "非流形边=%2，非流形顶点=%3，曲面欧拉特征=%4"
                    "（期望 %5）。已停止写入模型。")
                    .arg(final_topology.boundaryEdgeCount)
                    .arg(final_topology.nonManifoldEdgeCount)
                    .arg(final_topology.nonManifoldVertexCount)
                    .arg(final_topology.eulerCharacteristic)
                    .arg(expected_surface_euler);
                return result;
            }
        }

        const QJsonObject diagnostics = result.payload;
        QVector<MeshColorView> texture_views;
        texture_views.reserve(loaded.frames.size());
        for (const DepthTsdfFrame &frame : loaded.frames)
        {
            texture_views.push_back(textureViewFromFrame(frame));
        }
        if (request.progress)
        {
            request.progress(
                QStringLiteral("正在保存三角网格..."),
                output_algorithm == QStringLiteral(
                    "orbital_visual_hull_completion")
                    ? 99
                    : (request.exportObj ? 85 : 99));
        }
        std::function<void(const QString &, int)> output_progress = request.progress;
        if (output_algorithm == QStringLiteral(
                "orbital_visual_hull_completion") &&
            request.progress)
        {
            output_progress = [progress = request.progress](
                                  const QString &stage, int)
            {
                progress(stage, 99);
            };
        }
        else if (request.exportObj && request.progress)
        {
            output_progress = [progress = request.progress](
                                  const QString &stage, int percent)
            {
                const int bounded_percent = std::clamp(percent, 0, 100);
                progress(stage, 87 + bounded_percent * 12 / 100);
            };
        }
        result = saveMeshAndOptionalTexture(*output_mesh,
                                            output_algorithm.toStdString(),
                                            output_root,
                                            request.exportObj,
                                            request.texture,
                                            output_progress,
                                            &texture_views);
        const QJsonObject output_payload = result.payload;
        mergePayload(diagnostics, &result.payload);
        mergePayload(output_payload, &result.payload);
        if (result.ok && !tsdf.boundaryAttributionDebugMesh.empty())
        {
            const QString debug_ply_path = QDir(
                QDir(output_root).filePath(QStringLiteral("products")))
                .filePath(QStringLiteral("boundary_attribution_debug.ply"));
            std::string debug_error;
            if (tsdf.boundaryAttributionDebugMesh.savePLY(
                    xjw::common::io::toUtf8Path(debug_ply_path),
                    &debug_error))
            {
                result.payload[QStringLiteral(
                    "boundary_attribution_debug_ply")] = debug_ply_path;
            }
            else
            {
                result.payload[QStringLiteral(
                    "boundary_attribution_debug_error")] =
                    QString::fromUtf8(debug_error);
            }
        }
        if (result.ok && !tsdf.acquisitionGapReport.isEmpty())
        {
            const QString report_path = QDir(
                QDir(output_root).filePath(QStringLiteral("products")))
                .filePath(QStringLiteral("acquisition_gap_report.json"));
            QSaveFile report_file(report_path);
            if (report_file.open(QIODevice::WriteOnly))
            {
                report_file.write(QJsonDocument(tsdf.acquisitionGapReport)
                                      .toJson(QJsonDocument::Indented));
                if (report_file.commit())
                {
                    result.payload[QStringLiteral("acquisition_gap_report")] =
                        report_path;
                    result.payload[QStringLiteral(
                        "acquisition_gap_recommended_next_action")] =
                        tsdf.acquisitionGapReport.value(QStringLiteral(
                            "recommended_next_action"));
                    result.payload[QStringLiteral(
                        "acquisition_gap_highest_risk_sectors")] =
                        tsdf.acquisitionGapReport.value(QStringLiteral(
                            "highest_risk_sectors"));
                }
                else
                {
                    result.payload[QStringLiteral(
                        "acquisition_gap_report_error")] =
                        report_file.errorString();
                }
            }
            else
            {
                result.payload[QStringLiteral(
                    "acquisition_gap_report_error")] =
                    report_file.errorString();
            }
        }
        if (result.ok && request.progress)
        {
            request.progress(QStringLiteral("模型生成完成"), 100);
        }
        return result;
    }

    if (mode != QStringLiteral("visual_hull") &&
        mode != QStringLiteral("poisson_legacy"))
    {
        result.errorMessage = QStringLiteral("未知深度模型重建模式: %1").arg(mode);
        return result;
    }

    if (mode == QStringLiteral("visual_hull"))
    {
        DepthMapVisualHullOptions visual_hull_options;
        visual_hull_options.strictVolumetricMasks =
            request.settings.value(QStringLiteral("strictVolumetricMasks")).toBool(false);
        visual_hull_options.smoothingIterations = qBound(
            0,
            request.settings.value(QStringLiteral("visualHullSmoothingIterations"))
                .toInt(visual_hull_options.smoothingIterations),
            20);
        visual_hull_options.smoothingLambda = std::clamp(
            static_cast<float>(
                request.settings.value(QStringLiteral("visualHullSmoothingLambda"))
                    .toDouble(visual_hull_options.smoothingLambda)),
            0.0f,
            0.49f);
        const DepthMapVisualHullResult visual_hull = DepthMapMeshBuilder::buildVisualHull(
            request.depthMapSourcePath,
            request.reconstruction.resolution,
            visual_hull_options,
            request.progress);
        result.payload[QStringLiteral("visual_hull_views")] = visual_hull.usableViewCount;
        result.payload[QStringLiteral("visual_hull_depth_views")] = visual_hull.depthViewCount;
        result.payload[QStringLiteral("visual_hull_depth_carving")] =
            visual_hull.usedDepthFreeSpaceCarving;
        result.payload[QStringLiteral("visual_hull_retried_without_depth_carving")] =
            visual_hull.retriedWithoutDepthCarving;
        result.payload[QStringLiteral("visual_hull_component_count")] =
            visual_hull.connectivity.componentCount;
        result.payload[QStringLiteral("visual_hull_removed_satellite_components")] =
            visual_hull.removedSatelliteComponentCount;
        result.payload[QStringLiteral("visual_hull_largest_component_ratio")] =
            visual_hull.connectivity.largestComponentFaceRatio;
        if (!visual_hull.applicable || !visual_hull.ok)
        {
            result.errorMessage = visual_hull.message.isEmpty()
                ? QStringLiteral("显式 Visual Hull 模式不可用于当前深度数据")
                : visual_hull.message;
            return result;
        }

        const QJsonObject diagnostics = result.payload;
        result = saveMeshAndOptionalTexture(visual_hull.mesh,
                                            "silhouette_visual_hull",
                                            output_root,
                                            request.exportObj,
                                            request.texture,
                                            request.progress);
        mergePayload(diagnostics, &result.payload);
        return result;
    }

    QString resolveError;
    QString densePath = request.reusableDenseCloudPath.trimmed();
    if (!densePath.isEmpty() && !QFileInfo::exists(densePath))
    {
        resolveError = QStringLiteral("指定的可复用密集点云不存在: %1").arg(densePath);
        densePath.clear();
    }
    if (densePath.isEmpty())
    {
        densePath = xjw::mesh::DepthMapMeshBuilder::resolveReusableDenseCloud(
            request.depthMapSourcePath,
            &resolveError);
    }
    if (densePath.isEmpty())
    {
        result.errorMessage = resolveError.isEmpty()
            ? QStringLiteral("显式 poisson_legacy 模式缺少可复用密集点云")
            : resolveError;
        return result;
    }

    MeshBuildRequest meshRequest;
    meshRequest.pointCloudPath = densePath;
    meshRequest.outputRoot = output_root;
    meshRequest.reconstruction = request.reconstruction;
    meshRequest.exportObj = request.exportObj;
    meshRequest.texture = request.texture;
    meshRequest.progress = request.progress;

    const QJsonObject diagnostics = result.payload;
    result = buildMeshAndOptionalTexture(meshRequest);
    mergePayload(diagnostics, &result.payload);
    result.payload[QStringLiteral("source_point_cloud_path")] = densePath;
    return result;
}

WorkflowResult buildModel(const ModelBuildRequest &request)
{
    const QString source_data = request.sourceData.trimmed().isEmpty()
        ? QStringLiteral("point_cloud")
        : request.sourceData.trimmed();
    const ReconstructionConfig reconstruction =
        reconstructionConfigFromModelSettings(request.settings);

    WorkflowResult result;
    if (source_data == QStringLiteral("depth_maps"))
    {
        DepthMapMeshBuildRequest depth_request;
        depth_request.depthMapSourcePath = request.depthMapSourcePath.trimmed().isEmpty()
            ? request.requestedSourcePath
            : request.depthMapSourcePath;
        depth_request.reusableDenseCloudPath = request.sourcePointCloudPath;
        depth_request.outputRoot = request.outputRoot;
        depth_request.settings = request.settings;
        depth_request.reconstruction = reconstruction;
        depth_request.exportObj = false;
        depth_request.texture = defaultTextureConfig();
        depth_request.texture.isCancelled = request.isCancelled;
        depth_request.isCancelled = request.isCancelled;
        depth_request.progress = request.progress;
        result = buildMeshFromDepthMaps(depth_request);
    }
    else
    {
        MeshBuildRequest mesh_request;
        mesh_request.pointCloudPath = request.sourcePointCloudPath.trimmed().isEmpty()
            ? request.requestedSourcePath
            : request.sourcePointCloudPath;
        mesh_request.outputRoot = request.outputRoot;
        mesh_request.reconstruction = reconstruction;
        mesh_request.exportObj = false;
        mesh_request.texture = defaultTextureConfig();
        mesh_request.progress = request.progress;
        result = buildMeshAndOptionalTexture(mesh_request);
    }

    if (result.ok)
    {
        result.payload[QStringLiteral("source_data")] = source_data;
        result.payload[QStringLiteral("source_path")] = request.requestedSourcePath;
        result.payload[QStringLiteral("source_point_cloud_path")] = request.sourcePointCloudPath;
        if (source_data == QStringLiteral("depth_maps"))
        {
            result.payload[QStringLiteral("depth_map_source_path")] =
                request.depthMapSourcePath.trimmed().isEmpty()
                    ? request.requestedSourcePath
                    : request.depthMapSourcePath;
        }
    }
    return result;
}

WorkflowResult buildTextureOnly(const TextureBuildRequest &request)
{
    WorkflowResult result;

    if (request.meshPath.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("网格路径为空");
        return result;
    }

    xjw::mesh::TextureMappingConfig textureConfig = request.texture;
    textureConfig.isCancelled = request.isCancelled;
    if (request.progress)
    {
        textureConfig.progressFn = [cb = request.progress](const std::string &stage, int percent) {
            cb(QString::fromStdString(stage), percent);
        };
    }

    xjw::mesh::TextureMappingResult textureResult;
    std::string textureError;
    if (!request.depthMapSourcePath.trimmed().isEmpty())
    {
        const QVector<DepthFrameArtifact> artifacts =
            DepthMapMeshBuilder::discoverDepthFrames(request.depthMapSourcePath);
        const DepthTsdfFrameLoadResult loaded = DepthTsdfSurfaceBuilder::loadFrames(artifacts);
        if (!loaded.ok)
        {
            result.errorMessage = QStringLiteral("无法加载相机纹理源: %1").arg(loaded.errorMessage);
            return result;
        }
        QVector<MeshColorView> views;
        views.reserve(loaded.frames.size());
        for (const DepthTsdfFrame &frame : loaded.frames)
        {
            views.push_back(textureViewFromFrame(frame));
        }
        result.ok = xjw::mesh::TextureMapper::generateCameraTexturedModelFromMeshFile(
            xjw::common::io::toUtf8Path(request.meshPath),
            xjw::common::io::toUtf8Path(request.outputDir),
            textureConfig,
            views,
            &textureResult,
            &textureError);
    }
    else
    {
        if (!request.allowVertexColorFallback)
        {
            result.errorMessage = QStringLiteral(
                "当前模型没有可用的深度图与相机证据，已停止多视图纹理生成；"
                "如需使用顶点色平面纹理，请在界面中明确确认回退。");
            return result;
        }
        result.ok = xjw::mesh::TextureMapper::generateTexturedModelFromMeshFile(
            xjw::common::io::toUtf8Path(request.meshPath),
            xjw::common::io::toUtf8Path(request.outputDir),
            textureConfig,
            &textureResult,
            &textureError);
    }

    if (!result.ok)
    {
        result.errorMessage = QString::fromStdString(textureError);
        return result;
    }

    result.payload = textureResultToJson(textureResult, &textureConfig);
    return result;
}

} // namespace xjw::mesh::workflow
