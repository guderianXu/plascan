#include "ModelWorkflowService.h"

#include "DepthMapMeshBuilder.h"
#include "DepthTsdfSurfaceBuilder.h"
#include "MeshColorizer.h"
#include "SurfaceReconstructor.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <algorithm>
#include <cmath>

namespace xjw::mesh::workflow
{

namespace
{

QJsonObject textureResultToJson(const xjw::mesh::TextureMappingResult &result)
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
    object[QStringLiteral("texture_unmapped_face_count")] = result.unmappedFaceCount;
    return object;
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
            const QJsonObject texture_json = textureResultToJson(texture_result);
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
    options.enableQuadricSimplification = settings.value(
        QStringLiteral("tsdfQuadricSimplification")).toBool(false);
    options.simplifyTargetFaces = qBound(
        0,
        settings.value(QStringLiteral("simplifyTargetFaces"))
            .toInt(settings.value(QStringLiteral("targetFaces")).toInt(0)),
        2000000);
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
    options.minimumVoxelWeight = std::clamp(
        static_cast<float>(settings.value(QStringLiteral("tsdfMinimumVoxelWeight"))
                               .toDouble(options.minimumVoxelWeight)),
        0.05f,
        20.0f);
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
    options.minimumGeometrySupportCount = qBound(
        2,
        settings.value(QStringLiteral("tsdfMinimumGeometrySupportCount"))
            .toInt(options.minimumGeometrySupportCount),
        16);
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
            options.allowInvalidNearestPixelRecovery);
    options.enableSurfacePatchSupport = settings.value(
        QStringLiteral("tsdfSurfacePatchSupport")).toBool(false);
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
    options.minimumSurfacePatchWeightRatio = std::clamp(
        static_cast<float>(settings.value(
            QStringLiteral("tsdfMinimumSurfacePatchWeightRatio"))
                               .toDouble(options.minimumSurfacePatchWeightRatio)),
        0.01f,
        1.0f);
    options.enableSupportMaskFreeSpaceCarving = settings.value(
        QStringLiteral("tsdfSupportMaskFreeSpaceCarving")).toBool(false);
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

    const QString interpolation = settings.value(
        QStringLiteral("interpolation")).toString(QStringLiteral("enabled"));
    options.fillSmallBoundaryHoles = interpolation != QStringLiteral("disabled")
        && settings.value(QStringLiteral("holeFill")).toBool(true);
    const double maximum_hole_area = std::max(
        1.0, settings.value(QStringLiteral("maxHoleSize")).toDouble(100.0));
    const int conservative_edges = std::clamp(
        static_cast<int>(std::lround(std::sqrt(maximum_hole_area) * 1.6)),
        8,
        32);
    options.maximumHoleBoundaryEdges = interpolation == QStringLiteral("extrapolated")
        ? std::clamp(std::max(64, conservative_edges * 4), 64, 128)
        : conservative_edges;
    options.maximumHoleDiameterVoxels = interpolation == QStringLiteral("extrapolated")
        ? 16.0f
        : 4.0f;
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
    options.boundarySmoothingIterations = qBound(
        0,
        settings.value(QStringLiteral("tsdfBoundarySmoothingIterations")).toInt(1),
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
    const bool automatic_trim_weak_boundary_tips = options.resolution >= 384;
    options.trimWeakBoundaryTips = settings.value(
        QStringLiteral("tsdfTrimWeakBoundaryTips")).toBool(
            automatic_trim_weak_boundary_tips);
    options.weakBoundaryTipTrimPasses = qBound(
        1,
        settings.value(QStringLiteral("tsdfWeakBoundaryTipTrimPasses")).toInt(1),
        4);
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

} // namespace

xjw::mesh::DepthTsdfOptions depthTsdfOptionsFromSettings(const QJsonObject &settings,
                                                         int requestedResolution)
{
    return makeDepthTsdfOptions(settings, requestedResolution);
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
        config.padding = settings.value(QStringLiteral("padding")).toInt(config.padding);
        config.keepUnmapped = settings.value(QStringLiteral("keepUnmapped")).toBool(config.keepUnmapped);

        const QString blendMethod = settings.value(QStringLiteral("blendMethod")).toString();
        if (!blendMethod.isEmpty())
        {
            config.blendMethod = blendMethod.toStdString();
        }

        const QString uvMethod = settings.value(QStringLiteral("uvMethod")).toString();
        if (!uvMethod.isEmpty())
        {
            config.uvMethod = uvMethod.toStdString();
        }
    }

    return config;
}

bool exportObjRequested(const QJsonObject &settings)
{
    return settings.value(QStringLiteral("export_format")).toString().trimmed().toUpper() == QStringLiteral("OBJ");
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
        if (orbital_workspace &&
            !request.settings.contains(QStringLiteral("tsdfSupportMaskFreeSpaceCarving")))
        {
            options.enableSupportMaskFreeSpaceCarving = true;
        }
        if (orbital_workspace &&
            !request.settings.contains(QStringLiteral("tsdfMinimumSupportMaskFreeSpaceViews")))
        {
            options.minimumSupportMaskFreeSpaceViews = 5;
        }
        options.progress = request.progress;
        const DepthTsdfResult tsdf = DepthTsdfSurfaceBuilder::build(loaded.frames, options);
        mergePayload(DepthTsdfSurfaceBuilder::statisticsToJson(tsdf), &result.payload);
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

        const QJsonObject diagnostics = result.payload;
        QVector<MeshColorView> texture_views;
        texture_views.reserve(loaded.frames.size());
        for (const DepthTsdfFrame &frame : loaded.frames)
        {
            MeshColorView view;
            view.camera = frame.camera;
            view.colorBgr = frame.colorBgr;
            view.depth = frame.depth;
            view.confidence = frame.confidence;
            view.depthValidMask = frame.depthValidMask;
            view.supportMask = frame.supportMask;
            view.qualityWeight = frame.frameQualityWeight;
            texture_views.push_back(std::move(view));
        }
        result = saveMeshAndOptionalTexture(tsdf.mesh,
                                            "depth_tsdf",
                                            output_root,
                                            request.exportObj,
                                            request.texture,
                                            request.progress,
                                            &texture_views);
        mergePayload(diagnostics, &result.payload);
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
        depth_request.exportObj = exportObjRequested(request.settings);
        depth_request.texture = defaultTextureConfig();
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
        mesh_request.exportObj = exportObjRequested(request.settings);
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
            MeshColorView view;
            view.camera = frame.camera;
            view.colorBgr = frame.colorBgr;
            view.depth = frame.depth;
            view.confidence = frame.confidence;
            view.depthValidMask = frame.depthValidMask;
            view.supportMask = frame.supportMask;
            view.qualityWeight = frame.frameQualityWeight;
            views.push_back(std::move(view));
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

    result.payload = textureResultToJson(textureResult);
    return result;
}

} // namespace xjw::mesh::workflow
