#include "ModelWorkflowService.h"

#include "DepthMapMeshBuilder.h"
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
    object[QStringLiteral("texture_size")] = result.textureSize;
    object[QStringLiteral("texture_algorithm")] = QString::fromStdString(result.textureAlgorithm);
    object[QStringLiteral("uv_method")] = QString::fromStdString(result.uvMethod);
    object[QStringLiteral("blend_method")] = QString::fromStdString(result.blendMethod);
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
                                          const std::function<void(const QString &, int)> &progress)
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
        if (xjw::mesh::TextureMapper::generateTexturedModelFromMeshFile(
                xjw::common::io::toUtf8Path(mesh_ply_path),
                xjw::common::io::toUtf8Path(products_dir),
                texture_config,
                &texture_result,
                &texture_error))
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

} // namespace

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
    DepthMapVisualHullResult visual_hull = DepthMapMeshBuilder::buildVisualHull(
        request.depthMapSourcePath,
        request.reconstruction.resolution,
        request.progress);
    if (visual_hull.applicable && visual_hull.ok)
    {
        result = saveMeshAndOptionalTexture(visual_hull.mesh,
                                            "depth_constrained_visual_hull",
                                            output_root,
                                            request.exportObj,
                                            request.texture,
                                            request.progress);
        if (result.ok)
        {
            result.payload[QStringLiteral("visual_hull_views")] = visual_hull.usableViewCount;
            result.payload[QStringLiteral("depth_map_source_path")] = request.depthMapSourcePath;
            result.payload[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
        }
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
        const auto frames = xjw::mesh::DepthMapMeshBuilder::discoverDepthFrames(request.depthMapSourcePath);
        if (frames.isEmpty())
        {
            result.errorMessage = QStringLiteral("未找到可用于生成模型的深度图文件");
            return result;
        }

        result.errorMessage = QStringLiteral(
            "缺少深度图 metadata，无法从 raw depth 直接恢复相机、尺寸和尺度；"
            "请重新运行深度图估计或先执行深度图融合。");
        return result;
    }

    MeshBuildRequest meshRequest;
    meshRequest.pointCloudPath = densePath;
    meshRequest.outputRoot = output_root;
    meshRequest.reconstruction = request.reconstruction;
    meshRequest.exportObj = request.exportObj;
    meshRequest.texture = request.texture;
    meshRequest.progress = request.progress;

    result = buildMeshAndOptionalTexture(meshRequest);
    if (result.ok)
    {
        result.payload[QStringLiteral("depth_map_source_path")] = request.depthMapSourcePath;
        result.payload[QStringLiteral("source_point_cloud_path")] = densePath;
        result.payload[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
        if (visual_hull.applicable && !visual_hull.message.isEmpty())
        {
            result.payload[QStringLiteral("visual_hull_warning")] = visual_hull.message;
        }
    }
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
    result.ok = xjw::mesh::TextureMapper::generateTexturedModelFromMeshFile(
        xjw::common::io::toUtf8Path(request.meshPath),
        xjw::common::io::toUtf8Path(request.outputDir),
        textureConfig,
        &textureResult,
        &textureError);

    if (!result.ok)
    {
        result.errorMessage = QString::fromStdString(textureError);
        return result;
    }

    result.payload = textureResultToJson(textureResult);
    return result;
}

} // namespace xjw::mesh::workflow
