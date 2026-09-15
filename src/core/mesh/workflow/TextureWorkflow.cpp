#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;

    WorkflowResult buildTextureOnly(const TextureBuildRequest& input_request)
    {
        const auto request = bindWorkflowCallbacks(input_request);
        if (cancellationRequested(request.isCancelled))
        {
            return cancelledWorkflowResult();
        }
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
            textureConfig.progressFn = [cb = request.progress](const std::string& stage, int percent)
            { cb(QString::fromStdString(stage), percent); };
        }

        xjw::mesh::TextureMappingResult textureResult;
        std::string textureError;
        QJsonObject texture_source_diagnostics;
        if (!request.depthMapSourcePath.trimmed().isEmpty())
        {
            if (cancellationRequested(request.isCancelled))
            {
                return cancelledWorkflowResult();
            }
            const QVector<DepthFrameArtifact> artifacts =
                DepthMapMeshBuilder::discoverDepthFrames(request.depthMapSourcePath);
            const bool has_primary_frame = std::any_of(artifacts.cbegin(),
                                                       artifacts.cend(),
                                                       [](const DepthFrameArtifact& artifact)
                                                       { return xjw::mvs::isPrimaryFusionFrame(artifact.role); });
            if (!has_primary_frame)
            {
                result.errorMessage = QStringLiteral("无法加载相机纹理源：没有可用的主融合深度帧；"
                                                     "辅助验证帧不能独立发布纹理结果。");
                return result;
            }
            const DepthTsdfFrameLoadResult loaded = DepthTsdfSurfaceBuilder::loadFrames(artifacts);
            if (cancellationRequested(request.isCancelled))
            {
                return cancelledWorkflowResult();
            }
            if (!loaded.ok)
            {
                result.errorMessage = QStringLiteral("无法加载相机纹理源: %1").arg(loaded.errorMessage);
                return result;
            }
            if (loaded.primaryFrameCount == 0)
            {
                result.errorMessage = QStringLiteral("无法加载相机纹理源：实际载入的主融合深度帧为 0；"
                                                     "辅助验证帧不能独立发布纹理结果。");
                return result;
            }
            QVector<MeshColorView> views;
            views.reserve(loaded.frames.size());
            for (const DepthTsdfFrame& frame : loaded.frames)
            {
                views.push_back(textureViewFromFrame(frame));
            }
            QString texture_mesh_path = request.meshPath;
            QString temporary_colored_mesh_path;
            if (QFileInfo(request.meshPath).suffix().compare(QStringLiteral("ply"), Qt::CaseInsensitive) == 0)
            {
                TriMesh source_mesh;
                std::string source_mesh_error;
                if (TriMesh::loadPLY(xjw::common::io::toUtf8Path(request.meshPath), &source_mesh, &source_mesh_error) &&
                    !source_mesh.hasVertexColors)
                {
                    if (request.progress)
                    {
                        request.progress(QStringLiteral("正在为纹理未覆盖区域计算顶点颜色..."), 2);
                    }
                    QVector<MeshColorView> color_views;
                    color_views.reserve(loaded.frames.size());
                    for (const DepthTsdfFrame& frame : loaded.frames)
                    {
                        color_views.push_back(vertexColorViewFromFrame(frame));
                    }
                    MeshColorOptions color_options;
                    color_options.minimumConfidence = textureConfig.minimumConfidence;
                    const MeshColorStatistics color_statistics =
                        MeshColorizer::colorize(&source_mesh, color_views, color_options);
                    texture_source_diagnostics[QStringLiteral("texture_source_recolorized")] =
                        source_mesh.hasVertexColors;
                    texture_source_diagnostics[QStringLiteral("texture_source_reliably_colored_vertex_count")] =
                        color_statistics.reliablyColoredVertexCount;
                    texture_source_diagnostics[QStringLiteral("texture_source_fallback_color_vertex_count")] =
                        color_statistics.fallbackVertexCount;
                    if (source_mesh.hasVertexColors)
                    {
                        QDir().mkpath(request.outputDir);
                        temporary_colored_mesh_path =
                            QDir(request.outputDir).filePath(QStringLiteral(".texture_source_colored.ply"));
                        std::string save_error;
                        if (!source_mesh.savePLY(xjw::common::io::toUtf8Path(temporary_colored_mesh_path), &save_error))
                        {
                            QFile::remove(temporary_colored_mesh_path);
                            result.errorMessage = QStringLiteral("无法准备纹理未覆盖区域的顶点颜色: %1")
                                                      .arg(QString::fromStdString(save_error));
                            return result;
                        }
                        texture_mesh_path = temporary_colored_mesh_path;
                    }
                }
            }
            result.ok = xjw::mesh::TextureMapper::generateCameraTexturedModelFromMeshFile(
                xjw::common::io::toUtf8Path(texture_mesh_path),
                xjw::common::io::toUtf8Path(request.outputDir),
                textureConfig,
                views,
                &textureResult,
                &textureError);
            if (!temporary_colored_mesh_path.isEmpty())
            {
                QFile::remove(temporary_colored_mesh_path);
            }
        }
        else
        {
            if (!request.allowVertexColorFallback)
            {
                result.errorMessage = QStringLiteral("当前模型没有可用的深度图与相机证据，已停止多视图纹理生成；"
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

        if (cancellationRequested(request.isCancelled))
        {
            return cancelledWorkflowResult();
        }

        if (!result.ok)
        {
            result.errorMessage = QString::fromStdString(textureError);
            return result;
        }

        result.payload = textureResultToJson(textureResult, &textureConfig);
        mergePayload(texture_source_diagnostics, &result.payload);
        result.payload[QStringLiteral("source_model_path")] = request.meshPath;
        if (!request.textureRunId.trimmed().isEmpty())
        {
            if (cancellationRequested(request.isCancelled))
            {
                return cancelledWorkflowResult();
            }
            finalizeTextureRun(&result, request.textureRunId.trimmed(), request.outputDir);
            if (cancellationRequested(request.isCancelled))
            {
                return cancelledWorkflowResult();
            }
        }
        return result;
    }
} // namespace xjw::mesh::workflow
