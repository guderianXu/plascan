#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;

    WorkflowResult buildMeshFromDepthMaps(const DepthMapMeshBuildRequest& input_request)
    {
        const auto request = bindWorkflowCallbacks(input_request);
        if (cancellationRequested(request.isCancelled))
        {
            return cancelledWorkflowResult();
        }
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
        const QString output_root =
            request.outputRoot.isEmpty()
                ? (depth_source_info.isDir() ? depth_source_info.absoluteFilePath() : depth_source_info.absolutePath())
                : request.outputRoot;

        const QDir depth_directory(depth_source_info.isDir() ? depth_source_info.absoluteFilePath()
                                                             : depth_source_info.absolutePath());
        const QString recovered_input = depth_directory.filePath(QStringLiteral("recovered_model_input"));
        if (mode != QStringLiteral("recovered_ooc"))
        {
            result.errorMessage = QStringLiteral("深度图模型产品入口仅接受显式 recovered_ooc；旧 TSDF、Poisson、Visual "
                                                 "Hull 与稀疏 DEM 模式不会自动回退。请迁移项目设置。");
            return result;
        }
        if (request.exportObj)
        {
            result.errorMessage = QStringLiteral(
                "参考 recovered 生产链尚未实现 UV/纹理导出；请选择 PLY。不会自动混用旧 PlaScan 纹理算法。");
            return result;
        }
        if (!QFileInfo::exists(recovered_input))
        {
            result.errorMessage = QStringLiteral("recovered_ooc 需要有效的 recovered_model_input；不会回退到 "
                                                 "TSDF、Poisson、Visual Hull 或稀疏 DEM。请重新生成 recovered 深度。 ");
            return result;
        }
        return buildRecoveredDepthModel(request, std::move(result), output_root, recovered_input);
    }

} // namespace xjw::mesh::workflow
