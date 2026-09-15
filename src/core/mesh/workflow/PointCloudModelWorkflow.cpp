#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;

    WorkflowResult buildMeshAndOptionalTexture(const MeshBuildRequest& input_request)
    {
        const auto request = bindWorkflowCallbacks(input_request);
        if (cancellationRequested(request.isCancelled))
        {
            return cancelledWorkflowResult();
        }
        WorkflowResult result;

        if (request.pointCloudPath.trimmed().isEmpty())
        {
            result.errorMessage = QStringLiteral("点云路径为空");
            return result;
        }

        xjw::mesh::ReconstructionConfig reconstruction = request.reconstruction;
        reconstruction.isCancelled = request.isCancelled;
        if (request.progress)
        {
            reconstruction.progressFn = [cb = request.progress](const std::string& stage, float fraction)
            { cb(QString::fromStdString(stage), static_cast<int>(fraction * 100.0f)); };
        }

        xjw::mesh::TriMesh mesh;
        std::string meshError;
        std::string meshAlgorithm;
        if (!xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(
                xjw::common::io::toUtf8Path(request.pointCloudPath), reconstruction, mesh, &meshError, &meshAlgorithm))
        {
            if (cancellationRequested(request.isCancelled))
            {
                return cancelledWorkflowResult();
            }
            result.errorMessage = QStringLiteral("网格重建失败: %1").arg(QString::fromStdString(meshError));
            return result;
        }
        if (cancellationRequested(request.isCancelled))
        {
            return cancelledWorkflowResult();
        }

        return saveMeshAndOptionalTexture(mesh,
                                          meshAlgorithm,
                                          request.outputRoot,
                                          request.exportObj,
                                          request.texture,
                                          request.progress,
                                          request.isCancelled);
    }
} // namespace xjw::mesh::workflow
