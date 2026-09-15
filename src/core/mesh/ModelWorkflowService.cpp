#include "workflow/ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;

    WorkflowResult buildModel(const ModelBuildRequest& input_request)
    {
        const auto request = bindWorkflowCallbacks(input_request);
        if (cancellationRequested(request.isCancelled))
        {
            return cancelledWorkflowResult();
        }
        const auto started_at = std::chrono::steady_clock::now();
        const QString source_data =
            request.sourceData.trimmed().isEmpty() ? QStringLiteral("point_cloud") : request.sourceData.trimmed();
        ModelGenerationContract contract;
        QString contract_error;
        if (!resolveModelGenerationContract(request.settings, &contract, &contract_error))
        {
            WorkflowResult rejected;
            rejected.errorMessage = contract_error;
            rejected.payload[QStringLiteral("fallback")] = QStringLiteral("none");
            return rejected;
        }
        if ((contract.surfaceProfile == QStringLiteral("recovered_ooc") &&
             source_data != QStringLiteral("depth_maps")) ||
            (contract.surfaceProfile == QStringLiteral("rpc_height_plane_sweep") &&
             source_data != QStringLiteral("rpc_height_plane_sweep")))
        {
            WorkflowResult rejected;
            rejected.errorMessage =
                QStringLiteral("sourceData 与 surfaceQualityProfile 不匹配；recovered_ooc 必须使用 "
                               "depth_maps，rpc_height_plane_sweep 必须使用 rpc_height_plane_sweep。 ");
            rejected.payload[QStringLiteral("fallback")] = QStringLiteral("none");
            return rejected;
        }
        ReconstructionConfig reconstruction = reconstructionConfigFromModelSettings(request.settings);
        reconstruction.simplifyTargetFaces = contract.targetFaces;

        WorkflowResult result;
        const ModelOutputPolicy outputPolicy =
            request.outputPolicy.value_or(modelOutputPolicyFromSettings(request.settings));
        const QString baseOutputRoot = QDir::cleanPath(request.outputRoot.trimmed());
        QString runId;
        QString effectiveOutputRoot;
        if (!createModelRunOutputDirectory(
                baseOutputRoot, request.runId, &runId, &effectiveOutputRoot, &result.errorMessage))
        {
            return result;
        }

        bool ownsUnpublishedRun = true;
        const auto cleanupRun = [&]()
        {
            QString cleanupError;
            if (!ownsUnpublishedRun ||
                removeUnpublishedModelRunDirectory(baseOutputRoot, runId, effectiveOutputRoot, &cleanupError))
            {
                ownsUnpublishedRun = false;
                return QString();
            }
            return cleanupError;
        };
        const auto cleanupGuard = qScopeGuard([&cleanupRun]() { cleanupRun(); });
        const auto discardRun = [&cleanupRun](WorkflowResult failedResult)
        {
            const QString cleanupError = cleanupRun();
            if (!cleanupError.isEmpty())
            {
                failedResult.payload[QStringLiteral("run_cleanup_failed")] = true;
                failedResult.payload[QStringLiteral("run_cleanup_error")] = cleanupError;
                if (failedResult.errorMessage.isEmpty())
                {
                    failedResult.errorMessage =
                        QStringLiteral("模型运行失败，且未能清理隔离目录：%1").arg(cleanupError);
                }
                else
                {
                    failedResult.errorMessage += QStringLiteral("；未能清理隔离目录：%1").arg(cleanupError);
                }
            }
            return failedResult;
        };
        if (cancellationRequested(request.isCancelled))
        {
            return discardRun(cancelledWorkflowResult());
        }
        try
        {
            if (source_data == QStringLiteral("depth_maps"))
            {
                DepthMapMeshBuildRequest depth_request;
                depth_request.depthMapSourcePath = request.depthMapSourcePath.trimmed().isEmpty()
                                                       ? request.requestedSourcePath
                                                       : request.depthMapSourcePath;
                depth_request.reusableDenseCloudPath = request.sourcePointCloudPath;
                depth_request.sparseScaffoldPointCloudPath = request.sparseScaffoldPointCloudPath;
                depth_request.sparseScaffoldPointsPath = request.sparseScaffoldPointsPath;
                depth_request.outputRoot = effectiveOutputRoot;
                depth_request.settings = request.settings;
                depth_request.settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("recovered_ooc");
                depth_request.reconstruction = reconstruction;
                depth_request.exportObj = false;
                depth_request.texture = defaultTextureConfig();
                depth_request.texture.isCancelled = request.isCancelled;
                depth_request.execution = request.execution;
                result = buildMeshFromDepthMaps(depth_request);
            }
            else if (source_data == QStringLiteral("rpc_height_plane_sweep"))
            {
                const QString face_count_mode =
                    request.settings.value(QStringLiteral("faceCountMode")).toString(QStringLiteral("high"));
                const int rpc_target_faces = face_count_mode == QStringLiteral("low")      ? 20000
                                             : face_count_mode == QStringLiteral("medium") ? 100000
                                             : face_count_mode == QStringLiteral("high")   ? 200000
                                                                                           : contract.targetFaces;
                const auto rpc_model =
                    buildRpcPlaneSweepModel(request.settings, rpc_target_faces, request.isCancelled, request.progress);
                result = saveMeshAndOptionalTexture(rpc_model.mesh,
                                                    "rpc_height_plane_sweep",
                                                    effectiveOutputRoot,
                                                    false,
                                                    defaultTextureConfig(),
                                                    request.progress,
                                                    request.isCancelled);
                mergePayload(rpc_model.diagnostics, &result.payload);
            }
            else
            {
                result.errorMessage = QStringLiteral(
                    "模型产品入口不再接受点云/旧模式；请显式选择 recovered_ooc 或 rpc_height_plane_sweep。 ");
            }
        }
        catch (const std::exception& exception)
        {
            result.errorMessage = QStringLiteral("模型生成异常: %1").arg(QString::fromUtf8(exception.what()));
        }
        catch (...)
        {
            result.errorMessage = QStringLiteral("模型生成发生未知异常");
        }

        result.payload[QStringLiteral("model_core_elapsed_ms")] = static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at)
                .count());
        result.payload[QStringLiteral("requested_model_generation_contract")] = contract.requested;
        result.payload[QStringLiteral("effective_model_generation_contract")] = contract.effective;
        result.payload[QStringLiteral("face_count_mode")] = request.settings.value(QStringLiteral("faceCountMode"));
        result.payload[QStringLiteral("requested_target_faces")] = contract.targetFaces;
        result.payload[QStringLiteral("effective_target_faces")] = contract.targetFaces;
        // Legacy consumers use these shorter names.  They retain the same target
        // semantics as their explicit counterparts above.
        result.payload[QStringLiteral("requested_face_count")] = contract.targetFaces;
        result.payload[QStringLiteral("effective_face_count")] = contract.targetFaces;
        if (cancellationRequested(request.isCancelled))
        {
            return discardRun(cancelledWorkflowResult());
        }
        if (result.ok)
        {
            result.payload[QStringLiteral("actual_output_face_count")] =
                result.payload.value(QStringLiteral("face_count"));
            result.payload[QStringLiteral("source_data")] = source_data;
            result.payload[QStringLiteral("source_path")] = request.requestedSourcePath;
            result.payload[QStringLiteral("source_point_cloud_path")] = request.sourcePointCloudPath;
            if (source_data == QStringLiteral("depth_maps"))
            {
                result.payload[QStringLiteral("depth_map_source_path")] = request.depthMapSourcePath.trimmed().isEmpty()
                                                                              ? request.requestedSourcePath
                                                                              : request.depthMapSourcePath;
            }
            finalizeModelRun(&result, outputPolicy, runId, effectiveOutputRoot);
            if (cancellationRequested(request.isCancelled))
            {
                return discardRun(cancelledWorkflowResult());
            }
        }

        if (!result.ok)
        {
            return discardRun(result);
        }

        ownsUnpublishedRun = false;
        return result;
    }
} // namespace xjw::mesh::workflow
