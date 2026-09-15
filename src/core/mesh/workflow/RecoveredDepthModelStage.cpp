#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;
    WorkflowResult workflow_detail::buildRecoveredDepthModel(const BoundWorkflowRequest<DepthMapMeshBuildRequest>& request,
                                                             WorkflowResult result,
                                                             const QString& output_root,
                                                             const QString& recovered_input)
    {
        result.payload[QStringLiteral("actual_mesh_algorithm")] = QStringLiteral("recovered_ooc");
        try
        {
            const auto artifacts = DepthMapMeshBuilder::discoverDepthFrames(request.depthMapSourcePath);
            if (artifacts.isEmpty() || std::any_of(artifacts.begin(),
                                                   artifacts.end(),
                                                   [](const auto& artifact)
                                                   {
                                                       return artifact.status != QStringLiteral("completed") ||
                                                              !xjw::mvs::isPrimaryFusionFrame(artifact.role);
                                                   }))
            {
                result.errorMessage =
                    QStringLiteral("Recovered 模型需要完整、已完成的主深度帧批次，请重新生成深度图。");
                return result;
            }
            QJsonObject settings = request.settings;
            settings[QStringLiteral("interpolation")] = QStringLiteral("enabled");
            settings[QStringLiteral("strictVolumetricMasks")] = false;
            settings[QStringLiteral("splitIntoBlocks")] = false;
            settings[QStringLiteral("recovered_expected_camera_count")] = artifacts.size();
            QJsonArray expected_poses;
            QJsonArray source_images;
            for (qsizetype index = 0; index < artifacts.size(); ++index)
            {
                expected_poses.append(QJsonValue());
                source_images.append(QJsonValue());
            }
            for (const auto& artifact : artifacts)
            {
                if (!artifact.hasCameraModel || artifact.refIndex < 0 || artifact.refIndex >= artifacts.size() ||
                    !expected_poses[artifact.refIndex].isNull())
                {
                    result.errorMessage = QStringLiteral("Recovered 模型深度帧的相机索引不完整或重复。");
                    return result;
                }
                QJsonArray pose;
                for (const auto value : artifact.cameraModel.worldToCameraRotation())
                {
                    pose.append(value);
                }
                for (const auto value : artifact.cameraModel.worldToCameraTranslation())
                {
                    pose.append(value);
                }
                expected_poses[artifact.refIndex] = pose;
                source_images[artifact.refIndex] = artifact.sourceImage;
            }
            settings[QStringLiteral("recovered_expected_camera_poses")] = expected_poses;
            settings[QStringLiteral("recovered_source_images")] = source_images;
            auto recovered = buildRecoveredModel(recovered_input,
                                                 settings,
                                                 request.reconstruction.simplifyTargetFaces,
                                                 request.isCancelled,
                                                 request.progress);
            recovered.diagnostics[QStringLiteral("configured_interpolation")] =
                request.settings.value(QStringLiteral("interpolation")).toString(QStringLiteral("enabled"));
            recovered.diagnostics[QStringLiteral("effective_interpolation")] = QStringLiteral("enabled");
            recovered.diagnostics[QStringLiteral("effective_strict_volumetric_masks")] = false;
            recovered.diagnostics[QStringLiteral("effective_split_into_blocks")] = false;
            if (cancellationRequested(request.isCancelled))
            {
                return cancelledWorkflowResult();
            }
            result = saveMeshAndOptionalTexture(recovered.mesh,
                                                "recovered_ooc",
                                                output_root,
                                                request.exportObj,
                                                request.texture,
                                                request.progress,
                                                request.isCancelled,
                                                nullptr,
                                                &recovered);
            mergePayload(recovered.diagnostics, &result.payload);
            result.payload[QStringLiteral("reconstruction_mode")] = QStringLiteral("recovered_ooc");
            result.payload[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
            result.payload[QStringLiteral("depth_map_source_path")] = request.depthMapSourcePath;
            return result;
        }
        catch (const std::exception& exception)
        {
            if (cancellationRequested(request.isCancelled))
            {
                return cancelledWorkflowResult();
            }
            result.errorMessage = QStringLiteral("Recovered 模型生成失败：%1").arg(QString::fromUtf8(exception.what()));
            return result;
        }
    }

} // namespace xjw::mesh::workflow
