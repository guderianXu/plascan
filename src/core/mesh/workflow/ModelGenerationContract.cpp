#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow::workflow_detail
{
    using namespace workflow_detail;

    bool resolveModelGenerationContract(const QJsonObject& settings, ModelGenerationContract* contract, QString* error)
    {
        const auto fail = [error](const QString& message)
        {
            if (error)
                *error = message;
            return false;
        };
        if (!settings.value(QStringLiteral("modelGenerationContractRevision")).isDouble() ||
            settings.value(QStringLiteral("modelGenerationContractRevision")).toInt(-1) != 2)
        {
            return fail(
                QStringLiteral("模型生成需要 modelGenerationContractRevision=2；旧项目请重新确认生成模型参数。"));
        }
        const QString depth_profile = settings.value(QStringLiteral("depthQualityProfile")).toString();
        if (depth_profile != QStringLiteral("highest") && depth_profile != QStringLiteral("high") &&
            depth_profile != QStringLiteral("medium") && depth_profile != QStringLiteral("low") &&
            depth_profile != QStringLiteral("lowest"))
        {
            return fail(QStringLiteral("模型生成质量必须为 lowest、low、medium、high 或 highest。"));
        }
        const QString interpolation = settings.value(QStringLiteral("interpolation")).toString();
        if (interpolation != QStringLiteral("disabled") && interpolation != QStringLiteral("enabled") &&
            interpolation != QStringLiteral("extrapolated"))
        {
            return fail(QStringLiteral("模型插值必须为 disabled、enabled 或 extrapolated。"));
        }
        const QString surface_profile = settings.value(QStringLiteral("surfaceQualityProfile")).toString();
        if (surface_profile != QStringLiteral("recovered_ooc") &&
            surface_profile != QStringLiteral("rpc_height_plane_sweep"))
        {
            return fail(
                QStringLiteral("模型生成需要显式 surfaceQualityProfile=recovered_ooc 或 rpc_height_plane_sweep；旧 "
                               "TSDF/Poisson/Visual Hull 模式不可作为产品入口。"));
        }
        const QString mode = settings.value(QStringLiteral("faceCountMode")).toString();
        int target_faces = 0;
        if (mode == QStringLiteral("low") || mode == QStringLiteral("medium") || mode == QStringLiteral("high"))
        {
            // The three reference presets terminate QEM by an exact score
            // threshold. Only Custom has a numeric face-count target.
            target_faces = 0;
        }
        else if (mode == QStringLiteral("custom"))
        {
            if (!settings.value(QStringLiteral("faceCountCustom")).isDouble())
            {
                return fail(QStringLiteral("faceCountMode=custom 需要整数 faceCountCustom（1..2000000）。"));
            }
            target_faces = settings.value(QStringLiteral("faceCountCustom")).toInt(-1);
            if (target_faces < 1 || target_faces > 2000000)
            {
                return fail(QStringLiteral("faceCountCustom 必须在 1..2000000。"));
            }
        }
        else
        {
            return fail(QStringLiteral("模型面数档位必须为 low、medium、high 或 custom。"));
        }
        const QString legacy_mode = settings.value(QStringLiteral("reconstruction_mode")).toString();
        if (!legacy_mode.isEmpty() && legacy_mode != surface_profile)
        {
            return fail(QStringLiteral("reconstruction_mode 与 surfaceQualityProfile 不一致；请迁移旧项目设置。"));
        }
        contract->surfaceProfile = surface_profile;
        contract->targetFaces = target_faces;
        contract->requested = {{QStringLiteral("modelGenerationContractRevision"),
                                settings.value(QStringLiteral("modelGenerationContractRevision"))},
                               {QStringLiteral("depthQualityProfile"), depth_profile},
                               {QStringLiteral("surfaceQualityProfile"), surface_profile},
                               {QStringLiteral("faceCountMode"), mode},
                               {QStringLiteral("interpolation"), interpolation},
                               {QStringLiteral("requestedTargetFaces"), target_faces}};
        contract->effective = {{QStringLiteral("modelGenerationContractRevision"), 2},
                               {QStringLiteral("depthQualityProfile"), depth_profile},
                               {QStringLiteral("surfaceQualityProfile"), surface_profile},
                               {QStringLiteral("faceCountMode"), mode},
                               {QStringLiteral("interpolation"), interpolation},
                               {QStringLiteral("effectiveTargetFaces"), target_faces}};
        return true;
    }
} // namespace xjw::mesh::workflow::workflow_detail
