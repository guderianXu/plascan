/**
 * @file ProjectConfigManager.cpp
 * @brief ProjectConfigManager 的实现文件。
 *
 * 管理相机模型策略与 workflow 段；新建项目生成默认值，读取已有项目只校验、不迁移。
 */
#include "project/ProjectConfigManager.h"

namespace
{

constexpr auto CameraModelPolicyKey = "camera_model_policy";
constexpr auto FramePinholeToken = "frame_pinhole";
constexpr auto IsisUsgsCsmLineScanToken = "isis_usgscsm_linescan";

}

QString projectCameraModelPolicyToken(ProjectCameraModelPolicy policy)
{
    switch (policy)
    {
    case ProjectCameraModelPolicy::FramePinhole:
        return QString::fromLatin1(FramePinholeToken);
    case ProjectCameraModelPolicy::IsisUsgsCsmLineScan:
        return QString::fromLatin1(IsisUsgsCsmLineScanToken);
    }

    return {};
}

std::optional<ProjectCameraModelPolicy> parseProjectCameraModelPolicy(
    const QString &token)
{
    const QString normalized_token = token.trimmed();
    if (normalized_token == QString::fromLatin1(FramePinholeToken))
    {
        return ProjectCameraModelPolicy::FramePinhole;
    }
    if (normalized_token == QString::fromLatin1(IsisUsgsCsmLineScanToken))
    {
        return ProjectCameraModelPolicy::IsisUsgsCsmLineScan;
    }

    return std::nullopt;
}

/**
 * @brief 生成含有所有默认值的标准项目配置对象。
 *
 * 通过调用各子管理器的 default*Settings() 静态方法，
 * 将相机模型策略和 "workflow" 段聚合为完整配置，
 * 确保新建项目无需任何手动配置即可正常运行。
 *
 * @return 包含 "camera_model_policy" 和 "workflow" 的默认配置对象。
 */
QJsonObject ProjectConfigManager::defaultConfig()
{
    QJsonObject config;

    config[QString::fromLatin1(CameraModelPolicyKey)] =
        projectCameraModelPolicyToken(ProjectCameraModelPolicy::FramePinhole);
    config["workflow"] = ProjectWorkflowConfigManager::defaultWorkflowSettings();

    return config;
}

bool ProjectConfigManager::validateCurrentConfig(const QJsonObject& input, QString* errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    const auto fail = [errorMessage](const QString& field)
    {
        if (errorMessage)
        {
            *errorMessage =
                QStringLiteral("项目配置字段 %1 缺失或无效；不再补全旧项目配置，请新建工程并重新导入数据。").arg(field);
        }
        return false;
    };
    if (input.contains(QStringLiteral("ui")))
    {
        return fail(QStringLiteral("ui（视图状态应位于根 ui_state）"));
    }
    if (!input.value(QStringLiteral("camera_model_policy")).isString() ||
        !parseProjectCameraModelPolicy(input.value(QStringLiteral("camera_model_policy")).toString()))
    {
        return fail(QStringLiteral("camera_model_policy"));
    }
    if (!input.value(QStringLiteral("workflow")).isObject())
    {
        return fail(QStringLiteral("workflow"));
    }
    const auto workflow = input.value(QStringLiteral("workflow")).toObject();
    for (const auto& step : {QStringLiteral("bundle_adjust"), QStringLiteral("dem"), QStringLiteral("ortho")})
    {
        if (!workflow.value(step).isObject())
        {
            return fail(QStringLiteral("workflow.%1").arg(step));
        }
    }
    return validateBundleAdjustSettings(workflow.value(QStringLiteral("bundle_adjust")).toObject(), errorMessage);
}

bool ProjectConfigManager::validateBundleAdjustSettings(const QJsonObject& input, QString* errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    const auto fail = [errorMessage](const QString& field)
    {
        if (errorMessage)
        {
            *errorMessage =
                QStringLiteral("光束法平差配置 %1 已删除或无效，请使用当前 PlaMatrix 后端及参数。").arg(field);
        }
        return false;
    };
    for (const auto* field : {"max_point_iterations",
                              "max_camera_iterations",
                              "huber_delta",
                              "finite_diff_eps",
                              "damping",
                              "step_tolerance",
                              "ba_max_dense_schur_cameras",
                              "ba_compare_auto_backend_with_legacy"})
    {
        if (input.contains(QString::fromLatin1(field)))
        {
            return fail(QString::fromLatin1(field));
        }
    }
    const auto backend = input.value(QStringLiteral("ba_backend"));
    if (!backend.isUndefined())
    {
        const QString name = backend.toString().trimmed().toLower();
        if (!backend.isString() ||
            (name != QLatin1String("auto") && name != QLatin1String("plamatrix_cpu") &&
             name != QLatin1String("plamatrix_cuda") && name != QLatin1String("plamatrix_opencl")))
        {
            return fail(QStringLiteral("ba_backend=%1").arg(name));
        }
    }
    return true;
}

std::optional<ProjectCameraModelPolicy>
ProjectConfigManager::cameraModelPolicy() const
{
    return parseProjectCameraModelPolicy(
        _config.value(QString::fromLatin1(CameraModelPolicyKey)).toString());
}

void ProjectConfigManager::setCameraModelPolicy(
    ProjectCameraModelPolicy policy)
{
    _config[QString::fromLatin1(CameraModelPolicyKey)] =
        projectCameraModelPolicyToken(policy);
}

/**
 * @brief 获取指定工作流步骤的参数配置。
 *
 * @param step  步骤名称（例如 "bundle_adjust"、"dem"、"ortho"）。
 * @return      该步骤对应的参数对象；不存在时返回空 QJsonObject。
 */
QJsonObject ProjectConfigManager::workflowSettings(const QString &step) const
{
    // 临时创建子管理器，加载 "workflow" 段后查询指定步骤
    ProjectWorkflowConfigManager workflowManager;
    workflowManager.setData(_config.value("workflow").toObject());
    return workflowManager.settings(step);
}

/**
 * @brief 更新指定工作流步骤的参数配置（深度合并补丁）。
 *
 * @param step     步骤名称。
 * @param settings 仅含需要修改字段的 JSON 补丁对象。
 */
void ProjectConfigManager::setWorkflowSettings(const QString &step, const QJsonObject &settings)
{
    // 临时创建子管理器，加载 "workflow" 段后更新指定步骤
    ProjectWorkflowConfigManager workflowManager;
    workflowManager.setData(_config.value("workflow").toObject());
    workflowManager.setSettings(step, settings);
    // 将更新后的 "workflow" 写回完整配置
    _config["workflow"] = workflowManager.data();
}
