#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::gui::project
{

inline constexpr int kDenseFusionPipelineVersion = 2;

enum class ModelWorkflowAction
{
    RunMeshDirectly,
    GenerateDepthMapsThenMesh
};

struct ModelWorkflowDecision
{
    ModelWorkflowAction action = ModelWorkflowAction::RunMeshDirectly;
    QJsonObject modelSettings;
    QJsonObject depthSettings;
    QString depthMapSourcePath;
    QString reason;
};

int recommendedInteractiveModelWorkerCount(int idealThreadCount);

ModelWorkflowDecision decideModelGenerationWorkflow(const QJsonObject &settings,
                                                     const QJsonObject &project_metadata);

QJsonObject denseSettingsFromModelSettings(const QJsonObject &settings,
                                             const QString &depth_map_source_path);

QString projectDepthInputSignature(const QJsonObject &project_metadata,
                                   int aerial_triangulation_result_index = -1);

} // namespace xjw::gui::project
