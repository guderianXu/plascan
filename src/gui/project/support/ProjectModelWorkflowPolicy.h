#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::gui::project
{

inline constexpr int kDenseFusionPipelineVersion = 2;
inline constexpr int kProjectDepthInputSignatureVersion = 2;

struct StoredDepthBatchCompatibility
{
    bool compatible = false;
    int frameCount = 0;
    QString reason;
};

int recommendedInteractiveModelWorkerCount(int idealThreadCount);

QString projectDepthInputSignature(const QJsonObject &project_metadata,
                                   int aerial_triangulation_result_index = -1);

StoredDepthBatchCompatibility assessStoredDepthBatchCompatibility(
    const QJsonObject &project_metadata,
    const QString &depth_map_source_path = QString(),
    int aerial_triangulation_result_index = -1);

} // namespace xjw::gui::project
