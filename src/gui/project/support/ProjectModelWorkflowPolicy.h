#pragma once

#include <QJsonObject>
#include <QString>
#include <QtGlobal>

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

inline int recommendedInteractiveModelWorkerCount(int ideal_thread_count)
{
    const int available_threads = ideal_thread_count > 0
        ? ideal_thread_count
        : 4;
    // Reserve two logical processors for the GUI and operating system while
    // allowing model generation to use the rest of the machine by default.
    return qMax(1, available_threads - 2);
}

QString projectDepthInputSignature(const QJsonObject &project_metadata,
                                   int aerial_triangulation_result_index = -1);

StoredDepthBatchCompatibility assessStoredDepthBatchCompatibility(
    const QJsonObject &project_metadata,
    const QString &depth_map_source_path = QString(),
    int aerial_triangulation_result_index = -1,
    const QString &expected_scene_profile = QString());

} // namespace xjw::gui::project
