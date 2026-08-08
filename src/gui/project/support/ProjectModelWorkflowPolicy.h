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
    // QThread reports logical processors. On desktop CPUs with many logical
    // processors, use the conservative physical-core estimate and reserve two
    // cores for image decoding, the GUI and the operating system.
    const int compute_threads = available_threads >= 16
        ? available_threads / 2
        : available_threads;
    return qBound(1, compute_threads - 2, 32);
}

QString projectDepthInputSignature(const QJsonObject &project_metadata,
                                   int aerial_triangulation_result_index = -1);

StoredDepthBatchCompatibility assessStoredDepthBatchCompatibility(
    const QJsonObject &project_metadata,
    const QString &depth_map_source_path = QString(),
    int aerial_triangulation_result_index = -1);

} // namespace xjw::gui::project
