#pragma once

#include <QJsonObject>
#include <QString>
#include <QtGlobal>

namespace xjw::gui::project
{

inline constexpr int kDenseFusionPipelineVersion = 2;

struct StoredDepthBatchRequirements
{
    int minimumPrimaryFrameCount = 2;
    int minimumUsableFrameCount = 2;
};

inline constexpr StoredDepthBatchRequirements kPointCloudDepthBatchRequirements{
    2, 2};
inline constexpr StoredDepthBatchRequirements kDepthTsdfDepthBatchRequirements{
    1, 3};
inline constexpr StoredDepthBatchRequirements kVisualHullDepthBatchRequirements{
    6, 6};

struct StoredDepthBatchCompatibility
{
    bool compatible = false;
    int frameCount = 0;
    int usableFrameCount = 0;
    int primaryFrameCount = 0;
    QString reason;
};

struct SparseScaffoldSource
{
    QString pointCloudPath;
    QString pointsJsonPath;
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

StoredDepthBatchRequirements depthBatchRequirementsForModelSettings(
    const QJsonObject &settings);

StoredDepthBatchCompatibility assessStoredDepthBatchCompatibility(
    const QJsonObject &project_metadata,
    const QString &depth_map_source_path = QString(),
    int aerial_triangulation_result_index = -1,
    const QString &expected_scene_profile = QString(),
    bool allow_orbital_sparse_scaffold_fallback = false,
    StoredDepthBatchRequirements requirements =
        kPointCloudDepthBatchRequirements);

SparseScaffoldSource resolveSparseScaffoldSource(
    const QJsonObject &project_metadata,
    const QString &depth_map_source_path);

} // namespace xjw::gui::project
