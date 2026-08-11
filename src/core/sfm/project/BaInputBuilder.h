#pragma once

/**
 * @file BaInputBuilder.h
 * @brief 工程级 BA 输入聚合器。
 *
 * 构建顺序固定为：读取当前相机/匹配 -> 合并自动连接点轨迹 -> 追加旧测量控制 ->
 * 追加完整标记系统。输出把求解输入、统计和结果回写绑定放在同一索引空间中。
 */

#include "FramePinholeCamera.h"
#include "BundleAdjust.h"
#include "model/MarkerSet.h"
#include "project/ProjectMatchInputReader.h"
#include "registration/ControlNetworkSolver.h"

#include <QHash>
#include <QJsonObject>
#include <QMap>
#include <QStringList>

#include <vector>

namespace xjw::core::project
{

enum class BaInputBuildStatus
{
    Ok, ///< 至少两台相机且存在可用自动/人工轨迹。
    NotEnoughCameras, ///< 当前集合不足两台可解析相机。
    NoTracks ///< 输入不可读或没有满足条件的轨迹。
};

/**
 * @brief 从工程构建出的 BA 输入及来源统计。
 *
 * `cameras`、`imagePathByIndex` 和所有 BAObservation::cameraIndex 共享索引；
 * `tracks` 与 Marker/ScaleBar binding 中的 trackIndex 共享索引。调用方若过滤或重排
 * 任一数组，必须同步更新绑定。
 */
struct BaInputBuildResult
{
    std::vector<xjw::FramePinholeCamera> cameras; ///< BA 输入相机。
    QStringList imagePathByIndex; ///< 相机索引到影像路径。
    QMap<QString, QJsonObject> beforeCamMeta; ///< 更新前相机 JSON 快照。
    std::vector<xjw::BATrack> tracks; ///< 自动连接点和人工控制轨迹。
    std::vector<xjw::BAScaleBarConstraint> scaleBarConstraints; ///< 跨 track 尺度约束。

    // 自动匹配轨迹统计。
    int indexedObservationCount = 0; ///< `.pimatch` 中带稳定特征索引的观测数。
    int multiViewTrackCount = 0;
    int rejectedConflictTrackCount = 0;
    ProjectMatchInputDiagnostics matchDiagnostics;

    // 旧 survey_control 输入统计。
    int surveyControlTrackCount = 0;
    int surveyControlObservationCount = 0;
    int rejectedSurveyControlPointCount = 0;
    int surveyScaleBarConstraintCount = 0;
    int rejectedSurveyScaleBarCount = 0;

    // 完整标记系统统计和控制网解。
    control_points::ControlNetworkResult markerControlNetwork;
    int markerControlTrackCount = 0;
    int markerCheckTrackCount = 0;
    int rejectedMarkerTrackCount = 0;
    int markerControlPointConstraintCount = 0;
    int markerControlScaleBarConstraintCount = 0;
    int markerCheckScaleBarCount = 0;
    int rejectedMarkerScaleBarCount = 0;
    struct MarkerTrackBinding
    {
        control_points::MarkerId markerId; ///< 工程标记稳定 ID。
        control_points::MarkerRole role = control_points::MarkerRole::TieMarker; ///< 控制/检查/连接标记。
        int trackIndex = -1; ///< tracks 中对应轨迹。
        std::array<double, 3> referencePoint{{0.0, 0.0, 0.0}}; ///< 物方参考坐标。
        std::array<double, 3> sigma{{1.0, 1.0, 1.0}}; ///< 各轴先验标准差。
        bool usedAsConstraint = false; ///< 仅控制网内点控制点为 true。
    };
    struct MarkerScaleBarBinding
    {
        control_points::ScaleBarId scaleBarId; ///< 工程标尺稳定 ID。
        control_points::ScaleBarRole role = control_points::ScaleBarRole::Control; ///< 控制或检查标尺。
        int trackIndexA = -1; ///< 第一端点轨迹索引。
        int trackIndexB = -1; ///< 第二端点轨迹索引。
        double measuredDistance = 0.0; ///< 实测长度，米。
    };
    QVector<MarkerTrackBinding> markerTrackBindings;
    QVector<MarkerScaleBarBinding> markerScaleBarBindings;
};

/// 可选完整标记系统输入；只借用对象，不转移所有权。
struct MarkerBaInput
{
    const control_points::MarkerSet *markerSet = nullptr; ///< 调用期间必须保持有效。
    QHash<QString, QString> imagePathById; ///< 标记 imageId 到当前绝对路径。
};

/**
 * @brief 聚合当前工程可用的全部 BA 观测和约束。
 *
 * result 在开始时被完整清空。成功状态只表示输入可求解，不表示 BA 已收敛。
 */
BaInputBuildStatus buildBaInputFromMeta(const QJsonObject &meta,
                                        const QStringList &selectedImages,
                                        int minMatches,
                                        BaInputBuildResult *result,
                                        const MarkerBaInput *markerInput = nullptr);

} // namespace xjw::core::project
