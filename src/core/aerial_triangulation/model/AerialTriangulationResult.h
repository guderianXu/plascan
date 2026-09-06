#pragma once

/**
 * @file AerialTriangulationResult.h
 * @brief 空三各层共享的重建结果、回写数据和诊断结构。
 */

#include "model/AerialTriangulationResolvedConfig.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>

namespace xjw::aerial_triangulation
{

/// 仅表示正式 SfM/BA 与结果写出的输出，不包含连接点任务本身。
struct AerialTriangulationReconstructionResult
{
    bool success = false; ///< 正式稀疏模型通过质量门控且结果写出成功。
    QString errorMessage; ///< 失败时的用户可读原因。
    QString summary; ///< 简短结果摘要；失败时通常与 errorMessage 一致。
    int numRegisteredImages = 0; ///< 最终有效相机位姿数。
    int numPoints3D = 0; ///< 最终稀疏点数。
    double meanReprojError = 0.0; ///< 最终点平均重投影误差，像素。
    QMap<QString, QJsonObject> pendingCamUpdates; ///< 事务提交前的影像路径到相机 JSON。
    QString sparseCloudPath; ///< 正式 sfm_sparse.ply 路径。
    QString displaySparseCloudPath; ///< 可选的清理显示云，不替代正式算法点云。
    QJsonObject qualityMetadata; ///< MVS 门控和稀疏质量摘要。
    QJsonObject resultRecordExtra; ///< 工程结果记录的扩展字段。
    QJsonObject sfmDiagnostics; ///< 候选搜索、初始对、匹配网和 BA 诊断。

    // 统一 bundle_adjust 模块的关键统计。
    double baRmsBefore = 0.0;
    double baRmsAfter = 0.0;
    int baTracksTotal = 0;
    int baTracksOptimized = 0;
    int baTracksFiltered = 0;
    double durationSeconds = -1.0; ///< 整条正式管线耗时；<0 表示未测量。
    QJsonArray perCameraResiduals; ///< 每台已注册相机的观测数量和残差统计。
};

/// 最外层 Workflow 输出，同时保留解析配置和可选连接点准备结果。
struct AerialTriangulationResult
{
    AerialTriangulationResolvedConfig config; ///< 实际生效配置。
    AerialTriangulationReconstructionResult reconstructionResult; ///< 正式空三结果。
    bool tiePointPreparationExecuted = false; ///< 本次是否实际调用 matchphototask。
    matchphotos::MatchPhotosResult tiePointResult; ///< 未执行时保持默认值。
};

} // namespace xjw::aerial_triangulation
