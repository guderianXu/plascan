#pragma once

/**
 * @file AerialTriangulationResolvedConfig.h
 * @brief Workflow 对外部选项完成默认值、路径和职责拆分后的不可歧义配置。
 */

#include "model/AerialTriangulationOptions.h"
#include "matchphototask/task/MatchPhotosTask.h"

#include <QJsonObject>

namespace xjw::aerial_triangulation
{

    /**
     * @brief 同时承载连接点任务和正式 SfM 管线的解析结果。
     *
     * `prepareTiePoints` 控制本次是否调用 matchphototask；`forceRebuildTiePoints`
     * 进一步决定是否先删除当前影像集合的兼容缓存。重置旧相机对齐本身不会自动令
     * forceRebuildTiePoints 为 true。
     */
    struct AerialTriangulationResolvedConfig
    {
        PreparedAerialTriangulationInput pipelineInput;  ///< 连接点准备完成后交给 SfM。
        matchphotos::MatchPhotosOptions tiePointOptions; ///< 特征/匹配算法和配额。
        matchphotos::MatchPhotosContext tiePointContext; ///< 路径、蒙版、参考相机和回调。
        bool prepareTiePoints = false;                   ///< 需要重建或补齐连接点。
        bool forceRebuildTiePoints = false; ///< 清理当前影像相关匹配/连接点，并要求前端重算。
        int cachedTiePointLimit = -1; ///< 运行前持久化连接点配额；-1 表示无缓存或旧缓存未记录。
        QString tiePointLimitAction;  ///< reuse_cache/rebuild_track_selection/generate/full_rebuild 等稳定动作。
        QJsonObject resolvedSettings; ///< 实际生效值，用于工程报告和 GUI 回显。
    };

} // namespace xjw::aerial_triangulation
