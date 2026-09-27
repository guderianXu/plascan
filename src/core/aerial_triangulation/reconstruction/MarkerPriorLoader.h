#pragma once

/**
 * @file MarkerPriorLoader.h
 * @brief 将完整标记 sidecar 转为 IncrementalSfm 的人工先验轨迹/标尺。
 */

#include <placoordinate/context/CoordinateContext.h>
#include "registration/PriorTrack.h"

#include <QJsonObject>
#include <QMap>
#include <QString>

#include <vector>

namespace xjw::aerial_triangulation
{

    struct MarkerPriorLoadResult
    {
        bool ok = true;       ///< 文件不存在可按“无标记”成功；格式损坏时为 false。
        QString errorMessage; ///< 无法解析或引用当前影像集合失败的原因。
        std::vector<control_points::PriorTrack> tracks;       ///< 有至少两幅有效投影的标记轨迹。
        std::vector<control_points::PriorScaleBar> scaleBars; ///< 两端标记均可用的标尺。
    };

    // 将项目标记 sidecar 转换成 SfM 可消费的先验轨迹，不执行三角化或 BA。
    class MarkerPriorLoader
    {
    public:
        /**
         * @brief 加载标记并把 canonical ImageId 解析到当前 SfM ImageId。
         *
         * TieMarker 只增强轨迹；ControlPoint/CheckPoint 的参考坐标必须先通过米制笛卡尔
         * 归一化门禁。地理角坐标需要调用方显式传入包含源 CRS 和米制 solver reference 的
         * CoordinateContext；路径快照只用于审计，不参与绑定。
         */
        static MarkerPriorLoadResult load(const QString& path,
                                          const QJsonObject& projectMeta,
                                          const QMap<QString, ImageId>& imageIdByCanonicalId,
                                          const placoordinate::CoordinateContext* coordinateContext = nullptr);
    };

} // namespace xjw::aerial_triangulation
