#pragma once

#include "registration/PriorTrack.h"

#include <QJsonObject>
#include <QMap>
#include <QString>

#include <vector>

namespace xjw::aerial_triangulation
{

struct MarkerPriorLoadResult
{
    bool ok = true;
    QString errorMessage;
    std::vector<control_points::PriorTrack> tracks;
    std::vector<control_points::PriorScaleBar> scaleBars;
};

// 将项目标记 sidecar 转换成 SfM 可消费的先验轨迹，不执行三角化或 BA。
class MarkerPriorLoader
{
public:
    static MarkerPriorLoadResult load(const QString &path,
                                      const QJsonObject &projectMeta,
                                      const QMap<QString, ImageId> &imageIdByPath);
};

} // namespace xjw::aerial_triangulation
