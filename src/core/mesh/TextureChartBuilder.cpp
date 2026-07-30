#include "TextureMappingV4Internal.h"

#include "TextureAtlasPacker.h"

#include <algorithm>
#include <cmath>
#include <queue>

namespace xjw::mesh::texture_v4
{
namespace
{

bool cancelled(const TextureMappingConfig &config)
{
    return config.isCancelled && config.isCancelled();
}

bool projectedBounds(const PreparedView &view,
                     const QVector<FaceGeometry> &geometry,
                     const QVector<int> &faces,
                     int padding,
                     QRect *bounds)
{
    double minimum_x = view.colorBgr.cols;
    double minimum_y = view.colorBgr.rows;
    double maximum_x = -1.0;
    double maximum_y = -1.0;
    for (const int face_index : faces)
    {
        for (const auto &vertex : geometry[face_index].vertices)
        {
            double pixel[2]{};
            double depth = 0.0;
            if (!view.colorCamera.projectWorldPointWithDepth(
                    vertex.data(), pixel, depth))
            {
                return false;
            }
            minimum_x = std::min(minimum_x, pixel[0]);
            minimum_y = std::min(minimum_y, pixel[1]);
            maximum_x = std::max(maximum_x, pixel[0]);
            maximum_y = std::max(maximum_y, pixel[1]);
        }
    }

    const int left = std::clamp(
        static_cast<int>(std::floor(minimum_x)) - padding,
        0,
        view.colorBgr.cols - 1);
    const int top = std::clamp(
        static_cast<int>(std::floor(minimum_y)) - padding,
        0,
        view.colorBgr.rows - 1);
    const int right = std::clamp(
        static_cast<int>(std::ceil(maximum_x)) + padding,
        0,
        view.colorBgr.cols - 1);
    const int bottom = std::clamp(
        static_cast<int>(std::ceil(maximum_y)) + padding,
        0,
        view.colorBgr.rows - 1);
    *bounds = QRect(QPoint(left, top), QPoint(right, bottom));
    return bounds->isValid() && !bounds->isEmpty();
}

} // namespace

bool buildAndPackCharts(const TextureMappingConfig &config,
                        PipelineData *data,
                        TextureMappingResult *result,
                        std::string *errorMsg)
{
    if (!data || !result)
    {
        return false;
    }
    if (config.progressFn)
    {
        config.progressFn("正在构建连通纹理块...", 56);
    }

    QVector<bool> visited(data->geometry.size(), false);
    const int padding = std::clamp(config.padding, 2, 64);
    for (int seed = 0; seed < data->geometry.size(); ++seed)
    {
        if (visited[seed] || data->assignments[seed].primaryView < 0)
        {
            continue;
        }
        if ((seed % 2048 == 0) && cancelled(config))
        {
            result->cancelled = true;
            if (errorMsg)
            {
                *errorMsg = "纹理映射已取消";
            }
            return false;
        }

        TextureChart chart;
        chart.index = data->charts.size();
        chart.primaryView = data->assignments[seed].primaryView;
        std::queue<int> pending;
        pending.push(seed);
        visited[seed] = true;
        while (!pending.empty())
        {
            const int face_index = pending.front();
            pending.pop();
            chart.faces.push_back(face_index);
            for (const int neighbor : data->geometry[face_index].neighbors)
            {
                if (neighbor < 0 || neighbor >= visited.size() || visited[neighbor] ||
                    data->assignments[neighbor].primaryView != chart.primaryView)
                {
                    continue;
                }
                visited[neighbor] = true;
                pending.push(neighbor);
            }
        }

        if (!projectedBounds(data->views[chart.primaryView],
                             data->geometry,
                             chart.faces,
                             padding,
                             &chart.sourceBounds))
        {
            if (errorMsg)
            {
                *errorMsg = "纹理 v4 无法计算纹理块的影像范围";
            }
            return false;
        }
        data->charts.push_back(std::move(chart));
    }

    if (data->charts.isEmpty())
    {
        result->chartCount = 0;
        result->atlasOccupancy = 0.0;
        result->medianTexelDensity = 0.0;
        return true;
    }

    QVector<TextureAtlasItem> items;
    items.reserve(data->charts.size());
    for (const TextureChart &chart : data->charts)
    {
        items.push_back({chart.index, chart.sourceBounds.size(), QRect()});
    }
    const int atlas_size = std::clamp(config.textureSize, 1024, 16384);
    const int fallback_width = std::clamp(padding * 2 + 2, 8, 130);
    const TextureAtlasPackingResult packing =
        TextureAtlasPacker::pack(items, atlas_size, fallback_width);
    if (!packing.ok)
    {
        if (errorMsg)
        {
            *errorMsg = "纹理块无法装入指定大小的图集";
        }
        return false;
    }
    for (const TextureAtlasItem &item : packing.items)
    {
        TextureChart &chart = data->charts[item.id];
        chart.atlasBounds = item.packedRect;
        chart.atlasScale = std::min(
            static_cast<float>(item.packedRect.width()) /
                std::max(chart.sourceBounds.width(), 1),
            static_cast<float>(item.packedRect.height()) /
                std::max(chart.sourceBounds.height(), 1));
    }

    QVector<float> densities;
    densities.reserve(data->charts.size());
    for (const TextureChart &chart : data->charts)
    {
        densities.push_back(chart.atlasScale);
    }
    std::sort(densities.begin(), densities.end());
    data->atlasOccupancy = packing.occupancy;
    result->chartCount = data->charts.size();
    result->atlasOccupancy = packing.occupancy;
    result->medianTexelDensity = densities[densities.size() / 2];
    return true;
}

} // namespace xjw::mesh::texture_v4
