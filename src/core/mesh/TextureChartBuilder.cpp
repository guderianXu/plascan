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
                    vertex.data(), pixel, depth) ||
                !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
                !std::isfinite(depth) || depth <= 0.0)
            {
                return false;
            }
            minimum_x = std::min(minimum_x, pixel[0]);
            minimum_y = std::min(minimum_y, pixel[1]);
            maximum_x = std::max(maximum_x, pixel[0]);
            maximum_y = std::max(maximum_y, pixel[1]);
        }
    }

    const double last_column = static_cast<double>(view.colorBgr.cols - 1);
    const double last_row = static_cast<double>(view.colorBgr.rows - 1);
    const int left = static_cast<int>(
        std::clamp(std::floor(minimum_x), 0.0, last_column));
    const int top = static_cast<int>(
        std::clamp(std::floor(minimum_y), 0.0, last_row));
    const int right = static_cast<int>(
        std::clamp(std::ceil(maximum_x), 0.0, last_column));
    const int bottom = static_cast<int>(
        std::clamp(std::ceil(maximum_y), 0.0, last_row));
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
    int visited_face_count = 0;
    int chart_progress_percent = 56;
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
            ++visited_face_count;
            if ((visited_face_count % 4096 == 0) && cancelled(config))
            {
                result->cancelled = true;
                if (errorMsg)
                {
                    *errorMsg = "纹理映射已取消";
                }
                return false;
            }
            if ((visited_face_count % 4096 == 0) && config.progressFn)
            {
                const int next_percent = 56 + static_cast<int>(
                    static_cast<qint64>(visited_face_count) * 2 /
                    std::max<qsizetype>(data->geometry.size(), 1));
                if (next_percent > chart_progress_percent)
                {
                    chart_progress_percent = next_percent;
                    config.progressFn(
                        "正在构建连通纹理块...",
                        chart_progress_percent);
                }
            }
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
        items.push_back(
            {chart.index, chart.sourceBounds.size(), QRect(), padding});
    }
    const int atlas_size = std::clamp(config.textureSize, 1024, 16384);
    const int fallback_width = config.keepUnmapped
        && data->mesh && data->mesh->hasColors()
        ? kFallbackAtlasWidth
        : std::clamp(padding * 2 + 2, 8, kFallbackAtlasWidth);
    if (config.progressFn)
    {
        config.progressFn(
            "正在打包 " + std::to_string(items.size()) + " 个纹理块...",
            59);
    }
    const TextureAtlasPackingResult packing =
        TextureAtlasPacker::pack(
            items,
            atlas_size,
            fallback_width,
            config.isCancelled,
            config.progressFn
                ? std::function<void(int)>([&config](int percent)
                  {
                      config.progressFn(
                          "正在打包纹理图集...",
                          59 + std::clamp(percent, 0, 100) * 6 / 100);
                  })
                : std::function<void(int)>(),
            static_cast<float>(std::clamp(config.imageDownscale, 1, 8)));
    if (packing.cancelled)
    {
        result->cancelled = true;
        if (errorMsg)
        {
            *errorMsg = "纹理映射已取消";
        }
        return false;
    }
    if (!packing.ok)
    {
        if (errorMsg)
        {
            *errorMsg =
                "纹理块无法装入 " + std::to_string(atlas_size) +
                "x" + std::to_string(atlas_size) + " 图集（" +
                std::to_string(items.size()) + " 个纹理块，边距 " +
                std::to_string(padding) +
                " px）；请提高纹理分辨率或减小边距";
        }
        return false;
    }
    for (const TextureAtlasItem &item : packing.items)
    {
        TextureChart &chart = data->charts[item.id];
        chart.atlasBounds = item.packedRect;
        chart.atlasContentBounds = QRect(
            item.packedRect.x() + padding,
            item.packedRect.y() + padding,
            item.packedRect.width() - padding * 2,
            item.packedRect.height() - padding * 2);
        chart.atlasScale = packing.scale;
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
