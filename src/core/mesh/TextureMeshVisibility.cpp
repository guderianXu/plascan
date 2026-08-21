#include "TextureMappingV4Internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace xjw::mesh::texture_v4
{
namespace
{

bool cancelled(const TextureMappingConfig &config)
{
    return config.isCancelled && config.isCancelled();
}

bool projectTriangle(const PreparedView &view,
                     const FaceGeometry &face,
                     std::array<std::array<double, 2>, 3> *pixels,
                     std::array<double, 3> *depths)
{
    for (int corner = 0; corner < 3; ++corner)
    {
        if (!view.colorCamera.projectWorldPointWithDepth(
                face.vertices[corner].data(),
                (*pixels)[corner].data(),
                (*depths)[corner]) ||
            !std::isfinite((*pixels)[corner][0]) ||
            !std::isfinite((*pixels)[corner][1]) ||
            !std::isfinite((*depths)[corner]) ||
            (*depths)[corner] <= 1.0e-8)
        {
            return false;
        }
    }
    return true;
}

std::uint64_t rasterizeView(const PipelineData &data,
                            PreparedView *view,
                            const TextureMappingConfig &config)
{
    const int rows = view->colorBgr.rows;
    const int columns = view->colorBgr.cols;
    view->finalMeshFaceIds = cv::Mat(
        rows, columns, CV_32SC1, cv::Scalar(-1));
    view->finalMeshVisibleFaces.assign(
        static_cast<std::size_t>(data.geometry.size()), 0);
    cv::Mat depth_buffer(
        rows,
        columns,
        CV_64FC1,
        cv::Scalar(std::numeric_limits<double>::infinity()));
    for (int face_index = 0; face_index < data.geometry.size(); ++face_index)
    {
        if ((face_index % 2048 == 0) && cancelled(config))
        {
            return 0;
        }
        const FaceGeometry &face = data.geometry[face_index];
        std::array<std::array<double, 2>, 3> pixels{};
        std::array<double, 3> depths{};
        if (!projectTriangle(*view, face, &pixels, &depths))
        {
            continue;
        }

        const double minimum_x = std::min({pixels[0][0], pixels[1][0], pixels[2][0]});
        const double maximum_x = std::max({pixels[0][0], pixels[1][0], pixels[2][0]});
        const double minimum_y = std::min({pixels[0][1], pixels[1][1], pixels[2][1]});
        const double maximum_y = std::max({pixels[0][1], pixels[1][1], pixels[2][1]});
        if (maximum_x < 0.0 || maximum_y < 0.0 ||
            minimum_x > static_cast<double>(columns - 1) ||
            minimum_y > static_cast<double>(rows - 1))
        {
            continue;
        }
        const int first_column = static_cast<int>(std::clamp(
            std::ceil(minimum_x), 0.0, static_cast<double>(columns - 1)));
        const int last_column = static_cast<int>(std::clamp(
            std::floor(maximum_x), 0.0, static_cast<double>(columns - 1)));
        const int first_row = static_cast<int>(std::clamp(
            std::ceil(minimum_y), 0.0, static_cast<double>(rows - 1)));
        const int last_row = static_cast<int>(std::clamp(
            std::floor(maximum_y), 0.0, static_cast<double>(rows - 1)));
        if (first_column > last_column || first_row > last_row)
        {
            continue;
        }

        const double x0 = pixels[0][0];
        const double y0 = pixels[0][1];
        const double x1 = pixels[1][0];
        const double y1 = pixels[1][1];
        const double x2 = pixels[2][0];
        const double y2 = pixels[2][1];
        const double denominator =
            (y1 - y2) * (x0 - x2) +
            (x2 - x1) * (y0 - y2);
        if (!std::isfinite(denominator) || std::fabs(denominator) <= 1.0e-12)
        {
            continue;
        }

        for (int row = first_row; row <= last_row; ++row)
        {
            const float *support = view->supportDistance.ptr<float>(row);
            double *best_depth = depth_buffer.ptr<double>(row);
            int *best_face = view->finalMeshFaceIds.ptr<int>(row);
            for (int column = first_column; column <= last_column; ++column)
            {
                if (!std::isfinite(support[column]) || support[column] <= 0.0f)
                {
                    continue;
                }
                const double lambda0 =
                    ((y1 - y2) * (column - x2) +
                     (x2 - x1) * (row - y2)) /
                    denominator;
                const double lambda1 =
                    ((y2 - y0) * (column - x2) +
                     (x0 - x2) * (row - y2)) /
                    denominator;
                const double lambda2 = 1.0 - lambda0 - lambda1;
                if (lambda0 < -1.0e-9 || lambda1 < -1.0e-9 ||
                    lambda2 < -1.0e-9)
                {
                    continue;
                }
                const double inverse_depth =
                    lambda0 / depths[0] +
                    lambda1 / depths[1] +
                    lambda2 / depths[2];
                if (!std::isfinite(inverse_depth) || inverse_depth <= 0.0)
                {
                    continue;
                }
                const double candidate_depth = 1.0 / inverse_depth;
                const bool has_previous = std::isfinite(best_depth[column]);
                const double tie_tolerance = has_previous
                    ? 1.0e-9 * std::max({1.0, candidate_depth, best_depth[column]})
                    : 0.0;
                if (!has_previous ||
                    candidate_depth + tie_tolerance < best_depth[column] ||
                    (std::fabs(candidate_depth - best_depth[column]) <= tie_tolerance &&
                     face_index < best_face[column]))
                {
                    best_depth[column] = candidate_depth;
                    best_face[column] = face_index;
                }
            }
        }
    }

    std::uint64_t visible_pixel_count = 0;
    for (int row = 0; row < rows; ++row)
    {
        const int *visible_faces = view->finalMeshFaceIds.ptr<int>(row);
        for (int column = 0; column < columns; ++column)
        {
            const int face_index = visible_faces[column];
            if (face_index < 0)
            {
                continue;
            }
            ++visible_pixel_count;
            view->finalMeshVisibleFaces[static_cast<std::size_t>(face_index)] = 1;
        }
    }
    return visible_pixel_count;
}

} // namespace

bool buildFinalMeshVisibility(const TextureMappingConfig &config,
                              PipelineData *data,
                              TextureMappingResult *result,
                              std::string *errorMsg)
{
    if (!data || !result)
    {
        return false;
    }
    if (!config.enableFinalMeshVisibility)
    {
        for (PreparedView &view : data->views)
        {
            view.finalMeshFaceIds.release();
            view.finalMeshVisibleFaces.clear();
        }
        return true;
    }
    if (config.progressFn)
    {
        config.progressFn("正在构建最终网格逐视图可见性...", 12);
    }
    std::uint64_t allocated_pixels = 0;
    std::uint64_t largest_view_pixels = 0;
    for (const PreparedView &view : data->views)
    {
        const std::uint64_t pixels =
            static_cast<std::uint64_t>(view.colorBgr.total());
        allocated_pixels += pixels;
        largest_view_pixels = std::max(largest_view_pixels, pixels);
    }
    const double visibility_peak_mib =
        static_cast<double>(allocated_pixels * sizeof(int) +
                            largest_view_pixels * sizeof(double) +
                            static_cast<std::uint64_t>(data->views.size()) *
                                static_cast<std::uint64_t>(data->geometry.size()) *
                                sizeof(std::uint8_t)) /
        (1024.0 * 1024.0);
    constexpr double kMaximumEstimatedPeakMemoryMiB = 3072.0;
    if (result->peakMemoryEstimateMiB + visibility_peak_mib >
        kMaximumEstimatedPeakMemoryMiB)
    {
        if (errorMsg)
        {
            *errorMsg =
                "最终网格可见性与纹理图集预计需要超过 3 GiB 工作内存，"
                "请增大影像降采样倍率或降低纹理大小";
        }
        return false;
    }

    const int view_count = static_cast<int>(data->views.size());
    int visibility_progress_percent = 12;
    for (int view_index = 0; view_index < view_count; ++view_index)
    {
        if (cancelled(config))
        {
            result->cancelled = true;
            if (errorMsg)
            {
                *errorMsg = "纹理映射已取消";
            }
            return false;
        }
        PreparedView &view = data->views[view_index];
        const std::uint64_t visible_pixels =
            rasterizeView(*data, &view, config);
        if (cancelled(config))
        {
            result->cancelled = true;
            if (errorMsg)
            {
                *errorMsg = "纹理映射已取消";
            }
            return false;
        }
        result->visibilityRasterizedPixelCount += visible_pixels;
        if (config.progressFn)
        {
            const int next_percent = 12 + static_cast<int>(
                7LL * (view_index + 1) / std::max(1, view_count));
            if (next_percent > visibility_progress_percent)
            {
                visibility_progress_percent = next_percent;
                config.progressFn(
                    "正在构建最终网格逐视图可见性...",
                    visibility_progress_percent);
            }
        }
    }
    result->peakMemoryEstimateMiB += visibility_peak_mib;
    return true;
}

bool isFinalMeshFaceVisible(const PreparedView &view,
                            int face_index,
                            const std::array<double, 3> &world)
{
    if (view.finalMeshFaceIds.empty())
    {
        return true;
    }
    if (view.finalMeshFaceIds.type() != CV_32SC1)
    {
        return false;
    }
    double pixel[2]{};
    double depth = 0.0;
    if (!view.colorCamera.projectWorldPointWithDepth(
            world.data(), pixel, depth) ||
        !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
        !std::isfinite(depth) || depth <= 0.0 ||
        pixel[0] < 0.0 || pixel[1] < 0.0 ||
        pixel[0] > view.finalMeshFaceIds.cols - 1.0 ||
        pixel[1] > view.finalMeshFaceIds.rows - 1.0)
    {
        return false;
    }
    const int column = static_cast<int>(std::lround(pixel[0]));
    const int row = static_cast<int>(std::lround(pixel[1]));
    return view.finalMeshFaceIds.at<int>(row, column) == face_index;
}

bool isFinalMeshFaceVisibleSomewhere(const PreparedView &view,
                                     int face_index)
{
    return view.finalMeshFaceIds.empty() ||
        (face_index >= 0 &&
         static_cast<std::size_t>(face_index) <
             view.finalMeshVisibleFaces.size() &&
         view.finalMeshVisibleFaces[static_cast<std::size_t>(face_index)] != 0);
}

} // namespace xjw::mesh::texture_v4
