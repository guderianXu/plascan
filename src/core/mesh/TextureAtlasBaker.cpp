#include "TextureMappingV4Internal.h"

#include "TextureAtlasSampling.h"
#include "TextureSeamLeveling.h"

#include "io/PathIO.h"

#include <plapoint/io/obj_io.h>
#include <plamatrix/dense/dense_matrix.h>

#include <opencv2/imgproc.hpp>

#include <QDir>
#include <QFile>
#include <QImage>
#include <QUuid>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace xjw::mesh::texture_v4
{
namespace
{

bool cancelled(const TextureMappingConfig &config)
{
    return config.isCancelled && config.isCancelled();
}

float colorDistance(const cv::Vec3f &left, const cv::Vec3f &right)
{
    const cv::Vec3f difference = left - right;
    return std::sqrt(difference.dot(difference));
}

const FaceCandidate *candidateForView(const FaceAssignment &assignment,
                                      int viewIndex)
{
    const auto candidate = std::find_if(
        assignment.candidates.begin(),
        assignment.candidates.end(),
        [viewIndex](const FaceCandidate &value)
        {
            return value.viewIndex == viewIndex;
        });
    return candidate == assignment.candidates.end() ? nullptr : &*candidate;
}

bool barycentric(double x,
                 double y,
                 const std::array<QPointF, 3> &triangle,
                 std::array<double, 3> *weights)
{
    const double denominator =
        (triangle[1].y() - triangle[2].y()) *
            (triangle[0].x() - triangle[2].x()) +
        (triangle[2].x() - triangle[1].x()) *
            (triangle[0].y() - triangle[2].y());
    if (std::fabs(denominator) < 1.0e-12)
    {
        return false;
    }
    (*weights)[0] =
        ((triangle[1].y() - triangle[2].y()) * (x - triangle[2].x()) +
         (triangle[2].x() - triangle[1].x()) * (y - triangle[2].y())) /
        denominator;
    (*weights)[1] =
        ((triangle[2].y() - triangle[0].y()) * (x - triangle[2].x()) +
         (triangle[0].x() - triangle[2].x()) * (y - triangle[2].y())) /
        denominator;
    (*weights)[2] = 1.0 - (*weights)[0] - (*weights)[1];
    constexpr double tolerance = -1.0e-5;
    return (*weights)[0] >= tolerance &&
           (*weights)[1] >= tolerance &&
           (*weights)[2] >= tolerance;
}

cv::Vec3b fallbackColor(const PlaPointCloud &mesh)
{
    if (!mesh.hasColors() || !mesh.colors() || mesh.colors()->rows() == 0)
    {
        return cv::Vec3b(180, 180, 180);
    }
    std::uint64_t red = 0;
    std::uint64_t green = 0;
    std::uint64_t blue = 0;
    for (int index = 0; index < mesh.colors()->rows(); ++index)
    {
        red += mesh.colors()->getValue(index, 0);
        green += mesh.colors()->getValue(index, 1);
        blue += mesh.colors()->getValue(index, 2);
    }
    const std::uint64_t count = mesh.colors()->rows();
    return cv::Vec3b(
        static_cast<std::uint8_t>(blue / count),
        static_cast<std::uint8_t>(green / count),
        static_cast<std::uint8_t>(red / count));
}

cv::Vec3b meshVertexColor(const PlaPointCloud &mesh, int vertexIndex)
{
    const auto row = static_cast<plamatrix::Index>(vertexIndex);
    return cv::Vec3b(
        mesh.colors()->getValue(row, 2),
        mesh.colors()->getValue(row, 1),
        mesh.colors()->getValue(row, 0));
}

void bakeVertexColorTile(cv::Mat *atlas,
                         cv::Mat *mask,
                         int left,
                         int top,
                         const std::array<cv::Vec3b, 3> &colors)
{
    for (int row = 0; row < kFallbackTileSize; ++row)
    {
        for (int column = 0; column < kFallbackTileSize; ++column)
        {
            float second_weight = std::max(0.0f, static_cast<float>(column - 1));
            float third_weight = std::max(0.0f, static_cast<float>(row - 1));
            const float edge_sum = second_weight + third_weight;
            if (edge_sum > 1.0f)
            {
                second_weight /= edge_sum;
                third_weight /= edge_sum;
            }
            const float first_weight = 1.0f - second_weight - third_weight;
            cv::Vec3b &destination = atlas->at<cv::Vec3b>(top + row, left + column);
            for (int channel = 0; channel < 3; ++channel)
            {
                destination[channel] = cv::saturate_cast<std::uint8_t>(
                    first_weight * colors[0][channel]
                    + second_weight * colors[1][channel]
                    + third_weight * colors[2][channel]);
            }
            mask->at<std::uint8_t>(top + row, left + column) = 255;
        }
    }
}

void expandPadding(cv::Mat *atlas,
                   cv::Mat *mask,
                   const QRect &bounds,
                   int padding)
{
    cv::Rect roi(
        bounds.x(), bounds.y(), bounds.width(), bounds.height());
    roi &= cv::Rect(0, 0, atlas->cols, atlas->rows);
    if (padding <= 0 || roi.empty())
    {
        return;
    }
    cv::Mat atlas_roi = (*atlas)(roi);
    cv::Mat mask_roi = (*mask)(roi);
    if (cv::countNonZero(mask_roi) == 0)
    {
        return;
    }
    cv::Mat expanded_mask;
    const int kernel_size = padding * 2 + 1;
    cv::dilate(mask_roi,
               expanded_mask,
               cv::getStructuringElement(
                   cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size)));

    cv::Mat distance_source(mask_roi.size(), CV_8UC1, cv::Scalar(255));
    distance_source.setTo(0, mask_roi);
    cv::Mat distance;
    cv::Mat labels;
    cv::distanceTransform(
        distance_source,
        distance,
        labels,
        cv::DIST_L2,
        cv::DIST_MASK_5,
        cv::DIST_LABEL_PIXEL);

    double maximum_label_value = 0.0;
    cv::minMaxLoc(labels, nullptr, &maximum_label_value);
    const int maximum_label = static_cast<int>(maximum_label_value);
    std::vector<cv::Vec3b> colors(static_cast<std::size_t>(maximum_label + 1));
    for (int row = 0; row < mask_roi.rows; ++row)
    {
        for (int column = 0; column < mask_roi.cols; ++column)
        {
            if (mask_roi.at<std::uint8_t>(row, column) != 0)
            {
                colors[labels.at<int>(row, column)] =
                    atlas_roi.at<cv::Vec3b>(row, column);
            }
        }
    }
    for (int row = 0; row < mask_roi.rows; ++row)
    {
        for (int column = 0; column < mask_roi.cols; ++column)
        {
            if (mask_roi.at<std::uint8_t>(row, column) == 0 &&
                expanded_mask.at<std::uint8_t>(row, column) != 0)
            {
                atlas_roi.at<cv::Vec3b>(row, column) =
                    colors[labels.at<int>(row, column)];
            }
        }
    }
    expanded_mask.copyTo(mask_roi);
}

void sharpenTexture(cv::Mat *atlas,
                    const cv::Mat &mask,
                    const QRect &bounds,
                    float strength)
{
    cv::Rect roi(
        bounds.x(), bounds.y(), bounds.width(), bounds.height());
    roi &= cv::Rect(0, 0, atlas->cols, atlas->rows);
    if (strength <= 0.0f || roi.empty())
    {
        return;
    }
    cv::Mat atlas_roi = (*atlas)(roi);
    const cv::Mat mask_roi = mask(roi);
    cv::Mat blurred;
    cv::GaussianBlur(
        atlas_roi,
        blurred,
        cv::Size(),
        1.0,
        1.0,
        cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    cv::Mat sharpened;
    cv::addWeighted(
        atlas_roi, 1.0 + strength, blurred, -strength, 0.0, sharpened);
    sharpened.copyTo(atlas_roi, mask_roi);
}

bool writeMaterial(const QString &path)
{
    std::ofstream material = xjw::common::io::openOutputFile(
        path, std::ios::out | std::ios::trunc);
    if (!material)
    {
        return false;
    }
    material << "newmtl material0\n"
             << "Ka 1.000000 1.000000 1.000000\n"
             << "Kd 1.000000 1.000000 1.000000\n"
             << "Ks 0.000000 0.000000 0.000000\n"
             << "d 1.000000\n"
             << "illum 2\n"
             << "map_Kd textures/model_texture.png\n";
    return true;
}

bool commitFiles(
    const std::vector<std::pair<QString, QString>> &temporary_and_final,
    const QString &token)
{
    std::vector<QString> backups(temporary_and_final.size());
    for (std::size_t index = 0; index < temporary_and_final.size(); ++index)
    {
        const QString &final_path = temporary_and_final[index].second;
        if (!QFile::exists(final_path))
        {
            continue;
        }
        backups[index] = final_path + QStringLiteral(".backup-") + token;
        QFile::remove(backups[index]);
        if (!QFile::rename(final_path, backups[index]))
        {
            for (std::size_t restore = 0; restore < index; ++restore)
            {
                if (!backups[restore].isEmpty())
                {
                    QFile::rename(backups[restore],
                                  temporary_and_final[restore].second);
                }
            }
            return false;
        }
    }

    std::size_t committed = 0;
    for (; committed < temporary_and_final.size(); ++committed)
    {
        if (!QFile::rename(temporary_and_final[committed].first,
                           temporary_and_final[committed].second))
        {
            break;
        }
    }
    if (committed != temporary_and_final.size())
    {
        for (std::size_t index = 0; index < committed; ++index)
        {
            QFile::remove(temporary_and_final[index].second);
        }
        for (std::size_t index = 0; index < backups.size(); ++index)
        {
            if (!backups[index].isEmpty())
            {
                QFile::rename(backups[index],
                              temporary_and_final[index].second);
            }
        }
        return false;
    }
    for (const QString &backup : backups)
    {
        if (!backup.isEmpty())
        {
            QFile::remove(backup);
        }
    }
    return true;
}

double estimateSeamDifference(const PipelineData &data,
                              const TextureMappingConfig &config)
{
    double difference_sum = 0.0;
    int sample_count = 0;
    for (int face_index = 0; face_index < data.geometry.size(); ++face_index)
    {
        const int view_index = data.assignments[face_index].primaryView;
        if (view_index < 0)
        {
            continue;
        }
        for (const int neighbor : data.geometry[face_index].neighbors)
        {
            const int neighbor_view = data.assignments[neighbor].primaryView;
            if (neighbor <= face_index || neighbor_view < 0 ||
                neighbor_view == view_index)
            {
                continue;
            }
            const auto &world = data.geometry[face_index].centroid;
            const FaceCandidate *first_candidate = candidateForView(
                data.assignments[face_index], view_index);
            const FaceCandidate *second_candidate = candidateForView(
                data.assignments[neighbor], neighbor_view);
            if (!first_candidate || !second_candidate)
            {
                continue;
            }
            WeightedColor first;
            WeightedColor second;
            if (sampleTextureView(data.views[view_index],
                                  world,
                                  *first_candidate,
                                  config,
                                  data.medianEdgeLength,
                                  1,
                                  &first) == TextureSampleStatus::Sampled &&
                sampleTextureView(data.views[neighbor_view],
                                  world,
                                  *second_candidate,
                                  config,
                                  data.medianEdgeLength,
                                  1,
                                  &second) == TextureSampleStatus::Sampled)
            {
                difference_sum += colorDistance(first.color, second.color);
                ++sample_count;
            }
        }
    }
    return sample_count > 0 ? difference_sum / sample_count : 0.0;
}

cv::Vec3f atlasLinearColor(const cv::Mat &atlas, int row, int column)
{
    const cv::Vec3b encoded = atlas.at<cv::Vec3b>(row, column);
    cv::Vec3f linear;
    for (int channel = 0; channel < 3; ++channel)
    {
        const float value = encoded[channel] / 255.0f;
        linear[channel] = value <= 0.04045f
            ? value / 12.92f
            : std::pow((value + 0.055f) / 1.055f, 2.4f);
    }
    return linear;
}

QPointF atlasPointForChartVertex(
    int chart_index,
    int vertex_index,
    int atlas_size,
    const std::unordered_map<std::uint64_t, int> &texture_index_by_chart_vertex,
    const std::vector<std::array<float, 2>> &texture_coordinates)
{
    const std::uint64_t key =
        (static_cast<std::uint64_t>(
             static_cast<std::uint32_t>(chart_index)) << 32U) |
        static_cast<std::uint32_t>(vertex_index);
    const auto found = texture_index_by_chart_vertex.find(key);
    if (found == texture_index_by_chart_vertex.end() ||
        found->second < 0 ||
        found->second >= static_cast<int>(texture_coordinates.size()))
    {
        return QPointF(-1.0, -1.0);
    }
    const auto &uv = texture_coordinates[static_cast<std::size_t>(found->second)];
    return QPointF(
        static_cast<double>(uv[0]) * atlas_size,
        static_cast<double>(1.0f - uv[1]) * atlas_size);
}

std::vector<TextureSeamConstraint> buildSharedEdgeSeamConstraints(
    const PipelineData &data,
    const cv::Mat &atlas,
    const cv::Mat &filled_mask,
    int atlas_size,
    const std::vector<int> &chart_for_face,
    const std::unordered_map<std::uint64_t, int> &texture_index_by_chart_vertex,
    const std::vector<std::array<float, 2>> &texture_coordinates)
{
    std::vector<TextureSeamConstraint> constraints;
    std::set<std::tuple<int, int, int, int>> visited;
    for (int face_index = 0; face_index < data.geometry.size(); ++face_index)
    {
        const int first_chart = chart_for_face[static_cast<std::size_t>(face_index)];
        if (first_chart < 0)
        {
            continue;
        }
        for (const int neighbor : data.geometry[face_index].neighbors)
        {
            if (neighbor <= face_index || neighbor < 0 ||
                neighbor >= data.geometry.size())
            {
                continue;
            }
            const int second_chart =
                chart_for_face[static_cast<std::size_t>(neighbor)];
            if (second_chart < 0 || second_chart == first_chart)
            {
                continue;
            }
            std::array<int, 2> shared{{-1, -1}};
            int shared_count = 0;
            for (const int vertex : data.geometry[face_index].vertexIndices)
            {
                if (std::find(
                        data.geometry[neighbor].vertexIndices.cbegin(),
                        data.geometry[neighbor].vertexIndices.cend(),
                        vertex) != data.geometry[neighbor].vertexIndices.cend() &&
                    shared_count < 2)
                {
                    shared[static_cast<std::size_t>(shared_count++)] = vertex;
                }
            }
            if (shared_count != 2)
            {
                continue;
            }
            const int lower_chart = std::min(first_chart, second_chart);
            const int upper_chart = std::max(first_chart, second_chart);
            const int lower_vertex = std::min(shared[0], shared[1]);
            const int upper_vertex = std::max(shared[0], shared[1]);
            if (!visited.emplace(
                    lower_chart, upper_chart, lower_vertex, upper_vertex).second)
            {
                continue;
            }
            const QPointF first_start = atlasPointForChartVertex(
                first_chart, shared[0], atlas_size,
                texture_index_by_chart_vertex, texture_coordinates);
            const QPointF first_end = atlasPointForChartVertex(
                first_chart, shared[1], atlas_size,
                texture_index_by_chart_vertex, texture_coordinates);
            const QPointF second_start = atlasPointForChartVertex(
                second_chart, shared[0], atlas_size,
                texture_index_by_chart_vertex, texture_coordinates);
            const QPointF second_end = atlasPointForChartVertex(
                second_chart, shared[1], atlas_size,
                texture_index_by_chart_vertex, texture_coordinates);
            if (first_start.x() < 0.0 || first_end.x() < 0.0 ||
                second_start.x() < 0.0 || second_end.x() < 0.0)
            {
                continue;
            }
            std::array<std::vector<float>, 3> differences;
            for (int sample = 0; sample < 9; ++sample)
            {
                const double alpha = (sample + 0.5) / 9.0;
                const QPointF first = first_start * (1.0 - alpha) +
                    first_end * alpha;
                const QPointF second = second_start * (1.0 - alpha) +
                    second_end * alpha;
                const int first_column = std::clamp(
                    static_cast<int>(std::lround(first.x() - 0.5)),
                    0, atlas.cols - 1);
                const int first_row = std::clamp(
                    static_cast<int>(std::lround(first.y() - 0.5)),
                    0, atlas.rows - 1);
                const int second_column = std::clamp(
                    static_cast<int>(std::lround(second.x() - 0.5)),
                    0, atlas.cols - 1);
                const int second_row = std::clamp(
                    static_cast<int>(std::lround(second.y() - 0.5)),
                    0, atlas.rows - 1);
                if (filled_mask.at<std::uint8_t>(first_row, first_column) == 0 ||
                    filled_mask.at<std::uint8_t>(second_row, second_column) == 0)
                {
                    continue;
                }
                const cv::Vec3f first_color = atlasLinearColor(
                    atlas, first_row, first_column);
                const cv::Vec3f second_color = atlasLinearColor(
                    atlas, second_row, second_column);
                for (int channel = 0; channel < 3; ++channel)
                {
                    differences[static_cast<std::size_t>(channel)].push_back(
                        second_color[channel] - first_color[channel]);
                }
            }
            if (differences[0].size() < 3)
            {
                continue;
            }
            cv::Vec3f median_difference;
            for (int channel = 0; channel < 3; ++channel)
            {
                auto &values = differences[static_cast<std::size_t>(channel)];
                std::sort(values.begin(), values.end());
                median_difference[channel] =
                    values[values.size() / 2];
            }
            constraints.push_back({
                first_chart,
                second_chart,
                median_difference,
                static_cast<float>(differences[0].size())});
        }
    }
    return constraints;
}

bool sampleFaceTextureColor(const PipelineData &data,
                            const FaceAssignment &assignment,
                            int face_index,
                            const std::array<double, 3> &world,
                            const TextureMappingConfig &config,
                            int padding,
                            TextureMappingResult *result,
                            cv::Vec3b *color)
{
    if (!color)
    {
        return false;
    }
    std::vector<WeightedColor> samples;
    const int maximum_samples = config.blendMode == TextureBlendMode::BestView
        ? 1
        : std::clamp(config.maximumBlendedViews, 1, 8);
    WeightedColor missing_primary_sample;
    bool has_missing_primary_sample = false;
    auto append_sample = [&](const FaceCandidate &candidate)
    {
        WeightedColor sample;
        const TextureSampleStatus status = sampleTextureView(
            data.views[candidate.viewIndex],
            world,
            candidate,
            config,
            data.medianEdgeLength,
            padding,
            &sample,
            face_index);
        if (status == TextureSampleStatus::Sampled)
        {
            samples.push_back(sample);
        }
        else if (status == TextureSampleStatus::MissingDepthEvidence &&
                 candidate.strict &&
                 candidate.viewIndex == assignment.primaryView)
        {
            missing_primary_sample = sample;
            has_missing_primary_sample = true;
        }
    };
    const auto primary_candidate = std::find_if(
        assignment.candidates.begin(),
        assignment.candidates.end(),
        [&](const FaceCandidate &candidate)
        {
            return candidate.viewIndex == assignment.primaryView;
        });
    if (primary_candidate != assignment.candidates.end())
    {
        append_sample(*primary_candidate);
    }
    for (const FaceCandidate &candidate : assignment.candidates)
    {
        if (static_cast<int>(samples.size()) >= maximum_samples)
        {
            break;
        }
        if (candidate.viewIndex == assignment.primaryView)
        {
            continue;
        }
        append_sample(candidate);
    }
    if (samples.empty() && has_missing_primary_sample)
    {
        samples.push_back(missing_primary_sample);
    }
    if (samples.empty())
    {
        return false;
    }
    *color = blendTextureSamples(std::move(samples), config, result);
    return true;
}

} // namespace

bool perspectiveCorrectBarycentricWeights(
    const std::array<double, 3> &affine_weights,
    const std::array<double, 3> &camera_depths,
    std::array<double, 3> *corrected_weights)
{
    if (!corrected_weights)
    {
        return false;
    }
    double denominator = 0.0;
    for (int corner = 0; corner < 3; ++corner)
    {
        if (!std::isfinite(affine_weights[corner]) ||
            !std::isfinite(camera_depths[corner]) ||
            camera_depths[corner] <= 1.0e-12)
        {
            return false;
        }
        denominator += affine_weights[corner] / camera_depths[corner];
    }
    if (!std::isfinite(denominator) || denominator <= 1.0e-15)
    {
        return false;
    }
    for (int corner = 0; corner < 3; ++corner)
    {
        (*corrected_weights)[corner] =
            affine_weights[corner] / camera_depths[corner] / denominator;
    }
    return true;
}

bool bakeAndExport(const std::string &productsDir,
                   const TextureMappingConfig &config,
                   PipelineData *data,
                   TextureMappingResult *result,
                   std::string *errorMsg)
{
    if (!data || !data->mesh || !result)
    {
        return false;
    }
    const int atlas_size = std::clamp(config.textureSize, 1024, 16384);
    const int padding = std::clamp(config.padding, 2, 64);
    cv::Mat atlas(atlas_size, atlas_size, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat filled_mask(atlas_size, atlas_size, CV_8UC1, cv::Scalar(0));
    cv::Mat chart_index_map(atlas_size, atlas_size, CV_32SC1, cv::Scalar(-1));
    const int fallback_size = std::clamp(padding * 2, 6, 128);
    const cv::Vec3b fallback = config.keepUnmapped
        ? fallbackColor(*data->mesh)
        : cv::Vec3b(0, 0, 0);
    atlas(cv::Rect(1, 1, fallback_size, fallback_size)).setTo(fallback);
    filled_mask(cv::Rect(1, 1, fallback_size, fallback_size)).setTo(255);

    const int face_count = data->geometry.size();
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> texture_indices(
        static_cast<plamatrix::Index>(face_count), 3);
    const float fallback_u =
        atlasCoordinateToNormalizedUv(
            1.0 + fallback_size * 0.5, atlas_size);
    const float fallback_v =
        1.0f - atlasCoordinateToNormalizedUv(
            1.0 + fallback_size * 0.5, atlas_size);
    std::vector<std::array<float, 2>> texture_coordinate_values{
        {fallback_u, fallback_v}
    };
    std::unordered_map<std::uint64_t, int> texture_index_by_chart_vertex;
    for (int face_index = 0; face_index < face_count; ++face_index)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            texture_indices.setValue(face_index, corner, 0);
        }
    }

    const bool has_vertex_color_fallback =
        config.keepUnmapped && data->mesh->hasColors() && data->mesh->colors();
    const int fallback_columns =
        (kFallbackAtlasWidth - 2) / kFallbackTileSize;
    const int fallback_first_row = fallback_size + 2;
    const int fallback_rows = std::max(
        0,
        (atlas_size - fallback_first_row - 1) / kFallbackTileSize);
    const int fallback_tile_capacity = fallback_columns * fallback_rows;
    int fallback_tile_index = 0;
    const auto assign_vertex_color_fallback = [&](int face_index)
    {
        if (!has_vertex_color_fallback ||
            fallback_tile_index >= fallback_tile_capacity)
        {
            return false;
        }
        const FaceGeometry &face = data->geometry[face_index];
        const int left = 1 +
            (fallback_tile_index % fallback_columns) * kFallbackTileSize;
        const int top = fallback_first_row +
            (fallback_tile_index / fallback_columns) * kFallbackTileSize;
        const std::array<cv::Vec3b, 3> colors{
            meshVertexColor(*data->mesh, face.vertexIndices[0]),
            meshVertexColor(*data->mesh, face.vertexIndices[1]),
            meshVertexColor(*data->mesh, face.vertexIndices[2])};
        bakeVertexColorTile(&atlas, &filled_mask, left, top, colors);
        const std::array<QPointF, 3> centers{
            QPointF(left + 1.5, top + 1.5),
            QPointF(left + 2.5, top + 1.5),
            QPointF(left + 1.5, top + 2.5)};
        for (int corner = 0; corner < 3; ++corner)
        {
            const int texture_index =
                static_cast<int>(texture_coordinate_values.size());
            texture_coordinate_values.push_back({
                atlasCoordinateToNormalizedUv(
                    centers[corner].x(), atlas_size),
                1.0f - atlasCoordinateToNormalizedUv(
                    centers[corner].y(), atlas_size)});
            texture_indices.setValue(face_index, corner, texture_index);
        }
        ++fallback_tile_index;
        return true;
    };
    if (has_vertex_color_fallback)
    {
        for (int face_index = 0;
             face_index < face_count &&
             fallback_tile_index < fallback_tile_capacity;
             ++face_index)
        {
            if (data->assignments[face_index].primaryView >= 0)
            {
                continue;
            }
            assign_vertex_color_fallback(face_index);
        }
    }

    if (config.progressFn)
    {
        config.progressFn("正在烘焙多视角纹理...", 66);
    }
    int processed_faces = 0;
    int bake_progress_percent = 66;
    std::vector<int> chart_for_face(
        static_cast<std::size_t>(face_count), -1);
    for (const TextureChart &chart : data->charts)
    {
        const PreparedView &primary_view = data->views[chart.primaryView];
        for (const int face_index : chart.faces)
        {
            if ((processed_faces % 1024 == 0) && cancelled(config))
            {
                result->cancelled = true;
                if (errorMsg)
                {
                    *errorMsg = "纹理映射已取消";
                }
                return false;
            }
            ++processed_faces;
            const FaceGeometry &face = data->geometry[face_index];
            chart_for_face[static_cast<std::size_t>(face_index)] = chart.index;
            FaceAssignment &assignment = data->assignments[face_index];
            bool wrote_face_texel = false;
            std::array<QPointF, 3> atlas_triangle;
            std::array<double, 3> primary_camera_depths{};
            for (int corner = 0; corner < 3; ++corner)
            {
                double pixel[2]{};
                if (!primary_view.colorCamera.projectWorldPointWithDepth(
                        face.vertices[corner].data(),
                        pixel,
                        primary_camera_depths[corner]) ||
                    !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
                    !std::isfinite(primary_camera_depths[corner]) ||
                    primary_camera_depths[corner] <= 0.0)
                {
                    if (errorMsg)
                    {
                        *errorMsg = "纹理 v4 烘焙阶段遇到无效相机投影";
                    }
                    return false;
                }
                atlas_triangle[corner] = QPointF(
                    chart.atlasContentBounds.x() +
                        (pixel[0] - chart.sourceBounds.x()) * chart.atlasScale,
                    chart.atlasContentBounds.y() +
                        (pixel[1] - chart.sourceBounds.y()) * chart.atlasScale);
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(
                         static_cast<std::uint32_t>(chart.index)) << 32U) |
                    static_cast<std::uint32_t>(
                        face.vertexIndices[corner]);
                const auto existing =
                    texture_index_by_chart_vertex.find(key);
                int texture_index = 0;
                if (existing != texture_index_by_chart_vertex.end())
                {
                    texture_index = existing->second;
                }
                else
                {
                    texture_index =
                        static_cast<int>(texture_coordinate_values.size());
                    texture_coordinate_values.push_back({
                        atlasCoordinateToNormalizedUv(
                            atlas_triangle[corner].x(), atlas_size),
                        1.0f - atlasCoordinateToNormalizedUv(
                            atlas_triangle[corner].y(), atlas_size)});
                    texture_index_by_chart_vertex.emplace(key, texture_index);
                }
                texture_indices.setValue(face_index, corner, texture_index);
            }

            const int left = std::clamp(
                static_cast<int>(std::floor(std::min({
                    atlas_triangle[0].x(),
                    atlas_triangle[1].x(),
                    atlas_triangle[2].x()}))),
                0,
                atlas_size - 1);
            const int right = std::clamp(
                static_cast<int>(std::ceil(std::max({
                    atlas_triangle[0].x(),
                    atlas_triangle[1].x(),
                    atlas_triangle[2].x()}))),
                0,
                atlas_size - 1);
            const int top = std::clamp(
                static_cast<int>(std::floor(std::min({
                    atlas_triangle[0].y(),
                    atlas_triangle[1].y(),
                    atlas_triangle[2].y()}))),
                0,
                atlas_size - 1);
            const int bottom = std::clamp(
                static_cast<int>(std::ceil(std::max({
                    atlas_triangle[0].y(),
                    atlas_triangle[1].y(),
                    atlas_triangle[2].y()}))),
                0,
                atlas_size - 1);
            for (int row = top; row <= bottom; ++row)
            {
                for (int column = left; column <= right; ++column)
                {
                    std::array<double, 3> weights{};
                    if (!barycentric(
                            column + 0.5, row + 0.5, atlas_triangle, &weights))
                    {
                        continue;
                    }
                    std::array<double, 3> corrected_weights{};
                    if (!perspectiveCorrectBarycentricWeights(
                            weights,
                            primary_camera_depths,
                            &corrected_weights))
                    {
                        continue;
                    }
                    std::array<double, 3> world{};
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        world[axis] =
                            corrected_weights[0] * face.vertices[0][axis] +
                            corrected_weights[1] * face.vertices[1][axis] +
                            corrected_weights[2] * face.vertices[2][axis];
                    }
                    cv::Vec3b color;
                    if (!sampleFaceTextureColor(
                            *data,
                            assignment,
                            face_index,
                            world,
                            config,
                            padding,
                            result,
                            &color))
                    {
                        continue;
                    }
                    atlas.at<cv::Vec3b>(row, column) = color;
                    filled_mask.at<std::uint8_t>(row, column) = 255;
                    chart_index_map.at<int>(row, column) = chart.index;
                    wrote_face_texel = true;
                }
            }
            if (!wrote_face_texel)
            {
                ++result->noTexelFaceCount;
                const QPointF center =
                    (atlas_triangle[0] + atlas_triangle[1] + atlas_triangle[2]) /
                    3.0;
                const int column = std::clamp(
                    static_cast<int>(std::lround(center.x() - 0.5)),
                    0,
                    atlas_size - 1);
                const int row = std::clamp(
                    static_cast<int>(std::lround(center.y() - 0.5)),
                    0,
                    atlas_size - 1);
                const std::array<double, 3> affine_weights{
                    {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}};
                std::array<double, 3> corrected_weights{};
                std::array<double, 3> world{};
                const bool has_world = perspectiveCorrectBarycentricWeights(
                    affine_weights,
                    primary_camera_depths,
                    &corrected_weights);
                if (has_world)
                {
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        world[axis] =
                            corrected_weights[0] * face.vertices[0][axis] +
                            corrected_weights[1] * face.vertices[1][axis] +
                            corrected_weights[2] * face.vertices[2][axis];
                    }
                }
                cv::Vec3b color;
                if (has_world && sampleFaceTextureColor(
                        *data,
                        assignment,
                        face_index,
                        world,
                        config,
                        padding,
                        result,
                        &color))
                {
                    atlas.at<cv::Vec3b>(row, column) = color;
                    filled_mask.at<std::uint8_t>(row, column) = 255;
                    chart_index_map.at<int>(row, column) = chart.index;
                    ++result->centerRecoveredFaceCount;
                }
                else
                {
                    if (!assign_vertex_color_fallback(face_index))
                    {
                        for (int corner = 0; corner < 3; ++corner)
                        {
                            texture_indices.setValue(face_index, corner, 0);
                        }
                    }
                    ++result->unresolvedBakeFaceCount;
                    --result->mappedFaceCount;
                    ++result->unmappedFaceCount;
                    if (assignment.relaxed)
                    {
                        --result->fallbackMappedFaceCount;
                    }
                    else
                    {
                        --result->strictMappedFaceCount;
                    }
                    assignment.primaryView = -1;
                }
            }
        }
        if (config.progressFn)
        {
            const int next_percent = 66 + static_cast<int>(
                20.0 * processed_faces / std::max(face_count, 1));
            if (next_percent > bake_progress_percent)
            {
                bake_progress_percent = next_percent;
                config.progressFn(
                    "正在烘焙多视角纹理...",
                    bake_progress_percent);
            }
        }
    }

    std::vector<bool> baked_views(data->views.size(), false);
    for (const FaceAssignment &assignment : data->assignments)
    {
        if (assignment.primaryView >= 0 &&
            assignment.primaryView < static_cast<int>(baked_views.size()))
        {
            baked_views[static_cast<std::size_t>(assignment.primaryView)] = true;
        }
    }
    result->usedViewCount = static_cast<int>(std::count(
        baked_views.cbegin(), baked_views.cend(), true));

    if (config.enableSeamLeveling)
    {
        const std::vector<TextureSeamConstraint> seam_constraints =
            buildSharedEdgeSeamConstraints(
                *data,
                atlas,
                filled_mask,
                atlas_size,
                chart_for_face,
                texture_index_by_chart_vertex,
                texture_coordinate_values);
        const TextureSeamLevelingStats seam_stats = applyTextureSeamLeveling(
            &atlas,
            chart_index_map,
            seam_constraints,
            static_cast<int>(data->charts.size()),
            config.seamBorderBlendRadiusPixels,
            config.seamMaximumLinearCorrection);
        result->seamConstraintCount = seam_stats.constraintCount;
        result->seamAdjustedChartCount = seam_stats.adjustedChartCount;
        result->seamAdjustedPixelCount = seam_stats.adjustedPixelCount;
        result->seamMaximumAppliedLinearCorrection =
            seam_stats.maximumAbsoluteLinearCorrection;
    }

    if (config.progressFn)
    {
        config.progressFn("正在扩展纹理块边界并执行局部锐化...", 88);
    }
    for (const TextureChart &chart : data->charts)
    {
        expandPadding(&atlas, &filled_mask, chart.atlasBounds, padding);
        sharpenTexture(
            &atlas,
            filled_mask,
            chart.atlasBounds,
            std::clamp(config.sharpeningStrength, 0.0f, 2.0f));
    }
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coordinates(
        static_cast<plamatrix::Index>(texture_coordinate_values.size()), 2);
    for (std::size_t index = 0;
         index < texture_coordinate_values.size();
         ++index)
    {
        texture_coordinates.setValue(
            static_cast<plamatrix::Index>(index),
            0,
            texture_coordinate_values[index][0]);
        texture_coordinates.setValue(
            static_cast<plamatrix::Index>(index),
            1,
            texture_coordinate_values[index][1]);
    }
    data->mesh->setTextureCoords(std::move(texture_coordinates));
    data->mesh->setFaceTextureIndices(std::move(texture_indices));
    data->mesh->setMaterialLibraryFile(
        QStringLiteral("textured_model.mtl").toStdString());
    data->mesh->setTextureImageFile(
        QStringLiteral("textures/model_texture.png").toStdString());

    const QString output_dir = xjw::common::io::fromUtf8Path(productsDir);
    const QString textures_dir =
        QDir(output_dir).filePath(QStringLiteral("textures"));
    if (!QDir().mkpath(textures_dir))
    {
        if (errorMsg)
        {
            *errorMsg = "无法创建纹理输出目录";
        }
        return false;
    }
    const QString token =
        QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString texture_path =
        QDir(textures_dir).filePath(QStringLiteral("model_texture.png"));
    const QString obj_path =
        QDir(output_dir).filePath(QStringLiteral("textured_model.obj"));
    const QString mtl_path =
        QDir(output_dir).filePath(QStringLiteral("textured_model.mtl"));
    const QString temporary_texture =
        QDir(textures_dir).filePath(
            QStringLiteral("model_texture.%1.tmp.png").arg(token));
    const QString temporary_obj =
        QDir(output_dir).filePath(
            QStringLiteral("textured_model.%1.tmp.obj").arg(token));
    const QString temporary_mtl =
        QDir(output_dir).filePath(
            QStringLiteral("textured_model.%1.tmp.mtl").arg(token));

    cv::Mat rgb;
    cv::cvtColor(atlas, rgb, cv::COLOR_BGR2RGB);
    const QImage texture_image(
        rgb.data,
        rgb.cols,
        rgb.rows,
        static_cast<int>(rgb.step),
        QImage::Format_RGB888);
    if (!texture_image.save(temporary_texture) ||
        !writeMaterial(temporary_mtl))
    {
        QFile::remove(temporary_texture);
        QFile::remove(temporary_mtl);
        if (errorMsg)
        {
            *errorMsg = "无法写出纹理图集或材质文件";
        }
        return false;
    }
    try
    {
        plapoint::io::writeObj<float>(
            xjw::common::io::toNativeNarrowPath(temporary_obj), *data->mesh);
    }
    catch (const std::exception &exception)
    {
        QFile::remove(temporary_texture);
        QFile::remove(temporary_mtl);
        QFile::remove(temporary_obj);
        if (errorMsg)
        {
            *errorMsg = std::string("无法写出纹理 OBJ: ") + exception.what();
        }
        return false;
    }
    if (!QFile::exists(temporary_obj) ||
        !commitFiles({
            {temporary_texture, texture_path},
            {temporary_mtl, mtl_path},
            {temporary_obj, obj_path}},
            token))
    {
        QFile::remove(temporary_texture);
        QFile::remove(temporary_mtl);
        QFile::remove(temporary_obj);
        if (errorMsg)
        {
            *errorMsg = "无法提交纹理模型输出文件";
        }
        return false;
    }

    result->modelObjPath = xjw::common::io::toUtf8Path(obj_path);
    result->modelMtlPath = xjw::common::io::toUtf8Path(mtl_path);
    result->texturePngPath = xjw::common::io::toUtf8Path(texture_path);
    result->textureSize = atlas_size;
    result->seamColorDifference = estimateSeamDifference(*data, config);
    if (config.progressFn)
    {
        config.progressFn("纹理模型生成完成", 100);
    }
    return true;
}

} // namespace xjw::mesh::texture_v4
