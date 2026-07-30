#include "TextureMappingV4Internal.h"

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
#include <unordered_map>
#include <vector>

namespace xjw::mesh::texture_v4
{
namespace
{

struct WeightedColor
{
    cv::Vec3f color{};
    float weight = 0.0f;
};

bool cancelled(const TextureMappingConfig &config)
{
    return config.isCancelled && config.isCancelled();
}

cv::Vec3f bilinearColor(const cv::Mat &image, double x, double y)
{
    const int left = std::clamp(static_cast<int>(std::floor(x)), 0, image.cols - 1);
    const int top = std::clamp(static_cast<int>(std::floor(y)), 0, image.rows - 1);
    const int right = std::min(left + 1, image.cols - 1);
    const int bottom = std::min(top + 1, image.rows - 1);
    const float fx = static_cast<float>(x - left);
    const float fy = static_cast<float>(y - top);
    const cv::Vec3f first = image.at<cv::Vec3b>(top, left);
    const cv::Vec3f second = image.at<cv::Vec3b>(top, right);
    const cv::Vec3f third = image.at<cv::Vec3b>(bottom, left);
    const cv::Vec3f fourth = image.at<cv::Vec3b>(bottom, right);
    return (first * (1.0f - fx) + second * fx) * (1.0f - fy) +
           (third * (1.0f - fx) + fourth * fx) * fy;
}

float colorDistance(const cv::Vec3f &left, const cv::Vec3f &right)
{
    const cv::Vec3f difference = left - right;
    return std::sqrt(difference.dot(difference));
}

cv::Vec3f medianColor(const std::vector<WeightedColor> &samples)
{
    cv::Vec3f median;
    for (int channel = 0; channel < 3; ++channel)
    {
        std::vector<float> values;
        values.reserve(samples.size());
        for (const WeightedColor &sample : samples)
        {
            values.push_back(sample.color[channel]);
        }
        std::sort(values.begin(), values.end());
        median[channel] = values[values.size() / 2];
    }
    return median;
}

bool sampleView(const PreparedView &view,
                const std::array<double, 3> &world,
                float candidate_score,
                int padding,
                WeightedColor *sample)
{
    double pixel[2]{};
    double depth = 0.0;
    if (!view.colorCamera.projectWorldPointWithDepth(
            world.data(), pixel, depth) ||
        pixel[0] < 0.0 || pixel[1] < 0.0 ||
        pixel[0] > view.colorBgr.cols - 1.0 ||
        pixel[1] > view.colorBgr.rows - 1.0)
    {
        return false;
    }
    const int column = static_cast<int>(std::lround(pixel[0]));
    const int row = static_cast<int>(std::lround(pixel[1]));
    const float support_distance =
        view.supportDistance.at<float>(
            std::clamp(row, 0, view.supportDistance.rows - 1),
            std::clamp(column, 0, view.supportDistance.cols - 1));
    if (support_distance <= 0.0f)
    {
        return false;
    }
    sample->color = bilinearColor(view.colorBgr, pixel[0], pixel[1]) *
        view.exposureGain;
    const float border_weight = std::clamp(
        support_distance / std::max(static_cast<float>(padding), 1.0f),
        0.10f,
        1.0f);
    sample->weight = std::max(candidate_score, 1.0e-8f) * border_weight;
    return true;
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

cv::Vec3b blendSamples(std::vector<WeightedColor> samples,
                       const TextureMappingConfig &config,
                       TextureMappingResult *result)
{
    if (samples.empty())
    {
        return {};
    }
    if (config.blendMode == TextureBlendMode::BestView)
    {
        samples.resize(1);
    }
    else if (config.enableGhostFilter && samples.size() >= 3)
    {
        const cv::Vec3f median = medianColor(samples);
        samples.erase(
            std::remove_if(samples.begin(), samples.end(), [&](const auto &sample)
            {
                if (colorDistance(sample.color, median) <=
                    config.ghostColorThreshold)
                {
                    return false;
                }
                ++result->rejectedColorOutlierCount;
                return true;
            }),
            samples.end());
    }
    if (samples.empty())
    {
        return {};
    }

    cv::Vec3f color{};
    float total_weight = 0.0f;
    for (const WeightedColor &sample : samples)
    {
        color += sample.color * sample.weight;
        total_weight += sample.weight;
    }
    color *= 1.0f / std::max(total_weight, 1.0e-8f);
    return cv::Vec3b(
        cv::saturate_cast<std::uint8_t>(color[0]),
        cv::saturate_cast<std::uint8_t>(color[1]),
        cv::saturate_cast<std::uint8_t>(color[2]));
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

void expandPadding(cv::Mat *atlas, cv::Mat *mask, int padding)
{
    if (padding <= 0)
    {
        return;
    }
    cv::Mat expanded_mask;
    const int kernel_size = padding * 2 + 1;
    cv::dilate(*mask,
               expanded_mask,
               cv::getStructuringElement(
                   cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size)));

    cv::Mat distance_source(mask->size(), CV_8UC1, cv::Scalar(255));
    distance_source.setTo(0, *mask);
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
    for (int row = 0; row < mask->rows; ++row)
    {
        for (int column = 0; column < mask->cols; ++column)
        {
            if (mask->at<std::uint8_t>(row, column) != 0)
            {
                colors[labels.at<int>(row, column)] =
                    atlas->at<cv::Vec3b>(row, column);
            }
        }
    }
    for (int row = 0; row < mask->rows; ++row)
    {
        for (int column = 0; column < mask->cols; ++column)
        {
            if (mask->at<std::uint8_t>(row, column) == 0 &&
                expanded_mask.at<std::uint8_t>(row, column) != 0)
            {
                atlas->at<cv::Vec3b>(row, column) =
                    colors[labels.at<int>(row, column)];
            }
        }
    }
    *mask = std::move(expanded_mask);
}

void sharpenTexture(cv::Mat *atlas, const cv::Mat &mask, float strength)
{
    if (strength <= 0.0f)
    {
        return;
    }
    cv::Mat blurred;
    cv::GaussianBlur(*atlas, blurred, cv::Size(), 1.0);
    cv::Mat sharpened;
    cv::addWeighted(
        *atlas, 1.0 + strength, blurred, -strength, 0.0, sharpened);
    sharpened.copyTo(*atlas, mask);
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

double estimateSeamDifference(const PipelineData &data)
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
            WeightedColor first;
            WeightedColor second;
            if (sampleView(data.views[view_index], world, 1.0f, 1, &first) &&
                sampleView(data.views[neighbor_view], world, 1.0f, 1, &second))
            {
                difference_sum += colorDistance(first.color, second.color);
                ++sample_count;
            }
        }
    }
    return sample_count > 0 ? difference_sum / sample_count : 0.0;
}

} // namespace

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
        (1.0f + fallback_size * 0.5f) / static_cast<float>(atlas_size - 1);
    const float fallback_v =
        1.0f - (1.0f + fallback_size * 0.5f) /
            static_cast<float>(atlas_size - 1);
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

    if (config.progressFn)
    {
        config.progressFn("正在烘焙多视角纹理...", 66);
    }
    int processed_faces = 0;
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
            const FaceAssignment &assignment = data->assignments[face_index];
            std::array<QPointF, 3> atlas_triangle;
            for (int corner = 0; corner < 3; ++corner)
            {
                double pixel[2]{};
                double depth = 0.0;
                if (!primary_view.colorCamera.projectWorldPointWithDepth(
                        face.vertices[corner].data(), pixel, depth))
                {
                    continue;
                }
                atlas_triangle[corner] = QPointF(
                    chart.atlasBounds.x() +
                        (pixel[0] - chart.sourceBounds.x()) * chart.atlasScale,
                    chart.atlasBounds.y() +
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
                        std::clamp(
                            static_cast<float>(atlas_triangle[corner].x()) /
                                (atlas_size - 1),
                            0.0f,
                            1.0f),
                        1.0f - std::clamp(
                            static_cast<float>(atlas_triangle[corner].y()) /
                                (atlas_size - 1),
                            0.0f,
                            1.0f)});
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
                    std::array<double, 3> world{};
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        world[axis] =
                            weights[0] * face.vertices[0][axis] +
                            weights[1] * face.vertices[1][axis] +
                            weights[2] * face.vertices[2][axis];
                    }
                    std::vector<WeightedColor> samples;
                    const int maximum_samples =
                        config.blendMode == TextureBlendMode::BestView
                        ? 1
                        : std::clamp(config.maximumBlendedViews, 1, 8);
                    auto append_sample = [&](const FaceCandidate &candidate)
                    {
                        WeightedColor sample;
                        if (sampleView(data->views[candidate.viewIndex],
                                       world,
                                       candidate.score,
                                       padding,
                                       &sample))
                        {
                            samples.push_back(sample);
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
                        if (candidate.viewIndex == assignment.primaryView)
                        {
                            continue;
                        }
                        append_sample(candidate);
                        if (static_cast<int>(samples.size()) >= maximum_samples)
                        {
                            break;
                        }
                    }
                    if (samples.empty())
                    {
                        continue;
                    }
                    atlas.at<cv::Vec3b>(row, column) =
                        blendSamples(std::move(samples), config, result);
                    filled_mask.at<std::uint8_t>(row, column) = 255;
                }
            }
        }
        if (config.progressFn)
        {
            config.progressFn(
                "正在烘焙多视角纹理...",
                66 + static_cast<int>(
                    20.0 * processed_faces / std::max(face_count, 1)));
        }
    }

    if (config.holeFillMode != TextureHoleFillMode::Disabled)
    {
        if (config.progressFn)
        {
            config.progressFn("正在扩展纹理边界并填充小孔...", 88);
        }
        expandPadding(&atlas, &filled_mask, padding);
    }
    sharpenTexture(
        &atlas, filled_mask, std::clamp(config.sharpeningStrength, 0.0f, 2.0f));
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
    result->seamColorDifference = estimateSeamDifference(*data);
    if (config.progressFn)
    {
        config.progressFn("纹理模型生成完成", 100);
    }
    return true;
}

} // namespace xjw::mesh::texture_v4
