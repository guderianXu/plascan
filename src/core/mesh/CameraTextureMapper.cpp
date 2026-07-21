#include "TextureMapper.h"

#include "MeshColorizer.h"
#include "io/PathIO.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/obj_io.h>
#include <plapoint/io/ply_io.h>
#include <plamatrix/dense/dense_matrix.h>

#include <opencv2/imgproc.hpp>

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QPainter>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>

namespace xjw::mesh
{
namespace
{

using PlaPointCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

struct AtlasTile
{
    float scale = 1.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;
};

std::shared_ptr<PlaPointCloud> loadMesh(const std::string &mesh_path)
{
    const QString suffix = QFileInfo(xjw::common::io::fromUtf8Path(mesh_path)).suffix().toLower();
    if (suffix == QStringLiteral("ply"))
    {
        return plapoint::io::readPly<float>(xjw::common::io::toNativeNarrowPath(mesh_path));
    }
    return plapoint::io::readObj<float>(xjw::common::io::toNativeNarrowPath(mesh_path));
}

QImage toRgbImage(const cv::Mat &bgr)
{
    if (bgr.type() != CV_8UC3)
    {
        return {};
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data,
                  rgb.cols,
                  rgb.rows,
                  static_cast<int>(rgb.step),
                  QImage::Format_RGB888).copy();
}

float meshDiagonal(const PlaPointCloud &mesh)
{
    std::array<float, 3> minimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    std::array<float, 3> maximum{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    for (std::size_t index = 0; index < mesh.size(); ++index)
    {
        const auto point = mesh[index];
        const std::array<float, 3> position{point.x(), point.y(), point.z()};
        for (int axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], position[axis]);
            maximum[axis] = std::max(maximum[axis], position[axis]);
        }
    }
    const float dx = maximum[0] - minimum[0];
    const float dy = maximum[1] - minimum[1];
    const float dz = maximum[2] - minimum[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::array<double, 3> faceVertex(const PlaPointCloud &mesh, int index)
{
    return {mesh.points().getValue(index, 0),
            mesh.points().getValue(index, 1),
            mesh.points().getValue(index, 2)};
}

std::array<float, 3> faceNormal(const std::array<double, 3> &a,
                                const std::array<double, 3> &b,
                                const std::array<double, 3> &c)
{
    const float ux = static_cast<float>(b[0] - a[0]);
    const float uy = static_cast<float>(b[1] - a[1]);
    const float uz = static_cast<float>(b[2] - a[2]);
    const float vx = static_cast<float>(c[0] - a[0]);
    const float vy = static_cast<float>(c[1] - a[1]);
    const float vz = static_cast<float>(c[2] - a[2]);
    std::array<float, 3> normal{uy * vz - uz * vy,
                                uz * vx - ux * vz,
                                ux * vy - uy * vx};
    const float length = std::sqrt(normal[0] * normal[0] +
                                   normal[1] * normal[1] +
                                   normal[2] * normal[2]);
    if (length > 1.0e-10f)
    {
        normal[0] /= length; normal[1] /= length; normal[2] /= length;
    }
    return normal;
}

float cameraScore(const MeshColorView &view,
                  const std::array<double, 3> &centroid,
                  const std::array<float, 3> &normal,
                  float absolute_tolerance)
{
    double pixel[2]{};
    double camera_depth = 0.0;
    if (!view.camera.projectWorldPointWithDepth(centroid.data(), pixel, camera_depth))
    {
        return -1.0f;
    }
    const int column = static_cast<int>(std::lround(pixel[0]));
    const int row = static_cast<int>(std::lround(pixel[1]));
    if (row < 0 || column < 0 || row >= view.depth.rows || column >= view.depth.cols ||
        view.supportMask.at<std::uint8_t>(row, column) == 0 ||
        view.depthValidMask.at<std::uint8_t>(row, column) == 0)
    {
        return -1.0f;
    }
    const float observed_depth = view.depth.at<float>(row, column);
    const float confidence = view.confidence.at<float>(row, column);
    const float tolerance = std::max(absolute_tolerance,
                                     0.008f * std::fabs(static_cast<float>(camera_depth)));
    const float residual = std::fabs(observed_depth - static_cast<float>(camera_depth));
    if (!std::isfinite(observed_depth) || observed_depth <= 0.0f ||
        !std::isfinite(confidence) || confidence < 0.25f || residual > tolerance)
    {
        return -1.0f;
    }
    const std::array<double, 3> center = view.camera.cameraCenter();
    float dx = static_cast<float>(center[0] - centroid[0]);
    float dy = static_cast<float>(center[1] - centroid[1]);
    float dz = static_cast<float>(center[2] - centroid[2]);
    const float length = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (length <= 1.0e-10f)
    {
        return -1.0f;
    }
    dx /= length; dy /= length; dz /= length;
    const float cosine = std::fabs(normal[0] * dx + normal[1] * dy + normal[2] * dz);
    if (cosine < 0.15f)
    {
        return -1.0f;
    }
    const float residual_score = 1.0f /
        std::pow(1.0f + residual / std::max(tolerance, 1.0e-8f), 2.0f);
    return confidence * std::max(0.0f, view.qualityWeight) *
           std::pow(cosine, 4.0f) * residual_score;
}

bool projectFace(const MeshColorView &view,
                 const std::array<std::array<double, 3>, 3> &vertices,
                 std::array<std::array<double, 2>, 3> *pixels)
{
    if (!pixels)
    {
        return false;
    }
    for (int corner = 0; corner < 3; ++corner)
    {
        double depth = 0.0;
        if (!view.camera.projectWorldPointWithDepth(
                vertices[corner].data(), (*pixels)[corner].data(), depth))
        {
            return false;
        }
        const double x = (*pixels)[corner][0];
        const double y = (*pixels)[corner][1];
        if (x < 0.0 || y < 0.0 || x > view.colorBgr.cols - 1.0 ||
            y > view.colorBgr.rows - 1.0)
        {
            return false;
        }
    }
    return true;
}

float relaxedCameraScore(const MeshColorView &view,
                         const std::array<double, 3> &centroid,
                         const std::array<float, 3> &normal)
{
    double pixel[2]{};
    double camera_depth = 0.0;
    if (!view.camera.projectWorldPointWithDepth(centroid.data(), pixel, camera_depth))
    {
        return -1.0f;
    }
    const int column = static_cast<int>(std::lround(pixel[0]));
    const int row = static_cast<int>(std::lround(pixel[1]));
    if (row < 0 || column < 0 || row >= view.colorBgr.rows || column >= view.colorBgr.cols)
    {
        return -1.0f;
    }
    const std::array<double, 3> center = view.camera.cameraCenter();
    float dx = static_cast<float>(center[0] - centroid[0]);
    float dy = static_cast<float>(center[1] - centroid[1]);
    float dz = static_cast<float>(center[2] - centroid[2]);
    const float length = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (length <= 1.0e-10f)
    {
        return -1.0f;
    }
    dx /= length; dy /= length; dz /= length;
    const float cosine = std::fabs(normal[0] * dx + normal[1] * dy + normal[2] * dz);
    const bool supported = view.supportMask.type() == CV_8UC1 &&
        view.supportMask.at<std::uint8_t>(row, column) != 0;
    return std::pow(cosine, 4.0f) * (supported ? 2.0f : 1.0f);
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

} // namespace

bool TextureMapper::generateCameraTexturedModelFromMeshFile(
    const std::string &meshPath,
    const std::string &productsDir,
    const TextureMappingConfig &config,
    const QVector<MeshColorView> &views,
    TextureMappingResult *result,
    std::string *errorMsg)
{
    if (result)
    {
        *result = TextureMappingResult();
    }
    std::shared_ptr<PlaPointCloud> mesh = loadMesh(meshPath);
    if (!mesh || !mesh->hasFaces() || views.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "相机纹理映射缺少网格面或 MVS 相机影像";
        }
        return false;
    }

    const int texture_size = std::clamp(config.textureSize, 1024, 16384);
    const int view_count = static_cast<int>(views.size());
    const int columns = std::max(1, static_cast<int>(std::ceil(std::sqrt(view_count))));
    const int rows = std::max(1, (view_count + columns - 1) / columns);
    const int cell_width = texture_size / columns;
    const int cell_height = texture_size / rows;
    const int padding = std::clamp(config.padding, 2, 32);
    QImage atlas(texture_size, texture_size, QImage::Format_RGB32);
    atlas.fill(Qt::black);
    QPainter painter(&atlas);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    std::vector<AtlasTile> tiles(static_cast<std::size_t>(views.size()));
    for (int view_index = 0; view_index < views.size(); ++view_index)
    {
        const QImage image = toRgbImage(views[view_index].colorBgr);
        if (image.isNull())
        {
            continue;
        }
        const int cell_x = (view_index % columns) * cell_width;
        const int cell_y = (view_index / columns) * cell_height;
        const float scale = std::min(
            static_cast<float>(cell_width - padding * 2) / image.width(),
            static_cast<float>(cell_height - padding * 2) / image.height());
        const int draw_width = std::max(1, static_cast<int>(std::lround(image.width() * scale)));
        const int draw_height = std::max(1, static_cast<int>(std::lround(image.height() * scale)));
        const int offset_x = cell_x + (cell_width - draw_width) / 2;
        const int offset_y = cell_y + (cell_height - draw_height) / 2;
        painter.drawImage(QRect(offset_x, offset_y, draw_width, draw_height), image);
        tiles[static_cast<std::size_t>(view_index)] = {
            scale, static_cast<float>(offset_x), static_cast<float>(offset_y)};
    }
    painter.end();

    auto *faces = mesh->faces();
    const int face_count = static_cast<int>(faces->rows());
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coordinates(
        static_cast<plamatrix::Index>(face_count) * 3, 2);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> texture_indices(face_count, 3);
    const float absolute_tolerance = std::max(meshDiagonal(*mesh) / 64.0f, 1.0e-7f);
    int mapped_faces = 0;
    int fallback_mapped_faces = 0;
    int unmapped_faces = 0;
    for (int face_index = 0; face_index < face_count; ++face_index)
    {
        if (config.progressFn && (face_index % 4096 == 0 || face_index + 1 == face_count))
        {
            config.progressFn("正在选择每个三角面的最佳相机...",
                              20 + static_cast<int>(65.0 * (face_index + 1) /
                                                    std::max(face_count, 1)));
        }
        const std::array<int, 3> indices{
            faces->getValue(face_index, 0),
            faces->getValue(face_index, 1),
            faces->getValue(face_index, 2)};
        const std::array<std::array<double, 3>, 3> vertices{
            faceVertex(*mesh, indices[0]),
            faceVertex(*mesh, indices[1]),
            faceVertex(*mesh, indices[2])};
        const std::array<double, 3> centroid{
            (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3.0,
            (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3.0,
            (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3.0};
        const std::array<float, 3> normal = faceNormal(vertices[0], vertices[1], vertices[2]);
        int best_view = -1;
        float best_score = -1.0f;
        std::array<std::array<double, 2>, 3> best_pixels{};
        for (int view_index = 0; view_index < views.size(); ++view_index)
        {
            const float score = cameraScore(
                views[view_index], centroid, normal, absolute_tolerance);
            if (score <= best_score)
            {
                continue;
            }
            std::array<std::array<double, 2>, 3> pixels{};
            if (!projectFace(views[view_index], vertices, &pixels))
            {
                continue;
            }
            best_score = score;
            best_view = view_index;
            best_pixels = pixels;
        }
        bool used_fallback = false;
        if (best_view < 0)
        {
            for (int view_index = 0; view_index < views.size(); ++view_index)
            {
                const float score = relaxedCameraScore(
                    views[view_index], centroid, normal);
                if (score <= best_score)
                {
                    continue;
                }
                std::array<std::array<double, 2>, 3> pixels{};
                if (!projectFace(views[view_index], vertices, &pixels))
                {
                    continue;
                }
                best_score = score;
                best_view = view_index;
                best_pixels = pixels;
                used_fallback = true;
            }
        }

        for (int corner = 0; corner < 3; ++corner)
        {
            const int texture_index = face_index * 3 + corner;
            texture_indices.setValue(face_index, corner, texture_index);
            float u = 0.0f;
            float v = 0.0f;
            if (best_view >= 0)
            {
                const AtlasTile &tile = tiles[static_cast<std::size_t>(best_view)];
                u = (tile.offsetX + static_cast<float>(best_pixels[corner][0]) * tile.scale) /
                    (texture_size - 1);
                v = 1.0f -
                    (tile.offsetY + static_cast<float>(best_pixels[corner][1]) * tile.scale) /
                        (texture_size - 1);
            }
            texture_coordinates.setValue(texture_index, 0, std::clamp(u, 0.0f, 1.0f));
            texture_coordinates.setValue(texture_index, 1, std::clamp(v, 0.0f, 1.0f));
        }
        if (best_view >= 0)
        {
            ++mapped_faces;
            fallback_mapped_faces += used_fallback ? 1 : 0;
        }
        else
        {
            ++unmapped_faces;
        }
    }
    mesh->setTextureCoords(std::move(texture_coordinates));
    mesh->setFaceTextureIndices(std::move(texture_indices));

    const QString output_dir = xjw::common::io::fromUtf8Path(productsDir);
    const QString textures_dir = QDir(output_dir).filePath(QStringLiteral("textures"));
    QDir().mkpath(textures_dir);
    const QString texture_path = QDir(textures_dir).filePath(QStringLiteral("model_texture.png"));
    const QString obj_path = QDir(output_dir).filePath(QStringLiteral("textured_model.obj"));
    const QString mtl_path = QDir(output_dir).filePath(QStringLiteral("textured_model.mtl"));
    if (!atlas.save(texture_path) || !writeMaterial(mtl_path))
    {
        if (errorMsg)
        {
            *errorMsg = "无法写出相机纹理图集或 MTL";
        }
        return false;
    }
    mesh->setMaterialLibraryFile(QStringLiteral("textured_model.mtl").toStdString());
    mesh->setTextureImageFile(QStringLiteral("textures/model_texture.png").toStdString());
    plapoint::io::writeObj<float>(xjw::common::io::toNativeNarrowPath(obj_path), *mesh);

    if (result)
    {
        result->modelObjPath = xjw::common::io::toUtf8Path(obj_path);
        result->modelMtlPath = xjw::common::io::toUtf8Path(mtl_path);
        result->texturePngPath = xjw::common::io::toUtf8Path(texture_path);
        result->textureSize = texture_size;
        result->textureAlgorithm = "camera_projected_atlas_v1";
        result->uvMethod = "per_face_camera_projection";
        result->blendMethod = "best_view_depth_angle_score";
        result->sourceViewCount = view_count;
        result->mappedFaceCount = mapped_faces;
        result->fallbackMappedFaceCount = fallback_mapped_faces;
        result->unmappedFaceCount = unmapped_faces;
    }
    return true;
}

} // namespace xjw::mesh
