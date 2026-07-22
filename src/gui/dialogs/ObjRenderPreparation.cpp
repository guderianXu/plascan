#include "ObjRenderPreparation.h"

#include <QVector3D>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

ObjRenderPreparation prepareObjRenderData(const ObjRenderCloud &cloud,
                                          bool textureImageAvailable)
{
    ObjRenderPreparation result;
    if (cloud.size() == 0 || !cloud.hasFaces())
    {
        return result;
    }

    const auto *faces = cloud.faces();
    const std::size_t vertex_count = cloud.size();
    bool has_texture = textureImageAvailable && cloud.hasTextureCoords()
        && cloud.hasFaceTextureIndices()
        && cloud.faceTextureIndices()->rows() == faces->rows();
    if (has_texture)
    {
        const int texture_coordinate_count = cloud.textureCoords()->rows();
        for (int face_index = 0;
             face_index < cloud.faceTextureIndices()->rows() && has_texture;
             ++face_index)
        {
            for (int corner = 0; corner < 3; ++corner)
            {
                const int texture_index = cloud.faceTextureIndices()->getValue(
                    face_index, corner);
                if (texture_index < 0 || texture_index >= texture_coordinate_count)
                {
                    has_texture = false;
                    break;
                }
            }
        }
    }

    std::vector<QVector3D> vertex_normals(vertex_count);
    if (cloud.hasNormals())
    {
        for (std::size_t index = 0; index < vertex_count; ++index)
        {
            const auto row = static_cast<plamatrix::Index>(index);
            vertex_normals[index] = QVector3D(
                cloud.normals()->getValue(row, 0),
                cloud.normals()->getValue(row, 1),
                cloud.normals()->getValue(row, 2));
        }
    }
    else
    {
        for (int face_index = 0; face_index < faces->rows(); ++face_index)
        {
            const int first = faces->getValue(face_index, 0);
            const int second = faces->getValue(face_index, 1);
            const int third = faces->getValue(face_index, 2);
            if (first < 0 || second < 0 || third < 0
                || static_cast<std::size_t>(first) >= vertex_count
                || static_cast<std::size_t>(second) >= vertex_count
                || static_cast<std::size_t>(third) >= vertex_count)
            {
                continue;
            }
            const QVector3D p0(cloud.points()(first, 0),
                               cloud.points()(first, 1),
                               cloud.points()(first, 2));
            const QVector3D p1(cloud.points()(second, 0),
                               cloud.points()(second, 1),
                               cloud.points()(second, 2));
            const QVector3D p2(cloud.points()(third, 0),
                               cloud.points()(third, 1),
                               cloud.points()(third, 2));
            const QVector3D face_normal = QVector3D::crossProduct(p1 - p0, p2 - p0);
            vertex_normals[static_cast<std::size_t>(first)] += face_normal;
            vertex_normals[static_cast<std::size_t>(second)] += face_normal;
            vertex_normals[static_cast<std::size_t>(third)] += face_normal;
        }
        for (QVector3D &normal : vertex_normals)
        {
            normal.normalize();
        }
    }

    const int stride_floats = has_texture ? 11 : 9;
    const std::int64_t render_vertex_count = static_cast<std::int64_t>(faces->rows()) * 3;
    const std::int64_t byte_count = render_vertex_count * stride_floats
        * static_cast<std::int64_t>(sizeof(float));
    if (render_vertex_count <= 0 || render_vertex_count > std::numeric_limits<int>::max()
        || byte_count > std::numeric_limits<int>::max())
    {
        return result;
    }
    result.vertexData.resize(static_cast<int>(byte_count));
    float *output = reinterpret_cast<float *>(result.vertexData.data());

    const bool has_colors = cloud.hasColors();
    int appended_vertices = 0;
    for (int face_index = 0; face_index < faces->rows(); ++face_index)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const int vertex_index = faces->getValue(face_index, corner);
            if (vertex_index < 0 || static_cast<std::size_t>(vertex_index) >= vertex_count)
            {
                continue;
            }
            *output++ = cloud.points()(vertex_index, 0);
            *output++ = cloud.points()(vertex_index, 1);
            *output++ = cloud.points()(vertex_index, 2);
            const QVector3D &normal = vertex_normals[static_cast<std::size_t>(vertex_index)];
            *output++ = normal.x();
            *output++ = normal.y();
            *output++ = normal.z();
            if (has_colors)
            {
                *output++ = cloud.colors()->getValue(vertex_index, 0) / 255.0f;
                *output++ = cloud.colors()->getValue(vertex_index, 1) / 255.0f;
                *output++ = cloud.colors()->getValue(vertex_index, 2) / 255.0f;
            }
            else
            {
                *output++ = 0.55f;
                *output++ = 0.55f;
                *output++ = 0.58f;
            }
            if (has_texture)
            {
                const int texture_index = cloud.faceTextureIndices()->getValue(
                    face_index, corner);
                *output++ = cloud.textureCoords()->getValue(texture_index, 0);
                *output++ = cloud.textureCoords()->getValue(texture_index, 1);
            }
            ++appended_vertices;
        }
    }

    result.vertexCount = appended_vertices;
    result.strideBytes = stride_floats * static_cast<int>(sizeof(float));
    result.hasTexture = has_texture;
    result.vertexData.resize(result.vertexCount * result.strideBytes);
    return result;
}
