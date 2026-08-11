#include "ObjRenderPreparation.h"
#include "ObjPointPreviewPreparation.h"

#include <QVector3D>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace
{

constexpr std::size_t kCancellationCheckInterval = 4'096;

bool isCancellationRequested(const std::atomic_bool *flag)
{
    return flag && flag->load(std::memory_order_relaxed);
}

bool shouldCancel(const std::atomic_bool *flag, std::size_t index)
{
    return index % kCancellationCheckInterval == 0
        && isCancellationRequested(flag);
}

QVector3D normalizedVector(const QVector3D &vector)
{
    const float length_squared = vector.lengthSquared();
    if (!std::isfinite(length_squared) || length_squared <= 1.0e-30f)
    {
        return QVector3D(0.0f, 0.0f, 1.0f);
    }
    return vector / std::sqrt(length_squared);
}

template <typename Value>
QByteArray byteArrayFromVector(const std::vector<Value> &values)
{
    const std::size_t byte_count = values.size() * sizeof(Value);
    if (byte_count == 0 || byte_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    QByteArray bytes(static_cast<int>(byte_count), Qt::Uninitialized);
    std::memcpy(bytes.data(), values.data(), byte_count);
    return bytes;
}

bool hasValidTextureCoordinates(const ObjRenderCloud &cloud,
                                const std::atomic_bool *cancellationFlag)
{
    if (!cloud.hasTextureCoords() || !cloud.hasFaceTextureIndices()
        || !cloud.hasFaces()
        || cloud.faceTextureIndices()->rows() != cloud.faces()->rows())
    {
        return false;
    }

    const int texture_coordinate_count = cloud.textureCoords()->rows();
    for (int face_index = 0; face_index < cloud.faceTextureIndices()->rows(); ++face_index)
    {
        if (shouldCancel(cancellationFlag, static_cast<std::size_t>(face_index)))
        {
            return false;
        }
        for (int corner = 0; corner < 3; ++corner)
        {
            const int texture_index = cloud.faceTextureIndices()->getValue(face_index, corner);
            if (texture_index < 0 || texture_index >= texture_coordinate_count)
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace

ObjRenderPreparation prepareObjRenderData(const ObjRenderCloud &cloud,
                                          bool textureImageAvailable,
                                          const std::atomic_bool *cancellationFlag,
                                          const ObjPrepareProgressCallback &progress,
                                          const ObjRenderPreparationLimits &limits)
{
    ObjRenderPreparation result;
    if (cloud.size() == 0 || !cloud.hasFaces()
        || cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || cloud.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())
        || isCancellationRequested(cancellationFlag))
    {
        return result;
    }

    const auto *faces = cloud.faces();
    const std::size_t vertex_count = cloud.size();
    const std::size_t face_count = static_cast<std::size_t>(faces->rows());
    if (limits.maximumPreviewPoints == 0 || limits.previewPointsPerChunk == 0)
    {
        return {};
    }
    if (vertex_count > limits.maximumFullMeshVertices
        || face_count > limits.maximumFullMeshFaces)
    {
        return prepareObjPointPreview(cloud, cancellationFlag, progress, limits);
    }
    result.sourceVertexCount = vertex_count;
    result.sourceFaceCount = face_count;
    result.hasTexture = textureImageAvailable
        && hasValidTextureCoordinates(cloud, cancellationFlag);
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    result.hasVertexColors = cloud.hasColors();

    double minimum_elevation = std::numeric_limits<double>::infinity();
    double maximum_elevation = -std::numeric_limits<double>::infinity();
    std::vector<QVector3D> vertex_normals(vertex_count);
    if (cloud.hasNormals())
    {
        for (std::size_t index = 0; index < vertex_count; ++index)
        {
            if (shouldCancel(cancellationFlag, index))
            {
                return {};
            }
            const auto row = static_cast<plamatrix::Index>(index);
            vertex_normals[index] = normalizedVector(QVector3D(
                cloud.normals()->getValue(row, 0),
                cloud.normals()->getValue(row, 1),
                cloud.normals()->getValue(row, 2)));
        }
    }

    std::vector<std::uint32_t> triangle_indices;
    triangle_indices.reserve(static_cast<std::size_t>(faces->rows()) * 3);
    std::vector<int> valid_face_indices;
    valid_face_indices.reserve(static_cast<std::size_t>(faces->rows()));
    std::vector<std::uint64_t> edges;
    edges.reserve(static_cast<std::size_t>(faces->rows()) * 3);
    for (int face_index = 0; face_index < faces->rows(); ++face_index)
    {
        if (shouldCancel(cancellationFlag, static_cast<std::size_t>(face_index)))
        {
            return {};
        }
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
        if (!std::isfinite(face_normal.lengthSquared())
            || face_normal.lengthSquared() <= 1.0e-30f)
        {
            continue;
        }

        const std::uint32_t indices[] = {
            static_cast<std::uint32_t>(first),
            static_cast<std::uint32_t>(second),
            static_cast<std::uint32_t>(third)};
        valid_face_indices.push_back(face_index);
        triangle_indices.insert(triangle_indices.end(), std::begin(indices), std::end(indices));
        for (int edge = 0; edge < 3; ++edge)
        {
            const std::uint32_t a = std::min(indices[edge], indices[(edge + 1) % 3]);
            const std::uint32_t b = std::max(indices[edge], indices[(edge + 1) % 3]);
            edges.push_back((static_cast<std::uint64_t>(a) << 32U) | b);
        }

        if (!cloud.hasNormals())
        {
            vertex_normals[static_cast<std::size_t>(first)] += face_normal;
            vertex_normals[static_cast<std::size_t>(second)] += face_normal;
            vertex_normals[static_cast<std::size_t>(third)] += face_normal;
        }
    }
    if (triangle_indices.empty())
    {
        return {};
    }
    if (!cloud.hasNormals())
    {
        for (std::size_t index = 0; index < vertex_normals.size(); ++index)
        {
            if (shouldCancel(cancellationFlag, index))
            {
                return {};
            }
            vertex_normals[index] = normalizedVector(vertex_normals[index]);
        }
    }

    constexpr int stride_floats = 9;
    const std::size_t vertex_float_count = vertex_count * stride_floats;
    if (vertex_float_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / sizeof(float))
    {
        return {};
    }
    std::vector<float> vertices(vertex_float_count);
    for (std::size_t index = 0; index < vertex_count; ++index)
    {
        if (shouldCancel(cancellationFlag, index))
        {
            return {};
        }
        const auto row = static_cast<plamatrix::Index>(index);
        float *vertex = vertices.data() + index * stride_floats;
        vertex[0] = cloud.points()(row, 0);
        vertex[1] = cloud.points()(row, 1);
        vertex[2] = cloud.points()(row, 2);
        vertex[3] = vertex_normals[index].x();
        vertex[4] = vertex_normals[index].y();
        vertex[5] = vertex_normals[index].z();
        if (cloud.hasColors())
        {
            vertex[6] = cloud.colors()->getValue(row, 0) / 255.0f;
            vertex[7] = cloud.colors()->getValue(row, 1) / 255.0f;
            vertex[8] = cloud.colors()->getValue(row, 2) / 255.0f;
        }
        else
        {
            vertex[6] = 239.0f / 255.0f;
            vertex[7] = 236.0f / 255.0f;
            vertex[8] = 224.0f / 255.0f;
        }
        const double elevation = vertex[2];
        if (std::isfinite(elevation))
        {
            minimum_elevation = std::min(minimum_elevation, elevation);
            maximum_elevation = std::max(maximum_elevation, elevation);
        }
    }

    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    std::sort(edges.begin(), edges.end());
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    std::vector<std::uint32_t> wireframe_indices;
    wireframe_indices.reserve(edges.size() * 2);
    for (std::size_t index = 0; index < edges.size(); ++index)
    {
        if (shouldCancel(cancellationFlag, index))
        {
            return {};
        }
        const std::uint64_t edge = edges[index];
        wireframe_indices.push_back(static_cast<std::uint32_t>(edge >> 32U));
        wireframe_indices.push_back(static_cast<std::uint32_t>(edge));
    }

    result.vertexData = byteArrayFromVector(vertices);
    result.triangleIndexData = byteArrayFromVector(triangle_indices);
    result.wireframeIndexData = byteArrayFromVector(wireframe_indices);
    if (result.vertexData.isEmpty() || result.triangleIndexData.isEmpty()
        || result.wireframeIndexData.isEmpty())
    {
        return {};
    }
    result.vertexCount = static_cast<int>(vertex_count);
    result.triangleIndexCount = static_cast<int>(triangle_indices.size());
    result.wireframeIndexCount = static_cast<int>(wireframe_indices.size());
    result.elevationRange = {minimum_elevation, maximum_elevation};

    if (!result.hasTexture)
    {
        return result;
    }

    constexpr int texture_stride_floats = 11;
    const std::size_t textured_vertex_count = triangle_indices.size();
    const std::size_t texture_float_count = textured_vertex_count * texture_stride_floats;
    if (textured_vertex_count > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || texture_float_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / sizeof(float))
    {
        result.hasTexture = false;
        return result;
    }

    std::vector<float> textured_vertices;
    textured_vertices.reserve(texture_float_count);
    for (std::size_t valid_index = 0;
         valid_index < valid_face_indices.size();
         ++valid_index)
    {
        if (shouldCancel(cancellationFlag, valid_index))
        {
            return {};
        }
        const int face_index = valid_face_indices[valid_index];
        const int face_indices[] = {
            faces->getValue(face_index, 0),
            faces->getValue(face_index, 1),
            faces->getValue(face_index, 2)};
        for (int corner = 0; corner < 3; ++corner)
        {
            const int vertex_index = face_indices[corner];
            const auto row = static_cast<plamatrix::Index>(vertex_index);
            textured_vertices.push_back(cloud.points()(row, 0));
            textured_vertices.push_back(cloud.points()(row, 1));
            textured_vertices.push_back(cloud.points()(row, 2));
            const QVector3D &normal = vertex_normals[static_cast<std::size_t>(vertex_index)];
            textured_vertices.push_back(normal.x());
            textured_vertices.push_back(normal.y());
            textured_vertices.push_back(normal.z());
            if (cloud.hasColors())
            {
                textured_vertices.push_back(cloud.colors()->getValue(row, 0) / 255.0f);
                textured_vertices.push_back(cloud.colors()->getValue(row, 1) / 255.0f);
                textured_vertices.push_back(cloud.colors()->getValue(row, 2) / 255.0f);
            }
            else
            {
                textured_vertices.insert(textured_vertices.end(), {-1.0f, -1.0f, -1.0f});
            }
            const int texture_index = cloud.faceTextureIndices()->getValue(face_index, corner);
            textured_vertices.push_back(cloud.textureCoords()->getValue(texture_index, 0));
            textured_vertices.push_back(cloud.textureCoords()->getValue(texture_index, 1));
        }
    }
    if (isCancellationRequested(cancellationFlag))
    {
        return {};
    }
    result.texturedVertexData = byteArrayFromVector(textured_vertices);
    result.texturedVertexCount = static_cast<int>(textured_vertices.size() / texture_stride_floats);
    if (result.texturedVertexCount == 0)
    {
        result.hasTexture = false;
    }
    return result;
}
