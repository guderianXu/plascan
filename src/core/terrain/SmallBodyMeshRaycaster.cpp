#include "SmallBodyMeshRaycaster.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace xjw
{

namespace
{

constexpr std::size_t kLeafTriangleCount = 8;
constexpr double kDirectionTolerance = 1.0e-6;
constexpr double kBarycentricTolerance = 1.0e-10;
constexpr double kParallelTolerance = 1.0e-12;
constexpr double kDuplicateHitTolerance = 1.0e-9;

bool fail(QString *errorMsg, const QString &message)
{
    if (errorMsg)
    {
        *errorMsg = message;
    }
    return false;
}

bool isFinite(const cv::Vec3d &value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

std::uint8_t interpolatedByte(double value)
{
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0, 255.0)));
}

} // namespace

class SmallBodyMeshRaycaster::Impl
{
public:
    bool build(const TerrainMeshInput &input,
               const cv::Vec3d &bodyCenter,
               QString *errorMsg,
               const std::atomic_bool *cancelFlag);
    bool intersect(const cv::Vec3d &direction, Hit *hit) const;

private:
    struct Aabb
    {
        cv::Vec3d minimum = cv::Vec3d(
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity());
        cv::Vec3d maximum = cv::Vec3d(
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity());

        void include(const cv::Vec3d &point);
        void include(const Aabb &other);
    };

    struct Triangle
    {
        cv::Vec3d origin;
        cv::Vec3d edge1;
        cv::Vec3d edge2;
        cv::Vec3d normal;
        std::array<std::size_t, 3> vertexIndices{};
        std::array<std::size_t, 3> textureIndices{};
        std::size_t faceIndex = 0;
    };

    struct BuildPrimitive
    {
        Aabb bounds;
        cv::Vec3d centroid;
        std::size_t triangleIndex = 0;
    };

    struct Node
    {
        Aabb bounds;
        std::size_t first = 0;
        std::size_t count = 0;
        std::size_t left = 0;
        std::size_t right = 0;
    };

    std::size_t buildNode(std::vector<BuildPrimitive> &primitives,
                          std::size_t first,
                          std::size_t count);
    bool intersectsAabb(const Aabb &bounds,
                        const cv::Vec3d &direction,
                        double maximumDistance,
                        double *nearDistance) const;
    bool intersectsTriangle(const Triangle &triangle,
                            const cv::Vec3d &direction,
                            double maximumDistance,
                            double *distance,
                            cv::Vec3d *barycentric) const;
    cv::Vec3b sampleColor(const Triangle &triangle, const cv::Vec3d &barycentric) const;
    cv::Vec3b sampleTexture(const cv::Vec2d &uv) const;

    std::vector<Triangle> _triangles;
    std::vector<std::size_t> _triangleOrder;
    std::vector<Node> _nodes;
    std::vector<cv::Vec3b> _vertexColors;
    std::vector<cv::Vec2d> _textureCoordinates;
    cv::Mat _texture;
    double _rayEpsilon = std::numeric_limits<double>::epsilon();
    const std::atomic_bool *_buildCancelFlag = nullptr;
    bool _buildCancelled = false;
};

void SmallBodyMeshRaycaster::Impl::Aabb::include(const cv::Vec3d &point)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        minimum[axis] = std::min(minimum[axis], point[axis]);
        maximum[axis] = std::max(maximum[axis], point[axis]);
    }
}

void SmallBodyMeshRaycaster::Impl::Aabb::include(const Aabb &other)
{
    include(other.minimum);
    include(other.maximum);
}

bool SmallBodyMeshRaycaster::Impl::build(const TerrainMeshInput &input,
                                         const cv::Vec3d &bodyCenter,
                                         QString *errorMsg,
                                         const std::atomic_bool *cancelFlag)
{
    _buildCancelFlag = cancelFlag;
    _buildCancelled = false;
    if (!isFinite(bodyCenter))
    {
        return fail(errorMsg, QStringLiteral("小天体体心包含非有限坐标"));
    }
    const auto &mesh = input.mesh;
    if (mesh.size() == 0 || !mesh.hasFaces() || mesh.faces()->rows() == 0)
    {
        return fail(errorMsg, QStringLiteral("网格必须包含顶点和三角面"));
    }

    if (mesh.points().cols() != 3 || mesh.faces()->cols() != 3)
    {
        return fail(errorMsg, QStringLiteral("网格顶点和三角面必须分别为 Nx3 与 Fx3"));
    }

    std::vector<cv::Vec3d> vertices(mesh.size());
    double maximum_radius = 0.0;
    for (std::size_t index = 0; index < mesh.size(); ++index)
    {
        if ((index & 4095U) == 0U && _buildCancelFlag
            && _buildCancelFlag->load(std::memory_order_relaxed))
        {
            return fail(errorMsg, QStringLiteral("构建网格射线器已取消"));
        }
        const auto row = static_cast<plamatrix::Index>(index);
        const cv::Vec3d point(mesh.points().getValue(row, 0),
                             mesh.points().getValue(row, 1),
                             mesh.points().getValue(row, 2));
        if (!isFinite(point))
        {
            return fail(errorMsg, QStringLiteral("网格顶点 %1 包含非有限坐标").arg(index));
        }
        vertices[index] = point;
        const cv::Vec3d relative = point - bodyCenter;
        if (!isFinite(relative))
        {
            return fail(errorMsg, QStringLiteral("网格顶点 %1 无法相对体心表示").arg(index));
        }
        maximum_radius = std::max(maximum_radius,
                                  std::hypot(relative[0], relative[1], relative[2]));
    }
    if (!std::isfinite(maximum_radius))
    {
        return fail(errorMsg, QStringLiteral("网格相对体心的尺度超出数值范围"));
    }
    _rayEpsilon = 128.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, maximum_radius);

    if (mesh.hasColors())
    {
        if (mesh.colors()->rows() != static_cast<plamatrix::Index>(mesh.size())
            || mesh.colors()->cols() != 3)
        {
            return fail(errorMsg, QStringLiteral("顶点颜色必须与顶点数一致且为 RGB 三通道"));
        }
        _vertexColors.resize(mesh.size());
        for (std::size_t index = 0; index < mesh.size(); ++index)
        {
            const auto row = static_cast<plamatrix::Index>(index);
            _vertexColors[index] = cv::Vec3b(mesh.colors()->getValue(row, 2),
                                             mesh.colors()->getValue(row, 1),
                                             mesh.colors()->getValue(row, 0));
        }
    }

    const bool face_uv = mesh.hasFaceTextureIndices();
    const bool point_uv = mesh.hasPointAlignedTextureCoords();
    const bool use_texture = !input.texture.empty() && mesh.hasTextureCoords() && (face_uv || point_uv);
    if (use_texture)
    {
        if (mesh.textureCoords()->cols() != 2
            || (face_uv && (mesh.faceTextureIndices()->cols() != 3
                || mesh.faceTextureIndices()->rows() != mesh.faces()->rows())))
        {
            return fail(errorMsg, QStringLiteral("UV 表或每面纹理索引的形状无效"));
        }
        if (input.texture.depth() != CV_8U || (input.texture.channels() != 1
            && input.texture.channels() != 3 && input.texture.channels() != 4))
        {
            return fail(errorMsg, QStringLiteral("纹理必须是 8 位单通道、BGR 或 BGRA 图像"));
        }
        if (input.texture.channels() == 1)
        {
            cv::cvtColor(input.texture, _texture, cv::COLOR_GRAY2BGR);
        }
        else if (input.texture.channels() == 4)
        {
            cv::cvtColor(input.texture, _texture, cv::COLOR_BGRA2BGR);
        }
        else
        {
            _texture = input.texture.clone();
        }
        _textureCoordinates.resize(static_cast<std::size_t>(mesh.textureCoords()->rows()));
        for (std::size_t index = 0; index < _textureCoordinates.size(); ++index)
        {
            const auto row = static_cast<plamatrix::Index>(index);
            _textureCoordinates[index] = cv::Vec2d(mesh.textureCoords()->getValue(row, 0),
                                                   mesh.textureCoords()->getValue(row, 1));
            if (!std::isfinite(_textureCoordinates[index][0])
                || !std::isfinite(_textureCoordinates[index][1]))
            {
                return fail(errorMsg, QStringLiteral("UV 坐标 %1 包含非有限值").arg(index));
            }
            constexpr double coordinate_tolerance = 1.0e-9;
            if (_textureCoordinates[index][0] < -coordinate_tolerance
                || _textureCoordinates[index][0] > 1.0 + coordinate_tolerance
                || _textureCoordinates[index][1] < -coordinate_tolerance
                || _textureCoordinates[index][1] > 1.0 + coordinate_tolerance)
            {
                return fail(
                    errorMsg,
                    QStringLiteral(
                        "UV 坐标 %1 超出 [0,1]；当前全球 DOM 尚未解析 MTL 的纹理重复/夹取语义，"
                        "为避免静默生成错误颜色，请先烘焙为单一 [0,1] 纹理图集。")
                        .arg(index));
            }
        }
    }

    const auto *faces = mesh.faces();
    _triangles.reserve(static_cast<std::size_t>(faces->rows()));
    std::vector<BuildPrimitive> primitives;
    primitives.reserve(static_cast<std::size_t>(faces->rows()));
    for (plamatrix::Index face_index = 0; face_index < faces->rows(); ++face_index)
    {
        if ((static_cast<std::size_t>(face_index) & 4095U) == 0U
            && _buildCancelFlag && _buildCancelFlag->load(std::memory_order_relaxed))
        {
            return fail(errorMsg, QStringLiteral("构建网格射线器已取消"));
        }
        Triangle triangle;
        triangle.faceIndex = static_cast<std::size_t>(face_index);
        for (int corner = 0; corner < 3; ++corner)
        {
            const int vertex_index = faces->getValue(face_index, corner);
            if (vertex_index < 0 || static_cast<std::size_t>(vertex_index) >= vertices.size())
            {
                return fail(errorMsg, QStringLiteral("三角面 %1 的顶点索引越界").arg(face_index));
            }
            triangle.vertexIndices[corner] = static_cast<std::size_t>(vertex_index);
            if (use_texture)
            {
                const int texture_index = face_uv
                    ? mesh.faceTextureIndices()->getValue(face_index, corner) : vertex_index;
                if (texture_index < 0
                    || static_cast<std::size_t>(texture_index) >= _textureCoordinates.size())
                {
                    return fail(errorMsg, QStringLiteral("三角面 %1 的纹理索引越界").arg(face_index));
                }
                triangle.textureIndices[corner] = static_cast<std::size_t>(texture_index);
            }
        }

        triangle.origin = vertices[triangle.vertexIndices[0]] - bodyCenter;
        triangle.edge1 = vertices[triangle.vertexIndices[1]]
            - vertices[triangle.vertexIndices[0]];
        triangle.edge2 = vertices[triangle.vertexIndices[2]]
            - vertices[triangle.vertexIndices[0]];
        const cv::Vec3d cross = triangle.edge1.cross(triangle.edge2);
        const double cross_length = cv::norm(cross);
        const double edge_product = cv::norm(triangle.edge1) * cv::norm(triangle.edge2);
        if (!std::isfinite(cross_length) || cross_length <= 64.0
            * std::numeric_limits<double>::epsilon() * edge_product)
        {
            return fail(errorMsg, QStringLiteral("三角面 %1 退化或数值不稳定").arg(face_index));
        }
        triangle.normal = cross / cross_length;

        BuildPrimitive primitive;
        primitive.triangleIndex = _triangles.size();
        primitive.bounds.include(triangle.origin);
        primitive.bounds.include(triangle.origin + triangle.edge1);
        primitive.bounds.include(triangle.origin + triangle.edge2);
        primitive.centroid = triangle.origin + (triangle.edge1 + triangle.edge2) / 3.0;
        _triangles.push_back(triangle);
        primitives.push_back(primitive);
    }

    _nodes.reserve((_triangles.size() / kLeafTriangleCount + 1) * 2);
    buildNode(primitives, 0, primitives.size());
    if (_buildCancelled)
    {
        return fail(errorMsg, QStringLiteral("构建网格射线器已取消"));
    }
    _triangleOrder.resize(primitives.size());
    std::transform(primitives.begin(), primitives.end(), _triangleOrder.begin(),
                   [](const BuildPrimitive &primitive) { return primitive.triangleIndex; });
    return true;
}

std::size_t SmallBodyMeshRaycaster::Impl::buildNode(std::vector<BuildPrimitive> &primitives,
                                                     std::size_t first,
                                                     std::size_t count)
{
    if (_buildCancelFlag && _buildCancelFlag->load(std::memory_order_relaxed))
    {
        _buildCancelled = true;
        return 0;
    }
    const std::size_t node_index = _nodes.size();
    _nodes.emplace_back();
    Aabb centroid_bounds;
    for (std::size_t index = first; index < first + count; ++index)
    {
        _nodes[node_index].bounds.include(primitives[index].bounds);
        centroid_bounds.include(primitives[index].centroid);
    }
    if (count <= kLeafTriangleCount)
    {
        _nodes[node_index].first = first;
        _nodes[node_index].count = count;
        return node_index;
    }

    const cv::Vec3d extent = centroid_bounds.maximum - centroid_bounds.minimum;
    int axis = 0;
    if (extent[1] > extent[axis]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;
    const std::size_t middle = first + count / 2;
    std::nth_element(primitives.begin() + static_cast<std::ptrdiff_t>(first),
                     primitives.begin() + static_cast<std::ptrdiff_t>(middle),
                     primitives.begin() + static_cast<std::ptrdiff_t>(first + count),
                     [axis](const BuildPrimitive &left, const BuildPrimitive &right)
                     {
                         if (left.centroid[axis] != right.centroid[axis])
                         {
                             return left.centroid[axis] < right.centroid[axis];
                         }
                         return left.triangleIndex < right.triangleIndex;
                     });
    _nodes[node_index].left = buildNode(primitives, first, middle - first);
    if (_buildCancelled)
    {
        return node_index;
    }
    _nodes[node_index].right = buildNode(primitives, middle, first + count - middle);
    return node_index;
}

bool SmallBodyMeshRaycaster::Impl::intersectsAabb(const Aabb &bounds,
                                                   const cv::Vec3d &direction,
                                                   double maximumDistance,
                                                   double *nearDistance) const
{
    double near_distance = 0.0;
    double far_distance = maximumDistance;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double padding = 32.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(bounds.minimum[axis]), std::abs(bounds.maximum[axis])});
        const double minimum = bounds.minimum[axis] - padding;
        const double maximum = bounds.maximum[axis] + padding;
        if (std::abs(direction[axis]) <= std::numeric_limits<double>::epsilon())
        {
            if (minimum > 0.0 || maximum < 0.0) return false;
            continue;
        }
        double first = minimum / direction[axis];
        double second = maximum / direction[axis];
        if (first > second) std::swap(first, second);
        near_distance = std::max(near_distance, first);
        far_distance = std::min(far_distance, second);
        if (far_distance < near_distance) return false;
    }
    *nearDistance = near_distance;
    return far_distance >= _rayEpsilon;
}

bool SmallBodyMeshRaycaster::Impl::intersectsTriangle(const Triangle &triangle,
                                                       const cv::Vec3d &direction,
                                                       double maximumDistance,
                                                       double *distance,
                                                       cv::Vec3d *barycentric) const
{
    if (std::abs(triangle.normal.dot(direction)) <= kParallelTolerance) return false;
    const cv::Vec3d p = direction.cross(triangle.edge2);
    const double determinant = triangle.edge1.dot(p);
    if (determinant == 0.0) return false;
    const double inverse = 1.0 / determinant;
    const cv::Vec3d from_vertex = -triangle.origin;
    const double u = from_vertex.dot(p) * inverse;
    if (u < -kBarycentricTolerance || u > 1.0 + kBarycentricTolerance) return false;
    const cv::Vec3d q = from_vertex.cross(triangle.edge1);
    const double v = direction.dot(q) * inverse;
    if (v < -kBarycentricTolerance || u + v > 1.0 + kBarycentricTolerance) return false;
    const double ray_distance = triangle.edge2.dot(q) * inverse;
    if (!std::isfinite(ray_distance) || ray_distance <= _rayEpsilon
        || ray_distance > maximumDistance)
    {
        return false;
    }
    cv::Vec3d weights(std::clamp(1.0 - u - v, 0.0, 1.0),
                      std::clamp(u, 0.0, 1.0), std::clamp(v, 0.0, 1.0));
    weights /= weights[0] + weights[1] + weights[2];
    *distance = ray_distance;
    *barycentric = weights;
    return true;
}

cv::Vec3b SmallBodyMeshRaycaster::Impl::sampleTexture(const cv::Vec2d &uv) const
{
    const double x = std::clamp(uv[0], 0.0, 1.0) * static_cast<double>(_texture.cols - 1);
    const double y = (1.0 - std::clamp(uv[1], 0.0, 1.0)) * static_cast<double>(_texture.rows - 1);
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, _texture.cols - 1);
    const int y1 = std::min(y0 + 1, _texture.rows - 1);
    const double fx = x - x0;
    const double fy = y - y0;
    cv::Vec3d color(0.0, 0.0, 0.0);
    for (int channel = 0; channel < 3; ++channel)
    {
        color[channel] = (1.0 - fx) * (1.0 - fy) * _texture.at<cv::Vec3b>(y0, x0)[channel]
            + fx * (1.0 - fy) * _texture.at<cv::Vec3b>(y0, x1)[channel]
            + (1.0 - fx) * fy * _texture.at<cv::Vec3b>(y1, x0)[channel]
            + fx * fy * _texture.at<cv::Vec3b>(y1, x1)[channel];
    }
    return cv::Vec3b(interpolatedByte(color[0]), interpolatedByte(color[1]), interpolatedByte(color[2]));
}

cv::Vec3b SmallBodyMeshRaycaster::Impl::sampleColor(const Triangle &triangle,
                                                     const cv::Vec3d &barycentric) const
{
    if (!_texture.empty())
    {
        cv::Vec2d uv(0.0, 0.0);
        for (int corner = 0; corner < 3; ++corner)
        {
            uv += barycentric[corner] * _textureCoordinates[triangle.textureIndices[corner]];
        }
        return sampleTexture(uv);
    }
    if (!_vertexColors.empty())
    {
        cv::Vec3d color(0.0, 0.0, 0.0);
        for (int corner = 0; corner < 3; ++corner)
        {
            const cv::Vec3b &source = _vertexColors[triangle.vertexIndices[corner]];
            for (int channel = 0; channel < 3; ++channel)
            {
                color[channel] += barycentric[corner] * source[channel];
            }
        }
        return cv::Vec3b(interpolatedByte(color[0]), interpolatedByte(color[1]), interpolatedByte(color[2]));
    }
    return cv::Vec3b(128, 128, 128);
}

bool SmallBodyMeshRaycaster::Impl::intersect(const cv::Vec3d &direction, Hit *hit) const
{
    struct Candidate
    {
        double distance = std::numeric_limits<double>::infinity();
        std::size_t triangleIndex = 0;
        cv::Vec3d barycentric;
    } best;
    bool found = false;
    bool ambiguous = false;
    std::array<std::size_t, 128> stack{};
    std::size_t stack_size = 1;
    stack[0] = 0;
    while (stack_size > 0)
    {
        const Node &node = _nodes[stack[--stack_size]];
        double node_near = 0.0;
        const double search_limit = ambiguous ? best.distance : std::numeric_limits<double>::infinity();
        if (!intersectsAabb(node.bounds, direction, search_limit, &node_near)) continue;
        if (node.count == 0)
        {
            double left_near = 0.0;
            double right_near = 0.0;
            const bool hits_left = intersectsAabb(_nodes[node.left].bounds, direction,
                                                  search_limit, &left_near);
            const bool hits_right = intersectsAabb(_nodes[node.right].bounds, direction,
                                                   search_limit, &right_near);
            if (hits_left && hits_right)
            {
                const bool left_first = left_near <= right_near;
                stack[stack_size++] = left_first ? node.right : node.left;
                stack[stack_size++] = left_first ? node.left : node.right;
            }
            else if (hits_left) stack[stack_size++] = node.left;
            else if (hits_right) stack[stack_size++] = node.right;
            continue;
        }
        for (std::size_t offset = 0; offset < node.count; ++offset)
        {
            const std::size_t triangle_index = _triangleOrder[node.first + offset];
            const Triangle &triangle = _triangles[triangle_index];
            double distance = 0.0;
            cv::Vec3d barycentric;
            if (!intersectsTriangle(triangle, direction, search_limit, &distance, &barycentric)) continue;
            if (!found)
            {
                best = {distance, triangle_index, barycentric};
                found = true;
                continue;
            }
            const double tolerance = kDuplicateHitTolerance
                * std::max({1.0, std::abs(distance), std::abs(best.distance)});
            const bool same_hit = std::abs(distance - best.distance) <= tolerance;
            if (!same_hit) ambiguous = true;
            if (distance < best.distance - tolerance
                || (same_hit && triangle.faceIndex < _triangles[best.triangleIndex].faceIndex))
            {
                best = {distance, triangle_index, barycentric};
            }
        }
    }
    if (!found) return false;
    const Triangle &triangle = _triangles[best.triangleIndex];
    Hit result;
    result.radius = best.distance;
    result.faceIndex = triangle.faceIndex;
    result.barycentric = best.barycentric;
    result.colorBgr = sampleColor(triangle, best.barycentric);
    result.reliability = std::clamp(std::abs(triangle.normal.dot(direction)), 0.0, 1.0);
    result.ambiguous = ambiguous;
    *hit = result;
    return true;
}

SmallBodyMeshRaycaster::SmallBodyMeshRaycaster() = default;

SmallBodyMeshRaycaster::SmallBodyMeshRaycaster(const TerrainMeshInput &input,
                                               const cv::Vec3d &bodyCenter,
                                               QString *errorMsg)
{
    const bool initialized = initialize(input, bodyCenter, errorMsg);
    (void)initialized;
}

SmallBodyMeshRaycaster::~SmallBodyMeshRaycaster() = default;
SmallBodyMeshRaycaster::SmallBodyMeshRaycaster(SmallBodyMeshRaycaster &&) noexcept = default;
SmallBodyMeshRaycaster &SmallBodyMeshRaycaster::operator=(SmallBodyMeshRaycaster &&) noexcept = default;

bool SmallBodyMeshRaycaster::initialize(const TerrainMeshInput &input,
                                        const cv::Vec3d &bodyCenter,
                                        QString *errorMsg,
                                        const std::atomic_bool *cancelFlag)
{
    try
    {
        auto candidate = std::make_unique<Impl>();
        if (!candidate->build(input, bodyCenter, errorMsg, cancelFlag)) return false;
        _impl = std::move(candidate);
        if (errorMsg) errorMsg->clear();
        return true;
    }
    catch (const std::exception &exception)
    {
        return fail(errorMsg, QStringLiteral("构建网格射线器失败: %1")
            .arg(QString::fromUtf8(exception.what())));
    }
    catch (...)
    {
        return fail(errorMsg, QStringLiteral("构建网格射线器时发生未知错误"));
    }
}

bool SmallBodyMeshRaycaster::isInitialized() const noexcept
{
    return static_cast<bool>(_impl);
}

bool SmallBodyMeshRaycaster::intersect(const cv::Vec3d &unitDirection, Hit *hit) const
{
    if (!_impl || !hit || !isFinite(unitDirection)) return false;
    const double squared_norm = unitDirection.dot(unitDirection);
    if (!std::isfinite(squared_norm) || std::abs(squared_norm - 1.0) > kDirectionTolerance)
    {
        return false;
    }
    return _impl->intersect(unitDirection / std::sqrt(squared_norm), hit);
}

} // namespace xjw
