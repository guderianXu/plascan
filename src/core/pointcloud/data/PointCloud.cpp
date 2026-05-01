#include "data/PointCloud.h"
#include "data/PointCloudPoint.h"

#include "Logger.h"
#include "spatial/KDTree.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::pointcloud
{

// ── 空间索引 pimpl ───────────────────────────────────────────────────────────
struct PointCloudSpatialIndexImpl
{
    using Tree = xjw::common::spatial::KDTree<3, float>;
    Tree tree;
};

// ── 局部辅助 ─────────────────────────────────────────────────────────────────

namespace
{

Point3f makeZeroPoint()
{
    return Point3f{0.0f, 0.0f, 0.0f};
}

Point2f makeZeroTextureCoordinate()
{
    return Point2f{0.0f, 0.0f};
}

ColorRGBA makeDefaultColor()
{
    return ColorRGBA{255, 255, 255, 255};
}

PhotogrammetryPointAttributes makeDefaultPhotogrammetryAttributes()
{
    return PhotogrammetryPointAttributes{};
}

bool isValidIndex(std::size_t index, std::size_t size)
{
    return index < size;
}

Point3f applyLinearTransform(const std::array<float, 16> &matrix, const Point3f &value)
{
    return Point3f{
        matrix[0] * value.x + matrix[1] * value.y + matrix[2] * value.z,
        matrix[4] * value.x + matrix[5] * value.y + matrix[6] * value.z,
        matrix[8] * value.x + matrix[9] * value.y + matrix[10] * value.z};
}

Point3f applyPointTransform(const std::array<float, 16> &matrix, const Point3f &value)
{
    return Point3f{
        matrix[0] * value.x + matrix[1] * value.y + matrix[2] * value.z + matrix[3],
        matrix[4] * value.x + matrix[5] * value.y + matrix[6] * value.z + matrix[7],
        matrix[8] * value.x + matrix[9] * value.y + matrix[10] * value.z + matrix[11]};
}

Point3f normalizeVector(const Point3f &value)
{
    const float lengthSquared = value.x * value.x + value.y * value.y + value.z * value.z;
    if (lengthSquared <= 1e-20f)
    {
        return makeZeroPoint();
    }

    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
    return Point3f{value.x * inverseLength, value.y * inverseLength, value.z * inverseLength};
}

bool isInsideBounds(const Point3f &point, const PointCloudBounds &bounds)
{
    return bounds.valid &&
           point.x >= bounds.minCorner.x && point.x <= bounds.maxCorner.x &&
           point.y >= bounds.minCorner.y && point.y <= bounds.maxCorner.y &&
           point.z >= bounds.minCorner.z && point.z <= bounds.maxCorner.z;
}

} // namespace

// ── 默认/析构函数（在 .cpp 定义，不完整类型 delete 安全） ───────────────────

PointCloud::PointCloud() = default;
PointCloud::~PointCloud() = default;

// ── 移动构造/赋值（在 .cpp 定义以满足 pimpl 完整类型要求） ──────────────────

PointCloud::PointCloud(PointCloud &&) noexcept = default;
PointCloud &PointCloud::operator=(PointCloud &&) noexcept = default;

// ── 拷贝构造/赋值（跳过空间索引缓存） ────────────────────────────────────────

PointCloud::PointCloud(const PointCloud &other)
    : _positions(other._positions)
    , _normals(other._normals)
    , _colors(other._colors)
    , _photogrammetryAttributes(other._photogrammetryAttributes)
    , _textureCoordinates(other._textureCoordinates)
    , _faces(other._faces)
    , _materialLibraryFile(other._materialLibraryFile)
    , _textureImageFile(other._textureImageFile)
    , _metadata(other._metadata)
    // _spatialIndex 故意不拷贝；懒初始化，将在需要时重建
{
}

PointCloud &PointCloud::operator=(const PointCloud &other)
{
    if (this != &other)
    {
        _positions                  = other._positions;
        _normals                    = other._normals;
        _colors                     = other._colors;
        _photogrammetryAttributes   = other._photogrammetryAttributes;
        _textureCoordinates         = other._textureCoordinates;
        _faces                      = other._faces;
        _materialLibraryFile        = other._materialLibraryFile;
        _textureImageFile           = other._textureImageFile;
        _metadata                   = other._metadata;
        _spatialIndex.reset();  // 缓存失效，将在下次查询时重建
    }
    return *this;
}

PointCloudTransform PointCloudTransform::identity()
{
    return PointCloudTransform{};
}

PointCloudTransform PointCloudTransform::fromRotationTranslation(const std::array<float, 9> &rotation,
                                                                 const Point3f &translation)
{
    PointCloudTransform transform;
    transform.matrix = {
        rotation[0], rotation[1], rotation[2], translation.x,
        rotation[3], rotation[4], rotation[5], translation.y,
        rotation[6], rotation[7], rotation[8], translation.z,
        0.0f,        0.0f,        0.0f,        1.0f};
    return transform;
}

// ── 空间索引实现 ─────────────────────────────────────────────────────────────

void PointCloud::buildSpatialIndex() const
{
    if (_spatialIndex)
    {
        return; // 已有索引，无需重建
    }
    _spatialIndex = std::make_unique<PointCloudSpatialIndexImpl>();
    std::vector<PointCloudSpatialIndexImpl::Tree::CoordinateArray> coords;
    coords.reserve(_positions.size());
    for (const auto &p : _positions)
        coords.push_back({p.x, p.y, p.z});
    _spatialIndex->tree.build(coords);
}

bool PointCloud::hasSpatialIndex() const
{
    return static_cast<bool>(_spatialIndex);
}

void PointCloud::invalidateSpatialIndex()
{
    _spatialIndex.reset();
}

std::vector<std::size_t> PointCloud::kNearestIndices(std::size_t pointIndex, std::size_t k) const
{
    buildSpatialIndex();
    auto neighbors = _spatialIndex->tree.kNearestByPointIndex(pointIndex, k);
    std::vector<std::size_t> result;
    result.reserve(neighbors.size());
    for (const auto &nb : neighbors)
    {
        if (nb.index >= 0)
            result.push_back(static_cast<std::size_t>(nb.index));
    }
    return result;
}

std::vector<std::size_t> PointCloud::kNearestIndices(const Point3f &position, std::size_t k) const
{
    buildSpatialIndex();
    PointCloudSpatialIndexImpl::Tree::CoordinateArray query{position.x, position.y, position.z};
    auto neighbors = _spatialIndex->tree.kNearest(query, k);
    std::vector<std::size_t> result;
    result.reserve(neighbors.size());
    for (const auto &nb : neighbors)
    {
        if (nb.index >= 0)
            result.push_back(static_cast<std::size_t>(nb.index));
    }
    return result;
}

std::vector<std::size_t> PointCloud::radiusSearchIndices(const Point3f &position, float radius) const
{
    buildSpatialIndex();
    PointCloudSpatialIndexImpl::Tree::CoordinateArray query{position.x, position.y, position.z};
    auto indices = _spatialIndex->tree.radiusSearch(query, radius);
    std::vector<std::size_t> result;
    result.reserve(indices.size());
    for (int idx : indices)
    {
        if (idx >= 0)
            result.push_back(static_cast<std::size_t>(idx));
    }
    return result;
}

// ── PointCloud 成员函数 ─────────────────────────────────────────────────────

std::size_t PointCloud::size() const
{
    return _positions.size();
}

bool PointCloud::empty() const
{
    return _positions.empty();
}

bool PointCloud::hasNormals() const
{
    return !_normals.empty();
}

bool PointCloud::hasColors() const
{
    return !_colors.empty();
}

bool PointCloud::hasPhotogrammetryAttributes() const
{
    return !_photogrammetryAttributes.empty();
}

bool PointCloud::hasTextureCoordinates() const
{
    return !_textureCoordinates.empty();
}

bool PointCloud::hasFaces() const
{
    return !_faces.empty();
}

bool PointCloud::isConsistent() const
{
    if (hasNormals() && _normals.size() != _positions.size())
    {
        return false;
    }

    if (hasColors() && _colors.size() != _positions.size())
    {
        return false;
    }

    if (hasPhotogrammetryAttributes() && _photogrammetryAttributes.size() != _positions.size())
    {
        return false;
    }

    if (hasTextureCoordinates() && _textureCoordinates.size() != _positions.size())
    {
        return false;
    }

    for (const PointCloudFace &face : _faces)
    {
        for (std::size_t i = 0; i < face.vertexIndices.size(); ++i)
        {
            if (face.vertexIndices[i] >= _positions.size())
            {
                return false;
            }

            if (face.hasTextureIndices && face.textureIndices[i] >= _textureCoordinates.size())
            {
                return false;
            }
        }
    }

    return true;
}

void PointCloud::clear()
{
    LOG_INFO("PointCloud::clear: removing %zu points", size());
    _spatialIndex.reset();
    _positions.clear();
    _normals.clear();
    _colors.clear();
    _photogrammetryAttributes.clear();
    _textureCoordinates.clear();
    _faces.clear();
    _materialLibraryFile.clear();
    _textureImageFile.clear();
}

void PointCloud::reserve(std::size_t pointCount)
{
    LOG_DEBUG("PointCloud::reserve: pointCount=%zu", pointCount);
    _positions.reserve(pointCount);
    _normals.reserve(pointCount);
    _colors.reserve(pointCount);
    _photogrammetryAttributes.reserve(pointCount);
    _textureCoordinates.reserve(pointCount);
}

void PointCloud::addPoint(const Point3f &position)
{
    _spatialIndex.reset();
    _positions.push_back(position);

    if (hasNormals())
    {
        _normals.push_back(makeZeroPoint());
    }

    if (hasColors())
    {
        _colors.push_back(makeDefaultColor());
    }

    if (hasPhotogrammetryAttributes())
    {
        _photogrammetryAttributes.push_back(makeDefaultPhotogrammetryAttributes());
    }

    if (hasTextureCoordinates())
    {
        _textureCoordinates.push_back(makeZeroTextureCoordinate());
    }
}

void PointCloud::addPoint(const Point3f &position, const Point3f &normal)
{
    _spatialIndex.reset();
    ensureNormalsForExistingPoints();
    _positions.push_back(position);
    _normals.push_back(normal);

    if (hasColors())
    {
        _colors.push_back(makeDefaultColor());
    }

    if (hasPhotogrammetryAttributes())
    {
        _photogrammetryAttributes.push_back(makeDefaultPhotogrammetryAttributes());
    }

    if (hasTextureCoordinates())
    {
        _textureCoordinates.push_back(makeZeroTextureCoordinate());
    }
}

void PointCloud::addPoint(const Point3f &position, const ColorRGBA &color)
{
    _spatialIndex.reset();
    ensureColorsForExistingPoints();
    _positions.push_back(position);
    _colors.push_back(color);

    if (hasNormals())
    {
        _normals.push_back(makeZeroPoint());
    }

    if (hasPhotogrammetryAttributes())
    {
        _photogrammetryAttributes.push_back(makeDefaultPhotogrammetryAttributes());
    }

    if (hasTextureCoordinates())
    {
        _textureCoordinates.push_back(makeZeroTextureCoordinate());
    }
}

void PointCloud::addPoint(const Point3f &position, const Point3f &normal, const ColorRGBA &color)
{
    _spatialIndex.reset();
    ensureNormalsForExistingPoints();
    ensureColorsForExistingPoints();
    _positions.push_back(position);
    _normals.push_back(normal);
    _colors.push_back(color);

    if (hasPhotogrammetryAttributes())
    {
        _photogrammetryAttributes.push_back(makeDefaultPhotogrammetryAttributes());
    }

    if (hasTextureCoordinates())
    {
        _textureCoordinates.push_back(makeZeroTextureCoordinate());
    }
}

void PointCloud::addPoint(const Point3f &position,
                         const Point3f &normal,
                         const ColorRGBA &color,
                         const PhotogrammetryPointAttributes &photogrammetry)
{
    ensureNormalsForExistingPoints();
    ensureColorsForExistingPoints();
    ensurePhotogrammetryAttributesForExistingPoints();
    _positions.push_back(position);
    _normals.push_back(normal);
    _colors.push_back(color);
    _photogrammetryAttributes.push_back(photogrammetry);

    if (hasTextureCoordinates())
    {
        _textureCoordinates.push_back(makeZeroTextureCoordinate());
    }
}

bool PointCloud::removePoint(std::size_t index)
{
    _spatialIndex.reset();
    if (!isValidIndex(index, size()))
    {
        return false;
    }

    _positions.erase(_positions.begin() + static_cast<std::ptrdiff_t>(index));
    if (hasNormals())
    {
        _normals.erase(_normals.begin() + static_cast<std::ptrdiff_t>(index));
    }

    if (hasColors())
    {
        _colors.erase(_colors.begin() + static_cast<std::ptrdiff_t>(index));
    }

    if (hasPhotogrammetryAttributes())
    {
        _photogrammetryAttributes.erase(_photogrammetryAttributes.begin() + static_cast<std::ptrdiff_t>(index));
    }

    if (hasTextureCoordinates())
    {
        _textureCoordinates.erase(_textureCoordinates.begin() + static_cast<std::ptrdiff_t>(index));
    }

    reindexFacesAfterPointRemoval(index);

    LOG_DEBUG("PointCloud::removePoint: index=%zu remaining=%zu", index, size());
    return true;
}

bool PointCloud::setPosition(std::size_t index, const Point3f &position)
{
    if (!isValidIndex(index, size()))
    {
        return false;
    }

    _spatialIndex.reset();
    _positions[index] = position;
    return true;
}

bool PointCloud::setNormal(std::size_t index, const Point3f &normal)
{
    if (!isValidIndex(index, size()))
    {
        return false;
    }

    ensureNormalsForExistingPoints();
    _normals[index] = normal;
    return true;
}

bool PointCloud::setColor(std::size_t index, const ColorRGBA &color)
{
    if (!isValidIndex(index, size()))
    {
        return false;
    }

    ensureColorsForExistingPoints();
    _colors[index] = color;
    return true;
}

bool PointCloud::setPhotogrammetryAttributes(std::size_t index,
                                             const PhotogrammetryPointAttributes &photogrammetry)
{
    if (!isValidIndex(index, size()))
    {
        return false;
    }

    ensurePhotogrammetryAttributesForExistingPoints();
    _photogrammetryAttributes[index] = photogrammetry;
    return true;
}

bool PointCloud::setTextureCoordinate(std::size_t index, const Point2f &textureCoordinate)
{
    if (!isValidIndex(index, size()))
    {
        return false;
    }

    ensureTextureCoordinatesForExistingPoints();
    _textureCoordinates[index] = textureCoordinate;
    return true;
}

bool PointCloud::setPoint(std::size_t index, const PointCloudPoint &point)
{
    if (!isValidIndex(index, size()))
    {
        return false;
    }

    _positions[index] = point.position;

    if (point.hasNormal)
    {
        ensureNormalsForExistingPoints();
        _normals[index] = point.normal;
    }

    if (point.hasColor)
    {
        ensureColorsForExistingPoints();
        _colors[index] = point.color;
    }

    if (point.hasPhotogrammetry)
    {
        ensurePhotogrammetryAttributesForExistingPoints();
        _photogrammetryAttributes[index] = point.photogrammetry;
    }

    if (point.hasTextureCoordinate)
    {
        ensureTextureCoordinatesForExistingPoints();
        _textureCoordinates[index] = point.textureCoordinate;
    }

    return true;
}

void PointCloud::setNormals(const std::vector<Point3f> &normals)
{
    if (!normals.empty() && normals.size() != _positions.size())
    {
        LOG_WARN("PointCloud::setNormals: size mismatch normals=%zu points=%zu", normals.size(), _positions.size());
        return;
    }

    _normals = normals;
    LOG_INFO("PointCloud::setNormals: assigned=%zu", _normals.size());
}

void PointCloud::setColors(const std::vector<ColorRGBA> &colors)
{
    if (!colors.empty() && colors.size() != _positions.size())
    {
        LOG_WARN("PointCloud::setColors: size mismatch colors=%zu points=%zu", colors.size(), _positions.size());
        return;
    }

    _colors = colors;
    LOG_INFO("PointCloud::setColors: assigned=%zu", _colors.size());
}

void PointCloud::setPhotogrammetryAttributes(const std::vector<PhotogrammetryPointAttributes> &photogrammetryAttributes)
{
    if (!photogrammetryAttributes.empty() && photogrammetryAttributes.size() != _positions.size())
    {
        LOG_WARN("PointCloud::setPhotogrammetryAttributes: size mismatch attributes=%zu points=%zu",
                 photogrammetryAttributes.size(),
                 _positions.size());
        return;
    }

    _photogrammetryAttributes = photogrammetryAttributes;
    LOG_INFO("PointCloud::setPhotogrammetryAttributes: assigned=%zu", _photogrammetryAttributes.size());
}

void PointCloud::setTextureCoordinates(const std::vector<Point2f> &textureCoordinates)
{
    if (!textureCoordinates.empty() && textureCoordinates.size() != _positions.size())
    {
        LOG_WARN("PointCloud::setTextureCoordinates: size mismatch texCoords=%zu points=%zu",
                 textureCoordinates.size(),
                 _positions.size());
        return;
    }

    _textureCoordinates = textureCoordinates;
    LOG_INFO("PointCloud::setTextureCoordinates: assigned=%zu", _textureCoordinates.size());
}

bool PointCloud::addFace(const PointCloudFace &face)
{
    for (std::size_t i = 0; i < face.vertexIndices.size(); ++i)
    {
        if (face.vertexIndices[i] >= _positions.size())
        {
            return false;
        }

        if (face.hasTextureIndices)
        {
            if (!hasTextureCoordinates() || face.textureIndices[i] >= _textureCoordinates.size())
            {
                return false;
            }
        }
    }

    _faces.push_back(face);
    return true;
}

void PointCloud::clearFaces()
{
    _faces.clear();
}

void PointCloud::setMetadata(const PointCloudMetadata &metadata)
{
    _metadata = metadata;
    LOG_INFO("PointCloud::setMetadata: name=%s frame=%d registered=%d",
             _metadata.name.c_str(),
             static_cast<int>(_metadata.coordinateFrame),
             _metadata.isRegistered ? 1 : 0);
}

void PointCloud::append(const PointCloud &other)
{
    if (other.empty())
    {
        return;
    }

    _spatialIndex.reset();

    const std::size_t oldSize = size();
    const bool mergedNormals = hasNormals() || other.hasNormals();
    const bool mergedColors = hasColors() || other.hasColors();
    const bool mergedPhotogrammetry = hasPhotogrammetryAttributes() || other.hasPhotogrammetryAttributes();

    if (mergedNormals && !hasNormals())
    {
        _normals.assign(oldSize, makeZeroPoint());
    }

    if (mergedColors && !hasColors())
    {
        _colors.assign(oldSize, makeDefaultColor());
    }

    if (mergedPhotogrammetry && !hasPhotogrammetryAttributes())
    {
        _photogrammetryAttributes.assign(oldSize, makeDefaultPhotogrammetryAttributes());
    }

    _positions.insert(_positions.end(), other._positions.begin(), other._positions.end());

    if (mergedNormals)
    {
        if (other.hasNormals())
        {
            _normals.insert(_normals.end(), other._normals.begin(), other._normals.end());
        }
        else
        {
            _normals.insert(_normals.end(), other.size(), makeZeroPoint());
        }
    }

    if (mergedColors)
    {
        if (other.hasColors())
        {
            _colors.insert(_colors.end(), other._colors.begin(), other._colors.end());
        }
        else
        {
            _colors.insert(_colors.end(), other.size(), makeDefaultColor());
        }
    }

    if (mergedPhotogrammetry)
    {
        if (other.hasPhotogrammetryAttributes())
        {
            _photogrammetryAttributes.insert(_photogrammetryAttributes.end(),
                                             other._photogrammetryAttributes.begin(),
                                             other._photogrammetryAttributes.end());
        }
        else
        {
            _photogrammetryAttributes.insert(_photogrammetryAttributes.end(),
                                             other.size(),
                                             makeDefaultPhotogrammetryAttributes());
        }
    }

    const bool mergedTextureCoordinates = hasTextureCoordinates() || other.hasTextureCoordinates();
    if (mergedTextureCoordinates)
    {
        if (!hasTextureCoordinates())
        {
            _textureCoordinates.assign(oldSize, makeZeroTextureCoordinate());
        }

        if (other.hasTextureCoordinates())
        {
            _textureCoordinates.insert(_textureCoordinates.end(),
                                       other._textureCoordinates.begin(),
                                       other._textureCoordinates.end());
        }
        else
        {
            _textureCoordinates.insert(_textureCoordinates.end(), other.size(), makeZeroTextureCoordinate());
        }
    }

    if (other.hasFaces())
    {
        _faces.reserve(_faces.size() + other._faces.size());
        for (const PointCloudFace &otherFace : other._faces)
        {
            PointCloudFace remappedFace = otherFace;
            for (std::size_t i = 0; i < remappedFace.vertexIndices.size(); ++i)
            {
                remappedFace.vertexIndices[i] += oldSize;
                if (remappedFace.hasTextureIndices)
                {
                    remappedFace.textureIndices[i] += oldSize;
                }
            }
            _faces.push_back(remappedFace);
        }
    }

    if (_materialLibraryFile.empty())
    {
        _materialLibraryFile = other._materialLibraryFile;
    }
    if (_textureImageFile.empty())
    {
        _textureImageFile = other._textureImageFile;
    }

    LOG_INFO("PointCloud::append: old=%zu appended=%zu total=%zu", oldSize, other.size(), size());
}

PointCloudBounds PointCloud::computeBounds() const
{
    PointCloudBounds bounds;
    if (_positions.empty())
    {
        return bounds;
    }

    bounds.minCorner = _positions.front();
    bounds.maxCorner = _positions.front();
    bounds.valid = true;

    for (const Point3f &point : _positions)
    {
        bounds.minCorner.x = std::min(bounds.minCorner.x, point.x);
        bounds.minCorner.y = std::min(bounds.minCorner.y, point.y);
        bounds.minCorner.z = std::min(bounds.minCorner.z, point.z);
        bounds.maxCorner.x = std::max(bounds.maxCorner.x, point.x);
        bounds.maxCorner.y = std::max(bounds.maxCorner.y, point.y);
        bounds.maxCorner.z = std::max(bounds.maxCorner.z, point.z);
    }

    return bounds;
}

Point3f PointCloud::computeCentroid() const
{
    if (_positions.empty())
    {
        return makeZeroPoint();
    }

    Point3f centroid;
    for (const Point3f &point : _positions)
    {
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
    }

    const float inverseSize = 1.0f / static_cast<float>(_positions.size());
    centroid.x *= inverseSize;
    centroid.y *= inverseSize;
    centroid.z *= inverseSize;
    return centroid;
}

void PointCloud::translate(const Point3f &offset)
{
    for (Point3f &point : _positions)
    {
        point.x += offset.x;
        point.y += offset.y;
        point.z += offset.z;
    }

    _spatialIndex.reset();
    LOG_INFO("PointCloud::translate: offset=(%.6f, %.6f, %.6f) points=%zu",
             offset.x,
             offset.y,
             offset.z,
             size());
}

void PointCloud::scale(float factor)
{
    scale(factor, makeZeroPoint());
}

void PointCloud::scale(float factor, const Point3f &center)
{
    scale(Point3f{factor, factor, factor}, center);
}

void PointCloud::scale(const Point3f &factors, const Point3f &center)
{
    for (Point3f &point : _positions)
    {
        point.x = center.x + (point.x - center.x) * factors.x;
        point.y = center.y + (point.y - center.y) * factors.y;
        point.z = center.z + (point.z - center.z) * factors.z;
    }

    if (hasNormals())
    {
        for (Point3f &normal : _normals)
        {
            Point3f scaledNormal = normal;
            if (std::abs(factors.x) > 1e-12f)
            {
                scaledNormal.x /= factors.x;
            }
            if (std::abs(factors.y) > 1e-12f)
            {
                scaledNormal.y /= factors.y;
            }
            if (std::abs(factors.z) > 1e-12f)
            {
                scaledNormal.z /= factors.z;
            }
            normal = normalizeVector(scaledNormal);
        }
    }

    LOG_INFO("PointCloud::scale: factors=(%.6f, %.6f, %.6f) center=(%.6f, %.6f, %.6f) points=%zu",
             factors.x,
             factors.y,
             factors.z,
             center.x,
             center.y,
             center.z,
             size());
}

void PointCloud::transform(const PointCloudTransform &transformValue, bool transformNormals)
{
    for (Point3f &point : _positions)
    {
        point = applyPointTransform(transformValue.matrix, point);
    }

    if (transformNormals && hasNormals())
    {
        for (Point3f &normal : _normals)
        {
            normal = normalizeVector(applyLinearTransform(transformValue.matrix, normal));
        }
    }

    LOG_INFO("PointCloud::transform: points=%zu transformNormals=%d frame=%d",
             size(),
             transformNormals ? 1 : 0,
             static_cast<int>(_metadata.coordinateFrame));
    _spatialIndex.reset();
}

PointCloud PointCloud::transformed(const PointCloudTransform &transformValue, bool transformNormals) const
{
    PointCloud copy = *this;
    copy.transform(transformValue, transformNormals);
    return copy;
}

std::vector<std::size_t> PointCloud::selectIndices(const PointPredicate &predicate) const
{
    std::vector<std::size_t> selected;
    selected.reserve(size());

    for (std::size_t index = 0; index < size(); ++index)
    {
        const Point3f *normal = normalAt(index);
        const ColorRGBA *color = colorAt(index);
        const PhotogrammetryPointAttributes *photogrammetry = photogrammetryAttributesAt(index);

        if (!predicate || predicate(index, _positions[index], normal, color, photogrammetry))
        {
            selected.push_back(index);
        }
    }

    LOG_INFO("PointCloud::selectIndices: selected=%zu total=%zu", selected.size(), size());
    return selected;
}

PointCloud PointCloud::filter(const PointPredicate &predicate) const
{
    return copyByIndices(selectIndices(predicate));
}

void PointCloud::filterInPlace(const PointPredicate &predicate)
{
    auto selected = selectIndices(predicate);
    LOG_INFO("PointCloud::filterInPlace: before=%zu after=%zu", size(), selected.size());

    // 原地压缩：避免产生完整副本（百万点规模下可节省约 50% 峰值内存）
    std::size_t writeIdx = 0;
    for (std::size_t readIdx : selected)
    {
        if (writeIdx != readIdx)
        {
            _positions[writeIdx] = std::move(_positions[readIdx]);
            if (!_normals.empty())
                _normals[writeIdx] = std::move(_normals[readIdx]);
            if (!_colors.empty())
                _colors[writeIdx] = std::move(_colors[readIdx]);
            if (!_photogrammetryAttributes.empty())
                _photogrammetryAttributes[writeIdx] = std::move(_photogrammetryAttributes[readIdx]);
            if (!_textureCoordinates.empty())
                _textureCoordinates[writeIdx] = std::move(_textureCoordinates[readIdx]);
        }
        ++writeIdx;
    }

    _positions.resize(writeIdx);
    if (!_normals.empty())                 _normals.resize(writeIdx);
    if (!_colors.empty())                  _colors.resize(writeIdx);
    if (!_photogrammetryAttributes.empty()) _photogrammetryAttributes.resize(writeIdx);
    if (!_textureCoordinates.empty())      _textureCoordinates.resize(writeIdx);

    // 面索引在点移动后失效，一并清空
    _faces.clear();
    _spatialIndex.reset();
}

PointCloud PointCloud::crop(const PointCloudBounds &bounds) const
{
    if (!bounds.valid)
    {
        LOG_WARN("PointCloud::crop: invalid bounds");
        return PointCloud{};
    }

    return filter([&bounds](std::size_t,
                            const Point3f &position,
                            const Point3f *,
                            const ColorRGBA *,
                            const PhotogrammetryPointAttributes *) {
        return isInsideBounds(position, bounds);
    });
}

void PointCloud::cropInPlace(const PointCloudBounds &bounds)
{
    PointCloud cropped = crop(bounds);
    LOG_INFO("PointCloud::cropInPlace: before=%zu after=%zu", size(), cropped.size());
    *this = std::move(cropped);
}

PointCloudCudaExport PointCloud::exportToCuda() const
{
    PointCloudCudaExport exportData;
    exportData.positionsXyz.reserve(size() * 3);

    for (const Point3f &position : _positions)
    {
        exportData.positionsXyz.push_back(position.x);
        exportData.positionsXyz.push_back(position.y);
        exportData.positionsXyz.push_back(position.z);
    }

    if (hasNormals())
    {
        exportData.hasNormals = true;
        exportData.normalsXyz.reserve(_normals.size() * 3);
        for (const Point3f &normal : _normals)
        {
            exportData.normalsXyz.push_back(normal.x);
            exportData.normalsXyz.push_back(normal.y);
            exportData.normalsXyz.push_back(normal.z);
        }
    }

    if (hasColors())
    {
        exportData.hasColors = true;
        exportData.colorsRgba.reserve(_colors.size() * 4);
        for (const ColorRGBA &color : _colors)
        {
            exportData.colorsRgba.push_back(color.r);
            exportData.colorsRgba.push_back(color.g);
            exportData.colorsRgba.push_back(color.b);
            exportData.colorsRgba.push_back(color.a);
        }
    }

    if (hasPhotogrammetryAttributes())
    {
        exportData.hasPhotogrammetryAttributes = true;
        exportData.confidences.reserve(_photogrammetryAttributes.size());
        exportData.reprojectionErrors.reserve(_photogrammetryAttributes.size());
        for (const PhotogrammetryPointAttributes &attributes : _photogrammetryAttributes)
        {
            exportData.confidences.push_back(attributes.confidence);
            exportData.reprojectionErrors.push_back(attributes.reprojectionError);
        }
    }

    LOG_INFO("PointCloud::exportToCuda: points=%zu hasNormals=%d hasColors=%d hasPhotogrammetry=%d",
             size(),
             exportData.hasNormals ? 1 : 0,
             exportData.hasColors ? 1 : 0,
             exportData.hasPhotogrammetryAttributes ? 1 : 0);
    return exportData;
}

const PointCloudMetadata &PointCloud::metadata() const
{
    return _metadata;
}

const Point3f *PointCloud::normalAt(std::size_t index) const
{
    if (!hasNormals() || !isValidIndex(index, size()))
    {
        return nullptr;
    }

    return &_normals[index];
}

const ColorRGBA *PointCloud::colorAt(std::size_t index) const
{
    if (!hasColors() || !isValidIndex(index, size()))
    {
        return nullptr;
    }

    return &_colors[index];
}

const PhotogrammetryPointAttributes *PointCloud::photogrammetryAttributesAt(std::size_t index) const
{
    if (!hasPhotogrammetryAttributes() || !isValidIndex(index, size()))
    {
        return nullptr;
    }

    return &_photogrammetryAttributes[index];
}

const Point2f *PointCloud::textureCoordinateAt(std::size_t index) const
{
    if (!hasTextureCoordinates() || !isValidIndex(index, size()))
    {
        return nullptr;
    }

    return &_textureCoordinates[index];
}

PointCloudPoint PointCloud::pointAt(std::size_t index) const
{
    PointCloudPoint point;
    if (!isValidIndex(index, size()))
    {
        return point;
    }

    point.position = _positions[index];

    if (hasNormals())
    {
        point.setNormal(_normals[index]);
    }

    if (hasColors())
    {
        point.setColor(_colors[index]);
    }

    if (hasPhotogrammetryAttributes())
    {
        point.setPhotogrammetry(_photogrammetryAttributes[index]);
    }

    if (hasTextureCoordinates())
    {
        point.setTextureCoordinate(_textureCoordinates[index]);
    }

    return point;
}

const std::vector<Point3f> &PointCloud::positions() const
{
    return _positions;
}

const std::vector<Point3f> &PointCloud::normals() const
{
    return _normals;
}

const std::vector<ColorRGBA> &PointCloud::colors() const
{
    return _colors;
}

const std::vector<PhotogrammetryPointAttributes> &PointCloud::photogrammetryAttributes() const
{
    return _photogrammetryAttributes;
}

const std::vector<Point2f> &PointCloud::textureCoordinates() const
{
    return _textureCoordinates;
}

const std::vector<PointCloudFace> &PointCloud::faces() const
{
    return _faces;
}

void PointCloud::setMaterialLibraryFile(const std::string &materialLibraryFile)
{
    _materialLibraryFile = materialLibraryFile;
}

const std::string &PointCloud::materialLibraryFile() const
{
    return _materialLibraryFile;
}

void PointCloud::setTextureImageFile(const std::string &textureImageFile)
{
    _textureImageFile = textureImageFile;
}

const std::string &PointCloud::textureImageFile() const
{
    return _textureImageFile;
}

void PointCloud::ensureNormalsForExistingPoints()
{
    if (!hasNormals())
    {
        _normals.assign(_positions.size(), makeZeroPoint());
    }
}

void PointCloud::ensureColorsForExistingPoints()
{
    if (!hasColors())
    {
        _colors.assign(_positions.size(), makeDefaultColor());
    }
}

void PointCloud::ensurePhotogrammetryAttributesForExistingPoints()
{
    if (!hasPhotogrammetryAttributes())
    {
        _photogrammetryAttributes.assign(_positions.size(), makeDefaultPhotogrammetryAttributes());
    }
}

void PointCloud::ensureTextureCoordinatesForExistingPoints()
{
    if (!hasTextureCoordinates())
    {
        _textureCoordinates.assign(_positions.size(), makeZeroTextureCoordinate());
    }
}

void PointCloud::reindexFacesAfterPointRemoval(std::size_t removedIndex)
{
    if (_faces.empty())
    {
        return;
    }

    std::vector<PointCloudFace> updatedFaces;
    updatedFaces.reserve(_faces.size());

    for (const PointCloudFace &face : _faces)
    {
        bool shouldDropFace = false;
        PointCloudFace reindexedFace = face;

        for (std::size_t i = 0; i < 3; ++i)
        {
            if (face.vertexIndices[i] == removedIndex)
            {
                shouldDropFace = true;
                break;
            }

            if (face.vertexIndices[i] > removedIndex)
            {
                reindexedFace.vertexIndices[i] -= 1;
            }

            if (face.hasTextureIndices)
            {
                if (face.textureIndices[i] == removedIndex)
                {
                    shouldDropFace = true;
                    break;
                }

                if (face.textureIndices[i] > removedIndex)
                {
                    reindexedFace.textureIndices[i] -= 1;
                }
            }
        }

        if (!shouldDropFace)
        {
            updatedFaces.push_back(reindexedFace);
        }
    }

    _faces = std::move(updatedFaces);
}

PointCloud PointCloud::copyByIndices(const std::vector<std::size_t> &indices) const
{
    PointCloud result;
    result._metadata = _metadata;
    result._materialLibraryFile = _materialLibraryFile;
    result._textureImageFile = _textureImageFile;
    result.reserve(indices.size());

    std::vector<std::size_t> oldToNew(size(), std::numeric_limits<std::size_t>::max());

    for (std::size_t index : indices)
    {
        if (!isValidIndex(index, size()))
        {
            continue;
        }

        const std::size_t newIndex = result._positions.size();
        oldToNew[index] = newIndex;
        result._positions.push_back(_positions[index]);
        if (hasNormals())
        {
            result._normals.push_back(_normals[index]);
        }

        if (hasColors())
        {
            result._colors.push_back(_colors[index]);
        }

        if (hasPhotogrammetryAttributes())
        {
            result._photogrammetryAttributes.push_back(_photogrammetryAttributes[index]);
        }

        if (hasTextureCoordinates())
        {
            result._textureCoordinates.push_back(_textureCoordinates[index]);
        }
    }

    if (hasFaces())
    {
        for (const PointCloudFace &face : _faces)
        {
            PointCloudFace remappedFace = face;
            bool validFace = true;

            for (std::size_t i = 0; i < 3; ++i)
            {
                const std::size_t oldVertex = face.vertexIndices[i];
                if (!isValidIndex(oldVertex, oldToNew.size()) ||
                    oldToNew[oldVertex] == std::numeric_limits<std::size_t>::max())
                {
                    validFace = false;
                    break;
                }

                remappedFace.vertexIndices[i] = oldToNew[oldVertex];

                if (face.hasTextureIndices)
                {
                    const std::size_t oldTexture = face.textureIndices[i];
                    if (!isValidIndex(oldTexture, oldToNew.size()) ||
                        oldToNew[oldTexture] == std::numeric_limits<std::size_t>::max())
                    {
                        validFace = false;
                        break;
                    }
                    remappedFace.textureIndices[i] = oldToNew[oldTexture];
                }
            }

            if (validFace)
            {
                result._faces.push_back(remappedFace);
            }
        }
    }

    return result;
}

} // namespace xjw::pointcloud