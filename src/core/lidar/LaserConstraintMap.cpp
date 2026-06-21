#include "LaserConstraintMap.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>

namespace xjw
{
namespace lidar
{
namespace
{

struct PlyProperty
{
    std::string type;
    std::string name;
    std::size_t size = 0;
};

struct PlyHeader
{
    bool ascii = false;
    bool binaryLittleEndian = false;
    std::size_t vertexCount = 0;
    std::vector<PlyProperty> vertexProperties;
};

struct PropertyValues
{
    double x = std::numeric_limits<double>::quiet_NaN();
    double y = std::numeric_limits<double>::quiet_NaN();
    double z = std::numeric_limits<double>::quiet_NaN();
    double normalX = std::numeric_limits<double>::quiet_NaN();
    double normalY = std::numeric_limits<double>::quiet_NaN();
    double normalZ = std::numeric_limits<double>::quiet_NaN();
    double curvature = 0.0;
};

struct VoxelKey
{
    long long x = 0;
    long long y = 0;
    long long z = 0;

    bool operator<(const VoxelKey &other) const
    {
        if (x != other.x)
        {
            return x < other.x;
        }
        if (y != other.y)
        {
            return y < other.y;
        }
        return z < other.z;
    }
};

struct VoxelAccumulator
{
    std::array<double, 3> point{{0.0, 0.0, 0.0}};
    std::array<double, 3> normal{{0.0, 0.0, 0.0}};
    double curvature = 0.0;
    int sourceFrameIndex = -1;
    int count = 0;
};

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

std::size_t propertyTypeSize(const std::string &type)
{
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8")
    {
        return 1;
    }
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16")
    {
        return 2;
    }
    if (type == "int" || type == "uint" || type == "float" || type == "int32" || type == "uint32"
        || type == "float32")
    {
        return 4;
    }
    if (type == "double" || type == "float64")
    {
        return 8;
    }
    return 0;
}

bool parseHeader(std::ifstream &input, PlyHeader *header, std::string *errorMessage)
{
    if (!header)
    {
        setError(errorMessage, "PLY header output is null");
        return false;
    }

    std::string line;
    if (!std::getline(input, line) || line != "ply")
    {
        setError(errorMessage, "Not a PLY file");
        return false;
    }

    bool readingVertexProperties = false;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line == "end_header")
        {
            break;
        }

        std::istringstream stream(line);
        std::string keyword;
        stream >> keyword;
        if (keyword == "format")
        {
            std::string format;
            stream >> format;
            header->ascii = format == "ascii";
            header->binaryLittleEndian = format == "binary_little_endian";
        }
        else if (keyword == "element")
        {
            std::string name;
            stream >> name;
            if (name == "vertex")
            {
                stream >> header->vertexCount;
                readingVertexProperties = true;
            }
            else
            {
                readingVertexProperties = false;
            }
        }
        else if (keyword == "property" && readingVertexProperties)
        {
            std::string type;
            std::string name;
            stream >> type >> name;
            const std::size_t size = propertyTypeSize(type);
            if (size == 0)
            {
                setError(errorMessage, "Unsupported PLY vertex property type: " + type);
                return false;
            }
            header->vertexProperties.push_back(PlyProperty{type, name, size});
        }
    }

    if (!header->ascii && !header->binaryLittleEndian)
    {
        setError(errorMessage, "Unsupported PLY format");
        return false;
    }
    if (header->vertexCount == 0)
    {
        setError(errorMessage, "PLY has no vertices");
        return false;
    }
    if (header->vertexProperties.empty())
    {
        setError(errorMessage, "PLY has no vertex properties");
        return false;
    }
    return true;
}

double readLittleEndianValue(const char *bytes, const PlyProperty &property)
{
    if (property.type == "float" || property.type == "float32")
    {
        float value = 0.0f;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    if (property.type == "double" || property.type == "float64")
    {
        double value = 0.0;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }
    if (property.type == "uchar" || property.type == "uint8")
    {
        return static_cast<unsigned char>(bytes[0]);
    }
    if (property.type == "char" || property.type == "int8")
    {
        return static_cast<signed char>(bytes[0]);
    }
    if (property.type == "ushort" || property.type == "uint16")
    {
        std::uint16_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    if (property.type == "short" || property.type == "int16")
    {
        std::int16_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    if (property.type == "uint" || property.type == "uint32")
    {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    std::int32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<double>(value);
}

void assignProperty(PropertyValues *values, const std::string &name, double value)
{
    if (!values)
    {
        return;
    }

    if (name == "x")
    {
        values->x = value;
    }
    else if (name == "y")
    {
        values->y = value;
    }
    else if (name == "z")
    {
        values->z = value;
    }
    else if (name == "normal_x" || name == "nx")
    {
        values->normalX = value;
    }
    else if (name == "normal_y" || name == "ny")
    {
        values->normalY = value;
    }
    else if (name == "normal_z" || name == "nz")
    {
        values->normalZ = value;
    }
    else if (name == "curvature")
    {
        values->curvature = value;
    }
}

bool makeSample(const PropertyValues &values,
                const LaserConstraintMapOptions &options,
                LaserPlaneSample *sample)
{
    if (!sample)
    {
        return false;
    }

    const std::array<double, 3> point{{values.x, values.y, values.z}};
    std::array<double, 3> normal{{values.normalX, values.normalY, values.normalZ}};
    for (double value : point)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    bool normalIsMissing = false;
    for (double value : normal)
    {
        if (!std::isfinite(value))
        {
            normalIsMissing = true;
            break;
        }
    }
    if (normalIsMissing)
    {
        if (!options.useMissingNormalsAsHeightPlanes)
        {
            return false;
        }
        normal = {{0.0, 0.0, 1.0}};
    }
    if (options.maxCurvature >= 0.0 && values.curvature > options.maxCurvature)
    {
        return false;
    }

    const double norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (!std::isfinite(norm) || norm < 1e-9)
    {
        return false;
    }

    sample->point = point;
    sample->normal = {{normal[0] / norm, normal[1] / norm, normal[2] / norm}};
    sample->curvature = values.curvature;
    return true;
}

std::vector<LaserPlaneSample> readAsciiSamples(std::ifstream &input,
                                               const PlyHeader &header,
                                               const LaserConstraintMapOptions &options)
{
    std::vector<LaserPlaneSample> samples;
    samples.reserve(header.vertexCount);

    for (std::size_t vertex = 0; vertex < header.vertexCount; ++vertex)
    {
        PropertyValues values;
        for (const PlyProperty &property : header.vertexProperties)
        {
            double value = 0.0;
            input >> value;
            assignProperty(&values, property.name, value);
        }

        LaserPlaneSample sample;
        if (makeSample(values, options, &sample))
        {
            samples.push_back(sample);
        }
    }
    return samples;
}

std::vector<LaserPlaneSample> readBinaryLittleEndianSamples(std::ifstream &input,
                                                            const PlyHeader &header,
                                                            const LaserConstraintMapOptions &options)
{
    std::vector<LaserPlaneSample> samples;
    samples.reserve(header.vertexCount);

    std::size_t rowSize = 0;
    for (const PlyProperty &property : header.vertexProperties)
    {
        rowSize += property.size;
    }

    std::vector<char> row(rowSize);
    for (std::size_t vertex = 0; vertex < header.vertexCount; ++vertex)
    {
        input.read(row.data(), static_cast<std::streamsize>(row.size()));
        if (!input)
        {
            break;
        }

        PropertyValues values;
        std::size_t offset = 0;
        for (const PlyProperty &property : header.vertexProperties)
        {
            assignProperty(&values, property.name, readLittleEndianValue(row.data() + offset, property));
            offset += property.size;
        }

        LaserPlaneSample sample;
        if (makeSample(values, options, &sample))
        {
            samples.push_back(sample);
        }
    }
    return samples;
}

std::vector<LaserPlaneSample> voxelDownsample(const std::vector<LaserPlaneSample> &samples, double voxelSizeMeters)
{
    if (samples.empty() || !(voxelSizeMeters > 0.0))
    {
        return samples;
    }

    std::map<VoxelKey, VoxelAccumulator> voxels;
    for (const LaserPlaneSample &sample : samples)
    {
        const VoxelKey key{
            static_cast<long long>(std::floor(sample.point[0] / voxelSizeMeters)),
            static_cast<long long>(std::floor(sample.point[1] / voxelSizeMeters)),
            static_cast<long long>(std::floor(sample.point[2] / voxelSizeMeters))};
        VoxelAccumulator &accumulator = voxels[key];
        for (int i = 0; i < 3; ++i)
        {
            accumulator.point[static_cast<std::size_t>(i)] += sample.point[static_cast<std::size_t>(i)];
            accumulator.normal[static_cast<std::size_t>(i)] += sample.normal[static_cast<std::size_t>(i)];
        }
        accumulator.curvature += sample.curvature;
        if (accumulator.sourceFrameIndex < 0)
        {
            accumulator.sourceFrameIndex = sample.sourceFrameIndex;
        }
        ++accumulator.count;
    }

    std::vector<LaserPlaneSample> output;
    output.reserve(voxels.size());
    for (const auto &entry : voxels)
    {
        const VoxelAccumulator &accumulator = entry.second;
        if (accumulator.count <= 0)
        {
            continue;
        }
        LaserPlaneSample sample;
        for (int i = 0; i < 3; ++i)
        {
            sample.point[static_cast<std::size_t>(i)] =
                accumulator.point[static_cast<std::size_t>(i)] / static_cast<double>(accumulator.count);
            sample.normal[static_cast<std::size_t>(i)] =
                accumulator.normal[static_cast<std::size_t>(i)] / static_cast<double>(accumulator.count);
        }
        const double normalNorm = std::sqrt(sample.normal[0] * sample.normal[0]
                                            + sample.normal[1] * sample.normal[1]
                                            + sample.normal[2] * sample.normal[2]);
        if (normalNorm < 1e-9)
        {
            continue;
        }
        sample.normal = {{sample.normal[0] / normalNorm,
                          sample.normal[1] / normalNorm,
                          sample.normal[2] / normalNorm}};
        sample.curvature = accumulator.curvature / static_cast<double>(accumulator.count);
        sample.sourceFrameIndex = accumulator.sourceFrameIndex;
        output.push_back(sample);
    }
    return output;
}

std::vector<LaserPlaneSample> limitSamples(const std::vector<LaserPlaneSample> &samples, int maxSamples)
{
    if (maxSamples <= 0 || samples.size() <= static_cast<std::size_t>(maxSamples))
    {
        return samples;
    }

    std::vector<LaserPlaneSample> output;
    output.reserve(static_cast<std::size_t>(maxSamples));
    const double stride = static_cast<double>(samples.size()) / static_cast<double>(maxSamples);
    for (int i = 0; i < maxSamples; ++i)
    {
        const auto index = static_cast<std::size_t>(std::floor(static_cast<double>(i) * stride));
        output.push_back(samples[std::min(index, samples.size() - 1)]);
    }
    return output;
}

} // namespace

bool LaserConstraintMap::loadPly(const std::string &path,
                                 const LaserConstraintMapOptions &options,
                                 std::string *errorMessage)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        setError(errorMessage, "Cannot open LiDAR PLY: " + path);
        return false;
    }

    PlyHeader header;
    if (!parseHeader(input, &header, errorMessage))
    {
        return false;
    }

    std::vector<LaserPlaneSample> loaded = header.ascii
        ? readAsciiSamples(input, header, options)
        : readBinaryLittleEndianSamples(input, header, options);

    return build(std::move(loaded), options, errorMessage);
}

bool LaserConstraintMap::build(std::vector<LaserPlaneSample> samples,
                               const LaserConstraintMapOptions &options,
                               std::string *errorMessage)
{
    samples = voxelDownsample(samples, options.voxelSizeMeters);
    samples = limitSamples(samples, options.maxSamples);

    if (samples.empty())
    {
        m_samples.clear();
        m_index.clear();
        setError(errorMessage, "LiDAR constraint map has no valid plane samples");
        return false;
    }

    m_samples = std::move(samples);
    rebuildIndex();
    return true;
}

bool LaserConstraintMap::nearestPlane(const std::array<double, 3> &query,
                                      LaserPlaneSample *sample,
                                      double *distanceMeters) const
{
    if (m_samples.empty())
    {
        return false;
    }

    double distance = 0.0;
    const int index = m_index.nearest(query, &distance);
    if (index < 0 || index >= static_cast<int>(m_samples.size()))
    {
        return false;
    }

    if (sample)
    {
        *sample = m_samples[static_cast<std::size_t>(index)];
    }
    if (distanceMeters)
    {
        *distanceMeters = distance;
    }
    return true;
}

void LaserConstraintMap::rebuildIndex()
{
    using LaserKdTree3D = plapoint::search::SpatialKdTree<3, double>;

    std::vector<LaserKdTree3D::Point> points;
    points.reserve(m_samples.size());
    for (std::size_t i = 0; i < m_samples.size(); ++i)
    {
        points.push_back(LaserKdTree3D::Point{
            m_samples[i].point,
            static_cast<int>(i)});
    }
    m_index.build(points);
}

} // namespace lidar
} // namespace xjw
