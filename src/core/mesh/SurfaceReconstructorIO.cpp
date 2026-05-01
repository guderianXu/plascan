#include "SurfaceReconstructorIO.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace xjw
{
namespace mesh
{
namespace detail
{

namespace
{

bool endsWith(const std::string &text, const std::string &suffix)
{
    if (text.size() < suffix.size())
    {
        return false;
    }

    return std::equal(suffix.rbegin(),
                      suffix.rend(),
                      text.rbegin(),
                      [](char left, char right) {
                          return std::tolower(static_cast<unsigned char>(left))
                                 == std::tolower(static_cast<unsigned char>(right));
                      });
}

int typeSize(const std::string &type)
{
    if (type == "char" || type == "int8") return 1;
    if (type == "uchar" || type == "uint8") return 1;
    if (type == "short" || type == "int16") return 2;
    if (type == "ushort" || type == "uint16") return 2;
    if (type == "int" || type == "int32") return 4;
    if (type == "uint" || type == "uint32") return 4;
    if (type == "float" || type == "float32") return 4;
    if (type == "double" || type == "float64") return 8;
    return 0;
}

double readScalar(const std::string &type, const std::uint8_t *ptr)
{
    if (type == "char" || type == "int8") return *reinterpret_cast<const std::int8_t *>(ptr);
    if (type == "uchar" || type == "uint8") return *reinterpret_cast<const std::uint8_t *>(ptr);
    if (type == "short" || type == "int16") return *reinterpret_cast<const std::int16_t *>(ptr);
    if (type == "ushort" || type == "uint16") return *reinterpret_cast<const std::uint16_t *>(ptr);
    if (type == "int" || type == "int32") return *reinterpret_cast<const std::int32_t *>(ptr);
    if (type == "uint" || type == "uint32") return *reinterpret_cast<const std::uint32_t *>(ptr);
    if (type == "float" || type == "float32") return *reinterpret_cast<const float *>(ptr);
    if (type == "double" || type == "float64") return *reinterpret_cast<const double *>(ptr);
    return 0.0;
}

bool loadXYZText(const std::string &path,
                 std::vector<PointXYZRGB> &points,
                 std::string *errorMessage)
{
    std::ifstream stream(path);
    if (!stream)
    {
        if (errorMessage)
        {
            *errorMessage = "无法打开点云文件: " + path;
        }
        return false;
    }

    points.clear();
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream row(line);
        PointXYZRGB point;
        int red = 200;
        int green = 200;
        int blue = 200;
        if (!(row >> point.x >> point.y >> point.z))
        {
            continue;
        }

        if (row >> red >> green >> blue)
        {
            point.r = static_cast<std::uint8_t>(std::clamp(red, 0, 255));
            point.g = static_cast<std::uint8_t>(std::clamp(green, 0, 255));
            point.b = static_cast<std::uint8_t>(std::clamp(blue, 0, 255));
        }

        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            continue;
        }

        points.push_back(point);
    }

    if (points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "点云为空: " + path;
        }
        return false;
    }

    return true;
}

bool loadPLY(const std::string &path,
             std::vector<PointXYZRGB> &points,
             std::string *errorMessage)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        if (errorMessage)
        {
            *errorMessage = "无法打开 PLY 文件: " + path;
        }
        return false;
    }

    std::string line;
    if (!std::getline(stream, line) || line.find("ply") != 0)
    {
        if (errorMessage)
        {
            *errorMessage = "无效 PLY 头";
        }
        return false;
    }

    bool isBinaryLE = false;
    bool isAscii = false;
    int vertexCount = 0;
    bool inVertex = false;

    struct Property
    {
        std::string type;
        std::string name;
        int size = 0;
        bool isList = false;
    };
    std::vector<Property> properties;

    while (std::getline(stream, line))
    {
        if (line == "end_header")
        {
            break;
        }

        std::istringstream header(line);
        std::string key;
        header >> key;
        if (key == "format")
        {
            std::string format;
            header >> format;
            isBinaryLE = (format == "binary_little_endian");
            isAscii = (format == "ascii");
        }
        else if (key == "element")
        {
            std::string element;
            int count = 0;
            header >> element >> count;
            inVertex = (element == "vertex");
            if (inVertex)
            {
                vertexCount = count;
            }
        }
        else if (key == "property" && inVertex)
        {
            std::string type;
            header >> type;
            if (type == "list")
            {
                std::string countType;
                std::string valueType;
                std::string name;
                header >> countType >> valueType >> name;
                properties.push_back({valueType, name, typeSize(valueType), true});
            }
            else
            {
                std::string name;
                header >> name;
                properties.push_back({type, name, typeSize(type), false});
            }
        }
    }

    if (vertexCount <= 0 || properties.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "PLY 顶点信息缺失";
        }
        return false;
    }

    int idxX = -1;
    int idxY = -1;
    int idxZ = -1;
    int idxR = -1;
    int idxG = -1;
    int idxB = -1;
    for (int index = 0; index < static_cast<int>(properties.size()); ++index)
    {
        if (properties[static_cast<std::size_t>(index)].name == "x") idxX = index;
        else if (properties[static_cast<std::size_t>(index)].name == "y") idxY = index;
        else if (properties[static_cast<std::size_t>(index)].name == "z") idxZ = index;
        else if (properties[static_cast<std::size_t>(index)].name == "red"
                 || properties[static_cast<std::size_t>(index)].name == "r") idxR = index;
        else if (properties[static_cast<std::size_t>(index)].name == "green"
                 || properties[static_cast<std::size_t>(index)].name == "g") idxG = index;
        else if (properties[static_cast<std::size_t>(index)].name == "blue"
                 || properties[static_cast<std::size_t>(index)].name == "b") idxB = index;
    }

    if (idxX < 0 || idxY < 0 || idxZ < 0)
    {
        if (errorMessage)
        {
            *errorMessage = "PLY 缺少 x/y/z 属性";
        }
        return false;
    }

    points.clear();
    points.reserve(static_cast<std::size_t>(vertexCount));

    if (isAscii)
    {
        for (int rowIndex = 0; rowIndex < vertexCount; ++rowIndex)
        {
            if (!std::getline(stream, line))
            {
                break;
            }

            std::istringstream row(line);
            std::vector<double> values(properties.size(), 0.0);
            for (std::size_t propertyIndex = 0; propertyIndex < properties.size(); ++propertyIndex)
            {
                if (!(row >> values[propertyIndex]))
                {
                    values[propertyIndex] = 0.0;
                }
            }

            PointXYZRGB point;
            point.x = static_cast<float>(values[static_cast<std::size_t>(idxX)]);
            point.y = static_cast<float>(values[static_cast<std::size_t>(idxY)]);
            point.z = static_cast<float>(values[static_cast<std::size_t>(idxZ)]);
            if (idxR >= 0) point.r = static_cast<std::uint8_t>(std::clamp(static_cast<int>(values[static_cast<std::size_t>(idxR)]), 0, 255));
            if (idxG >= 0) point.g = static_cast<std::uint8_t>(std::clamp(static_cast<int>(values[static_cast<std::size_t>(idxG)]), 0, 255));
            if (idxB >= 0) point.b = static_cast<std::uint8_t>(std::clamp(static_cast<int>(values[static_cast<std::size_t>(idxB)]), 0, 255));
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
            {
                points.push_back(point);
            }
        }
    }
    else if (isBinaryLE)
    {
        int stride = 0;
        std::vector<int> offsets(properties.size(), 0);
        for (std::size_t index = 0; index < properties.size(); ++index)
        {
            offsets[index] = stride;
            if (properties[index].isList || properties[index].size <= 0)
            {
                if (errorMessage)
                {
                    *errorMessage = "当前暂不支持 list 顶点属性";
                }
                return false;
            }
            stride += properties[index].size;
        }

        std::vector<std::uint8_t> row(static_cast<std::size_t>(stride), 0);
        for (int rowIndex = 0; rowIndex < vertexCount; ++rowIndex)
        {
            stream.read(reinterpret_cast<char *>(row.data()), stride);
            if (!stream)
            {
                break;
            }

            PointXYZRGB point;
            point.x = static_cast<float>(readScalar(properties[static_cast<std::size_t>(idxX)].type,
                                                    row.data() + offsets[static_cast<std::size_t>(idxX)]));
            point.y = static_cast<float>(readScalar(properties[static_cast<std::size_t>(idxY)].type,
                                                    row.data() + offsets[static_cast<std::size_t>(idxY)]));
            point.z = static_cast<float>(readScalar(properties[static_cast<std::size_t>(idxZ)].type,
                                                    row.data() + offsets[static_cast<std::size_t>(idxZ)]));
            if (idxR >= 0) point.r = static_cast<std::uint8_t>(std::clamp(static_cast<int>(readScalar(properties[static_cast<std::size_t>(idxR)].type, row.data() + offsets[static_cast<std::size_t>(idxR)])), 0, 255));
            if (idxG >= 0) point.g = static_cast<std::uint8_t>(std::clamp(static_cast<int>(readScalar(properties[static_cast<std::size_t>(idxG)].type, row.data() + offsets[static_cast<std::size_t>(idxG)])), 0, 255));
            if (idxB >= 0) point.b = static_cast<std::uint8_t>(std::clamp(static_cast<int>(readScalar(properties[static_cast<std::size_t>(idxB)].type, row.data() + offsets[static_cast<std::size_t>(idxB)])), 0, 255));
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
            {
                points.push_back(point);
            }
        }
    }
    else
    {
        if (errorMessage)
        {
            *errorMessage = "仅支持 ascii 或 binary_little_endian PLY";
        }
        return false;
    }

    if (points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "PLY 中无有效点";
        }
        return false;
    }

    return true;
}

} // namespace

bool loadPointCloud(const std::string &path,
                    std::vector<PointXYZRGB> &points,
                    std::string *errorMessage)
{
    if (endsWith(path, ".ply"))
    {
        return loadPLY(path, points, errorMessage);
    }

    return loadXYZText(path, points, errorMessage);
}

} // namespace detail
} // namespace mesh
} // namespace xjw
