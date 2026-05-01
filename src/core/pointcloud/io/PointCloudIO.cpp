#include "io/PointCloudIO.h"

#include <QByteArray>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace xjw::pointcloud
{

namespace
{

// PLY 顶点属性描述。读取 header 后会把每个属性的类型、名字和偏移记录下来，
// 这样 ASCII 和二进制两条路径都可以复用同一份属性索引信息。
struct PlyProperty
{
    std::string countType;
    std::string type;
    std::string name;
    std::size_t size = 0;
    std::size_t offset = 0;
    bool isList = false;
};

std::string trim(const std::string &value)
{
    // 去掉行首尾空白，便于处理 header 关键字和 OBJ/PLY 文本行。
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0)
    {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0)
    {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::string toLower(std::string value)
{
    // 扩展名匹配不区分大小写，因此统一转小写再判断。
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

PointCloudFileFormat formatFromExtension(const std::string &path)
{
    // 目前模块支持 OBJ、PLY 与简单 XYZ 文本，未知扩展名保持 Auto。
    const std::string extension = toLower(std::filesystem::path(path).extension().string());
    if (extension == ".obj")
    {
        return PointCloudFileFormat::Obj;
    }

    if (extension == ".xyz" || extension == ".txt" || extension == ".pts" || extension == ".asc" ||
        extension == ".csv" || extension == ".pcd")
    {
        return PointCloudFileFormat::Xyz;
    }

    if (extension == ".ply")
    {
        return PointCloudFileFormat::PlyAscii;
    }

    return PointCloudFileFormat::Auto;
}

std::size_t plyScalarTypeSize(const std::string &type)
{
    // 按 PLY 标量类型返回字节宽度，供二进制 PLY 解析计算行跨度使用。
    if (type == "char" || type == "int8")
    {
        return 1;
    }
    if (type == "uchar" || type == "uint8")
    {
        return 1;
    }
    if (type == "short" || type == "int16")
    {
        return 2;
    }
    if (type == "ushort" || type == "uint16")
    {
        return 2;
    }
    if (type == "int" || type == "int32")
    {
        return 4;
    }
    if (type == "uint" || type == "uint32")
    {
        return 4;
    }
    if (type == "float" || type == "float32")
    {
        return 4;
    }
    if (type == "double" || type == "float64")
    {
        return 8;
    }

    return 0;
}

double readBinaryScalar(const std::string &type, const char *ptr)
{
    // 按 header 声明的类型从二进制缓冲区中读取一个标量。
    // 当前实现假定文件是 little-endian，并与写出路径保持一致。
    if (type == "char" || type == "int8")
    {
        int8_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        return static_cast<double>(value);
    }
    if (type == "uchar" || type == "uint8")
    {
        uint8_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        return static_cast<double>(value);
    }
    if (type == "short" || type == "int16")
    {
        int16_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        return static_cast<double>(value);
    }
    if (type == "ushort" || type == "uint16")
    {
        uint16_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        return static_cast<double>(value);
    }
    if (type == "int" || type == "int32")
    {
        int32_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        return static_cast<double>(value);
    }
    if (type == "uint" || type == "uint32")
    {
        uint32_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        return static_cast<double>(value);
    }
    if (type == "float" || type == "float32")
    {
        float value = 0.0f;
        std::memcpy(&value, ptr, sizeof(value));
        return static_cast<double>(value);
    }
    if (type == "double" || type == "float64")
    {
        double value = 0.0;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }

    return 0.0;
}

std::size_t findPropertyIndex(const std::vector<PlyProperty> &properties,
                              const std::vector<std::string> &names)
{
    // 一个语义属性可能在不同文件里有不同命名，例如 red/r/diffuse_red。
    // 因此这里接受别名列表，找到第一个匹配项就返回。
    for (std::size_t index = 0; index < properties.size(); ++index)
    {
        for (const std::string &name : names)
        {
            if (properties[index].name == name)
            {
                return index;
            }
        }
    }

    return std::numeric_limits<std::size_t>::max();
}

bool validPropertyIndex(std::size_t index)
{
    return index != std::numeric_limits<std::size_t>::max();
}

uint8_t decodeObjColorComponent(float value)
{
    // 一些 OBJ 以 [0,1] 存储颜色，也有文件直接写 [0,255]，这里两种都兼容。
    if (value <= 1.0f)
    {
        value *= 255.0f;
    }

    value = std::clamp(value, 0.0f, 255.0f);
    return static_cast<uint8_t>(value + 0.5f);
}

void fillResult(PointCloudIOResult *result,
                PointCloudFileFormat format,
                const std::string &errorMessage)
{
    // 所有对外接口统一通过该辅助函数返回格式识别结果和错误文本。
    if (!result)
    {
        return;
    }

    result->detectedFormat = format;
    result->errorMessage = errorMessage;
}

bool startsWithToken(const std::string &line, const std::string &token)
{
    if (line.size() < token.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < token.size(); ++i)
    {
        if (std::toupper(static_cast<unsigned char>(line[i])) !=
            std::toupper(static_cast<unsigned char>(token[i])))
        {
            return false;
        }
    }

    return line.size() == token.size() || std::isspace(static_cast<unsigned char>(line[token.size()])) != 0;
}

void normalizeDelimiters(std::string *line)
{
    if (!line)
    {
        return;
    }

    for (char &ch : *line)
    {
        if (ch == ',' || ch == ';' || ch == '\t')
        {
            ch = ' ';
        }
    }
}

bool parseObjIndexToken(const std::string &token,
                        std::size_t vertexCount,
                        std::size_t textureCount,
                        std::size_t normalCount,
                        std::size_t *vertexIndex,
                        std::size_t *textureIndex,
                        std::size_t *normalIndex,
                        bool *hasTextureIndex,
                        bool *hasNormalIndex)
{
    if (!vertexIndex || !textureIndex || !normalIndex || !hasTextureIndex || !hasNormalIndex)
    {
        return false;
    }

    *vertexIndex = 0;
    *textureIndex = 0;
    *normalIndex = 0;
    *hasTextureIndex = false;
    *hasNormalIndex = false;

    std::array<std::string, 3> parts;
    std::size_t partIndex = 0;
    std::size_t begin = 0;
    while (partIndex < parts.size())
    {
        const std::size_t slash = token.find('/', begin);
        if (slash == std::string::npos)
        {
            parts[partIndex++] = token.substr(begin);
            break;
        }

        parts[partIndex++] = token.substr(begin, slash - begin);
        begin = slash + 1;
        if (begin > token.size())
        {
            break;
        }

        if (slash == token.size() - 1)
        {
            parts[partIndex++] = std::string();
            break;
        }
    }

    while (partIndex < parts.size())
    {
        parts[partIndex++] = std::string();
    }

    auto parseSingleIndex = [](const std::string &value, std::size_t count, std::size_t *index) -> bool {
        if (!index)
        {
            return false;
        }

        if (value.empty() || count == 0)
        {
            return false;
        }

        int raw = 0;
        try
        {
            raw = std::stoi(value);
        }
        catch (...)
        {
            return false;
        }

        if (raw > 0)
        {
            const std::size_t converted = static_cast<std::size_t>(raw - 1);
            if (converted >= count)
            {
                return false;
            }
            *index = converted;
            return true;
        }

        if (raw < 0)
        {
            const int converted = static_cast<int>(count) + raw;
            if (converted < 0)
            {
                return false;
            }
            *index = static_cast<std::size_t>(converted);
            return true;
        }

        return false;
    };

    if (!parseSingleIndex(parts[0], vertexCount, vertexIndex))
    {
        return false;
    }

    if (!parts[1].empty())
    {
        *hasTextureIndex = parseSingleIndex(parts[1], textureCount, textureIndex);
        if (!*hasTextureIndex)
        {
            return false;
        }
    }

    if (!parts[2].empty())
    {
        *hasNormalIndex = parseSingleIndex(parts[2], normalCount, normalIndex);
        if (!*hasNormalIndex)
        {
            return false;
        }
    }

    return true;
}

} // namespace

bool readPointCloud(const std::string &path,
                    PointCloud *pointCloud,
                    const PointCloudReadOptions &options,
                    PointCloudIOResult *result)
{
    // 统一入口：如果调用方未显式指定格式，则根据扩展名分发到具体加载器。
    if (!pointCloud)
    {
        fillResult(result, PointCloudFileFormat::Auto, "输出点云对象为空");
        return false;
    }

    PointCloudFileFormat format = options.format;
    if (format == PointCloudFileFormat::Auto)
    {
        format = formatFromExtension(path);
    }

    if (format == PointCloudFileFormat::Obj)
    {
        return readObjPointCloud(path, pointCloud, result);
    }

    if (format == PointCloudFileFormat::Xyz)
    {
        return readXyzPointCloud(path, pointCloud, result);
    }

    if (format == PointCloudFileFormat::PlyAscii || format == PointCloudFileFormat::PlyBinaryLittleEndian)
    {
        return readPlyPointCloud(path, pointCloud, result);
    }

    fillResult(result, format, "无法根据扩展名识别点云格式");
    return false;
}

bool writePointCloud(const std::string &path,
                     const PointCloud &pointCloud,
                     const PointCloudWriteOptions &options,
                     PointCloudIOResult *result)
{
    // 与读取路径一致，写出时也先决定目标格式，再调用具体序列化实现。
    PointCloudFileFormat format = options.format;
    if (format == PointCloudFileFormat::Auto)
    {
        format = formatFromExtension(path);
    }

    if (format == PointCloudFileFormat::Obj)
    {
        return writeObjPointCloud(path, pointCloud, options, result);
    }

    if (format == PointCloudFileFormat::Xyz)
    {
        return writeXyzPointCloud(path, pointCloud, options, result);
    }

    if (format == PointCloudFileFormat::PlyAscii || format == PointCloudFileFormat::PlyBinaryLittleEndian)
    {
        return writePlyPointCloud(path, pointCloud, options, result);
    }

    fillResult(result, format, "无法根据扩展名识别点云格式");
    return false;
}

bool readPlyPointCloud(const std::string &path,
                       PointCloud *pointCloud,
                       PointCloudIOResult *result)
{
    if (!pointCloud)
    {
        fillResult(result, PointCloudFileFormat::Auto, "输出点云对象为空");
        return false;
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "无法打开 PLY 文件: " + path);
        return false;
    }

    std::string line;
    if (!std::getline(stream, line) || trim(line) != "ply")
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "无效的 PLY 文件头");
        return false;
    }

    bool isAscii = false;
    bool isBinaryLittleEndian = false;
    std::string currentElementName;
    std::size_t vertexCount = 0;
    std::size_t faceCount = 0;
    std::vector<PlyProperty> vertexProperties;
    std::vector<PlyProperty> faceProperties;
    std::size_t vertexStride = 0;

    // 逐行读取 header，抽取顶点数与属性布局信息。
    while (std::getline(stream, line))
    {
        line = trim(line);
        if (line == "end_header")
        {
            break;
        }

        std::istringstream lineStream(line);
        std::string key;
        lineStream >> key;

        if (key == "format")
        {
            std::string formatName;
            lineStream >> formatName;
            if (formatName == "ascii")
            {
                isAscii = true;
            }
            else if (formatName == "binary_little_endian")
            {
                isBinaryLittleEndian = true;
            }
        }
        else if (key == "element")
        {
            std::string elementName;
            std::size_t elementCount = 0;
            lineStream >> elementName >> elementCount;
            currentElementName = elementName;
            if (elementName == "vertex")
            {
                vertexCount = elementCount;
                vertexProperties.clear();
                vertexStride = 0;
            }
            else if (elementName == "face")
            {
                faceCount = elementCount;
                faceProperties.clear();
            }
        }
        else if (key == "property")
        {
            std::string typeName;
            lineStream >> typeName;

            std::vector<PlyProperty> *targetProperties = nullptr;
            std::size_t *targetStride = nullptr;
            if (currentElementName == "vertex")
            {
                targetProperties = &vertexProperties;
                targetStride = &vertexStride;
            }
            else if (currentElementName == "face")
            {
                targetProperties = &faceProperties;
            }

            if (!targetProperties)
            {
                continue;
            }

            if (typeName == "list")
            {
                std::string countType;
                std::string valueType;
                std::string propertyName;
                lineStream >> countType >> valueType >> propertyName;
                targetProperties->push_back(PlyProperty{countType, valueType, propertyName, 0, 0, true});
            }
            else
            {
                std::string propertyName;
                lineStream >> propertyName;
                const std::size_t size = plyScalarTypeSize(typeName);
                const std::size_t offset = targetStride ? *targetStride : 0;
                targetProperties->push_back(PlyProperty{std::string(), typeName, propertyName, size, offset, false});
                if (targetStride)
                {
                    *targetStride += size;
                }
            }
        }
    }

    if (vertexCount == 0)
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "PLY 文件中没有 vertex 元素");
        return false;
    }

    // 建立关键属性索引。法向量和颜色是可选的，位置必须存在。
    const std::size_t xIndex = findPropertyIndex(vertexProperties, {"x"});
    const std::size_t yIndex = findPropertyIndex(vertexProperties, {"y"});
    const std::size_t zIndex = findPropertyIndex(vertexProperties, {"z"});
    const std::size_t nxIndex = findPropertyIndex(vertexProperties, {"nx"});
    const std::size_t nyIndex = findPropertyIndex(vertexProperties, {"ny"});
    const std::size_t nzIndex = findPropertyIndex(vertexProperties, {"nz"});
    const std::size_t rIndex = findPropertyIndex(vertexProperties, {"red", "r", "diffuse_red"});
    const std::size_t gIndex = findPropertyIndex(vertexProperties, {"green", "g", "diffuse_green"});
    const std::size_t bIndex = findPropertyIndex(vertexProperties, {"blue", "b", "diffuse_blue"});
    const std::size_t aIndex = findPropertyIndex(vertexProperties, {"alpha", "a"});
    const std::size_t faceVertexIndex = findPropertyIndex(faceProperties, {"vertex_indices", "vertex_index"});

    if (!validPropertyIndex(xIndex) || !validPropertyIndex(yIndex) || !validPropertyIndex(zIndex))
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "PLY 文件缺少 x/y/z 属性");
        return false;
    }

    PointCloud cloud;
    cloud.reserve(vertexCount);
    std::vector<Point3f> normals;
    std::vector<ColorRGBA> colors;
    const bool readNormals = validPropertyIndex(nxIndex) && validPropertyIndex(nyIndex) && validPropertyIndex(nzIndex);
    const bool readColors = validPropertyIndex(rIndex) && validPropertyIndex(gIndex) && validPropertyIndex(bIndex);
    if (readNormals)
    {
        normals.reserve(vertexCount);
    }
    if (readColors)
    {
        colors.reserve(vertexCount);
    }

    if (isAscii)
    {
        // ASCII PLY：逐行按 header 顺序解析属性值。
        for (std::size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex)
        {
            if (!std::getline(stream, line))
            {
                break;
            }

            std::istringstream lineStream(line);
            std::vector<double> values(vertexProperties.size(), 0.0);
            for (std::size_t propertyIndex = 0; propertyIndex < vertexProperties.size(); ++propertyIndex)
            {
                if (vertexProperties[propertyIndex].isList)
                {
                    // 点云顶点一般不会使用 list 属性，但 ASCII 路径可以安全跳过。
                    int listCount = 0;
                    lineStream >> listCount;
                    for (int itemIndex = 0; itemIndex < listCount; ++itemIndex)
                    {
                        double ignoredValue = 0.0;
                        lineStream >> ignoredValue;
                    }
                }
                else
                {
                    lineStream >> values[propertyIndex];
                }
            }

            cloud.addPoint(Point3f{
                static_cast<float>(values[xIndex]),
                static_cast<float>(values[yIndex]),
                static_cast<float>(values[zIndex])
            });

            if (readNormals)
            {
                normals.push_back(Point3f{
                    static_cast<float>(values[nxIndex]),
                    static_cast<float>(values[nyIndex]),
                    static_cast<float>(values[nzIndex])
                });
            }

            if (readColors)
            {
                ColorRGBA color;
                color.r = static_cast<uint8_t>(std::clamp(static_cast<int>(values[rIndex]), 0, 255));
                color.g = static_cast<uint8_t>(std::clamp(static_cast<int>(values[gIndex]), 0, 255));
                color.b = static_cast<uint8_t>(std::clamp(static_cast<int>(values[bIndex]), 0, 255));
                if (validPropertyIndex(aIndex))
                {
                    color.a = static_cast<uint8_t>(std::clamp(static_cast<int>(values[aIndex]), 0, 255));
                }
                colors.push_back(color);
            }
        }
    }
    else if (isBinaryLittleEndian)
    {
        // 二进制路径要求顶点属性都为固定宽度标量，这样每个顶点的 stride 才可预计算。
        for (const PlyProperty &property : vertexProperties)
        {
            if (property.isList)
            {
                fillResult(result,
                           PointCloudFileFormat::PlyBinaryLittleEndian,
                           "暂不支持读取带 list 顶点属性的二进制 PLY");
                return false;
            }
        }

        std::vector<char> row(vertexStride);
        for (std::size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex)
        {
            // 每次读取一整行顶点记录，再按 offset 解码各字段。
            stream.read(row.data(), static_cast<std::streamsize>(row.size()));
            if (!stream)
            {
                break;
            }

            cloud.addPoint(Point3f{
                static_cast<float>(readBinaryScalar(vertexProperties[xIndex].type, row.data() + vertexProperties[xIndex].offset)),
                static_cast<float>(readBinaryScalar(vertexProperties[yIndex].type, row.data() + vertexProperties[yIndex].offset)),
                static_cast<float>(readBinaryScalar(vertexProperties[zIndex].type, row.data() + vertexProperties[zIndex].offset))
            });

            if (readNormals)
            {
                normals.push_back(Point3f{
                    static_cast<float>(readBinaryScalar(vertexProperties[nxIndex].type, row.data() + vertexProperties[nxIndex].offset)),
                    static_cast<float>(readBinaryScalar(vertexProperties[nyIndex].type, row.data() + vertexProperties[nyIndex].offset)),
                    static_cast<float>(readBinaryScalar(vertexProperties[nzIndex].type, row.data() + vertexProperties[nzIndex].offset))
                });
            }

            if (readColors)
            {
                ColorRGBA color;
                color.r = static_cast<uint8_t>(std::clamp(
                    static_cast<int>(readBinaryScalar(vertexProperties[rIndex].type, row.data() + vertexProperties[rIndex].offset)),
                    0,
                    255));
                color.g = static_cast<uint8_t>(std::clamp(
                    static_cast<int>(readBinaryScalar(vertexProperties[gIndex].type, row.data() + vertexProperties[gIndex].offset)),
                    0,
                    255));
                color.b = static_cast<uint8_t>(std::clamp(
                    static_cast<int>(readBinaryScalar(vertexProperties[bIndex].type, row.data() + vertexProperties[bIndex].offset)),
                    0,
                    255));
                if (validPropertyIndex(aIndex))
                {
                    color.a = static_cast<uint8_t>(std::clamp(
                        static_cast<int>(readBinaryScalar(vertexProperties[aIndex].type, row.data() + vertexProperties[aIndex].offset)),
                        0,
                        255));
                }
                colors.push_back(color);
            }
        }
    }
    else
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "仅支持 ASCII 或 binary_little_endian PLY");
        return false;
    }

    if (cloud.empty())
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "PLY 文件中没有有效点");
        return false;
    }

    // 如果文件中存在可选属性，则整体挂接到点云对象，避免逐点触发重复补齐逻辑。
    if (!normals.empty())
    {
        cloud.setNormals(normals);
    }
    if (!colors.empty())
    {
        cloud.setColors(colors);
    }

    if (faceCount > 0 && validPropertyIndex(faceVertexIndex))
    {
        if (isAscii)
        {
            for (std::size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
            {
                if (!std::getline(stream, line))
                {
                    break;
                }

                std::istringstream lineStream(line);
                std::vector<std::size_t> polygonIndices;
                for (std::size_t propertyIndex = 0; propertyIndex < faceProperties.size(); ++propertyIndex)
                {
                    const PlyProperty &property = faceProperties[propertyIndex];
                    if (property.isList)
                    {
                        int listCount = 0;
                        lineStream >> listCount;
                        std::vector<std::size_t> values;
                        values.reserve(std::max(listCount, 0));
                        for (int itemIndex = 0; itemIndex < listCount; ++itemIndex)
                        {
                            long long value = 0;
                            lineStream >> value;
                            values.push_back(static_cast<std::size_t>(std::max<long long>(0, value)));
                        }

                        if (propertyIndex == faceVertexIndex)
                        {
                            polygonIndices = std::move(values);
                        }
                    }
                    else
                    {
                        double ignoredValue = 0.0;
                        lineStream >> ignoredValue;
                    }
                }

                if (polygonIndices.size() < 3)
                {
                    continue;
                }

                for (std::size_t triIndex = 1; triIndex + 1 < polygonIndices.size(); ++triIndex)
                {
                    PointCloudFace face;
                    face.vertexIndices = {polygonIndices[0], polygonIndices[triIndex], polygonIndices[triIndex + 1]};
                    cloud.addFace(face);
                }
            }
        }
        else if (isBinaryLittleEndian)
        {
            for (std::size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
            {
                std::vector<std::size_t> polygonIndices;
                bool faceReadOk = true;
                for (std::size_t propertyIndex = 0; propertyIndex < faceProperties.size(); ++propertyIndex)
                {
                    const PlyProperty &property = faceProperties[propertyIndex];
                    if (property.isList)
                    {
                        const std::size_t countTypeSize = plyScalarTypeSize(property.countType);
                        if (countTypeSize == 0)
                        {
                            fillResult(result,
                                       PointCloudFileFormat::PlyBinaryLittleEndian,
                                       "PLY face 列表计数类型不受支持");
                            return false;
                        }

                        std::vector<char> countBuffer(countTypeSize);
                        stream.read(countBuffer.data(), static_cast<std::streamsize>(countBuffer.size()));
                        if (!stream)
                        {
                            faceReadOk = false;
                            break;
                        }

                        const std::size_t listCount = static_cast<std::size_t>(std::max<double>(
                            0.0,
                            readBinaryScalar(property.countType, countBuffer.data())));
                        const std::size_t valueTypeSize = plyScalarTypeSize(property.type);
                        if (valueTypeSize == 0)
                        {
                            fillResult(result,
                                       PointCloudFileFormat::PlyBinaryLittleEndian,
                                       "PLY face 列表值类型不受支持");
                            return false;
                        }

                        std::vector<std::size_t> values;
                        values.reserve(listCount);
                        std::vector<char> valueBuffer(valueTypeSize);
                        for (std::size_t itemIndex = 0; itemIndex < listCount; ++itemIndex)
                        {
                            stream.read(valueBuffer.data(), static_cast<std::streamsize>(valueBuffer.size()));
                            if (!stream)
                            {
                                faceReadOk = false;
                                break;
                            }

                            const auto value = static_cast<long long>(readBinaryScalar(property.type, valueBuffer.data()));
                            values.push_back(static_cast<std::size_t>(std::max<long long>(0, value)));
                        }

                        if (!faceReadOk)
                        {
                            break;
                        }

                        if (propertyIndex == faceVertexIndex)
                        {
                            polygonIndices = std::move(values);
                        }
                    }
                    else
                    {
                        std::vector<char> scalarBuffer(property.size);
                        stream.read(scalarBuffer.data(), static_cast<std::streamsize>(scalarBuffer.size()));
                        if (!stream)
                        {
                            faceReadOk = false;
                            break;
                        }
                    }
                }

                if (!faceReadOk)
                {
                    break;
                }

                if (polygonIndices.size() < 3)
                {
                    continue;
                }

                for (std::size_t triIndex = 1; triIndex + 1 < polygonIndices.size(); ++triIndex)
                {
                    PointCloudFace face;
                    face.vertexIndices = {polygonIndices[0], polygonIndices[triIndex], polygonIndices[triIndex + 1]};
                    cloud.addFace(face);
                }
            }
        }
    }

    *pointCloud = cloud;
    fillResult(result,
               isBinaryLittleEndian ? PointCloudFileFormat::PlyBinaryLittleEndian : PointCloudFileFormat::PlyAscii,
               std::string());
    return true;
}

bool writePlyPointCloud(const std::string &path,
                        const PointCloud &pointCloud,
                        const PointCloudWriteOptions &options,
                        PointCloudIOResult *result)
{
    if (pointCloud.empty())
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "点云为空，无法写出");
        return false;
    }

    const bool writeBinary = (options.format == PointCloudFileFormat::PlyBinaryLittleEndian);
    const bool writeNormals = options.writeNormals && pointCloud.hasNormals();
    const bool writeColors = options.writeColors && pointCloud.hasColors();
    const bool writeFaces = options.writeFaces && pointCloud.hasFaces();

    // 先根据属性可用性输出 header，再写实际顶点数据。
    std::ofstream stream(path, writeBinary ? std::ios::binary : std::ios::out);
    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "无法写出 PLY 文件: " + path);
        return false;
    }

    stream << "ply\n";
    stream << (writeBinary ? "format binary_little_endian 1.0\n" : "format ascii 1.0\n");
    stream << "element vertex " << pointCloud.size() << "\n";
    stream << "property float x\n";
    stream << "property float y\n";
    stream << "property float z\n";
    if (writeNormals)
    {
        stream << "property float nx\n";
        stream << "property float ny\n";
        stream << "property float nz\n";
    }
    if (writeColors)
    {
        stream << "property uchar red\n";
        stream << "property uchar green\n";
        stream << "property uchar blue\n";
        stream << "property uchar alpha\n";
    }
    if (writeFaces)
    {
        stream << "element face " << pointCloud.faces().size() << "\n";
        stream << "property list uchar int vertex_indices\n";
    }
    stream << "end_header\n";

    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::PlyAscii, "写入 PLY 头失败");
        return false;
    }

    if (writeBinary)
    {
        // 小端二进制写出能显著减小文件体积，也更适合大点云。
        for (std::size_t index = 0; index < pointCloud.size(); ++index)
        {
            const Point3f &position = pointCloud.positions()[index];
            stream.write(reinterpret_cast<const char *>(&position.x), sizeof(position.x));
            stream.write(reinterpret_cast<const char *>(&position.y), sizeof(position.y));
            stream.write(reinterpret_cast<const char *>(&position.z), sizeof(position.z));

            if (writeNormals)
            {
                const Point3f &normal = pointCloud.normals()[index];
                stream.write(reinterpret_cast<const char *>(&normal.x), sizeof(normal.x));
                stream.write(reinterpret_cast<const char *>(&normal.y), sizeof(normal.y));
                stream.write(reinterpret_cast<const char *>(&normal.z), sizeof(normal.z));
            }

            if (writeColors)
            {
                const ColorRGBA &color = pointCloud.colors()[index];
                stream.write(reinterpret_cast<const char *>(&color.r), sizeof(color.r));
                stream.write(reinterpret_cast<const char *>(&color.g), sizeof(color.g));
                stream.write(reinterpret_cast<const char *>(&color.b), sizeof(color.b));
                stream.write(reinterpret_cast<const char *>(&color.a), sizeof(color.a));
            }
        }

        if (writeFaces)
        {
            for (const PointCloudFace &face : pointCloud.faces())
            {
                const uint8_t vertexPerFace = 3;
                const int v0 = static_cast<int>(face.vertexIndices[0]);
                const int v1 = static_cast<int>(face.vertexIndices[1]);
                const int v2 = static_cast<int>(face.vertexIndices[2]);
                stream.write(reinterpret_cast<const char *>(&vertexPerFace), sizeof(vertexPerFace));
                stream.write(reinterpret_cast<const char *>(&v0), sizeof(v0));
                stream.write(reinterpret_cast<const char *>(&v1), sizeof(v1));
                stream.write(reinterpret_cast<const char *>(&v2), sizeof(v2));
            }
        }
    }
    else
    {
        // ASCII 路径更适合调试与人工检查。
        stream << std::fixed << std::setprecision(options.precision);
        for (std::size_t index = 0; index < pointCloud.size(); ++index)
        {
            const Point3f &position = pointCloud.positions()[index];
            stream << position.x << ' ' << position.y << ' ' << position.z;

            if (writeNormals)
            {
                const Point3f &normal = pointCloud.normals()[index];
                stream << ' ' << normal.x << ' ' << normal.y << ' ' << normal.z;
            }

            if (writeColors)
            {
                const ColorRGBA &color = pointCloud.colors()[index];
                stream << ' ' << static_cast<int>(color.r)
                       << ' ' << static_cast<int>(color.g)
                       << ' ' << static_cast<int>(color.b)
                       << ' ' << static_cast<int>(color.a);
            }

            stream << '\n';
        }

        if (writeFaces)
        {
            for (const PointCloudFace &face : pointCloud.faces())
            {
                stream << "3 "
                       << face.vertexIndices[0] << ' '
                       << face.vertexIndices[1] << ' '
                       << face.vertexIndices[2] << '\n';
            }
        }
    }

    if (!stream)
    {
        fillResult(result,
                   writeBinary ? PointCloudFileFormat::PlyBinaryLittleEndian : PointCloudFileFormat::PlyAscii,
                   "写出 PLY 数据失败");
        return false;
    }

    fillResult(result,
               writeBinary ? PointCloudFileFormat::PlyBinaryLittleEndian : PointCloudFileFormat::PlyAscii,
               std::string());
    return true;
}

bool readObjPointCloud(const std::string &path,
                       PointCloud *pointCloud,
                       PointCloudIOResult *result)
{
    if (!pointCloud)
    {
        fillResult(result, PointCloudFileFormat::Obj, "输出点云对象为空");
        return false;
    }

    std::ifstream stream(path);
    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::Obj, "无法打开 OBJ 文件: " + path);
        return false;
    }

    std::vector<Point3f> positions;
    std::vector<Point3f> normals;
    std::vector<Point2f> textureCoordinates;
    std::vector<ColorRGBA> colors;
    std::vector<PointCloudFace> faces;
    bool hasAnyColor = false;
    bool hasAnyTextureCoordinate = false;
    std::string materialLibraryFile;

    // 读取 OBJ 常见字段：mtllib / v / vt / vn / f。
    // 对于 f，支持三角形与多边形（扇形拆分）。
    std::string line;
    while (std::getline(stream, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream lineStream(line);
        std::string tag;
        lineStream >> tag;
        if (tag == "v")
        {
            Point3f position;
            if (!(lineStream >> position.x >> position.y >> position.z))
            {
                continue;
            }

            positions.push_back(position);

            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            if (lineStream >> r >> g >> b)
            {
                hasAnyColor = true;
                colors.push_back(ColorRGBA{
                    decodeObjColorComponent(r),
                    decodeObjColorComponent(g),
                    decodeObjColorComponent(b),
                    255
                });
            }
            else
            {
                colors.push_back(ColorRGBA{});
            }
        }
        else if (tag == "vt")
        {
            Point2f textureCoordinate;
            if (lineStream >> textureCoordinate.u >> textureCoordinate.v)
            {
                textureCoordinates.push_back(textureCoordinate);
            }
        }
        else if (tag == "vn")
        {
            Point3f normal;
            if (lineStream >> normal.x >> normal.y >> normal.z)
            {
                normals.push_back(normal);
            }
        }
        else if (tag == "mtllib")
        {
            lineStream >> materialLibraryFile;
        }
        else if (tag == "f")
        {
            std::vector<std::size_t> faceVertexIndices;
            std::vector<std::size_t> faceTextureIndices;
            faceVertexIndices.reserve(8);
            faceTextureIndices.reserve(8);

            std::string token;
            bool allCornersHaveTexture = true;
            while (lineStream >> token)
            {
                std::size_t vertexIndex = 0;
                std::size_t textureIndex = 0;
                std::size_t normalIndex = 0;
                bool hasTextureIndex = false;
                bool hasNormalIndex = false;

                if (!parseObjIndexToken(token,
                                        positions.size(),
                                        textureCoordinates.size(),
                                        normals.size(),
                                        &vertexIndex,
                                        &textureIndex,
                                        &normalIndex,
                                        &hasTextureIndex,
                                        &hasNormalIndex))
                {
                    faceVertexIndices.clear();
                    break;
                }

                faceVertexIndices.push_back(vertexIndex);
                if (hasTextureIndex)
                {
                    faceTextureIndices.push_back(textureIndex);
                }
                else
                {
                    allCornersHaveTexture = false;
                    faceTextureIndices.push_back(0);
                }
            }

            if (faceVertexIndices.size() < 3)
            {
                continue;
            }

            for (std::size_t i = 1; i + 1 < faceVertexIndices.size(); ++i)
            {
                PointCloudFace face;
                face.vertexIndices = {faceVertexIndices[0], faceVertexIndices[i], faceVertexIndices[i + 1]};
                if (allCornersHaveTexture)
                {
                    face.hasTextureIndices = true;
                    face.textureIndices = {faceTextureIndices[0], faceTextureIndices[i], faceTextureIndices[i + 1]};
                }
                faces.push_back(face);
            }
        }
    }

    if (positions.empty())
    {
        fillResult(result, PointCloudFileFormat::Obj, "OBJ 文件中没有顶点数据");
        return false;
    }

    PointCloud cloud;
    cloud.reserve(positions.size());
    for (const Point3f &position : positions)
    {
        cloud.addPoint(position);
    }

    if (normals.size() == positions.size())
    {
        cloud.setNormals(normals);
    }

    if (hasAnyColor)
    {
        cloud.setColors(colors);
    }

    if (!materialLibraryFile.empty())
    {
        cloud.setMaterialLibraryFile(materialLibraryFile);

        // 自动解析 MTL 文件，将首个 map_Kd 纹理文件名写入 textureImageFile。
        // 该路径相对于 OBJ 所在目录，调用方可直接通过 cloud.textureImageFile() 取得。
        const std::filesystem::path mtlPath =
            std::filesystem::path(path).parent_path() / materialLibraryFile;
        std::ifstream mtlStream(mtlPath);
        if (mtlStream.is_open())
        {
            std::string mtlLine;
            while (std::getline(mtlStream, mtlLine))
            {
                mtlLine = trim(mtlLine);
                std::istringstream mtlLineStream(mtlLine);
                std::string mtlTag;
                mtlLineStream >> mtlTag;
                if (mtlTag == "map_Kd")
                {
                    std::string textureName;
                    if (mtlLineStream >> textureName && !textureName.empty())
                    {
                        cloud.setTextureImageFile(textureName);
                    }
                    break;
                }
            }
        }
    }

    if (!textureCoordinates.empty() && !faces.empty())
    {
        std::vector<Point2f> perVertexTextureCoordinates(positions.size(), Point2f{});
        std::vector<bool> vertexHasTextureCoordinate(positions.size(), false);

        for (const PointCloudFace &face : faces)
        {
            if (!face.hasTextureIndices)
            {
                continue;
            }

            for (std::size_t i = 0; i < 3; ++i)
            {
                const std::size_t vertexIndex = face.vertexIndices[i];
                const std::size_t textureIndex = face.textureIndices[i];
                if (vertexIndex >= perVertexTextureCoordinates.size() || textureIndex >= textureCoordinates.size())
                {
                    continue;
                }

                if (!vertexHasTextureCoordinate[vertexIndex])
                {
                    perVertexTextureCoordinates[vertexIndex] = textureCoordinates[textureIndex];
                    vertexHasTextureCoordinate[vertexIndex] = true;
                    hasAnyTextureCoordinate = true;
                }
            }
        }

        if (hasAnyTextureCoordinate)
        {
            cloud.setTextureCoordinates(perVertexTextureCoordinates);

            for (const PointCloudFace &face : faces)
            {
                PointCloudFace perVertexFace;
                perVertexFace.vertexIndices = face.vertexIndices;

                if (face.hasTextureIndices)
                {
                    perVertexFace.hasTextureIndices = true;
                    perVertexFace.textureIndices = face.vertexIndices;
                }

                cloud.addFace(perVertexFace);
            }
        }
        else
        {
            for (const PointCloudFace &face : faces)
            {
                PointCloudFace meshFace;
                meshFace.vertexIndices = face.vertexIndices;
                cloud.addFace(meshFace);
            }
        }
    }
    else
    {
        for (const PointCloudFace &face : faces)
        {
            PointCloudFace meshFace;
            meshFace.vertexIndices = face.vertexIndices;
            cloud.addFace(meshFace);
        }
    }

    PointCloudMetadata metadata = cloud.metadata();
    metadata.sourcePath = path;
    metadata.name = std::filesystem::path(path).stem().string();
    cloud.setMetadata(metadata);

    *pointCloud = cloud;
    fillResult(result, PointCloudFileFormat::Obj, std::string());
    return true;
}

bool writeObjPointCloud(const std::string &path,
                        const PointCloud &pointCloud,
                        const PointCloudWriteOptions &options,
                        PointCloudIOResult *result)
{
    if (pointCloud.empty())
    {
        fillResult(result, PointCloudFileFormat::Obj, "点云为空，无法写出");
        return false;
    }

    std::ofstream stream(path);
    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::Obj, "无法写出 OBJ 文件: " + path);
        return false;
    }

    const bool writeNormals = options.writeNormals && pointCloud.hasNormals();
    const bool writeColors = options.writeColors && pointCloud.hasColors();
    const bool writeTextureCoordinates = options.writeTextureCoordinates && pointCloud.hasTextureCoordinates();
    const bool writeFaces = options.writeFaces && pointCloud.hasFaces();

    std::string materialLibraryFile = options.materialLibraryFile;
    if (materialLibraryFile.empty())
    {
        materialLibraryFile = pointCloud.materialLibraryFile();
    }

    const std::string textureImageFile = !options.textureImageFile.empty()
        ? options.textureImageFile
        : pointCloud.textureImageFile();

    if (materialLibraryFile.empty() && !textureImageFile.empty())
    {
        materialLibraryFile = std::filesystem::path(path).stem().string() + ".mtl";
    }

    if (!materialLibraryFile.empty())
    {
        stream << "mtllib " << materialLibraryFile << '\n';
    }

    stream << std::fixed << std::setprecision(options.precision);
    for (std::size_t index = 0; index < pointCloud.size(); ++index)
    {
        const Point3f &position = pointCloud.positions()[index];
        stream << "v " << position.x << ' ' << position.y << ' ' << position.z;
        if (writeColors)
        {
            const ColorRGBA &color = pointCloud.colors()[index];
            stream << ' ' << (static_cast<float>(color.r) / 255.0f)
                   << ' ' << (static_cast<float>(color.g) / 255.0f)
                   << ' ' << (static_cast<float>(color.b) / 255.0f);
        }
        stream << '\n';
    }

    if (writeTextureCoordinates)
    {
        for (const Point2f &textureCoordinate : pointCloud.textureCoordinates())
        {
            stream << "vt " << textureCoordinate.u << ' ' << textureCoordinate.v << '\n';
        }
    }

    if (writeNormals)
    {
        for (const Point3f &normal : pointCloud.normals())
        {
            stream << "vn " << normal.x << ' ' << normal.y << ' ' << normal.z << '\n';
        }
    }

    if (writeFaces)
    {
        if (!materialLibraryFile.empty())
        {
            stream << "usemtl " << options.materialName << '\n';
        }

        for (const PointCloudFace &face : pointCloud.faces())
        {
            stream << "f";
            for (std::size_t i = 0; i < 3; ++i)
            {
                const std::size_t vertexObjIndex = face.vertexIndices[i] + 1;
                const bool useTextureIndex = writeTextureCoordinates && face.hasTextureIndices;
                const bool useNormalIndex = writeNormals;

                stream << ' ' << vertexObjIndex;
                if (useTextureIndex || useNormalIndex)
                {
                    stream << '/';
                    if (useTextureIndex)
                    {
                        stream << (face.textureIndices[i] + 1);
                    }
                }

                if (useNormalIndex)
                {
                    stream << '/' << (face.vertexIndices[i] + 1);
                }
            }
            stream << '\n';
        }
    }

    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::Obj, "写出 OBJ 数据失败");
        return false;
    }

    if (!materialLibraryFile.empty())
    {
        std::filesystem::path mtlPath(materialLibraryFile);
        if (!mtlPath.is_absolute())
        {
            mtlPath = std::filesystem::path(path).parent_path() / mtlPath;
        }

        std::ofstream mtlStream(mtlPath.string());
        if (!mtlStream)
        {
            fillResult(result, PointCloudFileFormat::Obj, "OBJ 写出成功，但无法写出 MTL 文件: " + mtlPath.string());
            return false;
        }

        mtlStream << "newmtl " << options.materialName << '\n';
        mtlStream << "Ka 1.000000 1.000000 1.000000\n";
        mtlStream << "Kd 1.000000 1.000000 1.000000\n";
        mtlStream << "Ks 0.000000 0.000000 0.000000\n";
        mtlStream << "d 1.000000\n";
        mtlStream << "illum 2\n";
        if (!textureImageFile.empty())
        {
            mtlStream << "map_Kd " << textureImageFile << '\n';
        }

        if (!mtlStream)
        {
            fillResult(result, PointCloudFileFormat::Obj, "写出 MTL 数据失败: " + mtlPath.string());
            return false;
        }
    }

    fillResult(result, PointCloudFileFormat::Obj, std::string());
    return true;
}

bool readXyzPointCloud(const std::string &path,
                       PointCloud *pointCloud,
                       PointCloudIOResult *result)
{
    if (!pointCloud)
    {
        fillResult(result, PointCloudFileFormat::Xyz, "输出点云对象为空");
        return false;
    }

    std::ifstream stream(path);
    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::Xyz, "无法打开 XYZ 文件: " + path);
        return false;
    }

    PointCloud cloud;
    std::vector<ColorRGBA> colors;
    bool hasAnyColor = false;
    bool inPcdDataSection = false;
    bool maybePcd = false;

    std::string line;
    while (std::getline(stream, line))
    {
        line = trim(line);
        if (line.empty())
        {
            continue;
        }

        if (line[0] == '#')
        {
            if (line.find(".PCD") != std::string::npos)
            {
                maybePcd = true;
            }
            continue;
        }

        if (startsWithToken(line, "VERSION") || startsWithToken(line, "FIELDS") || startsWithToken(line, "SIZE") ||
            startsWithToken(line, "TYPE") || startsWithToken(line, "COUNT") || startsWithToken(line, "WIDTH") ||
            startsWithToken(line, "HEIGHT") || startsWithToken(line, "VIEWPOINT") || startsWithToken(line, "POINTS"))
        {
            maybePcd = true;
            continue;
        }

        if (startsWithToken(line, "DATA"))
        {
            maybePcd = true;
            inPcdDataSection = (toLower(line).find("ascii") != std::string::npos);
            continue;
        }

        if (maybePcd && !inPcdDataSection)
        {
            continue;
        }

        normalizeDelimiters(&line);

        std::istringstream lineStream(line);
        Point3f position;
        if (!(lineStream >> position.x >> position.y >> position.z))
        {
            continue;
        }

        cloud.addPoint(position);

        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        double a = 255.0;
        if (lineStream >> r >> g >> b)
        {
            hasAnyColor = true;
            if (!(lineStream >> a))
            {
                a = 255.0;
            }

            colors.push_back(ColorRGBA{
                decodeObjColorComponent(static_cast<float>(r)),
                decodeObjColorComponent(static_cast<float>(g)),
                decodeObjColorComponent(static_cast<float>(b)),
                decodeObjColorComponent(static_cast<float>(a))});
        }
        else
        {
            colors.push_back(ColorRGBA{});
        }
    }

    if (cloud.empty())
    {
        fillResult(result, PointCloudFileFormat::Xyz, "XYZ 文件中没有有效点");
        return false;
    }

    if (hasAnyColor)
    {
        cloud.setColors(colors);
    }

    PointCloudMetadata metadata = cloud.metadata();
    metadata.sourcePath = path;
    metadata.name = std::filesystem::path(path).stem().string();
    cloud.setMetadata(metadata);

    *pointCloud = cloud;
    fillResult(result, PointCloudFileFormat::Xyz, std::string());
    return true;
}

bool writeXyzPointCloud(const std::string &path,
                        const PointCloud &pointCloud,
                        const PointCloudWriteOptions &options,
                        PointCloudIOResult *result)
{
    if (pointCloud.empty())
    {
        fillResult(result, PointCloudFileFormat::Xyz, "点云为空，无法写出");
        return false;
    }

    std::ofstream stream(path);
    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::Xyz, "无法写出 XYZ 文件: " + path);
        return false;
    }

    const bool writeColors = options.writeColors && pointCloud.hasColors();
    stream << std::fixed << std::setprecision(options.precision);
    for (std::size_t index = 0; index < pointCloud.size(); ++index)
    {
        const Point3f &position = pointCloud.positions()[index];
        stream << position.x << ' ' << position.y << ' ' << position.z;
        if (writeColors)
        {
            const ColorRGBA &color = pointCloud.colors()[index];
            stream << ' ' << static_cast<int>(color.r)
                   << ' ' << static_cast<int>(color.g)
                   << ' ' << static_cast<int>(color.b)
                   << ' ' << static_cast<int>(color.a);
        }
        stream << '\n';
    }

    if (!stream)
    {
        fillResult(result, PointCloudFileFormat::Xyz, "写出 XYZ 数据失败");
        return false;
    }

    fillResult(result, PointCloudFileFormat::Xyz, std::string());
    return true;
}

bool readBaRunJsonPointCloud(const std::string &path,
                             PointCloud *pointCloud,
                             PointCloudIOResult *result)
{
    if (!pointCloud)
    {
        fillResult(result, PointCloudFileFormat::Auto, "输出点云对象为空");
        return false;
    }

    QFile file(QString::fromStdString(path));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        fillResult(result, PointCloudFileFormat::Auto, "无法打开 BA JSON 文件: " + path);
        return false;
    }

    const QByteArray content = file.readAll();
    file.close();

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(content, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        fillResult(result, PointCloudFileFormat::Auto, "BA JSON 解析失败: " + parseError.errorString().toStdString());
        return false;
    }

    const QJsonObject root = document.object();
    const QJsonArray points = root.value(QStringLiteral("points")).toArray();
    if (points.isEmpty())
    {
        fillResult(result, PointCloudFileFormat::Auto, "BA JSON 中缺少 points 数组或数组为空");
        return false;
    }

    PointCloud cloud;
    cloud.reserve(static_cast<std::size_t>(points.size()));
    std::vector<PhotogrammetryPointAttributes> photogrammetryAttributes;
    photogrammetryAttributes.reserve(static_cast<std::size_t>(points.size()));

    for (qsizetype index = 0; index < points.size(); ++index)
    {
        const QJsonObject pointObject = points.at(index).toObject();
        const QJsonArray xyz = pointObject.value(QStringLiteral("point_xyz")).toArray();
        if (xyz.size() < 3)
        {
            continue;
        }

        cloud.addPoint(Point3f{
            static_cast<float>(xyz.at(0).toDouble()),
            static_cast<float>(xyz.at(1).toDouble()),
            static_cast<float>(xyz.at(2).toDouble())});

        PhotogrammetryPointAttributes attributes;
        attributes.pointId = pointObject.value(QStringLiteral("index")).toInt(static_cast<int>(index));
        attributes.trackLength = pointObject.value(QStringLiteral("track_len")).toInt();
        attributes.reprojectionError = static_cast<float>(
            pointObject.value(QStringLiteral("rms_after")).toDouble(
                pointObject.value(QStringLiteral("rms_before")).toDouble()));
        attributes.confidence = pointObject.value(QStringLiteral("converged")).toBool(true) ? 1.0f : 0.0f;
        attributes.isControlPoint = false;
        attributes.isValid = pointObject.value(QStringLiteral("valid")).toBool(true);
        photogrammetryAttributes.push_back(attributes);
    }

    if (cloud.empty())
    {
        fillResult(result, PointCloudFileFormat::Auto, "BA JSON 中没有有效点");
        return false;
    }

    cloud.setPhotogrammetryAttributes(photogrammetryAttributes);

    PointCloudMetadata metadata;
    metadata.name = root.value(QStringLiteral("output_dir")).toString().isEmpty()
        ? std::filesystem::path(path).stem().string()
        : root.value(QStringLiteral("output_dir")).toString().toStdString();
    metadata.sourcePath = path;
    metadata.description = "Loaded from ba_run_summary.json";
    metadata.coordinateFrame = PointCloudCoordinateFrame::World;
    metadata.isRegistered = true;
    cloud.setMetadata(metadata);

    *pointCloud = cloud;
    fillResult(result, PointCloudFileFormat::Auto, std::string());
    return true;
}

} // namespace xjw::pointcloud