#include "ObjMtlLoader.h"
#include "io/PathIO.h"

#include <plapoint/io/obj_io.h>
#include <plapoint/core/point_cloud.h>

#include <QDir>
#include <QFileInfo>
#include <QString>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace xjw
{

namespace
{

struct ObjMaterialUsage
{
    std::set<std::string> libraries;
    std::set<std::string> materials;
    bool hasUnmaterialedFaces = false;
};

ObjMaterialUsage parseObjMaterialUsage(const QString &objPath)
{
    ObjMaterialUsage usage;
    std::ifstream file = xjw::common::io::openInputFile(objPath);
    std::string current_material;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream stream(line);
        std::string token;
        stream >> token;
        if (token == "mtllib")
        {
            std::string library;
            while (stream >> library)
            {
                usage.libraries.insert(library);
            }
        }
        else if (token == "usemtl")
        {
            stream >> current_material;
            if (!current_material.empty())
            {
                usage.materials.insert(current_material);
            }
        }
        else if (token == "f" && current_material.empty())
        {
            usage.hasUnmaterialedFaces = true;
        }
    }
    return usage;
}

using MaterialTextureMap = std::map<std::string, std::set<std::string>>;

void trimAsciiWhitespace(std::string *value)
{
    while (!value->empty()
           && std::isspace(static_cast<unsigned char>(value->back())) != 0)
    {
        value->pop_back();
    }
    const auto first = std::find_if(
        value->begin(), value->end(), [](unsigned char character)
        {
            return std::isspace(character) == 0;
        });
    value->erase(value->begin(), first);
}

MaterialTextureMap parseMtlTexturePaths(const QString &mtlPath)
{
    MaterialTextureMap textures;
    std::ifstream file = xjw::common::io::openInputFile(mtlPath);
    if (!file)
    {
        return textures;
    }

    std::string current_material;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream stream(line);
        std::string token;
        stream >> token;
        if (token == "newmtl")
        {
            stream >> current_material;
        }
        else if (token == "map_Kd")
        {
            std::string texture_file;
            std::getline(stream >> std::ws, texture_file);
            trimAsciiWhitespace(&texture_file);
            if (texture_file.size() >= 2 && texture_file.front() == '"'
                && texture_file.back() == '"')
            {
                texture_file = texture_file.substr(1, texture_file.size() - 2);
            }
            if (!texture_file.empty())
            {
                textures[current_material].insert(texture_file);
            }
        }
    }
    return textures;
}

std::set<std::string> texturesForUsage(const MaterialTextureMap &textures,
                                       const ObjMaterialUsage &usage)
{
    std::set<std::string> selected;
    if (usage.materials.empty())
    {
        for (const auto &[material, paths] : textures)
        {
            static_cast<void>(material);
            selected.insert(paths.begin(), paths.end());
        }
        return selected;
    }

    const auto iterator = textures.find(*usage.materials.begin());
    if (iterator != textures.end())
    {
        selected.insert(iterator->second.begin(), iterator->second.end());
    }
    return selected;
}

// Load texture image, converting to BGR 3-channel.
cv::Mat loadTextureBgr(const QString &texturePath)
{
    cv::Mat image = xjw::common::io::readImage(texturePath, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        return {};
    }

    if (image.channels() == 1)
    {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    }

    return image;
}

} // namespace

bool ObjMtlLoader::load(const QString &objPath, TerrainMeshInput *out, QString *errorMsg)
{
    if (!out)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("输出对象为空");
        }
        return false;
    }

    const ObjMaterialUsage material_usage = parseObjMaterialUsage(objPath);
    if (material_usage.libraries.size() > 1 || material_usage.materials.size() > 1
        || (material_usage.hasUnmaterialedFaces && !material_usage.materials.empty()))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral(
                "OBJ 使用了多个材质库、多个 usemtl 材质，或混合了无材质面。"
                "当前地形 DOM 仅支持单材质/单纹理图集，已拒绝生成可能颜色错误的产品：%1")
                            .arg(objPath);
        }
        return false;
    }

    // 1. Read OBJ geometry using plapoint IO
    std::shared_ptr<plapoint::PointCloud<float, plamatrix::Device::CPU>> cloudPtr;
    try
    {
        cloudPtr = plapoint::io::readObj<float>(xjw::common::io::toNativeNarrowPath(objPath));
    }
    catch (const std::exception &e)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("读取 OBJ 失败: %1").arg(QString::fromStdString(e.what()));
        }
        return false;
    }

    if (!cloudPtr || cloudPtr->size() == 0)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("OBJ 文件不含顶点数据: %1").arg(objPath);
        }
        return false;
    }

    // Save MTL path before moving cloud
    const std::string mtlLib = cloudPtr->materialLibraryFile();

    // Move-assign mesh
    out->mesh = std::move(*cloudPtr);

    // 2. Parse MTL for texture reference
    if (!mtlLib.empty())
    {
        const QString objDir = QFileInfo(objPath).absoluteDir().absolutePath();
        const QString mtlPath = QDir(objDir).filePath(xjw::common::io::fromUtf8Path(mtlLib));
        const MaterialTextureMap textures = parseMtlTexturePaths(mtlPath);
        const std::set<std::string> selected_textures =
            texturesForUsage(textures, material_usage);
        if (selected_textures.size() > 1)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral(
                    "OBJ/MTL 引用了多张漫反射纹理；当前地形 DOM 仅支持单纹理图集，"
                    "已拒绝静默套用错误纹理：%1")
                                .arg(objPath);
            }
            return false;
        }
        if (!selected_textures.empty())
        {
            const QString texture_file = xjw::common::io::fromUtf8Path(
                *selected_textures.begin());
            const QString texFullPath = QFileInfo(mtlPath).absoluteDir().filePath(texture_file);
            out->texture = loadTextureBgr(texFullPath);
        }
    }

    return true;
}

} // namespace xjw
