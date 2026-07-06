#include "ObjMtlLoader.h"
#include "io/PathIO.h"

#include <plapoint/io/obj_io.h>
#include <plapoint/core/point_cloud.h>

#include <QDir>
#include <QFileInfo>
#include <QString>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace xjw
{

namespace
{

// Parse MTL file for the first map_Kd texture filename.
// Returns empty string if not found.
std::string parseMtlTexturePath(const QString &mtlPath)
{
    std::ifstream f = xjw::common::io::openInputFile(mtlPath);
    if (!f)
    {
        return {};
    }

    std::string line;
    while (std::getline(f, line))
    {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "map_Kd")
        {
            std::string texFile;
            iss >> texFile;
            return texFile;
        }
    }
    return {};
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
        const std::string texFile = parseMtlTexturePath(mtlPath);
        if (!texFile.empty())
        {
            const QString texFullPath = QDir(objDir).filePath(xjw::common::io::fromUtf8Path(texFile));
            out->texture = loadTextureBgr(texFullPath);
        }
    }

    return true;
}

} // namespace xjw
