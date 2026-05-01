#include "io/ObjMtlLoader.h"

#include "io/PointCloudIO.h"

#include <QDir>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace xjw::pointcloud
{

namespace
{

// 加载 PNG/JPG/TIFF 纹理，统一转为 BGR 三通道。
// MTL 解析已由 PointCloudIO::readObjPointCloud 完成，此处只负责图像加载。
cv::Mat loadTextureBgr(const std::string &texturePath)
{
    cv::Mat image = cv::imread(texturePath, cv::IMREAD_UNCHANGED);
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

bool ObjMtlLoader::load(const QString &objPath, TexturedMesh *out, QString *errorMsg)
{
    if (!out)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("输出对象为空");
        }
        return false;
    }

    // 1. 读取 OBJ：PointCloudIO 负责解析几何、UV、三角面以及 MTL 中的 map_Kd，
    //    读取完成后 out->mesh.textureImageFile() 即为纹理文件名（相对于 OBJ 目录）。
    PointCloudIOResult ioResult;
    if (!readPointCloud(objPath.toStdString(), &out->mesh, {}, &ioResult))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("读取 OBJ 失败: %1")
                            .arg(QString::fromStdString(ioResult.errorMessage));
        }
        return false;
    }

    if (out->mesh.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("OBJ 文件不含顶点数据: %1").arg(objPath);
        }
        return false;
    }

    // 2. 加载纹理图像（路径由 IO 层解析 MTL 后填入 cloud.textureImageFile()）
    const std::string texFile = out->mesh.textureImageFile();
    if (!texFile.empty())
    {
        const QString objDir = QFileInfo(objPath).absoluteDir().absolutePath();
        const QString texPath = objDir + QLatin1Char('/') + QString::fromStdString(texFile);
        out->texture = loadTextureBgr(texPath.toStdString());
        // 纹理加载失败不视为错误，调用方检查 out->texture.empty() 即可
    }

    return true;
}

} // namespace xjw::pointcloud
