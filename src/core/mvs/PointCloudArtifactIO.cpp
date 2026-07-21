#include "PointCloudArtifactIO.h"

#include "io/PathIO.h"

#include <plapoint/io/ply_io.h>

#include <QDir>
#include <QFileInfo>

#include <exception>

namespace
{

xjw::mvs::DensePointCloud cloneWithoutNormals(const xjw::mvs::DensePointCloud &cloud)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(cloud.size(), 3);
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        const auto row = static_cast<plamatrix::Index>(index);
        for (int dimension = 0; dimension < 3; ++dimension)
        {
            points(row, dimension) = cloud.points()(row, dimension);
        }
    }
    xjw::mvs::DensePointCloud copy(std::move(points));
    if (cloud.hasColors())
    {
        copy.setColors(*cloud.colors());
    }
    if (cloud.hasIntensities()) copy.setIntensities(*cloud.intensities());
    if (cloud.hasScalarFields()) copy.setScalarFields(cloud.scalarFieldNames(), *cloud.scalarFields());
    if (cloud.hasFaces()) copy.setFaces(*cloud.faces());
    copy.setMaterialLibraryFile(cloud.materialLibraryFile());
    copy.setTextureImageFile(cloud.textureImageFile());
    return copy;
}

} // namespace

namespace xjw::mvs
{

bool writeDensePointCloudPly(const QString &path,
                             const DensePointCloud &pointCloud,
                             bool writeNormals,
                             QString *errorMessage)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建点云输出目录: %1").arg(QFileInfo(path).absolutePath());
        }
        return false;
    }

    try
    {
        if (writeNormals || !pointCloud.hasNormals())
        {
            plapoint::io::writePly(xjw::common::io::toNativeNarrowPath(path),
                                   pointCloud,
                                   plapoint::io::PlyFormat::BinaryLE);
        }
        else
        {
            const DensePointCloud withoutNormals = cloneWithoutNormals(pointCloud);
            plapoint::io::writePly(xjw::common::io::toNativeNarrowPath(path),
                                   withoutNormals,
                                   plapoint::io::PlyFormat::BinaryLE);
        }
        return true;
    }
    catch (const std::exception &exception)
    {
        if (errorMessage)
        {
            *errorMessage = QString::fromUtf8(exception.what());
        }
        return false;
    }
}

} // namespace xjw::mvs
