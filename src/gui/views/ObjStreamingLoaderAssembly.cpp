#include "ObjStreamingLoaderInternal.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <plamatrix/dense/dense_matrix.h>

namespace xjw::gui::obj_streaming
{
namespace
{

constexpr std::size_t kProgressIntervalItems = 1ULL * 1024ULL * 1024ULL;

bool isCancelled(const std::atomic_bool *flag)
{
    return flag && flag->load(std::memory_order_relaxed);
}

bool shouldCancel(const std::atomic_bool *flag, std::size_t index)
{
    return (index & 0xFFFFU) == 0 && isCancelled(flag);
}

void reportItemProgress(const ObjLoadProgressCallback &progress,
                        std::size_t completed,
                        std::size_t total,
                        int firstPercent,
                        int lastPercent,
                        const QString &stage,
                        std::size_t *nextReport)
{
    if (!progress || (completed < *nextReport && completed < total))
    {
        return;
    }
    const double ratio = total > 0
        ? std::min(1.0, static_cast<double>(completed) / static_cast<double>(total))
        : 1.0;
    progress(firstPercent + static_cast<int>(std::lround((lastPercent - firstPercent) * ratio)),
             stage);
    *nextReport = completed + kProgressIntervalItems;
}

} // namespace

std::shared_ptr<StreamingObjCloud> assembleObjCloud(
    ObjAssemblyInput input,
    const ObjLoadProgressCallback &progress,
    const std::atomic_bool *cancellationFlag)
{
    if (progress)
    {
        progress(65, QStringLiteral("正在组装 OBJ 顶点数据..."));
    }
    const std::size_t vertexCount = input.vx.size();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(vertexCount), 3);
    std::size_t nextItemReport = 0;
    for (std::size_t index = 0; index < vertexCount; ++index)
    {
        if (shouldCancel(cancellationFlag, index))
        {
            return {};
        }
        const auto row = static_cast<plamatrix::Index>(index);
        points(row, 0) = input.vx[index];
        points(row, 1) = input.vy[index];
        points(row, 2) = input.vz[index];
        reportItemProgress(progress,
                           index + 1,
                           vertexCount,
                           65,
                           68,
                           QStringLiteral("正在组装 OBJ 顶点数据..."),
                           &nextItemReport);
    }
    auto cloud = std::make_shared<StreamingObjCloud>(std::move(points));

    const bool completeColors = vertexCount > 0
        && input.hasVertexColor.size() == vertexCount
        && std::all_of(input.hasVertexColor.begin(),
                       input.hasVertexColor.end(),
                       [](bool value) { return value; });
    if (completeColors)
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
            static_cast<plamatrix::Index>(vertexCount), 3);
        nextItemReport = 0;
        for (std::size_t index = 0; index < vertexCount; ++index)
        {
            if (shouldCancel(cancellationFlag, index))
            {
                return {};
            }
            const auto row = static_cast<plamatrix::Index>(index);
            colors(row, 0) = input.vr[index];
            colors(row, 1) = input.vg[index];
            colors(row, 2) = input.vb[index];
            reportItemProgress(progress,
                               index + 1,
                               vertexCount,
                               68,
                               69,
                               QStringLiteral("正在组装 OBJ 顶点颜色..."),
                               &nextItemReport);
        }
        cloud->setColors(std::move(colors));
    }

    if (input.nx.size() == vertexCount)
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(
            static_cast<plamatrix::Index>(vertexCount), 3);
        nextItemReport = 0;
        for (std::size_t index = 0; index < vertexCount; ++index)
        {
            if (shouldCancel(cancellationFlag, index))
            {
                return {};
            }
            const auto row = static_cast<plamatrix::Index>(index);
            normals(row, 0) = input.nx[index];
            normals(row, 1) = input.ny[index];
            normals(row, 2) = input.nz[index];
            reportItemProgress(progress,
                               index + 1,
                               vertexCount,
                               69,
                               70,
                               QStringLiteral("正在组装 OBJ 顶点法线..."),
                               &nextItemReport);
        }
        cloud->setNormals(std::move(normals));
    }

    if (!input.tx.empty())
    {
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> textureCoordinates(
            static_cast<plamatrix::Index>(input.tx.size()), 2);
        nextItemReport = 0;
        for (std::size_t index = 0; index < input.tx.size(); ++index)
        {
            if (shouldCancel(cancellationFlag, index))
            {
                return {};
            }
            const auto row = static_cast<plamatrix::Index>(index);
            textureCoordinates(row, 0) = input.tx[index];
            textureCoordinates(row, 1) = input.ty[index];
            reportItemProgress(progress,
                               index + 1,
                               input.tx.size(),
                               70,
                               71,
                               QStringLiteral("正在组装 OBJ 纹理坐标..."),
                               &nextItemReport);
        }
        cloud->setTextureCoords(std::move(textureCoordinates));
    }

    if (!input.faceVertices.empty())
    {
        const auto faceCount = static_cast<plamatrix::Index>(input.faceVertices.size());
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(faceCount, 3);
        nextItemReport = 0;
        for (plamatrix::Index face = 0; face < faceCount; ++face)
        {
            const std::size_t index = static_cast<std::size_t>(face);
            if (shouldCancel(cancellationFlag, index))
            {
                return {};
            }
            faces(face, 0) = input.faceVertices[index][0];
            faces(face, 1) = input.faceVertices[index][1];
            faces(face, 2) = input.faceVertices[index][2];
            reportItemProgress(progress,
                               index + 1,
                               static_cast<std::size_t>(faceCount),
                               71,
                               78,
                               QStringLiteral("正在组装 OBJ 网格索引..."),
                               &nextItemReport);
        }
        cloud->setFaces(std::move(faces));

        if (input.faceTexturesComplete
            && input.faceTextures.size() == input.faceVertices.size())
        {
            plamatrix::DenseMatrix<int, plamatrix::Device::CPU> textureIndices(faceCount, 3);
            nextItemReport = 0;
            for (plamatrix::Index face = 0; face < faceCount; ++face)
            {
                const std::size_t index = static_cast<std::size_t>(face);
                if (shouldCancel(cancellationFlag, index))
                {
                    return {};
                }
                textureIndices(face, 0) = input.faceTextures[index][0];
                textureIndices(face, 1) = input.faceTextures[index][1];
                textureIndices(face, 2) = input.faceTextures[index][2];
                reportItemProgress(progress,
                                   index + 1,
                                   static_cast<std::size_t>(faceCount),
                                   78,
                                   80,
                                   QStringLiteral("正在组装 OBJ 纹理索引..."),
                                   &nextItemReport);
            }
            cloud->setFaceTextureIndices(std::move(textureIndices));
        }
    }
    if (!input.materialLibrary.empty())
    {
        cloud->setMaterialLibraryFile(input.materialLibrary);
    }
    if (progress)
    {
        progress(80, QStringLiteral("OBJ 几何读取完成"));
    }
    return cloud;
}

} // namespace xjw::gui::obj_streaming
