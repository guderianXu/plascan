#include "TextureMapper.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/obj_io.h>
#include <plamatrix/dense/dense_matrix.h>

#include <QDir>
#include <QImage>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xjw::mesh
{

namespace
{

using PlaPointCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

struct Bounds
{
    float minX = 0.0f;
    float minY = 0.0f;
    float minZ = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;
    float maxZ = 0.0f;
    bool valid = false;
};

Bounds computeCloudBounds(const PlaPointCloud &cloud)
{
    Bounds bounds;
    if (cloud.size() == 0)
    {
        return bounds;
    }

    bounds.valid = true;
    bounds.minX = bounds.minY = bounds.minZ = std::numeric_limits<float>::max();
    bounds.maxX = bounds.maxY = bounds.maxZ = -std::numeric_limits<float>::max();
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        auto pt = cloud[i];
        bounds.minX = std::min(bounds.minX, pt.x());
        bounds.minY = std::min(bounds.minY, pt.y());
        bounds.minZ = std::min(bounds.minZ, pt.z());
        bounds.maxX = std::max(bounds.maxX, pt.x());
        bounds.maxY = std::max(bounds.maxY, pt.y());
        bounds.maxZ = std::max(bounds.maxZ, pt.z());
    }
    return bounds;
}

float pointAxisValue(const PlaPointCloud &cloud, size_t index, int axis)
{
    auto pt = cloud[index];
    if (axis == 0)
    {
        return pt.x();
    }
    if (axis == 1)
    {
        return pt.y();
    }
    return pt.z();
}

struct ProjectionAxes
{
    int uAxis = 0;
    int vAxis = 1;
};

ProjectionAxes chooseProjectionAxes(const Bounds &bounds)
{
    ProjectionAxes axes;
    if (!bounds.valid)
    {
        return axes;
    }

    const std::array<float, 3> extents = {
        bounds.maxX - bounds.minX,
        bounds.maxY - bounds.minY,
        bounds.maxZ - bounds.minZ};
    std::array<int, 3> indices = {0, 1, 2};
    std::sort(indices.begin(), indices.end(), [&extents](int lhs, int rhs) {
        return extents[lhs] > extents[rhs];
    });
    axes.uAxis = indices[0];
    axes.vAxis = indices[1];
    return axes;
}

bool assignPlanarTextureCoordinates(PlaPointCloud *meshCloud,
                                    std::string *errorMessage)
{
    if (!meshCloud || meshCloud->size() == 0 || !meshCloud->hasFaces())
    {
        if (errorMessage)
        {
            *errorMessage = "网格缺少可用于纹理映射的三角面";
        }
        return false;
    }

    const Bounds bounds = computeCloudBounds(*meshCloud);
    const ProjectionAxes axes = chooseProjectionAxes(bounds);
    const float minU = pointAxisValue(*meshCloud, 0, axes.uAxis);
    const float maxU = pointAxisValue(*meshCloud, 0, axes.uAxis);
    const float minV = pointAxisValue(*meshCloud, 0, axes.vAxis);
    const float maxV = pointAxisValue(*meshCloud, 0, axes.vAxis);

    // Compute actual min/max for projection
    float actualMinU = std::numeric_limits<float>::max();
    float actualMaxU = -std::numeric_limits<float>::max();
    float actualMinV = std::numeric_limits<float>::max();
    float actualMaxV = -std::numeric_limits<float>::max();
    for (size_t i = 0; i < meshCloud->size(); ++i)
    {
        const float u = pointAxisValue(*meshCloud, i, axes.uAxis);
        const float v = pointAxisValue(*meshCloud, i, axes.vAxis);
        actualMinU = std::min(actualMinU, u);
        actualMaxU = std::max(actualMaxU, u);
        actualMinV = std::min(actualMinV, v);
        actualMaxV = std::max(actualMaxV, v);
    }
    Q_UNUSED(minU);
    Q_UNUSED(maxU);
    Q_UNUSED(minV);
    Q_UNUSED(maxV);

    const float rangeU = std::max(actualMaxU - actualMinU, 1e-6f);
    const float rangeV = std::max(actualMaxV - actualMinV, 1e-6f);

    const auto n = static_cast<plamatrix::Index>(meshCloud->size());
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texCoords(n, 2);
    for (size_t i = 0; i < meshCloud->size(); ++i)
    {
        const float u = (pointAxisValue(*meshCloud, i, axes.uAxis) - actualMinU) / rangeU;
        const float v = (pointAxisValue(*meshCloud, i, axes.vAxis) - actualMinV) / rangeV;
        const auto idx = static_cast<plamatrix::Index>(i);
        texCoords(idx, 0) = std::clamp(u, 0.0f, 1.0f);
        texCoords(idx, 1) = std::clamp(v, 0.0f, 1.0f);
    }
    meshCloud->setTextureCoords(std::move(texCoords));

    auto *faces = meshCloud->faces();
    if (!faces)
    {
        if (errorMessage)
        {
            *errorMessage = "网格缺少三角面数据";
        }
        return false;
    }

    const int faceCount = static_cast<int>(faces->rows());
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> texIdx(faceCount, 3);
    for (int fi = 0; fi < faceCount; ++fi)
    {
        texIdx(fi, 0) = (*faces)(fi, 0);
        texIdx(fi, 1) = (*faces)(fi, 1);
        texIdx(fi, 2) = (*faces)(fi, 2);
    }
    meshCloud->setFaceTextureIndices(std::move(texIdx));

    return true;
}

struct ColorRGBA
{
    std::uint8_t r = 200;
    std::uint8_t g = 200;
    std::uint8_t b = 200;
    std::uint8_t a = 255;
};

ColorRGBA averageMeshColor(const PlaPointCloud &meshCloud)
{
    if (!meshCloud.hasColors())
    {
        return ColorRGBA{180, 180, 180, 255};
    }

    auto *colors = meshCloud.colors();
    if (!colors || colors->rows() == 0)
    {
        return ColorRGBA{180, 180, 180, 255};
    }

    std::uint64_t sumR = 0;
    std::uint64_t sumG = 0;
    std::uint64_t sumB = 0;
    for (int i = 0; i < colors->rows(); ++i)
    {
        sumR += (*colors)(i, 0);
        sumG += (*colors)(i, 1);
        sumB += (*colors)(i, 2);
    }

    const std::uint64_t count = static_cast<std::uint64_t>(colors->rows());
    return ColorRGBA{
        static_cast<std::uint8_t>(sumR / count),
        static_cast<std::uint8_t>(sumG / count),
        static_cast<std::uint8_t>(sumB / count),
        255};
}

ColorRGBA vertexColorAt(const PlaPointCloud &meshCloud, std::size_t index)
{
    if (meshCloud.hasColors() && index < meshCloud.size())
    {
        auto pt = meshCloud[index];
        return ColorRGBA{pt.r(), pt.g(), pt.b(), 255};
    }
    return ColorRGBA{180, 180, 180, 255};
}

void expandTexturePadding(QImage *image, std::vector<std::uint8_t> *filledMask, int padding)
{
    if (!image || !filledMask || padding <= 0)
    {
        return;
    }

    const int width = image->width();
    const int height = image->height();
    for (int iteration = 0; iteration < padding; ++iteration)
    {
        bool changed = false;
        const QImage previous = image->copy();
        const std::vector<std::uint8_t> previousMask = *filledMask;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int index = y * width + x;
                if (previousMask[index] != 0)
                {
                    continue;
                }

                int sumR = 0;
                int sumG = 0;
                int sumB = 0;
                int samples = 0;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if (dx == 0 && dy == 0)
                        {
                            continue;
                        }

                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                        {
                            continue;
                        }

                        const int neighborIndex = ny * width + nx;
                        if (previousMask[neighborIndex] == 0)
                        {
                            continue;
                        }

                        const QRgb pixel = previous.pixel(nx, ny);
                        sumR += qRed(pixel);
                        sumG += qGreen(pixel);
                        sumB += qBlue(pixel);
                        ++samples;
                    }
                }

                if (samples == 0)
                {
                    continue;
                }

                image->setPixel(x, y, qRgb(sumR / samples, sumG / samples, sumB / samples));
                (*filledMask)[index] = 1;
                changed = true;
            }
        }

        if (!changed)
        {
            break;
        }
    }
}

bool bakeTextureFromVertexColors(const PlaPointCloud &meshCloud,
                                 const TextureMappingConfig &config,
                                 QImage *textureImage,
                                 std::string *errorMessage)
{
    if (!textureImage)
    {
        if (errorMessage)
        {
            *errorMessage = "内部错误：纹理图输出对象无效";
        }
        return false;
    }

    if (!meshCloud.hasFaces() || !meshCloud.hasTextureCoords())
    {
        if (errorMessage)
        {
            *errorMessage = "网格缺少三角面或纹理坐标";
        }
        return false;
    }

    auto *faces = meshCloud.faces();
    auto *texCoords = meshCloud.textureCoords();
    auto *faceTexIndices = meshCloud.faceTextureIndices();
    if (!faces || !texCoords)
    {
        if (errorMessage)
        {
            *errorMessage = "网格面或纹理坐标数据无效";
        }
        return false;
    }

    const int textureSize = std::clamp(config.textureSize, 512, 16384);
    const int padding = std::clamp(config.padding, 0, 32);
    const ColorRGBA background = config.keepUnmapped
        ? averageMeshColor(meshCloud)
        : ColorRGBA{0, 0, 0, 255};
    *textureImage = QImage(textureSize, textureSize, QImage::Format_RGB32);
    textureImage->fill(qRgb(background.r, background.g, background.b));

    std::vector<std::uint8_t> filledMask(static_cast<std::size_t>(textureSize) * textureSize, 0);

    const auto edgeFunction = [](const QPointF &a, const QPointF &b, const QPointF &c) {
        return (c.x() - a.x()) * (b.y() - a.y()) - (c.y() - a.y()) * (b.x() - a.x());
    };

    const int faceCount = faces->rows();
    for (int fi = 0; fi < faceCount; ++fi)
    {
        if (config.progressFn && (static_cast<std::size_t>(fi) % 256 == 0 || fi + 1 == faceCount))
        {
            const int percent = 25 + static_cast<int>((60.0 * (fi + 1)) / std::max(faceCount, 1));
            config.progressFn("正在烘焙纹理...", percent);
        }

        const int i0 = (*faces)(fi, 0);
        const int i1 = (*faces)(fi, 1);
        const int i2 = (*faces)(fi, 2);
        if (i0 >= static_cast<int>(meshCloud.size())
            || i1 >= static_cast<int>(meshCloud.size())
            || i2 >= static_cast<int>(meshCloud.size()))
        {
            continue;
        }

        const std::size_t t0Idx = faceTexIndices ? static_cast<std::size_t>((*faceTexIndices)(fi, 0)) : static_cast<std::size_t>(i0);
        const std::size_t t1Idx = faceTexIndices ? static_cast<std::size_t>((*faceTexIndices)(fi, 1)) : static_cast<std::size_t>(i1);
        const std::size_t t2Idx = faceTexIndices ? static_cast<std::size_t>((*faceTexIndices)(fi, 2)) : static_cast<std::size_t>(i2);
        if (t0Idx >= static_cast<std::size_t>(texCoords->rows())
            || t1Idx >= static_cast<std::size_t>(texCoords->rows())
            || t2Idx >= static_cast<std::size_t>(texCoords->rows()))
        {
            continue;
        }

        const QPointF p0((*texCoords)(static_cast<int>(t0Idx), 0) * (textureSize - 1),
                         (1.0f - (*texCoords)(static_cast<int>(t0Idx), 1)) * (textureSize - 1));
        const QPointF p1((*texCoords)(static_cast<int>(t1Idx), 0) * (textureSize - 1),
                         (1.0f - (*texCoords)(static_cast<int>(t1Idx), 1)) * (textureSize - 1));
        const QPointF p2((*texCoords)(static_cast<int>(t2Idx), 0) * (textureSize - 1),
                         (1.0f - (*texCoords)(static_cast<int>(t2Idx), 1)) * (textureSize - 1));
        const double area = edgeFunction(p0, p1, p2);
        if (std::abs(area) < 1e-8)
        {
            continue;
        }

        const int minX = std::max(0, static_cast<int>(std::floor(std::min({p0.x(), p1.x(), p2.x()}))));
        const int maxX = std::min(textureSize - 1, static_cast<int>(std::ceil(std::max({p0.x(), p1.x(), p2.x()}))));
        const int minY = std::max(0, static_cast<int>(std::floor(std::min({p0.y(), p1.y(), p2.y()}))));
        const int maxY = std::min(textureSize - 1, static_cast<int>(std::ceil(std::max({p0.y(), p1.y(), p2.y()}))));

        const ColorRGBA c0 = vertexColorAt(meshCloud, static_cast<std::size_t>(i0));
        const ColorRGBA c1 = vertexColorAt(meshCloud, static_cast<std::size_t>(i1));
        const ColorRGBA c2 = vertexColorAt(meshCloud, static_cast<std::size_t>(i2));

        for (int y = minY; y <= maxY; ++y)
        {
            for (int x = minX; x <= maxX; ++x)
            {
                const QPointF pixelCenter(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5);
                const double w0 = edgeFunction(p1, p2, pixelCenter) / area;
                const double w1 = edgeFunction(p2, p0, pixelCenter) / area;
                const double w2 = edgeFunction(p0, p1, pixelCenter) / area;
                if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6)
                {
                    continue;
                }

                int red = static_cast<int>(std::lround(w0 * c0.r + w1 * c1.r + w2 * c2.r));
                int green = static_cast<int>(std::lround(w0 * c0.g + w1 * c1.g + w2 * c2.g));
                int blue = static_cast<int>(std::lround(w0 * c0.b + w1 * c1.b + w2 * c2.b));
                const int maskIndex = y * textureSize + x;

                if (filledMask[maskIndex] != 0 && config.blendMethod.find("加权") != std::string::npos)
                {
                    const QRgb existing = textureImage->pixel(x, y);
                    red = (red + qRed(existing)) / 2;
                    green = (green + qGreen(existing)) / 2;
                    blue = (blue + qBlue(existing)) / 2;
                }

                textureImage->setPixel(x,
                                       y,
                                       qRgb(std::clamp(red, 0, 255),
                                            std::clamp(green, 0, 255),
                                            std::clamp(blue, 0, 255)));
                filledMask[maskIndex] = 1;
            }
        }
    }

    if (config.progressFn)
    {
        config.progressFn("正在扩展纹理边界...", 88);
    }
    expandTexturePadding(textureImage, &filledMask, padding);
    return true;
}

} // namespace

bool TextureMapper::generateTexturedModelFromMeshFile(const std::string &meshPath,
                                                      const std::string &productsDir,
                                                      const TextureMappingConfig &config,
                                                      TextureMappingResult *result,
                                                      std::string *errorMsg)
{
    if (result)
    {
        *result = TextureMappingResult();
    }

    auto meshCloudPtr = plapoint::io::readObj<float>(meshPath);
    if (!meshCloudPtr)
    {
        if (errorMsg)
        {
            *errorMsg = "无法读取网格模型: " + meshPath;
        }
        return false;
    }

    if (!assignPlanarTextureCoordinates(meshCloudPtr.get(), errorMsg))
    {
        return false;
    }

    if (config.progressFn)
    {
        config.progressFn("正在准备纹理图集...", 20);
    }

    QImage textureImage;
    if (!bakeTextureFromVertexColors(*meshCloudPtr, config, &textureImage, errorMsg))
    {
        return false;
    }

    const QString outputDir = QString::fromStdString(productsDir);
    const QString texturesDir = QDir(outputDir).filePath(QStringLiteral("textures"));
    QDir().mkpath(texturesDir);

    const QString texturePngPath = QDir(texturesDir).filePath(QStringLiteral("model_texture.png"));
    if (!textureImage.save(texturePngPath))
    {
        if (errorMsg)
        {
            *errorMsg = "无法写出纹理图像: " + texturePngPath.toStdString();
        }
        return false;
    }

    const QString objPath = QDir(outputDir).filePath(QStringLiteral("textured_model.obj"));
    const QString mtlPath = QDir(outputDir).filePath(QStringLiteral("textured_model.mtl"));
    meshCloudPtr->setMaterialLibraryFile(QStringLiteral("textured_model.mtl").toStdString());
    meshCloudPtr->setTextureImageFile(QStringLiteral("textures/model_texture.png").toStdString());

    plapoint::io::writeObj<float>(objPath.toStdString(), *meshCloudPtr);

    if (result)
    {
        result->modelObjPath = objPath.toStdString();
        result->modelMtlPath = mtlPath.toStdString();
        result->texturePngPath = texturePngPath.toStdString();
        result->textureSize = textureImage.width();
        result->textureAlgorithm = "vertex_color_planar_bake";
        result->uvMethod = config.uvMethod;
        result->blendMethod = config.blendMethod;
    }

    return true;
}

} // namespace xjw::mesh
