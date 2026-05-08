#include "DemDomIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QVector3D>

#include <gdal_priv.h>
#include <cpl_conv.h>

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace xjw
{

namespace
{

void ensureGdalRegistered()
{
    static bool registered = false;
    if (!registered)
    {
        GDALAllRegister();
        registered = true;
    }
}

double demMaxY(const DemGridData &demGrid)
{
    return demGrid.minY + demGrid.stepY * static_cast<double>(std::max(0, demGrid.height - 1));
}

cv::Mat flipForRasterWrite(const cv::Mat &input)
{
    cv::Mat output;
    cv::flip(input, output, 0);
    return output;
}

} // namespace

bool DemDomIO::writeDemPreviewPng(const DemGridData &demGrid,
                                  const QString &outputPath,
                                  QString *errorMsg)
{
    if (!demGrid.isValid())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 栅格无效，无法写出预览图");
        }
        return false;
    }

    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(demGrid.elevation, &minValue, &maxValue, nullptr, nullptr, demGrid.validMask);
    if (maxValue - minValue < 1e-9)
    {
        maxValue = minValue + 1.0;
    }

    cv::Mat preview(demGrid.height, demGrid.width, CV_8U, cv::Scalar(0));
    for (int row = 0; row < demGrid.height; ++row)
    {
        for (int col = 0; col < demGrid.width; ++col)
        {
            if (demGrid.validMask.at<uchar>(row, col) == 0)
            {
                continue;
            }

            const double normalized =
                (demGrid.elevation.at<float>(row, col) - minValue) / (maxValue - minValue);
            preview.at<uchar>(row, col) = static_cast<uchar>(qBound(0, static_cast<int>(std::round(normalized * 255.0)), 255));
        }
    }

    QDir().mkpath(QFileInfo(outputPath).absolutePath());
    cv::Mat previewFlipped = flipForRasterWrite(preview);
    if (!cv::imwrite(outputPath.toStdString(), previewFlipped))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 预览图写出失败: %1").arg(outputPath);
        }
        return false;
    }

    return true;
}

bool DemDomIO::writeDemRaster(const DemGridData &demGrid,
                              const QString &outputPath,
                              DemRasterFormat format,
                              QString *errorMsg)
{
    if (!demGrid.isValid())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 栅格无效，无法写出");
        }
        return false;
    }

    ensureGdalRegistered();
    QDir().mkpath(QFileInfo(outputPath).absolutePath());

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL GTiff driver 不可用");
        }
        return false;
    }

    char **createOptions = nullptr;
    createOptions = CSLSetNameValue(createOptions, "COMPRESS", "LZW");
    createOptions = CSLSetNameValue(createOptions, "TILED", "YES");

    const GDALDataType dataType = format == DemRasterFormat::UInt16Tiff ? GDT_UInt16 : GDT_Float32;
    const int nBands = (format == DemRasterFormat::Float32Tiff && demGrid.hasWorldXY()) ? 4 : 1;
    GDALDataset *dataset = driver->Create(outputPath.toStdString().c_str(),
                                          demGrid.width,
                                          demGrid.height,
                                          nBands,
                                          dataType,
                                          createOptions);
    CSLDestroy(createOptions);
    if (!dataset)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL 创建 DEM 文件失败: %1").arg(outputPath);
        }
        return false;
    }

    double geoTransform[6]{
        demGrid.minX - demGrid.stepX * 0.5,
        demGrid.stepX,
        0.0,
        demMaxY(demGrid) + demGrid.stepY * 0.5,
        0.0,
        -demGrid.stepY};
    dataset->SetGeoTransform(geoTransform);
    if (!demGrid.projection.projectionWkt.isEmpty())
    {
        dataset->SetProjection(demGrid.projection.projectionWkt.toStdString().c_str());
    }

    GDALRasterBand *band = dataset->GetRasterBand(1);
    if (!band)
    {
        GDALClose(dataset);
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL DEM 波段创建失败: %1").arg(outputPath);
        }
        return false;
    }

    if (format == DemRasterFormat::UInt16Tiff)
    {
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::minMaxLoc(demGrid.elevation, &minValue, &maxValue, nullptr, nullptr, demGrid.validMask);
        if (maxValue - minValue < 1e-9)
        {
            maxValue = minValue + 1.0;
        }

        cv::Mat u16(demGrid.height, demGrid.width, CV_16U, cv::Scalar(0));
        for (int row = 0; row < demGrid.height; ++row)
        {
            for (int col = 0; col < demGrid.width; ++col)
            {
                if (demGrid.validMask.at<uchar>(row, col) == 0)
                {
                    u16.at<uint16_t>(row, col) = 0;
                    continue;
                }
                const double normalized =
                    (demGrid.elevation.at<float>(row, col) - minValue) / (maxValue - minValue);
                u16.at<uint16_t>(row, col) = static_cast<uint16_t>(qBound(0, static_cast<int>(std::round(normalized * 65535.0)), 65535));
            }
        }
        cv::Mat writeRaster = flipForRasterWrite(u16);
        band->SetNoDataValue(0.0);
        const CPLErr error = band->RasterIO(GF_Write,
                                            0,
                                            0,
                                            demGrid.width,
                                            demGrid.height,
                                            writeRaster.data,
                                            demGrid.width,
                                            demGrid.height,
                                            GDT_UInt16,
                                            0,
                                            0);
        GDALClose(dataset);
        if (error != CE_None)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("GDAL 写出 UInt16 DEM 失败: %1").arg(outputPath);
            }
            return false;
        }

        return true;
    }

    if (nBands == 4)
    {
        // 4-band output: X, Y, Z, validity mask (like ASP run-PC.tif)
        const cv::Mat *bands[3] = { &demGrid.worldX, &demGrid.worldY, &demGrid.elevation };
        const float nodata = -3.40282e+38f;
        for (int b = 0; b < 3; ++b)
        {
            cv::Mat raster(demGrid.height, demGrid.width, CV_32F, cv::Scalar(nodata));
            for (int row = 0; row < demGrid.height; ++row)
                for (int col = 0; col < demGrid.width; ++col)
                    if (demGrid.validMask.at<uchar>(row, col) != 0)
                        raster.at<float>(row, col) = bands[b]->at<float>(row, col);
            cv::Mat writeRaster = flipForRasterWrite(raster);
            GDALRasterBand *gBand = dataset->GetRasterBand(b + 1);
            gBand->SetNoDataValue(static_cast<double>(nodata));
            gBand->RasterIO(GF_Write, 0, 0, demGrid.width, demGrid.height,
                            writeRaster.data, demGrid.width, demGrid.height,
                            GDT_Float32, 0, 0);
        }
        // Band 4: triangulation error (like ASP's intersection_error band)
        // Falls back to validity mask if no triangulation error available
        cv::Mat band4(demGrid.height, demGrid.width, CV_32F, cv::Scalar(0.0f));
        if (demGrid.hasTriangulationError())
        {
            const float nodata4 = -3.40282e+38f;
            for (int row = 0; row < demGrid.height; ++row)
                for (int col = 0; col < demGrid.width; ++col)
                    if (demGrid.validMask.at<uchar>(row, col) != 0)
                        band4.at<float>(row, col) = demGrid.triangulationError.at<float>(row, col);
                    else
                        band4.at<float>(row, col) = nodata4;
            cv::Mat writeBand4 = flipForRasterWrite(band4);
            GDALRasterBand *gBand4 = dataset->GetRasterBand(4);
            gBand4->SetNoDataValue(static_cast<double>(nodata4));
            gBand4->RasterIO(GF_Write, 0, 0, demGrid.width, demGrid.height,
                             writeBand4.data, demGrid.width, demGrid.height,
                             GDT_Float32, 0, 0);
        }
        else
        {
            for (int row = 0; row < demGrid.height; ++row)
                for (int col = 0; col < demGrid.width; ++col)
                    if (demGrid.validMask.at<uchar>(row, col) != 0)
                        band4.at<float>(row, col) = 1.0f;
            cv::Mat writeBand4 = flipForRasterWrite(band4);
            dataset->GetRasterBand(4)->RasterIO(GF_Write, 0, 0, demGrid.width, demGrid.height,
                                                writeBand4.data, demGrid.width, demGrid.height,
                                                GDT_Float32, 0, 0);
        }
        GDALClose(dataset);
        return true;
    }

    cv::Mat floatRaster(demGrid.height, demGrid.width, CV_32F, cv::Scalar(-9999.0f));
    for (int row = 0; row < demGrid.height; ++row)
    {
        for (int col = 0; col < demGrid.width; ++col)
        {
            if (demGrid.validMask.at<uchar>(row, col) != 0)
            {
                floatRaster.at<float>(row, col) = demGrid.elevation.at<float>(row, col);
            }
        }
    }
    cv::Mat writeRaster = flipForRasterWrite(floatRaster);
    band->SetNoDataValue(-9999.0);
    const CPLErr error = band->RasterIO(GF_Write,
                                        0,
                                        0,
                                        demGrid.width,
                                        demGrid.height,
                                        writeRaster.data,
                                        demGrid.width,
                                        demGrid.height,
                                        GDT_Float32,
                                        0,
                                        0);
    GDALClose(dataset);
    if (error != CE_None)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL 写出 Float32 DEM 失败: %1").arg(outputPath);
        }
        return false;
    }

    return true;
}

bool DemDomIO::readDemRaster(const QString &inputPath,
                             DemGridData *demGrid,
                             QString *errorMsg)
{
    if (!demGrid)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 读取目标为空");
        }
        return false;
    }

    ensureGdalRegistered();
    GDALDataset *dataset = static_cast<GDALDataset *>(GDALOpen(inputPath.toStdString().c_str(), GA_ReadOnly));
    if (!dataset)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 读取失败: %1").arg(inputPath);
        }
        return false;
    }

    const int width = dataset->GetRasterXSize();
    const int height = dataset->GetRasterYSize();
    if (width <= 0 || height <= 0 || dataset->GetRasterCount() <= 0)
    {
        GDALClose(dataset);
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 尺寸无效: %1").arg(inputPath);
        }
        return false;
    }

    cv::Mat raster(height, width, CV_32F, cv::Scalar(0));
    GDALRasterBand *band = dataset->GetRasterBand(1);
    if (!band || band->RasterIO(GF_Read,
                                0,
                                0,
                                width,
                                height,
                                raster.data,
                                width,
                                height,
                                GDT_Float32,
                                0,
                                0) != CE_None)
    {
        GDALClose(dataset);
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL 读取 DEM 波段失败: %1").arg(inputPath);
        }
        return false;
    }

    cv::flip(raster, raster, 0);

    int hasNoData = 0;
    const double noDataValue = band->GetNoDataValue(&hasNoData);

    demGrid->width = width;
    demGrid->height = height;
    demGrid->elevation = raster;
    demGrid->validMask = cv::Mat(height, width, CV_8U, cv::Scalar(255));

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            const float value = raster.at<float>(row, col);
            if ((hasNoData && std::fabs(static_cast<double>(value) - noDataValue) < 1e-6) || !std::isfinite(value))
            {
                demGrid->validMask.at<uchar>(row, col) = 0;
            }
        }
    }

    double geoTransform[6]{};
    if (dataset->GetGeoTransform(geoTransform) == CE_None)
    {
        demGrid->stepX = geoTransform[1];
        demGrid->stepY = std::fabs(geoTransform[5]);
        demGrid->minX = geoTransform[0] + demGrid->stepX * 0.5;
        demGrid->minY = geoTransform[3] + geoTransform[5] * (height - 0.5);
    }
    else
    {
        demGrid->minX = 0.0;
        demGrid->minY = 0.0;
        demGrid->stepX = 1.0;
        demGrid->stepY = 1.0;
    }

    const char *projectionRef = dataset->GetProjectionRef();
    if (projectionRef && projectionRef[0] != '\0')
    {
        demGrid->projection.projectionWkt = QString::fromStdString(projectionRef);
    }

    GDALClose(dataset);
    return true;
}

bool DemDomIO::writeDenseCloudXyz(const PlaPointCloud &denseCloud,
                                  const QString &outputPath,
                                  QString *errorMsg)
{
    if (denseCloud.size() == 0)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("密集点云为空，无法写出 XYZ");
        }
        return false;
    }

    QDir().mkpath(QFileInfo(outputPath).absolutePath());
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("密集点云 XYZ 写出失败: %1").arg(outputPath);
        }
        return false;
    }

    QTextStream stream(&file);
    for (size_t i = 0; i < denseCloud.size(); ++i)
    {
        auto pt = denseCloud[i];
        stream << pt.x() << ' ' << pt.y() << ' ' << pt.z() << '\n';
    }
    return true;
}

bool DemDomIO::writeMeshPlyFromDemGrid(const DemGridData &demGrid,
                                       const QString &outputPath,
                                       int *vertexCount,
                                       int *faceCount,
                                       QString *errorMsg)
{
    if (vertexCount)
    {
        *vertexCount = 0;
    }
    if (faceCount)
    {
        *faceCount = 0;
    }

    if (!demGrid.isValid() || demGrid.width <= 1 || demGrid.height <= 1)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 栅格无效，无法写出网格");
        }
        return false;
    }

    std::vector<QVector3D> vertices;
    std::vector<int> gridToVertex(static_cast<std::size_t>(demGrid.width * demGrid.height), -1);
    auto gridIndex = [&demGrid](int col, int row) {
        return static_cast<std::size_t>(row * demGrid.width + col);
    };

    for (int row = 0; row < demGrid.height; ++row)
    {
        for (int col = 0; col < demGrid.width; ++col)
        {
            if (demGrid.validMask.at<uchar>(row, col) == 0)
            {
                continue;
            }

            gridToVertex[gridIndex(col, row)] = static_cast<int>(vertices.size());
            vertices.push_back(QVector3D(static_cast<float>(demGrid.minX + demGrid.stepX * static_cast<double>(col)),
                                         static_cast<float>(demGrid.minY + demGrid.stepY * static_cast<double>(row)),
                                         demGrid.elevation.at<float>(row, col)));
        }
    }

    std::vector<std::array<int, 3>> faces;
    for (int row = 0; row < demGrid.height - 1; ++row)
    {
        for (int col = 0; col < demGrid.width - 1; ++col)
        {
            const int v00 = gridToVertex[gridIndex(col, row)];
            const int v10 = gridToVertex[gridIndex(col + 1, row)];
            const int v01 = gridToVertex[gridIndex(col, row + 1)];
            const int v11 = gridToVertex[gridIndex(col + 1, row + 1)];
            if (v00 >= 0 && v10 >= 0 && v11 >= 0)
            {
                faces.push_back({v00, v10, v11});
            }
            if (v00 >= 0 && v11 >= 0 && v01 >= 0)
            {
                faces.push_back({v00, v11, v01});
            }
        }
    }

    if (vertices.empty() || faces.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 网格顶点或面为空");
        }
        return false;
    }

    QDir().mkpath(QFileInfo(outputPath).absolutePath());
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("PLY 网格写出失败: %1").arg(outputPath);
        }
        return false;
    }

    QTextStream stream(&file);
    stream << "ply\n";
    stream << "format ascii 1.0\n";
    stream << "element vertex " << vertices.size() << "\n";
    stream << "property float x\n";
    stream << "property float y\n";
    stream << "property float z\n";
    stream << "element face " << faces.size() << "\n";
    stream << "property list uchar int vertex_indices\n";
    stream << "end_header\n";
    for (const QVector3D &vertex : vertices)
    {
        stream << vertex.x() << ' ' << vertex.y() << ' ' << vertex.z() << '\n';
    }
    for (const std::array<int, 3> &face : faces)
    {
        stream << "3 " << face[0] << ' ' << face[1] << ' ' << face[2] << '\n';
    }

    if (vertexCount)
    {
        *vertexCount = static_cast<int>(vertices.size());
    }
    if (faceCount)
    {
        *faceCount = static_cast<int>(faces.size());
    }
    return true;
}

bool DemDomIO::writeDenseCloudPly(const DemGridData &demGrid,
                                   const QString &outputPath,
                                   int *pointCount,
                                   QString *errorMsg)
{
    if (pointCount) *pointCount = 0;

    if (!demGrid.isValid() || !demGrid.hasWorldXY())
    {
        if (errorMsg) *errorMsg = QStringLiteral("DEM 栅格无效或缺少 XY 坐标");
        return false;
    }

    int count = 0;
    for (int row = 0; row < demGrid.height; ++row)
        for (int col = 0; col < demGrid.width; ++col)
            if (demGrid.validMask.at<uchar>(row, col) != 0)
                ++count;

    if (count == 0)
    {
        if (errorMsg) *errorMsg = QStringLiteral("无有效点");
        return false;
    }

    QDir().mkpath(QFileInfo(outputPath).absolutePath());
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (errorMsg) *errorMsg = QStringLiteral("无法写出 PLY: %1").arg(outputPath);
        return false;
    }

    QTextStream stream(&file);
    stream << "ply\n";
    stream << "format ascii 1.0\n";
    stream << "element vertex " << count << "\n";
    stream << "property float x\n";
    stream << "property float y\n";
    stream << "property float z\n";
    stream << "end_header\n";

    for (int row = 0; row < demGrid.height; ++row)
        for (int col = 0; col < demGrid.width; ++col)
            if (demGrid.validMask.at<uchar>(row, col) != 0)
                stream << demGrid.worldX.at<float>(row, col) << ' '
                       << demGrid.worldY.at<float>(row, col) << ' '
                       << demGrid.elevation.at<float>(row, col) << '\n';

    if (pointCount) *pointCount = count;
    return true;
}

bool DemDomIO::writeDomImage(const cv::Mat &domImage,
                             const QString &outputPath,
                             DomImageFormat format,
                             QString *errorMsg)
{
    if (domImage.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DOM 图像为空，无法写出");
        }
        return false;
    }

    QDir().mkpath(QFileInfo(outputPath).absolutePath());
    if (format == DomImageFormat::Tiff)
    {
        ensureGdalRegistered();
        GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!driver)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("GDAL GTiff driver 不可用，无法写出 DOM");
            }
            return false;
        }

        char **createOptions = nullptr;
        createOptions = CSLSetNameValue(createOptions, "COMPRESS", "LZW");
        createOptions = CSLSetNameValue(createOptions, "TILED", "YES");
        GDALDataset *dataset = driver->Create(outputPath.toStdString().c_str(),
                                              domImage.cols,
                                              domImage.rows,
                                              domImage.channels(),
                                              GDT_Byte,
                                              createOptions);
        CSLDestroy(createOptions);
        if (!dataset)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("GDAL 创建 DOM TIFF 失败: %1").arg(outputPath);
            }
            return false;
        }

        std::vector<cv::Mat> channels;
        cv::split(domImage, channels);
        for (int index = 0; index < static_cast<int>(channels.size()); ++index)
        {
            cv::Mat flipped = flipForRasterWrite(channels[static_cast<std::size_t>(index)]);
            GDALRasterBand *band = dataset->GetRasterBand(index + 1);
            if (!band || band->RasterIO(GF_Write,
                                        0,
                                        0,
                                        domImage.cols,
                                        domImage.rows,
                                        flipped.data,
                                        domImage.cols,
                                        domImage.rows,
                                        GDT_Byte,
                                        0,
                                        0) != CE_None)
            {
                GDALClose(dataset);
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("GDAL 写出 DOM TIFF 波段失败: %1").arg(outputPath);
                }
                return false;
            }
        }

        GDALClose(dataset);
        return true;
    }

    if (!cv::imwrite(outputPath.toStdString(), domImage))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DOM 图像写出失败: %1").arg(outputPath);
        }
        return false;
    }

    return true;
}

// =============================================================================
// writeDomGeoTiff — 带地理坐标参考的 DOM GeoTIFF
// =============================================================================

bool DemDomIO::writeDomGeoTiff(const cv::Mat &domImage,
                               const DemGridData &demGrid,
                               const QString &outputPath,
                               QString *errorMsg)
{
    if (domImage.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DOM 图像为空，无法写出 GeoTIFF");
        }
        return false;
    }

    if (!demGrid.isValid())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 栅格元数据无效，无法设置 DOM GeoTIFF 地理参考");
        }
        return false;
    }

    ensureGdalRegistered();
    QDir().mkpath(QFileInfo(outputPath).absolutePath());

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL GTiff driver 不可用");
        }
        return false;
    }

    char **createOptions = nullptr;
    createOptions = CSLSetNameValue(createOptions, "COMPRESS", "LZW");
    createOptions = CSLSetNameValue(createOptions, "TILED", "YES");

    GDALDataset *dataset = driver->Create(outputPath.toStdString().c_str(),
                                          domImage.cols,
                                          domImage.rows,
                                          domImage.channels(),
                                          GDT_Byte,
                                          createOptions);
    CSLDestroy(createOptions);
    if (!dataset)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL 创建 DOM GeoTIFF 失败: %1").arg(outputPath);
        }
        return false;
    }

    // 以 domImage 尺寸对 demGrid 的范围和步长做线性重采样（若尺寸不同）
    const double scaleX = static_cast<double>(domImage.cols) / static_cast<double>(demGrid.width);
    const double scaleY = static_cast<double>(domImage.rows) / static_cast<double>(demGrid.height);
    const double domStepX = demGrid.stepX / scaleX;
    const double domStepY = demGrid.stepY / scaleY;

    const double maxY = demGrid.minY + domStepY * static_cast<double>(std::max(0, domImage.rows - 1));

    double geoTransform[6]{
        demGrid.minX - domStepX * 0.5,
        domStepX,
        0.0,
        maxY + domStepY * 0.5,
        0.0,
        -domStepY};
    dataset->SetGeoTransform(geoTransform);

    if (!demGrid.projection.projectionWkt.isEmpty())
    {
        dataset->SetProjection(demGrid.projection.projectionWkt.toStdString().c_str());
    }

    std::vector<cv::Mat> channels;
    cv::split(domImage, channels);
    for (int index = 0; index < static_cast<int>(channels.size()); ++index)
    {
        cv::Mat flipped = flipForRasterWrite(channels[static_cast<std::size_t>(index)]);
        GDALRasterBand *band = dataset->GetRasterBand(index + 1);
        if (!band || band->RasterIO(GF_Write,
                                    0,
                                    0,
                                    domImage.cols,
                                    domImage.rows,
                                    flipped.data,
                                    domImage.cols,
                                    domImage.rows,
                                    GDT_Byte,
                                    0,
                                    0) != CE_None)
        {
            GDALClose(dataset);
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("GDAL 写出 DOM GeoTIFF 波段失败: %1").arg(outputPath);
            }
            return false;
        }
    }

    GDALClose(dataset);
    return true;
}

} // namespace xjw