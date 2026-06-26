#include "DemDomIO.h"

#include <plapoint/io/ply_io.h>
#include <plapoint/io/xyz_io.h>
#include <plapoint/mesh/height_grid.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <gdal_priv.h>
#include <cpl_conv.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
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

bool isRasterNoData(float value, bool hasNoData, double noDataValue)
{
    if (!std::isfinite(value))
    {
        return true;
    }

    if (!hasNoData)
    {
        return false;
    }

    return value == static_cast<float>(noDataValue) ||
           std::fabs(static_cast<double>(value) - noDataValue) < 1e-6;
}

plapoint::mesh::HeightGrid<float> heightGridFromDemGrid(const DemGridData &demGrid)
{
    plapoint::mesh::HeightGrid<float> grid;
    grid.width = demGrid.width;
    grid.height = demGrid.height;
    grid.minX = static_cast<float>(demGrid.minX);
    grid.minY = static_cast<float>(demGrid.minY);
    grid.stepX = static_cast<float>(demGrid.stepX);
    grid.stepY = static_cast<float>(demGrid.stepY);

    const std::size_t cellCount =
        static_cast<std::size_t>(std::max(0, grid.width)) *
        static_cast<std::size_t>(std::max(0, grid.height));
    grid.heights.assign(cellCount, 0.0f);
    grid.weights.assign(cellCount, 0.0f);
    grid.valid.assign(cellCount, 0);
    grid.fillPass.assign(cellCount, 0);
    if (demGrid.hasColor())
    {
        grid.colors.assign(cellCount * 3, 0);
    }

    for (int row = 0; row < demGrid.height; ++row)
    {
        for (int col = 0; col < demGrid.width; ++col)
        {
            const auto cell = static_cast<std::size_t>(row * demGrid.width + col);
            if (demGrid.validMask.at<uchar>(row, col) == 0)
            {
                continue;
            }
            grid.heights[cell] = demGrid.elevation.at<float>(row, col);
            grid.weights[cell] = 1.0f;
            grid.valid[cell] = 1;
            if (demGrid.hasColor())
            {
                const cv::Vec3b bgr = demGrid.color.at<cv::Vec3b>(row, col);
                grid.colors[cell * 3] = bgr[2];
                grid.colors[cell * 3 + 1] = bgr[1];
                grid.colors[cell * 3 + 2] = bgr[0];
            }
        }
    }
    return grid;
}

PlaPointCloud pointCloudFromDemGrid(const DemGridData &demGrid)
{
    std::vector<std::pair<int, int>> validCells;
    validCells.reserve(static_cast<std::size_t>(demGrid.width * demGrid.height));
    for (int row = 0; row < demGrid.height; ++row)
    {
        for (int col = 0; col < demGrid.width; ++col)
        {
            if (demGrid.validMask.at<uchar>(row, col) != 0)
            {
                validCells.emplace_back(row, col);
            }
        }
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(validCells.size(), 3);
    std::unique_ptr<plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>> colors;
    if (demGrid.hasColor())
    {
        colors = std::make_unique<plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>>(
            validCells.size(), 3);
    }
    std::unique_ptr<plamatrix::DenseMatrix<float, plamatrix::Device::CPU>> errors;
    if (demGrid.hasTriangulationError())
    {
        errors = std::make_unique<plamatrix::DenseMatrix<float, plamatrix::Device::CPU>>(
            validCells.size(), 1);
    }

    for (std::size_t i = 0; i < validCells.size(); ++i)
    {
        const auto [row, col] = validCells[i];
        const auto matrixRow = static_cast<plamatrix::Index>(i);
        const float x = demGrid.hasWorldXY()
            ? demGrid.worldX.at<float>(row, col)
            : static_cast<float>(demGrid.minX + demGrid.stepX * static_cast<double>(col));
        const float y = demGrid.hasWorldXY()
            ? demGrid.worldY.at<float>(row, col)
            : static_cast<float>(demGrid.minY + demGrid.stepY * static_cast<double>(row));
        points(matrixRow, 0) = x;
        points(matrixRow, 1) = y;
        points(matrixRow, 2) = demGrid.elevation.at<float>(row, col);
        if (colors)
        {
            const cv::Vec3b bgr = demGrid.color.at<cv::Vec3b>(row, col);
            (*colors)(matrixRow, 0) = bgr[2];
            (*colors)(matrixRow, 1) = bgr[1];
            (*colors)(matrixRow, 2) = bgr[0];
        }
        if (errors)
        {
            (*errors)(matrixRow, 0) = demGrid.triangulationError.at<float>(row, col);
        }
    }

    PlaPointCloud cloud(std::move(points));
    if (colors)
    {
        cloud.setColors(std::move(*colors));
    }
    if (errors)
    {
        cloud.setScalarFields(std::vector<std::string>{"error"}, std::move(*errors));
    }
    return cloud;
}

bool writeSingleBandQualityRaster(const DemGridData &demGrid,
                                  const cv::Mat &source,
                                  const QString &outputPath,
                                  GDALDataType dataType,
                                  double noDataValue,
                                  QString *errorMsg)
{
    if (source.empty())
    {
        return true;
    }
    if (source.rows != demGrid.height || source.cols != demGrid.width)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 质量栅格尺寸不一致: %1").arg(outputPath);
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
                                          demGrid.width,
                                          demGrid.height,
                                          1,
                                          dataType,
                                          createOptions);
    CSLDestroy(createOptions);
    if (!dataset)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL 创建 DEM 质量栅格失败: %1").arg(outputPath);
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
            *errorMsg = QStringLiteral("GDAL DEM 质量栅格波段创建失败: %1").arg(outputPath);
        }
        return false;
    }
    band->SetNoDataValue(noDataValue);

    cv::Mat raster;
    if (dataType == GDT_Float32)
    {
        source.convertTo(raster, CV_32F);
        for (int row = 0; row < demGrid.height; ++row)
        {
            for (int col = 0; col < demGrid.width; ++col)
            {
                if (demGrid.validMask.at<uchar>(row, col) == 0)
                {
                    raster.at<float>(row, col) = static_cast<float>(noDataValue);
                }
            }
        }
    }
    else if (dataType == GDT_Int32)
    {
        source.convertTo(raster, CV_32S);
        for (int row = 0; row < demGrid.height; ++row)
        {
            for (int col = 0; col < demGrid.width; ++col)
            {
                if (demGrid.validMask.at<uchar>(row, col) == 0)
                {
                    raster.at<int>(row, col) = static_cast<int>(noDataValue);
                }
            }
        }
    }
    else
    {
        source.convertTo(raster, CV_8U);
        for (int row = 0; row < demGrid.height; ++row)
        {
            for (int col = 0; col < demGrid.width; ++col)
            {
                if (demGrid.validMask.at<uchar>(row, col) == 0)
                {
                    raster.at<uchar>(row, col) = static_cast<uchar>(noDataValue);
                }
            }
        }
    }

    cv::Mat writeRaster = flipForRasterWrite(raster);
    const CPLErr error = band->RasterIO(GF_Write,
                                        0,
                                        0,
                                        demGrid.width,
                                        demGrid.height,
                                        writeRaster.data,
                                        demGrid.width,
                                        demGrid.height,
                                        dataType,
                                        0,
                                        0);
    GDALClose(dataset);
    if (error != CE_None)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL 写出 DEM 质量栅格失败: %1").arg(outputPath);
        }
        return false;
    }
    return true;
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
            const int roundedValue = static_cast<int>(std::round(normalized * 255.0));
            const int clampedValue = qBound(0, roundedValue, 255);
            preview.at<uchar>(row, col) = static_cast<uchar>(clampedValue);
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
                const int roundedValue = static_cast<int>(std::round(normalized * 65535.0));
                const int clampedValue = qBound(0, roundedValue, 65535);
                u16.at<uint16_t>(row, col) = static_cast<uint16_t>(clampedValue);
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
        bool writeOk = true;
        for (int b = 0; b < 3; ++b)
        {
            cv::Mat raster(demGrid.height, demGrid.width, CV_32F, cv::Scalar(nodata));
            for (int row = 0; row < demGrid.height; ++row)
                for (int col = 0; col < demGrid.width; ++col)
                    if (demGrid.validMask.at<uchar>(row, col) != 0)
                        raster.at<float>(row, col) = bands[b]->at<float>(row, col);
            cv::Mat writeRaster = flipForRasterWrite(raster);
            GDALRasterBand *gBand = dataset->GetRasterBand(b + 1);
            if (!gBand)
            {
                writeOk = false;
                continue;
            }
            gBand->SetNoDataValue(static_cast<double>(nodata));
            writeOk = writeOk &&
                      gBand->RasterIO(GF_Write, 0, 0, demGrid.width, demGrid.height,
                                      writeRaster.data, demGrid.width, demGrid.height,
                                      GDT_Float32, 0, 0) == CE_None;
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
            if (gBand4)
            {
                gBand4->SetNoDataValue(static_cast<double>(nodata4));
                writeOk = writeOk &&
                          gBand4->RasterIO(GF_Write, 0, 0, demGrid.width, demGrid.height,
                                           writeBand4.data, demGrid.width, demGrid.height,
                                           GDT_Float32, 0, 0) == CE_None;
            }
            else
            {
                writeOk = false;
            }
        }
        else
        {
            for (int row = 0; row < demGrid.height; ++row)
                for (int col = 0; col < demGrid.width; ++col)
                    if (demGrid.validMask.at<uchar>(row, col) != 0)
                        band4.at<float>(row, col) = 1.0f;
            cv::Mat writeBand4 = flipForRasterWrite(band4);
            GDALRasterBand *gBand4 = dataset->GetRasterBand(4);
            if (gBand4)
            {
                writeOk = writeOk &&
                          gBand4->RasterIO(GF_Write, 0, 0, demGrid.width, demGrid.height,
                                           writeBand4.data, demGrid.width, demGrid.height,
                                           GDT_Float32, 0, 0) == CE_None;
            }
            else
            {
                writeOk = false;
            }
        }
        GDALClose(dataset);
        if (!writeOk)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("GDAL 写出 4-band Float32 DEM 失败: %1").arg(outputPath);
            }
            return false;
        }

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

    auto readFloatBand = [&](int bandIndex, cv::Mat *raster) {
        if (!raster)
        {
            return false;
        }

        GDALRasterBand *band = dataset->GetRasterBand(bandIndex);
        if (!band)
        {
            return false;
        }

        *raster = cv::Mat(height, width, CV_32F, cv::Scalar(0));
        if (band->RasterIO(GF_Read,
                           0,
                           0,
                           width,
                           height,
                           raster->data,
                           width,
                           height,
                           GDT_Float32,
                           0,
                           0) != CE_None)
        {
            return false;
        }

        cv::flip(*raster, *raster, 0);
        return true;
    };

    const int rasterCount = dataset->GetRasterCount();
    cv::Mat elevationRaster;
    cv::Mat worldXRaster;
    cv::Mat worldYRaster;
    cv::Mat errorRaster;
    if (rasterCount >= 4)
    {
        if (!readFloatBand(1, &worldXRaster) ||
            !readFloatBand(2, &worldYRaster) ||
            !readFloatBand(3, &elevationRaster) ||
            !readFloatBand(4, &errorRaster))
        {
            GDALClose(dataset);
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("GDAL 读取 4-band DEM 波段失败: %1").arg(inputPath);
            }
            return false;
        }
    }
    else if (!readFloatBand(1, &elevationRaster))
    {
        GDALClose(dataset);
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("GDAL 读取 DEM 波段失败: %1").arg(inputPath);
        }
        return false;
    }

    demGrid->width = width;
    demGrid->height = height;
    demGrid->elevation = elevationRaster;
    demGrid->worldX.release();
    demGrid->worldY.release();
    demGrid->triangulationError.release();
    demGrid->color.release();
    demGrid->validMask = cv::Mat(height, width, CV_8U, cv::Scalar(255));

    GDALRasterBand *elevationBand = dataset->GetRasterBand(rasterCount >= 4 ? 3 : 1);
    int hasElevationNoData = 0;
    const double elevationNoDataValue = elevationBand->GetNoDataValue(&hasElevationNoData);

    int hasWorldXNoData = 0;
    int hasWorldYNoData = 0;
    int hasErrorNoData = 0;
    double worldXNoDataValue = 0.0;
    double worldYNoDataValue = 0.0;
    double errorNoDataValue = 0.0;
    if (rasterCount >= 4)
    {
        worldXNoDataValue = dataset->GetRasterBand(1)->GetNoDataValue(&hasWorldXNoData);
        worldYNoDataValue = dataset->GetRasterBand(2)->GetNoDataValue(&hasWorldYNoData);
        errorNoDataValue = dataset->GetRasterBand(4)->GetNoDataValue(&hasErrorNoData);
        demGrid->worldX = worldXRaster;
        demGrid->worldY = worldYRaster;
        demGrid->triangulationError = errorRaster;
    }

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            const float elevationValue = elevationRaster.at<float>(row, col);
            bool valid = !isRasterNoData(elevationValue, hasElevationNoData != 0, elevationNoDataValue);
            if (valid && rasterCount >= 4)
            {
                const float xValue = worldXRaster.at<float>(row, col);
                const float yValue = worldYRaster.at<float>(row, col);
                const float errorValue = errorRaster.at<float>(row, col);
                valid = !isRasterNoData(xValue, hasWorldXNoData != 0, worldXNoDataValue) &&
                        !isRasterNoData(yValue, hasWorldYNoData != 0, worldYNoDataValue) &&
                        !isRasterNoData(errorValue, hasErrorNoData != 0, errorNoDataValue);
            }

            if (!valid)
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

bool DemDomIO::writeDemQualityRasters(const DemGridData &demGrid,
                                      const QString &outputDir,
                                      DemQualityArtifacts *artifacts,
                                      QString *errorMsg)
{
    if (artifacts)
    {
        *artifacts = DemQualityArtifacts();
    }
    if (!demGrid.isValid())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 栅格无效，无法写出质量栅格");
        }
        return false;
    }
    if (!QDir().mkpath(outputDir))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法创建 DEM 质量栅格输出目录: %1").arg(outputDir);
        }
        return false;
    }

    const QDir dir(outputDir);
    if (demGrid.hasTriangulationError())
    {
        const QString path = dir.filePath(QStringLiteral("dem_error.tif"));
        if (!writeSingleBandQualityRaster(demGrid, demGrid.triangulationError, path, GDT_Float32, -9999.0, errorMsg))
        {
            return false;
        }
        if (artifacts)
        {
            artifacts->errorPath = path;
        }
    }
    if (demGrid.hasPointCount())
    {
        const QString path = dir.filePath(QStringLiteral("dem_count.tif"));
        if (!writeSingleBandQualityRaster(demGrid, demGrid.pointCount, path, GDT_Int32, 0.0, errorMsg))
        {
            return false;
        }
        if (artifacts)
        {
            artifacts->countPath = path;
        }
    }
    if (demGrid.hasConfidence())
    {
        const QString path = dir.filePath(QStringLiteral("dem_confidence.tif"));
        if (!writeSingleBandQualityRaster(demGrid, demGrid.confidence, path, GDT_Float32, -1.0, errorMsg))
        {
            return false;
        }
        if (artifacts)
        {
            artifacts->confidencePath = path;
        }
    }
    if (demGrid.hasCoverageMask())
    {
        const QString path = dir.filePath(QStringLiteral("dem_coverage.tif"));
        if (!writeSingleBandQualityRaster(demGrid, demGrid.coverageMask, path, GDT_Byte, 0.0, errorMsg))
        {
            return false;
        }
        if (artifacts)
        {
            artifacts->coveragePath = path;
        }
    }

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
    try
    {
        plapoint::io::writeXyz(outputPath.toStdString(), denseCloud);
        return true;
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = QStringLiteral("密集点云 XYZ 写出失败: %1 (%2)")
                                      .arg(outputPath, QString::fromStdString(e.what()));
        return false;
    }
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

    auto grid = heightGridFromDemGrid(demGrid);
    auto sourceCloud = plapoint::mesh::heightGridToPointCloud(grid);
    plapoint::mesh::HeightGridOptions<float> meshOptions;
    meshOptions.maxHeightJump = std::numeric_limits<float>::max();
    meshOptions.minAbsNormalZ = 0.0f;
    meshOptions.maxFillPassForFaces = std::numeric_limits<int>::max();
    auto mesh = plapoint::mesh::heightGridToMesh(grid, sourceCloud, meshOptions);

    if (mesh.size() == 0 || !mesh.hasFaces() || mesh.faces()->rows() == 0)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 网格顶点或面为空");
        }
        return false;
    }

    QDir().mkpath(QFileInfo(outputPath).absolutePath());
    try
    {
        plapoint::io::writePly(outputPath.toStdString(), mesh, plapoint::io::PlyFormat::ASCII);
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = QStringLiteral("PLY 网格写出失败: %1 (%2)")
                                      .arg(outputPath, QString::fromStdString(e.what()));
        return false;
    }
    if (vertexCount) *vertexCount = static_cast<int>(mesh.size());
    if (faceCount) *faceCount = static_cast<int>(mesh.faces()->rows());
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

    auto cloud = pointCloudFromDemGrid(demGrid);
    const int count = static_cast<int>(cloud.size());

    if (count == 0)
    {
        if (errorMsg) *errorMsg = QStringLiteral("无有效点");
        return false;
    }

    QDir().mkpath(QFileInfo(outputPath).absolutePath());
    try
    {
        plapoint::io::writePly(outputPath.toStdString(), cloud, plapoint::io::PlyFormat::ASCII);
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = QStringLiteral("无法写出 PLY: %1 (%2)")
                                      .arg(outputPath, QString::fromStdString(e.what()));
        return false;
    }

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
