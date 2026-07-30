#include "LayerImageLoader.h"

#include "Logger.h"
#include "io/PathIO.h"
#include "project/ProjectIO.h"

#include <opencv2/opencv.hpp>

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QImageReader>

#include <algorithm>
#include <cmath>
#include <vector>

#include <cpl_conv.h>
#include <gdal.h>
#include <gdal_priv.h>

namespace
{

QImage convertGray16ToGray8(const QImage &img16)
{
    if (img16.format() != QImage::Format_Grayscale16)
    {
        return QImage();
    }

    QImage out(img16.size(), QImage::Format_Grayscale8);
    if (out.isNull())
    {
        return QImage();
    }

    for (int y = 0; y < img16.height(); ++y)
    {
        const auto *src = reinterpret_cast<const quint16 *>(img16.constScanLine(y));
        uchar *dst = out.scanLine(y);
        for (int x = 0; x < img16.width(); ++x)
        {
            dst[x] = static_cast<uchar>(src[x] >> 8);
        }
    }

    return out;
}

float percentile(std::vector<float> &values, double p)
{
    if (values.empty())
    {
        return 0.0f;
    }
    if (p <= 0.0)
    {
        std::nth_element(values.begin(), values.begin(), values.end());
        return values.front();
    }
    if (p >= 1.0)
    {
        std::nth_element(values.begin(), values.end() - 1, values.end());
        return values.back();
    }

    const double pos = p * static_cast<double>(values.size() - 1);
    const size_t idx = static_cast<size_t>(pos);
    std::nth_element(values.begin(), values.begin() + static_cast<long>(idx), values.end());
    return values[idx];
}

bool convertTo8BitGeoTiff_GDAL(const QString &inputPath,
                               const QString &outputPath,
                               double lowP = 0.02,
                               double highP = 0.98,
                               bool forceUseBandNoData = true)
{
    static bool gdalInited = false;
    if (!gdalInited)
    {
        GDALAllRegister();
        gdalInited = true;
    }

    const std::string inputPathUtf8 = xjw::common::io::toUtf8Path(inputPath);
    GDALDataset *ds = static_cast<GDALDataset *>(GDALOpen(inputPathUtf8.c_str(), GA_ReadOnly));
    if (!ds)
    {
        return false;
    }

    const int width = ds->GetRasterXSize();
    const int height = ds->GetRasterYSize();
    const int bandCount = ds->GetRasterCount();
    if (width <= 0 || height <= 0 || bandCount <= 0)
    {
        GDALClose(ds);
        return false;
    }

    std::vector<std::vector<float>> bandBuffers;
    bandBuffers.resize(static_cast<size_t>(bandCount));

    std::vector<float> valid;
    valid.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));

    int hasNoData = 0;
    double noDataValue = 0.0;
    if (forceUseBandNoData)
    {
        GDALRasterBand *b0 = ds->GetRasterBand(1);
        if (b0)
        {
            noDataValue = b0->GetNoDataValue(&hasNoData);
        }
    }

    for (int b = 1; b <= bandCount; ++b)
    {
        GDALRasterBand *band = ds->GetRasterBand(b);
        if (!band)
        {
            GDALClose(ds);
            return false;
        }

        std::vector<float> buffer;
        buffer.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
        const CPLErr err = band->RasterIO(GF_Read,
                                          0,
                                          0,
                                          width,
                                          height,
                                          buffer.data(),
                                          width,
                                          height,
                                          GDT_Float32,
                                          0,
                                          0);
        if (err != CE_None)
        {
            GDALClose(ds);
            return false;
        }

        for (float v : buffer)
        {
            if (!std::isfinite(v))
            {
                continue;
            }
            if (hasNoData && static_cast<double>(v) == noDataValue)
            {
                continue;
            }
            if (std::fabs(static_cast<double>(v)) > 1.0e30)
            {
                continue;
            }
            valid.push_back(v);
        }

        bandBuffers[static_cast<size_t>(b - 1)] = std::move(buffer);
    }

    if (valid.empty())
    {
        GDALClose(ds);
        return false;
    }

    std::vector<float> tmp = valid;
    const float vmin = percentile(tmp, lowP);
    tmp = valid;
    const float vmax = percentile(tmp, highP);
    if (!(std::isfinite(vmin) && std::isfinite(vmax)) || vmax <= vmin)
    {
        GDALClose(ds);
        return false;
    }

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver)
    {
        GDALClose(ds);
        return false;
    }

    char **options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "TILED", "YES");

    const std::string outputPathUtf8 = xjw::common::io::toUtf8Path(outputPath);
    GDALDataset *outDs = driver->Create(outputPathUtf8.c_str(),
                                        width,
                                        height,
                                        bandCount,
                                        GDT_Byte,
                                        options);
    CSLDestroy(options);
    if (!outDs)
    {
        GDALClose(ds);
        return false;
    }

    double geoTransform[6];
    if (ds->GetGeoTransform(geoTransform) == CE_None)
    {
        outDs->SetGeoTransform(geoTransform);
    }
    const char *proj = ds->GetProjectionRef();
    if (proj && proj[0] != '\0')
    {
        outDs->SetProjection(proj);
    }

    std::vector<unsigned char> outBuf;
    outBuf.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int b = 1; b <= bandCount; ++b)
    {
        GDALRasterBand *outBand = outDs->GetRasterBand(b);
        if (!outBand)
        {
            GDALClose(ds);
            GDALClose(outDs);
            return false;
        }
        outBand->SetNoDataValue(0);

        const std::vector<float> &src = bandBuffers[static_cast<size_t>(b - 1)];
        for (size_t i = 0; i < src.size(); ++i)
        {
            const float v = src[i];
            bool isValid = std::isfinite(v);
            if (isValid && hasNoData && static_cast<double>(v) == noDataValue)
            {
                isValid = false;
            }
            if (isValid && std::fabs(static_cast<double>(v)) > 1.0e30)
            {
                isValid = false;
            }
            if (!isValid)
            {
                outBuf[i] = 0;
                continue;
            }

            double vv = static_cast<double>(v);
            if (vv < vmin)
            {
                vv = vmin;
            }
            if (vv > vmax)
            {
                vv = vmax;
            }
            const double n = (vv - static_cast<double>(vmin)) /
                             (static_cast<double>(vmax) - static_cast<double>(vmin));
            int outv = static_cast<int>(std::lround(n * 255.0));
            if (outv < 0)
            {
                outv = 0;
            }
            if (outv > 255)
            {
                outv = 255;
            }
            outBuf[i] = static_cast<unsigned char>(outv);
        }

        const CPLErr werr = outBand->RasterIO(GF_Write,
                                              0,
                                              0,
                                              width,
                                              height,
                                              outBuf.data(),
                                              width,
                                              height,
                                              GDT_Byte,
                                              0,
                                              0);
        if (werr != CE_None)
        {
            GDALClose(ds);
            GDALClose(outDs);
            return false;
        }
        outBand->FlushCache();
    }

    GDALClose(ds);
    GDALClose(outDs);
    return true;
}

bool needsConvertTo8Bit_GDAL(const QString &inputPath)
{
    static bool gdalInited = false;
    if (!gdalInited)
    {
        GDALAllRegister();
        gdalInited = true;
    }

    const std::string inputPathUtf8 = xjw::common::io::toUtf8Path(inputPath);
    GDALDataset *ds = static_cast<GDALDataset *>(GDALOpen(inputPathUtf8.c_str(), GA_ReadOnly));
    if (!ds)
    {
        return false;
    }

    const int bandCount = ds->GetRasterCount();
    if (bandCount <= 0)
    {
        GDALClose(ds);
        return false;
    }

    bool need = false;
    for (int b = 1; b <= bandCount; ++b)
    {
        GDALRasterBand *band = ds->GetRasterBand(b);
        if (!band)
        {
            continue;
        }
        const GDALDataType t = band->GetRasterDataType();
        if (t != GDT_Byte)
        {
            need = true;
            break;
        }
    }
    GDALClose(ds);
    return need;
}

QString hexSha1(const QByteArray &data)
{
    return QString::fromLatin1(QCryptographicHash::hash(data, QCryptographicHash::Sha1).toHex());
}

QImage imageFromOpenCvMat(const cv::Mat &mat)
{
    if (mat.empty())
    {
        return QImage();
    }

    cv::Mat displayMat;
    if (mat.depth() != CV_8U)
    {
        cv::Mat grayOrColor;
        const int channels = mat.channels();
        if (channels == 1 || channels == 3 || channels == 4)
        {
            cv::normalize(mat, grayOrColor, 0, 255, cv::NORM_MINMAX, CV_8U);
            displayMat = grayOrColor;
        }
        else
        {
            return QImage();
        }
    }
    else
    {
        displayMat = mat;
    }

    if (displayMat.channels() == 1)
    {
        return QImage(displayMat.data,
                      displayMat.cols,
                      displayMat.rows,
                      static_cast<int>(displayMat.step),
                      QImage::Format_Grayscale8).copy();
    }

    cv::Mat converted;
    if (displayMat.channels() == 3)
    {
        cv::cvtColor(displayMat, converted, cv::COLOR_BGR2RGB);
        return QImage(converted.data,
                      converted.cols,
                      converted.rows,
                      static_cast<int>(converted.step),
                      QImage::Format_RGB888).copy();
    }
    if (displayMat.channels() == 4)
    {
        cv::cvtColor(displayMat, converted, cv::COLOR_BGRA2RGBA);
        return QImage(converted.data,
                      converted.cols,
                      converted.rows,
                      static_cast<int>(converted.step),
                      QImage::Format_RGBA8888).copy();
    }

    return QImage();
}

QImage loadImageWithOpenCvByteDecode(const QString &path)
{
    return imageFromOpenCvMat(xjw::common::io::readImage(path, cv::IMREAD_UNCHANGED));
}

QString make8BitCachePath(const QString &inputPath, const QString &projectRoot)
{
    QFileInfo fi(inputPath);
    const QString abs = fi.absoluteFilePath();
    const QByteArray key = (abs + QStringLiteral("|") + QString::number(fi.size()) + QStringLiteral("|") +
                            QString::number(fi.lastModified().toMSecsSinceEpoch())).toUtf8();
    const QString id = hexSha1(key);
    const QString fileName = id + QStringLiteral("_") + fi.completeBaseName() + QStringLiteral("_8.tif");

    if (!projectRoot.trimmed().isEmpty())
    {
        QDir root(projectRoot);
        const QString outDir = root.filePath(QStringLiteral(".plascan_tmp/converted_images"));
        QDir d(outDir);
        d.mkpath(QStringLiteral("."));
        return d.filePath(fileName);
    }

    return QDir(fi.absolutePath()).filePath(fi.completeBaseName() + QStringLiteral("_8.tif"));
}

bool isCacheFresh(const QString &inputPath, const QString &cachePath)
{
    QFileInfo inFi(inputPath);
    QFileInfo outFi(cachePath);
    if (!outFi.exists())
    {
        return false;
    }
    return outFi.lastModified() >= inFi.lastModified();
}

} // namespace

namespace xjw::gui::views
{

QImage loadImageForDisplay(const QString &path, const QString &plascanPath)
{
    QImage img;
    QImageReader reader(path);
    reader.setAutoTransform(false);
    if (reader.canRead())
    {
        img = reader.read();

        if (!img.isNull() && img.format() == QImage::Format_Grayscale16)
        {
            QImage mapped = convertGray16ToGray8(img);
            if (!mapped.isNull())
            {
                img = mapped;
            }
        }
    }
    if (img.isNull())
    {
        img = loadImageWithOpenCvByteDecode(path);
    }
    if (img.isNull())
    {
        img = QImage(path);
    }

    const bool needConvert = needsConvertTo8Bit_GDAL(path);
    if (img.isNull() || needConvert)
    {
        const QString projectRoot = plascanPath.trimmed().isEmpty()
            ? QString()
            : xjw::common::project::ProjectIO::projectRootFromPlascan(
                  plascanPath);
        const QString out8 = make8BitCachePath(path, projectRoot);
        const bool fresh = isCacheFresh(path, out8);

        if (!fresh)
        {
            (void)convertTo8BitGeoTiff_GDAL(path, out8);
        }

        if (QFileInfo::exists(out8))
        {
            QImageReader r2(out8);
            if (r2.canRead())
            {
                img = r2.read();
            }
            if (img.isNull())
            {
                img = QImage(out8);
            }
        }
    }
    if (img.isNull())
    {
        LOG_WARN(QStringLiteral("addImageLayer: failed to load image %1").arg(path));
        return QImage();
    }
    return img;
}

} // namespace xjw::gui::views
