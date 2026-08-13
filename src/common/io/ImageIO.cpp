#include "io/ImageIO.h"

#include "io/PathIO.h"

#include <gdal_priv.h>
#include <opencv2/imgcodecs.hpp>

#include <QByteArray>
#include <QFileInfo>

#include <limits>
#include <memory>
#include <mutex>

namespace xjw::common::io
{
namespace
{

struct GdalDatasetDeleter
{
    void operator()(GDALDataset *dataset) const
    {
        if (dataset)
        {
            GDALClose(dataset);
        }
    }
};

using GdalDatasetPtr = std::unique_ptr<GDALDataset, GdalDatasetDeleter>;

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

void registerGdalOnce()
{
    static std::once_flag register_flag;
    std::call_once(register_flag, []()
    {
        GDALAllRegister();
    });
}

int cvDepthForGdalType(GDALDataType type)
{
    switch (type)
    {
    case GDT_Byte:
        return CV_8U;
    case GDT_UInt16:
        return CV_16U;
    case GDT_Int16:
        return CV_16S;
    case GDT_Int32:
        return CV_32S;
    case GDT_UInt32:
    case GDT_Float32:
        return CV_32F;
    case GDT_Float64:
        return CV_64F;
    default:
        return -1;
    }
}

GDALDataType gdalTypeForCvDepth(int depth)
{
    switch (depth)
    {
    case CV_8U:
        return GDT_Byte;
    case CV_16U:
        return GDT_UInt16;
    case CV_16S:
        return GDT_Int16;
    case CV_32S:
        return GDT_Int32;
    case CV_32F:
        return GDT_Float32;
    case CV_64F:
        return GDT_Float64;
    default:
        return GDT_Unknown;
    }
}

bool preservesSourceDepth(int flags)
{
    return flags == cv::IMREAD_UNCHANGED
        || (flags & cv::IMREAD_ANYDEPTH) != 0;
}

bool requestsGrayImage(int flags)
{
    return flags == cv::IMREAD_GRAYSCALE
        || flags == (cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
}

cv::Mat readImageWithGdal(const QString &path,
                          int flags,
                          QString *errorMessage)
{
    registerGdalOnce();
    GdalDatasetPtr dataset(static_cast<GDALDataset *>(
        GDALOpen(toUtf8Path(path).c_str(), GA_ReadOnly)));
    if (!dataset)
    {
        setError(errorMessage,
                 QStringLiteral("GDAL 无法打开影像 %1: %2")
                     .arg(path, QString::fromUtf8(CPLGetLastErrorMsg())));
        return {};
    }

    const int rows = dataset->GetRasterYSize();
    const int cols = dataset->GetRasterXSize();
    const int band_count = dataset->GetRasterCount();
    GDALRasterBand *first_band = dataset->GetRasterBand(1);
    if (rows <= 0 || cols <= 0 || band_count <= 0 || !first_band)
    {
        setError(errorMessage,
                 QStringLiteral("影像尺寸或波段无效: %1").arg(path));
        return {};
    }

    const GDALDataType source_type = first_band->GetRasterDataType();
    // GDAL converts UInt16 to Byte by saturation, while cv::imread scales the
    // full UInt16 range to 8 bit.  Saturation destroys nearly all texture in
    // scientific TIFF imagery (most non-zero samples become 255), which in
    // turn makes feature descriptors unusable.  Read UInt16 samples at their
    // native depth and apply the same 1/256 conversion as OpenCV.
    const bool scale_uint16_to_byte =
        !preservesSourceDepth(flags) && source_type == GDT_UInt16;
    const int cv_depth = preservesSourceDepth(flags) || scale_uint16_to_byte
        ? cvDepthForGdalType(source_type)
        : CV_8U;
    const GDALDataType gdal_type = gdalTypeForCvDepth(cv_depth);
    if (cv_depth < 0 || gdal_type == GDT_Unknown)
    {
        setError(errorMessage,
                 QStringLiteral("影像数据类型不受支持: %1").arg(path));
        return {};
    }

    // Match cv::imread semantics: IMREAD_COLOR always returns three BGR
    // channels, including for a single-band TIFF.  The previous GDAL path
    // returned CV_8UC1 for such files, so downstream vertex colorization
    // silently skipped every view that required CV_8UC3 input.
    const bool color_requested = flags == cv::IMREAD_COLOR;
    const bool gray = requestsGrayImage(flags)
        || (!color_requested && band_count < 3);
    const bool alpha = !gray
        && flags == cv::IMREAD_UNCHANGED
        && band_count >= 4;
    const int channels = gray ? 1 : (alpha ? 4 : 3);
    cv::Mat output(rows, cols, CV_MAKETYPE(cv_depth, channels));

    int band_map[4] = {1, 2, 3, 4};
    if (!gray && band_count >= 3)
    {
        band_map[0] = 3;
        band_map[1] = 2;
        band_map[2] = 1;
    }
    else if (!gray)
    {
        band_map[0] = 1;
        band_map[1] = 1;
        band_map[2] = 1;
    }
    const CPLErr result = dataset->RasterIO(
        GF_Read,
        0,
        0,
        cols,
        rows,
        output.data,
        cols,
        rows,
        gdal_type,
        channels,
        band_map,
        static_cast<GSpacing>(output.elemSize()),
        static_cast<GSpacing>(output.step),
        static_cast<GSpacing>(output.elemSize1()));
    if (result != CE_None)
    {
        setError(errorMessage,
                 QStringLiteral("GDAL 读取影像失败 %1: %2")
                     .arg(path, QString::fromUtf8(CPLGetLastErrorMsg())));
        return {};
    }
    if (scale_uint16_to_byte)
    {
        cv::Mat converted;
        output.convertTo(converted, CV_MAKETYPE(CV_8U, channels), 1.0 / 256.0);
        return converted;
    }
    return output;
}

bool prefersDirectGdalRead(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toCaseFolded();
    return suffix == QStringLiteral("tif")
        || suffix == QStringLiteral("tiff");
}

cv::Mat decodeImageBytes(const QString &path,
                         int flags,
                         QString *errorMessage)
{
    QString read_error;
    const QByteArray bytes = readFileBytes(path, &read_error);
    if (bytes.isEmpty()
        || bytes.size() > std::numeric_limits<int>::max())
    {
        setError(errorMessage,
                 read_error.isEmpty()
                     ? QStringLiteral("影像编码数据为空或过大: %1").arg(path)
                     : read_error);
        return {};
    }

    const cv::Mat encoded(
        1,
        static_cast<int>(bytes.size()),
        CV_8UC1,
        const_cast<char *>(bytes.constData()));
    cv::Mat decoded = cv::imdecode(encoded, flags);
    if (decoded.empty())
    {
        setError(errorMessage,
                 QStringLiteral("OpenCV 无法解码影像: %1").arg(path));
    }
    return decoded;
}

} // namespace

void ensureGdalRegistered()
{
    registerGdalOnce();
}

cv::Mat readImage(const QString &path, int flags)
{
    return readImage(path, flags, nullptr);
}

cv::Mat readImage(const QString &path, int flags, QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (path.trimmed().isEmpty())
    {
        setError(errorMessage, QStringLiteral("影像路径不能为空"));
        return {};
    }

    if (prefersDirectGdalRead(path))
    {
        cv::Mat image = readImageWithGdal(path, flags, errorMessage);
        if (!image.empty())
        {
            return image;
        }
    }

    cv::Mat image = decodeImageBytes(path, flags, errorMessage);
    if (!image.empty())
    {
        return image;
    }
    return readImageWithGdal(path, flags, errorMessage);
}

cv::Mat readImage(const std::string &path, int flags)
{
    return readImage(fromUtf8Path(path), flags);
}

cv::Mat readImage(const std::filesystem::path &path, int flags)
{
    return readImage(fromFilesystemPath(path), flags);
}

bool writeImage(const QString &path,
                const cv::Mat &image,
                const std::vector<int> &params)
{
    if (image.empty())
    {
        return false;
    }

    const QString suffix = QFileInfo(path).suffix().trimmed();
    if (suffix.isEmpty())
    {
        return false;
    }

    const QByteArray extension =
        QStringLiteral(".%1").arg(suffix).toLatin1();
    std::vector<uchar> encoded;
    if (!cv::imencode(extension.constData(), image, encoded, params))
    {
        return false;
    }

    const QByteArray bytes(
        reinterpret_cast<const char *>(encoded.data()),
        static_cast<qsizetype>(encoded.size()));
    return writeFileBytesAtomic(path, bytes);
}

bool writeImage(const std::string &path,
                const cv::Mat &image,
                const std::vector<int> &params)
{
    return writeImage(fromUtf8Path(path), image, params);
}

bool writeImage(const std::filesystem::path &path,
                const cv::Mat &image,
                const std::vector<int> &params)
{
    return writeImage(fromFilesystemPath(path), image, params);
}

} // namespace xjw::common::io
