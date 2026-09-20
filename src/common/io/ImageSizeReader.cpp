#include "io/ImageIO.h"
#include "io/PathIO.h"

#include <gdal_priv.h>

#include <QFile>
#include <QtEndian>

#include <limits>
#include <memory>

namespace xjw::common::io
{
    namespace
    {

        QSize readBmpHeaderSize(const QString& path)
        {
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly))
            {
                return {};
            }
            const QByteArray header = file.read(26);
            if (header.size() != 26 || !header.startsWith("BM"))
            {
                return {};
            }
            const quint32 dib_size = qFromLittleEndian<quint32>(header.constData() + 14);
            if (static_cast<qint64>(dib_size) + 14 > file.size())
            {
                return {};
            }
            if (dib_size == 12)
            {
                return QSize(qFromLittleEndian<quint16>(header.constData() + 18),
                             qFromLittleEndian<quint16>(header.constData() + 20));
            }
            if (dib_size < 40)
            {
                return {};
            }
            const qint32 width = qFromLittleEndian<qint32>(header.constData() + 18);
            const qint32 height = qFromLittleEndian<qint32>(header.constData() + 22);
            if (width <= 0 || height == 0 || height == std::numeric_limits<qint32>::min())
            {
                return {};
            }
            // A negative BMP height denotes top-down storage, not a rotated image.
            return QSize(width, height < 0 ? -height : height);
        }

    } // namespace

    QSize readImageSize(const QString& path, QString* errorMessage)
    {
        if (errorMessage)
        {
            errorMessage->clear();
        }
        const auto fail = [&](const QString& message) -> QSize
        {
            if (errorMessage)
            {
                *errorMessage = message;
            }
            return {};
        };
        if (path.trimmed().isEmpty())
        {
            return fail(QStringLiteral("影像路径不能为空"));
        }

        ensureGdalRegistered();
        CPLErrorReset();
        // Opening a dataset reads its header; no RasterIO or full-image allocation is needed.
        std::unique_ptr<GDALDataset, decltype(&GDALClose)> dataset(
            static_cast<GDALDataset*>(
                GDALOpenEx(toUtf8Path(path).c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)),
            &GDALClose);
        if (!dataset)
        {
            const QSize bmp_size = readBmpHeaderSize(path);
            if (bmp_size.width() > 0 && bmp_size.height() > 0)
            {
                return bmp_size;
            }
            const QString reason = QString::fromUtf8(CPLGetLastErrorMsg());
            return fail(QStringLiteral("无法读取影像尺寸 %1: %2")
                            .arg(path, reason.isEmpty() ? QStringLiteral("文件无法打开或格式不受 GDAL 支持") : reason));
        }
        const int width = dataset->GetRasterXSize();
        const int height = dataset->GetRasterYSize();
        if (width <= 0 || height <= 0)
        {
            return fail(QStringLiteral("影像尺寸无效: %1").arg(path));
        }
        return QSize(width, height);
    }

} // namespace xjw::common::io
