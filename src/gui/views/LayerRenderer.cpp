#include "LayerRenderer.h"
#include "Logger.h"
#include "ProjectIO.h"

#include <opencv2/opencv.hpp>

// 为避免Qt宏与LibTorch冲突,在包含SuperPoint.h前undef
#ifdef slots
  #undef slots
  #define NEED_RESTORE_SLOTS  
#endif
#ifdef signals
  #undef signals
  #define NEED_RESTORE_SIGNALS
#endif
#ifdef emit
  #undef emit
  #define NEED_RESTORE_EMIT
#endif

#include "SuperPoint.h"
#include "FeatureFileIO.h"

// 恢复Qt宏
#ifdef NEED_RESTORE_SLOTS
  #define slots Q_SLOTS
  #undef NEED_RESTORE_SLOTS
#endif
#ifdef NEED_RESTORE_SIGNALS
  #define signals Q_SIGNALS
  #undef NEED_RESTORE_SIGNALS
#endif
#ifdef NEED_RESTORE_EMIT
  #define emit Q_EMIT
  #undef NEED_RESTORE_EMIT
#endif

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QImageReader>
#include <QPixmap>
#include <QDir>
#include <QFileInfo>
#include <QCryptographicHash>
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <cmath>
#include <algorithm>

// GDAL：用于读取 Float32/高位深 GeoTIFF 并做分位裁剪归一化显示
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsLineItem>
#include <QVector>
#include <QPointF>
#include <QPainter>
#include <QGraphicsPixmapItem>



// 16 位灰度影像的简单显示映射：
// - GUI 显示通常以 8-bit 为主，因此需要把 16-bit 映射到 8-bit。
// - 这里使用线性缩放（value >> 8）作为“最小可用”方案，后续可扩展为直方图拉伸/百分位裁剪等。
static QImage convertGray16ToGray8(const QImage &img16)
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
        const quint16 *src = reinterpret_cast<const quint16*>(img16.constScanLine(y));
        uchar *dst = out.scanLine(y);
        for (int x = 0; x < img16.width(); ++x)
        {
            dst[x] = static_cast<uchar>(src[x] >> 8);
        }
    }

    return out;
}

// 计算分位数（p in [0,1]），会复制并部分排序数据。
static float percentile(std::vector<float> &values, double p)
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

// 使用 GDAL 将 32-bit/高位深影像转为 8-bit GeoTIFF（按 2%-98% 分位裁剪），并保留地理信息。
// 逻辑参考你提供的 Python 脚本：
// - 读取所有波段，合并计算分位数
// - clip 到 [min,max]
// - 归一化到 0~255 (uint8)
// - 输出 GTiff，带 COMPRESS=LZW/TILED=YES，并设置 NoData=0
static bool convertTo8BitGeoTiff_GDAL(const QString &inputPath,
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

    GDALDataset *ds = static_cast<GDALDataset*>(GDALOpen(inputPath.toStdString().c_str(), GA_ReadOnly));
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

    // 读取每个波段为 float32，并汇总有效值用于计算总体分位数
    std::vector<std::vector<float>> bandBuffers;
    bandBuffers.resize(static_cast<size_t>(bandCount));

    std::vector<float> valid;
    valid.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));

    // NoData：按你的脚本逻辑，默认优先使用 band1 的 nodata
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
                                         0, 0,
                                         width, height,
                                         buffer.data(),
                                         width, height,
                                         GDT_Float32,
                                         0, 0);
        if (err != CE_None)
        {
            GDALClose(ds);
            return false;
        }

        // 收集有效值
        for (size_t i = 0; i < buffer.size(); ++i)
        {
            const float v = buffer[i];
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

    // 创建输出 GeoTIFF（8-bit）
    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver)
    {
        GDALClose(ds);
        return false;
    }

    char **options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "TILED", "YES");

    GDALDataset *outDs = driver->Create(outputPath.toStdString().c_str(),
                                        width, height,
                                        bandCount,
                                        GDT_Byte,
                                        options);
    CSLDestroy(options);
    if (!outDs)
    {
        GDALClose(ds);
        return false;
    }

    // 复制地理信息
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

    // 将每个波段写出为 8-bit
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
            if (vv < vmin) vv = vmin;
            if (vv > vmax) vv = vmax;
            const double n = (vv - static_cast<double>(vmin)) / (static_cast<double>(vmax) - static_cast<double>(vmin));
            int outv = static_cast<int>(std::lround(n * 255.0));
            if (outv < 0) outv = 0;
            if (outv > 255) outv = 255;
            outBuf[i] = static_cast<unsigned char>(outv);
        }

        const CPLErr werr = outBand->RasterIO(GF_Write,
                                             0, 0,
                                             width, height,
                                             outBuf.data(),
                                             width, height,
                                             GDT_Byte,
                                             0, 0);
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

// 用 GDAL 快速探测是否需要转换为 8-bit 才能显示。
// 规则：
// - 只有当所有波段都是 Byte(8-bit) 时，认为无需转换。
// - 只要存在非 Byte（例如 UInt16/Float32），就需要转换。
static bool needsConvertTo8Bit_GDAL(const QString &inputPath)
{
    static bool gdalInited = false;
    if (!gdalInited)
    {
        GDALAllRegister();
        gdalInited = true;
    }

    GDALDataset *ds = static_cast<GDALDataset*>(GDALOpen(inputPath.toStdString().c_str(), GA_ReadOnly));
    if (!ds)
    {
        // 无法判断时，保持原逻辑：让 Qt 先尝试，失败后再走转换。
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

static QString hexSha1(const QByteArray &data)
{
    return QString::fromLatin1(QCryptographicHash::hash(data, QCryptographicHash::Sha1).toHex());
}

// 生成 8-bit 缓存路径：
// - 若已打开项目：写入 <projectRoot>/.plascan_tmp/converted_images/
// - 否则回退：与输入同目录、同文件名追加 _8。
// 命名：sha1(absolutePath|size|mtime) + "_" + 原始 baseName + "_8.tif"。
static QString make8BitCachePath(const QString &inputPath, const QString &projectRoot)
{
    QFileInfo fi(inputPath);
    const QString abs = fi.absoluteFilePath();
    const QByteArray key = (abs + QStringLiteral("|") + QString::number(fi.size()) + QStringLiteral("|") + QString::number(fi.lastModified().toMSecsSinceEpoch())).toUtf8();
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

static bool isCacheFresh(const QString &inputPath, const QString &cachePath)
{
    QFileInfo inFi(inputPath);
    QFileInfo outFi(cachePath);
    if (!outFi.exists())
    {
        return false;
    }
    return outFi.lastModified() >= inFi.lastModified();
}

namespace
{
class BatchedFeatureOverlayItem : public QGraphicsItem
{
public:
    BatchedFeatureOverlayItem(std::vector<cv::KeyPoint> keypoints,
                              const LayerRenderer::FeatureDisplayOptions &options,
                              const QRectF &imageBounds)
        : m_keypoints(std::move(keypoints))
        , m_options(options)
        , m_bounds(computeBounds(m_keypoints, options, imageBounds))
    {
        if (m_options.maxDisplayCount > 0)
        {
            if (m_options.showTopScores)
            {
                std::sort(m_keypoints.begin(), m_keypoints.end(),
                          [](const auto &a, const auto &b)
                          {
                              return a.response > b.response;
                          });
            }
            if (static_cast<int>(m_keypoints.size()) > m_options.maxDisplayCount)
            {
                m_keypoints.resize(static_cast<size_t>(m_options.maxDisplayCount));
                m_bounds = computeBounds(m_keypoints, m_options, imageBounds);
            }
        }
        setZValue(1000.0);
    }

    QRectF boundingRect() const override
    {
        return m_bounds;
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) override
    {
        if (!painter || m_keypoints.empty() || !m_options.showPoints)
        {
            return;
        }

        painter->setRenderHint(QPainter::Antialiasing, true);

        QColor pointColor = m_options.pointColor;
        pointColor.setAlpha(m_options.opacity);
        QPen pointPen(pointColor);
        pointPen.setWidthF(1.5);
        pointPen.setCosmetic(true);
        QBrush pointBrush = m_options.useFill
                                ? QBrush(pointColor)
                                : QBrush(Qt::NoBrush);

        for (const auto &kp : m_keypoints)
        {
            drawKeypoint(painter, kp, pointPen, pointBrush);
        }
    }

private:
    static double markerRadius(const cv::KeyPoint &keypoint,
                               const LayerRenderer::FeatureDisplayOptions &options)
    {
        const double sizeFactor = static_cast<double>(options.pointSize) * options.scaleMultiplier;
        return std::max(1.0, std::min(100.0, static_cast<double>(keypoint.size) * sizeFactor));
    }

    static QRectF computeBounds(const std::vector<cv::KeyPoint> &keypoints,
                                const LayerRenderer::FeatureDisplayOptions &options,
                                const QRectF &imageBounds)
    {
        QRectF bounds = imageBounds;
        for (const auto &kp : keypoints)
        {
            const double r = std::max(markerRadius(kp, options), 8.0);
            const QRectF kpRect(static_cast<qreal>(kp.pt.x - r),
                                static_cast<qreal>(kp.pt.y - r),
                                static_cast<qreal>(r * 2.0),
                                static_cast<qreal>(r * 2.0));
            bounds = bounds.isNull() ? kpRect : bounds.united(kpRect);
        }

        if (bounds.isNull())
        {
            return QRectF();
        }
        return bounds.adjusted(-4.0, -4.0, 4.0, 4.0);
    }

    void drawKeypoint(QPainter *painter,
                      const cv::KeyPoint &kp,
                      const QPen &pointPen,
                      const QBrush &pointBrush) const
    {
        const double r = markerRadius(kp, m_options);
        const QPointF center(static_cast<qreal>(kp.pt.x), static_cast<qreal>(kp.pt.y));

        painter->setPen(pointPen);
        painter->setBrush(pointBrush);

        if (m_options.markerShape == QLatin1String("circle"))
        {
            painter->drawEllipse(center, r, r);
        }
        else if (m_options.markerShape == QLatin1String("square"))
        {
            painter->drawRect(QRectF(center.x() - r, center.y() - r, r * 2.0, r * 2.0));
        }
        else if (m_options.markerShape == QLatin1String("cross"))
        {
            QPen crossPen(m_options.pointColor);
            crossPen.setWidthF(1.0);
            crossPen.setCosmetic(true);
            painter->setPen(crossPen);
            const double crossRadius = std::max(
                1.0,
                static_cast<double>(m_options.pointSize) * m_options.scaleMultiplier);
            painter->drawLine(QPointF(center.x() - crossRadius, center.y() - crossRadius),
                              QPointF(center.x() + crossRadius, center.y() + crossRadius));
            painter->drawLine(QPointF(center.x() - crossRadius, center.y() + crossRadius),
                              QPointF(center.x() + crossRadius, center.y() - crossRadius));
        }
        else if (m_options.markerShape == QLatin1String("dot"))
        {
            QPen dotPen(m_options.pointColor);
            dotPen.setWidthF(0.5);
            dotPen.setCosmetic(true);
            QColor fill = m_options.pointColor;
            fill.setAlpha(m_options.opacity);
            painter->setPen(dotPen);
            painter->setBrush(QBrush(fill));
            const double dotR = std::min(3.0, r * 0.4);
            painter->drawEllipse(center, dotR, dotR);
        }
        else if (m_options.markerShape == QLatin1String("point"))
        {
            QColor fill = m_options.pointColor;
            fill.setAlpha(m_options.opacity);
            QPen pointPixelPen(fill);
            pointPixelPen.setWidthF(0.0);
            pointPixelPen.setCosmetic(true);
            painter->setPen(pointPixelPen);
            painter->setBrush(QBrush(fill));
            painter->drawRect(QRectF(center.x(), center.y(), 1.0, 1.0));
        }
        else
        {
            painter->drawEllipse(center, r, r);
        }

        if (m_options.showScale)
        {
            QPen scalePen(m_options.scaleColor);
            scalePen.setWidthF(0.8);
            scalePen.setCosmetic(true);
            painter->setPen(scalePen);
            painter->setBrush(Qt::NoBrush);
            const double scaleRadius = static_cast<double>(kp.size) * m_options.scaleMultiplier;
            painter->drawEllipse(center, scaleRadius, scaleRadius);
        }

        if (m_options.showOrientation && kp.angle >= 0.0f)
        {
            QPen orientPen(m_options.orientColor);
            orientPen.setWidthF(1.5);
            orientPen.setCosmetic(true);
            painter->setPen(orientPen);

            const double orientRad = static_cast<double>(kp.angle) * M_PI / 180.0;
            const double arrowLen = r * 1.8;
            const QPointF end(center.x() + arrowLen * std::cos(orientRad),
                              center.y() + arrowLen * std::sin(orientRad));
            painter->drawLine(center, end);

            const double arrowHeadLen = r * 0.6;
            const double angle1 = orientRad + M_PI * 0.85;
            const double angle2 = orientRad - M_PI * 0.85;
            painter->drawLine(end,
                              QPointF(end.x() + arrowHeadLen * std::cos(angle1),
                                      end.y() + arrowHeadLen * std::sin(angle1)));
            painter->drawLine(end,
                              QPointF(end.x() + arrowHeadLen * std::cos(angle2),
                                      end.y() + arrowHeadLen * std::sin(angle2)));
        }
    }

    std::vector<cv::KeyPoint> m_keypoints;
    LayerRenderer::FeatureDisplayOptions m_options;
    QRectF m_bounds;
};
} // namespace

LayerRenderer::LayerRenderer(QGraphicsScene *scene, QObject *parent)
    : QObject(parent)
    , m_scene(scene)
{
}

void LayerRenderer::setFeatureDisplayOptions(const FeatureDisplayOptions &opts)
{
    // store options for future rendering
    // We'll use these in addFeatureItems when drawing items
    // keep a copy as a member variable
    // Use m_featureOpts (add member below)
    m_featureOpts = opts;
}

void LayerRenderer::setCurrentProjectPath(const QString &plascanPath)
{
    m_currentProjectPath = plascanPath;
}

bool LayerRenderer::addImageLayer(const QString &path, int z)
{
    return addImageLayer(loadImageForDisplay(path, m_currentProjectPath), z);
}

QImage LayerRenderer::loadImageForDisplay(const QString &path, const QString &plascanPath)
{
    QImage img;
    QImageReader reader(path);
    if (reader.canRead())
    {
        img = reader.read();

        // 尝试处理 16 位灰度显示
        // 说明：即使 reader 读到了 16-bit，Qt 的渲染也未必能直接在 QGraphicsPixmapItem 上正确显示。
        // 因此显式把 16-bit 映射为 8-bit，保证“至少能看”。
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
        img = QImage(path);
    }
    // 显示策略：
    // - 用户点击一个影像后，先判断它是否是 8-bit（GDAL 探测）。
    // - 若不是 8-bit，则在项目的 .plascan_tmp 下生成缓存的 8-bit GeoTIFF，再显示该缓存。
    // - 后续再次打开同一影像，若缓存存在且未过期，则直接使用缓存。
    // - 若当前未打开项目，则回退为“与输入同目录生成 _8.tif”。

    // 只有当 Qt 本身无法读，或者 GDAL 判断为非 8-bit，才触发转换。
    const bool needConvert = needsConvertTo8Bit_GDAL(path);
    if (img.isNull() || needConvert)
    {
        const QString projectRoot = plascanPath.trimmed().isEmpty()
                                        ? QString()
                                        : QFileInfo(plascanPath).absolutePath();
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

bool LayerRenderer::addImageLayer(const QImage &image, int z)
{
    if (!m_scene || image.isNull())
    {
        return false;
    }

    QPixmap pix = QPixmap::fromImage(image);
    auto *item = m_scene->addPixmap(pix);
    if (!item)
    {
        return false;
    }
    // Ensure newly added pixmap is positioned at origin and visible
    item->setPos(0, 0);
    item->setVisible(true);
    item->setZValue(z);
    m_layers.append(item);
    m_imageBounds = m_imageBounds.isNull() ? item->sceneBoundingRect() : m_imageBounds.united(item->sceneBoundingRect());
    return true;
}

bool LayerRenderer::addFeatureLayerFromVwip(const QString &imagePath)
{
    if (!m_scene) return false;
    
    // 使用ProjectIO查找特征文件(支持所有提取器后缀)
    QString spPath = ProjectIO::findFeatureForImage(m_currentProjectPath, imagePath);
    if (spPath.isEmpty()) {
        qDebug() << "LayerRenderer: No feature file found for" << imagePath;
        return false;
    }

    // 使用FeatureFileIO读取.sp文件
    QString imageName;
    FeatureOutput output;
    if (!FeatureFileIO::read(spPath, imageName, output)) {
        qWarning() << "Failed to read .sp file:" << spPath;
        return false;
    }
    
    // FeatureOutput.keypoints已经是std::vector<cv::KeyPoint>
    // scores已经存在output.scores中,将其存入KeyPoint.response字段
    for (size_t i = 0; i < output.keypoints.size() && i < output.scores.size(); ++i) {
        output.keypoints[i].response = output.scores[i];
    }
    
    if (output.keypoints.empty()) return false;
    addFeatureItems(output.keypoints);
    return true;
}

void LayerRenderer::clearFeatureLayers()
{
    // Remove items we explicitly tracked
    for (auto *it: std::as_const(m_featureItems))
    {
        if (it && m_scene)
        {
            m_scene->removeItem(it);
            delete it;
        }
    }
    m_featureItems.clear();
}

void LayerRenderer::addFeatureItems(const std::vector<cv::KeyPoint> &keypoints)
{
    if (!m_scene) return;
    // Debug incoming keypoints for troubleshooting feature rendering
    LOG_DEBUG(QStringLiteral("addFeatureItems: incoming keypoints=%1").arg(static_cast<int>(keypoints.size())));

    if (keypoints.empty() || !m_featureOpts.showPoints)
    {
        return;
    }

    auto *item = new BatchedFeatureOverlayItem(keypoints, m_featureOpts, m_imageBounds);
    m_scene->addItem(item);
    m_featureItems.append(item);
    const int added = 1;
    LOG_DEBUG(QStringLiteral("addFeatureItems: added items=%1 total_scene_items=%2").arg(added).arg(m_scene ? m_scene->items().size() : 0));
}

void LayerRenderer::clear()
{
    for (auto *it: std::as_const(m_layers))
    {
        if (it && m_scene)
        {
            m_scene->removeItem(it);
            delete it;
        }
    }
    m_layers.clear();
    m_imageBounds = QRectF();
}

bool LayerRenderer::addStitchedImagePair(const QString &pathA, const QString &pathB, QGraphicsPixmapItem **outA, QGraphicsPixmapItem **outB, int gap)
{
    if (!m_scene) return false;

    LOG_DEBUG(QStringLiteral("addStitchedImagePair: %1 <-> %2").arg(pathA, pathB));

    // clear existing image layers (we expect caller to manage state)
    // We'll add both images using addImageLayer then reposition the second.
    const int before = m_layers.size();
    if (!addImageLayer(pathA, 0)) return false;
    QGraphicsPixmapItem *itemA = nullptr;
    if (!m_layers.isEmpty()) itemA = m_layers.last();

    if (!addImageLayer(pathB, 0)) {
        // cleanup the first if second failed
        if (itemA) {
            m_scene->removeItem(itemA);
            m_layers.removeOne(itemA);
            delete itemA;
        }
        return false;
    }
    QGraphicsPixmapItem *itemB = nullptr;
    if (!m_layers.isEmpty() && m_layers.size() > before + 0) itemB = m_layers.last();

    if (!itemA || !itemB) return false;

    LOG_DEBUG(QStringLiteral("addStitchedImagePair: itemA size=%1x%2 itemB size=%3x%4").arg(itemA->pixmap().width()).arg(itemA->pixmap().height()).arg(itemB->pixmap().width()).arg(itemB->pixmap().height()));

    // position B to the right of A
    qreal bx = itemA->pixmap().width() + gap;
    itemB->setPos(bx, 0);
    m_imageBounds = itemA->sceneBoundingRect().united(itemB->sceneBoundingRect());

    if (outA) *outA = itemA;
    if (outB) *outB = itemB;

    // Debug: save a stitched composite image to disk for inspection and log scene items
    try {
        const int wa = itemA->pixmap().width();
        const int ha = itemA->pixmap().height();
        const int wb = itemB->pixmap().width();
        const int hb = itemB->pixmap().height();
        const int h = std::max(ha, hb);
        const int w = wa + gap + wb;

        QImage out(w, h, QImage::Format_ARGB32);
        out.fill(Qt::transparent);
        QPainter p(&out);
        p.drawPixmap(0, 0, itemA->pixmap());
        p.drawPixmap(wa + gap, 0, itemB->pixmap());
        p.end();

        // determine debug output directory
        QString debugDir;
        if (!m_currentProjectPath.trimmed().isEmpty()) {
            const QString projectRoot = QFileInfo(m_currentProjectPath).absolutePath();
            QDir d(projectRoot);
            debugDir = d.filePath(QStringLiteral(".plascan_tmp/debug"));
        } else {
            debugDir = QFileInfo(pathA).absolutePath() + QDir::separator() + QStringLiteral("plascan_debug");
        }
        QDir dd(debugDir);
        dd.mkpath(QStringLiteral("."));

        QByteArray key = (pathA + QStringLiteral("|") + pathB + QStringLiteral("|") + QString::number(wa) + QStringLiteral("x") + QString::number(ha) + QStringLiteral("|") + QString::number(wb) + QStringLiteral("x") + QString::number(hb)).toUtf8();
        const QString fname = dd.filePath(hexSha1(key) + QStringLiteral("_stitched.png"));
        if (out.save(fname)) {
            LOG_DEBUG(QStringLiteral("addStitchedImagePair: wrote debug stitched image %1").arg(fname));
        } else {
            LOG_WARN(QStringLiteral("addStitchedImagePair: failed to write debug stitched image %1").arg(fname));
        }

        // Log each item in the scene to find any with abnormal bounds
        if (m_scene) {
            const auto items = m_scene->items();
            LOG_DEBUG(QStringLiteral("addStitchedImagePair: scene has %1 items").arg(items.size()));
            for (int ii = 0; ii < items.size(); ++ii) {
                QGraphicsItem *it = items.at(ii);
                if (!it) continue;
                QRectF br = it->boundingRect();
                QPointF pos = it->pos();
                QString typeName = QStringLiteral("unknown");
                if (qgraphicsitem_cast<QGraphicsPixmapItem*>(it)) typeName = QStringLiteral("pixmap");
                else if (qgraphicsitem_cast<QGraphicsEllipseItem*>(it)) typeName = QStringLiteral("ellipse");
                else if (qgraphicsitem_cast<QGraphicsLineItem*>(it)) typeName = QStringLiteral("line");
                LOG_DEBUG(QStringLiteral("addStitchedImagePair: item[%1] type=%2 pos=(%3,%4) bound=(%5,%6,%7,%8)").arg(ii).arg(typeName).arg(pos.x()).arg(pos.y()).arg(br.x()).arg(br.y()).arg(br.width()).arg(br.height()));
            }
        }
    } catch (...) {
        LOG_WARN(QStringLiteral("addStitchedImagePair: exception while writing debug stitched image or logging items"));
    }
    return true;
}

void LayerRenderer::addMatchLines(const QVector<QPointF> &ptsA, const QVector<QPointF> &ptsB, qreal bOffsetX)
{
    if (!m_scene) return;
    if (!m_matchOpts.showLines) return; // 如果不显示匹配线,直接返回
    
    clearMatchLayers();

    LOG_DEBUG(QStringLiteral("addMatchLines: ptsA=%1 ptsB=%2 bOffsetX=%3").arg(ptsA.size()).arg(ptsB.size()).arg(bOffsetX));

    int n = qMin(ptsA.size(), ptsB.size());
    
    // 限制显示数量
    if (m_matchOpts.maxDisplayCount > 0 && n > m_matchOpts.maxDisplayCount) {
        n = m_matchOpts.maxDisplayCount;
    }
    
    QPen linePen(m_matchOpts.lineColor);
    linePen.setWidthF(static_cast<qreal>(m_matchOpts.lineWidth));
    linePen.setColor(QColor(m_matchOpts.lineColor.red(), m_matchOpts.lineColor.green(), 
                            m_matchOpts.lineColor.blue(), m_matchOpts.opacity));
    
    QPen ptPen(m_matchOpts.lineColor);
    QBrush ptBrush(QColor(m_matchOpts.lineColor.red(), m_matchOpts.lineColor.green(), 
                          m_matchOpts.lineColor.blue(), m_matchOpts.opacity));

    for (int i = 0; i < n; ++i) {
        QPointF a = ptsA.at(i);
        QPointF b = ptsB.at(i);
        // point on A
        QGraphicsEllipseItem *ea = m_scene->addEllipse(a.x()-3, a.y()-3, 6, 6, ptPen, ptBrush);
        ea->setZValue(1001.0);
        m_matchItems.append(ea);
        // point on B (apply offset)
        QGraphicsEllipseItem *eb = m_scene->addEllipse(bOffsetX + b.x()-3, b.y()-3, 6, 6, ptPen, ptBrush);
        eb->setZValue(1001.0);
        m_matchItems.append(eb);
        // line
        QGraphicsLineItem *ln = m_scene->addLine(a.x(), a.y(), bOffsetX + b.x(), b.y(), linePen);
        ln->setZValue(1000.5);
        m_matchItems.append(ln);
    }
}

void LayerRenderer::setMatchDisplayOptions(const MatchDisplayOptions &opts)
{
    m_matchOpts = opts;
}

void LayerRenderer::clearMatchLayers()
{
    for (auto *it: std::as_const(m_matchItems)) {
        if (it && m_scene) {
            m_scene->removeItem(it);
            delete it;
        }
    }
    m_matchItems.clear();
}
