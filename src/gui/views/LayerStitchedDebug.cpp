#include "LayerStitchedDebug.h"

#include "Logger.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsLineItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QImage>
#include <QPainter>
#include <QPixmap>

#include <algorithm>

namespace
{

QString hexSha1(const QByteArray &data)
{
    return QString::fromLatin1(QCryptographicHash::hash(data, QCryptographicHash::Sha1).toHex());
}

QString debugOutputDirectory(const QString &plascanPath, const QString &fallbackImagePath)
{
    if (!plascanPath.trimmed().isEmpty())
    {
        QDir projectRoot(QFileInfo(plascanPath).absolutePath());
        return projectRoot.filePath(QStringLiteral(".plascan_tmp/debug"));
    }
    return QFileInfo(fallbackImagePath).absolutePath() + QDir::separator() + QStringLiteral("plascan_debug");
}

QString itemTypeName(QGraphicsItem *item)
{
    if (qgraphicsitem_cast<QGraphicsPixmapItem *>(item))
    {
        return QStringLiteral("pixmap");
    }
    if (qgraphicsitem_cast<QGraphicsEllipseItem *>(item))
    {
        return QStringLiteral("ellipse");
    }
    if (qgraphicsitem_cast<QGraphicsLineItem *>(item))
    {
        return QStringLiteral("line");
    }
    return QStringLiteral("unknown");
}

} // namespace

namespace xjw::gui::views
{

void recordStitchedImagePairDebug(QGraphicsScene *scene,
                                  const QString &plascanPath,
                                  const QString &pathA,
                                  const QString &pathB,
                                  const QGraphicsPixmapItem *itemA,
                                  const QGraphicsPixmapItem *itemB,
                                  int gap)
{
    if (!itemA || !itemB)
    {
        return;
    }

    try
    {
        const int wa = itemA->pixmap().width();
        const int ha = itemA->pixmap().height();
        const int wb = itemB->pixmap().width();
        const int hb = itemB->pixmap().height();
        const int h = std::max(ha, hb);
        const int w = wa + gap + wb;

        QImage out(w, h, QImage::Format_ARGB32);
        out.fill(Qt::transparent);
        QPainter painter(&out);
        painter.drawPixmap(0, 0, itemA->pixmap());
        painter.drawPixmap(wa + gap, 0, itemB->pixmap());
        painter.end();

        QDir debugDir(debugOutputDirectory(plascanPath, pathA));
        debugDir.mkpath(QStringLiteral("."));

        const QByteArray key = (pathA + QStringLiteral("|") + pathB +
                                QStringLiteral("|") + QString::number(wa) +
                                QStringLiteral("x") + QString::number(ha) +
                                QStringLiteral("|") + QString::number(wb) +
                                QStringLiteral("x") + QString::number(hb)).toUtf8();
        const QString filePath = debugDir.filePath(hexSha1(key) + QStringLiteral("_stitched.png"));
        if (out.save(filePath))
        {
            LOG_DEBUG(QStringLiteral("addStitchedImagePair: wrote debug stitched image %1").arg(filePath));
        }
        else
        {
            LOG_WARN(QStringLiteral("addStitchedImagePair: failed to write debug stitched image %1").arg(filePath));
        }

        if (!scene)
        {
            return;
        }

        const auto items = scene->items();
        LOG_DEBUG(QStringLiteral("addStitchedImagePair: scene has %1 items").arg(items.size()));
        for (int ii = 0; ii < items.size(); ++ii)
        {
            QGraphicsItem *item = items.at(ii);
            if (!item)
            {
                continue;
            }
            const QRectF bounds = item->boundingRect();
            const QPointF pos = item->pos();
            LOG_DEBUG(QStringLiteral("addStitchedImagePair: item[%1] type=%2 pos=(%3,%4) bound=(%5,%6,%7,%8)")
                          .arg(ii)
                          .arg(itemTypeName(item))
                          .arg(pos.x())
                          .arg(pos.y())
                          .arg(bounds.x())
                          .arg(bounds.y())
                          .arg(bounds.width())
                          .arg(bounds.height()));
        }
    }
    catch (...)
    {
        LOG_WARN(QStringLiteral("addStitchedImagePair: exception while writing debug stitched image or logging items"));
    }
}

} // namespace xjw::gui::views
