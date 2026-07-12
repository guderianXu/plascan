#include "HenuBrandWidget.h"

#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QPaintEvent>
#include <QPainterPath>
#include <QSizePolicy>

HenuBrandWidget::HenuBrandWidget(QWidget *parent)
    : QWidget(parent)
    , _emblemPixmap(QStringLiteral(":/icons/henu_logo.png"))
{
    setObjectName(QStringLiteral("henuBrandWidget"));
    setToolTip(QStringLiteral("河南大学 / Henan University"));
    setMinimumSize(minimumSizeHint());
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

QSize HenuBrandWidget::minimumSizeHint() const
{
    return QSize(230, 40);
}

QSize HenuBrandWidget::sizeHint() const
{
    return QSize(250, 40);
}

void HenuBrandWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::TextAntialiasing, true);

    const QRectF contentRect = rect().adjusted(0.5, 1.5, -0.5, -1.5);
    painter.setPen(QPen(QColor(198, 212, 229), 1.0));
    painter.setBrush(QColor(255, 255, 255));
    painter.drawRoundedRect(contentRect, 4.0, 4.0);

    const QRectF emblemRect(contentRect.left() + 6.0,
                            contentRect.top() + 3.0,
                            contentRect.height() - 6.0,
                            contentRect.height() - 6.0);
    if (!_emblemPixmap.isNull())
    {
        painter.drawPixmap(emblemRect.toRect(), _emblemPixmap);
    }
    else
    {
        drawHenuEmblem(painter, emblemRect);
    }

    const QRectF textRect(emblemRect.right() + 7.0,
                          contentRect.top() + 3.0,
                          contentRect.right() - emblemRect.right() - 12.0,
                          contentRect.height() - 6.0);

    QFont titleFont = font();
    titleFont.setBold(true);
    titleFont.setPointSize(qMax(9, titleFont.pointSize()));
    painter.setFont(titleFont);
    painter.setPen(QColor(20, 46, 89));
    painter.drawText(QRectF(textRect.left(), textRect.top(), textRect.width(), 14.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     QStringLiteral("河南大学"));

    QFont subFont = font();
    subFont.setPointSize(qMax(7, subFont.pointSize() - 2));
    painter.setFont(subFont);
    painter.setPen(QColor(79, 91, 112));
    const QString subtitle = fontMetrics().elidedText(QStringLiteral("HENU · PlaScan 三维重建"),
                                                      Qt::ElideRight,
                                                      qMax(10, int(textRect.width())));
    painter.drawText(QRectF(textRect.left(), textRect.top() + 14.0, textRect.width(), 13.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     subtitle);
}

void HenuBrandWidget::drawHenuEmblem(QPainter &painter, const QRectF &rect) const
{
    painter.save();

    const QColor henuBlue(23, 71, 140);
    const QColor henuGreen(24, 138, 101);
    const QColor henuGold(226, 177, 74);

    painter.setPen(QPen(henuBlue, 1.8));
    painter.setBrush(QColor(244, 249, 255));
    painter.drawEllipse(rect);

    const QRectF inner = rect.adjusted(4.2, 4.2, -4.2, -4.2);
    painter.setPen(QPen(henuGreen, 1.1));
    painter.setBrush(Qt::NoBrush);
    painter.drawEllipse(inner);

    QPainterPath river;
    river.moveTo(inner.left() + 4.5, inner.center().y() + 4.0);
    river.cubicTo(inner.center().x() - 2.0,
                  inner.top() + 2.0,
                  inner.center().x() + 5.0,
                  inner.bottom() - 1.5,
                  inner.right() - 3.8,
                  inner.center().y() - 3.0);
    painter.setPen(QPen(henuGold, 1.5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawPath(river);

    QFont emblemFont = font();
    emblemFont.setBold(true);
    emblemFont.setPointSize(8);
    painter.setFont(emblemFont);
    painter.setPen(henuBlue);
    painter.drawText(inner.adjusted(0.0, -2.0, 0.0, -2.0),
                     Qt::AlignCenter,
                     QStringLiteral("河大"));

    QFont yearFont = font();
    yearFont.setBold(true);
    yearFont.setPointSize(5);
    painter.setFont(yearFont);
    painter.setPen(henuGreen);
    painter.drawText(QRectF(inner.left(), inner.bottom() - 7.0, inner.width(), 7.0),
                     Qt::AlignCenter,
                     QStringLiteral("1912"));

    painter.restore();
}
