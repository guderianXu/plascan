#pragma once

#include <QPixmap>
#include <QRectF>
#include <QWidget>

class QPainter;

class HenuBrandWidget : public QWidget
{
public:
    explicit HenuBrandWidget(QWidget *parent = nullptr);

    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    void drawHenuEmblem(QPainter &painter, const QRectF &rect) const;

    QPixmap _emblemPixmap;
};
