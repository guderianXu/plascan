#include "detection/NonCodedTargetDetector.h"

#include <gtest/gtest.h>

#include <QImage>
#include <QLineF>
#include <QPainter>

#include <atomic>

namespace xjw::control_points
{
namespace
{

QImage circleTarget(const QPointF &center, double radius)
{
    QImage image(240, 200, QImage::Format_Grayscale8);
    image.fill(Qt::white);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::black);
    painter.drawEllipse(center + QPointF(0.5, 0.5), radius, radius);
    return image;
}

QImage fourQuadrantTarget(const QPointF &center, double radius)
{
    QImage image(240, 200, QImage::Format_Grayscale8);
    image.fill(Qt::white);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::black);
    const QPointF painter_center = center + QPointF(0.5, 0.5);
    const QRectF bounds(painter_center.x() - radius,
                        painter_center.y() - radius,
                        2.0 * radius,
                        2.0 * radius);
    painter.drawPie(bounds, 0, 90 * 16);
    painter.drawPie(bounds, 180 * 16, 90 * 16);
    return image;
}

TEST(NonCodedTargetDetectorTest, LocatesCircularTargetAtSubpixelCenter)
{
    const QPointF expected(117.25, 91.75);
    const auto detections = NonCodedTargetDetector(NonCodedTargetType::Circle)
                                .detect(circleTarget(expected, 22.0), {}, {});

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections.front().targetId, -1);
    EXPECT_EQ(detections.front().family, MarkerTargetFamily::NonCodedCircle);
    EXPECT_LT(QLineF(detections.front().center, expected).length(), 0.35)
        << detections.front().center.x() << "," << detections.front().center.y();
    EXPECT_GT(detections.front().confidence, 0.8);
}

TEST(NonCodedTargetDetectorTest, LocatesFourQuadrantTargetWithoutAssigningIdentity)
{
    const QPointF expected(122.0, 96.0);
    const auto detections = NonCodedTargetDetector(NonCodedTargetType::FourQuadrant)
                                .detect(fourQuadrantTarget(expected, 26.0), {}, {});

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections.front().targetId, -1);
    EXPECT_EQ(detections.front().family, MarkerTargetFamily::NonCodedFourQuadrant);
    EXPECT_LT(QLineF(detections.front().center, expected).length(), 0.5)
        << detections.front().center.x() << "," << detections.front().center.y();
}

TEST(NonCodedTargetDetectorTest, AppliesMaskAndCancellation)
{
    const QImage image = circleTarget(QPointF(120.0, 100.0), 20.0);
    QImage mask(image.size(), QImage::Format_Grayscale8);
    mask.fill(0);
    mask.setPixelColor(QPoint(120, 100), Qt::white);

    const NonCodedTargetDetector detector(NonCodedTargetType::Circle);
    EXPECT_TRUE(detector.detect(image, mask, {}).isEmpty());

    std::atomic_bool cancelled = true;
    MarkerDetectionOptions options;
    options.cancelRequested = &cancelled;
    EXPECT_TRUE(detector.detect(image, {}, options).isEmpty());
}

} // namespace
} // namespace xjw::control_points
