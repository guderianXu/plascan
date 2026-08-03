#include "detection/AprilTagDetector.h"
#include "print/MarkerSheetRenderer.h"

#include <gtest/gtest.h>

#include <QImage>
#include <QLineF>
#include <QPainter>
#include <QTransform>

#include <atomic>

namespace xjw::control_points
{
namespace
{

QImage renderAprilTag(AprilTagFamily family, int id, int rotation)
{
    const QImage scaled = MarkerSheetRenderer::renderMarkerImage(
        markerTargetFamily(family), id, 144);
    QImage canvas(192, 192, QImage::Format_Grayscale8);
    canvas.fill(Qt::white);
    QPainter painter(&canvas);
    painter.drawImage(QPoint(24, 24), scaled);
    painter.end();
    return canvas.transformed(QTransform().rotate(rotation), Qt::FastTransformation);
}

TEST(AprilTagDetectorTest, DecodesSupportedFamiliesAcrossRotations)
{
    for (const AprilTagFamily family : supportedAprilTagFamilies())
    {
        for (const int rotation : {0, 90, 180, 270})
        {
            const QImage image = renderAprilTag(family, 7, rotation);
            const QVector<MarkerDetection> detections = AprilTagDetector(family).detect(image, {}, {});

            ASSERT_EQ(detections.size(), 1) << static_cast<int>(family) << " rotation=" << rotation;
            EXPECT_EQ(detections.front().targetId, 7);
            EXPECT_EQ(detections.front().family, markerTargetFamily(family));
            EXPECT_LT(QLineF(detections.front().center, QPointF(95.5, 95.5)).length(), 0.75);
            EXPECT_GT(detections.front().decisionMargin, 0.0);
            EXPECT_EQ(detections.front().corners.size(), 4);
        }
    }
}

TEST(AprilTagDetectorTest, RejectsDetectionWhenMaskExcludesCenterOrCorner)
{
    const QImage image = renderAprilTag(AprilTagFamily::Tag36h11, 7, 0);
    QImage mask(image.size(), QImage::Format_Grayscale8);
    mask.fill(0);

    EXPECT_EQ(AprilTagDetector(AprilTagFamily::Tag36h11).detect(image, mask, {}).size(), 1);

    mask.setPixelColor(QPoint(96, 96), Qt::white);
    EXPECT_TRUE(AprilTagDetector(AprilTagFamily::Tag36h11).detect(image, mask, {}).isEmpty());
}

TEST(AprilTagDetectorTest, HonorsPreexistingCancellation)
{
    std::atomic_bool cancelled = true;
    MarkerDetectionOptions options;
    options.cancelRequested = &cancelled;

    const QImage image = renderAprilTag(AprilTagFamily::Tag36h11, 7, 0);
    EXPECT_TRUE(AprilTagDetector(AprilTagFamily::Tag36h11).detect(image, {}, options).isEmpty());
}

} // namespace
} // namespace xjw::control_points
