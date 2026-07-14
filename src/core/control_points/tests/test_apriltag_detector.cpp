#include "detection/AprilTagDetector.h"

#include <gtest/gtest.h>

#include <apriltag/apriltag.h>
#include <apriltag/common/image_u8.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tag36h10.h>
#include <apriltag/tag36h11.h>
#include <apriltag/tagCircle21h7.h>
#include <apriltag/tagStandard41h12.h>
#include <apriltag/tagStandard52h13.h>

#include <QImage>
#include <QLineF>
#include <QPainter>
#include <QTransform>

#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace xjw::control_points
{
namespace
{

struct FamilyHandle
{
    apriltag_family_t *family = nullptr;
    void (*destroy)(apriltag_family_t *) = nullptr;

    FamilyHandle(apriltag_family_t *value, void (*deleter)(apriltag_family_t *))
        : family(value), destroy(deleter)
    {
    }

    FamilyHandle(const FamilyHandle &) = delete;
    FamilyHandle &operator=(const FamilyHandle &) = delete;

    FamilyHandle(FamilyHandle &&other) noexcept
        : family(other.family), destroy(other.destroy)
    {
        other.family = nullptr;
    }

    ~FamilyHandle()
    {
        if (family != nullptr && destroy != nullptr)
        {
            destroy(family);
        }
    }
};

FamilyHandle createFamily(AprilTagFamily family)
{
    switch (family)
    {
    case AprilTagFamily::Tag16h5:
        return {tag16h5_create(), tag16h5_destroy};
    case AprilTagFamily::Tag25h9:
        return {tag25h9_create(), tag25h9_destroy};
    case AprilTagFamily::Tag36h10:
        return {tag36h10_create(), tag36h10_destroy};
    case AprilTagFamily::Tag36h11:
        return {tag36h11_create(), tag36h11_destroy};
    case AprilTagFamily::Circle21h7:
        return {tagCircle21h7_create(), tagCircle21h7_destroy};
    case AprilTagFamily::Standard41h12:
        return {tagStandard41h12_create(), tagStandard41h12_destroy};
    case AprilTagFamily::Standard52h13:
        return {tagStandard52h13_create(), tagStandard52h13_destroy};
    }
    throw std::invalid_argument("Unsupported AprilTag family");
}

QImage renderAprilTag(AprilTagFamily family, int id, int rotation)
{
    FamilyHandle family_handle = createFamily(family);
    std::unique_ptr<image_u8_t, decltype(&image_u8_destroy)> raw_tag(
        apriltag_to_image(family_handle.family, static_cast<uint32_t>(id)),
        image_u8_destroy);

    QImage source(static_cast<int>(raw_tag->width),
                  static_cast<int>(raw_tag->height),
                  QImage::Format_Grayscale8);
    for (int row = 0; row < source.height(); ++row)
    {
        std::memcpy(source.scanLine(row),
                    raw_tag->buf + static_cast<std::size_t>(row) * raw_tag->stride,
                    static_cast<std::size_t>(source.width()));
    }

    const QImage scaled = source.scaled(144, 144, Qt::IgnoreAspectRatio, Qt::FastTransformation);
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
