#include "detection/MarkerImageAdapter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>

namespace
{
    class RecordingDetector final : public xjw::control_points::MarkerDetector
    {
    public:
        QVector<xjw::control_points::MarkerDetection>
        detect(const cv::Mat& image,
               const cv::Mat& mask,
               const xjw::control_points::MarkerDetectionOptions&) const override
        {
            ++calls;
            grayscale = image.clone();
            exclusions = mask.clone();
            return {};
        }

        mutable int calls = 0;
        mutable cv::Mat grayscale;
        mutable cv::Mat exclusions;
    };
} // namespace

TEST(MarkerImageAdapterTest, RetainsQtGrayscaleAndMaskPixelSemanticsAcrossFormats)
{
    const QImage::Format formats[] = {
        QImage::Format_RGB888, QImage::Format_ARGB32, QImage::Format_Grayscale8, QImage::Format_Indexed8};
    for (const auto format : formats)
    {
        QImage image(13, 7, format);
        if (format == QImage::Format_Indexed8)
        {
            image.setColorTable({qRgb(0, 0, 0), qRgb(255, 0, 0), qRgb(0, 255, 0), qRgb(0, 0, 255)});
        }
        for (int y = 0; y < image.height(); ++y)
        {
            for (int x = 0; x < image.width(); ++x)
            {
                if (format == QImage::Format_Indexed8)
                {
                    image.setPixel(x, y, static_cast<uint>((x + y) % 4));
                }
                else
                {
                    image.setPixelColor(x, y, QColor((x * 19) % 256, (y * 37) % 256, (x + y) * 11));
                }
            }
        }
        const QImage before = image.copy();
        const QImage expected = image.convertToFormat(QImage::Format_Grayscale8);
        RecordingDetector detector;
        xjw::app::markers::detectMarkers(detector, image, image, {});
        ASSERT_EQ(detector.calls, 1);
        ASSERT_EQ(detector.grayscale.type(), CV_8UC1);
        for (int y = 0; y < image.height(); ++y)
        {
            for (int x = 0; x < image.width(); ++x)
            {
                EXPECT_EQ(detector.grayscale.at<uchar>(y, x), expected.constScanLine(y)[x]);
                EXPECT_EQ(detector.exclusions.at<uchar>(y, x), qGray(image.pixel(x, y)));
            }
        }
        EXPECT_EQ(image, before);
    }
}

TEST(MarkerImageAdapterTest, RejectsMismatchedMaskAndBypassesEmptyOrCancelledInput)
{
    RecordingDetector detector;
    const QImage image(13, 7, QImage::Format_Grayscale8);
    EXPECT_THROW(xjw::app::markers::detectMarkers(detector, image, QImage(12, 7, QImage::Format_Grayscale8), {}),
                 std::invalid_argument);
    EXPECT_TRUE(xjw::app::markers::detectMarkers(detector, {}, {}, {}).isEmpty());
    std::atomic_bool cancelled = true;
    xjw::control_points::MarkerDetectionOptions options;
    options.cancelRequested = &cancelled;
    EXPECT_TRUE(xjw::app::markers::detectMarkers(detector, image, {}, options).isEmpty());
    EXPECT_EQ(detector.calls, 0);
}
