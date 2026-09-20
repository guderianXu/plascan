#include "io/ImageIO.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <gtest/gtest.h>

#include <cstdint>

namespace
{

using xjw::common::io::readImage;
using xjw::common::io::writeImage;

TEST(ImageIOTest, RoundTripsUnicodePngPath)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString path =
        QDir(temporary.path()).filePath(QStringLiteral("月球影像.png"));
    cv::Mat source(8, 12, CV_8UC3, cv::Scalar(10, 20, 30));
    ASSERT_TRUE(writeImage(path, source));

    QString error;
    const cv::Mat restored =
        readImage(path, cv::IMREAD_UNCHANGED, &error);
    ASSERT_FALSE(restored.empty()) << qPrintable(error);
    EXPECT_EQ(restored.type(), source.type());
    EXPECT_EQ(restored.size(), source.size());
    EXPECT_EQ(cv::norm(restored, source, cv::NORM_INF), 0.0);
}

TEST(ImageIOTest, PreservesSixteenBitTiffDepth)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString path =
        QDir(temporary.path()).filePath(QStringLiteral("高程.tiff"));
    cv::Mat source(7, 9, CV_16UC1, cv::Scalar(4096));
    ASSERT_TRUE(writeImage(path, source));

    QString error;
    const cv::Mat restored =
        readImage(path, cv::IMREAD_UNCHANGED, &error);
    ASSERT_FALSE(restored.empty()) << qPrintable(error);
    EXPECT_EQ(restored.type(), CV_16UC1);
    EXPECT_EQ(restored.size(), source.size());
    EXPECT_EQ(cv::norm(restored, source, cv::NORM_INF), 0.0);
}

TEST(ImageIOTest, ScalesSixteenBitTiffWhenEightBitGrayIsRequested)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString path =
        QDir(temporary.path()).filePath(QStringLiteral("十六位纹理.tiff"));
    cv::Mat source(1, 4, CV_16UC1);
    source.at<std::uint16_t>(0, 0) = 0;
    source.at<std::uint16_t>(0, 1) = 256;
    source.at<std::uint16_t>(0, 2) = 32768;
    source.at<std::uint16_t>(0, 3) = 65535;
    ASSERT_TRUE(writeImage(path, source));

    QString error;
    const cv::Mat restored = readImage(path, cv::IMREAD_GRAYSCALE, &error);
    ASSERT_FALSE(restored.empty()) << qPrintable(error);
    ASSERT_EQ(restored.type(), CV_8UC1);
    EXPECT_EQ(restored.at<std::uint8_t>(0, 0), 0);
    EXPECT_EQ(restored.at<std::uint8_t>(0, 1), 1);
    EXPECT_EQ(restored.at<std::uint8_t>(0, 2), 128);
    EXPECT_EQ(restored.at<std::uint8_t>(0, 3), 255);
}

TEST(ImageIOTest, ExpandsSingleBandTiffWhenColorIsRequested)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const QString path =
        QDir(temporary.path()).filePath(QStringLiteral("单波段影像.tif"));
    cv::Mat source(7, 9, CV_8UC1, cv::Scalar(137));
    ASSERT_TRUE(writeImage(path, source));

    QString error;
    const cv::Mat restored = readImage(path, cv::IMREAD_COLOR, &error);
    ASSERT_FALSE(restored.empty()) << qPrintable(error);
    ASSERT_EQ(restored.type(), CV_8UC3);
    EXPECT_EQ(restored.size(), source.size());
    EXPECT_EQ(restored.at<cv::Vec3b>(3, 4), cv::Vec3b(137, 137, 137));
}

TEST(ImageIOTest, ReportsMissingImagePath)
{
    QString error;
    const cv::Mat image = readImage(
        QStringLiteral("missing-image.tif"),
        cv::IMREAD_UNCHANGED,
        &error);
    EXPECT_TRUE(image.empty());
    EXPECT_TRUE(error.contains(QStringLiteral("missing-image.tif")));
}

TEST(ImageIOTest, ExpandsSingleBandTiffWithCombinedColorAndOrientationFlags)
{
    const QString root = QString::fromUtf8(PLASCAN_IMAGE_IO_TEST_TMP_DIR);
    ASSERT_TRUE(QDir().mkpath(root));
    QTemporaryDir temporary(QDir(root).filePath(QStringLiteral("run-XXXXXX")));
    ASSERT_TRUE(temporary.isValid());
    const QString path = temporary.filePath(QStringLiteral("单波段.tif"));
    ASSERT_TRUE(writeImage(path, cv::Mat(7, 9, CV_8UC1, cv::Scalar(137))));
    QString error;
    const cv::Mat restored = readImage(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION, &error);
    ASSERT_EQ(restored.type(), CV_8UC3) << qPrintable(error);
    EXPECT_EQ(restored.at<cv::Vec3b>(3, 4), cv::Vec3b(137, 137, 137));
}

TEST(ImageIOTest, CanIgnoreExifRotationForOriginalPixelCoordinates)
{
    const QString root = QString::fromUtf8(PLASCAN_IMAGE_IO_TEST_TMP_DIR);
    ASSERT_TRUE(QDir().mkpath(root));
    QTemporaryDir temporary(QDir(root).filePath(QStringLiteral("run-XXXXXX")));
    ASSERT_TRUE(temporary.isValid());
    std::vector<uchar> encoded;
    ASSERT_TRUE(cv::imencode(".jpg", cv::Mat(24, 32, CV_8UC3, cv::Scalar(20, 40, 80)), encoded));
    QByteArray jpeg(reinterpret_cast<const char*>(encoded.data()), static_cast<qsizetype>(encoded.size()));
    // APP1: Exif, little-endian TIFF, one SHORT orientation entry with value 6 (90 degrees).
    const QByteArray exif =
        QByteArray::fromHex("ffe1002245786966000049492a0008000000010012010300010000000600000000000000");
    jpeg.insert(2, exif);
    const QString path = temporary.filePath(QStringLiteral("rotated.jpg"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_EQ(file.write(jpeg), jpeg.size());
    file.close();
    EXPECT_EQ(xjw::common::io::readImageSize(path), QSize(32, 24));
    QString error;
    const cv::Mat original = readImage(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION, &error);
    ASSERT_FALSE(original.empty()) << qPrintable(error);
    EXPECT_EQ(original.size(), cv::Size(32, 24));
    const cv::Mat rotated = readImage(path, cv::IMREAD_COLOR, &error);
    ASSERT_FALSE(rotated.empty()) << qPrintable(error);
    EXPECT_EQ(rotated.size(), cv::Size(24, 32));
}

} // namespace
