#include "io/ImageIO.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QDir>
#include <QTemporaryDir>

#include <gtest/gtest.h>

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

} // namespace
