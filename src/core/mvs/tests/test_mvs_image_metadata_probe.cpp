#include <array>
#include <fstream>
#include <string>

#include <gdal_priv.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QDir>
#include <QTemporaryDir>

#include "MvsImageMetadataProbe.h"

namespace
{

    TEST(MvsImageMetadataProbeTest, RequiredRasterDriversAreRegistered)
    {
        GDALAllRegister();
        constexpr std::array<const char*, 10> required_drivers{
            "GTiff", "HFA", "ISIS2", "ISIS3", "JPEG", "JP2OpenJPEG", "PDS", "PDS4", "PNG", "VRT"};

        for (const char* driver_name : required_drivers)
        {
            SCOPED_TRACE(driver_name);
            EXPECT_NE(GetGDALDriverManager()->GetDriverByName(driver_name), nullptr);
        }
    }

    TEST(MvsImageMetadataProbeTest, ReadsJpegHeaderWrittenByOpenCv)
    {
        const QString temporary_root = QStringLiteral(PLASCAN_MVS_METADATA_TEST_TMP);
        ASSERT_TRUE(QDir().mkpath(temporary_root));
        QTemporaryDir temporary_directory(QDir(temporary_root).filePath(QStringLiteral("jpeg-header-XXXXXX")));
        ASSERT_TRUE(temporary_directory.isValid());

        const QString image_path = temporary_directory.filePath(QStringLiteral("input.JPG"));
        const cv::Mat image(19, 31, CV_8UC3, cv::Scalar(32, 128, 224));
        ASSERT_TRUE(cv::imwrite(image_path.toStdString(), image));

        xjw::mvs::MvsImageMetadata metadata;
        std::string error_message;
        EXPECT_TRUE(xjw::mvs::probeMvsImageMetadata(
            QDir::toNativeSeparators(image_path).toStdString(), &metadata, &error_message))
            << error_message;
        EXPECT_EQ(metadata.width, image.cols);
        EXPECT_EQ(metadata.height, image.rows);
        EXPECT_TRUE(error_message.empty());
    }

    TEST(MvsImageMetadataProbeTest, ReadsPngHeaderWrittenByOpenCv)
    {
        const QString temporary_root = QStringLiteral(PLASCAN_MVS_METADATA_TEST_TMP);
        ASSERT_TRUE(QDir().mkpath(temporary_root));
        QTemporaryDir temporary_directory(QDir(temporary_root).filePath(QStringLiteral("png-header-XXXXXX")));
        ASSERT_TRUE(temporary_directory.isValid());

        const QString image_path = temporary_directory.filePath(QStringLiteral("prepared.png"));
        const cv::Mat image(23, 37, CV_8UC1, cv::Scalar(96));
        ASSERT_TRUE(cv::imwrite(image_path.toStdString(), image));

        xjw::mvs::MvsImageMetadata metadata;
        std::string error_message;
        EXPECT_TRUE(xjw::mvs::probeMvsImageMetadata(
            QDir::toNativeSeparators(image_path).toStdString(), &metadata, &error_message))
            << error_message;
        EXPECT_EQ(metadata.width, image.cols);
        EXPECT_EQ(metadata.height, image.rows);
        EXPECT_TRUE(error_message.empty());
    }

    TEST(MvsImageMetadataProbeTest, ReportsGdalDetailForUnsupportedRaster)
    {
        const QString temporary_root = QStringLiteral(PLASCAN_MVS_METADATA_TEST_TMP);
        ASSERT_TRUE(QDir().mkpath(temporary_root));
        QTemporaryDir temporary_directory(QDir(temporary_root).filePath(QStringLiteral("unsupported-header-XXXXXX")));
        ASSERT_TRUE(temporary_directory.isValid());

        const QString image_path = temporary_directory.filePath(QStringLiteral("not-an-image.jpg"));
        std::ofstream output(image_path.toStdString(), std::ios::binary);
        output << "not a raster";
        output.close();
        ASSERT_TRUE(output.good());

        xjw::mvs::MvsImageMetadata metadata;
        std::string error_message;
        EXPECT_FALSE(xjw::mvs::probeMvsImageMetadata(
            QDir::toNativeSeparators(image_path).toStdString(), &metadata, &error_message));
        EXPECT_NE(error_message.find("unable to open image header"), std::string::npos);
        EXPECT_NE(error_message.find("; GDAL: "), std::string::npos);
    }

} // namespace
