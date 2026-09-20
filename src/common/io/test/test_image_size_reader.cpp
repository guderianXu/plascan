#include "io/ImageIO.h"
#include "io/PathIO.h"

#include <gdal_priv.h>
#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QtEndian>

#include <limits>

namespace
{

    class ImageSizeReaderTest : public testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_TRUE(QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR)));
        }
    };

    TEST_F(ImageSizeReaderTest, ReadsCommonRasterHeadersFromUnicodePaths)
    {
        QTemporaryDir temporary(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/size-XXXXXX"));
        ASSERT_TRUE(temporary.isValid());
        for (const QString& extension :
             {QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("bmp"), QStringLiteral("tif")})
        {
            const QString path = temporary.filePath(QStringLiteral("月球.") + extension);
            ASSERT_TRUE(xjw::common::io::writeImage(path, cv::Mat(24, 32, CV_8UC3, cv::Scalar(20, 40, 60))));
            QString error = QStringLiteral("old error");
            EXPECT_EQ(xjw::common::io::readImageSize(path, &error), QSize(32, 24)) << qPrintable(extension);
            EXPECT_TRUE(error.isEmpty()) << qPrintable(error);
        }
    }

    TEST_F(ImageSizeReaderTest, ReadsHugeSparseTiffWithoutAllocatingPixelBuffer)
    {
        QTemporaryDir temporary(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/size-XXXXXX"));
        ASSERT_TRUE(temporary.isValid());
        const QString path = temporary.filePath(QStringLiteral("huge.tif"));
        xjw::common::io::ensureGdalRegistered();
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        ASSERT_NE(driver, nullptr);
        char sparse[] = "SPARSE_OK=YES";
        char tiled[] = "TILED=YES";
        char* options[]{sparse, tiled, nullptr};
        GDALDataset* dataset =
            driver->Create(xjw::common::io::toUtf8Path(path).c_str(), 100000, 80000, 1, GDT_Byte, options);
        ASSERT_NE(dataset, nullptr);
        GDALClose(dataset);
        QString error;
        EXPECT_EQ(xjw::common::io::readImageSize(path, &error), QSize(100000, 80000)) << qPrintable(error);
        EXPECT_LT(QFileInfo(path).size(), 10 * 1024 * 1024);
    }

    TEST_F(ImageSizeReaderTest, ReportsEmptyMissingAndCorruptPaths)
    {
        QTemporaryDir temporary(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/size-XXXXXX"));
        ASSERT_TRUE(temporary.isValid());
        const QString corrupt = temporary.filePath(QStringLiteral("broken.png"));
        QFile file(corrupt);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_EQ(file.write("invalid image"), 13);
        file.close();
        for (const QString& path : {QString(), temporary.filePath(QStringLiteral("missing.tif")), corrupt})
        {
            QString error;
            EXPECT_FALSE(xjw::common::io::readImageSize(path, &error).isValid());
            EXPECT_FALSE(error.isEmpty());
            EXPECT_TRUE(error.contains(path));
        }
    }

    TEST_F(ImageSizeReaderTest, ReadsTopDownBmpAndRejectsInvalidDimensions)
    {
        QTemporaryDir temporary(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/size-XXXXXX"));
        ASSERT_TRUE(temporary.isValid());
        const QString path = temporary.filePath(QStringLiteral("top-down.bmp"));
        ASSERT_TRUE(xjw::common::io::writeImage(path, cv::Mat(24, 32, CV_8UC3, cv::Scalar(20, 40, 60))));
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::ReadWrite));
        QByteArray bytes = file.readAll();
        qToLittleEndian<qint32>(-24, bytes.data() + 22);
        ASSERT_TRUE(file.seek(0));
        ASSERT_EQ(file.write(bytes), bytes.size());
        file.close();
        EXPECT_EQ(xjw::common::io::readImageSize(path), QSize(32, 24));
        qToLittleEndian<qint32>(std::numeric_limits<qint32>::min(), bytes.data() + 22);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
        ASSERT_EQ(file.write(bytes), bytes.size());
        file.close();
        EXPECT_FALSE(xjw::common::io::readImageSize(path).isValid());
    }

} // namespace
