#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <filesystem>
#include <string>

TEST(CommonUnicodePathIOTest, ReadsAndWritesOpenCvImagesWithChinesePaths)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageDir = QDir(tempDir.path()).filePath(QStringLiteral("中文目录/影像输入"));
    ASSERT_TRUE(QDir().mkpath(imageDir));
    const QString imagePath = QDir(imageDir).filePath(QStringLiteral("火星影像.png"));

    cv::Mat image(3, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    ASSERT_TRUE(xjw::common::io::writeImage(imagePath, image));

    const cv::Mat loaded = xjw::common::io::readImage(imagePath, cv::IMREAD_COLOR);
    ASSERT_FALSE(loaded.empty());
    EXPECT_EQ(loaded.rows, image.rows);
    EXPECT_EQ(loaded.cols, image.cols);
    EXPECT_EQ(loaded.type(), image.type());
    EXPECT_EQ(loaded.at<cv::Vec3b>(1, 2), image.at<cv::Vec3b>(1, 2));
}

TEST(CommonUnicodePathIOTest, GenericImageApiAcceptsUtf8StdStringPaths)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageDir = QDir(tempDir.path()).filePath(QStringLiteral("中文目录/影像输入"));
    ASSERT_TRUE(QDir().mkpath(imageDir));
    const QString imagePath = QDir(imageDir).filePath(QStringLiteral("火星影像-std-string.png"));
    const std::string utf8Path = imagePath.toUtf8().toStdString();

    cv::Mat image(2, 5, CV_8UC3, cv::Scalar(3, 7, 11));
    ASSERT_TRUE(xjw::common::io::writeImage(utf8Path, image));

    const cv::Mat loaded = xjw::common::io::readImage(utf8Path, cv::IMREAD_COLOR);
    ASSERT_FALSE(loaded.empty());
    EXPECT_EQ(loaded.rows, image.rows);
    EXPECT_EQ(loaded.cols, image.cols);
    EXPECT_EQ(loaded.type(), image.type());
}

TEST(CommonUnicodePathIOTest, CreatesNativeFileStreamsFromQStringPaths)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString textDir = QDir(tempDir.path()).filePath(QStringLiteral("中文目录/参数"));
    ASSERT_TRUE(QDir().mkpath(textDir));
    const QString textPath = QDir(textDir).filePath(QStringLiteral("相机参数.txt"));

    {
        std::ofstream out = xjw::common::io::openOutputFile(textPath, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open());
        out << "unicode path ok";
    }

    std::ifstream in = xjw::common::io::openInputFile(textPath, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    std::string value;
    std::getline(in, value);
    EXPECT_EQ(value, "unicode path ok");
}

TEST(CommonUnicodePathIOTest, GenericStreamApiAcceptsUtf8StdStringPaths)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString textDir = QDir(tempDir.path()).filePath(QStringLiteral("中文目录/参数"));
    ASSERT_TRUE(QDir().mkpath(textDir));
    const QString textPath = QDir(textDir).filePath(QStringLiteral("相机参数-std-string.txt"));
    const std::string utf8Path = textPath.toUtf8().toStdString();

    {
        std::ofstream out = xjw::common::io::openOutputFile(utf8Path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open());
        out << "generic std::string path ok";
    }

    std::ifstream in = xjw::common::io::openInputFile(utf8Path, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    std::string value;
    std::getline(in, value);
    EXPECT_EQ(value, "generic std::string path ok");
}

TEST(CommonUnicodePathIOTest, ReadsAndWritesBinaryFilesAtomically)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString dataPath = QDir(tempDir.path()).filePath(QStringLiteral("中文目录/缓存/二进制.dat"));
    const QByteArray payload("abc\x00\x01\x02", 6);
    QString error;

    ASSERT_TRUE(xjw::common::io::writeFileBytesAtomic(dataPath, payload, &error)) << error.toStdString();
    const QByteArray loaded = xjw::common::io::readFileBytes(dataPath, &error);
    ASSERT_TRUE(error.isEmpty()) << error.toStdString();
    EXPECT_EQ(loaded, payload);
}

TEST(CommonUnicodePathIOTest, ConvertsFilesystemPathsWithoutLosingChineseCharacters)
{
    const QString original = QStringLiteral("E:/中文目录/相机参数.tsai");
    const std::filesystem::path nativePath = xjw::common::io::toFilesystemPath(original);

    EXPECT_EQ(xjw::common::io::fromFilesystemPath(nativePath), original);
    EXPECT_EQ(xjw::common::io::toUtf8Path(nativePath), original.toUtf8().toStdString());
}

TEST(CommonUnicodePathIOTest, ConvertsNativeNarrowPathForLegacyFileApis)
{
    const QString original = QStringLiteral("E:/中文目录/点云.ply");
    const std::string nativePath = xjw::common::io::toNativeNarrowPath(original);
    const QByteArray nativeBytes(nativePath.data(), static_cast<qsizetype>(nativePath.size()));
    const QString decodedPath = QFile::decodeName(nativeBytes);

    if (decodedPath.contains(QChar(0xfffd)))
    {
        GTEST_SKIP() << "Current Windows ANSI code page cannot represent this Unicode test path";
    }

#ifdef _WIN32
    EXPECT_EQ(decodedPath, QDir::toNativeSeparators(original));
#else
    EXPECT_EQ(decodedPath, original);
#endif
}
