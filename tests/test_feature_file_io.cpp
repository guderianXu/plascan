#include <gtest/gtest.h>

#include "compat/QtTorchMacroGuard.h"
#include "FeatureData.h"
#include "FeatureOutput.h"
#include "FeatureFileIO.h"

#include <QDataStream>
#include <QFile>
#include <QTemporaryDir>

TEST(FeatureFileIOTest, SiftRoundTripPreservesScaleAndOrientation)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    FeatureOutput output;
    cv::KeyPoint first;
    first.pt = cv::Point2f(12.5f, 34.25f);
    first.response = 0.75f;
    first.size = 4.5f;
    first.angle = 123.0f;

    cv::KeyPoint second;
    second.pt = cv::Point2f(56.0f, 78.0f);
    second.response = 0.50f;
    second.size = 9.0f;
    second.angle = 271.5f;

    output.keypoints = {first, second};
    output.scores = {first.response, second.response};
    output.descriptors = torch::tensor({{0.1f, 0.2f}, {0.3f, 0.4f}}, torch::kFloat32);
    output.imageWidth = 640;
    output.imageHeight = 480;

    const QString path = tempDir.path() + QStringLiteral("/features.sift");
    ASSERT_TRUE(FeatureFileIO::write(path, QStringLiteral("image.jpg"), output, "sift"));

    QString imageName;
    FeatureOutput loaded;
    std::string algorithmName;
    ASSERT_TRUE(FeatureFileIO::read(path, imageName, loaded, &algorithmName));

    EXPECT_EQ(imageName, QStringLiteral("image.jpg"));
    EXPECT_EQ(algorithmName, "sift");
    EXPECT_EQ(loaded.imageWidth, 640);
    EXPECT_EQ(loaded.imageHeight, 480);
    ASSERT_EQ(loaded.keypoints.size(), 2u);
    EXPECT_FLOAT_EQ(loaded.keypoints[0].size, 4.5f);
    EXPECT_FLOAT_EQ(loaded.keypoints[0].angle, 123.0f);
    EXPECT_FLOAT_EQ(loaded.keypoints[1].size, 9.0f);
    EXPECT_FLOAT_EQ(loaded.keypoints[1].angle, 271.5f);
    ASSERT_TRUE(loaded.descriptors.defined());
    ASSERT_EQ(loaded.descriptors.size(0), 2);
    ASSERT_EQ(loaded.descriptors.size(1), 2);
    EXPECT_FLOAT_EQ(loaded.descriptors[1][0].item<float>(), 0.3f);

    xjw::feature_extractors::FeatureData data;
    ASSERT_TRUE(FeatureFileIO::readData(path, imageName, data));
    EXPECT_EQ(data.sourceAlgorithm, "sift");
    EXPECT_EQ(data.imageWidth, 640);
    EXPECT_EQ(data.imageHeight, 480);
}

TEST(FeatureFileIOTest, VersionOneFilesRemainReadable)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString path = tempDir.path() + QStringLiteral("/legacy.sift");

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    QDataStream out(&file);
    out.setByteOrder(QDataStream::LittleEndian);
    out.setFloatingPointPrecision(QDataStream::SinglePrecision);
    out.writeRawData("SFTB", 4);
    out << quint32(1);
    const QByteArray nameBytes("legacy.jpg");
    out << quint32(nameBytes.size());
    out.writeRawData(nameBytes.constData(), nameBytes.size());
    out << quint32(1);
    out << float(10.0f) << float(20.0f) << float(0.8f);
    out << quint32(2);
    out << float(0.25f) << float(0.75f);
    file.close();

    QString imageName;
    FeatureOutput loaded;
    ASSERT_TRUE(FeatureFileIO::read(path, imageName, loaded));
    EXPECT_EQ(imageName, QStringLiteral("legacy.jpg"));
    ASSERT_EQ(loaded.keypoints.size(), 1u);
    EXPECT_FLOAT_EQ(loaded.keypoints[0].pt.x, 10.0f);
    EXPECT_FLOAT_EQ(loaded.keypoints[0].pt.y, 20.0f);
    EXPECT_FLOAT_EQ(loaded.keypoints[0].response, 0.8f);
    EXPECT_FLOAT_EQ(loaded.keypoints[0].size, 8.0f);
    EXPECT_FLOAT_EQ(loaded.keypoints[0].angle, -1.0f);
}

TEST(FeatureFileIOTest, DedodeRoundTripUsesDedicatedMagic)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    FeatureOutput output;
    cv::KeyPoint keypoint;
    keypoint.pt = cv::Point2f(12.0f, 24.0f);
    keypoint.response = 0.9f;
    keypoint.size = 8.0f;
    output.keypoints = {keypoint};
    output.scores = {0.9f};
    output.descriptors = torch::tensor({{0.1f, 0.2f, 0.3f}}, torch::kFloat32);
    output.imageWidth = 320;
    output.imageHeight = 240;

    const QString path = tempDir.path() + QStringLiteral("/features.dedode");
    ASSERT_TRUE(FeatureFileIO::write(path, QStringLiteral("image.jpg"), output, "dedode"));
    EXPECT_EQ(FeatureFileIO::peekAlgorithm(path), "dedode");
    EXPECT_EQ(FeatureFileIO::peekCount(path), 1);

    QString imageName;
    FeatureOutput loaded;
    std::string algorithmName;
    ASSERT_TRUE(FeatureFileIO::read(path, imageName, loaded, &algorithmName));
    EXPECT_EQ(algorithmName, "dedode");
    EXPECT_EQ(loaded.imageWidth, 320);
    EXPECT_EQ(loaded.imageHeight, 240);
}

TEST(FeatureFileIOTest, UnknownMagicIsRejected)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString path = tempDir.path() + QStringLiteral("/bad.sp");

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    file.write("BADC", 4);
    file.close();

    EXPECT_TRUE(FeatureFileIO::peekAlgorithm(path).empty());
    EXPECT_EQ(FeatureFileIO::peekCount(path), -1);

    QString imageName;
    FeatureOutput loaded;
    EXPECT_FALSE(FeatureFileIO::read(path, imageName, loaded));
}
