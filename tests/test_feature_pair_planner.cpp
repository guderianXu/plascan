#include "FeaturePairPlanner.h"

#include <gtest/gtest.h>

#include <QStringList>

#include <array>
#include <vector>

using xjw::gui::FeaturePairPlannerOptions;
using xjw::gui::planFeatureMatchPairs;
using xjw::gui::planFeatureMatchPairPaths;

namespace
{

QStringList numberedImages(int count)
{
    QStringList images;
    for (int i = 0; i < count; ++i)
    {
        images.append(QStringLiteral("image_%1").arg(i, 3, 10, QLatin1Char('0')));
    }
    return images;
}

QStringList numberedImagePaths(int count)
{
    QStringList images;
    for (int i = 0; i < count; ++i)
    {
        images.append(QStringLiteral("E:/dataset/images/image_%1.JPG")
                          .arg(i, 3, 10, QLatin1Char('0')));
    }
    return images;
}

} // namespace

TEST(FeaturePairPlannerTest, SmallImageSetUsesExhaustivePairs)
{
    const QStringList images = {QStringLiteral("a"), QStringLiteral("b"), QStringLiteral("c"), QStringLiteral("d")};

    const QStringList pairs = planFeatureMatchPairs(images);

    EXPECT_EQ(pairs.size(), 6);
    EXPECT_TRUE(pairs.contains(QStringLiteral("a__b")));
    EXPECT_TRUE(pairs.contains(QStringLiteral("a__d")));
    EXPECT_TRUE(pairs.contains(QStringLiteral("c__d")));
}

TEST(FeaturePairPlannerTest, LargeImageSetUsesSequentialWindow)
{
    FeaturePairPlannerOptions options;
    options.exhaustiveMaxImages = 10;
    options.sequentialWindow = 4;

    const QStringList pairs = planFeatureMatchPairs(numberedImages(100), options);

    EXPECT_EQ(pairs.size(), 390);
    EXPECT_TRUE(pairs.contains(QStringLiteral("image_000__image_004")));
    EXPECT_FALSE(pairs.contains(QStringLiteral("image_000__image_005")));
    EXPECT_FALSE(pairs.contains(QStringLiteral("image_000__image_099")));
}

TEST(FeaturePairPlannerTest, LargeImageSetUsesKnownCameraSpatialNeighbors)
{
    FeaturePairPlannerOptions options;
    options.exhaustiveMaxImages = 10;
    options.sequentialWindow = 3;
    options.spatialNeighborCount = 2;

    std::vector<std::array<double, 3>> centers;
    centers.reserve(25);
    for (int i = 0; i < 25; ++i)
    {
        centers.push_back({1000.0 * double(i), 0.0, 100.0});
    }
    centers[20] = {5.0, 0.0, 100.0};
    options.knownCameraCenters = centers;

    const QStringList pairs = planFeatureMatchPairs(numberedImages(25), options);

    EXPECT_TRUE(pairs.contains(QStringLiteral("image_000__image_003")));
    EXPECT_TRUE(pairs.contains(QStringLiteral("image_000__image_020")));
    EXPECT_FALSE(pairs.contains(QStringLiteral("image_000__image_004")));
}

TEST(FeaturePairPlannerTest, ExplicitOverlapPairsTakePriority)
{
    FeaturePairPlannerOptions options;
    options.exhaustiveMaxImages = 10;
    options.sequentialWindow = 3;
    options.spatialNeighborCount = 2;
    options.knownCameraOverlapPairs = {
        {0, 1},
        {0, 3},
        {10, 20},
    };

    std::vector<std::array<double, 3>> centers;
    centers.reserve(25);
    for (int i = 0; i < 25; ++i)
    {
        centers.push_back({1000.0 * double(i), 0.0, 100.0});
    }
    centers[20] = {5.0, 0.0, 100.0};
    options.knownCameraCenters = centers;

    const QStringList pairs = planFeatureMatchPairs(numberedImages(25), options);

    EXPECT_EQ(pairs.size(), 3);
    EXPECT_TRUE(pairs.contains(QStringLiteral("image_000__image_001")));
    EXPECT_TRUE(pairs.contains(QStringLiteral("image_000__image_003")));
    EXPECT_TRUE(pairs.contains(QStringLiteral("image_010__image_020")));
    EXPECT_FALSE(pairs.contains(QStringLiteral("image_000__image_020")));
}

TEST(FeaturePairPlannerTest, LargePathSetUsesBoundedPipelinePairs)
{
    FeaturePairPlannerOptions options;
    options.exhaustiveMaxImages = 10;
    options.sequentialWindow = 4;

    const QStringList images = numberedImagePaths(100);
    const QStringList pairs = planFeatureMatchPairPaths(images, options);

    EXPECT_EQ(pairs.size(), 390);
    EXPECT_TRUE(pairs.contains(QStringLiteral("E:/dataset/images/image_000.JPG|E:/dataset/images/image_004.JPG")));
    EXPECT_FALSE(pairs.contains(QStringLiteral("E:/dataset/images/image_000.JPG|E:/dataset/images/image_005.JPG")));
    EXPECT_FALSE(pairs.contains(QStringLiteral("E:/dataset/images/image_000.JPG|E:/dataset/images/image_099.JPG")));
    for (const QString &pair : pairs)
    {
        EXPECT_TRUE(pair.contains(QLatin1Char('|'))) << pair.toStdString();
        EXPECT_FALSE(pair.contains(QStringLiteral("__"))) << pair.toStdString();
    }
}

TEST(FeaturePairPlannerTest, LargePathSetUsesKnownCameraSpatialNeighbors)
{
    FeaturePairPlannerOptions options;
    options.exhaustiveMaxImages = 10;
    options.sequentialWindow = 3;
    options.spatialNeighborCount = 2;

    std::vector<std::array<double, 3>> centers;
    centers.reserve(25);
    for (int i = 0; i < 25; ++i)
    {
        centers.push_back({1000.0 * double(i), 0.0, 100.0});
    }
    centers[20] = {5.0, 0.0, 100.0};
    options.knownCameraCenters = centers;

    const QStringList pairs = planFeatureMatchPairPaths(numberedImagePaths(25), options);

    EXPECT_TRUE(pairs.contains(
        QStringLiteral("E:/dataset/images/image_000.JPG|E:/dataset/images/image_003.JPG")));
    EXPECT_TRUE(pairs.contains(
        QStringLiteral("E:/dataset/images/image_000.JPG|E:/dataset/images/image_020.JPG")));
    EXPECT_FALSE(pairs.contains(
        QStringLiteral("E:/dataset/images/image_000.JPG|E:/dataset/images/image_004.JPG")));
}
