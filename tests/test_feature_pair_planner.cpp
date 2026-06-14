#include "FeaturePairPlanner.h"

#include <gtest/gtest.h>

#include <QStringList>

using xjw::gui::FeaturePairPlannerOptions;
using xjw::gui::planFeatureMatchPairs;

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
