#include <gtest/gtest.h>

#include "SfmPairPlanner.h"

#include <QDir>
#include <QStringList>

namespace
{

QString imagePath(int index)
{
    return QDir::cleanPath(QStringLiteral("/tmp/plascan_pair_plan/img_%1.jpg").arg(index, 4, 10, QLatin1Char('0')));
}

QString cameraPath(int index)
{
    return QDir::cleanPath(QStringLiteral("/tmp/plascan_pair_plan/img_%1.tsai").arg(index, 4, 10, QLatin1Char('0')));
}

QStringList imagePaths(int count)
{
    QStringList result;
    for (int i = 0; i < count; ++i)
    {
        result.append(imagePath(i));
    }
    return result;
}

QStringList cameraPaths(int count)
{
    QStringList result;
    for (int i = 0; i < count; ++i)
    {
        result.append(cameraPath(i));
    }
    return result;
}

} // namespace

TEST(SfmPairPlannerTest, LargeKnownCameraSequenceUsesSlidingWindow)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 20;

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(25), cameraPaths(25), options);

    EXPECT_TRUE(plan.restrictPairs);
    EXPECT_TRUE(plan.autoRestricted);
    EXPECT_EQ(plan.allPairCount, 300);
    EXPECT_EQ(plan.allowedPairKeys.size(), 69);

    const QSet<QString> keys(plan.allowedPairKeys.begin(), plan.allowedPairKeys.end());
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(3))));
    EXPECT_FALSE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(4))));
}

TEST(SfmPairPlannerTest, SmallKnownCameraSetKeepsAllPairs)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 20;

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(12), cameraPaths(12), options);

    EXPECT_FALSE(plan.restrictPairs);
    EXPECT_FALSE(plan.autoRestricted);
    EXPECT_TRUE(plan.allowedPairKeys.isEmpty());
    EXPECT_EQ(plan.allPairCount, 66);
}

TEST(SfmPairPlannerTest, ExplicitPairRestrictionIsPreserved)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.restrictPairs = true;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 20;
    options.allowedPairs = {xjw::gui::canonicalSfmPairKey(imagePath(2), imagePath(9))};

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(25), cameraPaths(25), options);

    EXPECT_TRUE(plan.restrictPairs);
    EXPECT_FALSE(plan.autoRestricted);
    ASSERT_EQ(plan.allowedPairKeys.size(), 1);
    EXPECT_EQ(plan.allowedPairKeys.front(), xjw::gui::canonicalSfmPairKey(imagePath(2), imagePath(9)));
}

TEST(SfmPairPlannerTest, MissingCameraPathsKeepAllPairs)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 20;

    QStringList cameras = cameraPaths(25);
    cameras[4].clear();

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(25), cameras, options);

    EXPECT_FALSE(plan.restrictPairs);
    EXPECT_FALSE(plan.autoRestricted);
    EXPECT_TRUE(plan.allowedPairKeys.isEmpty());
    EXPECT_EQ(plan.allPairCount, 300);
}
