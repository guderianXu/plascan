#include <gtest/gtest.h>

#include "SfmMatchDiagnostics.h"
#include "SfmPairPlanner.h"

#include <QDir>
#include <QStringList>

#include <array>
#include <vector>

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

TEST(SfmPairPlannerTest, KnownCameraCentersAddSpatialNeighborsOutsideSequenceWindow)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 20;
    options.knownCameraSpatialNeighborCount = 2;

    std::vector<std::array<double, 3>> centers;
    centers.reserve(25);
    for (int i = 0; i < 25; ++i)
    {
        centers.push_back({1000.0 * double(i), 0.0, 100.0});
    }
    centers[20] = {5.0, 0.0, 100.0};
    options.knownCameraCenters = centers;

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(25), cameraPaths(25), options);

    EXPECT_TRUE(plan.restrictPairs);
    EXPECT_TRUE(plan.autoRestricted);
    EXPECT_TRUE(plan.usedSpatialCameraCenters);
    EXPECT_EQ(plan.knownCameraSpatialNeighborCount, 2);

    const QSet<QString> keys(plan.allowedPairKeys.begin(), plan.allowedPairKeys.end());
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(20))));
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(3))));
    EXPECT_FALSE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(4))));
}

TEST(SfmPairPlannerTest, KnownCameraCentersWithoutCameraFilesStillRestrictLargeProject)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 20;
    options.knownCameraSpatialNeighborCount = 2;

    std::vector<std::array<double, 3>> centers;
    centers.reserve(40);
    for (int i = 0; i < 40; ++i)
    {
        centers.push_back({1000.0 * double(i), 0.0, 100.0});
    }
    centers[30] = {5.0, 0.0, 100.0};
    options.knownCameraCenters = centers;

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(40), QStringList(), options);

    EXPECT_TRUE(plan.restrictPairs);
    EXPECT_TRUE(plan.autoRestricted);
    EXPECT_TRUE(plan.usedSpatialCameraCenters);
    EXPECT_EQ(plan.allPairCount, 780);
    EXPECT_LT(plan.allowedPairKeys.size(), plan.allPairCount / 2);

    const QSet<QString> keys(plan.allowedPairKeys.begin(), plan.allowedPairKeys.end());
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(30))));
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(3))));
    EXPECT_FALSE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(4))));
}

TEST(SfmPairPlannerTest, KnownCameraOverlapPairsTakePriorityOverCenterNeighbors)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 20;
    options.knownCameraSpatialNeighborCount = 2;
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

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(25), cameraPaths(25), options);

    EXPECT_TRUE(plan.restrictPairs);
    EXPECT_TRUE(plan.autoRestricted);
    EXPECT_TRUE(plan.usedCameraOverlapPairs);
    EXPECT_FALSE(plan.usedSpatialCameraCenters);
    EXPECT_EQ(plan.knownCameraOverlapPairCount, 3);

    const QSet<QString> keys(plan.allowedPairKeys.begin(), plan.allowedPairKeys.end());
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(1))));
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(3))));
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(10), imagePath(20))));
    EXPECT_FALSE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(20))));
    EXPECT_FALSE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(4))));
}

TEST(SfmPairPlannerTest, DenseKnownCameraOverlapPairsFallbackToBoundedNeighbors)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 20;
    options.knownCameraSpatialNeighborCount = 2;

    for (int i = 0; i < 25; ++i)
    {
        for (int j = i + 1; j < 25; ++j)
        {
            options.knownCameraOverlapPairs.push_back({i, j});
        }
    }

    std::vector<std::array<double, 3>> centers;
    centers.reserve(25);
    for (int i = 0; i < 25; ++i)
    {
        centers.push_back({1000.0 * double(i), 0.0, 100.0});
    }
    centers[20] = {5.0, 0.0, 100.0};
    options.knownCameraCenters = centers;

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(25), cameraPaths(25), options);

    EXPECT_TRUE(plan.restrictPairs);
    EXPECT_TRUE(plan.autoRestricted);
    EXPECT_FALSE(plan.usedCameraOverlapPairs);
    EXPECT_TRUE(plan.usedSpatialCameraCenters);
    EXPECT_LT(plan.allowedPairKeys.size(), plan.allPairCount / 2);

    const QSet<QString> keys(plan.allowedPairKeys.begin(), plan.allowedPairKeys.end());
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(20))));
    EXPECT_TRUE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(3))));
    EXPECT_FALSE(keys.contains(xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(24))));
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

TEST(SfmMatchDiagnosticsTest, SeparatesCandidateGraphFromActualMatchGraph)
{
    const QVector<int> imageIds = {0, 1, 2, 3, 4, 5, 6, 7};
    const QVector<xjw::gui::SfmMatchDiagnosticPair> pairs = {
        {0, 1, 120, true, false},
        {1, 2, 115, true, false},
        {2, 3, 98, true, false},
        {3, 4, 0, true, true},
        {4, 5, 101, true, false},
        {5, 6, 99, true, false},
        {6, 7, 104, true, false},
        {1, 6, 0, true, true},
    };

    const xjw::gui::SfmMatchDiagnostics diagnostics =
        xjw::gui::analyzeSfmMatchDiagnostics(imageIds, pairs);

    EXPECT_EQ(diagnostics.totalPairs, 8);
    EXPECT_EQ(diagnostics.actualMatchPairs, 6);
    EXPECT_EQ(diagnostics.noMatchCacheSkippedPairs, 2);
    EXPECT_EQ(diagnostics.pendingPairs, 0);
    EXPECT_EQ(diagnostics.emptyLoadedPairs, 0);

    EXPECT_EQ(diagnostics.candidateGraph.componentCount, 1);
    EXPECT_EQ(diagnostics.candidateGraph.largestComponentSize, 8);
    EXPECT_EQ(diagnostics.actualMatchGraph.componentCount, 2);
    EXPECT_EQ(diagnostics.actualMatchGraph.largestComponentSize, 4);
    ASSERT_GE(diagnostics.actualMatchGraph.componentSizes.size(), 2);
    EXPECT_EQ(diagnostics.actualMatchGraph.componentSizes.at(0), 4);
    EXPECT_EQ(diagnostics.actualMatchGraph.componentSizes.at(1), 4);
}
