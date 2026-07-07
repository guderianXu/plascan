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

TEST(SfmPairPlannerTest, SequenceLoopClosureConnectsTailBackToHead)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 2;
    options.knownCameraAllPairsMaxImages = 2;
    options.knownCameraSpatialNeighborCount = 0;
    options.knownCameraSequenceLoopClosure = true;

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(8), cameraPaths(8), options);

    const QString loopKey = xjw::gui::canonicalSfmPairKey(imagePath(7), imagePath(0));
    const auto loopIt = std::find_if(plan.pairCandidates.begin(), plan.pairCandidates.end(),
                                     [&](const xjw::gui::SfmPairCandidate &candidate) {
                                         return candidate.pairKey == loopKey;
                                     });

    ASSERT_NE(loopIt, plan.pairCandidates.end());
    EXPECT_TRUE(plan.usedSequenceLoopClosure);
    EXPECT_TRUE(loopIt->sourceTypes.contains(QStringLiteral("sequence_loop")));
    EXPECT_EQ(loopIt->sequenceDistance, 1);
    EXPECT_GT(loopIt->sequenceScore, 0.0);
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

TEST(SfmPairPlannerTest, PairPlanRecordsPerPairSourcesAndPriority)
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

    ASSERT_EQ(plan.pairCandidates.size(), plan.allowedPairKeys.size());

    const QString sequenceKey = xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(3));
    const QString spatialKey = xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(20));
    const auto sequenceIt = std::find_if(plan.pairCandidates.begin(), plan.pairCandidates.end(),
                                         [&](const xjw::gui::SfmPairCandidate &candidate) {
                                             return candidate.pairKey == sequenceKey;
                                         });
    const auto spatialIt = std::find_if(plan.pairCandidates.begin(), plan.pairCandidates.end(),
                                        [&](const xjw::gui::SfmPairCandidate &candidate) {
                                            return candidate.pairKey == spatialKey;
                                        });
    ASSERT_NE(sequenceIt, plan.pairCandidates.end());
    ASSERT_NE(spatialIt, plan.pairCandidates.end());

    EXPECT_TRUE(sequenceIt->sourceTypes.contains(QStringLiteral("sequence_window")));
    EXPECT_EQ(sequenceIt->sequenceDistance, 3);
    EXPECT_GT(sequenceIt->sequenceScore, 0.0);
    EXPECT_GT(sequenceIt->priorityScore, 0.0);

    EXPECT_TRUE(spatialIt->sourceTypes.contains(QStringLiteral("known_camera_spatial_neighbors")));
    EXPECT_GT(spatialIt->spatialScore, 0.0);
    EXPECT_GE(spatialIt->centerDistance, 0.0);
    EXPECT_GT(spatialIt->priorityScore, 0.0);
}

TEST(SfmPairPlannerTest, PairPlanUsesViewingDirectionAndBaselineScoresForSpatialPriority)
{
    xjw::gui::SfmPairPlannerOptions options;
    options.autoRestrictKnownCameraPairs = true;
    options.knownCameraPairWindow = 3;
    options.knownCameraAllPairsMaxImages = 2;
    options.knownCameraSpatialNeighborCount = 3;

    std::vector<std::array<double, 3>> centers;
    centers.reserve(8);
    for (int i = 0; i < 8; ++i)
    {
        centers.push_back({100.0 * double(i), 0.0, 100.0});
    }
    centers[2] = {1.0, 0.0, 100.0};
    centers[3] = {2.0, 0.0, 100.0};
    options.knownCameraCenters = centers;

    std::vector<std::array<double, 3>> viewDirs(8, {0.0, 0.0, -1.0});
    viewDirs[2] = {0.0, 0.0, 1.0};   // close, but opposite looking direction
    viewDirs[3] = {0.0, 0.0, -1.0};  // farther, but consistent nadir direction
    options.knownCameraViewingDirections = viewDirs;

    const xjw::gui::SfmPairPlan plan =
        xjw::gui::planSfmMatchPairs(imagePaths(8), cameraPaths(8), options);

    const QString oppositeKey = xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(2));
    const QString alignedKey = xjw::gui::canonicalSfmPairKey(imagePath(0), imagePath(3));
    const auto oppositeIt = std::find_if(plan.pairCandidates.begin(), plan.pairCandidates.end(),
                                         [&](const xjw::gui::SfmPairCandidate &candidate) {
                                             return candidate.pairKey == oppositeKey;
                                         });
    const auto alignedIt = std::find_if(plan.pairCandidates.begin(), plan.pairCandidates.end(),
                                        [&](const xjw::gui::SfmPairCandidate &candidate) {
                                            return candidate.pairKey == alignedKey;
                                        });

    ASSERT_NE(oppositeIt, plan.pairCandidates.end());
    ASSERT_NE(alignedIt, plan.pairCandidates.end());

    EXPECT_GT(alignedIt->orientationScore, oppositeIt->orientationScore);
    EXPECT_LT(alignedIt->orientationAngleDeg, oppositeIt->orientationAngleDeg);
    EXPECT_GT(alignedIt->baselineScore, 0.0);
    EXPECT_GT(oppositeIt->baselineScore, 0.0);
    EXPECT_GT(alignedIt->priorityScore, oppositeIt->priorityScore);
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

TEST(SfmGuidedMatchPlannerTest, PrioritizesRegisteredWeakOverlapPairsForEpipolarRematching)
{
    const QVector<int> imageIds = {0, 1, 2, 3, 4};
    const QVector<xjw::gui::SfmMatchDiagnosticPair> pairs = {
        {0, 1, 240, true, false},
        {1, 2, 18, true, false},
        {2, 3, 0, true, true},
        {3, 4, 0, false, false},
        {0, 4, 160, true, false},
    };

    xjw::gui::SfmGuidedMatchPlannerOptions options;
    options.minSeedMatches = 80;
    options.maxHealthyMatches = 60;
    options.maxCandidates = 8;
    options.registeredImageIds = {0, 1, 2, 3};

    const xjw::gui::SfmGuidedMatchPlan plan =
        xjw::gui::planSfmGuidedMatching(imageIds, pairs, options);

    ASSERT_EQ(plan.candidates.size(), 2);
    EXPECT_EQ(plan.seedPairCount, 2);
    EXPECT_EQ(plan.skippedUnregisteredPairs, 1);

    const xjw::gui::SfmGuidedMatchCandidate &first = plan.candidates.front();
    EXPECT_EQ(first.imageA, 2);
    EXPECT_EQ(first.imageB, 3);
    EXPECT_EQ(first.reason, QStringLiteral("skipped_no_match_cache"));
    EXPECT_TRUE(first.canUseEpipolarBand);
    EXPECT_GT(first.priorityScore, plan.candidates.back().priorityScore);

    const xjw::gui::SfmGuidedMatchCandidate &second = plan.candidates.back();
    EXPECT_EQ(second.imageA, 1);
    EXPECT_EQ(second.imageB, 2);
    EXPECT_EQ(second.reason, QStringLiteral("weak_geometric_inliers"));
    EXPECT_TRUE(second.canUseEpipolarBand);
}
