#include <gtest/gtest.h>

#include "FramePinholeCamera.h"
#include "graph/CorrespondenceGraph.h"
#include "reconstruction/SfmReconstruction.h"
#include "tracks/CorrespondenceTrackThinner.h"
#include "tracks/MultiViewTrackBuilder.h"
#include "triangulation/Triangulator.h"

namespace
{

    xjw::FramePinholeCamera makeCamera(double cx, double cy, double cz)
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(1200.0, 1200.0, 512.0, 384.0);
        const std::array<double, 9> rotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        const std::array<double, 3> center = {cx, cy, cz};
        camera.setPose(rotation, center);
        return camera;
    }

    xjw::FeatureKeypoint projectPoint(const xjw::FramePinholeCamera& camera, const std::array<double, 3>& xyz)
    {
        const double worldPoint[3] = {xyz[0], xyz[1], xyz[2]};
        double projected[2] = {0.0, 0.0};
        EXPECT_TRUE(camera.projectWorldPoint(worldPoint, projected));
        return xjw::FeatureKeypoint{static_cast<float>(projected[0]), static_cast<float>(projected[1])};
    }

} // namespace

TEST(MultiViewTrackBuilderTest, MergesConsistentPairwiseMatchesIntoTrack)
{
    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{10, 20}});
    builder.addMatchPair(1, 2, {{20, 30}});

    const xjw::MultiViewTrackBuildResult result = builder.build();

    ASSERT_EQ(result.tracks.size(), 1);
    EXPECT_EQ(result.acceptedComponents, 1);
    EXPECT_EQ(result.rejectedConflictComponents, 0);
    EXPECT_EQ(result.tracks.front().length(), 3);
    EXPECT_EQ(result.trackLengthHistogram.at(3), 1);
}

TEST(MultiViewTrackBuilderTest, DefaultBuildMatchesExplicitDefaultOptions)
{
    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{10, 20, 0.9f}});
    builder.addMatchPair(1, 2, {{20, 30, 0.7f}});

    const xjw::MultiViewTrackBuildResult default_result = builder.build();
    const xjw::MultiViewTrackBuildResult explicit_result = builder.build(xjw::MultiViewTrackBuilder::BuildOptions{});

    ASSERT_EQ(default_result.tracks.size(), explicit_result.tracks.size());
    ASSERT_EQ(default_result.trackConfidenceScores.size(), explicit_result.trackConfidenceScores.size());
    EXPECT_EQ(default_result.totalComponents, explicit_result.totalComponents);
    EXPECT_EQ(default_result.acceptedComponents, explicit_result.acceptedComponents);
    EXPECT_EQ(default_result.rejectedConflictComponents, explicit_result.rejectedConflictComponents);
    EXPECT_EQ(default_result.rejectedConflictEdges, explicit_result.rejectedConflictEdges);
    EXPECT_EQ(default_result.rejectedInconsistentBridgeEdges, explicit_result.rejectedInconsistentBridgeEdges);
    EXPECT_EQ(default_result.acceptedSupportedBridgeEdges, explicit_result.acceptedSupportedBridgeEdges);
    EXPECT_EQ(default_result.prunedByQualityThinning, explicit_result.prunedByQualityThinning);
    EXPECT_EQ(default_result.prunedStationaryTracks, explicit_result.prunedStationaryTracks);
    EXPECT_EQ(default_result.trackLengthHistogram, explicit_result.trackLengthHistogram);
    EXPECT_EQ(default_result.trackConfidenceScores, explicit_result.trackConfidenceScores);
    EXPECT_DOUBLE_EQ(default_result.meanTrackConfidence, explicit_result.meanTrackConfidence);

    for (std::size_t track_index = 0; track_index < default_result.tracks.size(); ++track_index)
    {
        const xjw::Track& default_track = default_result.tracks[track_index];
        const xjw::Track& explicit_track = explicit_result.tracks[track_index];
        ASSERT_EQ(default_track.elements.size(), explicit_track.elements.size());
        EXPECT_DOUBLE_EQ(default_track.confidence, explicit_track.confidence);
        for (std::size_t element_index = 0; element_index < default_track.elements.size(); ++element_index)
        {
            EXPECT_EQ(default_track.elements[element_index].imageId, explicit_track.elements[element_index].imageId);
            EXPECT_EQ(default_track.elements[element_index].featureIdx,
                      explicit_track.elements[element_index].featureIdx);
        }
    }
}

TEST(MultiViewTrackBuilderTest, RejectsUnsupportedLowConfidenceBridgeBetweenEstablishedTracks)
{
    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{10, 20, 0.95f}});
    builder.addMatchPair(2, 3, {{30, 40, 0.94f}});
    builder.addMatchPair(1, 2, {{20, 30, 0.20f}});

    const xjw::MultiViewTrackBuildResult result = builder.build();

    ASSERT_EQ(result.tracks.size(), 2u);
    EXPECT_EQ(result.rejectedInconsistentBridgeEdges, 1);
    EXPECT_EQ(result.acceptedSupportedBridgeEdges, 0);
    EXPECT_EQ(result.trackLengthHistogram.at(2), 2);
}

TEST(MultiViewTrackBuilderTest, AcceptsBridgeBackedByExactThreeObservationCycle)
{
    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{10, 20, 0.95f}});
    builder.addMatchPair(2, 3, {{30, 40, 0.94f}});
    builder.addMatchPair(1, 2, {{20, 30, 0.80f}});
    builder.addMatchPair(1, 4, {{20, 50, 0.70f}});
    builder.addMatchPair(2, 4, {{30, 50, 0.70f}});

    const xjw::MultiViewTrackBuildResult result = builder.build();

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().length(), 5u);
    EXPECT_EQ(result.rejectedInconsistentBridgeEdges, 0);
    EXPECT_EQ(result.acceptedSupportedBridgeEdges, 1);
}

TEST(MultiViewTrackBuilderTest, RejectsSingleEdgeBridgeBetweenMatureComponentsEvenAtHighConfidence)
{
    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{10, 20, 0.99f}});
    builder.addMatchPair(3, 4, {{40, 50, 0.99f}});
    builder.addMatchPair(1, 2, {{20, 30, 0.98f}});
    builder.addMatchPair(4, 5, {{50, 60, 0.98f}});
    builder.addMatchPair(2, 3, {{30, 40, 0.97f}});

    const xjw::MultiViewTrackBuildResult result = builder.build();

    ASSERT_EQ(result.tracks.size(), 2u);
    EXPECT_EQ(result.trackLengthHistogram.at(3), 2);
    EXPECT_EQ(result.rejectedInconsistentBridgeEdges, 1);
}

TEST(MultiViewTrackBuilderTest, SplitsConflictingComponentsInsteadOfDroppingAllTracks)
{
    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{10, 20, 0.9f}});
    builder.addMatchPair(0, 2, {{11, 30, 0.8f}});
    builder.addMatchPair(1, 2, {{20, 30, 0.1f}});

    const xjw::MultiViewTrackBuildResult result = builder.build();

    ASSERT_EQ(result.tracks.size(), 2);
    EXPECT_EQ(result.rejectedConflictEdges, 1);
    EXPECT_EQ(result.rejectedConflictComponents, 0);
    EXPECT_EQ(result.acceptedComponents, 2);
    EXPECT_EQ(result.trackLengthHistogram.at(2), 2);
}

TEST(MultiViewTrackBuilderTest, PrefersHighConfidenceEdgesWhenResolvingConflicts)
{
    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{10, 20, 0.95f}});
    builder.addMatchPair(1, 2, {{20, 30, 0.90f}});
    builder.addMatchPair(0, 2, {{10, 31, 0.05f}});

    const xjw::MultiViewTrackBuildResult result = builder.build();

    ASSERT_EQ(result.tracks.size(), 1);
    EXPECT_EQ(result.tracks.front().length(), 3);
    EXPECT_EQ(result.rejectedConflictEdges, 1);
    EXPECT_EQ(result.trackLengthHistogram.at(3), 1);
}

TEST(MultiViewTrackBuilderTest, PublishesTrackConfidenceFromAcceptedEdgeScores)
{
    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{10, 20, 0.90f}});
    builder.addMatchPair(1, 2, {{20, 30, 0.70f}});

    const xjw::MultiViewTrackBuildResult result = builder.build();

    ASSERT_EQ(result.tracks.size(), 1);
    ASSERT_EQ(result.trackConfidenceScores.size(), 1);
    EXPECT_NEAR(result.tracks.front().confidence, 0.80, 1e-6);
    EXPECT_NEAR(result.trackConfidenceScores.front(), 0.80, 1e-6);
    EXPECT_NEAR(result.meanTrackConfidence, 0.80, 1e-6);
}

TEST(MultiViewTrackBuilderTest, QualityThinningPrefersLongTracksOverDenseTwoViewTracks)
{
    xjw::MultiViewTrackBuilder builder;
    builder.setImageKeypoints(0, {{10.0f, 10.0f}, {12.0f, 10.0f}, {14.0f, 10.0f}});
    builder.setImageKeypoints(1, {{10.0f, 10.0f}, {12.0f, 10.0f}, {14.0f, 10.0f}});
    builder.setImageKeypoints(2, {{10.0f, 10.0f}});
    builder.addMatchPair(0, 1, {{0, 0, 0.50f}, {1, 1, 0.99f}, {2, 2, 0.98f}});
    builder.addMatchPair(1, 2, {{0, 0, 0.50f}});

    xjw::MultiViewTrackBuilder::BuildOptions options;
    options.enableQualityThinning = true;
    options.imageWidth = 100.0f;
    options.imageHeight = 100.0f;
    options.gridColumns = 1;
    options.gridRows = 1;
    options.maxTracksPerGridCell = 1;

    const xjw::MultiViewTrackBuildResult result = builder.build(options);

    ASSERT_EQ(result.tracks.size(), 1);
    EXPECT_EQ(result.tracks.front().length(), 3);
    EXPECT_EQ(result.prunedByQualityThinning, 2);
}

TEST(MultiViewTrackBuilderTest, QualityThinningKeepsSpatiallySeparatedTracks)
{
    xjw::MultiViewTrackBuilder builder;
    builder.setImageKeypoints(0, {{10.0f, 10.0f}, {90.0f, 10.0f}});
    builder.setImageKeypoints(1, {{10.0f, 10.0f}, {90.0f, 10.0f}});
    builder.addMatchPair(0, 1, {{0, 0, 0.90f}, {1, 1, 0.80f}});

    xjw::MultiViewTrackBuilder::BuildOptions options;
    options.enableQualityThinning = true;
    options.imageWidth = 100.0f;
    options.imageHeight = 100.0f;
    options.gridColumns = 2;
    options.gridRows = 1;
    options.maxTracksPerGridCell = 1;

    const xjw::MultiViewTrackBuildResult result = builder.build(options);

    ASSERT_EQ(result.tracks.size(), 2);
    EXPECT_EQ(result.prunedByQualityThinning, 0);
}

TEST(CorrespondenceGraphTrackRetentionTest, KeepsOnlyOriginalEdgesBelongingToSelectedTracks)
{
    xjw::CorrespondenceGraph graph;
    graph.addImage(0, 2);
    graph.addImage(1, 2);
    graph.addImage(2, 1);
    graph.addMatches(0, 1, {{0, 0, 0.9f}, {1, 1, 0.8f}});
    graph.addMatches(1, 2, {{0, 0, 0.9f}});

    xjw::Track selectedTrack;
    selectedTrack.elements = {{0, 0}, {1, 0}, {2, 0}};

    const std::size_t removed = graph.retainMatchesInTracks({selectedTrack});

    EXPECT_EQ(removed, 1u);
    EXPECT_EQ(graph.numImagePairs(), 2u);
    ASSERT_EQ(graph.matchesBetween(0, 1).size(), 1u);
    EXPECT_EQ(graph.matchesBetween(0, 1).front().idx1, 0u);
    EXPECT_EQ(graph.matchesBetween(0, 1).front().idx2, 0u);
    EXPECT_EQ(graph.matchesBetween(1, 2).size(), 1u);
    EXPECT_TRUE(graph.matchesBetween(0, 2).empty());
}

TEST(CorrespondenceGraphTest, CanonicalizesFeatureIndicesForReversedImagePair)
{
    xjw::CorrespondenceGraph graph;
    graph.addImage(0, 2);
    graph.addImage(1, 3);
    graph.addMatches(1, 0, {{2, 1, 0.8f}});
    graph.buildCorrespondences();

    ASSERT_EQ(graph.matchesBetween(0, 1).size(), 1u);
    EXPECT_EQ(graph.matchesBetween(0, 1).front().idx1, 1u);
    EXPECT_EQ(graph.matchesBetween(0, 1).front().idx2, 2u);

    const auto correspondences = graph.findCorrespondences(0, 1);
    ASSERT_EQ(correspondences.size(), 1u);
    EXPECT_EQ(correspondences.front().imageId, 1u);
    EXPECT_EQ(correspondences.front().featureIdx, 2u);
}

TEST(CorrespondenceGraphTest, UsesDeterministicCachedConnectivityCounts)
{
    xjw::CorrespondenceGraph graph;
    for (xjw::ImageId imageId = 0; imageId < 4; ++imageId)
    {
        graph.addImage(imageId, 4);
    }

    graph.addMatches(0, 2, {{0, 0, 0.9f}, {1, 1, 0.8f}});
    graph.addMatches(1, 0, {{0, 0, 0.9f}});
    graph.addMatches(0, 1, {{1, 1, 0.8f}, {2, 2, 0.7f}});
    graph.addMatches(3, 0, {{0, 0, 0.9f}, {1, 1, 0.8f}});

    const std::vector<xjw::ImageId> connected = graph.connectedImages(0);
    ASSERT_EQ(connected.size(), 3u);
    EXPECT_EQ(connected[0], 1u);
    EXPECT_EQ(connected[1], 2u);
    EXPECT_EQ(connected[2], 3u);

    const auto topConnected = graph.topConnectedImages(0, 3);
    ASSERT_EQ(topConnected.size(), 3u);
    EXPECT_EQ(topConnected[0].first, 1u);
    EXPECT_EQ(topConnected[0].second, 3u);
    EXPECT_EQ(topConnected[1].first, 2u);
    EXPECT_EQ(topConnected[1].second, 2u);
    EXPECT_EQ(topConnected[2].first, 3u);
    EXPECT_EQ(topConnected[2].second, 2u);
}

TEST(CorrespondenceGraphTest, RebuildsConnectivityAfterTrackRetention)
{
    xjw::CorrespondenceGraph graph;
    graph.addImage(0, 2);
    graph.addImage(1, 1);
    graph.addImage(2, 1);
    graph.addMatches(0, 1, {{0, 0, 0.9f}});
    graph.addMatches(0, 2, {{1, 0, 0.8f}});

    xjw::Track selectedTrack;
    selectedTrack.elements = {{0, 0}, {1, 0}};
    EXPECT_EQ(graph.retainMatchesInTracks({selectedTrack}), 1u);

    const std::vector<xjw::ImageId> connected = graph.connectedImages(0);
    ASSERT_EQ(connected.size(), 1u);
    EXPECT_EQ(connected.front(), 1u);
    EXPECT_TRUE(graph.connectedImages(2).empty());
    ASSERT_EQ(graph.topConnectedImages(0, 1).size(), 1u);
    EXPECT_EQ(graph.topConnectedImages(0, 1).front().first, 1u);
}

TEST(CorrespondenceTrackThinnerTest, EnforcesPerImageLimitAndPrefersLongTracks)
{
    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    for (xjw::ImageId imageId = 0; imageId < 3; ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        image.keypoints = imageId < 2
                              ? std::vector<xjw::FeatureKeypoint>{{10.0f, 10.0f}, {20.0f, 20.0f}, {30.0f, 30.0f}}
                              : std::vector<xjw::FeatureKeypoint>{{10.0f, 10.0f}};
        image.point3DIds.resize(image.keypoints.size(), xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        graph.addImage(imageId, image.keypoints.size());
    }
    graph.addMatches(0, 1, {{0, 0, 0.50f}, {1, 1, 0.99f}, {2, 2, 0.98f}});
    graph.addMatches(1, 2, {{0, 0, 0.50f}});

    xjw::CorrespondenceTrackThinningOptions options;
    options.maxTracksPerImage = 1;
    const xjw::CorrespondenceTrackThinningResult result =
        xjw::thinCorrespondenceTracks(reconstruction, &graph, options);

    EXPECT_EQ(result.inputTrackCount, 3);
    EXPECT_EQ(result.retainedTrackCount, 1);
    EXPECT_EQ(result.prunedTrackCount, 2);
    EXPECT_EQ(result.removedMatchCount, 2);
    ASSERT_EQ(graph.matchesBetween(0, 1).size(), 1u);
    EXPECT_EQ(graph.matchesBetween(1, 2).size(), 1u);
}

TEST(MultiViewTrackBuilderTest, ExcludesStationaryTracksWhenEnabled)
{
    xjw::MultiViewTrackBuilder builder;
    builder.setImageKeypoints(0, {{50.0f, 50.0f}, {10.0f, 10.0f}});
    builder.setImageKeypoints(1, {{50.2f, 49.8f}, {30.0f, 10.0f}});
    builder.setImageKeypoints(2, {{50.1f, 50.1f}, {50.0f, 10.0f}});
    builder.addMatchPair(0, 1, {{0, 0, 0.95f}, {1, 1, 0.90f}});
    builder.addMatchPair(1, 2, {{0, 0, 0.95f}, {1, 1, 0.90f}});

    xjw::MultiViewTrackBuilder::BuildOptions options;
    options.excludeStationaryTracks = true;
    options.stationaryTrackMaxPixelMotion = 1.0f;

    const xjw::MultiViewTrackBuildResult result = builder.build(options);

    ASSERT_EQ(result.tracks.size(), 1);
    EXPECT_EQ(result.prunedStationaryTracks, 1);
    ASSERT_EQ(result.tracks.front().elements.size(), 3u);
    EXPECT_EQ(result.tracks.front().elements.front().featureIdx, 1);
}

TEST(KnownPoseMultiViewTriangulationTest, CreatesSingleThreeViewTrack)
{
    const xjw::FramePinholeCamera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera1 = makeCamera(8.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera2 = makeCamera(16.0, 0.0, 0.0);
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    xjw::SfmReconstruction reconstruction;
    for (xjw::ImageId imageId = 0; imageId < 3; ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        image.keypoints.push_back(projectPoint(imageId == 0 ? camera0 : (imageId == 1 ? camera1 : camera2), xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
    }
    reconstruction.registerImage(0, camera0);
    reconstruction.registerImage(1, camera1);
    reconstruction.registerImage(2, camera2);

    xjw::CorrespondenceGraph graph;
    graph.addImage(0, 1);
    graph.addImage(1, 1);
    graph.addImage(2, 1);
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(1, 2, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::MultiViewTrackBuilder builder;
    builder.addMatchPair(0, 1, {{0, 0}});
    builder.addMatchPair(1, 2, {{0, 0}});
    const xjw::MultiViewTrackBuildResult trackBuild = builder.build();
    ASSERT_EQ(trackBuild.tracks.size(), 1);

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    options.maxReprojError = 2.0;
    const xjw::TriangulationStats stats = triangulator.triangulateTracks(trackBuild.tracks, options);

    EXPECT_EQ(stats.numCreated, 1);
    ASSERT_EQ(reconstruction.numPoints3D(), 1);
    const xjw::ScenePoint3D& point = reconstruction.points3D().begin()->second;
    EXPECT_EQ(point.track.length(), 3);
}

TEST(KnownPoseMultiViewTriangulationTest, SplitsGeometryInconsistentComponent)
{
    const xjw::FramePinholeCamera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera1 = makeCamera(8.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera2 = makeCamera(80.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera3 = makeCamera(88.0, 0.0, 0.0);
    const std::array<double, 3> leftPoint = {4.0, 0.5, 40.0};
    const std::array<double, 3> rightPoint = {84.0, -0.5, 40.0};

    const std::vector<xjw::FramePinholeCamera> cameras{camera0, camera1, camera2, camera3};

    xjw::SfmReconstruction reconstruction;
    for (xjw::ImageId imageId = 0; imageId < static_cast<xjw::ImageId>(cameras.size()); ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        const std::array<double, 3>& point = imageId < 2 ? leftPoint : rightPoint;
        image.keypoints.push_back(projectPoint(cameras[static_cast<std::size_t>(imageId)], point));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[static_cast<std::size_t>(imageId)]);
    }

    xjw::CorrespondenceGraph graph;
    for (xjw::ImageId imageId = 0; imageId < static_cast<xjw::ImageId>(cameras.size()); ++imageId)
    {
        graph.addImage(imageId, 1);
    }
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(1, 2, {xjw::FeatureMatch{0, 0}}); // bad bridge between two physical points
    graph.addMatches(2, 3, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::Track bridgedTrack;
    bridgedTrack.elements.push_back({0, 0});
    bridgedTrack.elements.push_back({1, 0});
    bridgedTrack.elements.push_back({2, 0});
    bridgedTrack.elements.push_back({3, 0});

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    options.maxReprojError = 2.0;
    options.completeMaxReprojError = 2.0;
    const xjw::TriangulationStats stats = triangulator.triangulateTracks({bridgedTrack}, options);

    EXPECT_EQ(stats.numCreated, 2);
    EXPECT_EQ(reconstruction.numPoints3D(), 2);
}

TEST(KnownPoseMultiViewTriangulationTest, KeepsNativeTwoViewTrack)
{
    const xjw::FramePinholeCamera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera1 = makeCamera(8.0, 0.0, 0.0);
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    xjw::Track track;
    const std::vector<xjw::FramePinholeCamera> cameras{camera0, camera1};
    for (xjw::ImageId imageId = 0; imageId < 2; ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        image.keypoints.push_back(projectPoint(cameras[imageId], xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[imageId]);
        graph.addImage(imageId, 1);
        track.elements.push_back({imageId, 0});
    }
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    const xjw::TriangulationStats stats = triangulator.triangulateTracks({track}, options);

    EXPECT_EQ(stats.numCreated, 1);
    EXPECT_EQ(stats.createdTwoViewTracks, 1);
    EXPECT_EQ(stats.suppressedTwoViewFragments, 0);
    ASSERT_EQ(reconstruction.numPoints3D(), 1);
    EXPECT_EQ(reconstruction.points3D().begin()->second.track.length(), 2);
}

TEST(KnownPoseMultiViewTriangulationTest, DefersPureTwoViewTrackAfterBootstrap)
{
    const std::vector<xjw::FramePinholeCamera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(16.0, 0.0, 0.0),
    };
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    xjw::Track track;
    for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        image.keypoints.push_back(projectPoint(cameras[imageId], xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[imageId]);
        graph.addImage(imageId, 1);
        if (imageId < 2)
        {
            track.elements.push_back({imageId, 0});
        }
    }
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    const xjw::TriangulationStats stats = xjw::Triangulator(reconstruction, graph).triangulateTracks({track}, options);

    EXPECT_EQ(stats.numCreated, 0);
    EXPECT_EQ(stats.deferredPureTwoViewTracks, 1);
    EXPECT_EQ(reconstruction.numPoints3D(), 0u);
}

TEST(KnownPoseMultiViewTriangulationTest, KeepsTwoViewCandidateWithPotentialThirdViewSupport)
{
    const std::vector<xjw::FramePinholeCamera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(16.0, 0.0, 0.0),
    };
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        image.keypoints.push_back(projectPoint(cameras[imageId], xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[imageId]);
        graph.addImage(imageId, 1);
    }
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(1, 2, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::Track track;
    track.elements.push_back({0, 0});
    track.elements.push_back({1, 0});
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    const xjw::TriangulationStats stats = xjw::Triangulator(reconstruction, graph).triangulateTracks({track}, options);

    EXPECT_EQ(stats.numCreated, 1);
    EXPECT_EQ(stats.deferredPureTwoViewTracks, 0);
    ASSERT_EQ(reconstruction.numPoints3D(), 1u);
}

TEST(IncrementalTriangulationTest, DefersNewPureTwoViewPointAfterThirdImageRegisters)
{
    const std::vector<xjw::FramePinholeCamera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(16.0, 0.0, 0.0),
    };
    const std::array<double, 3> xyz = {8.0, 0.5, 40.0};
    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        image.keypoints.push_back(projectPoint(cameras[imageId], xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[imageId]);
        graph.addImage(imageId, 1);
    }
    graph.addMatches(0, 2, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    const xjw::TriangulationStats stats = xjw::Triangulator(reconstruction, graph).triangulateImage(2, options);

    EXPECT_EQ(stats.numCreated, 0);
    EXPECT_EQ(stats.deferredPureTwoViewTracks, 1);
    EXPECT_EQ(reconstruction.numPoints3D(), 0u);
}

TEST(KnownPoseMultiViewTriangulationTest, SuppressesTwoViewResidualWhenLongConsensusExists)
{
    const std::vector<xjw::FramePinholeCamera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(16.0, 0.0, 0.0),
        makeCamera(80.0, 0.0, 0.0),
        makeCamera(88.0, 0.0, 0.0),
    };
    const std::array<double, 3> mainPoint = {8.0, 0.5, 40.0};
    const std::array<double, 3> residualPoint = {84.0, -0.5, 40.0};

    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    xjw::Track track;
    for (xjw::ImageId imageId = 0; imageId < static_cast<xjw::ImageId>(cameras.size()); ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        const auto& xyz = imageId < 3 ? mainPoint : residualPoint;
        image.keypoints.push_back(projectPoint(cameras[imageId], xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[imageId]);
        graph.addImage(imageId, 1);
        track.elements.push_back({imageId, 0});
    }
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(1, 2, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(2, 3, {xjw::FeatureMatch{0, 0}}); // erroneous bridge
    graph.addMatches(3, 4, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    options.maxReprojError = 1.0;
    options.completeMaxReprojError = 1.0;
    const xjw::TriangulationStats stats = triangulator.triangulateTracks({track}, options);

    EXPECT_EQ(stats.numCreated, 1);
    EXPECT_EQ(stats.createdLongTracks, 1);
    EXPECT_EQ(stats.createdTwoViewTracks, 0);
    EXPECT_EQ(stats.suppressedTwoViewFragments, 1);
    ASSERT_EQ(reconstruction.numPoints3D(), 1);
    EXPECT_EQ(reconstruction.points3D().begin()->second.track.length(), 3);
}

TEST(KnownPoseMultiViewTriangulationTest, RejectsTransitiveTwoViewCandidateWithoutDirectMatch)
{
    const std::vector<xjw::FramePinholeCamera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(80.0, 0.0, 0.0),
        makeCamera(16.0, 0.0, 0.0),
    };
    const std::array<double, 3> consistentPoint = {8.0, 0.5, 40.0};
    const std::array<double, 3> bridgePoint = {84.0, 12.0, 40.0};

    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    xjw::Track track;
    for (xjw::ImageId imageId = 0; imageId < 3; ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        const auto& xyz = imageId == 1 ? bridgePoint : consistentPoint;
        image.keypoints.push_back(projectPoint(cameras[imageId], xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[imageId]);
        graph.addImage(imageId, 1);
        track.elements.push_back({imageId, 0});
    }
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(1, 2, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    options.maxReprojError = 0.01;
    options.completeMaxReprojError = 0.01;
    const xjw::TriangulationStats stats = triangulator.triangulateTracks({track}, options);

    EXPECT_EQ(stats.numCreated, 0);
    EXPECT_GT(stats.indirectTwoViewCandidates, 0);
    EXPECT_EQ(reconstruction.numPoints3D(), 0);
}

TEST(KnownPoseMultiViewTriangulationTest, RejectsUnstableTwoViewFragmentFromLongTrack)
{
    const std::vector<xjw::FramePinholeCamera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(0.1, 0.0, 0.0),
        makeCamera(50.0, 0.0, 0.0),
    };
    const std::array<double, 3> shallowPoint = {0.05, 0.2, 100.0};
    const std::array<double, 3> bridgePoint = {52.0, 15.0, 35.0};

    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    xjw::Track track;
    for (xjw::ImageId imageId = 0; imageId < 3; ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        const auto& xyz = imageId < 2 ? shallowPoint : bridgePoint;
        image.keypoints.push_back(projectPoint(cameras[imageId], xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[imageId]);
        graph.addImage(imageId, 1);
        track.elements.push_back({imageId, 0});
    }
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(1, 2, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.001;
    options.maxReprojError = 0.01;
    options.completeMaxReprojError = 0.01;
    const xjw::TriangulationStats stats = triangulator.triangulateTracks({track}, options);

    EXPECT_EQ(stats.numCreated, 0);
    EXPECT_GT(stats.unstableTwoViewCandidates, 0);
    EXPECT_EQ(reconstruction.numPoints3D(), 0);
}

TEST(KnownPoseMultiViewTriangulationTest, UsesLocalMultiviewDepthOnlyWhenBothImagesHaveSupport)
{
    struct Outcome
    {
        xjw::TriangulationStats stats;
        std::size_t pointCount = 0;
    };
    const auto runCase = [](int referenceCount)
    {
        const std::vector<xjw::FramePinholeCamera> cameras{
            makeCamera(0.0, 0.0, 0.0),
            makeCamera(1.0, 0.0, 0.0),
            makeCamera(2.0, 0.0, 0.0),
        };
        xjw::SfmReconstruction reconstruction;
        xjw::CorrespondenceGraph graph;
        std::vector<xjw::Track> tracks;
        std::vector<xjw::ImageData> images(cameras.size());
        for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
        {
            images[imageId].id = imageId;
        }

        for (int index = 0; index < referenceCount; ++index)
        {
            const std::array<double, 3> referencePoint{
                0.02 * (index - referenceCount / 2),
                0.015 * (index % 3 - 1),
                40.0 + 0.05 * index,
            };
            xjw::Track track;
            for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
            {
                images[imageId].keypoints.push_back(projectPoint(cameras[imageId], referencePoint));
                track.elements.push_back({imageId, static_cast<xjw::FeatureIdx>(index)});
            }
            tracks.push_back(std::move(track));
            graph.addMatches(
                0, 1, {xjw::FeatureMatch{static_cast<xjw::FeatureIdx>(index), static_cast<xjw::FeatureIdx>(index)}});
            graph.addMatches(
                1, 2, {xjw::FeatureMatch{static_cast<xjw::FeatureIdx>(index), static_cast<xjw::FeatureIdx>(index)}});
        }

        const std::array<double, 3> inconsistentTwoViewPoint{0.0, 0.0, 80.0};
        xjw::Track twoViewTrack;
        const xjw::FeatureIdx twoViewFeature = static_cast<xjw::FeatureIdx>(referenceCount);
        for (xjw::ImageId imageId = 0; imageId < 2; ++imageId)
        {
            images[imageId].keypoints.push_back(projectPoint(cameras[imageId], inconsistentTwoViewPoint));
            twoViewTrack.elements.push_back({imageId, twoViewFeature});
        }
        tracks.push_back(twoViewTrack);
        graph.addMatches(0, 1, {xjw::FeatureMatch{twoViewFeature, twoViewFeature}});

        for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
        {
            images[imageId].point3DIds.resize(images[imageId].keypoints.size(), xjw::kInvalidPoint3DId);
            reconstruction.addImage(images[imageId]);
            reconstruction.registerImage(imageId, cameras[imageId]);
            graph.addImage(imageId, images[imageId].keypoints.size());
        }
        graph.buildCorrespondences();

        xjw::TriangulatorOptions options;
        options.minTriAngle = 0.1;
        options.maxReprojError = 0.1;
        options.completeMaxReprojError = 0.1;
        options.deferPureTwoViewTracks = false;
        const xjw::TriangulationStats stats =
            xjw::Triangulator(reconstruction, graph).triangulateTracks(tracks, options);
        return Outcome{stats, reconstruction.numPoints3D()};
    };

    const Outcome supported = runCase(5);
    EXPECT_EQ(supported.stats.localDepthInconsistentTwoViewPoints, 1);
    EXPECT_EQ(supported.stats.createdTwoViewTracks, 0);
    EXPECT_EQ(supported.pointCount, 5u);

    const Outcome sparse = runCase(2);
    EXPECT_EQ(sparse.stats.localDepthInconsistentTwoViewPoints, 0);
    EXPECT_EQ(sparse.stats.createdTwoViewTracks, 1);
    EXPECT_EQ(sparse.pointCount, 3u);
}

TEST(TriangulationFilterTest, RemovesBadObservationWithoutDeletingSupportedPoint)
{
    const std::vector<xjw::FramePinholeCamera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(16.0, 0.0, 0.0),
    };
    const std::array<double, 3> xyz = {4.0, 0.5, 40.0};

    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    xjw::Track track;
    for (xjw::ImageId imageId = 0; imageId < cameras.size(); ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        xjw::FeatureKeypoint keypoint = projectPoint(cameras[imageId], xyz);
        if (imageId == 2)
        {
            keypoint.x += 100.0f;
        }
        image.keypoints.push_back(keypoint);
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[imageId]);
        graph.addImage(imageId, 1);
        track.elements.push_back({imageId, 0});
    }
    graph.buildCorrespondences();

    const xjw::Point3DId pointId = reconstruction.addPoint3DWithTrack(xyz, track);
    xjw::Triangulator triangulator(reconstruction, graph);
    EXPECT_EQ(triangulator.filterPoints(2.0, 0.1), 0);

    ASSERT_TRUE(reconstruction.hasPoint3D(pointId));
    EXPECT_EQ(reconstruction.point3D(pointId).track.length(), 2);
    EXPECT_EQ(reconstruction.image(2).point3DIds[0], xjw::kInvalidPoint3DId);
    EXPECT_LT(reconstruction.point3D(pointId).error, 1.0e-6);
}

TEST(KnownPoseMultiViewTriangulationTest, RefinesNoisyThreeViewTrackBeforeRejectingObservation)
{
    const xjw::FramePinholeCamera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera1 = makeCamera(8.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera2 = makeCamera(16.0, 0.0, 0.0);
    const std::vector<xjw::FramePinholeCamera> cameras{camera0, camera1, camera2};
    const std::vector<xjw::FeatureKeypoint> observations{
        {632.324869f, 398.877649f},
        {391.894366f, 398.785406f},
        {152.173082f, 398.539692f},
    };

    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    for (xjw::ImageId imageId = 0; imageId < static_cast<xjw::ImageId>(cameras.size()); ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        image.keypoints.push_back(observations[static_cast<std::size_t>(imageId)]);
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, cameras[static_cast<std::size_t>(imageId)]);
        graph.addImage(imageId, 1);
    }
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(1, 2, {xjw::FeatureMatch{0, 0}});
    graph.addMatches(0, 2, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::Track track;
    track.elements.push_back({0, 0});
    track.elements.push_back({1, 0});
    track.elements.push_back({2, 0});

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    options.maxReprojError = 0.32;
    options.completeMaxReprojError = 0.32;
    const xjw::TriangulationStats stats = triangulator.triangulateTracks({track}, options);

    EXPECT_EQ(stats.numCreated, 1);
    EXPECT_EQ(stats.createdLongTracks, 1);
    ASSERT_EQ(reconstruction.numPoints3D(), 1);
    const xjw::ScenePoint3D& point = reconstruction.points3D().begin()->second;
    EXPECT_EQ(point.track.length(), 3);
    EXPECT_LT(point.error, 0.32);
}

TEST(KnownPoseMultiViewTriangulationTest, ConsistentLongTrackUsesSingleMultiviewSeed)
{
    constexpr xjw::ImageId kImageCount = 20;
    const std::array<double, 3> xyz = {5.0, 0.5, 80.0};
    xjw::SfmReconstruction reconstruction;
    xjw::CorrespondenceGraph graph;
    xjw::Track track;

    for (xjw::ImageId imageId = 0; imageId < kImageCount; ++imageId)
    {
        const xjw::FramePinholeCamera camera = makeCamera(static_cast<double>(imageId), 0.0, 0.0);
        xjw::ImageData image;
        image.id = imageId;
        image.keypoints.push_back(projectPoint(camera, xyz));
        image.point3DIds.resize(1, xjw::kInvalidPoint3DId);
        reconstruction.addImage(image);
        reconstruction.registerImage(imageId, camera);
        graph.addImage(imageId, 1);
        track.elements.push_back({imageId, 0});
    }
    graph.buildCorrespondences();

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    options.maxReprojError = 2.0;
    options.completeMaxReprojError = 2.0;
    const xjw::TriangulationStats stats = triangulator.triangulateTracks({track}, options);

    EXPECT_EQ(stats.numCreated, 1);
    EXPECT_EQ(stats.createdLongTracks, 1);
    EXPECT_EQ(stats.seedPairTests, 0);
    ASSERT_EQ(reconstruction.numPoints3D(), 1);
    EXPECT_EQ(reconstruction.points3D().begin()->second.track.length(), kImageCount);
}

TEST(IncrementalTriangulationTest, DoesNotReuseObservationOwnedByExistingPoint)
{
    const xjw::FramePinholeCamera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::FramePinholeCamera camera1 = makeCamera(8.0, 0.0, 0.0);
    const std::array<double, 3> existingPoint = {0.5, 0.2, 40.0};
    const std::array<double, 3> differentPoint = {6.0, 2.0, 35.0};

    xjw::SfmReconstruction reconstruction;
    xjw::ImageData image0;
    image0.id = 0;
    image0.keypoints.push_back(projectPoint(camera0, existingPoint));
    image0.point3DIds.resize(1, xjw::kInvalidPoint3DId);
    reconstruction.addImage(image0);
    reconstruction.registerImage(0, camera0);

    xjw::ImageData image1;
    image1.id = 1;
    image1.keypoints.push_back(projectPoint(camera1, differentPoint));
    image1.point3DIds.resize(1, xjw::kInvalidPoint3DId);
    reconstruction.addImage(image1);
    reconstruction.registerImage(1, camera1);

    xjw::Track existingTrack;
    existingTrack.elements.push_back({0, 0});
    const xjw::Point3DId existingId = reconstruction.addPoint3DWithTrack(existingPoint, existingTrack);

    xjw::CorrespondenceGraph graph;
    graph.addImage(0, 1);
    graph.addImage(1, 1);
    graph.addMatches(0, 1, {xjw::FeatureMatch{0, 0}});
    graph.buildCorrespondences();

    xjw::Triangulator triangulator(reconstruction, graph);
    xjw::TriangulatorOptions options;
    options.minTriAngle = 0.1;
    options.maxReprojError = 100.0;
    options.continueMaxReprojError = 0.1;
    const xjw::TriangulationStats stats = triangulator.triangulateImage(1, options);

    EXPECT_EQ(stats.numCreated, 0);
    EXPECT_EQ(reconstruction.numPoints3D(), 1u);
    EXPECT_EQ(reconstruction.image(0).point3DIds[0], existingId);
    EXPECT_EQ(reconstruction.image(1).point3DIds[0], xjw::kInvalidPoint3DId);
}
