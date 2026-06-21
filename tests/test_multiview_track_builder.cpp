#include <gtest/gtest.h>

#include "Camera.h"
#include "graph/CorrespondenceGraph.h"
#include "reconstruction/SfmReconstruction.h"
#include "tracks/MultiViewTrackBuilder.h"
#include "triangulation/Triangulator.h"

namespace
{

xjw::Camera makeCamera(double cx, double cy, double cz)
{
    xjw::Camera camera;
    camera.setIntrinsics(1200.0, 1200.0, 512.0, 384.0);
    const std::array<double, 9> rotation = {1.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0,
                                            0.0, 0.0, 1.0};
    const std::array<double, 3> center = {cx, cy, cz};
    camera.setPose(rotation, center);
    return camera;
}

xjw::FeatureKeypoint projectPoint(const xjw::Camera &camera, const std::array<double, 3> &xyz)
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

TEST(KnownPoseMultiViewTriangulationTest, CreatesSingleThreeViewTrack)
{
    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);
    const xjw::Camera camera2 = makeCamera(16.0, 0.0, 0.0);
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
    const xjw::ScenePoint3D &point = reconstruction.points3D().begin()->second;
    EXPECT_EQ(point.track.length(), 3);
}

TEST(KnownPoseMultiViewTriangulationTest, SplitsGeometryInconsistentComponent)
{
    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);
    const xjw::Camera camera2 = makeCamera(80.0, 0.0, 0.0);
    const xjw::Camera camera3 = makeCamera(88.0, 0.0, 0.0);
    const std::array<double, 3> leftPoint = {4.0, 0.5, 40.0};
    const std::array<double, 3> rightPoint = {84.0, -0.5, 40.0};

    const std::vector<xjw::Camera> cameras{camera0, camera1, camera2, camera3};

    xjw::SfmReconstruction reconstruction;
    for (xjw::ImageId imageId = 0; imageId < static_cast<xjw::ImageId>(cameras.size()); ++imageId)
    {
        xjw::ImageData image;
        image.id = imageId;
        const std::array<double, 3> &point = imageId < 2 ? leftPoint : rightPoint;
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

TEST(KnownPoseMultiViewTriangulationTest, RefinesNoisyThreeViewTrackBeforeRejectingObservation)
{
    const xjw::Camera camera0 = makeCamera(0.0, 0.0, 0.0);
    const xjw::Camera camera1 = makeCamera(8.0, 0.0, 0.0);
    const xjw::Camera camera2 = makeCamera(16.0, 0.0, 0.0);
    const std::vector<xjw::Camera> cameras{camera0, camera1, camera2};
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
    const xjw::ScenePoint3D &point = reconstruction.points3D().begin()->second;
    EXPECT_EQ(point.track.length(), 3);
    EXPECT_LT(point.error, 0.32);
}
