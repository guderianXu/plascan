#include <gtest/gtest.h>

#include "DepthMapMeshBuilder.h"
#include "DepthTsdfSurfaceBuilder.h"
#include "result/OperationResult.h"
#include "ModelWorkflowService.h"
#include "MeshColorizer.h"
#include "MeshQuadricSimplifier.h"
#include "PointCloudPreprocess.h"
#include "SurfaceReconstructor.h"
#include "SurfaceReconstructorPostprocess.h"
#include "TextureMapper.h"
#include "VisualHullReconstructor.h"

#include <plapoint/filters/preprocessing.h>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/io/obj_io.h>
#include <plapoint/io/ply_io.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QTemporaryDir>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace xjw::core::project
{
xjw::common::OperationResult writeDepthMatStorage(const QString &path, const cv::Mat &matrix);
}

namespace
{

QVector<xjw::mesh::DepthTsdfFrame> makeSyntheticPlaneFrames(bool addRejections)
{
    QVector<xjw::mesh::DepthTsdfFrame> frames;
    for (int index = 0; index < 3; ++index)
    {
        const double centerX = (index - 1) * 0.1;
        xjw::mesh::DepthTsdfFrame frame;
        frame.refIndex = index;
        frame.camera.setIntrinsics(40.0, 40.0, 24.0, 18.0);
        frame.camera.setPose(std::array<double, 9>{1.0, 0.0, 0.0,
                                                   0.0, 1.0, 0.0,
                                                   0.0, 0.0, 1.0},
                             std::array<double, 3>{centerX, 0.0, 0.0});
        frame.depth = cv::Mat(36, 48, CV_32FC1, cv::Scalar(2.0f));
        frame.confidence = cv::Mat(36, 48, CV_32FC1, cv::Scalar(0.9f));
        frame.geometrySupportCount = cv::Mat(36, 48, CV_16UC1, cv::Scalar(3));
        frame.depthValidMask = cv::Mat(36, 48, CV_8UC1, cv::Scalar(255));
        frame.supportMask = cv::Mat(36, 48, CV_8UC1, cv::Scalar(255));
        frame.frameQualityWeight = 1.0f;
        if (addRejections)
        {
            if (index == 0)
            {
                frame.confidence.colRange(0, 12).setTo(0.1f);
                frame.depthValidMask(cv::Rect(31, 13, 10, 11)).setTo(0);
            }
            const int supportCenter = static_cast<int>(std::lround(24.0 - centerX * 20.0));
            frame.supportMask(cv::Rect(supportCenter - 5, 12, 11, 13)).setTo(0);
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

xjw::mesh::DepthTsdfFrame makeSamplingFrame(const cv::Mat &depth)
{
    xjw::mesh::DepthTsdfFrame frame;
    frame.depth = depth.clone();
    frame.confidence = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.9f));
    frame.geometrySupportCount = cv::Mat(depth.size(), CV_16UC1, cv::Scalar(4));
    frame.inverseDepthRelativeSpread = cv::Mat(
        depth.size(), CV_32FC1, cv::Scalar(0.005f));
    frame.depthValidMask = cv::Mat(depth.size(), CV_8UC1, cv::Scalar(255));
    frame.supportMask = cv::Mat(depth.size(), CV_8UC1, cv::Scalar(255));
    return frame;
}

} // namespace

TEST(DepthTsdfSurfaceBuilderTest, ResolutionAppliesToLongestPhysicalAxis)
{
    const std::array<float, 3> minimum{0.0f, 0.0f, 0.0f};
    const std::array<float, 3> maximum{2.0f, 1.0f, 0.5f};
    const auto layout = xjw::mesh::DepthTsdfSurfaceBuilder::makeLayout(
        minimum, maximum, 320, false);

    ASSERT_TRUE(layout.ok);
    EXPECT_EQ(layout.cells[0], 320);
    EXPECT_EQ(layout.cells[1], 160);
    EXPECT_EQ(layout.cells[2], 80);
    EXPECT_GT(layout.requiredBytes, 0u);
}

TEST(DepthTsdfSurfaceBuilderTest, UsesLargestComponentRelativeCleanupByDefault)
{
    const xjw::mesh::DepthTsdfOptions options;

    EXPECT_FLOAT_EQ(options.minimumComponentFaceRatio, 0.025f);
    EXPECT_EQ(options.minimumDistinctCameraSupport, 2);
    EXPECT_FLOAT_EQ(options.minimumSingleObservationWeight, 0.70f);
    EXPECT_FLOAT_EQ(options.minimumGeometryVerifiedObservationWeight, 0.85f);
    EXPECT_EQ(options.minimumGeometrySupportCount, 4);
    EXPECT_FALSE(options.allowGeometryVerifiedSingleObservation);
    EXPECT_FALSE(options.enableDiscontinuityAwareSampling);
    EXPECT_FLOAT_EQ(options.maximumInterpolationRelativeDepthSpread, 0.02f);
    EXPECT_FLOAT_EQ(options.maximumObservationInverseDepthSpread, 0.0f);
    EXPECT_TRUE(options.allowInvalidNearestPixelRecovery);
    EXPECT_FLOAT_EQ(options.maximumInvalidNearestPixelRecoveryInverseDepthSpread, 0.0f);
    EXPECT_FALSE(options.enableCrossViewConsensusDepth);
    EXPECT_FLOAT_EQ(options.maximumCrossViewConsensusInverseDepthSpread, 0.02f);
    EXPECT_FALSE(options.enableGeometrySingleViewNeighborhoodGuard);
    EXPECT_EQ(options.minimumGeometrySingleViewNeighborCount, 2);
    EXPECT_EQ(options.geometrySingleViewGrowthPasses, 2);
    EXPECT_FLOAT_EQ(options.maximumGeometrySingleViewNeighborTsdfDelta, 0.35f);
    EXPECT_FLOAT_EQ(options.truncationVoxels, 7.5f);
    EXPECT_FALSE(options.enableSupportMaskFreeSpaceCarving);
    EXPECT_FALSE(options.fillSmallBoundaryHoles);
    EXPECT_EQ(options.boundarySmoothingIterations, 0);
    EXPECT_EQ(options.depthValidBoundaryErosionPixels, 0);
}

TEST(DepthTsdfSurfaceBuilderTest, SubpixelSamplingInterpolatesOnlyOneDepthSurface)
{
    const cv::Mat depth = (cv::Mat_<float>(2, 2) << 2.00f, 2.02f,
                                                     1.98f, 2.01f);
    const auto frame = makeSamplingFrame(depth);

    const auto sample = xjw::mesh::DepthTsdfSurfaceBuilder::sampleObservation(
        frame, frame.depthValidMask, cv::Point2d(0.5, 0.5), 0.25f, true, 0.02f);

    ASSERT_TRUE(sample.valid);
    EXPECT_EQ(sample.contributingPixelCount, 4);
    EXPECT_NEAR(sample.depth, 2.0025f, 0.02f);
    EXPECT_EQ(sample.discontinuityRejectedPixelCount, 0);
}

TEST(DepthTsdfSurfaceBuilderTest, SubpixelSamplingDoesNotAverageAcrossDiscontinuity)
{
    const cv::Mat depth = (cv::Mat_<float>(2, 2) << 2.0f, 10.0f,
                                                     2.0f, 10.0f);
    const auto frame = makeSamplingFrame(depth);

    const auto foreground = xjw::mesh::DepthTsdfSurfaceBuilder::sampleObservation(
        frame, frame.depthValidMask, cv::Point2d(0.49, 0.5), 0.25f, true, 0.02f);
    const auto background = xjw::mesh::DepthTsdfSurfaceBuilder::sampleObservation(
        frame, frame.depthValidMask, cv::Point2d(0.51, 0.5), 0.25f, true, 0.02f);

    ASSERT_TRUE(foreground.valid);
    ASSERT_TRUE(background.valid);
    EXPECT_NEAR(foreground.depth, 2.0f, 1.0e-5f);
    EXPECT_NEAR(background.depth, 10.0f, 1.0e-5f);
    EXPECT_EQ(foreground.discontinuityRejectedPixelCount, 2);
    EXPECT_EQ(background.discontinuityRejectedPixelCount, 2);
}

TEST(DepthTsdfSurfaceBuilderTest, SubpixelSamplingRecoversInvalidNearestPixelFromSameSurface)
{
    const cv::Mat depth(2, 2, CV_32FC1, cv::Scalar(2.0f));
    auto frame = makeSamplingFrame(depth);
    frame.depthValidMask.at<std::uint8_t>(0, 0) = 0;

    const auto sample = xjw::mesh::DepthTsdfSurfaceBuilder::sampleObservation(
        frame, frame.depthValidMask, cv::Point2d(0.1, 0.1), 0.25f, true, 0.02f);

    ASSERT_TRUE(sample.valid);
    EXPECT_TRUE(sample.recoveredFromInvalidNearestPixel);
    EXPECT_NEAR(sample.depth, 2.0f, 1.0e-5f);
    EXPECT_EQ(sample.contributingPixelCount, 3);
}

TEST(DepthTsdfSurfaceBuilderTest, CrossViewSpreadGateRejectsUnstableDepthObservation)
{
    const cv::Mat depth(2, 2, CV_32FC1, cv::Scalar(2.0f));
    auto frame = makeSamplingFrame(depth);
    frame.inverseDepthRelativeSpread.setTo(0.04f);

    const auto sample = xjw::mesh::DepthTsdfSurfaceBuilder::sampleObservation(
        frame,
        frame.depthValidMask,
        cv::Point2d(0.5, 0.5),
        0.25f,
        true,
        0.02f,
        0.03f);

    EXPECT_FALSE(sample.valid);
    EXPECT_EQ(sample.failure, xjw::mesh::DepthTsdfObservationFailure::GeometryConsistency);
}

TEST(DepthTsdfSurfaceBuilderTest, InvalidNearestPixelRecoveryCanBeDisabled)
{
    const cv::Mat depth(2, 2, CV_32FC1, cv::Scalar(2.0f));
    auto frame = makeSamplingFrame(depth);
    frame.depthValidMask.at<std::uint8_t>(0, 0) = 0;

    const auto sample = xjw::mesh::DepthTsdfSurfaceBuilder::sampleObservation(
        frame,
        frame.depthValidMask,
        cv::Point2d(0.1, 0.1),
        0.25f,
        true,
        0.02f,
        0.0f,
        false);

    EXPECT_FALSE(sample.valid);
    EXPECT_TRUE(sample.rejectedInvalidNearestPixelRecovery);
}

TEST(DepthTsdfSurfaceBuilderTest, InvalidNearestRecoveryRejectsCrossViewUnstableNeighbor)
{
    const cv::Mat depth(2, 2, CV_32FC1, cv::Scalar(2.0f));
    auto frame = makeSamplingFrame(depth);
    frame.depthValidMask.at<std::uint8_t>(0, 0) = 0;
    frame.inverseDepthRelativeSpread.setTo(0.02f);

    const auto sample = xjw::mesh::DepthTsdfSurfaceBuilder::sampleObservation(
        frame,
        frame.depthValidMask,
        cv::Point2d(0.1, 0.1),
        0.25f,
        true,
        0.02f,
        0.0f,
        true,
        0.015f);

    EXPECT_FALSE(sample.valid);
    EXPECT_TRUE(sample.rejectedInvalidNearestPixelRecovery);
    EXPECT_EQ(sample.failure, xjw::mesh::DepthTsdfObservationFailure::GeometryConsistency);
}

TEST(DepthTsdfSurfaceBuilderTest, StableCrossViewConsensusCanReplaceReferenceDepth)
{
    const cv::Mat depth(2, 2, CV_32FC1, cv::Scalar(2.0f));
    auto frame = makeSamplingFrame(depth);
    frame.inverseDepthMean = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.5f));
    frame.inverseDepthMean.at<float>(0, 0) = 0.4f;

    const auto sample = xjw::mesh::DepthTsdfSurfaceBuilder::sampleObservation(
        frame,
        frame.depthValidMask,
        cv::Point2d(0.0, 0.0),
        0.25f,
        true,
        0.02f,
        0.0f,
        true,
        0.0f,
        true,
        0.02f);

    ASSERT_TRUE(sample.valid);
    EXPECT_TRUE(sample.usedCrossViewConsensusDepth);
    EXPECT_NEAR(sample.depth, 2.5f, 1.0e-5f);
}

TEST(SurfaceReconstructorPostprocessTest, SmoothsOnlyOpenBoundaryVertices)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(5);
    mesh.vertices[0].x = -1.0f;
    mesh.vertices[0].y = -1.0f;
    mesh.vertices[1].x = 1.0f;
    mesh.vertices[1].y = -1.0f;
    mesh.vertices[2].x = 1.5f;
    mesh.vertices[2].y = 1.0f;
    mesh.vertices[3].x = -1.0f;
    mesh.vertices[3].y = 1.0f;
    mesh.faces = {{{0, 1, 4}}, {{1, 2, 4}}, {{2, 3, 4}}, {{3, 0, 4}}};

    const int moved = xjw::mesh::detail::smoothOpenBoundaryVertices(
        &mesh, 1, 0.25f, 0.20f);

    EXPECT_EQ(moved, 4);
    EXPECT_LT(mesh.vertices[2].x, 1.5f);
    EXPECT_FLOAT_EQ(mesh.vertices[4].x, 0.0f);
    EXPECT_FLOAT_EQ(mesh.vertices[4].y, 0.0f);
    EXPECT_EQ(mesh.faceCount(), 4);
}

TEST(SurfaceReconstructorPostprocessTest, CompactsOnlyUnreferencedVertices)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(5);
    mesh.vertices[0].x = 10.0f;
    mesh.vertices[1].x = 20.0f;
    mesh.vertices[2].x = 30.0f;
    mesh.vertices[3].x = 40.0f;
    mesh.vertices[4].x = 50.0f;
    xjw::mesh::Triangle face;
    face.v[0] = 3;
    face.v[1] = 1;
    face.v[2] = 4;
    mesh.faces.push_back(face);

    const int removed = xjw::mesh::detail::compactReferencedVertices(&mesh);

    EXPECT_EQ(removed, 2);
    ASSERT_EQ(mesh.vertices.size(), 3u);
    EXPECT_FLOAT_EQ(mesh.vertices[0].x, 20.0f);
    EXPECT_FLOAT_EQ(mesh.vertices[1].x, 40.0f);
    EXPECT_FLOAT_EQ(mesh.vertices[2].x, 50.0f);
    ASSERT_EQ(mesh.faces.size(), 1u);
    EXPECT_EQ(mesh.faces[0].v[0], 1);
    EXPECT_EQ(mesh.faces[0].v[1], 0);
    EXPECT_EQ(mesh.faces[0].v[2], 2);
}

TEST(MeshQuadricSimplifierTest, ReducesPlanarInteriorWithoutMovingOpenBoundary)
{
    constexpr int side = 16;
    xjw::mesh::TriMesh mesh;
    mesh.vertices.reserve(side * side);
    for (int row = 0; row < side; ++row)
    {
        for (int column = 0; column < side; ++column)
        {
            xjw::mesh::MeshVertex vertex;
            vertex.x = static_cast<float>(column);
            vertex.y = static_cast<float>(row);
            vertex.nz = 1.0f;
            mesh.vertices.push_back(vertex);
        }
    }
    for (int row = 0; row + 1 < side; ++row)
    {
        for (int column = 0; column + 1 < side; ++column)
        {
            const int first = row * side + column;
            const int second = first + 1;
            const int third = first + side;
            const int fourth = third + 1;
            xjw::mesh::Triangle left;
            left.v[0] = first; left.v[1] = second; left.v[2] = third;
            xjw::mesh::Triangle right;
            right.v[0] = second; right.v[1] = fourth; right.v[2] = third;
            mesh.faces.push_back(left);
            mesh.faces.push_back(right);
        }
    }

    xjw::mesh::QuadricSimplifyOptions options;
    options.targetFaceCount = 250;
    options.maximumPasses = 8;
    const auto statistics = xjw::mesh::simplifyMeshQuadric(&mesh, options);

    EXPECT_LT(statistics.outputFaceCount, statistics.inputFaceCount);
    EXPECT_LE(statistics.outputFaceCount, 275);
    EXPECT_GT(statistics.collapsedEdgeCount, 0);
    float minimum_x = std::numeric_limits<float>::max();
    float maximum_x = std::numeric_limits<float>::lowest();
    float minimum_y = std::numeric_limits<float>::max();
    float maximum_y = std::numeric_limits<float>::lowest();
    for (const auto &vertex : mesh.vertices)
    {
        minimum_x = std::min(minimum_x, vertex.x);
        maximum_x = std::max(maximum_x, vertex.x);
        minimum_y = std::min(minimum_y, vertex.y);
        maximum_y = std::max(maximum_y, vertex.y);
    }
    EXPECT_FLOAT_EQ(minimum_x, 0.0f);
    EXPECT_FLOAT_EQ(maximum_x, static_cast<float>(side - 1));
    EXPECT_FLOAT_EQ(minimum_y, 0.0f);
    EXPECT_FLOAT_EQ(maximum_y, static_cast<float>(side - 1));
}

TEST(MeshQuadricSimplifierTest, PreservesSharpCubeEdges)
{
    xjw::mesh::TriMesh mesh;
    for (int z = 0; z <= 1; ++z)
    {
        for (int y = 0; y <= 1; ++y)
        {
            for (int x = 0; x <= 1; ++x)
            {
                xjw::mesh::MeshVertex vertex;
                vertex.x = static_cast<float>(x);
                vertex.y = static_cast<float>(y);
                vertex.z = static_cast<float>(z);
                mesh.vertices.push_back(vertex);
            }
        }
    }
    const int face_indices[][3] = {
        {0, 1, 3}, {0, 3, 2}, {4, 6, 7}, {4, 7, 5},
        {0, 4, 5}, {0, 5, 1}, {2, 3, 7}, {2, 7, 6},
        {0, 2, 6}, {0, 6, 4}, {1, 5, 7}, {1, 7, 3}};
    for (const auto &indices : face_indices)
    {
        xjw::mesh::Triangle face;
        face.v[0] = indices[0]; face.v[1] = indices[1]; face.v[2] = indices[2];
        mesh.faces.push_back(face);
    }

    xjw::mesh::QuadricSimplifyOptions options;
    options.targetFaceCount = 4;
    options.featureAngleDegrees = 30.0f;
    const auto statistics = xjw::mesh::simplifyMeshQuadric(&mesh, options);

    EXPECT_EQ(statistics.outputFaceCount, 12);
    EXPECT_GT(statistics.rejectedFeatureEdgeCount, 0);
}

TEST(DepthTsdfSurfaceBuilderTest, StrongSingleObservationHasExplicitSupportPath)
{
    xjw::mesh::DepthTsdfOptions options;
    options.minimumVoxelWeight = 1.0f;
    options.minimumSingleObservationWeight = 0.70f;
    options.minimumDistinctCameraSupport = 1;

    bool single_view = false;
    bool multi_view = false;
    EXPECT_TRUE(xjw::mesh::DepthTsdfSurfaceBuilder::isSampleSupported(
        0.75f, 1, 0.75f, options, &single_view, &multi_view));
    EXPECT_TRUE(single_view);
    EXPECT_FALSE(multi_view);
    EXPECT_FALSE(xjw::mesh::DepthTsdfSurfaceBuilder::isSampleSupported(
        0.65f, 1, 0.65f, options));

    options.minimumDistinctCameraSupport = 2;
    EXPECT_FALSE(xjw::mesh::DepthTsdfSurfaceBuilder::isSampleSupported(
        0.90f, 1, 0.90f, options));
    EXPECT_TRUE(xjw::mesh::DepthTsdfSurfaceBuilder::isSampleSupported(
        1.20f, 2, 0.70f, options));
}

TEST(DepthTsdfSurfaceBuilderTest, GeometryVerifiedSingleObservationRequiresTwoSourceConfirmations)
{
    xjw::mesh::DepthTsdfOptions options;
    options.minimumDistinctCameraSupport = 2;
    options.allowGeometryVerifiedSingleObservation = true;
    options.minimumGeometryVerifiedObservationWeight = 0.85f;
    options.minimumGeometrySupportCount = 3;

    bool geometry_verified = false;
    EXPECT_TRUE(xjw::mesh::DepthTsdfSurfaceBuilder::isSampleSupported(
        0.9f, 1, 0.9f, options, nullptr, nullptr, 3, &geometry_verified));
    EXPECT_TRUE(geometry_verified);
    EXPECT_FALSE(xjw::mesh::DepthTsdfSurfaceBuilder::isSampleSupported(
        0.9f, 1, 0.9f, options, nullptr, nullptr, 2));
    EXPECT_FALSE(xjw::mesh::DepthTsdfSurfaceBuilder::isSampleSupported(
        0.8f, 1, 0.8f, options, nullptr, nullptr, 3));
}

TEST(DepthTsdfSurfaceBuilderTest, GeometrySingleViewGrowthKeepsOnlyCoreConnectedSamples)
{
    xjw::mesh::DepthTsdfLayout layout;
    layout.cells = {2, 2, 2};
    layout.sampleCount = 27;
    std::vector<float> tsdf(27, 0.0f);
    std::vector<std::uint8_t> supported(27, 0);
    const auto index = [](int x, int y, int z)
    {
        return static_cast<std::size_t>(z * 9 + y * 3 + x);
    };
    supported[index(0, 1, 1)] = 1;
    supported[index(1, 0, 1)] = 1;
    const std::vector<std::size_t> candidates{index(1, 1, 1), index(2, 2, 2)};

    const int accepted =
        xjw::mesh::DepthTsdfSurfaceBuilder::growGeometryVerifiedSingleViewSamples(
            layout, tsdf, candidates, 2, 1, 0.1f, &supported);

    EXPECT_EQ(accepted, 1);
    EXPECT_EQ(supported[index(1, 1, 1)], 1);
    EXPECT_EQ(supported[index(2, 2, 2)], 0);
}

TEST(DepthTsdfSurfaceBuilderTest, WeakBoundaryTrimTargetsOnlyDanglingFaces)
{
    EXPECT_TRUE(xjw::mesh::DepthTsdfSurfaceBuilder::shouldTrimWeakBoundaryFace(2, 1));
    EXPECT_TRUE(xjw::mesh::DepthTsdfSurfaceBuilder::shouldTrimWeakBoundaryFace(3, 1));
    EXPECT_FALSE(xjw::mesh::DepthTsdfSurfaceBuilder::shouldTrimWeakBoundaryFace(1, 3));
    EXPECT_FALSE(xjw::mesh::DepthTsdfSurfaceBuilder::shouldTrimWeakBoundaryFace(2, 0));
}

TEST(DepthTsdfSurfaceBuilderTest, MemoryFailureDoesNotLowerResolution)
{
    xjw::mesh::DepthTsdfOptions options;
    options.resolution = 320;
    options.availableMemoryBytes = 1024;

    const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::validateAllocation(
        std::array<float, 3>{0.0f, 0.0f, 0.0f},
        std::array<float, 3>{1.0f, 1.0f, 1.0f},
        options);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.layout.cells[0], 320);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("320")));
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("bytes")));
}

TEST(DepthTsdfSurfaceBuilderTest, LoadsProductionArtifactsAndEstimatesCameraAxisBounds)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    QVector<xjw::mesh::DepthFrameArtifact> artifacts;
    for (int index = 0; index < 4; ++index)
    {
        const QString prefix = directory.filePath(QStringLiteral("depth_%1").arg(index));
        const QString depthPath = prefix + QStringLiteral(".bin");
        const QString confidencePath = prefix + QStringLiteral("_conf.bin");
        const QString geometrySupportPath = prefix + QStringLiteral("_geometry_support.bin");
        const QString geometrySourceMaskPath =
            prefix + QStringLiteral("_geometry_source_mask.bin");
        const QString inverseDepthMeanPath =
            prefix + QStringLiteral("_inverse_depth_mean.bin");
        const QString inverseDepthSpreadPath =
            prefix + QStringLiteral("_inverse_depth_spread.bin");
        const QString repairedMaskPath =
            prefix + QStringLiteral("_cross_view_repaired_mask.png");
        const QString depthMaskPath = prefix + QStringLiteral("_mask.png");
        const QString supportMaskPath = prefix + QStringLiteral("_support_mask.png");
        const cv::Mat depth(24, 32, CV_32FC1, cv::Scalar(2.0f));
        const cv::Mat confidence(24, 32, CV_32FC1, cv::Scalar(0.9f));
        const cv::Mat geometrySupport(24, 32, CV_16UC1, cv::Scalar(4));
        const cv::Mat geometrySourceMask(24, 32, CV_16UC1, cv::Scalar(0x0005));
        const cv::Mat inverseDepthMean(24, 32, CV_32FC1, cv::Scalar(0.5f));
        const cv::Mat inverseDepthSpread(24, 32, CV_32FC1, cv::Scalar(0.01f));
        const cv::Mat repairedMask(24, 32, CV_8UC1, cv::Scalar(0));
        const cv::Mat depthMask(24, 32, CV_8UC1, cv::Scalar(255));
        const cv::Mat supportMask(24, 32, CV_8UC1, cv::Scalar(255));
        ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(depthPath, depth).ok);
        ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(confidencePath, confidence).ok);
        ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(
            geometrySupportPath, geometrySupport).ok);
        ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(
            geometrySourceMaskPath, geometrySourceMask).ok);
        ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(
            inverseDepthMeanPath, inverseDepthMean).ok);
        ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(
            inverseDepthSpreadPath, inverseDepthSpread).ok);
        ASSERT_TRUE(cv::imwrite(depthMaskPath.toStdString(), depthMask));
        ASSERT_TRUE(cv::imwrite(supportMaskPath.toStdString(), supportMask));
        ASSERT_TRUE(cv::imwrite(repairedMaskPath.toStdString(), repairedMask));

        xjw::Camera camera;
        camera.setIntrinsics(30.0, 30.0, 16.0, 12.0);
        camera.setPose(std::array<double, 9>{1.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0,
                                             0.0, 0.0, 1.0},
                       std::array<double, 3>{0.05 * index, 0.0, 0.0});

        xjw::mesh::DepthFrameArtifact artifact;
        artifact.refIndex = index;
        artifact.status = QStringLiteral("completed");
        artifact.acceptance = index == 3
            ? QStringLiteral("validation_only")
            : QStringLiteral("accepted");
        artifact.fusionEligible = true;
        artifact.depthPath = depthPath;
        artifact.confidencePath = confidencePath;
        artifact.geometrySupportPath = geometrySupportPath;
        if (index != 2)
        {
            artifact.geometrySourceMaskPath = geometrySourceMaskPath;
            artifact.inverseDepthMeanPath = inverseDepthMeanPath;
            artifact.inverseDepthSpreadPath = inverseDepthSpreadPath;
            artifact.crossViewRepairedMaskPath = repairedMaskPath;
            artifact.sourceIndices = {1, 2, 3};
        }
        artifact.validMaskPath = depthMaskPath;
        artifact.supportMaskPath = supportMaskPath;
        artifact.cameraModel = camera;
        artifact.hasCameraModel = true;
        artifacts.push_back(artifact);
    }

    const auto loaded = xjw::mesh::DepthTsdfSurfaceBuilder::loadFrames(artifacts);
    ASSERT_TRUE(loaded.ok) << loaded.errorMessage.toStdString();
    ASSERT_EQ(loaded.frames.size(), 3);
    EXPECT_EQ(loaded.frames.front().depth.type(), CV_32FC1);
    EXPECT_EQ(loaded.frames.front().confidence.type(), CV_32FC1);
    EXPECT_EQ(loaded.frames.front().geometrySupportCount.type(), CV_16UC1);
    EXPECT_EQ(loaded.frames.front().geometrySupportCount.at<std::uint16_t>(0, 0), 4);
    EXPECT_EQ(loaded.frames.front().geometrySourceMask.at<std::uint16_t>(0, 0), 0x0005);
    EXPECT_FLOAT_EQ(loaded.frames.front().inverseDepthMean.at<float>(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(
        loaded.frames.front().inverseDepthRelativeSpread.at<float>(0, 0), 0.01f);
    EXPECT_EQ(loaded.frames.front().crossViewRepairedMask.type(), CV_8UC1);
    EXPECT_EQ(loaded.frames.front().sourceIndices, QVector<int>({1, 2, 3}));
    EXPECT_EQ(cv::countNonZero(loaded.frames.at(2).geometrySourceMask), 0);
    EXPECT_EQ(cv::countNonZero(loaded.frames.at(2).inverseDepthMean), 0);
    EXPECT_EQ(cv::countNonZero(loaded.frames.at(2).inverseDepthRelativeSpread), 0);
    EXPECT_EQ(cv::countNonZero(loaded.frames.at(2).crossViewRepairedMask), 0);
    EXPECT_EQ(loaded.frames.front().depthValidMask.type(), CV_8UC1);
    EXPECT_EQ(loaded.frames.front().supportMask.type(), CV_8UC1);
    EXPECT_TRUE(loaded.frames.front().camera.isValid());


    const auto bounds = xjw::mesh::DepthTsdfSurfaceBuilder::estimateBounds(loaded.frames);
    ASSERT_TRUE(bounds.ok) << bounds.errorMessage.toStdString();
    EXPECT_LT(bounds.minimum[2], 2.0f);
    EXPECT_GT(bounds.maximum[2], 2.0f);
}

TEST(DepthTsdfSurfaceBuilderTest, BuildsFiniteSurfaceFromConfidenceWeightedPlane)
{
    const QVector<xjw::mesh::DepthTsdfFrame> frames = makeSyntheticPlaneFrames(false);
    xjw::mesh::DepthTsdfOptions options;
    options.resolution = 48;
    options.calculateVertexColors = false;
    options.workerCount = 2;
    options.availableMemoryBytes = 512ull * 1024ull * 1024ull;
    std::atomic_bool reported_integration_progress{false};
    options.progress = [&reported_integration_progress](const QString &, int percent)
    {
        if (percent > 5 && percent < 75)
        {
            reported_integration_progress.store(true, std::memory_order_relaxed);
        }
    };

    const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::build(frames, options);
    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    EXPECT_TRUE(reported_integration_progress.load(std::memory_order_relaxed));
    EXPECT_GT(result.statistics.integratedVoxelUpdates, 0u);
    EXPECT_GT(result.statistics.supportedSampleCount, 0u);
    EXPECT_FALSE(result.mesh.empty());
    EXPECT_EQ(result.statistics.vertexCount, result.mesh.vertexCount());
    EXPECT_EQ(result.statistics.faceCount, result.mesh.faceCount());
    EXPECT_GE(result.statistics.componentCount, 1);
    EXPECT_GE(result.statistics.largestComponentFaceRatio, 0.9);
    for (const auto &vertex : result.mesh.vertices)
    {
        EXPECT_TRUE(std::isfinite(vertex.x));
        EXPECT_TRUE(std::isfinite(vertex.y));
        EXPECT_TRUE(std::isfinite(vertex.z));
        const float normalLength = std::sqrt(vertex.nx * vertex.nx +
                                             vertex.ny * vertex.ny +
                                             vertex.nz * vertex.nz);
        EXPECT_TRUE(std::isfinite(normalLength));
        EXPECT_NEAR(normalLength, 1.0f, 1.0e-3f);
    }
}

TEST(DepthTsdfSurfaceBuilderTest, RecoversOneVoxelSurfacePatchWithStableSourceEvidence)
{
    QVector<xjw::mesh::DepthTsdfFrame> frames = makeSyntheticPlaneFrames(false);
    for (int index = 0; index < frames.size(); ++index)
    {
        xjw::mesh::DepthTsdfFrame &frame = frames[index];
        for (int row = 0; row < frame.depth.rows; ++row)
        {
            for (int column = 0; column < frame.depth.cols; ++column)
            {
                frame.depth.at<float>(row, column) =
                    2.0f + 0.005f * static_cast<float>(column - 24);
            }
        }
        frame.sourceIndices = {0, 1, 2};
        frame.geometrySourceMask = cv::Mat(
            frame.depth.size(), CV_16UC1, cv::Scalar(0x0003));
        frame.inverseDepthRelativeSpread = cv::Mat(
            frame.depth.size(), CV_32FC1, cv::Scalar(0.005f));
        if (index > 0)
        {
            frame.depthValidMask.colRange(22, 30).setTo(0);
        }
    }
    xjw::mesh::DepthTsdfOptions baseline_options;
    baseline_options.resolution = 48;
    baseline_options.calculateVertexColors = false;
    baseline_options.availableMemoryBytes = 512ull * 1024ull * 1024ull;
    const auto baseline = xjw::mesh::DepthTsdfSurfaceBuilder::build(
        frames, baseline_options);
    ASSERT_TRUE(baseline.ok) << baseline.errorMessage.toStdString();

    xjw::mesh::DepthTsdfOptions patch_options = baseline_options;
    patch_options.enableSurfacePatchSupport = true;
    patch_options.maximumSurfacePatchInverseDepthSpread = 0.01f;
    patch_options.maximumSurfacePatchNormalAngleDegrees = 20.0f;
    patch_options.minimumGeometryVerifiedObservationWeight = 0.10f;
    const auto patched = xjw::mesh::DepthTsdfSurfaceBuilder::build(
        frames, patch_options);

    ASSERT_TRUE(patched.ok) << patched.errorMessage.toStdString();
    EXPECT_GT(patched.statistics.surfacePatchRecoveredSampleCount, 0U)
        << "normal=" << patched.statistics.surfacePatchRejectedNormalCount
        << " source=" << patched.statistics.surfacePatchRejectedSourceOverlapCount
        << " spread=" << patched.statistics.surfacePatchRejectedDepthSpreadCount
        << " free=" << patched.statistics.surfacePatchRejectedFreeSpaceCount
        << " singleRejected="
        << patched.statistics.rejectedSingleObservationWeightCount
        << " considered=" << patched.statistics.surfacePatchConsideredSampleCount
        << " weight=" << patched.statistics.surfacePatchRejectedWeightCount
        << " multi=" << patched.statistics.multiViewSupportedSampleCount;
    EXPECT_TRUE(patched.statistics.effectiveSurfacePatchSupport);
    EXPECT_EQ(patched.statistics.surfacePatchCreatedComponentCount, 0);
    EXPECT_GE(patched.statistics.supportedSampleCount,
              baseline.statistics.supportedSampleCount);
    EXPECT_GE(patched.mesh.faceCount(), baseline.mesh.faceCount());
}

TEST(DepthTsdfSurfaceBuilderTest, SeparatesSupportDepthValidAndConfidenceRejections)
{
    const QVector<xjw::mesh::DepthTsdfFrame> frames = makeSyntheticPlaneFrames(true);
    xjw::mesh::DepthTsdfOptions options;
    options.resolution = 48;
    options.calculateVertexColors = false;
    options.availableMemoryBytes = 512ull * 1024ull * 1024ull;

    const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::build(frames, options);
    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    EXPECT_GT(result.statistics.rejectedConfidenceCount, 0u);
    EXPECT_GT(result.statistics.rejectedDepthValidCount, 0u);
    EXPECT_GT(result.statistics.rejectedSupportMaskCount, 0u);
    EXPECT_EQ(result.statistics.supportMaskFreeSpaceUpdateCount, 0u);
    EXPECT_FALSE(result.mesh.empty());
    const QJsonObject statistics =
        xjw::mesh::DepthTsdfSurfaceBuilder::statisticsToJson(result);
    EXPECT_TRUE(statistics.contains(QStringLiteral("rejected_projection_count")));
    EXPECT_TRUE(statistics.contains(QStringLiteral("rejected_depth_valid_count")));
    EXPECT_TRUE(statistics.contains(QStringLiteral("single_view_supported_sample_count")));
    EXPECT_TRUE(statistics.contains(QStringLiteral("effective_minimum_single_observation_weight")));
    EXPECT_FLOAT_EQ(statistics.value(QStringLiteral("effective_truncation_voxels")).toDouble(),
                    7.5);
    EXPECT_FLOAT_EQ(
        statistics.value(QStringLiteral("effective_maximum_free_space_voxels")).toDouble(),
        36.0);
    EXPECT_EQ(statistics.value(QStringLiteral("support_mask_free_space_update_count"))
                  .toInteger(),
              0);

    options.enableSupportMaskFreeSpaceCarving = true;
    const auto carved = xjw::mesh::DepthTsdfSurfaceBuilder::build(frames, options);
    ASSERT_TRUE(carved.ok) << carved.errorMessage.toStdString();
    EXPECT_GT(carved.statistics.supportMaskFreeSpaceUpdateCount, 0u);

    options.minimumSupportMaskFreeSpaceViews = static_cast<int>(frames.size()) + 1;
    const auto consensus_guarded = xjw::mesh::DepthTsdfSurfaceBuilder::build(frames, options);
    ASSERT_TRUE(consensus_guarded.ok) << consensus_guarded.errorMessage.toStdString();
    EXPECT_EQ(consensus_guarded.statistics.supportMaskFreeSpaceUpdateCount, 0u);
    EXPECT_EQ(consensus_guarded.statistics.effectiveMinimumSupportMaskFreeSpaceViews,
              static_cast<int>(frames.size()) + 1);
}

TEST(DepthTsdfSurfaceBuilderTest, RobustVertexColorRejectsAnInconsistentCamera)
{
    QVector<xjw::mesh::DepthTsdfFrame> frames = makeSyntheticPlaneFrames(false);
    frames[0].colorBgr = cv::Mat(36, 48, CV_8UC3, cv::Scalar(240, 240, 240));
    frames[1].colorBgr = cv::Mat(36, 48, CV_8UC3, cv::Scalar(10, 20, 30));
    frames[2].colorBgr = cv::Mat(36, 48, CV_8UC3, cv::Scalar(10, 20, 30));
    xjw::mesh::DepthTsdfOptions options;
    options.resolution = 48;
    options.calculateVertexColors = true;
    options.availableMemoryBytes = 512ull * 1024ull * 1024ull;

    const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::build(frames, options);
    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    ASSERT_FALSE(result.mesh.empty());
    EXPECT_TRUE(result.mesh.hasVertexColors);
    EXPECT_GT(result.statistics.colorCandidateObservationCount, 0u);
    EXPECT_GT(result.statistics.colorRejectedOutlierCount, 0u);
    EXPECT_GT(result.statistics.reliablyColoredVertexCount, 0);
    EXPECT_TRUE(std::any_of(result.mesh.vertices.begin(),
                            result.mesh.vertices.end(),
                            [](const xjw::mesh::MeshVertex &vertex)
                            {
                                return vertex.r == 30 && vertex.g == 20 && vertex.b == 10;
                            }));
    const QJsonObject statistics =
        xjw::mesh::DepthTsdfSurfaceBuilder::statisticsToJson(result);
    EXPECT_EQ(statistics.value(QStringLiteral("reliably_colored_vertex_count")).toInt(),
              result.statistics.reliablyColoredVertexCount);
    EXPECT_TRUE(statistics.contains(QStringLiteral("fallback_color_vertex_count")));
}

TEST(MeshColorizerTest, ExposureCompensationAlsoAppliesToBestViewFallback)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(3);
    mesh.vertices[0].x = -0.1f; mesh.vertices[0].y = -0.1f; mesh.vertices[0].z = 2.0f;
    mesh.vertices[1].x = 0.1f; mesh.vertices[1].y = -0.1f; mesh.vertices[1].z = 2.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 0.1f; mesh.vertices[2].z = 2.0f;
    for (auto &vertex : mesh.vertices)
    {
        vertex.nz = 1.0f;
    }
    xjw::mesh::Triangle face;
    face.v[0] = 0; face.v[1] = 1; face.v[2] = 2;
    mesh.faces.push_back(face);

    auto make_view = [](std::uint8_t value)
    {
        xjw::mesh::MeshColorView view;
        view.camera.setIntrinsics(40.0, 40.0, 8.0, 8.0);
        view.camera.setPose(std::array<double, 9>{1.0, 0.0, 0.0,
                                                   0.0, 1.0, 0.0,
                                                   0.0, 0.0, 1.0},
                            std::array<double, 3>{0.0, 0.0, 0.0});
        view.colorBgr = cv::Mat(16, 16, CV_8UC3, cv::Scalar(value, value, value));
        view.depth = cv::Mat(16, 16, CV_32FC1, cv::Scalar(2.0f));
        view.confidence = cv::Mat(16, 16, CV_32FC1, cv::Scalar(0.9f));
        view.depthValidMask = cv::Mat(16, 16, CV_8UC1, cv::Scalar(255));
        view.supportMask = cv::Mat(16, 16, CV_8UC1, cv::Scalar(255));
        return view;
    };

    xjw::mesh::MeshColorOptions options;
    options.maximumVoxelSize = 0.01f;
    options.minimumViewCosine = 1.1f;
    options.minimumConsistentViews = 2;
    options.propagationPasses = 0;
    options.speckleCleanupPasses = 0;
    options.compensateExposure = true;

    const auto statistics = xjw::mesh::MeshColorizer::colorize(
        &mesh, QVector<xjw::mesh::MeshColorView>{make_view(50), make_view(100)}, options);

    EXPECT_EQ(statistics.bestViewFallbackVertexCount, 3);
    for (const auto &vertex : mesh.vertices)
    {
        EXPECT_NEAR(vertex.r, 67, 1);
        EXPECT_NEAR(vertex.g, 67, 1);
        EXPECT_NEAR(vertex.b, 67, 1);
    }
}

TEST(MeshColorizerTest, CoherentFaceColorUsesOnePrimaryViewForTheWholeFace)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(3);
    mesh.vertices[0].x = -0.1f; mesh.vertices[0].y = -0.1f; mesh.vertices[0].z = 2.0f;
    mesh.vertices[1].x = 0.1f; mesh.vertices[1].y = -0.1f; mesh.vertices[1].z = 2.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 0.1f; mesh.vertices[2].z = 2.0f;
    for (auto &vertex : mesh.vertices)
    {
        vertex.nz = 1.0f;
    }
    xjw::mesh::Triangle face;
    face.v[0] = 0; face.v[1] = 1; face.v[2] = 2;
    mesh.faces.push_back(face);

    auto make_view = [](const cv::Scalar &color, float quality)
    {
        xjw::mesh::MeshColorView view;
        view.camera.setIntrinsics(40.0, 40.0, 8.0, 8.0);
        view.camera.setPose(std::array<double, 9>{1.0, 0.0, 0.0,
                                                   0.0, 1.0, 0.0,
                                                   0.0, 0.0, 1.0},
                            std::array<double, 3>{0.0, 0.0, 0.0});
        view.colorBgr = cv::Mat(16, 16, CV_8UC3, color);
        view.depth = cv::Mat(16, 16, CV_32FC1, cv::Scalar(2.0f));
        view.confidence = cv::Mat(16, 16, CV_32FC1, cv::Scalar(0.9f));
        view.depthValidMask = cv::Mat(16, 16, CV_8UC1, cv::Scalar(255));
        view.supportMask = cv::Mat(16, 16, CV_8UC1, cv::Scalar(255));
        view.qualityWeight = quality;
        return view;
    };

    xjw::mesh::MeshColorOptions options;
    options.maximumVoxelSize = 0.01f;
    options.propagationPasses = 0;
    options.speckleCleanupPasses = 0;
    options.coherentFacePrimaryViews = true;
    const auto statistics = xjw::mesh::MeshColorizer::colorize(
        &mesh,
        QVector<xjw::mesh::MeshColorView>{
            make_view(cv::Scalar(10, 20, 30), 0.5f),
            make_view(cv::Scalar(100, 150, 200), 1.0f)},
        options);

    EXPECT_EQ(statistics.coherentPrimaryViewFaceCount, 1);
    EXPECT_EQ(statistics.coherentPrimaryViewVertexCount, 3);
    for (const auto &vertex : mesh.vertices)
    {
        EXPECT_EQ(vertex.r, 200);
        EXPECT_EQ(vertex.g, 150);
        EXPECT_EQ(vertex.b, 100);
    }
}

namespace
{

std::filesystem::path writeNoNormalsPointCloud(const std::filesystem::path &root)
{
    namespace fs = std::filesystem;
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "dense_no_normals.ply";

    constexpr int N = 24;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(N * N, 3);
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            const int row = y * N + x;
            const float fx = (static_cast<float>(x) - N * 0.5f) / static_cast<float>(N);
            const float fy = (static_cast<float>(y) - N * 0.5f) / static_cast<float>(N);
            points(row, 0) = fx;
            points(row, 1) = fy;
            points(row, 2) = 0.08f * std::sin(fx * 7.0f) + 0.05f * std::cos(fy * 9.0f);
        }
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);
    return plyPath;
}

std::filesystem::path writeSpherePointCloudWithNormals(const std::filesystem::path &root,
                                                       bool alternateNormalDirection = false,
                                                       bool injectInvalidNormals = false,
                                                       bool varyColors = false)
{
    namespace fs = std::filesystem;
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "sphere_with_normals.ply";

    constexpr int rings = 16;
    constexpr int segments = 16;
    constexpr int pointCount = rings * segments;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(pointCount, 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(pointCount, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(pointCount, 3);

    int row = 0;
    for (int ring = 0; ring < rings; ++ring)
    {
        const float phi = static_cast<float>(ring + 1) * static_cast<float>(M_PI) /
                          static_cast<float>(rings + 1);
        for (int segment = 0; segment < segments; ++segment)
        {
            const float theta = static_cast<float>(segment) * 2.0f * static_cast<float>(M_PI) /
                                static_cast<float>(segments);
            const float nx = std::sin(phi) * std::cos(theta);
            const float ny = std::sin(phi) * std::sin(theta);
            const float nz = std::cos(phi);
            points(row, 0) = 2.0f * nx;
            points(row, 1) = 2.0f * ny;
            points(row, 2) = 2.0f * nz;
            const float normal_sign = alternateNormalDirection && (row % 2 != 0) ? -1.0f : 1.0f;
            normals(row, 0) = normal_sign * nx;
            normals(row, 1) = normal_sign * ny;
            normals(row, 2) = normal_sign * nz;
            if (varyColors && nx < 0.0f)
            {
                colors(row, 0) = 240;
                colors(row, 1) = 40;
                colors(row, 2) = 30;
            }
            else if (varyColors)
            {
                colors(row, 0) = 30;
                colors(row, 1) = 60;
                colors(row, 2) = 240;
            }
            else
            {
                colors(row, 0) = 180;
                colors(row, 1) = 190;
                colors(row, 2) = 210;
            }
            ++row;
        }
    }

    if (injectInvalidNormals)
    {
        for (int invalid_index = 0; invalid_index < 32; ++invalid_index)
        {
            normals(invalid_index, 0) = 0.0f;
            normals(invalid_index, 1) = 0.0f;
            normals(invalid_index, 2) = 0.0f;
        }
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    cloud.setNormals(std::move(normals));
    cloud.setColors(std::move(colors));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);
    return plyPath;
}

std::filesystem::path writeDenseGridPointCloud(const std::filesystem::path &root)
{
    namespace fs = std::filesystem;
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "dense_grid.ply";

    constexpr int N = 32;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(N * N, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(N * N, 3);
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            const int row = y * N + x;
            const float fx = static_cast<float>(x) / static_cast<float>(N - 1);
            const float fy = static_cast<float>(y) / static_cast<float>(N - 1);
            points(row, 0) = fx;
            points(row, 1) = fy;
            points(row, 2) = 0.04f * std::sin(fx * 8.0f) + 0.03f * std::cos(fy * 6.0f);
            colors(row, 0) = static_cast<std::uint8_t>(50 + x * 4);
            colors(row, 1) = static_cast<std::uint8_t>(80 + y * 4);
            colors(row, 2) = 160;
        }
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    cloud.setColors(std::move(colors));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);
    return plyPath;
}

std::filesystem::path writeFlatGridPointCloudWithSparseVerticalSpikes(const std::filesystem::path &root)
{
    namespace fs = std::filesystem;
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "flat_grid_with_spikes.ply";

    constexpr int N = 24;
    constexpr int baseSamples = 5;
    constexpr int spikeSamples = 1;
    constexpr int perCellSamples = baseSamples + spikeSamples;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(N * N * perCellSamples, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(N * N * perCellSamples, 3);

    int row = 0;
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            const float fx = static_cast<float>(x) / static_cast<float>(N - 1);
            const float fy = static_cast<float>(y) / static_cast<float>(N - 1);
            for (int sample = 0; sample < baseSamples; ++sample)
            {
                points(row, 0) = fx + 0.0002f * static_cast<float>(sample);
                points(row, 1) = fy;
                points(row, 2) = 0.0f;
                colors(row, 0) = 80;
                colors(row, 1) = 160;
                colors(row, 2) = 80;
                ++row;
            }
            const bool hasSpike = (x % 6 == 0) && (y % 6 == 0);
            points(row, 0) = fx;
            points(row, 1) = fy;
            points(row, 2) = hasSpike ? 4.0f : 0.0f;
            colors(row, 0) = 220;
            colors(row, 1) = 80;
            colors(row, 2) = 80;
            ++row;
        }
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    cloud.setColors(std::move(colors));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);
    return plyPath;
}

xjw::mesh::ReconstructionConfig fallbackMeshConfig()
{
    xjw::mesh::ReconstructionConfig config;
    config.forcePoisson = true;
    config.poissonDepth = 12;
    config.resolution = 48;
    config.enableDenoise = false;
    config.enableDownsample = false;
    config.preprocessingDevice = plapoint::ProcessingDevice::CPU;
    config.cleanSmallComponents = false;
    config.smoothIterations = 0;
    config.holeFillPasses = 2;
    return config;
}

xjw::Camera makeLookAtCamera(const std::array<float, 3> &center)
{
    auto normalize = [](std::array<float, 3> value)
    {
        const float length = std::sqrt(value[0] * value[0] +
                                       value[1] * value[1] +
                                       value[2] * value[2]);
        for (float &component : value)
        {
            component /= length;
        }
        return value;
    };
    auto cross = [](const std::array<float, 3> &lhs, const std::array<float, 3> &rhs)
    {
        return std::array<float, 3>{
            lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[0] * rhs[1] - lhs[1] * rhs[0]};
    };

    const std::array<float, 3> forward = normalize({-center[0], -center[1], -center[2]});
    const std::array<float, 3> provisional_up = std::fabs(forward[1]) > 0.9f
        ? std::array<float, 3>{0.0f, 0.0f, 1.0f}
        : std::array<float, 3>{0.0f, 1.0f, 0.0f};
    const std::array<float, 3> right = normalize(cross(forward, provisional_up));
    const std::array<float, 3> down = normalize(cross(forward, right));

    const std::array<std::array<float, 3>, 3> rows{right, down, forward};
    std::array<double, 9> cameraToWorld{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            cameraToWorld[static_cast<std::size_t>(column * 3 + row)] =
                rows[static_cast<std::size_t>(row)][column];
        }
    }
    xjw::Camera camera;
    camera.setIntrinsics(100.0, 100.0, 64.0, 64.0);
    camera.setPose(cameraToWorld,
                   {center[0], center[1], center[2]});
    return camera;
}

} // namespace

TEST(VisualHullReconstructorTest, ReconstructsClosedBodyFromSixSilhouettes)
{
    const std::array<std::array<float, 3>, 6> centers{{
        {{3.0f, 0.0f, 0.0f}}, {{-3.0f, 0.0f, 0.0f}},
        {{0.0f, 3.0f, 0.0f}}, {{0.0f, -3.0f, 0.0f}},
        {{0.0f, 0.0f, 3.0f}}, {{0.0f, 0.0f, -3.0f}}}};

    std::vector<xjw::mesh::VisualHullView> views;
    for (const auto &center : centers)
    {
        xjw::mesh::VisualHullView view;
        view.camera = makeLookAtCamera(center);
        view.silhouetteMask = cv::Mat::zeros(128, 128, CV_8UC1);
        cv::circle(view.silhouetteMask, cv::Point(64, 64), 21, cv::Scalar(255), cv::FILLED);
        views.push_back(std::move(view));
    }

    xjw::mesh::VisualHullConfig config;
    config.boundsMin = {-1.0f, -1.0f, -1.0f};
    config.boundsMax = {1.0f, 1.0f, 1.0f};
    config.resolution = 32;
    config.minimumVisibleViews = 4;
    config.allowedSilhouetteViolations = 0;

    xjw::mesh::TriMesh mesh;
    std::string error;
    ASSERT_TRUE(xjw::mesh::VisualHullReconstructor::reconstruct(views, config, &mesh, &error)) << error;
    ASSERT_FALSE(mesh.empty());
    EXPECT_GT(mesh.vertexCount(), 500);
    EXPECT_GT(mesh.faceCount(), 500);

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    for (const xjw::mesh::MeshVertex &vertex : mesh.vertices)
    {
        min_x = std::min(min_x, vertex.x);
        max_x = std::max(max_x, vertex.x);
    }
    EXPECT_LT(min_x, -0.45f);
    EXPECT_GT(max_x, 0.45f);
    EXPECT_GT(min_x, -0.85f);
    EXPECT_LT(max_x, 0.85f);
}

TEST(VisualHullReconstructorTest, KeepsDepthCarvingDisabledByDefault)
{
    const xjw::mesh::VisualHullConfig config;
    EXPECT_FALSE(config.enableDepthFreeSpaceCarving);
}

TEST(VisualHullReconstructorTest, DetectsFragmentedMeshForSafeRetry)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(6);
    mesh.faces = {
        {{0, 1, 2}},
        {{3, 4, 5}}
    };

    const xjw::mesh::MeshConnectivityStats stats =
        xjw::mesh::VisualHullReconstructor::analyzeConnectivity(mesh);
    EXPECT_EQ(stats.componentCount, 2);
    EXPECT_DOUBLE_EQ(stats.largestComponentFaceRatio, 0.5);
    ASSERT_EQ(stats.componentFaceCounts.size(), 2);
    EXPECT_EQ(stats.componentFaceCounts[0], 1);
    EXPECT_EQ(stats.componentFaceCounts[1], 1);
    ASSERT_EQ(stats.components.size(), 2);
    EXPECT_EQ(stats.components[0].faceCount, 1);
    EXPECT_EQ(stats.components[1].faceCount, 1);
    EXPECT_TRUE(xjw::mesh::VisualHullReconstructor::requiresSilhouetteOnlyRetry(
        stats, 0.85, 12));

    EXPECT_TRUE(xjw::mesh::VisualHullReconstructor::retainLargestConnectedComponent(&mesh));
    EXPECT_EQ(mesh.faces.size(), 1);
    EXPECT_EQ(mesh.vertices.size(), 3);
    const xjw::mesh::MeshConnectivityStats cleaned =
        xjw::mesh::VisualHullReconstructor::analyzeConnectivity(mesh);
    EXPECT_EQ(cleaned.componentCount, 1);
    EXPECT_DOUBLE_EQ(cleaned.largestComponentFaceRatio, 1.0);
}

TEST(MeshReconstructorTest, UsesStreamingHeightGridForOversizedPly)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_streaming_input_cap_test";
    const fs::path plyPath = writeDenseGridPointCloud(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.forcePoisson = false;
    config.resolution = 32;
    config.maxInputPointsForMeshing = 300;
    config.streamingThreads = 3;
    config.streamingChunkBytes = 256;
    std::vector<std::string> progressMessages;
    config.progressFn = [&](const std::string &stage, float) {
        progressMessages.push_back(stage);
    };

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(algorithmUsed, "streaming_tiled_height_grid");
    EXPECT_GT(mesh.faceCount(), 0);
    const auto streamed = std::find_if(progressMessages.begin(), progressMessages.end(), [](const std::string &msg) {
        return msg.find("并行分块") != std::string::npos &&
               msg.find("1024") != std::string::npos &&
               msg.find("3") != std::string::npos;
    });
    EXPECT_NE(streamed, progressMessages.end());
    const auto tiled = std::find_if(progressMessages.begin(), progressMessages.end(), [](const std::string &msg) {
        return msg.find("瓦片") != std::string::npos &&
               msg.find("2x2") != std::string::npos;
    });
    EXPECT_NE(tiled, progressMessages.end());
}

TEST(MeshReconstructorTest, OversizedArbitraryCloudUsesBoundedPoissonSample)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_bounded_poisson_sample_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;
    config.allowHeightGridFallback = false;
    config.orientNormalsForClosedSurface = true;
    config.enableDenoise = false;
    config.enableDownsample = false;
    config.maxInputPointsForMeshing = 240;
    std::vector<std::string> progressMessages;
    config.progressFn = [&](const std::string &stage, float) {
        progressMessages.push_back(stage);
    };

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(algorithmUsed, "poisson");
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_NE(std::find_if(progressMessages.begin(), progressMessages.end(), [](const std::string &message) {
                  return message.find("Poisson") != std::string::npos
                      && message.find("抽样") != std::string::npos;
              }),
              progressMessages.end());
}

TEST(MeshReconstructorTest, StreamingHeightGridSuppressesSparseVerticalSpikes)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_streaming_spike_test";
    const fs::path plyPath = writeFlatGridPointCloudWithSparseVerticalSpikes(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.forcePoisson = false;
    config.resolution = 32;
    config.maxInputPointsForMeshing = 300;
    config.streamingThreads = 2;
    config.streamingChunkBytes = 512;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    ASSERT_FALSE(mesh.vertices.empty());
    EXPECT_EQ(algorithmUsed, "streaming_tiled_height_grid");

    float maxZ = -std::numeric_limits<float>::max();
    for (const auto &vertex : mesh.vertices)
    {
        maxZ = std::max(maxZ, vertex.z);
    }
    EXPECT_LT(maxZ, 0.4f)
        << "Sparse vertical outliers should not become visible terrain spikes.";
}

TEST(MeshReconstructorTest, ForcePoissonUsesInputNormalsWhenPresent)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_normals_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.vertexCount(), 0);
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "poisson");
}

TEST(MeshReconstructorTest, PoissonTransfersSourceColorsToMeshVertices)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_color_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root, false, false, true);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;
    config.allowHeightGridFallback = false;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    ASSERT_FALSE(mesh.vertices.empty());
    EXPECT_EQ(algorithmUsed, "poisson");

    int red_dominant = 0;
    int blue_dominant = 0;
    for (const auto &vertex : mesh.vertices)
    {
        red_dominant += vertex.r > vertex.b ? 1 : 0;
        blue_dominant += vertex.b > vertex.r ? 1 : 0;
    }
    EXPECT_GT(red_dominant, 0);
    EXPECT_GT(blue_dominant, 0);
}

TEST(MeshReconstructorTest, PoissonDepthAdaptsToPointCloudDensity)
{
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(4000, 8), 6);
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(20000, 8), 7);
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(50000, 8), 8);
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(250000, 8), 8);
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(1000000, 8), 8);
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(50000, 5), 5);
}

TEST(MeshReconstructorTest, PoissonInputDropsOnlyPointsWithInvalidNormals)
{
    std::vector<xjw::mesh::detail::PointXYZRGB> points(3);
    for (auto &point : points)
    {
        point.hasNormal = true;
        point.nz = 2.0f;
    }
    points[1].nx = 0.0f;
    points[1].ny = 0.0f;
    points[1].nz = 0.0f;
    points[2].nx = std::numeric_limits<float>::quiet_NaN();

    EXPECT_EQ(xjw::mesh::detail::removeInvalidPoissonPoints(&points), 2u);
    ASSERT_EQ(points.size(), 1u);
    EXPECT_NEAR(points.front().nz, 1.0f, 1.0e-6f);
}

TEST(MeshReconstructorTest, PoissonInputRemovesDisconnectedPointFragments)
{
    std::vector<xjw::mesh::detail::PointXYZRGB> points;
    points.reserve(228);
    for (int z = 0; z < 6; ++z)
    {
        for (int y = 0; y < 6; ++y)
        {
            for (int x = 0; x < 6; ++x)
            {
                xjw::mesh::detail::PointXYZRGB point;
                point.x = 0.05f * static_cast<float>(x);
                point.y = 0.05f * static_cast<float>(y);
                point.z = 0.05f * static_cast<float>(z);
                point.hasNormal = true;
                point.nz = 1.0f;
                point.r = 40;
                point.g = 120;
                point.b = 220;
                points.push_back(point);
            }
        }
    }
    for (int index = 0; index < 12; ++index)
    {
        xjw::mesh::detail::PointXYZRGB point;
        point.x = 5.0f + 0.02f * static_cast<float>(index % 3);
        point.y = 0.02f * static_cast<float>((index / 3) % 2);
        point.z = 0.02f * static_cast<float>(index / 6);
        point.hasNormal = true;
        point.nz = 1.0f;
        point.r = 220;
        point.g = 40;
        point.b = 30;
        points.push_back(point);
    }

    const auto stats = xjw::mesh::detail::removeSmallPoissonPointComponents(
        &points, 32, 4.0f);

    EXPECT_EQ(stats.componentCount, 2u);
    EXPECT_EQ(stats.removedComponentCount, 1u);
    EXPECT_EQ(stats.removedPointCount, 12u);
    ASSERT_EQ(points.size(), 216u);
    EXPECT_TRUE(std::all_of(points.begin(), points.end(), [](const auto &point)
    {
        return point.r == 40 && point.g == 120 && point.b == 220 && point.hasNormal;
    }));
}

TEST(MeshReconstructorTest, PoissonOrientsMixedNormalsForClosedSurface)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_mixed_normals_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root, true);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;
    config.allowHeightGridFallback = false;
    config.orientNormalsForClosedSurface = true;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "poisson");
}

TEST(MeshReconstructorTest, PoissonRepairsInvalidNormalsBeforeReconstruction)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_invalid_normals_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root, false, true);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;
    config.allowHeightGridFallback = false;
    config.orientNormalsForClosedSurface = true;
    config.enableDenoise = false;
    config.enableDownsample = false;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "poisson");
}

TEST(MeshReconstructorTest, SmallComponentCleanupKeepsLargestComponent)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(7);
    mesh.vertices[0].x = 0.0f; mesh.vertices[0].y = 0.0f; mesh.vertices[0].z = 0.0f;
    mesh.vertices[1].x = 1.0f; mesh.vertices[1].y = 0.0f; mesh.vertices[1].z = 0.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 1.0f; mesh.vertices[2].z = 0.0f;
    mesh.vertices[3].x = 1.0f; mesh.vertices[3].y = 1.0f; mesh.vertices[3].z = 0.0f;
    mesh.vertices[4].x = 10.0f; mesh.vertices[4].y = 0.0f; mesh.vertices[4].z = 0.0f;
    mesh.vertices[5].x = 11.0f; mesh.vertices[5].y = 0.0f; mesh.vertices[5].z = 0.0f;
    mesh.vertices[6].x = 10.0f; mesh.vertices[6].y = 1.0f; mesh.vertices[6].z = 0.0f;

    xjw::mesh::Triangle firstA;
    firstA.v[0] = 0; firstA.v[1] = 1; firstA.v[2] = 2;
    xjw::mesh::Triangle secondA;
    secondA.v[0] = 1; secondA.v[1] = 3; secondA.v[2] = 2;
    xjw::mesh::Triangle firstB;
    firstB.v[0] = 4; firstB.v[1] = 5; firstB.v[2] = 6;
    mesh.faces = {firstA, secondA, firstB};

    xjw::mesh::detail::removeSmallConnectedComponents(&mesh, 10);

    EXPECT_EQ(mesh.faceCount(), 2);
    EXPECT_EQ(mesh.vertexCount(), 4);
}

TEST(MeshReconstructorTest, SmallComponentCleanupUsesLargestComponentRelativeThreshold)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(10);
    mesh.faces = {
        xjw::mesh::Triangle{{0, 1, 2}},
        xjw::mesh::Triangle{{1, 3, 2}},
        xjw::mesh::Triangle{{1, 4, 3}},
        xjw::mesh::Triangle{{1, 5, 4}},
        xjw::mesh::Triangle{{6, 7, 8}},
        xjw::mesh::Triangle{{7, 9, 8}},
    };

    xjw::mesh::detail::removeSmallConnectedComponents(&mesh, 2, 0.75f);

    EXPECT_EQ(mesh.faceCount(), 4);
    EXPECT_EQ(mesh.vertexCount(), 6);
}

TEST(MeshReconstructorTest, SmallBoundaryHoleFillingClosesTetrahedronOpening)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(4);
    mesh.vertices[0].x = 0.0f; mesh.vertices[0].y = 0.0f; mesh.vertices[0].z = 0.0f;
    mesh.vertices[1].x = 1.0f; mesh.vertices[1].y = 0.0f; mesh.vertices[1].z = 0.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 1.0f; mesh.vertices[2].z = 0.0f;
    mesh.vertices[3].x = 0.0f; mesh.vertices[3].y = 0.0f; mesh.vertices[3].z = 1.0f;
    mesh.faces = {
        xjw::mesh::Triangle{{0, 1, 3}},
        xjw::mesh::Triangle{{1, 2, 3}},
        xjw::mesh::Triangle{{2, 0, 3}},
    };

    const int filled = xjw::mesh::detail::fillSmallBoundaryHoles(&mesh, 8);

    EXPECT_EQ(filled, 1);
    EXPECT_EQ(mesh.vertexCount(), 5);
    EXPECT_EQ(mesh.faceCount(), 6);
}

TEST(MeshReconstructorTest, BoundaryHoleDiameterLimitPreservesLargeOpening)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(4);
    mesh.vertices[0].x = 0.0f; mesh.vertices[0].y = 0.0f; mesh.vertices[0].z = 0.0f;
    mesh.vertices[1].x = 10.0f; mesh.vertices[1].y = 0.0f; mesh.vertices[1].z = 0.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 10.0f; mesh.vertices[2].z = 0.0f;
    mesh.vertices[3].x = 0.0f; mesh.vertices[3].y = 0.0f; mesh.vertices[3].z = 10.0f;
    mesh.faces = {
        xjw::mesh::Triangle{{0, 1, 3}},
        xjw::mesh::Triangle{{1, 2, 3}},
        xjw::mesh::Triangle{{2, 0, 3}},
    };

    const int filled = xjw::mesh::detail::fillSmallBoundaryHoles(&mesh, 8, 2.0f);

    EXPECT_EQ(filled, 0);
    EXPECT_EQ(mesh.vertexCount(), 4);
    EXPECT_EQ(mesh.faceCount(), 3);
}

TEST(MeshPostprocessTest, WeldsCoincidentVerticesBeforeComponentFiltering)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(6);
    mesh.vertices[0].x = 0.0f; mesh.vertices[0].y = 0.0f;
    mesh.vertices[1].x = 1.0f; mesh.vertices[1].y = 0.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 1.0f;
    mesh.vertices[3].x = 1.0f; mesh.vertices[3].y = 0.0f;
    mesh.vertices[4].x = 1.0f; mesh.vertices[4].y = 1.0f;
    mesh.vertices[5].x = 0.0f; mesh.vertices[5].y = 1.0f;

    xjw::mesh::Triangle first;
    first.v[0] = 0; first.v[1] = 1; first.v[2] = 2;
    xjw::mesh::Triangle second;
    second.v[0] = 3; second.v[1] = 4; second.v[2] = 5;
    mesh.faces = {first, second};

    xjw::mesh::detail::weldCoincidentVertices(&mesh, 1.0e-6f);
    xjw::mesh::detail::removeSmallConnectedComponents(&mesh, 2);

    EXPECT_EQ(mesh.vertexCount(), 4);
    EXPECT_EQ(mesh.faceCount(), 2);
}

TEST(MeshReconstructorTest, WeldedPoissonMeshSurvivesOversizedComponentThreshold)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_tiny_cleanup_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.cleanSmallComponents = true;
    config.minComponentFaces = 100000;
    config.poissonDepth = 4;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(algorithmUsed, "poisson");
    EXPECT_GT(mesh.faceCount(), 100);
}

TEST(MeshReconstructorTest, ForcePoissonFallsBackWhenCloudHasNoNormals)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_no_normals_test";
    const fs::path plyPath = writeNoNormalsPointCloud(root);

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    bool ok = false;
    ASSERT_NO_THROW({
        ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                            fallbackMeshConfig(),
                                                                            mesh,
                                                                            &error,
                                                                            &algorithmUsed);
    });

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.vertexCount(), 0);
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "height_grid");
}

TEST(MeshReconstructorTest, Arbitrary3dEstimatesMissingNormalsBeforePoisson)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_no_arbitrary_fallback_test";
    const fs::path plyPath = writeNoNormalsPointCloud(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.allowHeightGridFallback = false;
    config.orientNormalsForClosedSurface = true;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.vertexCount(), 0);
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "poisson");
}

TEST(MeshReconstructorTest, PoissonFallbackProgressIncludesFailureReason)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_reason_test";
    const fs::path plyPath = writeNoNormalsPointCloud(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    std::vector<std::string> progressMessages;
    config.progressFn = [&](const std::string &stage, float) {
        progressMessages.push_back(stage);
    };

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(algorithmUsed, "height_grid");
    const auto it = std::find_if(progressMessages.begin(), progressMessages.end(), [](const std::string &message) {
        return message.find("Poisson 重建失败(") != std::string::npos &&
               message.find("改用高度格网") != std::string::npos;
    });
    EXPECT_NE(it, progressMessages.end());
}

TEST(MeshWorkflowServiceTest, RecordsActualFallbackAlgorithmInPayload)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_workflow_algorithm_test";
    const fs::path plyPath = writeNoNormalsPointCloud(root / "input");

    xjw::mesh::workflow::MeshBuildRequest request;
    request.pointCloudPath = QString::fromStdString(plyPath.string());
    request.outputRoot = QString::fromStdString((root / "model").string());
    request.reconstruction = fallbackMeshConfig();
    request.exportObj = false;

    const auto result = xjw::mesh::workflow::buildMeshAndOptionalTexture(request);
    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    EXPECT_EQ(result.payload.value(QStringLiteral("mesh_algorithm")).toString().toStdString(), "height_grid");
    EXPECT_TRUE(fs::exists(result.payload.value(QStringLiteral("model_ply")).toString().toStdString()));
}

TEST(MeshWorkflowServiceTest, SharedModelEntryMapsSettingsAndBuildsPointCloudSource)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_shared_model_entry_test";
    const fs::path plyPath = writeDenseGridPointCloud(root);

    xjw::mesh::workflow::ModelBuildRequest request;
    request.sourceData = QStringLiteral("point_cloud");
    request.requestedSourcePath = QString::fromStdString(plyPath.string());
    request.sourcePointCloudPath = request.requestedSourcePath;
    request.outputRoot = QString::fromStdString((root / "model").string());
    request.settings[QStringLiteral("surface_type")] = QStringLiteral("height_field");
    request.settings[QStringLiteral("method")] = QStringLiteral("Height Grid");
    request.settings[QStringLiteral("octreeDepth")] = 7;
    request.settings[QStringLiteral("cleanSmall")] = false;
    request.settings[QStringLiteral("smoothIter")] = 0;
    request.settings[QStringLiteral("depthFiltering")] = QStringLiteral("disabled");
    request.settings[QStringLiteral("qualityProfile")] = QStringLiteral("detail");

    const auto result = xjw::mesh::workflow::buildModel(request);

    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    EXPECT_EQ(result.payload.value(QStringLiteral("source_data")).toString(),
              QStringLiteral("point_cloud"));
    EXPECT_EQ(result.payload.value(QStringLiteral("source_point_cloud_path")).toString(),
              request.sourcePointCloudPath);
    EXPECT_TRUE(fs::exists(result.payload.value(QStringLiteral("mesh_ply")).toString().toStdString()));
}

TEST(MeshWorkflowSettingsTest, HeightFieldSourceDisablesPoissonAndMapsFiltering)
{
    QJsonObject settings;
    settings[QStringLiteral("surface_type")] = QStringLiteral("height_field");
    settings[QStringLiteral("quality")] = QStringLiteral("ultra");
    settings[QStringLiteral("qualityProfile")] = QStringLiteral("detail");
    settings[QStringLiteral("meshResolution")] = 384;
    settings[QStringLiteral("octreeDepth")] = 11;
    settings[QStringLiteral("targetFaces")] = 0;
    settings[QStringLiteral("interpolation")] = QStringLiteral("disabled");
    settings[QStringLiteral("depthFiltering")] = QStringLiteral("aggressive");

    const auto config = xjw::mesh::workflow::reconstructionConfigFromModelSettings(settings);

    EXPECT_FALSE(config.forcePoisson);
    EXPECT_GE(config.resolution, 384);
    EXPECT_FALSE(config.fillHoles);
    EXPECT_TRUE(config.enableDenoise);
    EXPECT_LE(config.denoiseStdMul, 0.95f);
    EXPECT_EQ(config.simplifyTargetFaces, 0);
}

TEST(MeshWorkflowSettingsTest, ArbitrarySourceKeepsPoissonAndHonorsFaceBudget)
{
    QJsonObject settings;
    settings[QStringLiteral("surface_type")] = QStringLiteral("arbitrary_3d");
    settings[QStringLiteral("quality")] = QStringLiteral("high");
    settings[QStringLiteral("qualityProfile")] = QStringLiteral("detail");
    settings[QStringLiteral("meshResolution")] = 320;
    settings[QStringLiteral("octreeDepth")] = 10;
    settings[QStringLiteral("targetFaces")] = 240000;
    settings[QStringLiteral("interpolation")] = QStringLiteral("extrapolated");
    settings[QStringLiteral("depthFiltering")] = QStringLiteral("mild");

    const auto config = xjw::mesh::workflow::reconstructionConfigFromModelSettings(settings);

    EXPECT_TRUE(config.forcePoisson);
    EXPECT_FALSE(config.allowHeightGridFallback);
    EXPECT_TRUE(config.orientNormalsForClosedSurface);
    EXPECT_GE(config.poissonDepth, 10);
    EXPECT_TRUE(config.fillHoles);
    EXPECT_GE(config.holeFillPasses, 16);
    EXPECT_EQ(config.simplifyTargetFaces, 240000);
    EXPECT_EQ(config.maxInputPointsForMeshing, 400000);
    EXPECT_FALSE(config.enableDownsample);
    EXPECT_GE(config.kNormals, 18);
    EXPECT_GE(config.smoothIterations, 4);
}

TEST(MeshWorkflowSettingsTest, DepthMapsDefaultToDepthTsdf)
{
    const QJsonObject settings{
        {QStringLiteral("source_data"), QStringLiteral("depth_maps")},
        {QStringLiteral("surface_type"), QStringLiteral("arbitrary_3d")},
        {QStringLiteral("meshResolution"), 320}};

    EXPECT_EQ(xjw::mesh::workflow::depthReconstructionModeFromSettings(settings),
              QStringLiteral("depth_tsdf"));
}

TEST(MeshWorkflowSettingsTest, DepthTsdfSupportThresholdIsConfigurable)
{
    const QJsonObject settings{
        {QStringLiteral("tsdfMinimumDistinctCameraSupport"), 1},
        {QStringLiteral("tsdfSurfaceSupportBandVoxels"), 2.5},
        {QStringLiteral("tsdfMinimumSupportMaskFreeSpaceViews"), 3},
        {QStringLiteral("tsdfMinimumComponentFaceRatio"), 0.025}};

    const auto options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 320);

    EXPECT_EQ(options.minimumDistinctCameraSupport, 1);
    EXPECT_FLOAT_EQ(options.surfaceSupportBandVoxels, 2.5f);
    EXPECT_EQ(options.minimumSupportMaskFreeSpaceViews, 3);
    EXPECT_FLOAT_EQ(options.minimumComponentFaceRatio, 0.025f);
    EXPECT_FLOAT_EQ(options.minimumSingleObservationWeight, 0.70f);
    EXPECT_FALSE(options.enableSupportMaskFreeSpaceCarving);
    EXPECT_FALSE(options.compensateColorExposure);
    EXPECT_FLOAT_EQ(options.maximumFreeSpaceVoxels, 36.0f);
    EXPECT_TRUE(options.fillSmallBoundaryHoles);
    EXPECT_EQ(options.boundarySmoothingIterations, 1);
}

TEST(MeshWorkflowSettingsTest, DepthTsdfObservationEdgeGatesAreConfigurable)
{
    const QJsonObject settings{
        {QStringLiteral("tsdfMaximumObservationInverseDepthSpread"), 0.03},
        {QStringLiteral("tsdfAllowInvalidNearestPixelRecovery"), false},
        {QStringLiteral("tsdfMaximumInvalidNearestPixelRecoveryInverseDepthSpread"), 0.015},
        {QStringLiteral("tsdfCrossViewConsensusDepth"), true},
        {QStringLiteral("tsdfMaximumCrossViewConsensusInverseDepthSpread"), 0.025},
        {QStringLiteral("tsdfGeometrySingleViewNeighborhoodGuard"), true},
        {QStringLiteral("tsdfMinimumGeometrySingleViewNeighborCount"), 4},
        {QStringLiteral("tsdfGeometrySingleViewGrowthPasses"), 3},
        {QStringLiteral("tsdfMaximumGeometrySingleViewNeighborTsdfDelta"), 0.2}};

    const auto options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 384);

    EXPECT_FLOAT_EQ(options.maximumObservationInverseDepthSpread, 0.03f);
    EXPECT_FALSE(options.allowInvalidNearestPixelRecovery);
    EXPECT_FLOAT_EQ(options.maximumInvalidNearestPixelRecoveryInverseDepthSpread, 0.015f);
    EXPECT_TRUE(options.enableCrossViewConsensusDepth);
    EXPECT_FLOAT_EQ(options.maximumCrossViewConsensusInverseDepthSpread, 0.025f);
    EXPECT_TRUE(options.enableGeometrySingleViewNeighborhoodGuard);
    EXPECT_EQ(options.minimumGeometrySingleViewNeighborCount, 4);
    EXPECT_EQ(options.geometrySingleViewGrowthPasses, 3);
    EXPECT_FLOAT_EQ(options.maximumGeometrySingleViewNeighborTsdfDelta, 0.2f);
}

TEST(MeshWorkflowSettingsTest, DepthTsdfExposureCompensationIsExplicitOptIn)
{
    const QJsonObject settings{
        {QStringLiteral("tsdfCompensateColorExposure"), true}};

    const auto options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 320);

    EXPECT_TRUE(options.compensateColorExposure);
}

TEST(MeshWorkflowSettingsTest, DepthTsdfQuadricSimplificationIsExplicitOptIn)
{
    const QJsonObject settings{
        {QStringLiteral("tsdfQuadricSimplification"), true},
        {QStringLiteral("simplifyTargetFaces"), 240000}};

    const auto options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 384);

    EXPECT_TRUE(options.enableQuadricSimplification);
    EXPECT_EQ(options.simplifyTargetFaces, 240000);
}

TEST(MeshWorkflowSettingsTest, DepthTsdfCoherentFaceColorsAreExplicitOptIn)
{
    const QJsonObject settings{
        {QStringLiteral("tsdfCoherentFacePrimaryViewColors"), true}};

    const auto options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 384);

    EXPECT_TRUE(options.coherentFacePrimaryViewColors);
}

TEST(MeshWorkflowSettingsTest, UltraEnablesOnlyGeometryVerifiedSingleObservations)
{
    const auto high = xjw::mesh::workflow::depthTsdfOptionsFromSettings(
        QJsonObject{}, 320);
    const auto ultra = xjw::mesh::workflow::depthTsdfOptionsFromSettings(
        QJsonObject{}, 384);

    EXPECT_FALSE(high.allowGeometryVerifiedSingleObservation);
    EXPECT_FALSE(high.enableDiscontinuityAwareSampling);
    EXPECT_TRUE(ultra.allowGeometryVerifiedSingleObservation);
    EXPECT_TRUE(ultra.enableDiscontinuityAwareSampling);
    EXPECT_FLOAT_EQ(ultra.maximumInterpolationRelativeDepthSpread, 0.02f);
    EXPECT_FALSE(ultra.allowInvalidNearestPixelRecovery);
    EXPECT_TRUE(high.allowInvalidNearestPixelRecovery);
    EXPECT_EQ(ultra.minimumGeometrySupportCount, 3);
    EXPECT_EQ(high.minimumGeometrySupportCount, 4);
    EXPECT_FLOAT_EQ(ultra.minimumGeometryVerifiedObservationWeight, 0.85f);
    EXPECT_EQ(ultra.minimumDistinctCameraSupport, 2);
}

TEST(MeshWorkflowSettingsTest, UltraDepthTsdfRequiresThreeCameraSupportByDefault)
{
    const auto ultra_options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(
        QJsonObject{}, 384);
    const auto high_options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(
        QJsonObject{}, 320);

    EXPECT_EQ(ultra_options.minimumDistinctCameraSupport, 2);
    EXPECT_EQ(high_options.minimumDistinctCameraSupport, 2);
    EXPECT_EQ(ultra_options.boundarySmoothingIterations, 2);
    EXPECT_EQ(high_options.boundarySmoothingIterations, 1);
    EXPECT_EQ(ultra_options.depthValidBoundaryErosionPixels, 2);
    EXPECT_EQ(high_options.depthValidBoundaryErosionPixels, 1);
    EXPECT_TRUE(ultra_options.trimWeakBoundaryTips);
    EXPECT_FALSE(high_options.trimWeakBoundaryTips);
    EXPECT_EQ(ultra_options.weakBoundaryTipTrimPasses, 2);
    EXPECT_EQ(high_options.weakBoundaryTipTrimPasses, 1);

    const QJsonObject disabled_settings{
        {QStringLiteral("tsdfTrimWeakBoundaryTips"), false}};
    const auto disabled_ultra_options =
        xjw::mesh::workflow::depthTsdfOptionsFromSettings(disabled_settings, 384);
    EXPECT_FALSE(disabled_ultra_options.trimWeakBoundaryTips);
}

TEST(MeshWorkflowSettingsTest, DepthTsdfBoundaryErosionIsBoundedAndConfigurable)
{
    QJsonObject settings{{QStringLiteral("tsdfDepthValidBoundaryErosionPixels"), 9}};
    auto options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 384);
    EXPECT_EQ(options.depthValidBoundaryErosionPixels, 4);

    settings[QStringLiteral("tsdfDepthValidBoundaryErosionPixels")] = 1;
    options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 384);
    EXPECT_EQ(options.depthValidBoundaryErosionPixels, 1);
}

TEST(MeshWorkflowSettingsTest, DepthTsdfInterpolationControlsBoundedHoleFilling)
{
    QJsonObject settings{
        {QStringLiteral("interpolation"), QStringLiteral("disabled")},
        {QStringLiteral("maxHoleSize"), 100.0}
    };
    auto options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 320);
    EXPECT_FALSE(options.fillSmallBoundaryHoles);

    settings[QStringLiteral("interpolation")] = QStringLiteral("enabled");
    options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 320);
    EXPECT_TRUE(options.fillSmallBoundaryHoles);
    EXPECT_EQ(options.maximumHoleBoundaryEdges, 16);
    EXPECT_FLOAT_EQ(options.maximumHoleDiameterVoxels, 4.0f);

    settings[QStringLiteral("interpolation")] = QStringLiteral("extrapolated");
    options = xjw::mesh::workflow::depthTsdfOptionsFromSettings(settings, 320);
    EXPECT_TRUE(options.fillSmallBoundaryHoles);
    EXPECT_GE(options.maximumHoleBoundaryEdges, 64);
    EXPECT_FLOAT_EQ(options.maximumHoleDiameterVoxels, 16.0f);
}

TEST(DepthMapMeshBuilderTest, TsdfFailureDoesNotFallBackToVisualHullOrDenseCloud)
{
    xjw::mesh::workflow::DepthMapMeshBuildRequest request;
    request.depthMapSourcePath = QStringLiteral("E:/missing/depth");
    request.settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("depth_tsdf");
    request.reconstruction.resolution = 320;

    const auto result = xjw::mesh::workflow::buildMeshFromDepthMaps(request);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.payload.value(QStringLiteral("actual_mesh_algorithm")).toString(),
              QStringLiteral("depth_tsdf"));
    EXPECT_FALSE(result.payload.contains(QStringLiteral("source_point_cloud_path")));
    EXPECT_FALSE(result.payload.contains(QStringLiteral("visual_hull_fallback_reason")));
}

TEST(MeshWorkflowSettingsTest, DenseAerialSceneUsesDetailedHeightFieldPolicy)
{
    const auto config = xjw::mesh::workflow::reconstructionConfigForDenseScene(
        224,
        true,
        true);

    EXPECT_FALSE(config.forcePoisson);
    EXPECT_TRUE(config.allowHeightGridFallback);
    EXPECT_FALSE(config.orientNormalsForClosedSurface);
    EXPECT_FALSE(config.enableDownsample);
    EXPECT_GE(config.resolution, 320);
    EXPECT_GE(config.simplifyTargetFaces, 65000);
}

TEST(MeshWorkflowSettingsTest, DenseOrbitalSceneKeepsPoissonPolicy)
{
    const auto config = xjw::mesh::workflow::reconstructionConfigForDenseScene(
        224,
        false,
        true);

    EXPECT_TRUE(config.forcePoisson);
    EXPECT_FALSE(config.allowHeightGridFallback);
    EXPECT_TRUE(config.orientNormalsForClosedSurface);
    EXPECT_EQ(config.resolution, 224);
}

TEST(MeshWorkflowSettingsTest, DepthMapMeshRequestPreservesSourceAndSettings)
{
    xjw::mesh::workflow::DepthMapMeshBuildRequest request;
    request.depthMapSourcePath = QStringLiteral("E:/tmp/mvs_output");
    request.reusableDenseCloudPath = QStringLiteral("E:/tmp/mvs_output/dense_cloud.ply");
    request.outputRoot = QStringLiteral("E:/tmp/model");
    request.settings[QStringLiteral("surface_type")] = QStringLiteral("height_field");
    request.settings[QStringLiteral("quality")] = QStringLiteral("high");
    request.settings[QStringLiteral("reuseDepthMaps")] = true;

    EXPECT_EQ(request.depthMapSourcePath, QStringLiteral("E:/tmp/mvs_output"));
    EXPECT_EQ(request.reusableDenseCloudPath, QStringLiteral("E:/tmp/mvs_output/dense_cloud.ply"));
    EXPECT_EQ(request.outputRoot, QStringLiteral("E:/tmp/model"));
    EXPECT_TRUE(request.settings.value(QStringLiteral("reuseDepthMaps")).toBool());
}

TEST(DepthMapMeshBuilderTest, DiscoversDepthFramesFromOutputDirectory)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_discovery_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream(root / "depth_001.bin").put('\0');
    std::ofstream(root / "depth_001_conf.bin").put('\0');
    std::ofstream(root / "depth_001_geometry_source_mask.bin").put('\0');
    std::ofstream(root / "depth_001_inverse_depth_mean.bin").put('\0');
    std::ofstream(root / "depth_001_inverse_depth_spread.bin").put('\0');
    std::ofstream(root / "depth_001.png").put('\0');
    std::ofstream(root / "depth_001_mask.png").put('\0');
    std::ofstream(root / "depth_002.bin").put('\0');

    const auto frames =
        xjw::mesh::DepthMapMeshBuilder::discoverDepthFrames(QString::fromStdString(root.string()));

    ASSERT_EQ(frames.size(), 2);
    EXPECT_TRUE(frames.at(0).depthPath.endsWith(QStringLiteral("depth_001.bin")));
    EXPECT_TRUE(frames.at(0).confidencePath.endsWith(QStringLiteral("depth_001_conf.bin")));
    EXPECT_TRUE(frames.at(0).previewPath.endsWith(QStringLiteral("depth_001.png")));
    EXPECT_TRUE(frames.at(0).validMaskPath.endsWith(QStringLiteral("depth_001_mask.png")));
    EXPECT_TRUE(frames.at(1).depthPath.endsWith(QStringLiteral("depth_002.bin")));
}

TEST(DepthMapMeshBuilderTest, LoadsDepthGridCameraFromWorkspaceManifest)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_manifest_camera_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream(root / "depth_0.bin").put('\0');
    std::ofstream manifest(root / "mvs_manifest.json");
    manifest << R"({
        "frames": [{
            "ref_index": 0,
            "status": "completed",
            "scene_profile": "orbital_object",
            "ref_image": "frame.png",
            "raw_depth_path": "depth_0.bin",
            "raw_geometry_source_mask_path": "depth_0_geometry_source_mask.bin",
            "raw_inverse_depth_mean_path": "depth_0_inverse_depth_mean.bin",
            "raw_inverse_depth_spread_path": "depth_0_inverse_depth_spread.bin",
            "cross_view_repaired_mask_path": "depth_0_cross_view_repaired_mask.png",
            "source_indices": [3, 7],
            "grid_width": 320,
            "grid_height": 240,
            "camera_model": {
                "fx": 250.0,
                "fy": 252.0,
                "cx": 160.0,
                "cy": 120.0,
                "rotation_world_to_camera": [1,0,0,0,1,0,0,0,1],
                "translation_world_to_camera": [0,0,3],
                "camera_center": [0,0,-3]
            }
        }]
    })";
    manifest.close();

    const auto frames =
        xjw::mesh::DepthMapMeshBuilder::discoverDepthFrames(QString::fromStdString(root.string()));

    ASSERT_EQ(frames.size(), 1);
    EXPECT_TRUE(frames.front().hasCameraModel);
    EXPECT_DOUBLE_EQ(frames.front().cameraModel.focalX(), 250.0);
    EXPECT_DOUBLE_EQ(frames.front().cameraModel.principalY(), 120.0);
    EXPECT_EQ(frames.front().gridWidth, 320);
    EXPECT_EQ(frames.front().gridHeight, 240);
    EXPECT_EQ(frames.front().sceneProfile, QStringLiteral("orbital_object"));
    EXPECT_TRUE(frames.front().geometrySourceMaskPath.endsWith(
        QStringLiteral("depth_0_geometry_source_mask.bin")));
    EXPECT_TRUE(frames.front().inverseDepthMeanPath.endsWith(
        QStringLiteral("depth_0_inverse_depth_mean.bin")));
    EXPECT_TRUE(frames.front().inverseDepthSpreadPath.endsWith(
        QStringLiteral("depth_0_inverse_depth_spread.bin")));
    EXPECT_TRUE(frames.front().crossViewRepairedMaskPath.endsWith(
        QStringLiteral("depth_0_cross_view_repaired_mask.png")));
    EXPECT_EQ(frames.front().sourceIndices, QVector<int>({3, 7}));
}

TEST(DepthMapMeshBuilderTest, DoesNotTreatFullFrameAerialImagesAsStudioSilhouettes)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_aerial_branch_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream manifest(root / "mvs_manifest.json");
    manifest << "{\"frames\":[";
    for (int index = 0; index < 6; ++index)
    {
        const std::string image_name = "aerial_" + std::to_string(index) + ".png";
        const std::string depth_name = "depth_" + std::to_string(index) + ".bin";
        cv::Mat aerial_image(48, 64, CV_8UC3, cv::Scalar(70, 150, 90));
        ASSERT_TRUE(cv::imwrite((root / image_name).string(), aerial_image));
        std::ofstream(root / depth_name).put('\0');
        if (index > 0)
        {
            manifest << ',';
        }
        manifest << "{\"ref_index\":" << index
                 << ",\"status\":\"completed\",\"ref_image\":\"" << image_name
                 << "\",\"raw_depth_path\":\"" << depth_name
                 << "\",\"grid_width\":64,\"grid_height\":48,\"camera_model\":{"
                    "\"fx\":50,\"fy\":50,\"cx\":32,\"cy\":24,"
                    "\"rotation_world_to_camera\":[1,0,0,0,1,0,0,0,1],"
                    "\"translation_world_to_camera\":[0,0,3],"
                    "\"camera_center\":[0,0,-3]}}";
    }
    manifest << "]}";
    manifest.close();

    const auto result = xjw::mesh::DepthMapMeshBuilder::buildVisualHull(
        QString::fromStdString(root.string()), 96);

    EXPECT_FALSE(result.applicable);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.usableViewCount, 0);
}

TEST(DepthMapMeshBuilderTest, UsesExistingDenseCloudWhenPresent)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_existing_dense_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path densePath = writeDenseGridPointCloud(root);
    fs::rename(densePath, root / "dense_cloud.ply");

    QString error;
    const QString resolved = xjw::mesh::DepthMapMeshBuilder::resolveReusableDenseCloud(
        QString::fromStdString(root.string()),
        &error);

    EXPECT_TRUE(error.isEmpty());
    EXPECT_TRUE(resolved.endsWith(QStringLiteral("dense_cloud.ply")));
}

TEST(DepthMapMeshBuilderTest, ReportsActionableErrorWhenDepthFrameMetadataIsMissing)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_no_dense_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream(root / "depth_001.bin").put('\0');

    xjw::mesh::workflow::DepthMapMeshBuildRequest request;
    request.depthMapSourcePath = QString::fromStdString(root.string());
    request.outputRoot = QString::fromStdString(root.string());
    request.reconstruction = fallbackMeshConfig();

    const auto result = xjw::mesh::workflow::buildMeshFromDepthMaps(request);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.payload.value(QStringLiteral("actual_mesh_algorithm")).toString(),
              QStringLiteral("depth_tsdf"));
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("camera is invalid")));
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("depth_001.bin")));
    EXPECT_FALSE(result.payload.contains(QStringLiteral("source_point_cloud_path")));
}

TEST(TextureMapperTest, ReadsPlyMeshFacesForTextureMapping)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_texture_mapper_ply_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "mesh_with_faces.ply";

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points(0, 0) = 0.0f; points(0, 1) = 0.0f; points(0, 2) = 0.0f;
    points(1, 0) = 1.0f; points(1, 1) = 0.0f; points(1, 2) = 0.0f;
    points(2, 0) = 0.0f; points(2, 1) = 1.0f; points(2, 2) = 0.0f;
    points(3, 0) = 1.0f; points(3, 1) = 1.0f; points(3, 2) = 0.0f;

    plapoint::PointCloud<float, plamatrix::Device::CPU> meshCloud(std::move(points));

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(4, 3);
    colors(0, 0) = 255; colors(0, 1) = 0;   colors(0, 2) = 0;
    colors(1, 0) = 0;   colors(1, 1) = 255; colors(1, 2) = 0;
    colors(2, 0) = 0;   colors(2, 1) = 0;   colors(2, 2) = 255;
    colors(3, 0) = 255; colors(3, 1) = 255; colors(3, 2) = 255;
    meshCloud.setColors(std::move(colors));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 3);
    faces(0, 0) = 0; faces(0, 1) = 1; faces(0, 2) = 2;
    faces(1, 0) = 1; faces(1, 1) = 3; faces(1, 2) = 2;
    meshCloud.setFaces(std::move(faces));
    plapoint::io::writePly<float>(plyPath.string(), meshCloud, plapoint::io::PlyFormat::BinaryLE);

    xjw::mesh::TextureMappingConfig config;
    config.textureSize = 512;
    xjw::mesh::TextureMappingResult result;
    std::string error;
    ASSERT_TRUE(xjw::mesh::TextureMapper::generateTexturedModelFromMeshFile(plyPath.string(),
                                                                            root.string(),
                                                                            config,
                                                                            &result,
                                                                            &error))
        << error;
    EXPECT_TRUE(fs::exists(result.modelObjPath));
    EXPECT_TRUE(fs::exists(result.modelMtlPath));
    EXPECT_TRUE(fs::exists(result.texturePngPath));

    std::ifstream objFile(result.modelObjPath);
    std::stringstream objBuffer;
    objBuffer << objFile.rdbuf();
    EXPECT_NE(objBuffer.str().find("usemtl material0"), std::string::npos);

    std::ifstream mtlFile(result.modelMtlPath);
    std::stringstream mtlBuffer;
    mtlBuffer << mtlFile.rdbuf();
    EXPECT_NE(mtlBuffer.str().find("map_Kd textures/model_texture.png"), std::string::npos);
}

TEST(TextureMapperTest, CameraAtlasUsesPerFaceProjectedUvWithoutPlanarOverlap)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_camera_texture_mapper_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path ply_path = root / "mesh.ply";

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points(0, 0) = -0.4f; points(0, 1) = -0.3f; points(0, 2) = 2.0f;
    points(1, 0) = 0.4f;  points(1, 1) = -0.3f; points(1, 2) = 2.0f;
    points(2, 0) = -0.4f; points(2, 1) = 0.3f;  points(2, 2) = 2.0f;
    points(3, 0) = 0.4f;  points(3, 1) = 0.3f;  points(3, 2) = 2.0f;
    plapoint::PointCloud<float, plamatrix::Device::CPU> mesh(std::move(points));
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 3);
    faces(0, 0) = 0; faces(0, 1) = 1; faces(0, 2) = 2;
    faces(1, 0) = 1; faces(1, 1) = 3; faces(1, 2) = 2;
    mesh.setFaces(std::move(faces));
    plapoint::io::writePly<float>(ply_path.string(), mesh, plapoint::io::PlyFormat::BinaryLE);

    xjw::mesh::MeshColorView view;
    view.camera.setIntrinsics(40.0, 40.0, 24.0, 18.0);
    view.camera.setPose(std::array<double, 9>{1.0, 0.0, 0.0,
                                              0.0, 1.0, 0.0,
                                              0.0, 0.0, 1.0},
                        std::array<double, 3>{0.0, 0.0, 0.0});
    view.colorBgr = cv::Mat(36, 48, CV_8UC3, cv::Scalar(10, 80, 220));
    view.depth = cv::Mat(36, 48, CV_32FC1, cv::Scalar(2.0f));
    view.confidence = cv::Mat(36, 48, CV_32FC1, cv::Scalar(0.9f));
    view.depthValidMask = cv::Mat(36, 48, CV_8UC1, cv::Scalar(255));
    view.supportMask = cv::Mat(36, 48, CV_8UC1, cv::Scalar(255));

    xjw::mesh::TextureMappingConfig config;
    config.textureSize = 1024;
    xjw::mesh::TextureMappingResult result;
    std::string error;
    ASSERT_TRUE(xjw::mesh::TextureMapper::generateCameraTexturedModelFromMeshFile(
        ply_path.string(), root.string(), config, QVector<xjw::mesh::MeshColorView>{view},
        &result, &error)) << error;
    EXPECT_EQ(result.textureAlgorithm, "camera_projected_atlas_v2");
    EXPECT_EQ(result.uvMethod, "indexed_vertex_view_projection");
    EXPECT_EQ(result.sourceViewCount, 1);
    EXPECT_EQ(result.mappedFaceCount, 2);
    EXPECT_EQ(result.fallbackMappedFaceCount, 0);
    EXPECT_EQ(result.unmappedFaceCount, 0);
    EXPECT_TRUE(fs::exists(result.modelObjPath));
    EXPECT_TRUE(fs::exists(result.texturePngPath));

    const auto textured_mesh = plapoint::io::readObj<float>(result.modelObjPath);
    ASSERT_TRUE(textured_mesh != nullptr);
    ASSERT_TRUE(textured_mesh->hasTextureCoords());
    ASSERT_TRUE(textured_mesh->hasFaceTextureIndices());
    EXPECT_LT(textured_mesh->textureCoords()->rows(),
              textured_mesh->faces()->rows() * 3);

    std::ifstream obj_file(result.modelObjPath);
    std::stringstream obj_buffer;
    obj_buffer << obj_file.rdbuf();
    EXPECT_NE(obj_buffer.str().find("vt "), std::string::npos);
    EXPECT_NE(obj_buffer.str().find("usemtl material0"), std::string::npos);
}

TEST(TextureMapperTest, CameraAtlasDoesNotSampleBackgroundAcrossMaskedFaceCorner)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        "plascan_camera_texture_masked_corner_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path ply_path = root / "mesh.ply";

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(3, 3);
    points(0, 0) = -0.4f; points(0, 1) = -0.3f; points(0, 2) = 2.0f;
    points(1, 0) = 0.4f;  points(1, 1) = -0.3f; points(1, 2) = 2.0f;
    points(2, 0) = -0.4f; points(2, 1) = 0.3f;  points(2, 2) = 2.0f;
    plapoint::PointCloud<float, plamatrix::Device::CPU> mesh(std::move(points));
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces(0, 0) = 0; faces(0, 1) = 1; faces(0, 2) = 2;
    mesh.setFaces(std::move(faces));
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(3, 3);
    for (int row = 0; row < 3; ++row)
    {
        colors(row, 0) = 180;
        colors(row, 1) = 130;
        colors(row, 2) = 80;
    }
    mesh.setColors(std::move(colors));
    plapoint::io::writePly<float>(
        ply_path.string(), mesh, plapoint::io::PlyFormat::BinaryLE);

    xjw::mesh::MeshColorView view;
    view.camera.setIntrinsics(40.0, 40.0, 24.0, 18.0);
    view.camera.setPose(std::array<double, 9>{1.0, 0.0, 0.0,
                                              0.0, 1.0, 0.0,
                                              0.0, 0.0, 1.0},
                        std::array<double, 3>{0.0, 0.0, 0.0});
    view.colorBgr = cv::Mat(36, 48, CV_8UC3, cv::Scalar(0, 0, 0));
    view.depth = cv::Mat(36, 48, CV_32FC1, cv::Scalar(2.0f));
    view.confidence = cv::Mat(36, 48, CV_32FC1, cv::Scalar(0.9f));
    view.depthValidMask = cv::Mat(36, 48, CV_8UC1, cv::Scalar(255));
    view.supportMask = cv::Mat(36, 48, CV_8UC1, cv::Scalar(255));
    view.supportMask.at<std::uint8_t>(12, 16) = 0;

    xjw::mesh::TextureMappingConfig config;
    config.textureSize = 1024;
    xjw::mesh::TextureMappingResult result;
    std::string error;
    ASSERT_TRUE(xjw::mesh::TextureMapper::generateCameraTexturedModelFromMeshFile(
        ply_path.string(), root.string(), config,
        QVector<xjw::mesh::MeshColorView>{view}, &result, &error)) << error;
    EXPECT_EQ(result.mappedFaceCount, 0);
    EXPECT_EQ(result.unmappedFaceCount, 1);

    const cv::Mat atlas = cv::imread(result.texturePngPath, cv::IMREAD_COLOR);
    ASSERT_FALSE(atlas.empty());
    const cv::Vec3b fallback = atlas.at<cv::Vec3b>(2, 2);
    EXPECT_GT(fallback[0], 0);
    EXPECT_GT(fallback[1], 0);
    EXPECT_GT(fallback[2], 0);
}
