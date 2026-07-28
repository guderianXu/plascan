#include "MeshFaceOrientation.h"

#include <gtest/gtest.h>

TEST(MeshFaceOrientationTest, RepairsSameDirectionSharedEdge)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(4);
    mesh.faces = {
        {{0, 1, 2}},
        {{1, 2, 3}}};

    const xjw::mesh::MeshFaceOrientationStatistics statistics =
        xjw::mesh::repairMeshFaceOrientation(&mesh);

    EXPECT_TRUE(statistics.succeeded);
    EXPECT_EQ(statistics.sharedEdgeCount, 1);
    EXPECT_EQ(statistics.inconsistentSharedEdgeCountBefore, 1);
    EXPECT_EQ(statistics.inconsistentSharedEdgeCountAfter, 0);
    EXPECT_EQ(statistics.flippedFaceCount, 1);
}

TEST(MeshFaceOrientationTest, RejectsNonManifoldSharedEdge)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(5);
    mesh.faces = {
        {{0, 1, 2}},
        {{1, 0, 3}},
        {{0, 1, 4}}};

    const xjw::mesh::MeshFaceOrientationStatistics statistics =
        xjw::mesh::repairMeshFaceOrientation(&mesh);

    EXPECT_FALSE(statistics.succeeded);
    EXPECT_EQ(statistics.nonManifoldEdgeCount, 1);
    EXPECT_EQ(statistics.flippedFaceCount, 0);
}

TEST(MeshFaceOrientationTest, RemovesMinimalContradictoryMobiusFace)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(6);
    mesh.faces = {
        {{0, 1, 2}},
        {{1, 3, 2}},
        {{2, 3, 4}},
        {{3, 5, 4}},
        {{4, 5, 1}},
        {{5, 0, 1}}};

    const xjw::mesh::MeshFaceOrientationStatistics statistics =
        xjw::mesh::repairMeshFaceOrientation(&mesh);

    EXPECT_TRUE(statistics.succeeded);
    EXPECT_EQ(statistics.orientationConflictCount, 1);
    EXPECT_EQ(statistics.removedContradictoryFaceCount, 1);
    EXPECT_EQ(mesh.faceCount(), 5);
    EXPECT_EQ(statistics.inconsistentSharedEdgeCountAfter, 0);
}
