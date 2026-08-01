#include "RobustSurfaceDisplacementSolver.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

xjw::mesh::TriMesh makeStripMesh(int column_count)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.reserve(static_cast<std::size_t>(column_count * 2));
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < column_count; ++column)
        {
            xjw::mesh::MeshVertex vertex;
            vertex.x = static_cast<float>(column);
            vertex.y = static_cast<float>(row);
            vertex.nz = 1.0f;
            mesh.vertices.push_back(vertex);
        }
    }
    for (int column = 0; column + 1 < column_count; ++column)
    {
        const int upper_left = column;
        const int upper_right = column + 1;
        const int lower_left = column_count + column;
        const int lower_right = column_count + column + 1;
        mesh.faces.push_back(
            {{upper_left, upper_right, lower_left}});
        mesh.faces.push_back(
            {{upper_right, lower_right, lower_left}});
    }
    return mesh;
}

xjw::mesh::RobustSurfaceDisplacementOptions baseOptions()
{
    xjw::mesh::RobustSurfaceDisplacementOptions options;
    options.irlsIterations = 5;
    options.maximumPcgIterations = 200;
    options.convergenceTolerance = 1.0e-7f;
    options.maximumDisplacement = 0.10f;
    options.minimumNormalDot = 0.0f;
    return options;
}

std::vector<xjw::mesh::RobustSurfaceDisplacementObservation>
makeCrossingLayerObservations(int column_count)
{
    std::vector<xjw::mesh::RobustSurfaceDisplacementObservation>
        observations;
    observations.reserve(
        static_cast<std::size_t>(column_count * 4));
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < column_count; ++column)
        {
            const int vertex_index = row * column_count + column;
            const float positive_weight =
                static_cast<float>(column) /
                static_cast<float>(column_count - 1);
            observations.push_back(
                {vertex_index, -0.02f, 1.0f - positive_weight});
            observations.push_back(
                {vertex_index, 0.02f, positive_weight});
        }
    }
    return observations;
}

} // namespace

TEST(
    RobustSurfaceDisplacementSolverTest,
    CrossingDepthLayerWeightsProduceContinuousDisplacement)
{
    constexpr int column_count = 7;
    const xjw::mesh::TriMesh mesh = makeStripMesh(column_count);
    const auto observations =
        makeCrossingLayerObservations(column_count);
    auto options = baseOptions();
    options.robustScale = 0.08f;
    options.laplacianWeight = 0.45f;
    options.hullPriorWeight = 0.002f;
    options.maximumDisplacement = 0.05f;
    std::vector<float> displacement(mesh.vertices.size(), 0.0f);

    const auto statistics =
        xjw::mesh::RobustSurfaceDisplacementSolver::solve(
            mesh, observations, options, &displacement);

    ASSERT_TRUE(statistics.solved);
    EXPECT_EQ(statistics.anchoredVertexCount, 14U);
    EXPECT_EQ(statistics.priorOnlyVertexCount, 0U);
    std::vector<float> column_average(column_count, 0.0f);
    for (int column = 0; column < column_count; ++column)
    {
        column_average[static_cast<std::size_t>(column)] =
            0.5f *
            (displacement[static_cast<std::size_t>(column)] +
             displacement[static_cast<std::size_t>(
                 column_count + column)]);
    }
    EXPECT_LT(column_average.front(), -0.015f);
    EXPECT_GT(column_average.back(), 0.015f);
    EXPECT_NEAR(column_average[3], 0.0f, 0.002f);
    float maximum_adjacent_jump = 0.0f;
    for (int column = 1; column < column_count; ++column)
    {
        EXPECT_GE(
            column_average[static_cast<std::size_t>(column)] +
                1.0e-5f,
            column_average[static_cast<std::size_t>(column - 1)]);
        maximum_adjacent_jump = std::max(
            maximum_adjacent_jump,
            std::abs(
                column_average[static_cast<std::size_t>(column)] -
                column_average[
                    static_cast<std::size_t>(column - 1)]));
    }
    EXPECT_LT(maximum_adjacent_jump, 0.012f);
    EXPECT_LT(statistics.finalEnergy, statistics.initialEnergy);
}

TEST(
    RobustSurfaceDisplacementSolverTest,
    CauchyIrlsRejectsFarEqualWeightOutlier)
{
    const xjw::mesh::TriMesh mesh = makeStripMesh(2);
    std::vector<xjw::mesh::RobustSurfaceDisplacementObservation>
        observations;
    for (int vertex_index = 0;
         vertex_index < mesh.vertexCount();
         ++vertex_index)
    {
        observations.push_back({vertex_index, 0.011f, 1.0f});
        observations.push_back({vertex_index, 0.012f, 1.0f});
        observations.push_back({vertex_index, 0.090f, 1.0f});
    }
    auto options = baseOptions();
    options.robustScale = 0.004f;
    options.laplacianWeight = 0.2f;
    options.hullPriorWeight = 0.001f;
    std::vector<float> displacement(mesh.vertices.size(), 0.0f);

    const auto statistics =
        xjw::mesh::RobustSurfaceDisplacementSolver::solve(
            mesh, observations, options, &displacement);

    ASSERT_TRUE(statistics.solved);
    EXPECT_EQ(statistics.observationCount, 12U);
    for (const float value : displacement)
    {
        EXPECT_NEAR(value, 0.0115f, 0.0015f);
        EXPECT_GT(std::abs(value - 0.0376667f), 0.02f);
    }
}

TEST(
    RobustSurfaceDisplacementSolverTest,
    PriorOnlyDisconnectedComponentReturnsToHullWithoutLeakage)
{
    xjw::mesh::TriMesh mesh;
    for (int index = 0; index < 6; ++index)
    {
        xjw::mesh::MeshVertex vertex;
        vertex.x = static_cast<float>(index % 3);
        vertex.y = static_cast<float>(index / 3) * 10.0f;
        vertex.nz = 1.0f;
        mesh.vertices.push_back(vertex);
    }
    mesh.faces.push_back({{0, 1, 2}});
    mesh.faces.push_back({{3, 4, 5}});
    std::vector<xjw::mesh::RobustSurfaceDisplacementObservation>
        observations;
    for (int vertex_index = 0; vertex_index < 3; ++vertex_index)
    {
        observations.push_back({vertex_index, 0.025f, 1.0f});
    }
    auto options = baseOptions();
    options.robustScale = 0.05f;
    options.laplacianWeight = 0.5f;
    options.hullPriorWeight = 0.01f;
    std::vector<float> displacement =
        {0.0f, 0.0f, 0.0f, 0.03f, -0.02f, 0.01f};

    const auto statistics =
        xjw::mesh::RobustSurfaceDisplacementSolver::solve(
            mesh, observations, options, &displacement);

    ASSERT_TRUE(statistics.solved);
    EXPECT_EQ(statistics.anchoredVertexCount, 3U);
    EXPECT_EQ(statistics.priorOnlyVertexCount, 3U);
    for (int index = 0; index < 3; ++index)
    {
        EXPECT_NEAR(
            displacement[static_cast<std::size_t>(index)],
            0.0247f,
            0.002f);
    }
    for (int index = 3; index < 6; ++index)
    {
        EXPECT_NEAR(
            displacement[static_cast<std::size_t>(index)],
            0.0f,
            1.0e-5f);
    }
}

TEST(
    RobustSurfaceDisplacementSolverTest,
    CancellationLeavesCallerDisplacementUnchanged)
{
    constexpr int column_count = 7;
    const xjw::mesh::TriMesh mesh = makeStripMesh(column_count);
    const auto observations =
        makeCrossingLayerObservations(column_count);
    auto options = baseOptions();
    options.robustScale = 0.08f;
    options.laplacianWeight = 0.45f;
    options.hullPriorWeight = 0.002f;
    std::vector<float> displacement(mesh.vertices.size(), 0.003f);
    const std::vector<float> original = displacement;
    int cancellation_checks = 0;

    const auto statistics =
        xjw::mesh::RobustSurfaceDisplacementSolver::solve(
            mesh,
            observations,
            options,
            &displacement,
            [&cancellation_checks]()
            {
                return ++cancellation_checks >= 3;
            });

    EXPECT_TRUE(statistics.cancelled);
    EXPECT_FALSE(statistics.solved);
    EXPECT_GE(cancellation_checks, 3);
    EXPECT_EQ(displacement, original);
}

TEST(
    RobustSurfaceDisplacementSolverTest,
    IgnoresNonFiniteObservationsWithoutPollutingSolution)
{
    const xjw::mesh::TriMesh mesh = makeStripMesh(2);
    std::vector<xjw::mesh::RobustSurfaceDisplacementObservation>
        observations;
    for (int vertex_index = 0;
         vertex_index < mesh.vertexCount();
         ++vertex_index)
    {
        observations.push_back({vertex_index, 0.01f, 1.0f});
    }
    observations.push_back(
        {0, std::numeric_limits<float>::quiet_NaN(), 1.0f});
    observations.push_back(
        {1, 0.02f, std::numeric_limits<float>::infinity()});
    auto options = baseOptions();
    options.robustScale = 0.02f;
    options.hullPriorWeight = 0.001f;
    std::vector<float> displacement(mesh.vertices.size(), 0.0f);

    const auto statistics =
        xjw::mesh::RobustSurfaceDisplacementSolver::solve(
            mesh, observations, options, &displacement);

    ASSERT_TRUE(statistics.solved);
    EXPECT_EQ(statistics.observationCount, 4U);
    for (const float value : displacement)
    {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_NEAR(value, 0.01f, 0.0015f);
    }
}
