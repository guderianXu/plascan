#include "ModelVisualization.h"

#include <gtest/gtest.h>

namespace
{

using xjw::gui::model_views::ColorMode;
using xjw::gui::model_views::GeometryInput;
using xjw::gui::model_views::ModelVisualizationManager;
using xjw::gui::model_views::Triangle;
using xjw::gui::model_views::confidenceColor;
using xjw::gui::model_views::surfaceColor;

void expectVertexColor(const QVector<float> &vertices,
                       qsizetype cornerIndex,
                       const QColor &expected)
{
    const qsizetype colorOffset = cornerIndex * 9 + 6;
    ASSERT_GE(vertices.size(), colorOffset + 3);
    EXPECT_NEAR(vertices.at(colorOffset), expected.redF(), 1.0e-5f);
    EXPECT_NEAR(vertices.at(colorOffset + 1), expected.greenF(), 1.0e-5f);
    EXPECT_NEAR(vertices.at(colorOffset + 2), expected.blueF(), 1.0e-5f);
}

TEST(ModelVisualizationTest, UsesDistinctMetashapeStyleSurfaceColors)
{
    EXPECT_GT(surfaceColor(ColorMode::Shaded).value(), 220);
    EXPECT_GT(surfaceColor(ColorMode::Shaded).red(),
              surfaceColor(ColorMode::Shaded).blue());
    EXPECT_GT(surfaceColor(ColorMode::Solid).blue(),
              surfaceColor(ColorMode::Solid).red());
    EXPECT_LT(surfaceColor(ColorMode::Wireframe).value(), 120);
}

TEST(ModelVisualizationTest, ReservedConfidencePaletteRunsFromRedLowToBlueHigh)
{
    const QColor low = confidenceColor(1);
    const QColor high = confidenceColor(100);
    EXPECT_GT(low.red(), low.blue());
    EXPECT_GT(high.blue(), high.red());
}

TEST(ModelVisualizationTest, DefaultsToShadedMode)
{
    ModelVisualizationManager manager;
    EXPECT_EQ(manager.mode(), ColorMode::Shaded);
}

TEST(ModelVisualizationTest, ShadedModePreservesProvidedVertexColors)
{
    GeometryInput input;
    input.positions = {
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(1.0f, 0.0f, 0.0f),
        QVector3D(0.0f, 1.0f, 0.0f)
    };
    input.faces = {Triangle{{0, 1, 2}}};
    input.vertexColors = {
        QColor(230, 20, 40),
        QColor(30, 210, 70),
        QColor(50, 80, 240)
    };

    ModelVisualizationManager manager;
    manager.setMode(ColorMode::Shaded);
    const auto geometry = manager.buildGeometry(input);

    ASSERT_EQ(geometry.filledVertices.size(), 27);
    for (qsizetype cornerIndex = 0;
         cornerIndex < input.vertexColors.size();
         ++cornerIndex)
    {
        expectVertexColor(geometry.filledVertices,
                          cornerIndex,
                          input.vertexColors.at(cornerIndex));
    }
}

TEST(ModelVisualizationTest, ShadedModeFallsBackToNeutralSurfaceWithoutVertexColors)
{
    GeometryInput input;
    input.positions = {
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(1.0f, 0.0f, 0.0f),
        QVector3D(0.0f, 1.0f, 0.0f)
    };
    input.faces = {Triangle{{0, 1, 2}}};

    ModelVisualizationManager manager;
    manager.setMode(ColorMode::Shaded);
    const auto geometry = manager.buildGeometry(input);

    ASSERT_EQ(geometry.filledVertices.size(), 27);
    const QColor fallbackColor = surfaceColor(ColorMode::Shaded);
    for (qsizetype cornerIndex = 0; cornerIndex < 3; ++cornerIndex)
    {
        expectVertexColor(geometry.filledVertices, cornerIndex, fallbackColor);
    }
}

TEST(ModelVisualizationTest, DedicatedManagerBuildsSmoothAndFacetedNormals)
{
    GeometryInput input;
    input.positions = {
        QVector3D(1.0f, 1.0f, 1.0f),
        QVector3D(-1.0f, -1.0f, 1.0f),
        QVector3D(-1.0f, 1.0f, -1.0f),
        QVector3D(1.0f, -1.0f, -1.0f)
    };
    input.faces = {
        Triangle{{0, 2, 1}},
        Triangle{{0, 1, 3}},
        Triangle{{0, 3, 2}},
        Triangle{{1, 2, 3}}
    };

    ModelVisualizationManager manager;
    manager.setMode(ColorMode::Shaded);
    const auto shaded = manager.buildGeometry(input);
    manager.setMode(ColorMode::Solid);
    const auto solid = manager.buildGeometry(input);

    ASSERT_EQ(shaded.filledVertices.size(), 4 * 3 * 9);
    ASSERT_EQ(solid.filledVertices.size(), 4 * 3 * 9);

    const QVector3D shadedFirst(
        shaded.filledVertices.at(3),
        shaded.filledVertices.at(4),
        shaded.filledVertices.at(5));
    const QVector3D shadedSecond(
        shaded.filledVertices.at(12),
        shaded.filledVertices.at(13),
        shaded.filledVertices.at(14));
    const QVector3D solidFirst(
        solid.filledVertices.at(3),
        solid.filledVertices.at(4),
        solid.filledVertices.at(5));
    const QVector3D solidSecond(
        solid.filledVertices.at(12),
        solid.filledVertices.at(13),
        solid.filledVertices.at(14));

    EXPECT_NE(shadedFirst, shadedSecond);
    EXPECT_EQ(solidFirst, solidSecond);
    EXPECT_NEAR(shadedFirst.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(solidFirst.length(), 1.0f, 1.0e-5f);
}

TEST(ModelVisualizationTest, SolidModeNormalizesSmallScaleFaces)
{
    GeometryInput input;
    input.positions = {
        QVector3D(0.0000f, 0.0000f, 0.0000f),
        QVector3D(0.0010f, 0.0000f, 0.0000f),
        QVector3D(0.0000f, 0.0010f, 0.0000f)
    };
    input.faces = {Triangle{{0, 1, 2}}};

    ModelVisualizationManager manager;
    manager.setMode(ColorMode::Solid);
    const auto geometry = manager.buildGeometry(input);

    ASSERT_EQ(geometry.filledVertices.size(), 27);
    const QVector3D normal(
        geometry.filledVertices.at(3),
        geometry.filledVertices.at(4),
        geometry.filledVertices.at(5));
    EXPECT_NEAR(normal.length(), 1.0f, 1.0e-5f);
}

TEST(ModelVisualizationTest, ShadedModePreservesProvidedVertexNormals)
{
    GeometryInput input;
    input.positions = {
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(1.0f, 0.0f, 0.0f),
        QVector3D(0.0f, 1.0f, 0.0f)
    };
    input.faces = {Triangle{{0, 1, 2}}};
    input.vertexNormals = {
        QVector3D(1.0f, 1.0f, 1.0f),
        QVector3D(0.0f, 1.0f, 1.0f),
        QVector3D(1.0f, 0.0f, 1.0f)
    };

    ModelVisualizationManager manager;
    manager.setMode(ColorMode::Shaded);
    const auto geometry = manager.buildGeometry(input);

    ASSERT_EQ(geometry.filledVertices.size(), 27);
    const QVector3D firstNormal(
        geometry.filledVertices.at(3),
        geometry.filledVertices.at(4),
        geometry.filledVertices.at(5));
    const QVector3D expectedNormal = input.vertexNormals.first().normalized();
    EXPECT_NEAR(firstNormal.x(), expectedNormal.x(), 1.0e-5f);
    EXPECT_NEAR(firstNormal.y(), expectedNormal.y(), 1.0e-5f);
    EXPECT_NEAR(firstNormal.z(), expectedNormal.z(), 1.0e-5f);
}

} // namespace
