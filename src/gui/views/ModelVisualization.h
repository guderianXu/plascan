#pragma once

#include <QColor>
#include <QVector>
#include <QVector3D>

#include <array>

#include "TiePointVisualization.h"

namespace xjw::gui::model_views
{

enum class ColorMode
{
    Texture,
    Shaded,
    Solid,
    Wireframe,
    Elevation,
    Confidence,
    AssignedImage
};

struct Triangle
{
    std::array<int, 3> vertexIndices{};
};

struct GeometryInput
{
    QVector<QVector3D> positions;
    QVector<Triangle> faces;
    QVector<QColor> vertexColors;
    QVector<QVector3D> vertexNormals;
};

struct GeometryResult
{
    QVector<float> filledVertices;
    QVector<float> wireframeVertices;
    tie_points::ScalarRange elevationRange;
};

class ModelVisualizationManager
{
public:
    void setMode(ColorMode mode);
    ColorMode mode() const;
    GeometryResult buildGeometry(const GeometryInput &input) const;

private:
    ColorMode _mode = ColorMode::Shaded;
};

QColor surfaceColor(ColorMode mode);
QColor elevationColor(double elevation, const tie_points::ScalarRange &range);
QColor confidenceColor(int confidence);

} // namespace xjw::gui::model_views
