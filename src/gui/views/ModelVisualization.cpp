#include "ModelVisualization.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::gui::model_views
{

namespace
{

QVector3D normalizedVector(const QVector3D &vector)
{
    const float lengthSquared = vector.lengthSquared();
    if (!std::isfinite(lengthSquared) || lengthSquared <= 1.0e-30f)
    {
        return {};
    }
    return vector / std::sqrt(lengthSquared);
}

QColor vertexColorForMode(ColorMode mode,
                          const QVector3D &position,
                          const QColor &sourceColor,
                          const tie_points::ScalarRange &elevationRange)
{
    switch (mode)
    {
    case ColorMode::Shaded:
    case ColorMode::Solid:
        return surfaceColor(mode);
    case ColorMode::Elevation:
        return xjw::gui::model_views::elevationColor(
            position.z(), elevationRange);
    case ColorMode::Confidence:
    case ColorMode::AssignedImage:
        return surfaceColor(ColorMode::Shaded);
    case ColorMode::Wireframe:
        return surfaceColor(ColorMode::Wireframe);
    case ColorMode::Texture:
        return sourceColor.isValid() ? sourceColor : QColor(140, 140, 148);
    }
    return QColor(140, 140, 148);
}

} // namespace

void ModelVisualizationManager::setMode(ColorMode mode)
{
    _mode = mode;
}

ColorMode ModelVisualizationManager::mode() const
{
    return _mode;
}

GeometryResult ModelVisualizationManager::buildGeometry(const GeometryInput &input) const
{
    GeometryResult result;
    if (input.positions.isEmpty() || input.faces.isEmpty())
    {
        return result;
    }

    double minimumElevation = std::numeric_limits<double>::infinity();
    double maximumElevation = -std::numeric_limits<double>::infinity();
    for (const QVector3D &position : input.positions)
    {
        if (std::isfinite(position.z()))
        {
            minimumElevation = std::min(minimumElevation, double(position.z()));
            maximumElevation = std::max(maximumElevation, double(position.z()));
        }
    }
    result.elevationRange = {minimumElevation, maximumElevation};

    QVector<QVector3D> faceNormals(input.faces.size());
    QVector<QVector3D> generatedVertexNormals(input.positions.size());
    QVector<bool> validFaces(input.faces.size(), false);

    for (qsizetype faceIndex = 0; faceIndex < input.faces.size(); ++faceIndex)
    {
        const Triangle &face = input.faces.at(faceIndex);
        const int i0 = face.vertexIndices[0];
        const int i1 = face.vertexIndices[1];
        const int i2 = face.vertexIndices[2];
        if (i0 < 0 || i1 < 0 || i2 < 0 ||
            i0 >= input.positions.size() ||
            i1 >= input.positions.size() ||
            i2 >= input.positions.size())
        {
            continue;
        }

        const QVector3D &p0 = input.positions.at(i0);
        const QVector3D &p1 = input.positions.at(i1);
        const QVector3D &p2 = input.positions.at(i2);
        const QVector3D faceNormal =
            QVector3D::crossProduct(p1 - p0, p2 - p0);
        if (faceNormal.lengthSquared() <= 1.0e-30f)
        {
            continue;
        }

        faceNormals[faceIndex] = normalizedVector(faceNormal);
        generatedVertexNormals[i0] += faceNormal;
        generatedVertexNormals[i1] += faceNormal;
        generatedVertexNormals[i2] += faceNormal;
        validFaces[faceIndex] = true;
    }

    const bool hasSourceNormals =
        input.vertexNormals.size() == input.positions.size();
    QVector<QVector3D> displayVertexNormals(input.positions.size());
    for (qsizetype vertexIndex = 0;
         vertexIndex < input.positions.size();
         ++vertexIndex)
    {
        QVector3D normal;
        if (hasSourceNormals)
        {
            normal = normalizedVector(input.vertexNormals.at(vertexIndex));
        }
        if (normal.isNull())
        {
            normal = normalizedVector(generatedVertexNormals.at(vertexIndex));
        }
        if (normal.isNull())
        {
            normal = QVector3D(0.0f, 0.0f, 1.0f);
        }
        displayVertexNormals[vertexIndex] = normal;
    }

    result.filledVertices.reserve(input.faces.size() * 3 * 9);
    if (_mode == ColorMode::Wireframe)
    {
        result.wireframeVertices.reserve(input.faces.size() * 6 * 6);
    }

    for (qsizetype faceIndex = 0; faceIndex < input.faces.size(); ++faceIndex)
    {
        if (!validFaces.at(faceIndex))
        {
            continue;
        }
        const Triangle &face = input.faces.at(faceIndex);
        const QVector3D faceNormal = faceNormals.at(faceIndex);
        if (_mode == ColorMode::Wireframe)
        {
            const QColor color = surfaceColor(ColorMode::Wireframe);
            const int edgeVertices[6] = {0, 1, 1, 2, 2, 0};
            for (int corner : edgeVertices)
            {
                const QVector3D &position =
                    input.positions.at(face.vertexIndices[corner]);
                result.wireframeVertices
                    << position.x() << position.y() << position.z()
                    << color.redF() << color.greenF() << color.blueF();
            }
        }

        for (int corner = 0; corner < 3; ++corner)
        {
            const int vertexIndex = face.vertexIndices[corner];
            const QVector3D &position = input.positions.at(vertexIndex);
            const QVector3D normal =
                _mode == ColorMode::Solid
                ? faceNormal
                : displayVertexNormals.at(vertexIndex);
            const QColor sourceColor =
                vertexIndex < input.vertexColors.size()
                ? input.vertexColors.at(vertexIndex)
                : QColor();
            const QColor color = vertexColorForMode(
                _mode, position, sourceColor, result.elevationRange);
            result.filledVertices
                << position.x() << position.y() << position.z()
                << normal.x() << normal.y() << normal.z()
                << color.redF() << color.greenF() << color.blueF();
        }
    }
    return result;
}

QColor surfaceColor(ColorMode mode)
{
    switch (mode)
    {
    case ColorMode::Shaded:
        return QColor(239, 236, 224);
    case ColorMode::Solid:
        return QColor(160, 156, 205);
    case ColorMode::Wireframe:
        return QColor(54, 50, 94);
    default:
        return QColor(145, 145, 150);
    }
}

QColor elevationColor(double elevation, const tie_points::ScalarRange &range)
{
    return tie_points::scalarRampColor(range.normalize(elevation));
}

QColor confidenceColor(int confidence)
{
    const tie_points::ScalarRange range{1.0, 100.0};
    return tie_points::imageCountColor(std::clamp(confidence, 1, 100), range);
}

} // namespace xjw::gui::model_views
