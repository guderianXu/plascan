#pragma once

#include <QColor>
#include <QString>
#include <QVector>

namespace xjw::gui::tie_points
{

enum class ColorMode
{
    Color,
    Elevation,
    ImageCount
};

struct ScalarRange
{
    double minimum = 0.0;
    double maximum = 0.0;

    bool isValid() const;
    double normalize(double value) const;
};

struct ImageCountMetadata
{
    QVector<int> counts;
    QString errorMessage;

    bool isValidFor(qsizetype pointCount) const;
};

QColor elevationColor(double elevation, const ScalarRange &range);
QColor imageCountColor(int imageCount, const ScalarRange &range);
QColor scalarRampColor(double normalizedValue);
float pointSizeForMode(ColorMode mode);
QString inferSidecarPath(const QString &pointCloudPath);
ImageCountMetadata loadImageCountMetadata(const QString &sidecarPath);

} // namespace xjw::gui::tie_points
