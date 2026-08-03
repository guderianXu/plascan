#pragma once

#include "detection/MarkerDetector.h"

#include <QImage>
#include <QSizeF>
#include <QString>
#include <QVector>

namespace xjw::control_points
{

struct MarkerPrintRequest
{
    MarkerTargetFamily family = MarkerTargetFamily::AprilTag36h11;
    QVector<int> ids;
    double targetDiameterMm = 30.0;
    QSizeF pageSizeMm = QSizeF(210.0, 297.0);
    double marginMm = 12.0;
    double spacingMm = 8.0;
    bool showLabels = true;
};

struct MarkerSheetRenderResult
{
    bool ok = false;
    QVector<QImage> pages;
    QString error;
    int dpi = 0;
};

class MarkerSheetRenderer final
{
public:
    static QImage renderMarkerImage(MarkerTargetFamily family, int id, int targetPixels);
    static MarkerSheetRenderResult render(const MarkerPrintRequest &request, int dpi);
};

} // namespace xjw::control_points
