#pragma once

#include "MarkerGeometry.h"

namespace xjw::control_points
{

struct MarkerPredictionResult
{
    MarkerTriangulation triangulation;
    QVector<MarkerProjection> predictions;
};

class MarkerProjectionPredictor
{
public:
    static MarkerPredictionResult predict(
        const Marker &marker,
        const QVector<MarkerCamera> &cameras,
        const MarkerTriangulationOptions &options = {});
};

} // namespace xjw::control_points
