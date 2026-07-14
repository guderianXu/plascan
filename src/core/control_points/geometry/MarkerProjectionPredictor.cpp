#include "MarkerProjectionPredictor.h"

#include <algorithm>

namespace xjw::control_points
{

MarkerPredictionResult MarkerProjectionPredictor::predict(
    const Marker &marker,
    const QVector<MarkerCamera> &cameras,
    const MarkerTriangulationOptions &options)
{
    MarkerPredictionResult result;
    result.triangulation = triangulateMarker(marker, cameras, options);
    if (!result.triangulation.success) return result;

    for (const MarkerCamera &camera : cameras)
    {
        const bool already_observed = std::any_of(
            marker.projections.cbegin(), marker.projections.cend(), [&camera](const MarkerProjection &projection)
        {
            return projection.imageId == camera.imageId;
        });
        if (already_observed || camera.depth(result.triangulation.point) <= 0.0) continue;

        const QPointF pixel = camera.project(result.triangulation.point);
        if (!camera.contains(pixel)) continue;
        if (camera.acceptsPixel && !camera.acceptsPixel(pixel)) continue;

        MarkerProjection projection;
        projection.imageId = camera.imageId;
        projection.imagePathSnapshot = camera.imagePath;
        projection.xy = pixel;
        projection.state = ProjectionState::Predicted;
        projection.sigmaPx = std::max(1.0, result.triangulation.rmsReprojectionPx);
        projection.confidence = 1.0 / (1.0 + result.triangulation.rmsReprojectionPx);
        projection.source = QStringLiteral("geometry_prediction");
        result.predictions.push_back(projection);
    }
    return result;
}

} // namespace xjw::control_points
