#include "MarkerProjectionPredictor.h"

#include <algorithm>
#include <string>

namespace xjw::control_points
{

    MarkerPredictionResult MarkerProjectionPredictor::predict(const Marker& marker,
                                                              const QVector<MarkerImageView>& views,
                                                              const MarkerTriangulationOptions& options)
    {
        MarkerPredictionResult result;
        result.triangulation = triangulateMarker(marker, views, options);
        if (!result.triangulation.success || !result.triangulation.groundFrame)
        {
            return result;
        }

        const auto& point = result.triangulation.point;
        const placamera::GroundCoordinate ground{*result.triangulation.groundFrame, {point.x, point.y, point.z}};
        placamera::EvaluationOptions evaluation_options;
        evaluation_options.requireInsideImage = true;

        for (const MarkerImageView& view : views)
        {
            if (!view.camera)
            {
                continue;
            }
            const QString image_id = QString::fromStdString(std::string(view.camera->imageId().value()));
            const bool already_observed =
                std::any_of(marker.projections.cbegin(),
                            marker.projections.cend(),
                            [&image_id](const MarkerProjection& projection) { return projection.imageId == image_id; });
            if (already_observed)
            {
                continue;
            }

            const auto projected = view.camera->groundToImage(ground, evaluation_options);
            if (!projected)
            {
                continue;
            }
            const QPointF pixel(projected.value().image.sample, projected.value().image.line);
            if (view.acceptsPixel && !view.acceptsPixel(pixel))
            {
                continue;
            }

            MarkerProjection projection;
            projection.imageId = image_id;
            projection.imagePathSnapshot = view.imagePath;
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
