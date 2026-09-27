#pragma once

#include "model/MarkerTypes.h"

#include <placamera/frame_camera.h>

#include <QHash>

#include <opencv2/core.hpp>

#include <functional>
#include <memory>
#include <optional>

namespace xjw::control_points
{

    struct MarkerImageView
    {
        QString imagePath;
        std::shared_ptr<const placamera::FramePinholeModel> camera;
        std::function<bool(const QPointF&)> acceptsPixel;
    };

    struct MarkerTriangulationOptions
    {
        double maximumReprojectionErrorPx = 4.0;
        double minimumIntersectionAngleDegrees = 0.5;
    };

    struct MarkerTriangulation
    {
        bool success = false;
        cv::Point3d point;
        std::optional<placamera::FrameId> groundFrame;
        double rmsReprojectionPx = std::numeric_limits<double>::quiet_NaN();
        double minimumIntersectionAngleDegrees = 0.0;
        QHash<QString, double> residualByImage;
        QStringList usedImageIds;
        QString error;
    };

    struct EpipolarBand
    {
        bool valid = false;
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        double halfWidthPx = 2.0;

        double distanceTo(const QPointF& pixel) const;
        bool contains(const QPointF& pixel) const;
    };

    MarkerTriangulation triangulateMarker(const Marker& marker,
                                          const QVector<MarkerImageView>& views,
                                          const MarkerTriangulationOptions& options = {});

    // A two-depth secant of the PlaCamera epipolar locus; exact without lens distortion.
    EpipolarBand epipolarSearchBand(const QPointF& sourcePixel,
                                    const MarkerImageView& sourceView,
                                    const MarkerImageView& targetView,
                                    double halfWidthPx = 2.0);

} // namespace xjw::control_points
