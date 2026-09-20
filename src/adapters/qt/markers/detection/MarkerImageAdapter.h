#pragma once

#include "detection/MarkerDetector.h"

#include <QImage>

namespace xjw::app::markers
{
    // Retains Qt's source grayscale conversion and qGray(pixel) mask semantics.
    // Converted storage is owned locally until the synchronous detector returns.
    QVector<control_points::MarkerDetection> detectMarkers(const control_points::MarkerDetector& detector,
                                                           const QImage& image,
                                                           const QImage& mask,
                                                           const control_points::MarkerDetectionOptions& options);
} // namespace xjw::app::markers
