#pragma once

#include "MarkerDetector.h"

namespace xjw::control_points
{

class DetectionMerger
{
public:
    static QVector<MarkerDetection> merge(const QVector<MarkerDetection> &detections,
                                           double centerDistanceThresholdPx = 2.0);
};

} // namespace xjw::control_points
