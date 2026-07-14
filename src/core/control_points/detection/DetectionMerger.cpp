#include "DetectionMerger.h"

#include <QLineF>

#include <algorithm>

namespace xjw::control_points
{

QVector<MarkerDetection> DetectionMerger::merge(const QVector<MarkerDetection> &detections,
                                                 double centerDistanceThresholdPx)
{
    QVector<MarkerDetection> ordered = detections;
    std::sort(ordered.begin(), ordered.end(), [](const MarkerDetection &left, const MarkerDetection &right)
    {
        if (left.confidence != right.confidence)
        {
            return left.confidence > right.confidence;
        }
        return left.decisionMargin > right.decisionMargin;
    });

    QVector<MarkerDetection> result;
    for (const MarkerDetection &candidate : ordered)
    {
        const bool duplicate = std::any_of(result.cbegin(), result.cend(), [&](const MarkerDetection &accepted)
        {
            const bool same_code = candidate.family == accepted.family
                                   && candidate.targetId >= 0
                                   && candidate.targetId == accepted.targetId;
            const bool same_location = QLineF(candidate.center, accepted.center).length()
                                       <= centerDistanceThresholdPx;
            return same_code || same_location;
        });
        if (!duplicate)
        {
            result.push_back(candidate);
        }
    }
    return result;
}

} // namespace xjw::control_points
