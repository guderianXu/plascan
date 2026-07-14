#pragma once

#include "MarkerDetector.h"
#include "model/MarkerSet.h"

#include <QHash>

namespace xjw::control_points
{

struct MarkerDetectionObservation
{
    QString imageId;
    QString imagePathSnapshot;
    QString imageContentSignature;
    MarkerDetection detection;
};

struct DetectionConflict
{
    QString reason;
    QString message;
    MarkerDetectionObservation observation;
};

struct DetectionIntegrationResult
{
    MarkerSet markerSet;
    QVector<DetectionConflict> conflicts;
    QVector<MarkerDetectionObservation> pendingReview;
    int appliedDetections = 0;
    int createdMarkers = 0;
};

class DetectionIntegrator
{
public:
    static DetectionIntegrationResult integrate(
        const MarkerSet &base,
        const QVector<MarkerDetectionObservation> &observations,
        const QHash<QString, QString> &currentImageSignatures,
        double manualConflictThresholdPx = 2.0);
};

} // namespace xjw::control_points
