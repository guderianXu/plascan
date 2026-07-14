#pragma once

#include "Camera.h"
#include "BundleAdjust.h"
#include "model/MarkerSet.h"
#include "registration/ControlNetworkSolver.h"

#include <QHash>
#include <QJsonObject>
#include <QMap>
#include <QStringList>

#include <vector>

namespace xjw::core::project
{

enum class BaInputBuildStatus
{
    Ok,
    NotEnoughCameras,
    NoTracks
};

struct BaInputBuildResult
{
    std::vector<xjw::Camera> cameras;
    QStringList imagePathByIndex;
    QMap<QString, QJsonObject> beforeCamMeta;
    std::vector<xjw::BATrack> tracks;
    std::vector<xjw::BAScaleBarConstraint> scaleBarConstraints;
    int sidecarV2PairCount = 0;
    int multiViewTrackCount = 0;
    int rejectedConflictTrackCount = 0;
    int surveyControlTrackCount = 0;
    int surveyControlObservationCount = 0;
    int rejectedSurveyControlPointCount = 0;
    int surveyScaleBarConstraintCount = 0;
    int rejectedSurveyScaleBarCount = 0;
    control_points::ControlNetworkResult markerControlNetwork;
    int markerControlTrackCount = 0;
    int markerCheckTrackCount = 0;
    int rejectedMarkerTrackCount = 0;
    int markerControlPointConstraintCount = 0;
    int markerControlScaleBarConstraintCount = 0;
    int markerCheckScaleBarCount = 0;
    int rejectedMarkerScaleBarCount = 0;
    struct MarkerTrackBinding
    {
        control_points::MarkerId markerId;
        control_points::MarkerRole role = control_points::MarkerRole::TieMarker;
        int trackIndex = -1;
        std::array<double, 3> referencePoint{{0.0, 0.0, 0.0}};
        std::array<double, 3> sigma{{1.0, 1.0, 1.0}};
        bool usedAsConstraint = false;
    };
    struct MarkerScaleBarBinding
    {
        control_points::ScaleBarId scaleBarId;
        control_points::ScaleBarRole role = control_points::ScaleBarRole::Control;
        int trackIndexA = -1;
        int trackIndexB = -1;
        double measuredDistance = 0.0;
    };
    QVector<MarkerTrackBinding> markerTrackBindings;
    QVector<MarkerScaleBarBinding> markerScaleBarBindings;
};

struct MarkerBaInput
{
    const control_points::MarkerSet *markerSet = nullptr;
    QHash<QString, QString> imagePathById;
};

BaInputBuildStatus buildBaInputFromMeta(const QJsonObject &meta,
                                        const QStringList &selectedImages,
                                        int minMatches,
                                        BaInputBuildResult *result,
                                        const MarkerBaInput *markerInput = nullptr);

} // namespace xjw::core::project
