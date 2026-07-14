#pragma once

#include "common/SfmTypes.h"
#include "model/MarkerTypes.h"

#include <array>
#include <string>
#include <vector>

namespace xjw::control_points
{

enum class PriorObservationState
{
    ManualPinned,
    AutoDetected,
    Predicted,
    Blocked,
    Disabled
};

struct PriorObservation
{
    ImageId imageId = kInvalidImageId;
    double x = 0.0;
    double y = 0.0;
    PriorObservationState state = PriorObservationState::Predicted;
    double confidence = 1.0;
    bool stale = false;
};

struct PriorTrack
{
    std::string markerId;
    std::vector<PriorObservation> observations;
    double confidence = 1.0;
    MarkerRole role = MarkerRole::TieMarker;
    bool hasReference = false;
    bool referenceUsable = false;
    std::array<double, 3> referencePoint{{0.0, 0.0, 0.0}};
    std::array<double, 3> referenceSigma{{1.0, 1.0, 1.0}};
};

struct PriorScaleBar
{
    std::string scaleBarId;
    std::string firstMarkerId;
    std::string secondMarkerId;
    ScaleBarRole role = ScaleBarRole::Control;
    bool enabled = true;
    double measuredDistance = 0.0;
    double sigma = 1.0;
};

struct PriorTrackDiagnostics
{
    int tracksSubmitted = 0;
    int tracksAccepted = 0;
    int tracksRejected = 0;
    int observationsAccepted = 0;
    int observationsRejected = 0;
    int observationConflicts = 0;
    std::vector<std::string> messages;
};

inline bool priorObservationParticipates(PriorObservationState state)
{
    return state == PriorObservationState::ManualPinned
        || state == PriorObservationState::AutoDetected;
}

} // namespace xjw::control_points
