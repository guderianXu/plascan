#include "BaInputBuilder.h"

#include "project/BaTrackBuilder.h"
#include "project/MarkerBaAdapter.h"
#include "project/ProjectMatchInputReader.h"
#include "project/SurveyControlBaAdapter.h"

#include <utility>

namespace xjw::core::project
{
namespace
{

void resetBuildResult(BaInputBuildResult *result)
{
    result->cameras.clear();
    result->imagePathByIndex.clear();
    result->beforeCamMeta.clear();
    result->tracks.clear();
    result->scaleBarConstraints.clear();
    result->sidecarV2PairCount = 0;
    result->multiViewTrackCount = 0;
    result->rejectedConflictTrackCount = 0;
    result->surveyControlTrackCount = 0;
    result->surveyControlObservationCount = 0;
    result->rejectedSurveyControlPointCount = 0;
    result->surveyScaleBarConstraintCount = 0;
    result->rejectedSurveyScaleBarCount = 0;
    result->markerControlNetwork = {};
    result->markerControlTrackCount = 0;
    result->markerCheckTrackCount = 0;
    result->rejectedMarkerTrackCount = 0;
    result->markerControlPointConstraintCount = 0;
    result->markerControlScaleBarConstraintCount = 0;
    result->markerCheckScaleBarCount = 0;
    result->rejectedMarkerScaleBarCount = 0;
    result->markerTrackBindings.clear();
    result->markerScaleBarBindings.clear();
}

} // namespace

BaInputBuildStatus buildBaInputFromMeta(const QJsonObject &meta,
                                        const QStringList &selectedImages,
                                        int minMatches,
                                        BaInputBuildResult *result,
                                        const MarkerBaInput *markerInput)
{
    if (!result)
    {
        return BaInputBuildStatus::NoTracks;
    }
    resetBuildResult(result);

    ProjectMatchInput matchInput;
    if (!readProjectMatchInput(meta, selectedImages, minMatches, &matchInput))
    {
        return BaInputBuildStatus::NoTracks;
    }
    if (matchInput.cameras.size() < 2)
    {
        return BaInputBuildStatus::NotEnoughCameras;
    }

    appendBaTracks(matchInput, result);
    result->sidecarV2PairCount = matchInput.sidecarV2PairCount;
    result->cameras = std::move(matchInput.cameras);
    result->imagePathByIndex = std::move(matchInput.imagePathByIndex);
    result->beforeCamMeta = std::move(matchInput.beforeCamMeta);

    appendSurveyControlBaInput(meta, matchInput.cameraIndexByPath, result);
    appendMarkerBaInput(markerInput, matchInput.cameraIndexByPath, result);
    return result->tracks.empty()
        ? BaInputBuildStatus::NoTracks
        : BaInputBuildStatus::Ok;
}

} // namespace xjw::core::project
