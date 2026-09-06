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
    result->indexedObservationCount = 0;
    result->multiViewTrackCount = 0;
    result->matchDiagnostics = {};
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
    // 第一阶段建立统一的相机索引空间；后续自动轨迹、控制点和标记都只能引用
    // 该索引，禁止各适配器再次按文件名自行排序相机。
    if (!readProjectMatchInput(meta, selectedImages, minMatches, &matchInput))
    {
        return BaInputBuildStatus::NoTracks;
    }
    if (matchInput.cameras.size() < 2)
    {
        return BaInputBuildStatus::NotEnoughCameras;
    }

    // 第二阶段先生成自动连接点轨迹，再移动相机和工程快照。appendBaTracks
    // 需要读取 matchInput.cameras，移动顺序不可提前。
    appendBaTracks(matchInput, result);
    result->indexedObservationCount = matchInput.indexedObservationCount;
    result->matchDiagnostics = matchInput.diagnostics;
    result->cameras = std::move(matchInput.cameras);
    result->imagePathByIndex = std::move(matchInput.imagePathByIndex);
    result->beforeCamMeta = std::move(matchInput.beforeCamMeta);

    // 第三阶段追加物方约束。人工标记可能先估计控制网 Sim(3) 并同时变换已有
    // 自动轨迹与相机，因此必须在全部自动轨迹完成后执行。
    appendSurveyControlBaInput(meta, matchInput.cameraIndexByPath, result);
    appendMarkerBaInput(markerInput, matchInput.cameraIndexByPath, result);
    return result->tracks.empty()
        ? BaInputBuildStatus::NoTracks
        : BaInputBuildStatus::Ok;
}

} // namespace xjw::core::project
