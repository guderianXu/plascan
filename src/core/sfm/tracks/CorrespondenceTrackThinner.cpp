#include "tracks/CorrespondenceTrackThinner.h"

#include "graph/CorrespondenceGraph.h"
#include "reconstruction/SfmReconstruction.h"
#include "tracks/MultiViewTrackBuilder.h"

#include <algorithm>
#include <cmath>

namespace xjw
{

CorrespondenceTrackThinningResult thinCorrespondenceTracks(
    const SfmReconstruction &reconstruction,
    CorrespondenceGraph *graph,
    const CorrespondenceTrackThinningOptions &options)
{
    CorrespondenceTrackThinningResult result;
    if (!graph)
    {
        return result;
    }

    MultiViewTrackBuilder builder;
    float maxKeypointX = 0.0f;
    float maxKeypointY = 0.0f;

    // 先把对应图恢复成完整多视轨迹。直接按 pair 截断会打断跨多幅影像的同一
    // 物点，并让 SfM 中的 track length 统计失真。
    for (ImageId imageId : reconstruction.allImageIds())
    {
        const ImageData &image = reconstruction.image(imageId);
        builder.setImageKeypoints(imageId, image.keypoints);
        for (const FeatureKeypoint &keypoint : image.keypoints)
        {
            if (std::isfinite(keypoint.x))
            {
                maxKeypointX = std::max(maxKeypointX, keypoint.x);
            }
            if (std::isfinite(keypoint.y))
            {
                maxKeypointY = std::max(maxKeypointY, keypoint.y);
            }
        }
    }

    for (const ImagePair &pair : graph->imagePairs())
    {
        const std::vector<FeatureMatch> &matches = graph->matchesBetween(pair.first, pair.second);
        std::vector<MultiViewTrackBuilder::MatchIndexPair> indexedMatches;
        indexedMatches.reserve(matches.size());
        for (const FeatureMatch &match : matches)
        {
            if (match.idx1 != kInvalidFeatureIdx && match.idx2 != kInvalidFeatureIdx)
            {
                indexedMatches.emplace_back(match.idx1, match.idx2, match.score);
            }
        }
        if (!indexedMatches.empty())
        {
            builder.addMatchPair(pair.first, pair.second, indexedMatches);
        }
    }

    MultiViewTrackBuilder::BuildOptions buildOptions;
    buildOptions.enableQualityThinning =
        options.maxTracksPerImage > 0 || options.maxTracksPerGridCell > 0;
    buildOptions.maxTracksPerImage = options.maxTracksPerImage;
    buildOptions.maxTracksPerGridCell = options.maxTracksPerGridCell;
    buildOptions.gridColumns = options.gridColumns;
    buildOptions.gridRows = options.gridRows;
    buildOptions.imageWidth = std::max(1.0f, maxKeypointX + 1.0f);
    buildOptions.imageHeight = std::max(1.0f, maxKeypointY + 1.0f);

    // MultiViewTrackBuilder 在轨迹层应用每图/网格配额。保留集合确定后，再从原图
    // 删除不属于这些轨迹的边，确保剩余边仍全部来自真实 pairwise match。
    const MultiViewTrackBuildResult tracks = builder.build(buildOptions);
    result.retainedTrackCount = static_cast<int>(tracks.tracks.size());
    result.prunedTrackCount = tracks.prunedByQualityThinning;
    result.inputTrackCount = result.retainedTrackCount + result.prunedTrackCount;
    result.removedMatchCount = graph->retainMatchesInTracks(tracks.tracks);
    for (const ImagePair &pair : graph->imagePairs())
    {
        result.retainedMatchCount += graph->numMatchesBetween(pair.first, pair.second);
    }
    return result;
}

} // namespace xjw
