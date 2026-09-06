#include "tracks/CorrespondenceTrackThinner.h"

#include "graph/CorrespondenceGraph.h"
#include "reconstruction/SfmReconstruction.h"
#include "tracks/ReferenceTrackBuilder.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw
{

    CorrespondenceTrackThinningResult thinCorrespondenceTracks(const SfmReconstruction& reconstruction,
                                                               CorrespondenceGraph* graph,
                                                               const CorrespondenceTrackThinningOptions& options)
    {
        CorrespondenceTrackThinningResult result;
        if (!graph)
        {
            return result;
        }

        ReferenceTrackBuilder builder;

        // 先把对应图恢复成完整多视轨迹。直接按 pair 截断会打断跨多幅影像的同一
        // 物点，并让 SfM 中的 track length 统计失真。
        for (ImageId imageId : reconstruction.allImageIds())
        {
            const ImageData& image = reconstruction.image(imageId);
            builder.setImageKeypoints(imageId, image.keypoints);
        }

        for (const ImagePair& pair : graph->imagePairs())
        {
            const std::vector<FeatureMatch>& matches = graph->matchesBetween(pair.first, pair.second);
            std::vector<ReferenceTrackBuilder::MatchIndexPair> indexedMatches;
            indexedMatches.reserve(matches.size());
            for (const FeatureMatch& match : matches)
            {
                if (match.idx1 != kInvalidFeatureIdx && match.idx2 != kInvalidFeatureIdx)
                {
                    indexedMatches.push_back({match.idx1, match.idx2});
                }
            }
            if (!indexedMatches.empty())
            {
                builder.addMatchPair(pair.first, pair.second, indexedMatches);
            }
        }

        ReferenceTrackBuildOptions buildOptions;
        buildOptions.tiePointLimit = static_cast<std::size_t>(std::max(0, options.maxTracksPerImage));
        ReferenceTrackBuildResult tracks = builder.build(buildOptions);
        result.retainedTrackCount = static_cast<int>(tracks.tracks.size());
        result.prunedTrackCount = tracks.prunedBySpatialSelection;
        result.inputTrackCount = tracks.tracksBeforeSpatialSelection;
        result.removedMatchCount = graph->retainMatchesInTracks(tracks.tracks);
        for (const ImagePair& pair : graph->imagePairs())
        {
            result.retainedMatchCount += graph->numMatchesBetween(pair.first, pair.second);
        }
        result.retainedTracks = std::move(tracks.tracks);
        return result;
    }

} // namespace xjw
