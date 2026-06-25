#pragma once

#include "common/SfmTypes.h"

#include <map>
#include <utility>
#include <vector>

namespace xjw
{

struct MultiViewTrackBuildResult
{
    std::vector<Track> tracks;
    int totalComponents = 0;
    int acceptedComponents = 0;
    int rejectedConflictComponents = 0;
    int rejectedConflictEdges = 0;
    std::map<int, int> trackLengthHistogram;
    std::vector<double> trackConfidenceScores;
    double meanTrackConfidence = 0.0;
};

class MultiViewTrackBuilder
{
public:
    struct MatchIndexPair
    {
        FeatureIdx first = kInvalidFeatureIdx;
        FeatureIdx second = kInvalidFeatureIdx;
        float score = 1.0f;

        MatchIndexPair() = default;
        MatchIndexPair(FeatureIdx featureA, FeatureIdx featureB, float matchScore = 1.0f)
            : first(featureA), second(featureB), score(matchScore)
        {
        }
    };

    struct ObservationKey
    {
        ImageId imageId = kInvalidImageId;
        FeatureIdx featureIdx = kInvalidFeatureIdx;

        bool operator<(const ObservationKey &other) const
        {
            if (imageId != other.imageId)
            {
                return imageId < other.imageId;
            }
            return featureIdx < other.featureIdx;
        }
    };

    void addMatchPair(ImageId imageA, ImageId imageB, const std::vector<MatchIndexPair> &matches);
    MultiViewTrackBuildResult build() const;

private:
    struct Edge
    {
        ObservationKey first;
        ObservationKey second;
        float score = 1.0f;
        int insertionOrder = 0;
    };

    std::vector<Edge> _edges;
};

} // namespace xjw
