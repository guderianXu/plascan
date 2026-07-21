#include "BaTrackBuilder.h"

#include "geometry/TriangulationQuality.h"
#include "tracks/MultiViewTrackBuilder.h"

#include <algorithm>
#include <map>
#include <utility>

namespace xjw::core::project
{
namespace
{

struct IndexedObservation
{
    int cameraIndex = -1;
    std::array<double, 2> pixel{{0.0, 0.0}};
};

using IndexedFeatureKey = std::pair<int, xjw::FeatureIdx>;

void rememberObservation(std::map<IndexedFeatureKey, IndexedObservation> *observations,
                         int cameraIndex,
                         xjw::FeatureIdx featureIndex,
                         const std::array<double, 2> &pixel)
{
    if (!observations || featureIndex == xjw::kInvalidFeatureIdx)
    {
        return;
    }
    observations->emplace(IndexedFeatureKey{cameraIndex, featureIndex},
                          IndexedObservation{cameraIndex, pixel});
}

std::array<double, 3> midpointBetweenCameras(const xjw::Camera &cameraA,
                                             const xjw::Camera &cameraB)
{
    const auto centerA = cameraA.cameraCenter();
    const auto centerB = cameraB.cameraCenter();
    return {{0.5 * (centerA[0] + centerB[0]),
             0.5 * (centerA[1] + centerB[1]),
             0.5 * (centerA[2] + centerB[2])}};
}

xjw::BATrack makeBaTrackFromIndexedTrack(
    const xjw::Track &track,
    const std::map<IndexedFeatureKey, IndexedObservation> &observationsByIndexedFeature,
    const std::vector<xjw::Camera> &cameras)
{
    xjw::BATrack baTrack;
    std::vector<IndexedObservation> observations;
    observations.reserve(track.elements.size());
    for (const xjw::TrackElement &element : track.elements)
    {
        const auto found = observationsByIndexedFeature.find(
            IndexedFeatureKey{static_cast<int>(element.imageId), element.featureIdx});
        if (found != observationsByIndexedFeature.end())
        {
            observations.push_back(found->second);
        }
    }

    if (observations.size() < 2)
    {
        return baTrack;
    }

    bool initialized = false;
    for (std::size_t leftIndex = 0; leftIndex < observations.size() && !initialized; ++leftIndex)
    {
        for (std::size_t rightIndex = leftIndex + 1;
             rightIndex < observations.size();
             ++rightIndex)
        {
            const IndexedObservation &left = observations[leftIndex];
            const IndexedObservation &right = observations[rightIndex];
            if (left.cameraIndex < 0 || right.cameraIndex < 0
                || left.cameraIndex >= static_cast<int>(cameras.size())
                || right.cameraIndex >= static_cast<int>(cameras.size()))
            {
                continue;
            }
            const xjw::PairIntersectionCandidate candidate =
                xjw::triangulatePairWithDirectionFallback(
                    cameras[static_cast<std::size_t>(left.cameraIndex)], left.pixel,
                    cameras[static_cast<std::size_t>(right.cameraIndex)], right.pixel);
            if (candidate.valid)
            {
                baTrack.initialPoint = candidate.point;
                initialized = true;
                break;
            }
        }
    }

    if (!initialized)
    {
        baTrack.initialPoint = midpointBetweenCameras(
            cameras[static_cast<std::size_t>(observations[0].cameraIndex)],
            cameras[static_cast<std::size_t>(observations[1].cameraIndex)]);
    }

    for (const IndexedObservation &observation : observations)
    {
        xjw::BAObservation baObservation;
        baObservation.cameraIndex = observation.cameraIndex;
        baObservation.u = observation.pixel[0];
        baObservation.v = observation.pixel[1];
        baObservation.weight = track.confidence;
        baTrack.observations.push_back(baObservation);
    }
    return baTrack;
}

} // namespace

void appendBaTracks(const ProjectMatchInput &input, BaInputBuildResult *result)
{
    if (!result)
    {
        return;
    }

    xjw::MultiViewTrackBuilder multiViewTrackBuilder;
    std::map<IndexedFeatureKey, IndexedObservation> observationsByIndexedFeature;
    for (const ProjectMatchPair &pair : input.pairs)
    {
        const xjw::Camera &cameraA = input.cameras.at(
            static_cast<std::size_t>(pair.cameraIndexA));
        const xjw::Camera &cameraB = input.cameras.at(
            static_cast<std::size_t>(pair.cameraIndexB));

        if (pair.indexed)
        {
            std::vector<xjw::MultiViewTrackBuilder::MatchIndexPair> indexedMatches;
            indexedMatches.reserve(pair.observations.size());
            for (const ProjectMatchObservationPair &observation : pair.observations)
            {
                indexedMatches.emplace_back(observation.featureA,
                                            observation.featureB,
                                            static_cast<float>(observation.score));
                rememberObservation(&observationsByIndexedFeature,
                                    pair.cameraIndexA,
                                    observation.featureA,
                                    observation.pixelA);
                rememberObservation(&observationsByIndexedFeature,
                                    pair.cameraIndexB,
                                    observation.featureB,
                                    observation.pixelB);
            }
            if (!indexedMatches.empty())
            {
                multiViewTrackBuilder.addMatchPair(
                    static_cast<xjw::ImageId>(pair.cameraIndexA),
                    static_cast<xjw::ImageId>(pair.cameraIndexB),
                    indexedMatches);
            }
            continue;
        }

        for (const ProjectMatchObservationPair &observation : pair.observations)
        {
            xjw::BATrack track;
            const xjw::PairIntersectionCandidate candidate =
                xjw::triangulatePairWithDirectionFallback(
                    cameraA, observation.pixelA, cameraB, observation.pixelB);
            track.initialPoint = candidate.valid
                ? candidate.point
                : midpointBetweenCameras(cameraA, cameraB);

            const double weight = std::clamp(observation.score, 0.0, 1.0);
            track.observations.push_back({pair.cameraIndexA,
                                          observation.pixelA[0],
                                          observation.pixelA[1],
                                          weight});
            track.observations.push_back({pair.cameraIndexB,
                                          observation.pixelB[0],
                                          observation.pixelB[1],
                                          weight});
            result->tracks.push_back(std::move(track));
        }
    }

    const xjw::MultiViewTrackBuildResult multiViewResult = multiViewTrackBuilder.build();
    result->multiViewTrackCount = static_cast<int>(multiViewResult.tracks.size());
    result->rejectedConflictTrackCount = multiViewResult.rejectedConflictComponents;
    for (const xjw::Track &track : multiViewResult.tracks)
    {
        xjw::BATrack baTrack = makeBaTrackFromIndexedTrack(
            track, observationsByIndexedFeature, input.cameras);
        if (baTrack.observations.size() >= 2)
        {
            result->tracks.push_back(std::move(baTrack));
        }
    }
}

} // namespace xjw::core::project
