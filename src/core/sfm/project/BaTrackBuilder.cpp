#include "BaTrackBuilder.h"

#include "geometry/TriangulationQuality.h"
#include "tracks/ReferenceTrackBuilder.h"

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
            double measurementScale = 1.0;
        };

        using IndexedFeatureKey = std::pair<int, xjw::FeatureIdx>;

        xjw::FeatureIdx
        rememberObservation(std::map<IndexedFeatureKey, xjw::FeatureIdx>* compactIndexByOriginal,
                            std::map<IndexedFeatureKey, IndexedObservation>* observationsByCompactFeature,
                            std::map<int, std::vector<xjw::FeatureKeypoint>>* keypointsByCamera,
                            int cameraIndex,
                            xjw::FeatureIdx featureIndex,
                            const std::array<double, 2>& pixel)
        {
            if (!compactIndexByOriginal || !observationsByCompactFeature || !keypointsByCamera || cameraIndex < 0 ||
                featureIndex == xjw::kInvalidFeatureIdx)
            {
                return xjw::kInvalidFeatureIdx;
            }

            const IndexedFeatureKey originalKey{cameraIndex, featureIndex};
            const auto existing = compactIndexByOriginal->find(originalKey);
            if (existing != compactIndexByOriginal->end())
            {
                return existing->second;
            }

            std::vector<xjw::FeatureKeypoint>& keypoints = (*keypointsByCamera)[cameraIndex];
            if (keypoints.size() >= static_cast<std::size_t>(xjw::kInvalidFeatureIdx))
            {
                return xjw::kInvalidFeatureIdx;
            }
            const xjw::FeatureIdx compactIndex = static_cast<xjw::FeatureIdx>(keypoints.size());
            keypoints.push_back({static_cast<float>(pixel[0]), static_cast<float>(pixel[1]), 1.0f});
            compactIndexByOriginal->emplace(originalKey, compactIndex);
            observationsByCompactFeature->emplace(IndexedFeatureKey{cameraIndex, compactIndex},
                                                  IndexedObservation{cameraIndex, pixel});
            return compactIndex;
        }

        std::array<double, 3> midpointBetweenCameras(const xjw::FramePinholeCamera& cameraA,
                                                     const xjw::FramePinholeCamera& cameraB)
        {
            const auto centerA = cameraA.cameraCenter();
            const auto centerB = cameraB.cameraCenter();
            return {
                {0.5 * (centerA[0] + centerB[0]), 0.5 * (centerA[1] + centerB[1]), 0.5 * (centerA[2] + centerB[2])}};
        }

        xjw::BATrack
        makeBaTrackFromIndexedTrack(const xjw::Track& track,
                                    const std::map<IndexedFeatureKey, IndexedObservation>& observationsByIndexedFeature,
                                    const std::vector<xjw::FramePinholeCamera>& cameras)
        {
            xjw::BATrack baTrack;
            std::vector<IndexedObservation> observations;
            observations.reserve(track.elements.size());
            for (const xjw::TrackElement& element : track.elements)
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

            // 依次寻找可交会的观测对，而不是固定使用轨迹前两个观测。弱基线或错误深度轴
            // 的首对不应让整条长轨迹失去可用初值。
            bool initialized = false;
            for (std::size_t leftIndex = 0; leftIndex < observations.size() && !initialized; ++leftIndex)
            {
                for (std::size_t rightIndex = leftIndex + 1; rightIndex < observations.size(); ++rightIndex)
                {
                    const IndexedObservation& left = observations[leftIndex];
                    const IndexedObservation& right = observations[rightIndex];
                    if (left.cameraIndex < 0 || right.cameraIndex < 0 ||
                        left.cameraIndex >= static_cast<int>(cameras.size()) ||
                        right.cameraIndex >= static_cast<int>(cameras.size()))
                    {
                        continue;
                    }
                    const xjw::PairIntersectionCandidate candidate =
                        xjw::triangulatePairWithDirectionFallback(cameras[static_cast<std::size_t>(left.cameraIndex)],
                                                                  left.pixel,
                                                                  cameras[static_cast<std::size_t>(right.cameraIndex)],
                                                                  right.pixel);
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
                baTrack.initialPoint =
                    midpointBetweenCameras(cameras[static_cast<std::size_t>(observations[0].cameraIndex)],
                                           cameras[static_cast<std::size_t>(observations[1].cameraIndex)]);
            }

            for (const IndexedObservation& observation : observations)
            {
                xjw::BAObservation baObservation;
                baObservation.cameraIndex = observation.cameraIndex;
                baObservation.u = observation.pixel[0];
                baObservation.v = observation.pixel[1];
                baObservation.weight = track.confidence;
                baObservation.measurementScale = observation.measurementScale;
                baTrack.observations.push_back(baObservation);
            }
            return baTrack;
        }

    } // namespace

    void appendBaTracks(const ProjectMatchInput& input, BaInputBuildResult* result)
    {
        if (!result)
        {
            return;
        }

        // 新格式匹配具有稳定特征索引，可跨多个 pair 合并为同一物点。旧格式只有
        // 浮点坐标，没有可靠身份，只能保守地保留为双视 BA track。
        xjw::ReferenceTrackBuilder referenceTrackBuilder;
        std::map<IndexedFeatureKey, xjw::FeatureIdx> compactIndexByOriginal;
        std::map<IndexedFeatureKey, IndexedObservation> observationsByIndexedFeature;
        std::map<int, std::vector<xjw::FeatureKeypoint>> keypointsByCamera;
        for (const ProjectMatchPair& pair : input.pairs)
        {
            const xjw::FramePinholeCamera& cameraA = input.cameras.at(static_cast<std::size_t>(pair.cameraIndexA));
            const xjw::FramePinholeCamera& cameraB = input.cameras.at(static_cast<std::size_t>(pair.cameraIndexB));

            if (pair.indexed)
            {
                std::vector<xjw::ReferenceTrackBuilder::MatchIndexPair> indexedMatches;
                indexedMatches.reserve(pair.observations.size());
                for (const ProjectMatchObservationPair& observation : pair.observations)
                {
                    const xjw::FeatureIdx featureA = rememberObservation(&compactIndexByOriginal,
                                                                         &observationsByIndexedFeature,
                                                                         &keypointsByCamera,
                                                                         pair.cameraIndexA,
                                                                         observation.featureA,
                                                                         observation.pixelA);
                    const xjw::FeatureIdx featureB = rememberObservation(&compactIndexByOriginal,
                                                                         &observationsByIndexedFeature,
                                                                         &keypointsByCamera,
                                                                         pair.cameraIndexB,
                                                                         observation.featureB,
                                                                         observation.pixelB);
                    if (featureA != xjw::kInvalidFeatureIdx && featureB != xjw::kInvalidFeatureIdx)
                    {
                        indexedMatches.push_back({featureA, featureB});
                    }
                }
                if (!indexedMatches.empty())
                {
                    referenceTrackBuilder.addMatchPair(static_cast<xjw::ImageId>(pair.cameraIndexA),
                                                       static_cast<xjw::ImageId>(pair.cameraIndexB),
                                                       indexedMatches);
                }
                continue;
            }

            for (const ProjectMatchObservationPair& observation : pair.observations)
            {
                xjw::BATrack track;
                const xjw::PairIntersectionCandidate candidate =
                    xjw::triangulatePairWithDirectionFallback(cameraA, observation.pixelA, cameraB, observation.pixelB);
                track.initialPoint = candidate.valid ? candidate.point : midpointBetweenCameras(cameraA, cameraB);

                const double weight = std::clamp(observation.score, 0.0, 1.0);
                track.observations.push_back({pair.cameraIndexA, observation.pixelA[0], observation.pixelA[1], weight});
                track.observations.push_back({pair.cameraIndexB, observation.pixelB[0], observation.pixelB[1], weight});
                result->tracks.push_back(std::move(track));
            }
        }

        for (const auto& [cameraIndex, keypoints] : keypointsByCamera)
        {
            referenceTrackBuilder.setImageKeypoints(static_cast<xjw::ImageId>(cameraIndex), keypoints);
        }

        // 与 MatchPhotos/SfM 使用相同的参考构轨语义：先合并全部已验证边，再删除
        // 同一轨迹内来自重复影像的全部冲突观测；独立 BA 不额外执行空间限额。
        const xjw::ReferenceTrackBuildResult referenceTracks = referenceTrackBuilder.build();
        result->multiViewTrackCount = static_cast<int>(referenceTracks.tracks.size());
        for (const xjw::Track& track : referenceTracks.tracks)
        {
            xjw::BATrack baTrack = makeBaTrackFromIndexedTrack(track, observationsByIndexedFeature, input.cameras);
            if (baTrack.observations.size() >= 2)
            {
                result->tracks.push_back(std::move(baTrack));
            }
        }
    }

} // namespace xjw::core::project
