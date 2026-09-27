#include "BaTrackBuilder.h"

#include "geometry/TriangulationQuality.h"
#include "tracks/ReferenceTrackBuilder.h"

#include <placamera/frame_numeric_state.h>

#include <algorithm>
#include <exception>
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

        using NumericCamera = placamera::FramePinholeNumericState;

        std::array<double, 3> midpointBetweenCameras(const NumericCamera& cameraA,
                                                     const NumericCamera& cameraB)
        {
            const auto& centerA = cameraA.pose().center;
            const auto& centerB = cameraB.pose().center;
            return {
                {0.5 * (centerA[0] + centerB[0]), 0.5 * (centerA[1] + centerB[1]), 0.5 * (centerA[2] + centerB[2])}};
        }

        plabundle::Track
        makeBaTrackFromIndexedTrack(const xjw::Track& track,
                                    const std::map<IndexedFeatureKey, IndexedObservation>& observationsByIndexedFeature,
                                    const std::vector<NumericCamera>& cameras)
        {
            plabundle::Track baTrack;
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
                    const auto candidate = NumericCamera::triangulatePair(
                        cameras[static_cast<std::size_t>(left.cameraIndex)],
                        placamera::ImageCoordinate{left.pixel[0], left.pixel[1]},
                        cameras[static_cast<std::size_t>(right.cameraIndex)],
                        placamera::ImageCoordinate{right.pixel[0], right.pixel[1]});
                    if (candidate)
                    {
                        baTrack.initialPoint = candidate.value().point.position;
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
                plabundle::Observation baObservation;
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

    bool appendBaTracks(const ProjectMatchInput& input, BaInputBuildResult* result)
    {
        if (!result)
        {
            return false;
        }

        std::vector<NumericCamera> cameras;
        cameras.reserve(input.cameraInstances.size());
        for (const auto& instance : input.cameraInstances)
        {
            if (!instance)
            {
                result->matchDiagnostics.firstCameraError = QStringLiteral("BA 轨迹包含空 PlaCamera 实例");
                return false;
            }
            try
            {
                cameras.push_back(NumericCamera::fromModel(*instance));
            }
            catch (const std::exception& exception)
            {
                result->matchDiagnostics.firstCameraError =
                    QStringLiteral("BA 轨迹无法建立 PlaCamera 数值状态：%1")
                        .arg(QString::fromUtf8(exception.what()));
                return false;
            }
        }
        if (cameras.size() != input.imageIdByIndex.size())
        {
            result->matchDiagnostics.firstCameraError =
                QStringLiteral("BA 轨迹的 PlaCamera 实例与 canonical ImageId 数量不一致");
            return false;
        }
        for (std::size_t index = 0; index < cameras.size(); ++index)
        {
            if (input.cameraInstances[index]->imageId() != input.imageIdByIndex[index])
            {
                result->matchDiagnostics.firstCameraError =
                    QStringLiteral("BA 轨迹的 PlaCamera 实例与 canonical ImageId 不一致（索引 %1）")
                        .arg(static_cast<qulonglong>(index));
                return false;
            }
        }

        // 新格式匹配具有稳定特征索引，可跨多个 pair 合并为同一物点。旧格式只有
        // 浮点坐标，没有可靠身份，只能保守地保留为双视 BA track。
        xjw::ReferenceTrackBuilder referenceTrackBuilder;
        std::map<IndexedFeatureKey, xjw::FeatureIdx> compactIndexByOriginal;
        std::map<IndexedFeatureKey, IndexedObservation> observationsByIndexedFeature;
        std::map<int, std::vector<xjw::FeatureKeypoint>> keypointsByCamera;
        for (const ProjectMatchPair& pair : input.pairs)
        {
            const NumericCamera& cameraA = cameras.at(static_cast<std::size_t>(pair.cameraIndexA));
            const NumericCamera& cameraB = cameras.at(static_cast<std::size_t>(pair.cameraIndexB));

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
                plabundle::Track track;
                const auto candidate = NumericCamera::triangulatePair(
                    cameraA,
                    placamera::ImageCoordinate{observation.pixelA[0], observation.pixelA[1]},
                    cameraB,
                    placamera::ImageCoordinate{observation.pixelB[0], observation.pixelB[1]});
                track.initialPoint = candidate ? candidate.value().point.position
                                               : midpointBetweenCameras(cameraA, cameraB);

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
            plabundle::Track baTrack = makeBaTrackFromIndexedTrack(track, observationsByIndexedFeature, cameras);
            if (baTrack.observations.size() >= 2)
            {
                result->tracks.push_back(std::move(baTrack));
            }
        }
        return true;
    }

} // namespace xjw::core::project
