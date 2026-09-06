#include "Triangulator.h"
#include "concurrency/SafeWorkerGroup.h"

#include "log/Logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <limits>
#include <thread>
#include <unordered_map>
#include <utility>

namespace xjw
{

    Triangulator::Triangulator(SfmReconstruction& reconstruction, const CorrespondenceGraph& graph, int threadCount)
        : _reconstruction(reconstruction), _correspondenceGraph(graph),
          _threadCount(threadCount > 0 ? threadCount
                                       : static_cast<int>(std::max(1u, std::thread::hardware_concurrency())))
    {
    }

    /**
     * @brief 三角化器构造函数。
     *
     * 持有对 `SfmReconstruction` 与 `CorrespondenceGraph` 的引用，
     * 在增量注册过程中执行三角化与过滤任务。
     */

    // ---- 核心：对新注册图像的特征执行三角化 ----

    TriangulationStats Triangulator::triangulateImage(ImageId imageId, const TriangulatorOptions& options)
    {
        TriangulationStats stats;

        if (!_reconstruction.isRegistered(imageId))
        {
            return stats;
        }
        const ImageData& imgData = _reconstruction.image(imageId);

        const size_t numFeatures = imgData.keypoints.size();
        for (FeatureIdx fi = 0; fi < static_cast<FeatureIdx>(numFeatures); ++fi)
        {
            // 如果该特征已经关联三维点，跳过
            if (fi < imgData.point3DIds.size() && imgData.point3DIds[fi] != kInvalidPoint3DId)
            {
                continue;
            }

            // 查找该特征在其他图像中的对应点
            auto corrs = _correspondenceGraph.findCorrespondences(imageId, fi);
            if (corrs.empty())
            {
                continue;
            }

            bool created = false;

            for (const auto& corr : corrs)
            {
                // 对应图像必须已注册
                if (!_reconstruction.isRegistered(corr.imageId))
                {
                    continue;
                }
                const ImageData& otherImg = _reconstruction.image(corr.imageId);

                // 情况1：对应特征已关联三维点 → 延续轨迹
                if (corr.featureIdx < otherImg.point3DIds.size() &&
                    otherImg.point3DIds[corr.featureIdx] != kInvalidPoint3DId)
                {
                    Point3DId p3dId = otherImg.point3DIds[corr.featureIdx];
                    if (!_reconstruction.hasPoint3D(p3dId))
                    {
                        continue;
                    }

                    // 检查重投影误差
                    const auto& pt = _reconstruction.point3D(p3dId);
                    double reprErr = normalizedReprojError(computeReprojError(pt.xyz, imageId, fi),
                                                           imageId,
                                                           fi,
                                                           options.normalizeReprojectionByFeatureScale);
                    if (reprErr <= options.continueMaxReprojError)
                    {
                        // 追加观测到轨迹
                        auto& mutPt = _reconstruction.point3D(p3dId);
                        mutPt.track.elements.push_back({imageId, fi});

                        // 更新图像的三维点关联
                        auto& p3dIds = _reconstruction.image(imageId).point3DIds;
                        if (fi < p3dIds.size())
                        {
                            p3dIds[fi] = p3dId;
                        }

                        stats.numContinued++;
                        created = true;
                        break;
                    }
                }
            }

            if (created)
            {
                continue;
            }

            // 情况2：尝试与已注册图像的未关联特征进行三角化
            for (const auto& corr : corrs)
            {
                if (!_reconstruction.isRegistered(corr.imageId))
                {
                    continue;
                }

                // 新点只能使用尚未归属三维点的两端观测。若另一端已经属于已有点，
                // 即使轨迹延续因重投影误差失败，也不能覆盖其 point3DId，否则同一
                // 2D 特征会同时出现在多个三维点轨迹中。
                const ImageData& otherImg = _reconstruction.image(corr.imageId);
                if (corr.featureIdx >= otherImg.point3DIds.size() ||
                    otherImg.point3DIds[corr.featureIdx] != kInvalidPoint3DId)
                {
                    continue;
                }

                Track track;
                track.elements.push_back({imageId, fi});
                track.elements.push_back({corr.imageId, corr.featureIdx});
                const std::string first_source = _correspondenceGraph.priorTrackId(imageId, fi);
                const std::string second_source = _correspondenceGraph.priorTrackId(corr.imageId, corr.featureIdx);
                if (!first_source.empty() && first_source == second_source)
                {
                    track.source = TrackSource::PriorMarker;
                    track.sourceId = first_source;
                }

                // 初始影像对必须建立启动点；注册第三张影像后，孤立的纯两视组件
                // 先留在对应图中等待第三视图，不再立即固化为容易飞散的三维点。
                const bool bootstrapPair = _reconstruction.numRegisteredImages() <= 2;
                if (options.deferPureTwoViewTracks && !bootstrapPair && isPureTwoViewComponent(track))
                {
                    ++stats.deferredPureTwoViewTracks;
                    continue;
                }

                std::array<double, 3> xyz;
                if (triangulatePair(imageId, fi, corr.imageId, corr.featureIdx, options, xyz))
                {
                    _reconstruction.addPoint3DWithTrack(xyz, track);

                    stats.numCreated++;
                    break;
                }
            }
        }

        return stats;
    }

    TriangulationStats Triangulator::triangulateTracks(const std::vector<Track>& tracks,
                                                       const TriangulatorOptions& options)
    {
        enum class Decision : std::uint8_t
        {
            Skipped,
            Unusable,
            Deferred,
            GeometryRejected,
            DepthRejected,
            ReprojectionRejected,
            Accepted
        };
        struct Candidate
        {
            std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
            std::vector<TrackElement> registeredElements;
            bool usesCompleteInputTrack = true;
            double meanError = 0.0;
            Decision decision = Decision::Skipped;
        };
        struct Scratch
        {
            std::vector<TrackElement> registeredElements;
        };

        TriangulationStats stats;
        stats.inputTracks = static_cast<int>(tracks.size());
        stats.inputLongTracks = static_cast<int>(
            std::count_if(tracks.begin(), tracks.end(), [](const Track& track) { return track.length() >= 3; }));

        std::vector<std::size_t> availableTrackIndices;
        std::vector<Point3DId> availablePointSlots;
        availableTrackIndices.reserve(tracks.size());
        availablePointSlots.reserve(tracks.size());
        for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
        {
            const Track& track = tracks[trackIndex];
            Point3DId pointSlot = kInvalidPoint3DId;
            if (options.bindCompleteInputTrack && !track.elements.empty())
            {
                for (const TrackElement& element : track.elements)
                {
                    if (!_reconstruction.hasImage(element.imageId))
                    {
                        continue;
                    }
                    const ImageData& image = _reconstruction.image(element.imageId);
                    if (element.featureIdx >= image.point3DIds.size())
                    {
                        continue;
                    }
                    const Point3DId pointId = image.point3DIds[element.featureIdx];
                    if (pointId == kInvalidPoint3DId)
                    {
                        continue;
                    }
                    pointSlot = pointId;
                    break;
                }
                if (pointSlot != kInvalidPoint3DId && _reconstruction.hasPoint3D(pointSlot))
                {
                    continue;
                }
                if (pointSlot != kInvalidPoint3DId && !_reconstruction.hasInactivePoint3D(pointSlot))
                {
                    pointSlot = kInvalidPoint3DId;
                }
            }
            availableTrackIndices.push_back(trackIndex);
            availablePointSlots.push_back(pointSlot);
        }

        if (availableTrackIndices.empty())
        {
            return stats;
        }

        std::unordered_map<ImageId, double> sensorThresholds;
        for (const ImageId imageId : _reconstruction.registeredImageIds())
        {
            if (!_reconstruction.hasCamera(imageId) || !_reconstruction.hasImage(imageId))
            {
                continue;
            }
            const FramePinholeCamera& camera = _reconstruction.camera(imageId);
            double width = 0.0;
            double height = 0.0;
            if (camera.imageSize() && camera.imageSize()->samples > 0 && camera.imageSize()->lines > 0)
            {
                width = static_cast<double>(camera.imageSize()->samples);
                height = static_cast<double>(camera.imageSize()->lines);
            }
            else
            {
                const ImageData& image = _reconstruction.image(imageId);
                for (const FeatureKeypoint& keypoint : image.keypoints)
                {
                    width = std::max(width, static_cast<double>(keypoint.x) + 1.0);
                    height = std::max(height, static_cast<double>(keypoint.y) + 1.0);
                }
            }
            sensorThresholds.emplace(imageId, 0.002 * 0.5 * (std::max(1.0, width) + std::max(1.0, height)));
        }

        std::vector<Candidate> candidates(availableTrackIndices.size());
        const auto triangulateTrack = [&](std::size_t candidateIndex, Scratch& scratch)
        {
            const std::size_t trackIndex = availableTrackIndices[candidateIndex];
            const Point3DId pointSlot = availablePointSlots[candidateIndex];
            const Track& inputTrack = tracks[trackIndex];
            Candidate& candidate = candidates[candidateIndex];
            if (inputTrack.length() < 2)
            {
                return;
            }

            scratch.registeredElements.clear();
            scratch.registeredElements.reserve(inputTrack.elements.size());
            for (const TrackElement& element : inputTrack.elements)
            {
                if (!_reconstruction.isRegistered(element.imageId))
                {
                    continue;
                }
                if (!_reconstruction.hasCamera(element.imageId) || !_reconstruction.hasImage(element.imageId))
                {
                    candidate.decision = Decision::Unusable;
                    return;
                }
                const ImageData& image = _reconstruction.image(element.imageId);
                if (element.featureIdx >= image.keypoints.size() || element.featureIdx >= image.point3DIds.size())
                {
                    candidate.decision = Decision::Unusable;
                    return;
                }
                const Point3DId observationPoint = image.point3DIds[element.featureIdx];
                if (observationPoint != kInvalidPoint3DId && observationPoint != pointSlot)
                {
                    candidate.decision = Decision::Unusable;
                    return;
                }
                scratch.registeredElements.push_back(element);
            }
            if (scratch.registeredElements.size() < 2)
            {
                candidate.decision = Decision::Skipped;
                return;
            }

            const bool allElementsRegistered = scratch.registeredElements.size() == inputTrack.elements.size();
            candidate.usesCompleteInputTrack = options.bindCompleteInputTrack || allElementsRegistered;
            const std::vector<TrackElement>& registeredElements =
                allElementsRegistered ? inputTrack.elements : scratch.registeredElements;
            const bool bootstrapPair = _reconstruction.numRegisteredImages() <= 2;
            Track registeredTrack;
            if (!allElementsRegistered)
            {
                registeredTrack.source = inputTrack.source;
                registeredTrack.sourceId = inputTrack.sourceId;
                registeredTrack.elements = registeredElements;
            }
            const Track& activeTrack = allElementsRegistered ? inputTrack : registeredTrack;
            if (options.deferPureTwoViewTracks && !bootstrapPair && isPureTwoViewComponent(activeTrack))
            {
                candidate.decision = Decision::Deferred;
                return;
            }

            if (!triangulateMultiView(registeredElements, candidate.xyz))
            {
                candidate.decision = Decision::GeometryRejected;
                return;
            }

            double errorSum = 0.0;
            for (const TrackElement& element : registeredElements)
            {
                if (!hasPositiveDepth(candidate.xyz, element.imageId))
                {
                    candidate.decision = Decision::DepthRejected;
                    return;
                }
                const double error = computeReprojError(candidate.xyz, element.imageId, element.featureIdx);
                const double normalizedError = normalizedReprojError(error, element.imageId, element.featureIdx, true);
                const auto thresholdIt = sensorThresholds.find(element.imageId);
                const double sensorThreshold =
                    thresholdIt != sensorThresholds.end() ? thresholdIt->second : options.completeMaxReprojError;
                const double requestedThreshold =
                    options.normalizeReprojectionByFeatureScale && options.completeMaxReprojError > 0.0
                        ? std::min(options.completeMaxReprojError, sensorThreshold)
                        : sensorThreshold;
                if (!std::isfinite(normalizedError) || normalizedError > requestedThreshold)
                {
                    candidate.decision = Decision::ReprojectionRejected;
                    return;
                }
                errorSum += error;
            }

            candidate.meanError = errorSum / static_cast<double>(registeredElements.size());
            if (!allElementsRegistered)
            {
                candidate.registeredElements = registeredElements;
            }
            candidate.decision = Decision::Accepted;
        };

        constexpr std::size_t kChunkSize = 100;
        const std::size_t workerCount =
            std::min(availableTrackIndices.size(), static_cast<std::size_t>(std::max(1, _threadCount)));
        if (workerCount <= 1)
        {
            Scratch scratch;
            for (std::size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex)
            {
                triangulateTrack(candidateIndex, scratch);
            }
        }
        else
        {
            std::atomic<std::size_t> nextTrack{0};
            common::concurrency::runWorkerGroup(
                workerCount,
                [&](std::stop_token stopToken)
                {
                    Scratch scratch;
                    while (!stopToken.stop_requested())
                    {
                        const std::size_t begin = nextTrack.fetch_add(kChunkSize, std::memory_order_relaxed);
                        if (begin >= candidates.size())
                        {
                            break;
                        }
                        const std::size_t end = std::min(candidates.size(), begin + kChunkSize);
                        for (std::size_t candidateIndex = begin; candidateIndex < end; ++candidateIndex)
                        {
                            triangulateTrack(candidateIndex, scratch);
                        }
                    }
                });
        }

        for (std::size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex)
        {
            const std::size_t trackIndex = availableTrackIndices[candidateIndex];
            const Point3DId pointSlot = availablePointSlots[candidateIndex];
            Candidate& candidate = candidates[candidateIndex];
            switch (candidate.decision)
            {
            case Decision::Skipped:
                continue;
            case Decision::Unusable:
                ++stats.unusableTracks;
                continue;
            case Decision::Deferred:
                ++stats.deferredPureTwoViewTracks;
                continue;
            case Decision::GeometryRejected:
                ++stats.noCandidateTracks;
                continue;
            case Decision::DepthRejected:
                ++stats.noCandidateTracks;
                ++stats.depthObservationRejected;
                continue;
            case Decision::ReprojectionRejected:
                ++stats.noCandidateTracks;
                ++stats.reprojObservationRejected;
                continue;
            case Decision::Accepted:
                break;
            }

            Track partialTrack;
            if (!candidate.usesCompleteInputTrack)
            {
                partialTrack.source = tracks[trackIndex].source;
                partialTrack.sourceId = tracks[trackIndex].sourceId;
                partialTrack.elements = std::move(candidate.registeredElements);
            }
            const Track& acceptedTrack = candidate.usesCompleteInputTrack ? tracks[trackIndex] : partialTrack;
            const bool observationsStillAvailable =
                std::all_of(acceptedTrack.elements.begin(),
                            acceptedTrack.elements.end(),
                            [this, pointSlot](const TrackElement& element)
                            {
                                if (!_reconstruction.hasImage(element.imageId))
                                {
                                    return false;
                                }
                                const ImageData& image = _reconstruction.image(element.imageId);
                                if (element.featureIdx >= image.point3DIds.size())
                                {
                                    return false;
                                }
                                const Point3DId observationPoint = image.point3DIds[element.featureIdx];
                                return observationPoint == kInvalidPoint3DId || observationPoint == pointSlot;
                            });
            if (!observationsStillAvailable)
            {
                ++stats.unusableTracks;
                continue;
            }

            Point3DId pointId = pointSlot;
            if (pointSlot != kInvalidPoint3DId && _reconstruction.hasInactivePoint3D(pointSlot))
            {
                if (!_reconstruction.restorePoint3DWithTrack(
                        pointSlot, candidate.xyz, acceptedTrack, candidate.meanError))
                {
                    ++stats.unusableTracks;
                    continue;
                }
                ++stats.numRestored;
            }
            else
            {
                pointId = _reconstruction.addPoint3DWithTrack(candidate.xyz, acceptedTrack);
                _reconstruction.point3D(pointId).error = candidate.meanError;
            }
            ++stats.numCreated;
            stats.numContinued += static_cast<int>(acceptedTrack.length() - 2);
            if (acceptedTrack.length() >= 3)
            {
                ++stats.createdLongTracks;
            }
            else
            {
                ++stats.createdTwoViewTracks;
            }
        }
        return stats;
    }

    // ---- 双目三角化 ----

    bool Triangulator::triangulatePair(ImageId imgId1,
                                       FeatureIdx featIdx1,
                                       ImageId imgId2,
                                       FeatureIdx featIdx2,
                                       const TriangulatorOptions& options,
                                       std::array<double, 3>& outXyz)
    {
        if (!_reconstruction.hasCamera(imgId1) || !_reconstruction.hasCamera(imgId2))
        {
            return false;
        }

        const FramePinholeCamera& cam1 = _reconstruction.camera(imgId1);
        const FramePinholeCamera& cam2 = _reconstruction.camera(imgId2);

        const ImageData& img1 = _reconstruction.image(imgId1);
        const ImageData& img2 = _reconstruction.image(imgId2);

        if (featIdx1 >= img1.keypoints.size() || featIdx2 >= img2.keypoints.size())
        {
            return false;
        }

        double u1 = img1.keypoints[featIdx1].x;
        double v1 = img1.keypoints[featIdx1].y;
        double u2 = img2.keypoints[featIdx2].x;
        double v2 = img2.keypoints[featIdx2].y;

        // 使用 Intersection 模块进行前方交汇
        auto result = Intersection::intersectPair(cam1, u1, v1, cam2, u2, v2);

        if (!result.valid)
        {
            return false;
        }
        if (result.angle_deg < options.minTriAngle)
        {
            return false;
        }
        double threshold_error = result.reproj_error_rms;
        if (options.normalizeReprojectionByFeatureScale && result.valid)
        {
            const double first_error =
                normalizedReprojError(computeReprojError(result.point, imgId1, featIdx1), imgId1, featIdx1, true);
            const double second_error =
                normalizedReprojError(computeReprojError(result.point, imgId2, featIdx2), imgId2, featIdx2, true);
            threshold_error = std::sqrt(0.5 * (first_error * first_error + second_error * second_error));
        }
        if (!std::isfinite(threshold_error) || threshold_error > options.maxReprojError)
            return false;

        outXyz = result.point;
        return true;
    }

    // ---- 重投影误差计算 ----

    double
    Triangulator::computeReprojError(const std::array<double, 3>& xyz, ImageId imageId, FeatureIdx featureIdx) const
    {
        if (!_reconstruction.hasCamera(imageId))
        {
            return 1e9;
        }
        const FramePinholeCamera& cam = _reconstruction.camera(imageId);
        const ImageData& img = _reconstruction.image(imageId);

        if (featureIdx >= img.keypoints.size())
        {
            return 1e9;
        }

        double uv[2];
        double world[3] = {xyz[0], xyz[1], xyz[2]};
        if (!cam.projectWorldPoint(world, uv))
        {
            return 1e9;
        }

        double du = uv[0] - img.keypoints[featureIdx].x;
        double dv = uv[1] - img.keypoints[featureIdx].y;
        return std::sqrt(du * du + dv * dv);
    }

    double Triangulator::normalizedReprojError(double error,
                                               ImageId imageId,
                                               FeatureIdx featureIdx,
                                               bool normalizeByFeatureScale) const
    {
        if (!normalizeByFeatureScale || !std::isfinite(error) || !_reconstruction.hasImage(imageId))
        {
            return error;
        }
        const ImageData& image = _reconstruction.image(imageId);
        if (featureIdx >= image.keypoints.size())
        {
            return error;
        }
        const float stored_scale = image.keypoints[featureIdx].scale;
        const double scale =
            std::isfinite(stored_scale) && stored_scale > 0.0F ? static_cast<double>(stored_scale) : 1.0;
        return error / scale;
    }

    bool Triangulator::isPureTwoViewComponent(const Track& track) const
    {
        if (track.source != TrackSource::FeatureMatch || track.length() != 2)
        {
            return false;
        }

        const TrackElement& first = track.elements[0];
        const TrackElement& second = track.elements[1];
        const auto isOnlyPeer =
            [](std::span<const CorrespondenceGraph::Correspondence> correspondences, const TrackElement& peer)
        {
            return correspondences.size() == 1 && correspondences.front().imageId == peer.imageId &&
                   correspondences.front().featureIdx == peer.featureIdx;
        };

        return isOnlyPeer(_correspondenceGraph.findCorrespondences(first.imageId, first.featureIdx), second) &&
               isOnlyPeer(_correspondenceGraph.findCorrespondences(second.imageId, second.featureIdx), first);
    }

    // ---- 过滤低质量三维点 ----

    int Triangulator::filterPoints(double maxReprojError, double minTriAngle, bool normalizeByFeatureScale)
    {
        auto allIds = _reconstruction.allPoint3DIds();
        struct FilterDecision
        {
            bool deletePoint = false;
            std::vector<TrackElement> observationsToRemove;
            double retainedRms = std::numeric_limits<double>::infinity();
        };
        std::vector<FilterDecision> decisions(allIds.size());

        common::concurrency::parallelForIndices(
            allIds.size(),
            static_cast<std::size_t>(_threadCount),
            [&](std::size_t pointIndex)
            {
                const Point3DId pid = allIds[pointIndex];
                if (!_reconstruction.hasPoint3D(pid))
                {
                    return;
                }
                const auto& pt = _reconstruction.point3D(pid);
                FilterDecision decision;
                std::vector<TrackElement> retainedObservations;
                retainedObservations.reserve(pt.track.elements.size());
                double squaredErrorSum = 0.0;
                for (const TrackElement& element : pt.track.elements)
                {
                    const double error =
                        normalizedReprojError(computeReprojError(pt.xyz, element.imageId, element.featureIdx),
                                              element.imageId,
                                              element.featureIdx,
                                              normalizeByFeatureScale);
                    if (!std::isfinite(error) || error > maxReprojError)
                    {
                        decision.observationsToRemove.push_back(element);
                        continue;
                    }
                    retainedObservations.push_back(element);
                    squaredErrorSum += error * error;
                }

                if (retainedObservations.size() < 2)
                {
                    decision.deletePoint = true;
                }
                else
                {
                    const double maxAngle = computeMaxTriangulationAngle(pt.xyz, retainedObservations);
                    decision.deletePoint = maxAngle < minTriAngle;
                    decision.retainedRms =
                        std::sqrt(squaredErrorSum / static_cast<double>(retainedObservations.size()));
                }
                decisions[pointIndex] = std::move(decision);
            });

        int numFiltered = 0;
        int numRemovedObservations = 0;
        for (std::size_t pointIndex = 0; pointIndex < allIds.size(); ++pointIndex)
        {
            const Point3DId pointId = allIds[pointIndex];
            if (!_reconstruction.hasPoint3D(pointId))
            {
                continue;
            }
            const FilterDecision& decision = decisions[pointIndex];
            if (decision.deletePoint)
            {
                _reconstruction.deletePoint3D(pointId);
                ++numFiltered;
                continue;
            }

            for (const TrackElement& element : decision.observationsToRemove)
            {
                if (_reconstruction.removeObservation(pointId, element.imageId, element.featureIdx))
                {
                    ++numRemovedObservations;
                }
            }
            if (_reconstruction.hasPoint3D(pointId))
            {
                _reconstruction.point3D(pointId).error = decision.retainedRms;
            }
        }

        if (numFiltered > 0 || numRemovedObservations > 0)
        {
            Logger::instance()->infof("[Triangulator] filterPoints: deletedPoints=%d removedObservations=%d",
                                      numFiltered,
                                      numRemovedObservations);
        }

        return numFiltered;
    }

    // ---- 过滤短轨迹 ----

    int Triangulator::filterShortTracks(int minTrackLen)
    {
        auto allIds = _reconstruction.allPoint3DIds();
        std::vector<char> shouldFilter(allIds.size(), 0);

        common::concurrency::parallelForIndices(allIds.size(),
                                                static_cast<std::size_t>(_threadCount),
                                                [&](std::size_t pointIndex)
                                                {
                                                    const Point3DId pid = allIds[pointIndex];
                                                    if (!_reconstruction.hasPoint3D(pid))
                                                    {
                                                        return;
                                                    }
                                                    const auto& pt = _reconstruction.point3D(pid);

                                                    // 统计有效观测数量（对应已注册图像的观测）
                                                    int validObs = 0;
                                                    for (const auto& elem : pt.track.elements)
                                                    {
                                                        if (_reconstruction.isRegistered(elem.imageId))
                                                        {
                                                            ++validObs;
                                                        }
                                                    }

                                                    if (validObs < minTrackLen)
                                                    {
                                                        shouldFilter[pointIndex] = 1;
                                                    }
                                                });

        int numFiltered = 0;
        for (std::size_t pointIndex = 0; pointIndex < allIds.size(); ++pointIndex)
        {
            if (shouldFilter[pointIndex])
            {
                _reconstruction.deletePoint3D(allIds[pointIndex]);
                ++numFiltered;
            }
        }

        return numFiltered;
    }

    // ---- 补全轨迹 ----

    int Triangulator::completeTracks(const TriangulatorOptions& options)
    {
        int numCompleted = 0;
        auto regIds = _reconstruction.registeredImageIds();

        for (ImageId imgId : regIds)
        {
            const ImageData& imgData = _reconstruction.image(imgId);
            const size_t numFeatures = imgData.keypoints.size();

            for (FeatureIdx fi = 0; fi < static_cast<FeatureIdx>(numFeatures); ++fi)
            {
                // 只处理未关联三维点的特征
                if (fi < imgData.point3DIds.size() && imgData.point3DIds[fi] != kInvalidPoint3DId)
                {
                    continue;
                }

                auto corrs = _correspondenceGraph.findCorrespondences(imgId, fi);
                for (const auto& corr : corrs)
                {
                    if (!_reconstruction.isRegistered(corr.imageId))
                    {
                        continue;
                    }
                    const ImageData& otherImg = _reconstruction.image(corr.imageId);
                    if (corr.featureIdx >= otherImg.point3DIds.size())
                    {
                        continue;
                    }

                    Point3DId p3dId = otherImg.point3DIds[corr.featureIdx];
                    if (p3dId == kInvalidPoint3DId)
                    {
                        continue;
                    }
                    if (!_reconstruction.hasPoint3D(p3dId))
                    {
                        continue;
                    }

                    const auto& pt = _reconstruction.point3D(p3dId);
                    const bool alreadyHasObservationInImage =
                        std::any_of(pt.track.elements.begin(),
                                    pt.track.elements.end(),
                                    [imgId](const TrackElement& element) { return element.imageId == imgId; });
                    if (alreadyHasObservationInImage)
                    {
                        continue;
                    }

                    // 检查重投影误差
                    double reprErr = normalizedReprojError(
                        computeReprojError(pt.xyz, imgId, fi), imgId, fi, options.normalizeReprojectionByFeatureScale);
                    if (reprErr > options.completeMaxReprojError)
                    {
                        continue;
                    }

                    // 深度一致性检查：确保 3D 点在该相机前方
                    if (!hasPositiveDepth(pt.xyz, imgId))
                    {
                        continue;
                    }

                    auto& mutPt = _reconstruction.point3D(p3dId);
                    mutPt.track.elements.push_back({imgId, fi});
                    auto& p3dIds = _reconstruction.image(imgId).point3DIds;
                    if (fi < p3dIds.size())
                    {
                        p3dIds[fi] = p3dId;
                    }
                    ++numCompleted;
                    break;
                }
            }
        }

        return numCompleted;
    }

    // ---- 深度一致性检查 ----

    bool Triangulator::hasPositiveDepth(const std::array<double, 3>& xyz, ImageId imageId) const
    {
        if (!_reconstruction.hasCamera(imageId))
            return false;
        const FramePinholeCamera& cam = _reconstruction.camera(imageId);
        const double world[3] = {xyz[0], xyz[1], xyz[2]};
        return cam.isPointInFront(world);
    }

    double Triangulator::computeMaxTriangulationAngle(const std::array<double, 3>& xyz,
                                                      const std::vector<TrackElement>& observations) const
    {
        double maxAngle = 0.0;

        for (size_t i = 0; i < observations.size(); ++i)
        {
            if (!_reconstruction.hasCamera(observations[i].imageId))
            {
                continue;
            }

            const FramePinholeCamera& cameraI = _reconstruction.camera(observations[i].imageId);
            const auto centerI = cameraI.cameraCenter();

            for (size_t j = i + 1; j < observations.size(); ++j)
            {
                if (!_reconstruction.hasCamera(observations[j].imageId))
                {
                    continue;
                }

                const FramePinholeCamera& cameraJ = _reconstruction.camera(observations[j].imageId);
                const auto centerJ = cameraJ.cameraCenter();

                const double rayI[3] = {xyz[0] - centerI[0], xyz[1] - centerI[1], xyz[2] - centerI[2]};
                const double rayJ[3] = {xyz[0] - centerJ[0], xyz[1] - centerJ[1], xyz[2] - centerJ[2]};
                const double lenI = std::sqrt(rayI[0] * rayI[0] + rayI[1] * rayI[1] + rayI[2] * rayI[2]);
                const double lenJ = std::sqrt(rayJ[0] * rayJ[0] + rayJ[1] * rayJ[1] + rayJ[2] * rayJ[2]);
                if (lenI <= 1e-9 || lenJ <= 1e-9)
                {
                    continue;
                }

                double cosAngle = (rayI[0] * rayJ[0] + rayI[1] * rayJ[1] + rayI[2] * rayJ[2]) / (lenI * lenJ);
                cosAngle = std::max(-1.0, std::min(1.0, cosAngle));
                const double angle = std::acos(cosAngle) * 180.0 / M_PI;
                maxAngle = std::max(maxAngle, angle);
            }
        }

        return maxAngle;
    }

    // ---- 多视图最近射线三角化 ----

    bool Triangulator::triangulateMultiView(const std::vector<TrackElement>& observations,
                                            std::array<double, 3>& outXyz) const
    {
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        double d = 0.0;
        double e = 0.0;
        double f = 0.0;
        double g = 0.0;
        double h = 0.0;
        double i = 0.0;
        double rhsX = 0.0;
        double rhsY = 0.0;
        double rhsZ = 0.0;
        std::size_t rayCount = 0;

        for (const TrackElement& element : observations)
        {
            if (!_reconstruction.isRegistered(element.imageId) || !_reconstruction.hasCamera(element.imageId) ||
                !_reconstruction.hasImage(element.imageId))
            {
                continue;
            }
            const ImageData& image = _reconstruction.image(element.imageId);
            if (element.featureIdx >= image.keypoints.size())
            {
                continue;
            }

            const FeatureKeypoint& keypoint = image.keypoints[element.featureIdx];
            CameraImagingRay ray;
            if (!_reconstruction.camera(element.imageId)
                     .rayForPixel(CameraImageCoordinate{keypoint.x, keypoint.y}, &ray))
            {
                continue;
            }
            const double squaredLength = ray.direction[0] * ray.direction[0] + ray.direction[1] * ray.direction[1] +
                                         ray.direction[2] * ray.direction[2];
            const double length = std::sqrt(squaredLength);
            if (!std::isfinite(length) || length < std::numeric_limits<double>::epsilon())
            {
                continue;
            }
            const double x = ray.direction[0] / length;
            const double y = ray.direction[1] / length;
            const double z = ray.direction[2] / length;
            const double xx = x * x;
            const double xy = x * y;
            const double xz = x * z;
            const double yy = y * y;
            const double yz = y * z;
            const double zz = z * z;
            const double oneMinusXx = 1.0 - xx;
            const double oneMinusYy = 1.0 - yy;
            const double oneMinusZz = 1.0 - zz;

            a += oneMinusXx;
            b -= xy;
            c -= xz;
            d -= xy;
            e += oneMinusYy;
            f -= yz;
            g -= xz;
            h -= yz;
            i += oneMinusZz;
            rhsX += (oneMinusXx * ray.originMeters[0] - xy * ray.originMeters[1]) - xz * ray.originMeters[2];
            rhsY += (-xy * ray.originMeters[0] + oneMinusYy * ray.originMeters[1]) - yz * ray.originMeters[2];
            rhsZ += (-xz * ray.originMeters[0] - yz * ray.originMeters[1]) + oneMinusZz * ray.originMeters[2];
            ++rayCount;
        }

        if (rayCount < 2)
        {
            return false;
        }

        const double cofactor00 = e * i - f * h;
        const double cofactor10 = d * i - f * g;
        const double cofactor20 = d * h - e * g;
        const double determinant = (a * cofactor00 - b * cofactor10) + c * cofactor20;
        if (!std::isfinite(determinant) || determinant == 0.0)
        {
            return false;
        }
        const double inverseDeterminant = 1.0 / determinant;
        outXyz = {{
            (cofactor00 * inverseDeterminant) * rhsX + ((c * h - b * i) * inverseDeterminant) * rhsY +
                ((b * f - c * e) * inverseDeterminant) * rhsZ,
            ((f * g - d * i) * inverseDeterminant) * rhsX + ((a * i - c * g) * inverseDeterminant) * rhsY +
                ((c * d - a * f) * inverseDeterminant) * rhsZ,
            (cofactor20 * inverseDeterminant) * rhsX + ((b * g - a * h) * inverseDeterminant) * rhsY +
                ((a * e - b * d) * inverseDeterminant) * rhsZ,
        }};

        return std::isfinite(outXyz[0]) && std::isfinite(outXyz[1]) && std::isfinite(outXyz[2]);
    }

    // ---- 重三角化所有点 ----

    int Triangulator::retriangulatePoints(double maxReprojError, bool normalizeByFeatureScale)
    {
        const auto started = std::chrono::steady_clock::now();
        auto allIds = _reconstruction.allPoint3DIds();
        struct RetriangulationUpdate
        {
            bool apply = false;
            std::array<double, 3> xyz{};
            double error = 0.0;
        };
        std::vector<RetriangulationUpdate> updates(allIds.size());

        common::concurrency::parallelForIndices(
            allIds.size(),
            static_cast<std::size_t>(_threadCount),
            [&](std::size_t pointIndex)
            {
                const Point3DId pid = allIds[pointIndex];
                if (!_reconstruction.hasPoint3D(pid))
                    return;
                const auto& pt = _reconstruction.point3D(pid);

                if (pt.track.length() < 2)
                    return;

                // 计算原始平均重投影误差
                double oldAvgErr = 0.0;
                int oldValidObs = 0;
                for (const auto& elem : pt.track.elements)
                {
                    double err = normalizedReprojError(computeReprojError(pt.xyz, elem.imageId, elem.featureIdx),
                                                       elem.imageId,
                                                       elem.featureIdx,
                                                       normalizeByFeatureScale);
                    if (err < 1e8)
                    {
                        oldAvgErr += err;
                        ++oldValidObs;
                    }
                }
                if (oldValidObs > 0)
                    oldAvgErr /= oldValidObs;
                else
                    oldAvgErr = 1e9; // 无有效投影说明旧点极差（如在相机后方）

                // 多视图重三角化
                std::array<double, 3> newXyz;
                if (!triangulateMultiView(pt.track.elements, newXyz))
                    return;

                // 检查新坐标在所有观测相机中深度为正
                bool allPositive = true;
                for (const auto& elem : pt.track.elements)
                {
                    if (!hasPositiveDepth(newXyz, elem.imageId))
                    {
                        allPositive = false;
                        break;
                    }
                }
                if (!allPositive)
                    return;

                // 计算新的平均重投影误差
                double newAvgErr = 0.0;
                int newValidObs = 0;
                for (const auto& elem : pt.track.elements)
                {
                    double err = normalizedReprojError(computeReprojError(newXyz, elem.imageId, elem.featureIdx),
                                                       elem.imageId,
                                                       elem.featureIdx,
                                                       normalizeByFeatureScale);
                    if (err < 1e8)
                    {
                        newAvgErr += err;
                        ++newValidObs;
                    }
                }
                if (newValidObs > 0)
                    newAvgErr /= newValidObs;

                // 如果新误差超过阈值，或者比旧误差更差且旧误差本身已经不错，跳过
                if (newAvgErr > maxReprojError && newAvgErr >= oldAvgErr)
                    return;

                // 只在新结果更好时更新
                if (newAvgErr < oldAvgErr || oldAvgErr > maxReprojError)
                {
                    updates[pointIndex] = {true, newXyz, newAvgErr};
                }
            });

        int improved = 0;
        for (std::size_t pointIndex = 0; pointIndex < allIds.size(); ++pointIndex)
        {
            if (!updates[pointIndex].apply || !_reconstruction.hasPoint3D(allIds[pointIndex]))
            {
                continue;
            }
            auto& point = _reconstruction.point3D(allIds[pointIndex]);
            point.xyz = updates[pointIndex].xyz;
            point.error = updates[pointIndex].error;
            ++improved;
        }

        const double elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        Logger::instance()->infof(
            "[Triangulator] retriangulatePoints: improved %d / %zu points, threads=%d, seconds=%.3f",
            improved,
            allIds.size(),
            _threadCount,
            elapsedSeconds);
        return improved;
    }

    // ---- 重算所有点的重投影误差 ----

    void Triangulator::recomputeReprojErrors()
    {
        auto allIds = _reconstruction.allPoint3DIds();
        std::vector<double> errors(allIds.size(), 0.0);
        std::vector<char> hasPoint(allIds.size(), 0);
        common::concurrency::parallelForIndices(allIds.size(),
                                                static_cast<std::size_t>(_threadCount),
                                                [&](std::size_t pointIndex)
                                                {
                                                    const Point3DId pid = allIds[pointIndex];
                                                    if (!_reconstruction.hasPoint3D(pid))
                                                        return;
                                                    const auto& pt = _reconstruction.point3D(pid);

                                                    double sumErr = 0.0;
                                                    int cnt = 0;
                                                    for (const auto& elem : pt.track.elements)
                                                    {
                                                        double err =
                                                            computeReprojError(pt.xyz, elem.imageId, elem.featureIdx);
                                                        if (err < 1e8)
                                                        {
                                                            sumErr += err;
                                                            ++cnt;
                                                        }
                                                    }
                                                    errors[pointIndex] = (cnt > 0) ? (sumErr / cnt) : 0.0;
                                                    hasPoint[pointIndex] = 1;
                                                });

        for (std::size_t pointIndex = 0; pointIndex < allIds.size(); ++pointIndex)
        {
            if (hasPoint[pointIndex] && _reconstruction.hasPoint3D(allIds[pointIndex]))
            {
                _reconstruction.point3D(allIds[pointIndex]).error = errors[pointIndex];
            }
        }
    }

} // namespace xjw
