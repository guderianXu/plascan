#include "IndependentBlockReconstructor.h"

#include "ImageRegistrationEngine.h"
#include "IncrementalSfmDetail.h"
#include "ReferenceModelQuality.h"
#include "graph/ReferenceCameraGroupPartitioner.h"
#include "triangulation/Triangulator.h"

#include "log/Logger.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xjw
{
    namespace
    {

        using incremental_sfm_detail::SimilarityTransform3d;

        constexpr std::size_t kAmbiguousTrack = std::numeric_limits<std::size_t>::max();

        std::uint64_t observationKey(ImageId imageId, FeatureIdx featureIdx)
        {
            return (static_cast<std::uint64_t>(imageId) << 32) | static_cast<std::uint64_t>(featureIdx);
        }

        std::unordered_map<std::uint64_t, std::size_t> buildTrackIdentity(const std::vector<Track>& tracks)
        {
            std::unordered_map<std::uint64_t, std::size_t> identity;
            for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
            {
                for (const TrackElement& element : tracks[trackIndex].elements)
                {
                    const std::uint64_t key = observationKey(element.imageId, element.featureIdx);
                    const auto [it, inserted] = identity.emplace(key, trackIndex);
                    if (!inserted && it->second != trackIndex)
                    {
                        it->second = kAmbiguousTrack;
                    }
                }
            }
            return identity;
        }

        std::optional<std::size_t> persistentTrackId(const Track& track,
                                                     const std::unordered_map<std::uint64_t, std::size_t>& identity)
        {
            std::optional<std::size_t> result;
            for (const TrackElement& element : track.elements)
            {
                const auto it = identity.find(observationKey(element.imageId, element.featureIdx));
                if (it == identity.end() || it->second == kAmbiguousTrack)
                {
                    return std::nullopt;
                }
                if (result && *result != it->second)
                {
                    return std::nullopt;
                }
                result = it->second;
            }
            return result;
        }

        std::map<std::size_t, std::array<double, 3>>
        pointsByPersistentTrack(const SfmReconstruction& reconstruction,
                                const std::unordered_map<std::uint64_t, std::size_t>& identity)
        {
            std::map<std::size_t, std::array<double, 3>> result;
            for (Point3DId pointId : reconstruction.allPoint3DIds())
            {
                const ScenePoint3D& point = reconstruction.point3D(pointId);
                const std::optional<std::size_t> trackId = persistentTrackId(point.track, identity);
                if (trackId)
                {
                    result.emplace(*trackId, point.xyz);
                }
            }
            return result;
        }

        SimilarityTransform3d estimatePointSimilarity(const std::map<std::size_t, std::array<double, 3>>& source,
                                                      const std::map<std::size_t, std::array<double, 3>>& target)
        {
            std::vector<std::array<double, 3>> sourceValues;
            std::vector<std::array<double, 3>> targetValues;
            for (const auto& [trackId, targetPoint] : target)
            {
                const auto sourceIt = source.find(trackId);
                if (sourceIt != source.end())
                {
                    sourceValues.push_back(sourceIt->second);
                    targetValues.push_back(targetPoint);
                }
            }
            Logger::instance()->infof("[SFM] Block Sim3 persistent correspondences=%zu source=%zu anchor=%zu",
                                      sourceValues.size(),
                                      source.size(),
                                      target.size());
            if (sourceValues.size() < 10)
            {
                return {};
            }
            SimilarityTransform3d transform =
                incremental_sfm_detail::estimateRobustCameraCenterSimilarity(sourceValues, targetValues);
            return transform.valid && transform.inlierCount >= 10 ? transform : SimilarityTransform3d{};
        }

        SimilarityTransform3d estimateCameraSimilarity(const SfmReconstruction& source,
                                                       const SfmReconstruction& target,
                                                       const std::vector<ImageId>& imageIds)
        {
            std::vector<std::array<double, 3>> sourceCenters;
            std::vector<std::array<double, 3>> targetCenters;
            for (ImageId imageId : imageIds)
            {
                if (source.isRegistered(imageId) && target.isRegistered(imageId))
                {
                    sourceCenters.push_back(source.camera(imageId).cameraCenter());
                    targetCenters.push_back(target.camera(imageId).cameraCenter());
                }
            }
            return incremental_sfm_detail::estimateRobustCameraCenterSimilarity(sourceCenters, targetCenters);
        }

        FramePinholeCamera transformCamera(const FramePinholeCamera& camera, const SimilarityTransform3d& transform)
        {
            FramePinholeCamera result = camera;
            result.setPose(incremental_sfm_detail::multiplyRotation(transform.rotation, camera.cameraToWorldRotation()),
                           incremental_sfm_detail::transformPoint(transform, camera.cameraCenter()));
            return result;
        }

        SimilarityTransform3d composeSimilarity(const SimilarityTransform3d& first, const SimilarityTransform3d& second)
        {
            // first: source -> intermediate, second: intermediate -> anchor.
            SimilarityTransform3d result;
            if (!first.valid || !second.valid)
            {
                return result;
            }
            result.valid = true;
            result.scale = first.scale * second.scale;
            result.rotation = incremental_sfm_detail::multiplyRotation(second.rotation, first.rotation);
            std::array<double, 3> rotatedTranslation{};
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    rotatedTranslation[row] += second.rotation[row * 3 + column] * first.translation[column];
                }
            }
            for (std::size_t axis = 0; axis < result.translation.size(); ++axis)
            {
                result.translation[axis] = second.scale * rotatedTranslation[axis] + second.translation[axis];
            }
            result.inlierCount = first.inlierCount;
            result.rmse = first.rmse;
            return result;
        }

        std::vector<Track> filterTracks(const std::vector<Track>& tracks, const std::unordered_set<ImageId>& allowed)
        {
            std::vector<Track> result;
            result.reserve(tracks.size());
            for (const Track& track : tracks)
            {
                Track filtered = track;
                filtered.elements.erase(std::remove_if(filtered.elements.begin(),
                                                       filtered.elements.end(),
                                                       [&](const TrackElement& element)
                                                       { return !allowed.contains(element.imageId); }),
                                        filtered.elements.end());
                if (filtered.elements.size() >= 2)
                {
                    result.push_back(std::move(filtered));
                }
            }
            return result;
        }

        double median(std::vector<double> values)
        {
            if (values.empty())
            {
                return 0.0;
            }
            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        }

        std::string independentBlockSensorKey(std::size_t blockIndex, const std::string& inputSensorKey)
        {
            return "__independent_block_" + std::to_string(blockIndex) + "/" +
                   (inputSensorKey.empty() ? std::string("__default__") : inputSensorKey);
        }

        void collapseSensorCalibrations(SfmReconstruction& reconstruction)
        {
            std::map<std::string, std::vector<ImageId>> groups;
            for (ImageId imageId : reconstruction.registeredImageIds())
            {
                const std::string& sensor_key = reconstruction.image(imageId).sensorKey;
                groups[sensor_key.empty() ? std::string("__default__") : sensor_key].push_back(imageId);
            }
            for (const auto& [sensor_key, image_ids] : groups)
            {
                (void)sensor_key;
                std::array<std::vector<double>, 9> parameters;
                for (ImageId imageId : image_ids)
                {
                    const FramePinholeCamera& camera = reconstruction.camera(imageId);
                    const auto intrinsics = camera.intrinsics();
                    const auto distortion = camera.distortion();
                    parameters[0].push_back(intrinsics.focalX);
                    parameters[1].push_back(intrinsics.focalY);
                    parameters[2].push_back(intrinsics.principalX);
                    parameters[3].push_back(intrinsics.principalY);
                    parameters[4].push_back(distortion.radialK1);
                    parameters[5].push_back(distortion.radialK2);
                    parameters[6].push_back(distortion.radialK3);
                    parameters[7].push_back(distortion.tangentialP1);
                    parameters[8].push_back(distortion.tangentialP2);
                }
                for (ImageId imageId : image_ids)
                {
                    FramePinholeCamera& camera = reconstruction.camera(imageId);
                    camera.setIntrinsics(
                        median(parameters[0]), median(parameters[1]), median(parameters[2]), median(parameters[3]));
                    camera.setDistortion(median(parameters[4]),
                                         median(parameters[5]),
                                         median(parameters[6]),
                                         median(parameters[7]),
                                         median(parameters[8]));
                }
            }
        }

    } // namespace

    IndependentBlockReconstructor::IndependentBlockReconstructor(IncrementalSfm& owner) : _owner(owner)
    {
    }

    std::optional<IncrementalSfmResult> IndependentBlockReconstructor::runIfNeeded(SfmProgressCallback progressCb)
    {
        if (!_owner._sfmOptions.enableIndependentCameraBlocks || _owner._inputMultiViewTracks.empty())
        {
            return std::nullopt;
        }
        const std::vector<ImageId> allImageIds = _owner._reconstruction->allImageIds();
        if (allImageIds.size() < static_cast<std::size_t>(std::max(2, _owner._sfmOptions.independentBlockMinImages)))
        {
            return std::nullopt;
        }
        ReferenceCameraGroupPartitionOptions partitionOptions;
        partitionOptions.maximumGroupSize =
            static_cast<std::size_t>(std::max(1, _owner._sfmOptions.independentBlockGroupSize));
        partitionOptions.minimumLargeSide =
            static_cast<std::size_t>(std::max(1, _owner._sfmOptions.independentBlockMinimumLargeSide));
        const std::vector<ReferenceCameraGroup> groups =
            partitionReferenceCameraGroups(allImageIds, _owner._inputMultiViewTracks, partitionOptions);
        if (groups.size() <= 1)
        {
            return std::nullopt;
        }

        Logger::instance()->infof(
            "[SFM] Reference independent reconstruction: images=%zu blocks=%zu", allImageIds.size(), groups.size());
        const auto identity = buildTrackIdentity(_owner._inputMultiViewTracks);
        const SfmReconstruction inputReconstruction = *_owner._reconstruction;

        auto runSubset = [&](const std::vector<ImageId>& subsetIds, const std::string& label) -> IncrementalSfmResult
        {
            IncrementalSfmOptions childOptions = _owner._sfmOptions;
            childOptions.enableIndependentCameraBlocks = false;
            childOptions.enableHierarchicalBA = false;
            childOptions.maxRegisteredImages = 0;
            childOptions.useKnownCameraPoses = false;
            IncrementalSfm child(childOptions);
            const std::unordered_set<ImageId> allowed(subsetIds.begin(), subsetIds.end());
            for (ImageId imageId : subsetIds)
            {
                FramePinholeCamera camera;
                if (!_owner.getCamera(imageId, camera))
                {
                    IncrementalSfmResult failed;
                    failed.summary = label + ": cannot load camera " + std::to_string(imageId);
                    return failed;
                }
                const ImageData& image = inputReconstruction.image(imageId);
                child.addImageWithCamera(imageId, image.imagePath, camera, image.keypoints, image.sensorKey);
            }
            for (const ImagePair& pair : _owner._correspondenceGraph.imagePairs())
            {
                if (allowed.contains(pair.first) && allowed.contains(pair.second))
                {
                    child.addMatches(
                        pair.first, pair.second, _owner._correspondenceGraph.matchesBetween(pair.first, pair.second));
                }
            }
            child.setInputMultiViewTracks(filterTracks(_owner._inputMultiViewTracks, allowed));
            return child.run(
                [&](int registered, int total, const std::string& message)
                {
                    if (!progressCb)
                    {
                        return true;
                    }
                    const bool keepGoing = progressCb(registered, total, label + ": " + message);
                    if (!keepGoing)
                    {
                        _owner._isAborted = true;
                    }
                    return keepGoing;
                });
        };

        auto runProducer = [&](const IncrementalSfmResult& coreResult, const std::string& label) -> IncrementalSfmResult
        {
            IncrementalSfmOptions childOptions = _owner._sfmOptions;
            childOptions.enableIndependentCameraBlocks = false;
            childOptions.enableHierarchicalBA = false;
            childOptions.maxRegisteredImages = 0;
            childOptions.useKnownCameraPoses = false;
            IncrementalSfm child(childOptions);
            for (ImageId imageId : allImageIds)
            {
                FramePinholeCamera camera;
                if (!_owner.getCamera(imageId, camera))
                {
                    IncrementalSfmResult failed;
                    failed.summary = label + ": cannot load camera " + std::to_string(imageId);
                    return failed;
                }
                const ImageData& image = inputReconstruction.image(imageId);
                child.addImageWithCamera(imageId, image.imagePath, camera, image.keypoints, image.sensorKey);
            }
            for (const ImagePair& pair : _owner._correspondenceGraph.imagePairs())
            {
                child.addMatches(
                    pair.first, pair.second, _owner._correspondenceGraph.matchesBetween(pair.first, pair.second));
            }
            // producer 不调用 child.run()，必须在所有匹配写入后显式建立逐特征邻接索引。
            // 否则全轨三角化看不到任何直接边，新增相机也无法生成共同持久轨迹点。
            child._correspondenceGraph.buildCorrespondences();
            child.setInputMultiViewTracks(_owner._inputMultiViewTracks);
            // producer 不调用 child.run()，因此需直接激活已经由父流程构建并校验的轨迹；
            // setInputMultiViewTracks() 只填充 run() 的待处理输入槽。
            child._inputMultiViewTracks = _owner._inputMultiViewTracks;
            child.rebuildInputTrackObservationIndex();
            child._reconstruction = std::make_shared<SfmReconstruction>(*coreResult.reconstruction);
            for (ImageId imageId : allImageIds)
            {
                if (child._reconstruction->hasImage(imageId))
                {
                    continue;
                }
                ImageData image = inputReconstruction.image(imageId);
                image.registered = false;
                std::fill(image.point3DIds.begin(), image.point3DIds.end(), kInvalidPoint3DId);
                child._reconstruction->addImage(image);
            }

            // 参考 producer 只负责建立各核心块之间可比较的坐标系：继承核心块标定，
            // 最多执行两轮“冻结模型 PnP -> 全量重三角化”。这里禁止 BA，
            // 否则每个 producer 会独立漂移，随后还需要额外的 core->producer Sim3。
            std::map<std::string, FramePinholeCamera> calibrationBySensor;
            std::optional<FramePinholeCamera> defaultCalibration;
            for (ImageId imageId : child._reconstruction->registeredImageIds())
            {
                const ImageData& image = child._reconstruction->image(imageId);
                const FramePinholeCamera& camera = child._reconstruction->camera(imageId);
                calibrationBySensor.try_emplace(image.sensorKey, camera);
                if (!defaultCalibration)
                {
                    defaultCalibration = camera;
                }
            }
            for (ImageId imageId : allImageIds)
            {
                if (child._reconstruction->isRegistered(imageId))
                {
                    continue;
                }
                auto preloaded = child._preloadedCameras.find(imageId);
                if (preloaded == child._preloadedCameras.end())
                {
                    continue;
                }
                FramePinholeCamera camera = preloaded->second;
                const std::string& sensorKey = child._reconstruction->image(imageId).sensorKey;
                const auto calibration = calibrationBySensor.find(sensorKey);
                const FramePinholeCamera* source = calibration != calibrationBySensor.end()
                                                       ? &calibration->second
                                                       : (defaultCalibration ? &*defaultCalibration : nullptr);
                if (source)
                {
                    const auto intrinsics = source->intrinsics();
                    const auto distortion = source->distortion();
                    camera.setIntrinsics(
                        intrinsics.focalX, intrinsics.focalY, intrinsics.principalX, intrinsics.principalY);
                    camera.setDistortion(distortion.radialK1,
                                         distortion.radialK2,
                                         distortion.radialK3,
                                         distortion.tangentialP1,
                                         distortion.tangentialP2);
                    preloaded->second = camera;
                }
            }
            // 参考 producer 在第一次刷新前清空局部核心点，并只用当前已注册影像上的
            // 持久轨迹观测重建结构。不能保留 core 点：不同 producer 的 core 覆盖互不
            // 相同，保留它们会让后续新增相机虽能完成 PnP，却产生不了共同轨迹点。
            bool firstRefresh = true;
            auto refreshProducerStructure = [&]()
            {
                if (firstRefresh)
                {
                    child._reconstruction->clearPoints3D();
                    firstRefresh = false;
                }

                Triangulator triangulator(
                    *child._reconstruction, child._correspondenceGraph, child._sfmOptions.baOptions.numThreads);
                TriangulatorOptions triangulatorOptions = child._sfmOptions.triangulatorOptions;
                triangulatorOptions.bindCompleteInputTrack = true;
                triangulatorOptions.deferPureTwoViewTracks = false;
                const TriangulationStats triangulation =
                    triangulator.triangulateTracks(child._inputMultiViewTracks, triangulatorOptions);
                child.invalidateVisibilityCache();
                child.rebuildVisibilityCache();
                Logger::instance()->infof("[SFM] %s refresh: available=%zu added=%d restored=%d unusable=%d "
                                          "noCandidate=%d reprojRejected=%d depthRejected=%d valid=%zu",
                                          label.c_str(),
                                          child._inputMultiViewTracks.size(),
                                          triangulation.numCreated,
                                          triangulation.numRestored,
                                          triangulation.unusableTracks,
                                          triangulation.noCandidateTracks,
                                          triangulation.reprojObservationRejected,
                                          triangulation.depthObservationRejected,
                                          child._reconstruction->numPoints3D());
            };
            refreshProducerStructure();
            for (int round = 0; round < 2 && !child._isAborted; ++round)
            {
                const int remaining = static_cast<int>(allImageIds.size()) -
                                      static_cast<int>(child._reconstruction->numRegisteredImages());
                if (remaining <= 0)
                {
                    break;
                }
                const std::vector<ImageId> registered = child.registerImageBatch(remaining);
                if (registered.empty())
                {
                    break;
                }
                refreshProducerStructure();
                if (progressCb && !progressCb(static_cast<int>(child._reconstruction->numRegisteredImages()),
                                              static_cast<int>(allImageIds.size()),
                                              label + ": producer round " + std::to_string(round + 1)))
                {
                    child._isAborted = true;
                    _owner._isAborted = true;
                }
            }

            IncrementalSfmResult result;
            result.success = !child._isAborted && child._reconstruction->numRegisteredImages() >= 2;
            result.numRegisteredImages = static_cast<int>(child._reconstruction->numRegisteredImages());
            result.numPoints3D = static_cast<int>(child._reconstruction->numPoints3D());
            result.meanReprojError = child._reconstruction->meanReprojError();
            result.reconstruction = child._reconstruction;
            result.summary = child._reconstruction->summary();
            return result;
        };

        struct Block
        {
            ReferenceCameraGroup core;
            IncrementalSfmResult result;
            std::map<std::size_t, std::array<double, 3>> points;
            IncrementalSfmResult producer;
            std::map<std::size_t, std::array<double, 3>> producerPoints;
        };
        std::vector<Block> blocks;
        blocks.reserve(groups.size());
        for (std::size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex)
        {
            IncrementalSfmResult blockResult =
                runSubset(groups[groupIndex], "Independent block " + std::to_string(groupIndex + 1));
            if (_owner._isAborted)
            {
                return blockResult;
            }
            if (!blockResult.success || !blockResult.reconstruction)
            {
                Logger::instance()->warnf("[SFM] Independent block %zu failed; falling back to one model: %s",
                                          groupIndex + 1,
                                          blockResult.summary.c_str());
                return std::nullopt;
            }
            auto persistentPoints = pointsByPersistentTrack(*blockResult.reconstruction, identity);
            IncrementalSfmResult producer =
                runProducer(blockResult, "Similarity producer " + std::to_string(groupIndex + 1));
            if (_owner._isAborted)
            {
                return producer;
            }
            if (!producer.success || !producer.reconstruction)
            {
                Logger::instance()->warnf("[SFM] Similarity producer %zu failed; falling back to one model: %s",
                                          groupIndex + 1,
                                          producer.summary.c_str());
                return std::nullopt;
            }
            auto producerPoints = pointsByPersistentTrack(*producer.reconstruction, identity);
            blocks.push_back({groups[groupIndex],
                              std::move(blockResult),
                              std::move(persistentPoints),
                              std::move(producer),
                              std::move(producerPoints)});
        }

        std::map<ImageId, FramePinholeCamera> mergedCameras;
        std::map<ImageId, std::string> mergedSensorKeys;
        std::map<std::size_t, std::array<double, 3>> mergedPoints = blocks.front().points;
        for (ImageId imageId : blocks.front().core)
        {
            if (blocks.front().result.reconstruction->isRegistered(imageId))
            {
                mergedCameras.emplace(imageId, blocks.front().result.reconstruction->camera(imageId));
                const std::string& sensorKey = inputReconstruction.image(imageId).sensorKey;
                mergedSensorKeys.emplace(imageId, independentBlockSensorKey(0, sensorKey));
            }
        }

        // 优先保持参考实现的 producer->anchor 直连；若空间分块形成链而两端没有
        // 共同轨迹，则允许通过已经对齐的相邻 producer 传播到 anchor。每个块仍只
        // 接受一条鲁棒 Sim3 边，避免重复估计 core->producer 等漂移变换。
        std::vector<std::optional<SimilarityTransform3d>> transformToAnchor(blocks.size());
        SimilarityTransform3d anchorIdentity;
        anchorIdentity.valid = true;
        transformToAnchor.front() = anchorIdentity;
        int totalMergeInliers = 0;
        std::size_t remainingBlocks = blocks.size() - 1;
        while (remainingBlocks > 0)
        {
            bool madeProgress = false;
            for (std::size_t blockIndex = 1; blockIndex < blocks.size(); ++blockIndex)
            {
                if (transformToAnchor[blockIndex])
                {
                    continue;
                }

                SimilarityTransform3d bestEdge;
                std::size_t bestTarget = 0;
                for (std::size_t targetIndex = 0; targetIndex < blocks.size(); ++targetIndex)
                {
                    if (!transformToAnchor[targetIndex])
                    {
                        continue;
                    }
                    SimilarityTransform3d edge =
                        estimatePointSimilarity(blocks[blockIndex].producerPoints, blocks[targetIndex].producerPoints);
                    if (!edge.valid)
                    {
                        edge = estimateCameraSimilarity(*blocks[blockIndex].producer.reconstruction,
                                                        *blocks[targetIndex].producer.reconstruction,
                                                        allImageIds);
                    }
                    if (edge.valid && (!bestEdge.valid || edge.inlierCount > bestEdge.inlierCount ||
                                       (edge.inlierCount == bestEdge.inlierCount && edge.rmse < bestEdge.rmse)))
                    {
                        bestEdge = edge;
                        bestTarget = targetIndex;
                    }
                }
                if (!bestEdge.valid)
                {
                    continue;
                }

                transformToAnchor[blockIndex] = composeSimilarity(bestEdge, *transformToAnchor[bestTarget]);
                totalMergeInliers += bestEdge.inlierCount;
                --remainingBlocks;
                madeProgress = true;
                Logger::instance()->infof(
                    "[SFM] Independent block %zu Sim3 edge -> block %zu: inliers=%d scale=%.9f rms=%.6f",
                    blockIndex + 1,
                    bestTarget + 1,
                    bestEdge.inlierCount,
                    bestEdge.scale,
                    bestEdge.rmse);
            }
            if (!madeProgress)
            {
                Logger::instance()->warn(
                    "[SFM] Independent producer overlap graph is disconnected; falling back to one model");
                return std::nullopt;
            }
        }

        for (std::size_t blockIndex = 1; blockIndex < blocks.size(); ++blockIndex)
        {
            Block& block = blocks[blockIndex];
            const SimilarityTransform3d& transform = *transformToAnchor[blockIndex];
            for (ImageId imageId : block.core)
            {
                if (block.result.reconstruction->isRegistered(imageId))
                {
                    mergedCameras.emplace(imageId,
                                          transformCamera(block.result.reconstruction->camera(imageId), transform));
                    const std::string& sensorKey = inputReconstruction.image(imageId).sensorKey;
                    mergedSensorKeys.emplace(imageId, independentBlockSensorKey(blockIndex, sensorKey));
                }
            }
            for (const auto& [trackId, point] : block.points)
            {
                mergedPoints.emplace(trackId, incremental_sfm_detail::transformPoint(transform, point));
            }
            Logger::instance()->infof(
                "[SFM] Independent block %zu merged to anchor: scale=%.9f", blockIndex + 1, transform.scale);
        }

        auto merged = std::make_shared<SfmReconstruction>();
        for (ImageId imageId : allImageIds)
        {
            ImageData image = inputReconstruction.image(imageId);
            image.registered = false;
            std::fill(image.point3DIds.begin(), image.point3DIds.end(), kInvalidPoint3DId);
            const auto sensorKey = mergedSensorKeys.find(imageId);
            if (sensorKey != mergedSensorKeys.end())
            {
                image.sensorKey = sensorKey->second;
            }
            merged->addImage(image);
        }
        for (const auto& [imageId, camera] : mergedCameras)
        {
            merged->registerImage(imageId, camera);
        }
        collapseSensorCalibrations(*merged);
        Triangulator triangulator(*merged, _owner._correspondenceGraph, _owner._sfmOptions.baOptions.numThreads);
        TriangulatorOptions triangulatorOptions = _owner._sfmOptions.triangulatorOptions;
        triangulatorOptions.bindCompleteInputTrack = true;
        triangulatorOptions.deferPureTwoViewTracks = false;
        const TriangulationStats triangulation =
            triangulator.triangulateTracks(_owner._inputMultiViewTracks, triangulatorOptions);
        if (triangulation.numCreated == 0)
        {
            Logger::instance()->warn(
                "[SFM] Independent blocks created no global points; falling back to one incremental model");
            return std::nullopt;
        }
        const ReferenceStructureFilterResult filterResult =
            filterReferenceStructurePoints(*merged, 0.0, _owner._sfmOptions.baOptions.numThreads);
        Logger::instance()->infof("[SFM] Merged reference filters far=%d inaccurate=%d weak=%d",
                                  filterResult.farPoints,
                                  filterResult.inaccuratePoints,
                                  filterResult.weakPoints);

        _owner._reconstruction = std::move(merged);
        _owner.invalidateVisibilityCache();
        IncrementalSfmResult result = _owner.runRegistrationFromCurrentInitialization(
            static_cast<int>(allImageIds.size()), std::move(progressCb), 0, true, false, true, false);
        result.independentCameraBlocks = static_cast<int>(groups.size());
        result.independentBlockMergeInliers = totalMergeInliers;
        return result;
    }

} // namespace xjw
