#include "Triangulator.h"
#include "geometry/OpenCvCameraAdapter.h"
#include "concurrency/SafeWorkerGroup.h"

#include "log/Logger.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
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
                    double reprErr = computeReprojError(pt.xyz, imageId, fi);
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
        TriangulationStats stats;
        std::vector<Point3DId> createdTwoViewPointIds;

        for (const Track& track : tracks)
        {
            ++stats.inputTracks;
            if (track.length() >= 3)
            {
                ++stats.inputLongTracks;
            }

            if (track.length() < 2)
            {
                continue;
            }

            bool usable = true;
            for (const TrackElement& element : track.elements)
            {
                if (!_reconstruction.isRegistered(element.imageId) || !_reconstruction.hasCamera(element.imageId) ||
                    !_reconstruction.hasImage(element.imageId))
                {
                    usable = false;
                    break;
                }

                const ImageData& image = _reconstruction.image(element.imageId);
                if (element.featureIdx >= image.keypoints.size() || element.featureIdx >= image.point3DIds.size() ||
                    image.point3DIds[element.featureIdx] != kInvalidPoint3DId)
                {
                    usable = false;
                    break;
                }
            }
            if (!usable)
            {
                ++stats.unusableTracks;
                continue;
            }

            const bool bootstrapPair = _reconstruction.numRegisteredImages() <= 2;
            if (options.deferPureTwoViewTracks && !bootstrapPair && isPureTwoViewComponent(track))
            {
                ++stats.deferredPureTwoViewTracks;
                continue;
            }

            std::vector<TrackElement> remainingElements = track.elements;
            bool createdAnyPointForTrack = false;
            struct Candidate
            {
                std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
                Track inlierTrack;
                double rmsError = std::numeric_limits<double>::infinity();
                double triangulationAngle = 0.0;
                double relativeDepthUncertainty = std::numeric_limits<double>::infinity();
                double nearestRejectedReprojError = std::numeric_limits<double>::infinity();
                bool valid = false;
            };
            const auto hasDirectCorrespondence = [this](const TrackElement& first, const TrackElement& second)
            {
                const auto correspondences = _correspondenceGraph.findCorrespondences(first.imageId, first.featureIdx);
                return std::any_of(correspondences.begin(),
                                   correspondences.end(),
                                   [&second](const CorrespondenceGraph::Correspondence& correspondence)
                                   {
                                       return correspondence.imageId == second.imageId &&
                                              correspondence.featureIdx == second.featureIdx;
                                   });
            };
            const auto estimateTwoViewRelativeDepthUncertainty =
                [this, &options](double triangulationAngle, const std::vector<TrackElement>& observations)
            {
                if (observations.size() != 2 || triangulationAngle <= 0.0 || !std::isfinite(triangulationAngle) ||
                    options.twoViewFragmentPixelSigma <= 0.0)
                {
                    return std::numeric_limits<double>::infinity();
                }

                double effectiveFocal = std::numeric_limits<double>::infinity();
                for (const TrackElement& observation : observations)
                {
                    if (!_reconstruction.hasCamera(observation.imageId))
                    {
                        return std::numeric_limits<double>::infinity();
                    }
                    const FramePinholeCamera& camera = _reconstruction.camera(observation.imageId);
                    const double focal = std::sqrt(std::abs(camera.focalX() * camera.focalY()));
                    if (!std::isfinite(focal) || focal <= 0.0)
                    {
                        return std::numeric_limits<double>::infinity();
                    }
                    effectiveFocal = std::min(effectiveFocal, focal);
                }

                const double angleRadians = triangulationAngle * M_PI / 180.0;
                const double geometricStrength = effectiveFocal * std::sin(angleRadians);
                if (!std::isfinite(geometricStrength) || geometricStrength <= 1e-12)
                {
                    return std::numeric_limits<double>::infinity();
                }
                return std::sqrt(2.0) * options.twoViewFragmentPixelSigma / geometricStrength;
            };
            auto collectCandidateFromPoint = [&](const std::array<double, 3>& xyz, bool countRejects) -> Candidate
            {
                Candidate candidate;
                double squaredErrorSum = 0.0;
                for (const TrackElement& element : remainingElements)
                {
                    if (!hasPositiveDepth(xyz, element.imageId))
                    {
                        if (countRejects)
                        {
                            ++stats.depthObservationRejected;
                        }
                        continue;
                    }

                    const double error = computeReprojError(xyz, element.imageId, element.featureIdx);
                    if (!std::isfinite(error) || error > options.completeMaxReprojError)
                    {
                        if (std::isfinite(error))
                        {
                            candidate.nearestRejectedReprojError =
                                std::min(candidate.nearestRejectedReprojError, error);
                        }
                        if (countRejects)
                        {
                            ++stats.reprojObservationRejected;
                        }
                        continue;
                    }

                    candidate.inlierTrack.elements.push_back(element);
                    squaredErrorSum += error * error;
                }

                if (candidate.inlierTrack.length() < 2)
                {
                    return Candidate{};
                }
                candidate.triangulationAngle = computeMaxTriangulationAngle(xyz, candidate.inlierTrack.elements);
                if (candidate.triangulationAngle < options.minTriAngle)
                {
                    return Candidate{};
                }

                // 多视轨迹由直接匹配边的传递闭包构成。若一个候选最终只剩两个观测，
                // 两者必须确实存在上游几何验证过的匹配边，不能把 A-B-C 的传递关系
                // 当成未经验证的 A-C 匹配。原生双影像轨迹仍会正常通过此检查。
                if (track.source == TrackSource::FeatureMatch && candidate.inlierTrack.length() == 2 &&
                    !hasDirectCorrespondence(candidate.inlierTrack.elements[0], candidate.inlierTrack.elements[1]))
                {
                    return Candidate{};
                }

                // 内点重建不能丢失人工标记的来源，否则后续控制点/检查点角色无法传入 BA。
                candidate.inlierTrack.confidence = track.confidence;
                candidate.inlierTrack.source = track.source;
                candidate.inlierTrack.sourceId = track.sourceId;
                candidate.xyz = xyz;
                candidate.rmsError = std::sqrt(squaredErrorSum / static_cast<double>(candidate.inlierTrack.length()));
                if (track.source == TrackSource::FeatureMatch && track.length() >= 3 &&
                    candidate.inlierTrack.length() == 2)
                {
                    candidate.relativeDepthUncertainty = estimateTwoViewRelativeDepthUncertainty(
                        candidate.triangulationAngle, candidate.inlierTrack.elements);
                    const double fragmentMaxReprojError =
                        options.twoViewFragmentMaxReprojError > 0.0
                            ? std::min(options.maxReprojError, options.twoViewFragmentMaxReprojError)
                            : options.maxReprojError;
                    const bool unstableReprojection =
                        !std::isfinite(candidate.rmsError) || candidate.rmsError > fragmentMaxReprojError;
                    const bool unstableDepth =
                        options.twoViewFragmentMaxRelativeDepthUncertainty > 0.0 &&
                        (!std::isfinite(candidate.relativeDepthUncertainty) ||
                         candidate.relativeDepthUncertainty > options.twoViewFragmentMaxRelativeDepthUncertainty);
                    if (unstableReprojection || unstableDepth)
                    {
                        if (countRejects)
                        {
                            ++stats.unstableTwoViewCandidates;
                        }
                        return Candidate{};
                    }
                }
                candidate.valid = true;
                return candidate;
            };
            auto betterCandidate = [](const Candidate& candidate, const Candidate& best)
            {
                return candidate.valid && (!best.valid || candidate.inlierTrack.length() > best.inlierTrack.length() ||
                                           (candidate.inlierTrack.length() == best.inlierTrack.length() &&
                                            candidate.rmsError < best.rmsError));
            };
            auto refineCandidate = [&](Candidate candidate) -> Candidate
            {
                if (!candidate.valid || candidate.inlierTrack.length() < 3)
                {
                    return candidate;
                }

                std::array<double, 3> refinedXyz;
                if (!triangulateMultiView(candidate.inlierTrack.elements, refinedXyz))
                {
                    return candidate;
                }

                Candidate refined = collectCandidateFromPoint(refinedXyz, false);
                if (betterCandidate(refined, candidate))
                {
                    return refined;
                }
                return candidate;
            };

            std::vector<Candidate> candidates;
            while (remainingElements.size() >= 2)
            {
                Candidate best;
                double nearestRejectedExtraForTrack = std::numeric_limits<double>::infinity();

                // 先用完整组件做一次多视 DLT。几何一致的正常轨迹通常一次即可得到
                // 最长内点集，避免为每个二元/三元组合重复求解。
                std::array<double, 3> allViewXyz;
                if (triangulateMultiView(remainingElements, allViewXyz))
                {
                    best = refineCandidate(collectCandidateFromPoint(allViewXyz, false));
                }

                // 对坏桥或含离群观测的组件仍需双视种子。小轨迹穷举全部像对；
                // 长轨迹只取相邻、跨距和均匀采样的确定性代表像对，限制最坏复杂度。
                constexpr std::size_t kMaximumSeedPairs = 96;
                std::vector<std::pair<std::size_t, std::size_t>> seedPairs;
                seedPairs.reserve(
                    std::min(kMaximumSeedPairs, remainingElements.size() * (remainingElements.size() - 1) / 2));
                const auto addSeedPair = [&seedPairs](std::size_t left, std::size_t right)
                {
                    if (left == right || seedPairs.size() >= kMaximumSeedPairs)
                    {
                        return;
                    }
                    if (left > right)
                    {
                        std::swap(left, right);
                    }
                    const std::pair<std::size_t, std::size_t> pair{left, right};
                    if (std::find(seedPairs.begin(), seedPairs.end(), pair) == seedPairs.end())
                    {
                        seedPairs.push_back(pair);
                    }
                };

                const std::size_t elementCount = remainingElements.size();
                const std::size_t allPairCount = elementCount * (elementCount - 1) / 2;
                const bool allViewCandidateUsesEveryObservation =
                    best.valid && best.inlierTrack.length() == elementCount;
                if (!allViewCandidateUsesEveryObservation && allPairCount <= kMaximumSeedPairs)
                {
                    for (std::size_t left = 0; left < elementCount; ++left)
                    {
                        for (std::size_t right = left + 1; right < elementCount; ++right)
                        {
                            addSeedPair(left, right);
                        }
                    }
                }
                else if (!allViewCandidateUsesEveryObservation)
                {
                    for (std::size_t index = 0; index + 1 < elementCount; ++index)
                    {
                        addSeedPair(index, index + 1);
                    }
                    for (std::size_t index = 0; index < elementCount / 2; ++index)
                    {
                        addSeedPair(index, elementCount - 1 - index);
                    }

                    const std::size_t sampleStride = std::max<std::size_t>(1, allPairCount / kMaximumSeedPairs);
                    std::size_t flatPairIndex = 0;
                    for (std::size_t left = 0; left < elementCount && seedPairs.size() < kMaximumSeedPairs; ++left)
                    {
                        for (std::size_t right = left + 1; right < elementCount && seedPairs.size() < kMaximumSeedPairs;
                             ++right, ++flatPairIndex)
                        {
                            if (flatPairIndex % sampleStride == 0)
                            {
                                addSeedPair(left, right);
                            }
                        }
                    }
                }

                for (const auto& [left, right] : seedPairs)
                {
                    std::array<double, 3> seedXyz;
                    const TrackElement& leftElement = remainingElements[left];
                    const TrackElement& rightElement = remainingElements[right];
                    if (track.source == TrackSource::FeatureMatch &&
                        !hasDirectCorrespondence(leftElement, rightElement))
                    {
                        ++stats.indirectTwoViewCandidates;
                        continue;
                    }
                    ++stats.seedPairTests;
                    if (!triangulatePair(leftElement.imageId,
                                         leftElement.featureIdx,
                                         rightElement.imageId,
                                         rightElement.featureIdx,
                                         options,
                                         seedXyz))
                    {
                        ++stats.seedPairRejected;
                        continue;
                    }

                    Candidate seedCandidate = refineCandidate(collectCandidateFromPoint(seedXyz, true));
                    if (betterCandidate(seedCandidate, best))
                    {
                        best = seedCandidate;
                    }
                    if (track.length() >= 3 && seedCandidate.valid && seedCandidate.inlierTrack.length() == 2 &&
                        std::isfinite(seedCandidate.nearestRejectedReprojError))
                    {
                        nearestRejectedExtraForTrack =
                            std::min(nearestRejectedExtraForTrack, seedCandidate.nearestRejectedReprojError);
                    }
                }

                if (!best.valid)
                {
                    break;
                }
                best.nearestRejectedReprojError =
                    std::min(best.nearestRejectedReprojError, nearestRejectedExtraForTrack);
                candidates.push_back(best);

                remainingElements.erase(std::remove_if(remainingElements.begin(),
                                                       remainingElements.end(),
                                                       [&best](const TrackElement& element)
                                                       {
                                                           return std::any_of(
                                                               best.inlierTrack.elements.begin(),
                                                               best.inlierTrack.elements.end(),
                                                               [&element](const TrackElement& used)
                                                               {
                                                                   return used.imageId == element.imageId &&
                                                                          used.featureIdx == element.featureIdx;
                                                               });
                                                       }),
                                        remainingElements.end());
            }

            const bool hasLongCandidate =
                std::any_of(candidates.begin(),
                            candidates.end(),
                            [](const Candidate& candidate) { return candidate.inlierTrack.length() >= 3; });
            for (const Candidate& candidate : candidates)
            {
                // 一条输入轨迹表达一个物理特征。它已经形成多视共识点时，剩余的
                // 两观测候选通常是离群观测被再次配对产生的飞点。只抑制这种残片；
                // 原生双视轨迹，以及完全无法形成三视共识的稀疏区域仍保留二视结果。
                if (track.length() >= 3 && hasLongCandidate && candidate.inlierTrack.length() == 2)
                {
                    ++stats.suppressedTwoViewFragments;
                    continue;
                }

                const Point3DId pointId = _reconstruction.addPoint3DWithTrack(candidate.xyz, candidate.inlierTrack);
                if (_reconstruction.hasPoint3D(pointId))
                {
                    _reconstruction.point3D(pointId).error = candidate.rmsError;
                }
                ++stats.numCreated;
                stats.numContinued += static_cast<int>(std::max<std::size_t>(0, candidate.inlierTrack.length() - 2));
                createdAnyPointForTrack = true;
                if (candidate.inlierTrack.length() >= 3)
                {
                    ++stats.createdLongTracks;
                    continue;
                }

                ++stats.createdTwoViewTracks;
                createdTwoViewPointIds.push_back(pointId);
                if (track.length() < 3)
                {
                    continue;
                }

                ++stats.longTrackTwoViewOnly;
                if (!std::isfinite(candidate.nearestRejectedReprojError))
                {
                    continue;
                }

                const double rejectedError = candidate.nearestRejectedReprojError;
                ++stats.longTrackRejectedExtraSamples;
                stats.longTrackRejectedExtraErrorSum += rejectedError;
                stats.longTrackRejectedExtraErrorMax = std::max(stats.longTrackRejectedExtraErrorMax, rejectedError);
                if (rejectedError <= 5.0)
                {
                    ++stats.longTrackRejectedExtraLe5;
                }
                else if (rejectedError <= 10.0)
                {
                    ++stats.longTrackRejectedExtraLe10;
                }
                else if (rejectedError <= 25.0)
                {
                    ++stats.longTrackRejectedExtraLe25;
                }
                else
                {
                    ++stats.longTrackRejectedExtraGt25;
                }
            }

            if (!createdAnyPointForTrack)
            {
                ++stats.noCandidateTracks;
            }
        }

        if (options.enableTwoViewLocalDepthConsistency && _reconstruction.numRegisteredImages() > 2 &&
            options.twoViewLocalDepthRadiusPixels > 0.0 && options.twoViewLocalDepthMinReferences > 0 &&
            options.twoViewLocalDepthMaxDeviation > 0.0 && !createdTwoViewPointIds.empty())
        {
            struct DepthReference
            {
                double x = 0.0;
                double y = 0.0;
                double depth = 0.0;
            };
            using CellReferences = std::unordered_map<std::uint64_t, std::vector<DepthReference>>;
            std::unordered_map<ImageId, CellReferences> referencesByImage;
            const double cellSize = options.twoViewLocalDepthRadiusPixels;
            const auto cellKey = [cellSize](double x, double y)
            {
                const auto column = static_cast<std::int32_t>(std::floor(x / cellSize));
                const auto row = static_cast<std::int32_t>(std::floor(y / cellSize));
                return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(column)) << 32U) |
                       static_cast<std::uint32_t>(row);
            };
            const auto median = [](std::vector<double> values)
            {
                if (values.empty())
                {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const std::size_t middle = values.size() / 2;
                std::nth_element(values.begin(), values.begin() + middle, values.end());
                const double upper = values[middle];
                if (values.size() % 2 != 0)
                {
                    return upper;
                }
                std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
                return 0.5 * (values[middle - 1] + upper);
            };

            for (const auto& [pointId, point] : _reconstruction.points3D())
            {
                (void)pointId;
                if (point.track.source != TrackSource::FeatureMatch || point.track.length() < 3)
                {
                    continue;
                }
                for (const TrackElement& element : point.track.elements)
                {
                    if (!_reconstruction.hasCamera(element.imageId) || !_reconstruction.hasImage(element.imageId))
                    {
                        continue;
                    }
                    const ImageData& image = _reconstruction.image(element.imageId);
                    if (element.featureIdx >= image.keypoints.size())
                    {
                        continue;
                    }
                    double projected[2]{};
                    double depth = 0.0;
                    if (!_reconstruction.camera(element.imageId)
                             .projectWorldPointWithDepth(point.xyz.data(), projected, depth) ||
                        !std::isfinite(depth) || depth <= 0.0)
                    {
                        continue;
                    }
                    const FeatureKeypoint& keypoint = image.keypoints[element.featureIdx];
                    referencesByImage[element.imageId][cellKey(keypoint.x, keypoint.y)].push_back(
                        {keypoint.x, keypoint.y, depth});
                }
            }

            const double radiusSquared = options.twoViewLocalDepthRadiusPixels * options.twoViewLocalDepthRadiusPixels;
            const auto localDepthAssessment = [&](const TrackElement& element,
                                                  const std::array<double, 3>& xyz) -> std::optional<double>
            {
                const auto imageReferences = referencesByImage.find(element.imageId);
                if (imageReferences == referencesByImage.end() || !_reconstruction.hasCamera(element.imageId) ||
                    !_reconstruction.hasImage(element.imageId))
                {
                    return std::nullopt;
                }
                const ImageData& image = _reconstruction.image(element.imageId);
                if (element.featureIdx >= image.keypoints.size())
                {
                    return std::nullopt;
                }

                double projected[2]{};
                double candidateDepth = 0.0;
                if (!_reconstruction.camera(element.imageId)
                         .projectWorldPointWithDepth(xyz.data(), projected, candidateDepth) ||
                    !std::isfinite(candidateDepth) || candidateDepth <= 0.0)
                {
                    return std::nullopt;
                }

                const FeatureKeypoint& keypoint = image.keypoints[element.featureIdx];
                const int centerColumn = static_cast<int>(std::floor(keypoint.x / cellSize));
                const int centerRow = static_cast<int>(std::floor(keypoint.y / cellSize));
                std::vector<double> localDepths;
                for (int columnOffset = -1; columnOffset <= 1; ++columnOffset)
                {
                    for (int rowOffset = -1; rowOffset <= 1; ++rowOffset)
                    {
                        const std::uint64_t key =
                            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(centerColumn + columnOffset))
                             << 32U) |
                            static_cast<std::uint32_t>(centerRow + rowOffset);
                        const auto cell = imageReferences->second.find(key);
                        if (cell == imageReferences->second.end())
                        {
                            continue;
                        }
                        for (const DepthReference& reference : cell->second)
                        {
                            const double dx = reference.x - keypoint.x;
                            const double dy = reference.y - keypoint.y;
                            if (dx * dx + dy * dy <= radiusSquared)
                            {
                                localDepths.push_back(reference.depth);
                            }
                        }
                    }
                }
                if (static_cast<int>(localDepths.size()) < options.twoViewLocalDepthMinReferences)
                {
                    return std::nullopt;
                }

                const double localMedian = median(localDepths);
                if (!std::isfinite(localMedian) || localMedian <= 0.0)
                {
                    return std::nullopt;
                }
                std::vector<double> absoluteDeviations;
                absoluteDeviations.reserve(localDepths.size());
                for (const double depth : localDepths)
                {
                    absoluteDeviations.push_back(std::abs(depth - localMedian));
                }
                const double madFraction = median(std::move(absoluteDeviations)) / localMedian;
                if (!std::isfinite(madFraction) || madFraction > options.twoViewLocalDepthMaxMadFraction)
                {
                    return std::nullopt;
                }
                return std::abs(candidateDepth - localMedian) / localMedian;
            };

            for (const Point3DId pointId : createdTwoViewPointIds)
            {
                if (!_reconstruction.hasPoint3D(pointId))
                {
                    continue;
                }
                const ScenePoint3D& point = _reconstruction.point3D(pointId);
                if (point.track.source != TrackSource::FeatureMatch || point.track.length() != 2)
                {
                    continue;
                }
                const std::optional<double> first = localDepthAssessment(point.track.elements[0], point.xyz);
                const std::optional<double> second = localDepthAssessment(point.track.elements[1], point.xyz);
                if (!first || !second || *first <= options.twoViewLocalDepthMaxDeviation ||
                    *second <= options.twoViewLocalDepthMaxDeviation)
                {
                    continue;
                }

                _reconstruction.deletePoint3D(pointId);
                ++stats.localDepthInconsistentTwoViewPoints;
                ++stats.numFiltered;
                --stats.numCreated;
                --stats.createdTwoViewTracks;
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
        if (!std::isfinite(result.reproj_error_rms) || result.reproj_error_rms > options.maxReprojError)
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

    int Triangulator::filterPoints(double maxReprojError, double minTriAngle)
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
                    const double error = computeReprojError(pt.xyz, element.imageId, element.featureIdx);
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
                    double reprErr = computeReprojError(pt.xyz, imgId, fi);
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

    // ---- 多视图 DLT 三角化 ----

    bool Triangulator::triangulateMultiView(const std::vector<TrackElement>& observations,
                                            std::array<double, 3>& outXyz) const
    {
        // 收集有效观测的投影矩阵和像点坐标
        // 投影矩阵 P = K * [R | -R*C]，其中 R 是 camera-to-world 的逆
        std::vector<cv::Mat> projMats;
        std::vector<cv::Point2d> pts;

        for (const auto& elem : observations)
        {
            if (!_reconstruction.isRegistered(elem.imageId))
                continue;
            if (!_reconstruction.hasCamera(elem.imageId))
                continue;
            const FramePinholeCamera& cam = _reconstruction.camera(elem.imageId);
            const ImageData& img = _reconstruction.image(elem.imageId);
            if (elem.featureIdx >= img.keypoints.size())
                continue;

            projMats.push_back(openCvProjectionMatrix(cam));
            pts.emplace_back(img.keypoints[elem.featureIdx].x, img.keypoints[elem.featureIdx].y);
        }

        if (projMats.size() < 2)
            return false;

        // 构造 DLT 线性系统: 对每个观测 (P, x)，添加两行到 A
        //   x * P[2,:] - P[0,:]
        //   y * P[2,:] - P[1,:]
        const int nObs = static_cast<int>(projMats.size());
        cv::Mat A(2 * nObs, 4, CV_64F);

        for (int i = 0; i < nObs; ++i)
        {
            const cv::Mat& P = projMats[i];
            double x = pts[i].x, y = pts[i].y;
            for (int j = 0; j < 4; ++j)
            {
                A.at<double>(2 * i, j) = x * P.at<double>(2, j) - P.at<double>(0, j);
                A.at<double>(2 * i + 1, j) = y * P.at<double>(2, j) - P.at<double>(1, j);
            }
        }

        // SVD 求解: X 对应最小奇异值的右奇异向量
        cv::Mat W, U, Vt;
        cv::SVD::compute(A, W, U, Vt, cv::SVD::FULL_UV);
        cv::Mat X4 = Vt.row(3).t();

        if (std::fabs(X4.at<double>(3)) < 1e-10)
            return false;

        outXyz[0] = X4.at<double>(0) / X4.at<double>(3);
        outXyz[1] = X4.at<double>(1) / X4.at<double>(3);
        outXyz[2] = X4.at<double>(2) / X4.at<double>(3);

        return std::isfinite(outXyz[0]) && std::isfinite(outXyz[1]) && std::isfinite(outXyz[2]);
    }

    // ---- 重三角化所有点 ----

    int Triangulator::retriangulatePoints(double maxReprojError)
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

        common::concurrency::parallelForIndices(allIds.size(),
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
                                                        double err =
                                                            computeReprojError(pt.xyz, elem.imageId, elem.featureIdx);
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
                                                        double err =
                                                            computeReprojError(newXyz, elem.imageId, elem.featureIdx);
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
