#include "Triangulator.h"
#include "geometry/OpenCvCameraAdapter.h"
#include "concurrency/SafeWorkerGroup.h"

#include "log/Logger.h"

#include "OpenCvCompat.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>

namespace xjw
{

Triangulator::Triangulator(SfmReconstruction &reconstruction,
                           const CorrespondenceGraph &graph,
                           int threadCount)
    : _reconstruction(reconstruction),
      _correspondenceGraph(graph),
      _threadCount(threadCount > 0
                       ? threadCount
                       : static_cast<int>(
                             std::max(1u, std::thread::hardware_concurrency())))
{
}

/**
 * @brief 三角化器构造函数。
 *
 * 持有对 `SfmReconstruction` 与 `CorrespondenceGraph` 的引用，
 * 在增量注册过程中执行三角化与过滤任务。
 */

// ---- 核心：对新注册图像的特征执行三角化 ----

TriangulationStats Triangulator::triangulateImage(ImageId imageId, const TriangulatorOptions &options)
{
    TriangulationStats stats;

    if (!_reconstruction.isRegistered(imageId))
    {
        return stats;
    }
    const ImageData &imgData = _reconstruction.image(imageId);

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

        for (const auto &corr : corrs)
        {
            // 对应图像必须已注册
            if (!_reconstruction.isRegistered(corr.imageId))
            {
                continue;
            }
            const ImageData &otherImg = _reconstruction.image(corr.imageId);

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
                const auto &pt = _reconstruction.point3D(p3dId);
                double reprErr = computeReprojError(pt.xyz, imageId, fi);
                if (reprErr <= options.continueMaxReprojError)
                {
                    // 追加观测到轨迹
                    auto &mutPt = _reconstruction.point3D(p3dId);
                    mutPt.track.elements.push_back({imageId, fi});

                    // 更新图像的三维点关联
                    auto &p3dIds = _reconstruction.image(imageId).point3DIds;
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
        for (const auto &corr : corrs)
        {
            if (!_reconstruction.isRegistered(corr.imageId))
            {
                continue;
            }

            // 新点只能使用尚未归属三维点的两端观测。若另一端已经属于已有点，
            // 即使轨迹延续因重投影误差失败，也不能覆盖其 point3DId，否则同一
            // 2D 特征会同时出现在多个三维点轨迹中。
            const ImageData &otherImg = _reconstruction.image(corr.imageId);
            if (corr.featureIdx >= otherImg.point3DIds.size() ||
                otherImg.point3DIds[corr.featureIdx] != kInvalidPoint3DId)
            {
                continue;
            }

            std::array<double, 3> xyz;
            if (triangulatePair(imageId, fi, corr.imageId, corr.featureIdx, options, xyz))
            {
                // 创建新三维点
                Track track;
                track.elements.push_back({imageId, fi});
                track.elements.push_back({corr.imageId, corr.featureIdx});
                const std::string first_source = _correspondenceGraph.priorTrackId(imageId, fi);
                const std::string second_source =
                    _correspondenceGraph.priorTrackId(corr.imageId, corr.featureIdx);
                if (!first_source.empty() && first_source == second_source)
                {
                    track.source = TrackSource::PriorMarker;
                    track.sourceId = first_source;
                }

                _reconstruction.addPoint3DWithTrack(xyz, track);

                stats.numCreated++;
                break;
            }
        }
    }

    return stats;
}

TriangulationStats Triangulator::triangulateTracks(const std::vector<Track> &tracks,
                                                   const TriangulatorOptions &options)
{
    TriangulationStats stats;

    for (const Track &track : tracks)
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
        for (const TrackElement &element : track.elements)
        {
            if (!_reconstruction.isRegistered(element.imageId) ||
                !_reconstruction.hasCamera(element.imageId) ||
                !_reconstruction.hasImage(element.imageId))
            {
                usable = false;
                break;
            }

            const ImageData &image = _reconstruction.image(element.imageId);
            if (element.featureIdx >= image.keypoints.size() ||
                element.featureIdx >= image.point3DIds.size() ||
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

        std::vector<TrackElement> remainingElements = track.elements;
        bool createdAnyPointForTrack = false;
        struct Candidate
        {
            std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
            Track inlierTrack;
            double rmsError = std::numeric_limits<double>::infinity();
            double nearestRejectedReprojError = std::numeric_limits<double>::infinity();
            bool valid = false;
        };
        auto collectCandidateFromPoint = [&](const std::array<double, 3> &xyz, bool countRejects) -> Candidate
        {
            Candidate candidate;
            double squaredErrorSum = 0.0;
            for (const TrackElement &element : remainingElements)
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

            if (candidate.inlierTrack.length() < 2 ||
                computeMaxTriangulationAngle(xyz, candidate.inlierTrack.elements) < options.minTriAngle)
            {
                return Candidate{};
            }

            // 内点重建不能丢失人工标记的来源，否则后续控制点/检查点角色无法传入 BA。
            candidate.inlierTrack.confidence = track.confidence;
            candidate.inlierTrack.source = track.source;
            candidate.inlierTrack.sourceId = track.sourceId;
            candidate.xyz = xyz;
            candidate.rmsError = std::sqrt(squaredErrorSum / static_cast<double>(candidate.inlierTrack.length()));
            candidate.valid = true;
            return candidate;
        };
        auto betterCandidate = [](const Candidate &candidate, const Candidate &best)
        {
            return candidate.valid &&
                   (!best.valid ||
                    candidate.inlierTrack.length() > best.inlierTrack.length() ||
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
            seedPairs.reserve(std::min(
                kMaximumSeedPairs,
                remainingElements.size() * (remainingElements.size() - 1) / 2));
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
            if (!allViewCandidateUsesEveryObservation &&
                allPairCount <= kMaximumSeedPairs)
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

                const std::size_t sampleStride = std::max<std::size_t>(
                    1, allPairCount / kMaximumSeedPairs);
                std::size_t flatPairIndex = 0;
                for (std::size_t left = 0;
                     left < elementCount && seedPairs.size() < kMaximumSeedPairs;
                     ++left)
                {
                    for (std::size_t right = left + 1;
                         right < elementCount && seedPairs.size() < kMaximumSeedPairs;
                         ++right, ++flatPairIndex)
                    {
                        if (flatPairIndex % sampleStride == 0)
                        {
                            addSeedPair(left, right);
                        }
                    }
                }
            }

            for (const auto &[left, right] : seedPairs)
            {
                std::array<double, 3> seedXyz;
                const TrackElement &leftElement = remainingElements[left];
                const TrackElement &rightElement = remainingElements[right];
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
                if (track.length() >= 3 &&
                    seedCandidate.valid &&
                    seedCandidate.inlierTrack.length() == 2 &&
                    std::isfinite(seedCandidate.nearestRejectedReprojError))
                {
                    nearestRejectedExtraForTrack = std::min(
                        nearestRejectedExtraForTrack,
                        seedCandidate.nearestRejectedReprojError);
                }
            }

            if (!best.valid)
            {
                break;
            }

            const Point3DId pointId = _reconstruction.addPoint3DWithTrack(best.xyz, best.inlierTrack);
            if (_reconstruction.hasPoint3D(pointId))
            {
                _reconstruction.point3D(pointId).error = best.rmsError;
            }
            ++stats.numCreated;
            stats.numContinued += static_cast<int>(std::max<std::size_t>(0, best.inlierTrack.length() - 2));
            createdAnyPointForTrack = true;
            if (best.inlierTrack.length() >= 3)
            {
                ++stats.createdLongTracks;
            }
            else
            {
                ++stats.createdTwoViewTracks;
                if (track.length() >= 3)
                {
                    ++stats.longTrackTwoViewOnly;
                    if (std::isfinite(nearestRejectedExtraForTrack))
                    {
                        ++stats.longTrackRejectedExtraSamples;
                        stats.longTrackRejectedExtraErrorSum += nearestRejectedExtraForTrack;
                        stats.longTrackRejectedExtraErrorMax = std::max(stats.longTrackRejectedExtraErrorMax,
                                                                        nearestRejectedExtraForTrack);
                        if (nearestRejectedExtraForTrack <= 5.0)
                        {
                            ++stats.longTrackRejectedExtraLe5;
                        }
                        else if (nearestRejectedExtraForTrack <= 10.0)
                        {
                            ++stats.longTrackRejectedExtraLe10;
                        }
                        else if (nearestRejectedExtraForTrack <= 25.0)
                        {
                            ++stats.longTrackRejectedExtraLe25;
                        }
                        else
                        {
                            ++stats.longTrackRejectedExtraGt25;
                        }
                    }
                }
            }

            remainingElements.erase(
                std::remove_if(remainingElements.begin(),
                               remainingElements.end(),
                               [&best](const TrackElement &element)
                               {
                                   return std::any_of(best.inlierTrack.elements.begin(),
                                                      best.inlierTrack.elements.end(),
                                                      [&element](const TrackElement &used)
                                                      {
                                                          return used.imageId == element.imageId &&
                                                                 used.featureIdx == element.featureIdx;
                                                      });
                               }),
                remainingElements.end());
        }

        if (!createdAnyPointForTrack)
        {
            ++stats.noCandidateTracks;
        }
    }

    return stats;
}

// ---- 双目三角化 ----

bool Triangulator::triangulatePair(ImageId imgId1, FeatureIdx featIdx1, ImageId imgId2, FeatureIdx featIdx2,
                                   const TriangulatorOptions &options, std::array<double, 3> &outXyz)
{
    if (!_reconstruction.hasCamera(imgId1) || !_reconstruction.hasCamera(imgId2))
    {
        return false;
    }

    const Camera &cam1 = _reconstruction.camera(imgId1);
    const Camera &cam2 = _reconstruction.camera(imgId2);

    const ImageData &img1 = _reconstruction.image(imgId1);
    const ImageData &img2 = _reconstruction.image(imgId2);

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

double Triangulator::computeReprojError(const std::array<double, 3> &xyz, ImageId imageId, FeatureIdx featureIdx) const
{
    if (!_reconstruction.hasCamera(imageId))
    {
        return 1e9;
    }
    const Camera &cam = _reconstruction.camera(imageId);
    const ImageData &img = _reconstruction.image(imageId);

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

// ---- 过滤低质量三维点 ----

int Triangulator::filterPoints(double maxReprojError, double minTriAngle)
{
    auto allIds = _reconstruction.allPoint3DIds();
    std::vector<char> shouldFilter(allIds.size(), 0);

    common::concurrency::parallelForIndices(
        allIds.size(), static_cast<std::size_t>(_threadCount), [&](std::size_t pointIndex)
        {
        const Point3DId pid = allIds[pointIndex];
        if (!_reconstruction.hasPoint3D(pid))
        {
            return;
        }
        const auto &pt = _reconstruction.point3D(pid);

        bool filterPoint = false;

        // 使用已存储的重投影 RMS 误差（BA 后由 recomputeReprojErrors 更新）
        if (!std::isfinite(pt.error) || pt.error > maxReprojError)
        {
            filterPoint = true;
        }

        // 检查三角化角：遍历所有观测相机对，取最大角
        if (!filterPoint && pt.track.length() >= 2)
        {
            double maxAngle = 0.0;
            const auto &elems = pt.track.elements;

            for (size_t i = 0; i < elems.size() && maxAngle < minTriAngle; ++i)
            {
                if (!_reconstruction.hasCamera(elems[i].imageId))
                {
                    continue;
                }
                const Camera &ci = _reconstruction.camera(elems[i].imageId);
                auto Ci = ci.cameraCenter();

                for (size_t j = i + 1; j < elems.size(); ++j)
                {
                    if (!_reconstruction.hasCamera(elems[j].imageId))
                    {
                        continue;
                    }
                    const Camera &cj = _reconstruction.camera(elems[j].imageId);
                    auto Cj = cj.cameraCenter();

                    double v0[3] = {pt.xyz[0] - Ci[0], pt.xyz[1] - Ci[1], pt.xyz[2] - Ci[2]};
                    double v1[3] = {pt.xyz[0] - Cj[0], pt.xyz[1] - Cj[1], pt.xyz[2] - Cj[2]};
                    double len0 = std::sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
                    double len1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);

                    if (len0 > 1e-9 && len1 > 1e-9)
                    {
                        double cosAngle = (v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]) / (len0 * len1);
                        cosAngle = std::max(-1.0, std::min(1.0, cosAngle));
                        double angle = std::acos(cosAngle) * 180.0 / M_PI;
                        if (angle > maxAngle)
                        {
                            maxAngle = angle;
                        }
                    }
                }
            }

            if (maxAngle < minTriAngle)
            {
                filterPoint = true;
            }
        }

        shouldFilter[pointIndex] = filterPoint ? 1 : 0;
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

// ---- 过滤短轨迹 ----

int Triangulator::filterShortTracks(int minTrackLen)
{
    auto allIds = _reconstruction.allPoint3DIds();
    std::vector<char> shouldFilter(allIds.size(), 0);

    common::concurrency::parallelForIndices(
        allIds.size(), static_cast<std::size_t>(_threadCount), [&](std::size_t pointIndex)
        {
        const Point3DId pid = allIds[pointIndex];
        if (!_reconstruction.hasPoint3D(pid))
        {
            return;
        }
        const auto &pt = _reconstruction.point3D(pid);

        // 统计有效观测数量（对应已注册图像的观测）
        int validObs = 0;
        for (const auto &elem : pt.track.elements)
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

int Triangulator::completeTracks(const TriangulatorOptions &options)
{
    int numCompleted = 0;
    auto regIds = _reconstruction.registeredImageIds();

    for (ImageId imgId : regIds)
    {
        const ImageData &imgData = _reconstruction.image(imgId);
        const size_t numFeatures = imgData.keypoints.size();

        for (FeatureIdx fi = 0; fi < static_cast<FeatureIdx>(numFeatures); ++fi)
        {
            // 只处理未关联三维点的特征
            if (fi < imgData.point3DIds.size() && imgData.point3DIds[fi] != kInvalidPoint3DId)
            {
                continue;
            }

            auto corrs = _correspondenceGraph.findCorrespondences(imgId, fi);
            for (const auto &corr : corrs)
            {
                if (!_reconstruction.isRegistered(corr.imageId))
                {
                    continue;
                }
                const ImageData &otherImg = _reconstruction.image(corr.imageId);
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

                const auto &pt = _reconstruction.point3D(p3dId);
                const bool alreadyHasObservationInImage =
                    std::any_of(pt.track.elements.begin(),
                                pt.track.elements.end(),
                                [imgId](const TrackElement &element)
                                {
                                    return element.imageId == imgId;
                                });
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

                auto &mutPt = _reconstruction.point3D(p3dId);
                mutPt.track.elements.push_back({imgId, fi});
                auto &p3dIds = _reconstruction.image(imgId).point3DIds;
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

bool Triangulator::hasPositiveDepth(const std::array<double, 3> &xyz, ImageId imageId) const
{
    if (!_reconstruction.hasCamera(imageId))
        return false;
    const Camera &cam = _reconstruction.camera(imageId);
    const double world[3] = {xyz[0], xyz[1], xyz[2]};
    return cam.isPointInFront(world);
}

double Triangulator::computeMaxTriangulationAngle(const std::array<double, 3> &xyz,
                                                  const std::vector<TrackElement> &observations) const
{
    double maxAngle = 0.0;

    for (size_t i = 0; i < observations.size(); ++i)
    {
        if (!_reconstruction.hasCamera(observations[i].imageId))
        {
            continue;
        }

        const Camera &cameraI = _reconstruction.camera(observations[i].imageId);
        const auto centerI = cameraI.cameraCenter();

        for (size_t j = i + 1; j < observations.size(); ++j)
        {
            if (!_reconstruction.hasCamera(observations[j].imageId))
            {
                continue;
            }

            const Camera &cameraJ = _reconstruction.camera(observations[j].imageId);
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

bool Triangulator::triangulateMultiView(const std::vector<TrackElement> &observations,
                                        std::array<double, 3> &outXyz) const
{
    // 收集有效观测的投影矩阵和像点坐标
    // 投影矩阵 P = K * [R | -R*C]，其中 R 是 camera-to-world 的逆
    std::vector<cv::Mat> projMats;
    std::vector<cv::Point2d> pts;

    for (const auto &elem : observations)
    {
        if (!_reconstruction.isRegistered(elem.imageId))
            continue;
        if (!_reconstruction.hasCamera(elem.imageId))
            continue;
        const Camera &cam = _reconstruction.camera(elem.imageId);
        const ImageData &img = _reconstruction.image(elem.imageId);
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
        const cv::Mat &P = projMats[i];
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

    common::concurrency::parallelForIndices(
        allIds.size(), static_cast<std::size_t>(_threadCount), [&](std::size_t pointIndex)
        {
        const Point3DId pid = allIds[pointIndex];
        if (!_reconstruction.hasPoint3D(pid))
            return;
        const auto &pt = _reconstruction.point3D(pid);

        if (pt.track.length() < 2)
            return;

        // 计算原始平均重投影误差
        double oldAvgErr = 0.0;
        int oldValidObs = 0;
        for (const auto &elem : pt.track.elements)
        {
            double err = computeReprojError(pt.xyz, elem.imageId, elem.featureIdx);
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
        for (const auto &elem : pt.track.elements)
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
        for (const auto &elem : pt.track.elements)
        {
            double err = computeReprojError(newXyz, elem.imageId, elem.featureIdx);
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
        if (!updates[pointIndex].apply ||
            !_reconstruction.hasPoint3D(allIds[pointIndex]))
        {
            continue;
        }
        auto &point = _reconstruction.point3D(allIds[pointIndex]);
        point.xyz = updates[pointIndex].xyz;
        point.error = updates[pointIndex].error;
        ++improved;
    }

    const double elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
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
    common::concurrency::parallelForIndices(
        allIds.size(), static_cast<std::size_t>(_threadCount), [&](std::size_t pointIndex)
        {
        const Point3DId pid = allIds[pointIndex];
        if (!_reconstruction.hasPoint3D(pid))
            return;
        const auto &pt = _reconstruction.point3D(pid);

        double sumErr = 0.0;
        int cnt = 0;
        for (const auto &elem : pt.track.elements)
        {
            double err = computeReprojError(pt.xyz, elem.imageId, elem.featureIdx);
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
        if (hasPoint[pointIndex] &&
            _reconstruction.hasPoint3D(allIds[pointIndex]))
        {
            _reconstruction.point3D(allIds[pointIndex]).error =
                errors[pointIndex];
        }
    }
}

} // namespace xjw
