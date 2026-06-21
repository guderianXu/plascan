#include "IncrementalSfm.h"
#include "Intersection.h"
#include "tracks/MultiViewTrackBuilder.h"

#include "log/Logger.h"

#include "OpenCvCompat.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace xjw
{

namespace
{

struct KnownPoseTriangulationPolicy
{
    TriangulatorOptions triangulatorOptions;
    double filterMinTriAngle = 2.0;
    bool adapted = false;
    int validCandidates = 0;
    int acceptedWithDefault = 0;
    int acceptedWithAdapted = 0;
    double chosenMinTriAngle = 2.0;
};

constexpr int kKnownPoseMinLongInputTracksForQualityGate = 10;
constexpr int kKnownPoseMinPointsForTrackRatioGate = 20;
constexpr double kKnownPoseMaxTwoViewTrackRatio = 0.95;

double percentile(std::vector<double> values, double ratio)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const double clampedRatio = std::max(0.0, std::min(1.0, ratio));
    const auto index = static_cast<size_t>(std::round(clampedRatio * static_cast<double>(values.size() - 1)));
    return values[index];
}

KnownPoseTriangulationPolicy resolveKnownPoseTriangulationPolicy(
    const std::shared_ptr<SfmReconstruction> &reconstruction,
    const CorrespondenceGraph &correspondenceGraph,
    const std::vector<ImageId> &imageIds,
    const IncrementalSfmOptions &options)
{
    KnownPoseTriangulationPolicy policy;
    policy.triangulatorOptions = options.triangulatorOptions;
    policy.filterMinTriAngle = options.filterMinTriAngle;
    policy.chosenMinTriAngle = options.triangulatorOptions.minTriAngle;

    constexpr int minUsefulTriangulations = 20;
    constexpr double relaxedMinTriAngle = 0.1;
    constexpr double absoluteMinTriAngle = 0.05;

    std::vector<double> validAngles;
    validAngles.reserve(1024);

    for (ImageId imageId : imageIds)
    {
        if (!reconstruction->isRegistered(imageId) || !reconstruction->hasCamera(imageId))
        {
            continue;
        }

        const ImageData &image = reconstruction->image(imageId);
        const Camera &camera = reconstruction->camera(imageId);

        for (FeatureIdx featureIdx = 0; featureIdx < static_cast<FeatureIdx>(image.keypoints.size()); ++featureIdx)
        {
            const auto correspondences = correspondenceGraph.findCorrespondences(imageId, featureIdx);
            for (const auto &correspondence : correspondences)
            {
                if (correspondence.imageId <= imageId ||
                    !reconstruction->isRegistered(correspondence.imageId) ||
                    !reconstruction->hasCamera(correspondence.imageId))
                {
                    continue;
                }

                const ImageData &otherImage = reconstruction->image(correspondence.imageId);
                if (correspondence.featureIdx >= otherImage.keypoints.size())
                {
                    continue;
                }

                const Camera &otherCamera = reconstruction->camera(correspondence.imageId);
                const auto &keypoint = image.keypoints[featureIdx];
                const auto &otherKeypoint = otherImage.keypoints[correspondence.featureIdx];
                const auto triResult = Intersection::intersectPair(camera,
                                                                    keypoint.x,
                                                                    keypoint.y,
                                                                    otherCamera,
                                                                    otherKeypoint.x,
                                                                    otherKeypoint.y);
                if (!triResult.valid ||
                    !std::isfinite(triResult.angle_deg) ||
                    !std::isfinite(triResult.reproj_error_rms) ||
                    triResult.reproj_error_rms > options.triangulatorOptions.maxReprojError)
                {
                    continue;
                }

                validAngles.push_back(triResult.angle_deg);
            }
        }
    }

    policy.validCandidates = static_cast<int>(validAngles.size());
    policy.acceptedWithDefault = static_cast<int>(std::count_if(validAngles.begin(),
                                                                validAngles.end(),
                                                                [&](double angle)
                                                                {
                                                                    return angle >= options.triangulatorOptions.minTriAngle;
                                                                }));

    if (policy.acceptedWithDefault >= minUsefulTriangulations ||
        policy.validCandidates < minUsefulTriangulations)
    {
        policy.acceptedWithAdapted = policy.acceptedWithDefault;
        return policy;
    }

    int acceptedWithRelaxed = static_cast<int>(std::count_if(validAngles.begin(),
                                                            validAngles.end(),
                                                            [](double angle)
                                                            {
                                                                return angle >= relaxedMinTriAngle;
                                                            }));

    double adaptedMinTriAngle = relaxedMinTriAngle;
    if (acceptedWithRelaxed < minUsefulTriangulations)
    {
        adaptedMinTriAngle = std::max(absoluteMinTriAngle, percentile(validAngles, 0.20) * 0.8);
        adaptedMinTriAngle = std::min(adaptedMinTriAngle, relaxedMinTriAngle);
    }

    policy.triangulatorOptions.minTriAngle = std::min(options.triangulatorOptions.minTriAngle, adaptedMinTriAngle);
    policy.filterMinTriAngle = std::min(options.filterMinTriAngle, policy.triangulatorOptions.minTriAngle);
    policy.chosenMinTriAngle = policy.triangulatorOptions.minTriAngle;
    policy.adapted = policy.chosenMinTriAngle < options.triangulatorOptions.minTriAngle;
    policy.acceptedWithAdapted = static_cast<int>(std::count_if(validAngles.begin(),
                                                                validAngles.end(),
                                                                [&](double angle)
                                                                {
                                                                    return angle >= policy.chosenMinTriAngle;
                                                                }));

    return policy;
}

bool knownPoseMatchPassesGeometry(const SfmReconstruction &reconstruction,
                                  ImageId imageId,
                                  ImageId otherImageId,
                                  const FeatureMatch &match,
                                  const TriangulatorOptions &options)
{
    if (!reconstruction.hasCamera(imageId) ||
        !reconstruction.hasCamera(otherImageId) ||
        !reconstruction.hasImage(imageId) ||
        !reconstruction.hasImage(otherImageId))
    {
        return false;
    }

    const ImageData &image = reconstruction.image(imageId);
    const ImageData &otherImage = reconstruction.image(otherImageId);
    if (match.idx1 >= image.keypoints.size() || match.idx2 >= otherImage.keypoints.size())
    {
        return false;
    }

    const Camera &camera = reconstruction.camera(imageId);
    const Camera &otherCamera = reconstruction.camera(otherImageId);
    const FeatureKeypoint &keypoint = image.keypoints[match.idx1];
    const FeatureKeypoint &otherKeypoint = otherImage.keypoints[match.idx2];
    const auto triResult = Intersection::intersectPair(camera,
                                                       keypoint.x,
                                                       keypoint.y,
                                                       otherCamera,
                                                       otherKeypoint.x,
                                                       otherKeypoint.y);
    if (!triResult.valid ||
        !std::isfinite(triResult.angle_deg) ||
        !std::isfinite(triResult.reproj_error_rms))
    {
        return false;
    }

    return triResult.angle_deg >= options.minTriAngle &&
           triResult.reproj_error_rms <= options.maxReprojError;
}

struct SimilarityTransform3d
{
    bool valid = false;
    double scale = 1.0;
    std::array<double, 9> rotation{{1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0}};
    std::array<double, 3> translation{{0.0, 0.0, 0.0}};
    int inlierCount = 0;
    double rmse = 0.0;
};

std::array<double, 3> transformPoint(const SimilarityTransform3d &transform,
                                     const std::array<double, 3> &point)
{
    std::array<double, 3> out{};
    for (int r = 0; r < 3; ++r)
    {
        out[static_cast<size_t>(r)] = transform.translation[static_cast<size_t>(r)];
        for (int c = 0; c < 3; ++c)
        {
            out[static_cast<size_t>(r)] +=
                transform.scale *
                transform.rotation[static_cast<size_t>(r * 3 + c)] *
                point[static_cast<size_t>(c)];
        }
    }
    return out;
}

std::array<double, 9> multiplyRotation(const std::array<double, 9> &left,
                                       const std::array<double, 9> &right)
{
    std::array<double, 9> result{};
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            double value = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                value += left[static_cast<size_t>(r * 3 + k)] *
                         right[static_cast<size_t>(k * 3 + c)];
            }
            result[static_cast<size_t>(r * 3 + c)] = value;
        }
    }
    return result;
}

double pointDistance(const std::array<double, 3> &a,
                     const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double centerExtent(const std::vector<std::array<double, 3>> &points)
{
    double extent = 0.0;
    for (size_t i = 0; i < points.size(); ++i)
    {
        for (size_t j = i + 1; j < points.size(); ++j)
        {
            extent = std::max(extent, pointDistance(points[i], points[j]));
        }
    }
    return extent;
}

SimilarityTransform3d estimateSimilarityUmeyama(const std::vector<std::array<double, 3>> &source,
                                                const std::vector<std::array<double, 3>> &target)
{
    SimilarityTransform3d transform;
    if (source.size() != target.size() || source.size() < 3)
    {
        return transform;
    }

    std::array<double, 3> sourceMean{{0.0, 0.0, 0.0}};
    std::array<double, 3> targetMean{{0.0, 0.0, 0.0}};
    for (size_t i = 0; i < source.size(); ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            sourceMean[static_cast<size_t>(k)] += source[i][static_cast<size_t>(k)];
            targetMean[static_cast<size_t>(k)] += target[i][static_cast<size_t>(k)];
        }
    }
    for (int k = 0; k < 3; ++k)
    {
        sourceMean[static_cast<size_t>(k)] /= static_cast<double>(source.size());
        targetMean[static_cast<size_t>(k)] /= static_cast<double>(target.size());
    }

    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    double sourceVariance = 0.0;
    for (size_t i = 0; i < source.size(); ++i)
    {
        std::array<double, 3> srcCentered{};
        std::array<double, 3> dstCentered{};
        for (int k = 0; k < 3; ++k)
        {
            srcCentered[static_cast<size_t>(k)] =
                source[i][static_cast<size_t>(k)] - sourceMean[static_cast<size_t>(k)];
            dstCentered[static_cast<size_t>(k)] =
                target[i][static_cast<size_t>(k)] - targetMean[static_cast<size_t>(k)];
            sourceVariance += srcCentered[static_cast<size_t>(k)] * srcCentered[static_cast<size_t>(k)];
        }

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                covariance.at<double>(r, c) +=
                    dstCentered[static_cast<size_t>(r)] * srcCentered[static_cast<size_t>(c)];
            }
        }
    }

    sourceVariance /= static_cast<double>(source.size());
    if (!(sourceVariance > 1e-18))
    {
        return transform;
    }
    covariance /= static_cast<double>(source.size());

    cv::SVD svd(covariance, cv::SVD::FULL_UV);
    cv::Mat sign = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat rotation = svd.u * svd.vt;
    if (cv::determinant(rotation) < 0.0)
    {
        sign.at<double>(2, 2) = -1.0;
        rotation = svd.u * sign * svd.vt;
    }

    double scaleNumerator = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        scaleNumerator += svd.w.at<double>(i) * sign.at<double>(i, i);
    }
    const double scale = scaleNumerator / sourceVariance;
    if (!(scale > 0.0) || !std::isfinite(scale))
    {
        return transform;
    }

    transform.valid = true;
    transform.scale = scale;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            transform.rotation[static_cast<size_t>(r * 3 + c)] = rotation.at<double>(r, c);
        }
    }

    const auto mappedSourceMean = transformPoint(transform, sourceMean);
    for (int k = 0; k < 3; ++k)
    {
        transform.translation[static_cast<size_t>(k)] +=
            targetMean[static_cast<size_t>(k)] - mappedSourceMean[static_cast<size_t>(k)];
    }

    double sum2 = 0.0;
    for (size_t i = 0; i < source.size(); ++i)
    {
        const double residual = pointDistance(transformPoint(transform, source[i]), target[i]);
        sum2 += residual * residual;
    }
    transform.inlierCount = static_cast<int>(source.size());
    transform.rmse = std::sqrt(sum2 / static_cast<double>(source.size()));
    return transform;
}

SimilarityTransform3d estimateRobustCameraCenterSimilarity(
    const std::vector<std::array<double, 3>> &source,
    const std::vector<std::array<double, 3>> &target)
{
    SimilarityTransform3d best;
    if (source.size() != target.size() || source.size() < 3)
    {
        return best;
    }

    const double extent = std::max(centerExtent(source), centerExtent(target));
    const double inlierThreshold = std::max(1e-4, extent * 0.03);
    constexpr int maxSamples = 256;
    int sampleCount = 0;

    for (size_t i = 0; i + 2 < source.size() && sampleCount < maxSamples; ++i)
    {
        for (size_t j = i + 1; j + 1 < source.size() && sampleCount < maxSamples; ++j)
        {
            for (size_t k = j + 1; k < source.size() && sampleCount < maxSamples; ++k)
            {
                ++sampleCount;
                std::vector<std::array<double, 3>> sampleSource{source[i], source[j], source[k]};
                std::vector<std::array<double, 3>> sampleTarget{target[i], target[j], target[k]};
                const SimilarityTransform3d candidate = estimateSimilarityUmeyama(sampleSource, sampleTarget);
                if (!candidate.valid)
                {
                    continue;
                }

                std::vector<std::array<double, 3>> inlierSource;
                std::vector<std::array<double, 3>> inlierTarget;
                inlierSource.reserve(source.size());
                inlierTarget.reserve(target.size());
                double sum2 = 0.0;
                for (size_t idx = 0; idx < source.size(); ++idx)
                {
                    const double residual = pointDistance(transformPoint(candidate, source[idx]), target[idx]);
                    if (residual <= inlierThreshold)
                    {
                        inlierSource.push_back(source[idx]);
                        inlierTarget.push_back(target[idx]);
                        sum2 += residual * residual;
                    }
                }

                if (inlierSource.size() < 3)
                {
                    continue;
                }

                const double rmse = std::sqrt(sum2 / static_cast<double>(inlierSource.size()));
                if (!best.valid ||
                    inlierSource.size() > static_cast<size_t>(best.inlierCount) ||
                    (inlierSource.size() == static_cast<size_t>(best.inlierCount) && rmse < best.rmse))
                {
                    best = estimateSimilarityUmeyama(inlierSource, inlierTarget);
                    best.inlierCount = static_cast<int>(inlierSource.size());
                    best.rmse = rmse;
                }
            }
        }
    }

    if (!best.valid)
    {
        best = estimateSimilarityUmeyama(source, target);
    }
    return best;
}

} // namespace

// ============================================================
// 构造
// ============================================================

IncrementalSfm::IncrementalSfm(const IncrementalSfmOptions &options)
    : _sfmOptions(options), _reconstruction(std::make_shared<SfmReconstruction>())
{
}

/**
 * @brief 构造函数，使用给定选项初始化内部状态。
 *
 * 仅调整内部成员，实际重建在 run() 中执行。
 */

// ============================================================
// 数据输入
// ============================================================

void IncrementalSfm::addImage(ImageId id, const std::string &imagePath, const std::string &cameraPath,
                              const std::vector<FeatureKeypoint> &keypoints)
{
    ImageData data;
    data.id = id;
    data.imagePath = imagePath;
    data.cameraPath = cameraPath;
    data.keypoints = keypoints;
    data.point3DIds.resize(keypoints.size(), kInvalidPoint3DId);

    _reconstruction->addImage(data);
    _correspondenceGraph.addImage(id, keypoints.size());
    _cameraPaths[id] = cameraPath;
}

void IncrementalSfm::addImageWithCamera(ImageId id, const std::string &imagePath, const Camera &camera,
                                        const std::vector<FeatureKeypoint> &keypoints)
{
    ImageData data;
    data.id = id;
    data.imagePath = imagePath;
    data.cameraPath = ""; // 无文件
    data.keypoints = keypoints;
    data.point3DIds.resize(keypoints.size(), kInvalidPoint3DId);

    _reconstruction->addImage(data);
    _correspondenceGraph.addImage(id, keypoints.size());
    _preloadedCameras[id] = camera;
}

void IncrementalSfm::addMatches(ImageId id1, ImageId id2, const std::vector<FeatureMatch> &matches)
{
    _correspondenceGraph.addMatches(id1, id2, matches);
}

// ============================================================
// 主流程
// ============================================================

IncrementalSfmResult IncrementalSfm::run(SfmProgressCallback progressCb)
{
    IncrementalSfmResult result;
    _isAborted = false;

    const int totalImages = static_cast<int>(_reconstruction->numImages());
    if (totalImages < 2)
    {
        result.summary = "At least 2 images required for SfM";
        return result;
    }

    Logger::instance()->infof(
        "[SFM] Incremental run started: images=%d, useKnownCameraPoses=%s, initMinMatches=%d, "
        "initMinInliers=%d, initMinTriAngle=%.3f, pnpMinInliers=%d, pnpMaxReproj=%.3f",
        totalImages,
        _sfmOptions.useKnownCameraPoses ? "true" : "false",
        _sfmOptions.initMinNumMatches,
        _sfmOptions.initMinNumInliers,
        _sfmOptions.initMinTriAngle,
        _sfmOptions.pnpOptions.minNumInliers,
        _sfmOptions.pnpOptions.maxReprojError);

    // ---- 步骤 0：构建对应关系图 ----
    _correspondenceGraph.buildCorrespondences();
    Logger::instance()->infof("[SFM] Correspondence graph ready: images=%zu, imagePairs=%zu",
                              _correspondenceGraph.numImages(),
                              _correspondenceGraph.numImagePairs());

    if (!reportProgress(0, totalImages, "Building correspondence graph...", progressCb))
        return result;

    if (_sfmOptions.useKnownCameraPoses)
    {
        return runKnownCameraPoseReconstruction(progressCb);
    }

    // ---- 步骤 1：选择并初始化初始像对 ----
    ImageId initId1, initId2;
    bool initOk = false;

    if (_sfmOptions.autoSelectInitPair)
    {
        // ── COLMAP 式多候选重试 ──
        auto candidates = selectInitialPairCandidates(_sfmOptions.maxInitPairCandidates);
        if (candidates.empty())
        {
            result.summary = "Failed to find a suitable initial image pair";
            return result;
        }
        for (size_t ci = 0; ci < candidates.size(); ++ci)
        {
            initId1 = candidates[ci].first;
            initId2 = candidates[ci].second;
            Logger::instance()->infof("[SFM] Trying init pair candidate %zu/%zu: (%u, %u)", ci + 1,
                                      candidates.size(), initId1, initId2);
            if (initializeFromPair(initId1, initId2))
            {
                initOk = true;
                break;
            }
            Logger::instance()->warnf("[SFM] Candidate (%u, %u) failed: %s", initId1, initId2,
                                      _lastErrorMessage.c_str());
        }
    }
    else
    {
        initId1 = _sfmOptions.initImageId1;
        initId2 = _sfmOptions.initImageId2;
        initOk = initializeFromPair(initId1, initId2);
    }

    if (!initOk)
    {
        result.summary = "Failed to initialize from image pair: " + _lastErrorMessage;
        return result;
    }

    if (!reportProgress(2, totalImages, "Initialized from pair", progressCb))
        return result;

    rebuildVisibilityCache();

    // ---- 步骤 2：逐帧注册循环 ----
    int regCount = static_cast<int>(_reconstruction->numRegisteredImages());
    int iterSinceLastLocalBA = 0;
    int iterSinceLastGlobalBA = 0;

    while (regCount < totalImages && !_isAborted)
    {
        ImageId nextId = selectNextImage();
        if (nextId == kInvalidImageId)
        {
            Logger::instance()->warnf(
                "[SFM] Registration stopped: no selectable next image (registered=%d/%d, points=%zu, "
                "permanentlyFailed=%zu, minPnPInliers=%d)",
                regCount,
                totalImages,
                _reconstruction->numPoints3D(),
                _permanentlyFailedImages.size(),
                _sfmOptions.pnpOptions.minNumInliers);
            break; // 无更多可注册图像
        }

        if (!registerImage(nextId))
        {
            int &failCount = _registerFailCount[nextId];
            ++failCount;
            Logger::instance()->warnf("[SFM] Image %u registration failed: %s",
                                      nextId,
                                      _lastErrorMessage.empty() ? "unknown error" : _lastErrorMessage.c_str());
            // 区分暂时失败（无足够 3D-2D 点）和永久失败（相机文件缺失等）
            // 超过 5 次仍无法注册视为永久失败，不再尝试
            const int kMaxRetries = 5;
            if (failCount >= kMaxRetries)
            {
                _permanentlyFailedImages.insert(nextId);
                Logger::instance()->warnf(
                    "[SFM] Image %u permanently skipped after %d failed registration attempts",
                    nextId, failCount);
            }
            else
            {
                Logger::instance()->infof(
                    "[SFM] Image %u registration attempt %d/%d failed, will retry later",
                    nextId, failCount, kMaxRetries);
            }
            continue;
        }
        // 注册成功，清除失败计数
        _registerFailCount.erase(nextId);

        regCount = static_cast<int>(_reconstruction->numRegisteredImages());

        // 三角化新注册图像
        std::vector<Point3DId> previousPointIds = _reconstruction->image(nextId).point3DIds;
        Triangulator triangulator(*_reconstruction, _correspondenceGraph);
        triangulator.triangulateImage(nextId, _sfmOptions.triangulatorOptions);
        updateVisibilityCacheForImage(nextId, previousPointIds);

        ++iterSinceLastLocalBA;
        ++iterSinceLastGlobalBA;

        // 局部 BA
        if (iterSinceLastLocalBA >= _sfmOptions.localBAInterval)
        {
            runBundleAdjust(true, {nextId});
            triangulator.filterPoints(_sfmOptions.filterMaxReprojError, _sfmOptions.filterMinTriAngle);
            invalidateVisibilityCache();
            iterSinceLastLocalBA = 0;
        }

        // 全局 BA（使用迭代精化策略）
        if (iterSinceLastGlobalBA >= _sfmOptions.globalBAInterval)
        {
            iterativeGlobalBA();
            invalidateVisibilityCache();
            iterSinceLastGlobalBA = 0;
        }

        std::ostringstream msg;
        msg << "Registered " << regCount << "/" << totalImages << " images, " << _reconstruction->numPoints3D()
            << " 3D points";

        if (!reportProgress(regCount, totalImages, msg.str(), progressCb))
            return result;
    }

    // ---- 步骤 3：最终全局 BA 和清理（迭代精化） ----
    if (!_isAborted && _reconstruction->numRegisteredImages() >= 2)
    {
        iterativeGlobalBA();

        // 过滤轨迹长度过短的不可靠三维点
        Triangulator finalTri(*_reconstruction, _correspondenceGraph);
        if (_sfmOptions.filterMinTrackLen > 1)
        {
            int nShort = finalTri.filterShortTracks(_sfmOptions.filterMinTrackLen);
            Logger::instance()->infof("[SFM] Filtered %d short-track points (minTrackLen=%d)", nShort,
                                      _sfmOptions.filterMinTrackLen);
        }
    }

    // ---- 步骤 4：用最新相机位姿重算重投影误差（确保统计精确） ----
    if (!_isAborted && _reconstruction->numRegisteredImages() >= 2)
    {
        Triangulator finalReprojTri(*_reconstruction, _correspondenceGraph);
        finalReprojTri.recomputeReprojErrors();
    }

    // ---- 步骤 5：组装结果 ----
    result.success = _reconstruction->numRegisteredImages() >= 2;
    result.numRegisteredImages = static_cast<int>(_reconstruction->numRegisteredImages());
    result.numPoints3D = static_cast<int>(_reconstruction->numPoints3D());
    result.meanReprojError = _reconstruction->meanReprojError();
    result.reconstruction = _reconstruction;
    result.summary = _reconstruction->summary();

    if (!_permanentlyFailedImages.empty())
    {
        Logger::instance()->warnf("[SFM] %zu image(s) permanently failed to register",
                                  _permanentlyFailedImages.size());
    }

    // 填充最后一轮全局 BA 统计
    result.baRmsBefore = _lastGlobalBARmsBefore;
    result.baRmsAfter = _lastGlobalBARmsAfter;
    result.baTracksTotal = _lastGlobalBATracksTotal;
    result.baTracksOptimized = _lastGlobalBATracksOptimized;
    result.baTracksFiltered = _lastGlobalBATracksFiltered;

    return result;
}

IncrementalSfmResult IncrementalSfm::runKnownCameraPoseReconstruction(SfmProgressCallback progressCb)
{
    IncrementalSfmResult result;
    const int totalImages = static_cast<int>(_reconstruction->numImages());

    auto imageIds = _reconstruction->allImageIds();
    std::sort(imageIds.begin(), imageIds.end());

    int registeredCount = 0;
    for (ImageId imageId : imageIds)
    {
        Camera camera;
        if (!getCamera(imageId, camera))
        {
            result.summary = "Failed to load known camera pose for image " + std::to_string(imageId);
            return result;
        }

        _reconstruction->registerImage(imageId, camera);
        ++registeredCount;

        if (!reportProgress(registeredCount, totalImages, "Registered known camera pose", progressCb))
        {
            result.summary = "Known camera pose reconstruction aborted";
            return result;
        }
    }

    const auto triangulationPolicy = resolveKnownPoseTriangulationPolicy(_reconstruction,
                                                                         _correspondenceGraph,
                                                                         imageIds,
                                                                         _sfmOptions);
    if (triangulationPolicy.adapted)
    {
        Logger::instance()->infof(
            "[SFM] Known-pose narrow-baseline adaptation: minTriAngle %.3f -> %.3f deg "
            "(valid=%d defaultAccepted=%d adaptedAccepted=%d)",
            _sfmOptions.triangulatorOptions.minTriAngle,
            triangulationPolicy.chosenMinTriAngle,
            triangulationPolicy.validCandidates,
            triangulationPolicy.acceptedWithDefault,
            triangulationPolicy.acceptedWithAdapted);
    }

    Triangulator triangulator(*_reconstruction, _correspondenceGraph);
    int createdPoints = 0;
    int continuedObservations = 0;
    int completedObservations = 0;
    int inputMultiViewTrackCount = 0;
    int createdLongTrackCount = 0;
    int longTrackTwoViewOnlyCount = 0;

    MultiViewTrackBuilder trackBuilder;
    int indexedPairCount = 0;
    int indexedMatchCount = 0;
    int rawIndexedMatchCount = 0;
    int geometryRejectedMatchCount = 0;
    for (ImageId imageId : imageIds)
    {
        const std::vector<ImageId> connected = _correspondenceGraph.connectedImages(imageId);
        for (ImageId otherImageId : connected)
        {
            if (otherImageId <= imageId)
            {
                continue;
            }

            const auto &matches = _correspondenceGraph.matchesBetween(imageId, otherImageId);
            if (matches.empty())
            {
                continue;
            }

            std::vector<MultiViewTrackBuilder::MatchIndexPair> indexedMatches;
            indexedMatches.reserve(matches.size());
            for (const FeatureMatch &match : matches)
            {
                if (match.idx1 == kInvalidFeatureIdx || match.idx2 == kInvalidFeatureIdx)
                {
                    continue;
                }
                ++rawIndexedMatchCount;
                if (!knownPoseMatchPassesGeometry(*_reconstruction,
                                                  imageId,
                                                  otherImageId,
                                                  match,
                                                  triangulationPolicy.triangulatorOptions))
                {
                    ++geometryRejectedMatchCount;
                    continue;
                }
                indexedMatches.emplace_back(match.idx1, match.idx2, match.score);
            }

            if (!indexedMatches.empty())
            {
                trackBuilder.addMatchPair(imageId, otherImageId, indexedMatches);
                ++indexedPairCount;
                indexedMatchCount += static_cast<int>(indexedMatches.size());
            }
        }
    }

    const MultiViewTrackBuildResult multiViewTracks = trackBuilder.build();
    if (!multiViewTracks.tracks.empty())
    {
        std::ostringstream trackLengthHistogram;
        int emittedTrackLengthBins = 0;
        for (const auto &entry : multiViewTracks.trackLengthHistogram)
        {
            if (emittedTrackLengthBins > 0)
            {
                trackLengthHistogram << ",";
            }
            trackLengthHistogram << entry.first << ":" << entry.second;
            ++emittedTrackLengthBins;
            if (emittedTrackLengthBins >= 8)
            {
                break;
            }
        }

        const auto trackStats = triangulator.triangulateTracks(multiViewTracks.tracks,
                                                               triangulationPolicy.triangulatorOptions);
        createdPoints += trackStats.numCreated;
        continuedObservations += trackStats.numContinued;
        inputMultiViewTrackCount = trackStats.inputLongTracks;
        createdLongTrackCount = trackStats.createdLongTracks;
        longTrackTwoViewOnlyCount = trackStats.longTrackTwoViewOnly;
        Logger::instance()->infof(
            "[SFM] Known-pose multiview tracks: pairs=%d matches=%d rawMatches=%d "
            "geometryRejected=%d components=%d accepted=%d rejectedConflict=%d "
            "rejectedConflictEdges=%d hist=%s created=%d addedObservations=%d",
            indexedPairCount,
            indexedMatchCount,
            rawIndexedMatchCount,
            geometryRejectedMatchCount,
            multiViewTracks.totalComponents,
            multiViewTracks.acceptedComponents,
            multiViewTracks.rejectedConflictComponents,
            multiViewTracks.rejectedConflictEdges,
            trackLengthHistogram.str().c_str(),
            trackStats.numCreated,
            trackStats.numContinued);
        Logger::instance()->infof(
            "[SFM] Known-pose triangulation stats: inputTracks=%d inputLong=%d unusable=%d "
            "noCandidate=%d createdTwoView=%d createdLong=%d seedTests=%d seedRejected=%d "
            "depthRejected=%d reprojRejected=%d longTwoView=%d rejectedExtraSamples=%d "
            "rejectedExtraAvg=%.3f rejectedExtraMax=%.3f rejectedExtraHist=<=5:%d,<=10:%d,<=25:%d,>25:%d",
            trackStats.inputTracks,
            trackStats.inputLongTracks,
            trackStats.unusableTracks,
            trackStats.noCandidateTracks,
            trackStats.createdTwoViewTracks,
            trackStats.createdLongTracks,
            trackStats.seedPairTests,
            trackStats.seedPairRejected,
            trackStats.depthObservationRejected,
            trackStats.reprojObservationRejected,
            trackStats.longTrackTwoViewOnly,
            trackStats.longTrackRejectedExtraSamples,
            trackStats.longTrackRejectedExtraSamples > 0
                ? trackStats.longTrackRejectedExtraErrorSum /
                      static_cast<double>(trackStats.longTrackRejectedExtraSamples)
                : 0.0,
            trackStats.longTrackRejectedExtraErrorMax,
            trackStats.longTrackRejectedExtraLe5,
            trackStats.longTrackRejectedExtraLe10,
            trackStats.longTrackRejectedExtraLe25,
            trackStats.longTrackRejectedExtraGt25);
    }

    if (createdPoints == 0)
    {
        Logger::instance()->warnf(
            "[SFM] Known-pose multiview triangulation produced no points; falling back to pairwise triangulation");
        for (ImageId imageId : imageIds)
        {
            const auto stats = triangulator.triangulateImage(imageId, triangulationPolicy.triangulatorOptions);
            createdPoints += stats.numCreated;
            continuedObservations += stats.numContinued;
        }
    }

    completedObservations = triangulator.completeTracks(triangulationPolicy.triangulatorOptions);
    triangulator.retriangulatePoints(triangulationPolicy.triangulatorOptions.maxReprojError);
    triangulator.recomputeReprojErrors();
    const int filteredPoints = triangulator.filterPoints(_sfmOptions.filterMaxReprojError,
                                                         triangulationPolicy.filterMinTriAngle);
    const int shortTrackFiltered = triangulator.filterShortTracks(_sfmOptions.filterMinTrackLen);
    triangulator.recomputeReprojErrors();

    int baRetriangulated = 0;
    int baCompletedObservations = 0;
    int baFilteredPoints = 0;
    int baShortTrackFiltered = 0;
    if (_reconstruction->numRegisteredImages() >= 2 && _reconstruction->numPoints3D() > 0)
    {
        Logger::instance()->infof("[SFM] Known-pose global BA: tracks=%zu cameras=%zu",
                                  _reconstruction->numPoints3D(),
                                  _reconstruction->numRegisteredImages());
        if (!reportProgress(totalImages, totalImages, "Running known-pose bundle adjustment...", progressCb))
        {
            result.summary = "Known camera pose reconstruction aborted before bundle adjustment";
            return result;
        }

        refineKnownCameraPosesWithPnp();
        runBundleAdjust(false);

        Triangulator baTriangulator(*_reconstruction, _correspondenceGraph);
        baRetriangulated = baTriangulator.retriangulatePoints(triangulationPolicy.triangulatorOptions.maxReprojError);
        baCompletedObservations = baTriangulator.completeTracks(triangulationPolicy.triangulatorOptions);
        baFilteredPoints = baTriangulator.filterPoints(_sfmOptions.filterMaxReprojError,
                                                       triangulationPolicy.filterMinTriAngle);
        baShortTrackFiltered = baTriangulator.filterShortTracks(_sfmOptions.filterMinTrackLen);
        baTriangulator.recomputeReprojErrors();
    }

    result.numRegisteredImages = static_cast<int>(_reconstruction->numRegisteredImages());
    result.numPoints3D = static_cast<int>(_reconstruction->numPoints3D());
    result.meanReprojError = _reconstruction->meanReprojError();
    result.reconstruction = _reconstruction;
    result.summary = _reconstruction->summary();
    result.baRmsBefore = _lastGlobalBARmsBefore;
    result.baRmsAfter = _lastGlobalBARmsAfter;
    result.baTracksTotal = _lastGlobalBATracksTotal;
    result.baTracksOptimized = _lastGlobalBATracksOptimized;
    result.baTracksFiltered = _lastGlobalBATracksFiltered;

    int finalLongTrackCount = 0;
    int finalTwoViewTrackCount = 0;
    for (Point3DId pointId : _reconstruction->allPoint3DIds())
    {
        if (!_reconstruction->hasPoint3D(pointId))
        {
            continue;
        }

        const size_t trackLength = _reconstruction->point3D(pointId).track.length();
        if (trackLength >= 3)
        {
            ++finalLongTrackCount;
        }
        else if (trackLength == 2)
        {
            ++finalTwoViewTrackCount;
        }
    }

    const int finalTrackCount = finalLongTrackCount + finalTwoViewTrackCount;
    const double finalTwoViewTrackRatio = finalTrackCount > 0
        ? static_cast<double>(finalTwoViewTrackCount) / static_cast<double>(finalTrackCount)
        : 0.0;
    const bool hasEnoughMultiViewInputForQualityGate =
        inputMultiViewTrackCount >= kKnownPoseMinLongInputTracksForQualityGate &&
        result.numPoints3D >= kKnownPoseMinPointsForTrackRatioGate;
    const bool hasOnlyTwoViewOutputFromMultiViewInput =
        result.numRegisteredImages >= 3 &&
        inputMultiViewTrackCount > 0 &&
        result.numPoints3D > 0 &&
        finalLongTrackCount == 0;
    const bool hasTooLittleMultiViewOutputFromMultiViewInput =
        result.numRegisteredImages >= 3 &&
        hasEnoughMultiViewInputForQualityGate &&
        finalTwoViewTrackRatio >= kKnownPoseMaxTwoViewTrackRatio;
    result.success = result.numRegisteredImages >= 2 &&
        result.numPoints3D > 0 &&
        !hasOnlyTwoViewOutputFromMultiViewInput &&
        !hasTooLittleMultiViewOutputFromMultiViewInput;

    Logger::instance()->infof(
        "[SFM] Known-pose reconstruction: registered=%d/%d created=%d continued=%d completed=%d "
        "filtered=%d shortTrackFiltered=%d baTracks=%d baOptimized=%d baRms=%.4f->%.4f "
        "baRetriangulated=%d baCompleted=%d baFiltered=%d baShortTrackFiltered=%d points=%d "
        "finalLongTracks=%d finalTwoViewTracks=%d finalTwoViewRatio=%.4f",
        result.numRegisteredImages,
        totalImages,
        createdPoints,
        continuedObservations,
        completedObservations,
        filteredPoints,
        shortTrackFiltered,
        result.baTracksTotal,
        result.baTracksOptimized,
        result.baRmsBefore,
        result.baRmsAfter,
        baRetriangulated,
        baCompletedObservations,
        baFilteredPoints,
        baShortTrackFiltered,
        result.numPoints3D,
        finalLongTrackCount,
        finalTwoViewTrackCount,
        finalTwoViewTrackRatio);

    if (hasOnlyTwoViewOutputFromMultiViewInput)
    {
        Logger::instance()->warnf(
            "[SFM] Known-pose reconstruction rejected all-two-view output: inputLongTracks=%d "
            "createdLongTracks=%d longTrackTwoViewOnly=%d finalTwoViewTracks=%d",
            inputMultiViewTrackCount,
            createdLongTrackCount,
            longTrackTwoViewOnlyCount,
            finalTwoViewTrackCount);
        result.summary =
            "Known camera pose reconstruction produced only two-view tracks despite multi-view matches; "
            "check camera poses, match consistency, or rerun SfM/BA with adjusted cameras.";
    }
    else if (hasTooLittleMultiViewOutputFromMultiViewInput)
    {
        Logger::instance()->warnf(
            "[SFM] Known-pose reconstruction rejected weak multi-view output: inputLongTracks=%d "
            "createdLongTracks=%d longTrackTwoViewOnly=%d finalLongTracks=%d finalTwoViewTracks=%d "
            "finalTwoViewRatio=%.4f threshold=%.4f",
            inputMultiViewTrackCount,
            createdLongTrackCount,
            longTrackTwoViewOnlyCount,
            finalLongTrackCount,
            finalTwoViewTrackCount,
            finalTwoViewTrackRatio,
            kKnownPoseMaxTwoViewTrackRatio);
        result.summary =
            "Known camera pose reconstruction produced mostly two-view tracks despite multi-view matches; "
            "check camera poses, match consistency, or rerun SfM/BA with adjusted cameras.";
    }

    if (!result.success)
    {
        if (result.summary.empty())
        {
            result.summary = "Known camera pose reconstruction produced insufficient sparse points";
        }
    }

    return result;
}

// ============================================================
// 内部：加载相机
// ============================================================

bool IncrementalSfm::loadCamera(const std::string &cameraPath, Camera &cam) const
{
    return cam.loadFromFile(cameraPath);
}

bool IncrementalSfm::getCamera(ImageId imageId, Camera &cam) const
{
    // 优先使用预设相机对象
    auto pit = _preloadedCameras.find(imageId);
    if (pit != _preloadedCameras.end())
    {
        cam = pit->second;
        return true;
    }
    // 其次从 .tsai 文件加载
    auto cit = _cameraPaths.find(imageId);
    if (cit != _cameraPaths.end() && !cit->second.empty())
    {
        return loadCamera(cit->second, cam);
    }
    return false;
}

std::vector<BACameraPosePrior> IncrementalSfm::buildCameraPosePriorsFromInputCameras(
    const std::vector<ImageId> &imageIds) const
{
    std::vector<Camera> inputCameras;
    inputCameras.reserve(imageIds.size());
    std::vector<std::array<double, 3>> inputCenters;
    inputCenters.reserve(imageIds.size());
    for (ImageId imageId : imageIds)
    {
        Camera inputCamera;
        if (getCamera(imageId, inputCamera))
        {
            inputCenters.push_back(inputCamera.cameraCenter());
        }
        inputCameras.push_back(inputCamera);
    }

    const double inputExtent = centerExtent(inputCenters);
    const double adaptivePositionSigmaMeters = std::max(0.25, inputExtent * 0.01);

    std::vector<BACameraPosePrior> priors;
    priors.reserve(imageIds.size());
    for (size_t i = 0; i < imageIds.size(); ++i)
    {
        BACameraPosePrior prior;
        const Camera &inputCamera = inputCameras[i];
        if (inputCamera.isValid())
        {
            prior.enabled = true;
            prior.cameraToWorldRotation = inputCamera.cameraToWorldRotation();
            prior.cameraCenter = inputCamera.cameraCenter();
            prior.positionSigmaMeters = std::max(1e-6, adaptivePositionSigmaMeters);
            prior.rotationSigmaDegrees = std::max(1e-6, prior.rotationSigmaDegrees);
        }
        priors.push_back(prior);
    }
    return priors;
}

void IncrementalSfm::alignReconstructionToKnownPosePriors(const std::vector<ImageId> &imageIds,
                                                          std::vector<Camera> *baCameras)
{
    if (!baCameras || imageIds.size() != baCameras->size() || imageIds.size() < 3)
    {
        return;
    }

    std::vector<std::array<double, 3>> currentCenters;
    std::vector<std::array<double, 3>> inputCenters;
    currentCenters.reserve(imageIds.size());
    inputCenters.reserve(imageIds.size());
    for (size_t i = 0; i < imageIds.size(); ++i)
    {
        Camera inputCamera;
        if (!getCamera(imageIds[i], inputCamera))
        {
            continue;
        }
        currentCenters.push_back((*baCameras)[i].cameraCenter());
        inputCenters.push_back(inputCamera.cameraCenter());
    }

    if (currentCenters.size() < 3)
    {
        return;
    }

    const SimilarityTransform3d transform = estimateRobustCameraCenterSimilarity(currentCenters, inputCenters);
    if (!transform.valid || transform.inlierCount < 3)
    {
        Logger::instance()->warnf("[SFM] Known-pose Sim3 alignment skipped: insufficient robust camera-center inliers");
        return;
    }

    for (ImageId imageId : _reconstruction->registeredImageIds())
    {
        if (!_reconstruction->hasCamera(imageId))
        {
            continue;
        }
        Camera &camera = _reconstruction->camera(imageId);
        camera.setPose(multiplyRotation(transform.rotation, camera.cameraToWorldRotation()),
                       transformPoint(transform, camera.cameraCenter()));
    }

    for (Point3DId pointId : _reconstruction->allPoint3DIds())
    {
        if (!_reconstruction->hasPoint3D(pointId))
        {
            continue;
        }
        _reconstruction->point3D(pointId).xyz =
            transformPoint(transform, _reconstruction->point3D(pointId).xyz);
    }

    for (size_t i = 0; i < imageIds.size(); ++i)
    {
        if (_reconstruction->hasCamera(imageIds[i]))
        {
            (*baCameras)[i] = _reconstruction->camera(imageIds[i]);
        }
    }

    Logger::instance()->infof(
        "[SFM] Known-pose Sim3/RANSAC alignment before BA: cameras=%zu inliers=%d scale=%.9f rmse=%.6f",
        currentCenters.size(),
        transform.inlierCount,
        transform.scale,
        transform.rmse);
}

void IncrementalSfm::refineKnownCameraPosesWithPnp()
{
    if (!_sfmOptions.useKnownCameraPoses || !_reconstruction)
    {
        return;
    }

    const std::vector<ImageId> imageIds = _reconstruction->registeredImageIds();
    for (ImageId imageId : imageIds)
    {
        if (!_reconstruction->hasCamera(imageId) || !_reconstruction->hasImage(imageId))
        {
            continue;
        }

        std::vector<std::array<double, 3>> worldPoints;
        std::vector<std::array<double, 2>> imagePoints;
        const ImageData &image = _reconstruction->image(imageId);
        for (Point3DId pointId : _reconstruction->allPoint3DIds())
        {
            if (!_reconstruction->hasPoint3D(pointId))
            {
                continue;
            }

            const ScenePoint3D &point = _reconstruction->point3D(pointId);
            for (const TrackElement &element : point.track.elements)
            {
                if (element.imageId != imageId || element.featureIdx >= image.keypoints.size())
                {
                    continue;
                }

                const FeatureKeypoint &keypoint = image.keypoints[element.featureIdx];
                worldPoints.push_back(point.xyz);
                imagePoints.push_back({{static_cast<double>(keypoint.x), static_cast<double>(keypoint.y)}});
                break;
            }
        }

        if (static_cast<int>(worldPoints.size()) < std::max(6, _sfmOptions.pnpOptions.minNumInliers))
        {
            continue;
        }

        PnpOptions pnpOptions = _sfmOptions.pnpOptions;
        pnpOptions.maxReprojError = std::max(pnpOptions.maxReprojError, _sfmOptions.filterMaxReprojError * 2.0);
        pnpOptions.minNumInliers = std::min(pnpOptions.minNumInliers, static_cast<int>(worldPoints.size()));
        pnpOptions.minInlierRatio = std::min(pnpOptions.minInlierRatio, 0.10);

        const Camera before = _reconstruction->camera(imageId);
        const PnpResult pnp = PnpSolver::solveWithCamera(worldPoints, imagePoints, before, pnpOptions);
        if (!pnp.success)
        {
            continue;
        }

        Camera inputCamera;
        const double inputExtent = [&]()
        {
            std::vector<std::array<double, 3>> centers;
            for (ImageId id : imageIds)
            {
                Camera cam;
                if (getCamera(id, cam))
                {
                    centers.push_back(cam.cameraCenter());
                }
            }
            return centerExtent(centers);
        }();
        const double maxAcceptedMove = std::max(0.5, inputExtent * 0.05);
        Camera candidate = before;
        candidate.setPose(pnp.R, pnp.C);
        if (getCamera(imageId, inputCamera) &&
            pointDistance(candidate.cameraCenter(), inputCamera.cameraCenter()) > maxAcceptedMove)
        {
            Logger::instance()->warnf(
                "[SFM] Known-pose PnP refinement rejected for image %u: move from prior %.3f > %.3f",
                imageId,
                pointDistance(candidate.cameraCenter(), inputCamera.cameraCenter()),
                maxAcceptedMove);
            continue;
        }

        _reconstruction->camera(imageId) = candidate;
        Logger::instance()->infof("[SFM] Known-pose PnP refinement accepted: image=%u inliers=%d/%zu",
                                  imageId,
                                  pnp.numInliers,
                                  worldPoints.size());
    }
}

// ============================================================
// 内部：选择初始像对
// ============================================================

std::vector<std::pair<ImageId, ImageId>> IncrementalSfm::selectInitialPairCandidates(int maxCandidates) const
{
    // 收集所有满足最少匹配数的像对，按匹配数降序排序
    struct PairInfo
    {
        ImageId id1;
        ImageId id2;
        size_t numMatches;
    };
    std::vector<PairInfo> pairs;

    auto allIds = _reconstruction->allImageIds();
    for (size_t i = 0; i < allIds.size(); ++i)
    {
        for (size_t j = i + 1; j < allIds.size(); ++j)
        {
            size_t nm = _correspondenceGraph.numMatchesBetween(allIds[i], allIds[j]);
            if (nm >= static_cast<size_t>(_sfmOptions.initMinNumMatches))
            {
                pairs.push_back({allIds[i], allIds[j], nm});
            }
        }
    }

    // 按匹配数降序排序
    std::sort(pairs.begin(), pairs.end(),
              [](const PairInfo &a, const PairInfo &b) { return a.numMatches > b.numMatches; });

    // COLMAP 式退化对过滤：计算 H_inlier / F_inlier 比值，
    // 比值过高说明像对接近纯旋转或平面场景，对极几何退化，提前拒绝。
    const double maxHFRatio = 0.8;
    const double ransacThresh = 3.0;

    std::vector<std::pair<ImageId, ImageId>> result;
    result.reserve(maxCandidates);

    for (const auto &pair : pairs)
    {
        if (static_cast<int>(result.size()) >= maxCandidates)
        {
            break;
        }

        const auto &matches = _correspondenceGraph.matchesBetween(pair.id1, pair.id2);
        const ImageData &img1 = _reconstruction->image(pair.id1);
        const ImageData &img2 = _reconstruction->image(pair.id2);
        const bool swapped = (pair.id1 > pair.id2);

        std::vector<cv::Point2d> pts1, pts2;
        pts1.reserve(matches.size());
        pts2.reserve(matches.size());
        for (const auto &m : matches)
        {
            FeatureIdx fi1 = swapped ? m.idx2 : m.idx1;
            FeatureIdx fi2 = swapped ? m.idx1 : m.idx2;
            if (fi1 >= img1.keypoints.size() || fi2 >= img2.keypoints.size())
            {
                continue;
            }
            pts1.emplace_back(img1.keypoints[fi1].x, img1.keypoints[fi1].y);
            pts2.emplace_back(img2.keypoints[fi2].x, img2.keypoints[fi2].y);
        }

        if (static_cast<int>(pts1.size()) < _sfmOptions.initMinNumMatches)
        {
            continue;
        }

        cv::Mat maskF, maskH;
        cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, ransacThresh, 0.999, maskF);
        cv::findHomography(pts1, pts2, cv::RANSAC, ransacThresh, maskH);

        int fInliers = maskF.empty() ? 0 : cv::countNonZero(maskF);
        int hInliers = maskH.empty() ? 0 : cv::countNonZero(maskH);

        double hfRatio = (fInliers > 0) ? static_cast<double>(hInliers) / fInliers : 1.0;

        Logger::instance()->debugf("[SFM] Pair (%u, %u): F_inliers=%d, H_inliers=%d, H/F=%.3f",
                                   pair.id1, pair.id2, fInliers, hInliers, hfRatio);

        if (hfRatio > maxHFRatio)
        {
            Logger::instance()->infof("[SFM] Rejected degenerate pair (%u, %u): H/F=%.3f > %.2f",
                                      pair.id1, pair.id2, hfRatio, maxHFRatio);
            continue;
        }

        result.emplace_back(pair.id1, pair.id2);
    }

    Logger::instance()->infof("[SFM] selectInitialPairCandidates: %d candidates from %zu pairs",
                              static_cast<int>(result.size()), pairs.size());
    return result;
}

// ============================================================
// 内部：初始化初始像对
// ============================================================

bool IncrementalSfm::initializeFromPair(ImageId id1, ImageId id2)
{
    // 加载两台相机内参
    Camera cam1, cam2;
    if (!getCamera(id1, cam1))
    {
        _lastErrorMessage = "getCamera(" + std::to_string(id1) + ") failed";
        return false;
    }
    if (!getCamera(id2, cam2))
    {
        _lastErrorMessage = "getCamera(" + std::to_string(id2) + ") failed";
        return false;
    }

        Logger::instance()->infof("[SFM] initializeFromPair: id1=%d, id2=%d", id1, id2);
        Logger::instance()->debugf("[SFM] cam1: fu=%.4f fv=%.4f cu=%.4f cv=%.4f uDir=%d vDir=%d", cam1.focalX(),
                       cam1.focalY(), cam1.principalX(), cam1.principalY(), cam1.uAxisSign(),
                       cam1.vAxisSign());
        Logger::instance()->debugf("[SFM] cam2: fu=%.4f fv=%.4f cu=%.4f cv=%.4f uDir=%d vDir=%d", cam2.focalX(),
                       cam2.focalY(), cam2.principalX(), cam2.principalY(), cam2.uAxisSign(),
                       cam2.vAxisSign());

    // ---- 收集匹配特征点 ----
    const auto &matches = _correspondenceGraph.matchesBetween(id1, id2);
    const ImageData &img1 = _reconstruction->image(id1);
    const ImageData &img2 = _reconstruction->image(id2);
    const bool swapped = (id1 > id2);

    std::vector<cv::Point2d> pts1, pts2;
    std::vector<size_t> matchIndices;
    for (size_t mi = 0; mi < matches.size(); ++mi)
    {
        const auto &m = matches[mi];
        FeatureIdx fi1 = swapped ? m.idx2 : m.idx1;
        FeatureIdx fi2 = swapped ? m.idx1 : m.idx2;
        if (fi1 >= img1.keypoints.size() || fi2 >= img2.keypoints.size())
        {
            continue;
        }
        pts1.emplace_back(img1.keypoints[fi1].x, img1.keypoints[fi1].y);
        pts2.emplace_back(img2.keypoints[fi2].x, img2.keypoints[fi2].y);
        matchIndices.push_back(mi);
    }

    const int nPts = static_cast<int>(pts1.size());
    Logger::instance()->infof("[SFM] valid point pairs: %d (matches=%zu)", nPts, matches.size());

    // 使用 chirality 专用阈值而非 initMinNumInliers
    const int chiralityThreshold = std::max(5, _sfmOptions.initMinChiralityInliers);

    if (nPts < chiralityThreshold)
    {
        _lastErrorMessage =
            "valid_points=" + std::to_string(nPts) + " < chiralityThreshold=" + std::to_string(chiralityThreshold);
        return false;
    }

    // ---- 构造内参矩阵 K1, K2（分别使用两台相机的内参）----
    // 当 depthAxisFlipped 时，额外翻转 fx/fy 符号，使归一化坐标处于正深度约定
    // 这样 OpenCV 的 recoverPose / decomposeHomography 的 chirality 检查 (Z>0) 才正确
    const bool depthFlipped = cam1.depthAxisFlipped();
    const double depthSign = depthFlipped ? -1.0 : 1.0;
    const double fx1 = depthSign * (cam1.uAxisSign() < 0 ? -1.0 : 1.0) * cam1.focalX();
    const double fy1 = depthSign * (cam1.vAxisSign() < 0 ? -1.0 : 1.0) * cam1.focalY();
    const double cx1 = cam1.principalX();
    const double cy1 = cam1.principalY();
    cv::Mat K1 = (cv::Mat_<double>(3, 3) << fx1, 0.0, cx1, 0.0, fy1, cy1, 0.0, 0.0, 1.0);

    const double fx2 = depthSign * (cam2.uAxisSign() < 0 ? -1.0 : 1.0) * cam2.focalX();
    const double fy2 = depthSign * (cam2.vAxisSign() < 0 ? -1.0 : 1.0) * cam2.focalY();
    const double cx2 = cam2.principalX();
    const double cy2 = cam2.principalY();
    cv::Mat K2 = (cv::Mat_<double>(3, 3) << fx2, 0.0, cx2, 0.0, fy2, cy2, 0.0, 0.0, 1.0);

    // 当两台相机内参相同时直接用 K1；否则先归一化到 K1 坐标系
    // 对于 findEssentialMat 使用 K1（假设近似相同或归一化后处理）
    // 注：实际行星影像通常同一台相机，内参完全相同
    cv::Mat K = K1;

    // ---- COLMAP 式同时估计 E 和 H ----
    const double ransacThresh = 1.0; // 像素
    cv::Mat maskE, maskH;

    cv::Mat E = xjw::opencv_compat::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, ransacThresh, maskE);
    int E_inliers = cv::countNonZero(maskE);

    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, ransacThresh, maskH);
    int H_inliers = cv::countNonZero(maskH);

    Logger::instance()->infof("[SFM] E_inliers=%d, H_inliers=%d, ratio=%.3f", E_inliers, H_inliers,
                              E_inliers > 0 ? static_cast<double>(H_inliers) / E_inliers : 999.0);

    // ─── 同时尝试 E 和 H 两条路径，选最优（参考 COLMAP） ───
    cv::Mat R_E, t_E, R_H, t_H;
    int chirality_E = 0, chirality_H = 0;

    // ── E 路径 ──
    if (E_inliers >= chiralityThreshold && !E.empty())
    {
        chirality_E = cv::recoverPose(E, pts1, pts2, K, R_E, t_E, maskE);
        Logger::instance()->infof("[SFM] E path: chirality_inliers=%d", chirality_E);
    }

    // ── H 路径 (Faugeras decomposition → 4 solutions) ──
    if (H_inliers >= chiralityThreshold && !H.empty())
    {
        std::vector<cv::Mat> Rs, ts, normals;
        int nSolutions = cv::decomposeHomographyMat(H, K, Rs, ts, normals);
        (void)nSolutions;

        int bestIdx = -1;
        int bestChirality = 0;
        double bestReprojSum = std::numeric_limits<double>::max();

        for (int si = 0; si < static_cast<int>(Rs.size()); ++si)
        {
            const cv::Mat &Ri = Rs[si];
            const cv::Mat &ti = ts[si];

            int chiralityOk = 0;
            double reprojSum = 0.0;

            for (int pi = 0; pi < nPts; ++pi)
            {
                if (maskH.at<uchar>(pi) == 0)
                {
                    continue;
                }

                cv::Mat x1 = (cv::Mat_<double>(3, 1) << (pts1[pi].x - cx1) / fx1, (pts1[pi].y - cy1) / fy1, 1.0);
                cv::Mat x2 = (cv::Mat_<double>(3, 1) << (pts2[pi].x - cx1) / fx1, (pts2[pi].y - cy1) / fy1, 1.0);

                cv::Mat A(4, 4, CV_64F);
                A.row(0) = x1.at<double>(0) * cv::Mat::eye(3, 4, CV_64F).row(2) - cv::Mat::eye(3, 4, CV_64F).row(0);
                A.row(1) = x1.at<double>(1) * cv::Mat::eye(3, 4, CV_64F).row(2) - cv::Mat::eye(3, 4, CV_64F).row(1);

                cv::Mat P2row(3, 4, CV_64F);
                Ri.copyTo(P2row(cv::Rect(0, 0, 3, 3)));
                ti.copyTo(P2row(cv::Rect(3, 0, 1, 3)));
                A.row(2) = x2.at<double>(0) * P2row.row(2) - P2row.row(0);
                A.row(3) = x2.at<double>(1) * P2row.row(2) - P2row.row(1);

                cv::Mat W, U, Vt;
                cv::SVD::compute(A, W, U, Vt, cv::SVD::FULL_UV);
                cv::Mat X4 = Vt.row(3).t();
                if (std::fabs(X4.at<double>(3)) < 1e-10)
                {
                    continue;
                }
                cv::Mat X3 = X4.rowRange(0, 3) / X4.at<double>(3);

                double z1 = X3.at<double>(2);
                cv::Mat X3_cam2 = Ri * X3 + ti;
                double z2 = X3_cam2.at<double>(2);

                if (z1 > 0 && z2 > 0)
                {
                    ++chiralityOk;
                    double u2_proj = fx1 * X3_cam2.at<double>(0) / z2 + cx1;
                    double v2_proj = fy1 * X3_cam2.at<double>(1) / z2 + cy1;
                    double du = u2_proj - pts2[pi].x;
                    double dv = v2_proj - pts2[pi].y;
                    reprojSum += du * du + dv * dv;
                }
            }

            Logger::instance()->debugf("[SFM] H decomp[%d]: chirality=%d, reprojSum=%.2f", si, chiralityOk,
                                       reprojSum);

            if (chiralityOk > bestChirality || (chiralityOk == bestChirality && reprojSum < bestReprojSum))
            {
                bestIdx = si;
                bestChirality = chiralityOk;
                bestReprojSum = reprojSum;
            }
        }

        if (bestIdx >= 0)
        {
            R_H = Rs[bestIdx].clone();
            t_H = ts[bestIdx].clone();
            chirality_H = bestChirality;
            Logger::instance()->infof("[SFM] H best solution: idx=%d, chirality=%d", bestIdx, bestChirality);
        }
    }

    // ── 选择最优路径（参考 COLMAP：优先 chirality 数量最多的方案） ──
    cv::Mat R21_cv, t21_cv;
    int poseInliers = 0;

    if (chirality_E >= chirality_H && chirality_E >= chiralityThreshold)
    {
        R21_cv = R_E;
        t21_cv = t_E;
        poseInliers = chirality_E;
        Logger::instance()->infof("[SFM] Selected E path (chirality=%d)", chirality_E);
    }
    else if (chirality_H >= chiralityThreshold)
    {
        R21_cv = R_H;
        t21_cv = t_H;
        poseInliers = chirality_H;
        Logger::instance()->infof("[SFM] Selected H path (chirality=%d)", chirality_H);
    }
    else
    {
        _lastErrorMessage = "Both E and H paths failed: E_chirality=" + std::to_string(chirality_E) +
                            " H_chirality=" + std::to_string(chirality_H) +
                            " < threshold=" + std::to_string(chiralityThreshold);
        return false;
    }

    // ---- R21_cv, t21_cv 是 OpenCV 的 world-to-cam2: X_cam2 = R21*X_world + t21 ----
    // 若使用了正深度约定 K（depthFlipped），需先转回物理相机坐标约定：
    //   R_physical = D * R_positive * D,  t_physical = D * t_positive
    //   其中 D = diag(1, 1, -1)
    if (depthFlipped)
    {
        R21_cv.at<double>(0, 2) *= -1;
        R21_cv.at<double>(1, 2) *= -1;
        R21_cv.at<double>(2, 0) *= -1;
        R21_cv.at<double>(2, 1) *= -1;
        t21_cv.at<double>(2) *= -1;
    }

    // 转换到 PlaScan 约定: R_c2w = R21^T, C = -R21^T * t21

    // 相机1: 世界原点
    cam1.setPose({1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0});

    // 相机2: camera-to-world
    std::array<double, 9> R2;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R2[i * 3 + j] = R21_cv.at<double>(j, i); // transpose

    std::array<double, 3> C2;
    for (int i = 0; i < 3; ++i)
    {
        C2[i] = 0;
        for (int j = 0; j < 3; ++j)
        {
            C2[i] -= R21_cv.at<double>(j, i) * t21_cv.at<double>(j);
        }
    }
    cam2.setPose(R2, C2);

    Logger::instance()->infof("[SFM] cam2 pose: C=[%.6f, %.6f, %.6f], poseInliers=%d", C2[0], C2[1], C2[2],
                              poseInliers);

    // ---- 注册两幅图像 ----
    _reconstruction->registerImage(id1, cam1);
    _reconstruction->registerImage(id2, cam2);

    // ---- 三角化初始点云 ----
    {
        int validCount = 0, invalidCount = 0, angleFailCount = 0, reprojFailCount = 0;
        for (size_t mi = 0; mi < matches.size(); ++mi)
        {
            const auto &m = matches[mi];
            FeatureIdx fi1 = swapped ? m.idx2 : m.idx1;
            FeatureIdx fi2 = swapped ? m.idx1 : m.idx2;
            if (fi1 >= img1.keypoints.size() || fi2 >= img2.keypoints.size())
            {
                continue;
            }
            double u1 = img1.keypoints[fi1].x, v1 = img1.keypoints[fi1].y;
            double u2 = img2.keypoints[fi2].x, v2 = img2.keypoints[fi2].y;

            auto triResult = Intersection::intersectPair(cam1, u1, v1, cam2, u2, v2);
            if (triResult.valid)
            {
                if (triResult.angle_deg < _sfmOptions.triangulatorOptions.minTriAngle)
                {
                    ++angleFailCount;
                }
                else if (!std::isfinite(triResult.reproj_error_rms) ||
                         triResult.reproj_error_rms > _sfmOptions.triangulatorOptions.maxReprojError)
                {
                    ++reprojFailCount;
                }
                else
                {
                    ++validCount;
                }
            }
            else
            {
                ++invalidCount;
            }
        }
        Logger::instance()->infof("[SFM] Triangulation preview: valid=%d invalid=%d angleFail=%d reprojFail=%d",
                      validCount, invalidCount, angleFailCount, reprojFailCount);
    }

    Triangulator triangulator(*_reconstruction, _correspondenceGraph);
    auto triStats = triangulator.triangulateImage(id1, _sfmOptions.triangulatorOptions);
    triangulator.triangulateImage(id2, _sfmOptions.triangulatorOptions);

    Logger::instance()->infof("[SFM] After triangulation: numPoints3D=%zu", _reconstruction->numPoints3D());

    // 验证初始三角化质量
    // 降低阈值：使用估算内参时三角化成功率偏低，3 个点足以启动 BA 精化
    if (_reconstruction->numPoints3D() < 3)
    {
        std::ostringstream oss;
        oss << "numPoints3D=" << _reconstruction->numPoints3D() << " < 3 after triangulation";
        _lastErrorMessage = oss.str();
        // 初始化失败，回退
        _reconstruction->deregisterImage(id1);
        _reconstruction->deregisterImage(id2);
        return false;
    }

    // 初始 BA
    runBundleAdjust(false);

    return true;
}

// ============================================================
// 内部：选择下一幅图像
// ============================================================

ImageId IncrementalSfm::selectNextImage() const
{
    // 使用增量维护的可见性缓存
    if (_visibilityCacheDirty)
    {
        const_cast<IncrementalSfm *>(this)->rebuildVisibilityCache();
    }

    ImageId bestId = kInvalidImageId;
    size_t bestVisible = 0;

    for (const auto &[id, count] : _visibilityCache)
    {
        if (_reconstruction->isRegistered(id))
            continue;
        if (_permanentlyFailedImages.count(id))
            continue;
        if (count > bestVisible)
        {
            bestVisible = count;
            bestId = id;
        }
    }

    // 至少需要若干可见三维点才值得注册
    if (bestVisible < static_cast<size_t>(_sfmOptions.pnpOptions.minNumInliers))
    {
        Logger::instance()->warnf(
            "[SFM] selectNextImage rejected all candidates: bestId=%u, bestVisible=%zu, minPnPInliers=%d, "
            "registered=%zu/%zu, points=%zu, permanentlyFailed=%zu",
            bestId,
            bestVisible,
            _sfmOptions.pnpOptions.minNumInliers,
            _reconstruction->numRegisteredImages(),
            _reconstruction->numImages(),
            _reconstruction->numPoints3D(),
            _permanentlyFailedImages.size());
        return kInvalidImageId;
    }

    Logger::instance()->infof("[SFM] selectNextImage: id=%u, visible3D=%zu, minPnPInliers=%d",
                              bestId,
                              bestVisible,
                              _sfmOptions.pnpOptions.minNumInliers);
    return bestId;
}

// ============================================================
// 内部：注册图像（PnP）
// ============================================================

bool IncrementalSfm::registerImage(ImageId imageId)
{
    // 加载相机内参
    Camera cam;
    if (!getCamera(imageId, cam))
    {
        _lastErrorMessage = "getCamera(" + std::to_string(imageId) + ") failed";
        return false;
    }

    const ImageData &img = _reconstruction->image(imageId);

    // 收集 3D-2D 对应关系
    std::vector<std::array<double, 3>> worldPts;
    std::vector<std::array<double, 2>> imagePts;

    auto neighbors = _correspondenceGraph.connectedImages(imageId);
    // 用 set 避免重复添加同一个三维点
    std::unordered_set<Point3DId> addedPoints;

    for (ImageId nid : neighbors)
    {
        if (!_reconstruction->isRegistered(nid))
            continue;
        const auto &matches = _correspondenceGraph.matchesBetween(imageId, nid);
        const ImageData &nimg = _reconstruction->image(nid);

        for (const auto &m : matches)
        {
            FeatureIdx myFeat = (imageId < nid) ? m.idx1 : m.idx2;
            FeatureIdx nFeat = (imageId < nid) ? m.idx2 : m.idx1;

            if (nFeat >= nimg.point3DIds.size())
                continue;
            Point3DId p3dId = nimg.point3DIds[nFeat];
            if (p3dId == kInvalidPoint3DId)
                continue;
            if (!_reconstruction->hasPoint3D(p3dId))
                continue;
            if (addedPoints.count(p3dId))
                continue;

            if (myFeat >= img.keypoints.size())
                continue;

            addedPoints.insert(p3dId);
            worldPts.push_back(_reconstruction->point3D(p3dId).xyz);
            imagePts.push_back(
                {static_cast<double>(img.keypoints[myFeat].x), static_cast<double>(img.keypoints[myFeat].y)});
        }
    }

    if (static_cast<int>(worldPts.size()) < _sfmOptions.pnpOptions.minNumInliers)
    {
        std::ostringstream oss;
        oss << "not enough 2D-3D observations for PnP: observations=" << worldPts.size()
            << ", minPnPInliers=" << _sfmOptions.pnpOptions.minNumInliers
            << ", connectedNeighbors=" << neighbors.size();
        _lastErrorMessage = oss.str();
        return false;
    }

    // PnP 求解
    auto pnpResult = PnpSolver::solveWithCamera(worldPts, imagePts, cam, _sfmOptions.pnpOptions);
    if (!pnpResult.success)
    {
        std::ostringstream oss;
        oss << "PnP failed: observations=" << worldPts.size()
            << ", inliers=" << pnpResult.numInliers
            << ", inlierRatio=" << pnpResult.inlierRatio
            << ", minPnPInliers=" << _sfmOptions.pnpOptions.minNumInliers
            << ", minInlierRatio=" << _sfmOptions.pnpOptions.minInlierRatio
            << ", maxReprojError=" << _sfmOptions.pnpOptions.maxReprojError;
        _lastErrorMessage = oss.str();
        return false;
    }

    Logger::instance()->infof("[SFM] Image %u PnP success: observations=%zu, inliers=%d, inlierRatio=%.3f",
                              imageId,
                              worldPts.size(),
                              pnpResult.numInliers,
                              pnpResult.inlierRatio);

    // 应用 PnP 结果更新相机外参
    cam.setPose(pnpResult.R, pnpResult.C);

    // 注册到重建
    _reconstruction->registerImage(imageId, cam);

    return true;
}

// ============================================================
// 内部：光束法平差
// ============================================================

void IncrementalSfm::runBundleAdjust(bool localOnly, const std::vector<ImageId> &anchorIds)
{
    // 收集参与 BA 的图像 ID
    std::vector<ImageId> baImageIds;
    if (localOnly && !anchorIds.empty())
    {
        // 局部 BA：收集锚定图像及其邻居
        std::unordered_set<ImageId> baSet;
        for (ImageId aid : anchorIds)
        {
            baSet.insert(aid);
            // 按匹配数选取前 N 个已注册邻居
            auto topN = _correspondenceGraph.topConnectedImages(aid, static_cast<size_t>(_sfmOptions.localBANumImages));
            for (auto &[nid, _] : topN)
            {
                if (_reconstruction->isRegistered(nid))
                    baSet.insert(nid);
            }
        }
        baImageIds.assign(baSet.begin(), baSet.end());
    }
    else
    {
        // 全局 BA：所有已注册图像
        baImageIds = _reconstruction->registeredImageIds();
    }

    if (baImageIds.size() < 2)
        return;

    // 构造 imageId → BA 内部相机索引的映射
    std::unordered_map<ImageId, int> idToIdx;
    std::vector<Camera> baCameras;
    for (size_t i = 0; i < baImageIds.size(); ++i)
    {
        idToIdx[baImageIds[i]] = static_cast<int>(i);
        baCameras.push_back(_reconstruction->camera(baImageIds[i]));
    }

    if (_sfmOptions.useKnownCameraPoses && !localOnly)
    {
        alignReconstructionToKnownPosePriors(baImageIds, &baCameras);
    }

    // 收集轨迹（同时记录 trackIdx → Point3DId 的映射，用于回写和过滤）
    std::vector<BATrack> baTracks;
    std::vector<Point3DId> trackToPid; // 与 baTracks 索引对应

    const auto allPtIds = _reconstruction->allPoint3DIds();
    for (Point3DId pid : allPtIds)
    {
        if (!_reconstruction->hasPoint3D(pid))
            continue;
        const ScenePoint3D &pt = _reconstruction->point3D(pid);
        BATrack track;
        track.initialPoint = pt.xyz;

        for (const auto &elem : pt.track.elements)
        {
            auto idxIt = idToIdx.find(elem.imageId);
            if (idxIt == idToIdx.end())
                continue;

            const ImageData &img = _reconstruction->image(elem.imageId);
            if (elem.featureIdx >= img.keypoints.size())
                continue;

            BAObservation obs;
            obs.cameraIndex = idxIt->second;
            obs.u = img.keypoints[elem.featureIdx].x;
            obs.v = img.keypoints[elem.featureIdx].y;
            obs.weight = pt.track.confidence;
            track.observations.push_back(obs);
        }

        // 至少 2 个观测才有意义
        if (track.observations.size() >= 2)
        {
            baTracks.push_back(std::move(track));
            trackToPid.push_back(pid);
        }
    }

    if (baTracks.empty())
        return;

    // 构造本次 BA 选项：
    //   - 全局 BA 固定 index=0 的相机（gauge 固定，消除坐标系漂移）
    //   - 局部 BA 不固定（局部块坐标由全局约束）
    BAOptions baOpt = _sfmOptions.baOptions;
    if (_sfmOptions.useKnownCameraPoses && baOpt.cameraPosePriors.empty())
    {
        baOpt.cameraPosePriors = buildCameraPosePriorsFromInputCameras(baImageIds);
        baOpt.refineCameraPose = false;
    }
    if (!localOnly && !baImageIds.empty())
    {
        baOpt.fixedCameraIndices = {0}; // gauge: 第一个相机位姿保持不变
    }

    // 执行 BA
    const BAResult baResult = BundleAdjust::optimizePoints(baCameras, baTracks, baOpt);

    // 回写优化后的相机位姿（跳过被 gauge 固定的相机）
    for (size_t i = 0; i < baImageIds.size(); ++i)
    {
        if (i < baResult.refinedCameras.size())
        {
            _reconstruction->camera(baImageIds[i]) = baResult.refinedCameras[i];
        }
    }

    // 回写优化后的三维点坐标、误差；并进行观测级过滤（参考 COLMAP）
    // BA 标记为 invalid 的点 → 先检查各观测的个别重投影误差，
    // 移除高误差观测后若轨迹仍 >= 2，则保留点但缩短轨迹；否则删除整个点。
    int deletedPts = 0;
    int removedObs = 0;
    for (size_t ti = 0; ti < baTracks.size() && ti < baResult.points.size(); ++ti)
    {
        const Point3DId pid = trackToPid[ti];
        if (!_reconstruction->hasPoint3D(pid))
            continue;

        const BARefinedPoint &bp = baResult.points[ti];
        if (!bp.valid)
        {
            // ── 观测级过滤：逐观测检查重投影误差 ──
            auto &pt = _reconstruction->point3D(pid);
            const double filterThresh = _sfmOptions.baOptions.filterMaxReprojError;
            std::vector<size_t> badObsIndices;

            for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
            {
                const auto &elem = pt.track.elements[oi];
                if (!_reconstruction->isRegistered(elem.imageId))
                    continue;
                if (!_reconstruction->hasCamera(elem.imageId))
                    continue;

                const Camera &cam = _reconstruction->camera(elem.imageId);
                const ImageData &imgData = _reconstruction->image(elem.imageId);
                if (elem.featureIdx >= imgData.keypoints.size())
                    continue;

                double u_obs = imgData.keypoints[elem.featureIdx].x;
                double v_obs = imgData.keypoints[elem.featureIdx].y;

                // 计算重投影
                double world[3] = {bp.point[0], bp.point[1], bp.point[2]};
                double uv_proj[2] = {0, 0};
                bool projected = cam.projectWorldPoint(world, uv_proj);
                if (!projected)
                {
                    badObsIndices.push_back(oi);
                    continue;
                }
                double du = uv_proj[0] - u_obs;
                double dv = uv_proj[1] - v_obs;
                double reproj = std::sqrt(du * du + dv * dv);
                if (reproj > filterThresh * 1.5)
                {
                    badObsIndices.push_back(oi);
                }
            }

            size_t goodObs = pt.track.elements.size() - badObsIndices.size();
            if (goodObs >= 2 && !badObsIndices.empty())
            {
                // 保留点，仅移除坏观测
                for (auto it = badObsIndices.rbegin(); it != badObsIndices.rend(); ++it)
                {
                    const auto &elem = pt.track.elements[*it];
                    if (_reconstruction->hasImage(elem.imageId))
                    {
                        auto &imgd = _reconstruction->image(elem.imageId);
                        if (elem.featureIdx < imgd.point3DIds.size())
                        {
                            imgd.point3DIds[elem.featureIdx] = kInvalidPoint3DId;
                        }
                    }
                    pt.track.elements.erase(pt.track.elements.begin() + static_cast<long>(*it));
                    ++removedObs;
                }
                // 更新点坐标和误差为 BA 结果
                pt.xyz = bp.point;
                pt.error = bp.rmsAfter;
            }
            else
            {
                // 彻底删除
                _reconstruction->deletePoint3D(pid);
                ++deletedPts;
            }
        }
        else
        {
            // ── 有效点也进行观测级过滤：移除残差特别高的观测 ──
            auto &pt = _reconstruction->point3D(pid);
            const double filterThresh = _sfmOptions.baOptions.filterMaxReprojError;
            std::vector<size_t> badObsIndices;

            for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
            {
                const auto &elem = pt.track.elements[oi];
                if (!_reconstruction->isRegistered(elem.imageId))
                    continue;
                if (!_reconstruction->hasCamera(elem.imageId))
                    continue;

                const Camera &cam = _reconstruction->camera(elem.imageId);
                const ImageData &imgData = _reconstruction->image(elem.imageId);
                if (elem.featureIdx >= imgData.keypoints.size())
                    continue;

                double u_obs = imgData.keypoints[elem.featureIdx].x;
                double v_obs = imgData.keypoints[elem.featureIdx].y;

                double world[3] = {bp.point[0], bp.point[1], bp.point[2]};
                double uv_proj[2] = {0, 0};
                bool projected = cam.projectWorldPoint(world, uv_proj);
                if (!projected)
                {
                    badObsIndices.push_back(oi);
                    continue;
                }
                double du = uv_proj[0] - u_obs;
                double dv = uv_proj[1] - v_obs;
                double reproj = std::sqrt(du * du + dv * dv);
                // 对有效点使用 2 倍阈值剔除明显错误观测
                if (reproj > filterThresh * 2.0)
                {
                    badObsIndices.push_back(oi);
                }
            }

            if (!badObsIndices.empty())
            {
                size_t goodObs = pt.track.elements.size() - badObsIndices.size();
                if (goodObs >= 2)
                {
                    for (auto it = badObsIndices.rbegin(); it != badObsIndices.rend(); ++it)
                    {
                        const auto &elem = pt.track.elements[*it];
                        if (_reconstruction->hasImage(elem.imageId))
                        {
                            auto &imgd = _reconstruction->image(elem.imageId);
                            if (elem.featureIdx < imgd.point3DIds.size())
                            {
                                imgd.point3DIds[elem.featureIdx] = kInvalidPoint3DId;
                            }
                        }
                        pt.track.elements.erase(pt.track.elements.begin() + static_cast<long>(*it));
                        ++removedObs;
                    }
                }
            }

            pt.xyz = bp.point;
            pt.error = bp.rmsAfter;
        }
    }

    Logger::instance()->infof("[BA] deletedPts=%d, removedObs=%d", deletedPts, removedObs);

    // 全局 BA：记录统计供最终结果使用（lastGlobalBA* 成员变量）
    if (!localOnly)
    {
        _lastGlobalBARmsBefore = baResult.meanRmsBefore;
        _lastGlobalBARmsAfter = baResult.meanRmsAfter;
        _lastGlobalBATracksTotal = baResult.totalTracks;
        _lastGlobalBATracksOptimized = baResult.optimizedTracks;
        _lastGlobalBATracksFiltered = (int)baTracks.size() - baResult.optimizedTracks;
        if (_lastGlobalBATracksFiltered < 0)
            _lastGlobalBATracksFiltered = deletedPts;
    }
}

// ============================================================
// 内部：迭代全局 BA 精化（参考 COLMAP IterativeGlobalRefinement）
// ============================================================

void IncrementalSfm::iterativeGlobalBA()
{
    const int maxRounds = std::max(1, _sfmOptions.iterativeBARounds);
    size_t prevNumPoints = _reconstruction->numPoints3D();

    for (int round = 0; round < maxRounds; ++round)
    {
        Logger::instance()->infof("[SFM] IterativeGlobalBA round %d/%d: numPts=%zu", round + 1, maxRounds,
                      _reconstruction->numPoints3D());

        // (1) 过滤负深度点
        if (_sfmOptions.filterNegativeDepth)
        {
            int nNeg = filterNegativeDepthPoints();
            if (nNeg > 0)
            {
                Logger::instance()->infof("[SFM]   Filtered %d negative-depth points", nNeg);
            }
        }

        // (2) 执行全局 BA
        runBundleAdjust(false);

        // (3) 利用 BA 后更新的相机位姿重三角化所有 3D 点（参考 COLMAP Retriangulate）
        Triangulator tri(*_reconstruction, _correspondenceGraph);
        int nRetri = tri.retriangulatePoints(_sfmOptions.filterMaxReprojError);
        if (nRetri > 0)
        {
            Logger::instance()->infof("[SFM]   Retriangulated %d points with updated poses", nRetri);
        }

        // (4) 过滤点质量（重投影误差 + 三角化角）
        tri.filterPoints(_sfmOptions.filterMaxReprojError, _sfmOptions.filterMinTriAngle);

        // (5) 补三角化（尝试延伸已有轨迹，含深度检查）
        tri.completeTracks(_sfmOptions.triangulatorOptions);

        // (6) 收敛判断：3D 点数变化 < 1%
        size_t curNumPoints = _reconstruction->numPoints3D();
        double changeRate = (prevNumPoints > 0)
                                ? std::fabs(static_cast<double>(curNumPoints) - static_cast<double>(prevNumPoints)) /
                                      static_cast<double>(prevNumPoints)
                                : 1.0;

        Logger::instance()->infof("[SFM]   After round %d: numPts=%zu, changeRate=%.4f", round + 1, curNumPoints,
                      changeRate);

        if (round >= 1 && changeRate < 0.01)
        {
            Logger::instance()->info("[SFM]   Converged (changeRate < 1%)");
            break;
        }
        prevNumPoints = curNumPoints;
    }
}

// ============================================================
// 内部：过滤负深度点（参考 COLMAP FilterObservationsWithNegativeDepth）
// ============================================================

int IncrementalSfm::filterNegativeDepthPoints()
{
    int deletedCount = 0;
    auto allPtIds = _reconstruction->allPoint3DIds();

    for (Point3DId pid : allPtIds)
    {
        if (!_reconstruction->hasPoint3D(pid))
            continue;
        auto &pt = _reconstruction->point3D(pid);
        const auto &xyz = pt.xyz;

        // 检查该点在每个观测相机中的深度
        bool hasNegativeDepth = false;
        std::vector<size_t> badObsIndices;

        for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
        {
            const auto &elem = pt.track.elements[oi];
            if (!_reconstruction->isRegistered(elem.imageId))
                continue;
            if (!_reconstruction->hasCamera(elem.imageId))
                continue;

            const Camera &cam = _reconstruction->camera(elem.imageId);
            const double world[3] = {xyz[0], xyz[1], xyz[2]};
            double cameraPoint[3] = {0.0, 0.0, 0.0};
            cam.worldToCamera(world, cameraPoint);

            if (cameraPoint[2] < 0.0)
            {
                badObsIndices.push_back(oi);
                hasNegativeDepth = true;
            }
        }

        if (!hasNegativeDepth)
            continue;

        // 如果全部观测都是负深度或移除坏观测后不足 2 个，删除整个点
        size_t goodObs = pt.track.elements.size() - badObsIndices.size();
        if (goodObs < 2)
        {
            _reconstruction->deletePoint3D(pid);
            ++deletedCount;
        }
        else
        {
            // 移除坏观测（从后往前删）
            for (auto it = badObsIndices.rbegin(); it != badObsIndices.rend(); ++it)
            {
                const auto &elem = pt.track.elements[*it];
                // 清理 ImageData 中的关联
                if (_reconstruction->hasImage(elem.imageId))
                {
                    auto &imgData = _reconstruction->image(elem.imageId);
                    if (elem.featureIdx < imgData.point3DIds.size())
                    {
                        imgData.point3DIds[elem.featureIdx] = kInvalidPoint3DId;
                    }
                }
                pt.track.elements.erase(pt.track.elements.begin() + static_cast<long>(*it));
            }
        }
    }

    return deletedCount;
}

// ============================================================
// 内部：可见性缓存管理
// ============================================================

void IncrementalSfm::rebuildVisibilityCache()
{
    _visibilityCache.clear();
    auto allIds = _reconstruction->allImageIds();

    for (ImageId id : allIds)
    {
        if (_reconstruction->isRegistered(id))
        {
            continue;
        }

        size_t numVisible = 0;
        auto neighbors = _correspondenceGraph.connectedImages(id);
        for (ImageId nid : neighbors)
        {
            if (!_reconstruction->isRegistered(nid))
            {
                continue;
            }
            const auto &matches = _correspondenceGraph.matchesBetween(id, nid);
            const ImageData &nimg = _reconstruction->image(nid);

            for (const auto &m : matches)
            {
                FeatureIdx nfeat = (id < nid) ? m.idx2 : m.idx1;
                if (nfeat < nimg.point3DIds.size() && nimg.point3DIds[nfeat] != kInvalidPoint3DId)
                {
                    ++numVisible;
                }
            }
        }

        _visibilityCache[id] = numVisible;
    }
    _visibilityCacheDirty = false;
}

void IncrementalSfm::invalidateVisibilityCache()
{
    _visibilityCacheDirty = true;
}

void IncrementalSfm::updateVisibilityCacheForImage(ImageId imageId, const std::vector<Point3DId> &previousPointIds)
{
    if (_visibilityCacheDirty || !_reconstruction->hasImage(imageId))
    {
        return;
    }

    const auto &currentPointIds = _reconstruction->image(imageId).point3DIds;
    const size_t numFeatures = std::min(previousPointIds.size(), currentPointIds.size());

    for (size_t featureIndex = 0; featureIndex < numFeatures; ++featureIndex)
    {
        const Point3DId previousId = previousPointIds[featureIndex];
        const Point3DId currentId = currentPointIds[featureIndex];
        if (currentId == kInvalidPoint3DId || currentId == previousId || !_reconstruction->hasPoint3D(currentId))
        {
            continue;
        }

        const ScenePoint3D &point = _reconstruction->point3D(currentId);
        for (const auto &trackElem : point.track.elements)
        {
            auto correspondences = _correspondenceGraph.findCorrespondences(trackElem.imageId, trackElem.featureIdx);
            for (const auto &corr : correspondences)
            {
                if (_reconstruction->isRegistered(corr.imageId))
                {
                    continue;
                }
                ++_visibilityCache[corr.imageId];
            }
        }
    }
}

// ============================================================
// 内部：进度报告
// ============================================================

bool IncrementalSfm::reportProgress(int numRegistered, int numTotal, const std::string &msg, SfmProgressCallback &cb)
{
    if (!cb)
        return true;
    bool continueRun = cb(numRegistered, numTotal, msg);
    if (!continueRun)
        _isAborted = true;
    return continueRun;
}

} // namespace xjw
