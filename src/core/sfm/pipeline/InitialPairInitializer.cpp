#include "InitialPairInitializer.h"
#include "IncrementalSfmDetail.h"
#include "SfmBundleAdjustCoordinator.h"
#include "geometry/OpenCvCameraAdapter.h"
#include "Intersection.h"
#include "tracks/CorrespondenceTrackThinner.h"
#include "tracks/MultiViewTrackBuilder.h"

#include "log/Logger.h"

#include "DeterministicOpenCvRansac.h"
#include "OpenCvCompat.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xjw
{

using namespace incremental_sfm_detail;

InitialPairInitializer::InitialPairInitializer(IncrementalSfm &owner)
    : _owner(owner)
{
}

std::vector<std::pair<ImageId, ImageId>> InitialPairInitializer::selectCandidates(
    int maxCandidates) const
{
    return _owner.selectInitialPairCandidates(maxCandidates);
}

bool InitialPairInitializer::initialize(ImageId id1, ImageId id2)
{
    return _owner.initializeFromPair(id1, id2);
}

void InitialPairInitializer::resetTrial(const SfmReconstruction &baseReconstruction)
{
    _owner.resetForInitialPairTrial(baseReconstruction);
}

std::vector<std::pair<ImageId, ImageId>> IncrementalSfm::selectInitialPairCandidates(int maxCandidates) const
{
    // 收集所有满足最少匹配数的像对，按匹配数降序排序
    struct PairInfo
    {
        ImageId id1;
        ImageId id2;
        size_t numMatches;
        double initialPairScore;
        double matchCoverage;
        size_t endpointDegree = 0;
        size_t localGraphReach = 0;
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
                const ImageData &img1 = _reconstruction->image(allIds[i]);
                const ImageData &img2 = _reconstruction->image(allIds[j]);
                const size_t minKeypoints = std::min(img1.keypoints.size(), img2.keypoints.size());
                const double coverage = minKeypoints > 0
                    ? static_cast<double>(nm) / static_cast<double>(minKeypoints)
                    : 0.0;
                double score = static_cast<double>(nm);
                if (coverage > 0.75)
                {
                    // 高覆盖匹配通常意味着两图极高重叠甚至近重复。它们对后续补点有价值，
                    // 但作为初始像对基线很弱，容易把无相机 SfM 初始化到病态解。
                    score *= 0.25;
                }
                pairs.push_back({allIds[i], allIds[j], nm, score, coverage});
            }
        }
    }

    std::unordered_map<ImageId, std::vector<ImageId>> initGraph;
    initGraph.reserve(pairs.size() * 2);
    for (const auto &pair : pairs)
    {
        initGraph[pair.id1].push_back(pair.id2);
        initGraph[pair.id2].push_back(pair.id1);
    }

    for (auto &pair : pairs)
    {
        std::unordered_set<ImageId> localReach;
        localReach.reserve(16);
        localReach.insert(pair.id1);
        localReach.insert(pair.id2);

        const auto addNeighbors = [&](ImageId imageId)
        {
            const auto it = initGraph.find(imageId);
            if (it == initGraph.end())
            {
                return size_t{0};
            }
            for (ImageId neighborId : it->second)
            {
                localReach.insert(neighborId);
            }
            return it->second.size();
        };

        const size_t degree1 = addNeighbors(pair.id1);
        const size_t degree2 = addNeighbors(pair.id2);
        pair.endpointDegree = degree1 + degree2;
        pair.localGraphReach = localReach.size();

        // 初始像对不能只看两图之间的匹配数，还要看它能否把增量 SfM 带到
        // 更大的连通区域。局部可达影像少的强匹配对容易初始化成功后困在小团里。
        const double reachBonus =
            std::min(1.0, static_cast<double>(pair.localGraphReach > 2 ? pair.localGraphReach - 2 : 0) / 8.0);
        const double degreeBonus = std::min(1.0, static_cast<double>(pair.endpointDegree) / 12.0);
        const double graphConnectivityFactor = 1.0 + 0.35 * reachBonus + 0.20 * degreeBonus;
        pair.initialPairScore *= graphConnectivityFactor;
    }

    // 按“可用于初始化的分数”降序排序，而不是只看匹配数。
    // 近重复视角可能匹配数最高，但不是好的初始基线。
    std::sort(pairs.begin(), pairs.end(),
              [](const PairInfo &a, const PairInfo &b)
              {
                  if (std::abs(a.initialPairScore - b.initialPairScore) > 1e-9)
                  {
                      return a.initialPairScore > b.initialPairScore;
                  }
                  return a.numMatches > b.numMatches;
              });

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
        const int pairSeed = opencv_compat::stableRansacSeed(pair.id1, pair.id2, 0x53454c45u);
        opencv_compat::runDeterministicRansac(pairSeed, [&]()
        {
            cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, ransacThresh, 0.999, maskF);
            return 0;
        });
        opencv_compat::runDeterministicRansac(pairSeed ^ 0x484f4d4fu, [&]()
        {
            cv::findHomography(pts1, pts2, cv::RANSAC, ransacThresh, maskH);
            return 0;
        });

        int fInliers = maskF.empty() ? 0 : cv::countNonZero(maskF);
        int hInliers = maskH.empty() ? 0 : cv::countNonZero(maskH);

        double hfRatio = (fInliers > 0) ? static_cast<double>(hInliers) / fInliers : 1.0;

        Logger::instance()->debugf("[SFM] Pair (%u, %u): matches=%zu, coverage=%.3f, localReach=%zu, "
                                   "endpointDegree=%zu, initScore=%.1f, F_inliers=%d, H_inliers=%d, H/F=%.3f",
                                   pair.id1,
                                   pair.id2,
                                   pair.numMatches,
                                   pair.matchCoverage,
                                   pair.localGraphReach,
                                   pair.endpointDegree,
                                   pair.initialPairScore,
                                   fInliers,
                                   hInliers,
                                   hfRatio);

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
    const cv::Mat K1 = openCvCameraMatrix(
        cam1.focalX(), cam1.focalY(), cam1.principalX(), cam1.principalY(),
        cam1.uAxisSign(), cam1.vAxisSign(), depthFlipped, true);
    const cv::Mat K2 = openCvCameraMatrix(
        cam2.focalX(), cam2.focalY(), cam2.principalX(), cam2.principalY(),
        cam2.uAxisSign(), cam2.vAxisSign(), depthFlipped, true);
    const double fx1 = K1.at<double>(0, 0);
    const double fy1 = K1.at<double>(1, 1);
    const double cx1 = K1.at<double>(0, 2);
    const double cy1 = K1.at<double>(1, 2);

    // 当两台相机内参相同时直接用 K1；否则先归一化到 K1 坐标系
    // 对于 findEssentialMat 使用 K1（假设近似相同或归一化后处理）
    // 注：实际行星影像通常同一台相机，内参完全相同
    cv::Mat K = K1;

    // ---- COLMAP 式同时估计 E 和 H ----
    const double ransacThresh = 1.0; // 像素
    cv::Mat maskE, maskH;

    const int pairSeed = opencv_compat::stableRansacSeed(id1, id2, 0x494e4954u);
    cv::Mat E = opencv_compat::runDeterministicRansac(pairSeed, [&]()
    {
        return xjw::opencv_compat::findEssentialMat(
            pts1, pts2, K, cv::RANSAC, 0.999, ransacThresh, maskE);
    });
    int E_inliers = cv::countNonZero(maskE);

    cv::Mat H = opencv_compat::runDeterministicRansac(pairSeed ^ 0x484f4d4fu, [&]()
    {
        return cv::findHomography(pts1, pts2, cv::RANSAC, ransacThresh, maskH);
    });
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
    SfmBundleAdjustCoordinator(*this).run(false);

    return true;
}

// ============================================================
// 内部：选择下一幅图像
// ============================================================


} // namespace xjw
