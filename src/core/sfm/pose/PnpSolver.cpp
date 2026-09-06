#include "PnpSolver.h"

#include "DeterministicOpenCvRansac.h"
#include "ReferenceResectionSolver.h"
#include "geometry/OpenCvCameraAdapter.h"
#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>

#include <algorithm>
#include <cmath>

namespace xjw
{
    namespace
    {
        PnpResult runReferenceResection(const std::vector<std::array<double, 3>>& worldPoints,
                                        const std::vector<std::array<double, 2>>& imagePoints,
                                        const FramePinholeCamera& camera,
                                        const PnpOptions& options)
        {
            PnpResult result;
            result.inputCandidateCount = static_cast<int>(worldPoints.size());
            result.usedReferenceResection = true;
            const ReferenceResectionResult reference =
                solveReferenceResection(worldPoints, imagePoints, camera, options.maxReprojError);
            result.inlierMask = reference.inlierMask;
            result.numInliers = reference.numInliers;
            result.inlierRatio = worldPoints.empty()
                                     ? 0.0
                                     : static_cast<double>(result.numInliers) / static_cast<double>(worldPoints.size());
            result.referenceThresholdLevel = reference.selectedThresholdLevel;
            result.ransacIterations = reference.ransacIterations;
            if (!reference.success || result.numInliers < std::max(5, options.minNumInliers))
            {
                return result;
            }
            result.R = reference.cameraToWorldRotation;
            result.C = reference.cameraCenter;
            result.success = true;
            return result;
        }
    } // namespace

    PnpResult PnpSolver::solve(const std::vector<std::array<double, 3>>& worldPoints,
                               const std::vector<std::array<double, 2>>& imagePoints,
                               double fu,
                               double fv,
                               double cu,
                               double cv,
                               int uDir,
                               int vDir,
                               bool depthFlipped,
                               const PnpOptions& options)
    {
        return solveWithDistortion(worldPoints,
                                   imagePoints,
                                   fu,
                                   fv,
                                   cu,
                                   cv,
                                   uDir,
                                   vDir,
                                   depthFlipped,
                                   FramePinholeCamera::Distortion{},
                                   options);
    }

    PnpResult PnpSolver::solveWithDistortion(const std::vector<std::array<double, 3>>& worldPoints,
                                             const std::vector<std::array<double, 2>>& imagePoints,
                                             double fu,
                                             double fv,
                                             double cu,
                                             double cv,
                                             int uDir,
                                             int vDir,
                                             bool depthFlipped,
                                             const FramePinholeCamera::Distortion& distortion,
                                             const PnpOptions& options)
    {
        if (options.useReferenceResection)
        {
            FramePinholeCamera camera;
            camera.setIntrinsics(fu, fv, cu, cv);
            camera.setAxisDirections(uDir, vDir);
            camera.setDepthAxisFlipped(depthFlipped);
            camera.setDistortion(distortion);
            return runReferenceResection(worldPoints, imagePoints, camera, options);
        }

        PnpResult result;
        const size_t n = worldPoints.size();
        result.inputCandidateCount = static_cast<int>(n);
        if (n < 4 || n != imagePoints.size())
            return result;

        // ---- 准备 OpenCV 数据 ----
        std::vector<cv::Point3d> objPts(n);
        std::vector<cv::Point2d> imgPts(n);
        for (size_t i = 0; i < n; ++i)
        {
            objPts[i] = cv::Point3d(worldPoints[i][0], worldPoints[i][1], worldPoints[i][2]);
            imgPts[i] = cv::Point2d(imagePoints[i][0], imagePoints[i][1]);
        }

        // 构造内参矩阵
        // TSai 模型：u = uDir * fu * x + cu → OpenCV 等效：fx = uDir * fu
        // 当 depthFlipped 时需额外翻转符号，使归一化坐标处于正深度约定
        const cv::Mat cameraMatrix = openCvCameraMatrix(fu, fv, cu, cv, uDir, vDir, depthFlipped, true);

        const std::array<double, 5> distortionCoefficients{{
            distortion.radialK1,
            distortion.radialK2,
            distortion.tangentialP1,
            distortion.tangentialP2,
            distortion.radialK3,
        }};
        if (!std::all_of(distortionCoefficients.begin(),
                         distortionCoefficients.end(),
                         [](double coefficient) { return std::isfinite(coefficient); }))
        {
            return result;
        }

        // OpenCV 的 Brown-Conrady 参数顺序固定为 k1, k2, p1, p2, k3。
        const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << distortionCoefficients[0],
                                    distortionCoefficients[1],
                                    distortionCoefficients[2],
                                    distortionCoefficients[3],
                                    distortionCoefficients[4]);

        cv::Mat rvec, tvec;
        cv::Mat inliers;
        const bool useExtrinsicGuess = options.useInitialPose;
        if (useExtrinsicGuess)
        {
            rvec = openCvRvecFromCameraToWorldPose(options.initialCameraToWorldRotation, depthFlipped);
            tvec = openCvTvecFromCameraPose(
                options.initialCameraToWorldRotation, options.initialCameraCenter, depthFlipped);
        }

        const std::vector<cv::Point3d>* ransacObjPts = &objPts;
        const std::vector<cv::Point2d>* ransacImgPts = &imgPts;
        std::vector<cv::Point3d> guidedObjPts;
        std::vector<cv::Point2d> guidedImgPts;
        std::vector<int> guidedOriginalIndices;
        if (useExtrinsicGuess && options.useInitialPosePrefilter)
        {
            std::vector<cv::Point2d> projected;
            cv::projectPoints(objPts, rvec, tvec, cameraMatrix, distCoeffs, projected);
            const double maxError = std::max(options.maxReprojError, options.initialPosePrefilterMaxReprojError);
            const double maxErrorSquared = maxError * maxError;
            guidedObjPts.reserve(n);
            guidedImgPts.reserve(n);
            guidedOriginalIndices.reserve(n);
            for (std::size_t index = 0; index < n; ++index)
            {
                const double dx = projected[index].x - imgPts[index].x;
                const double dy = projected[index].y - imgPts[index].y;
                if (dx * dx + dy * dy > maxErrorSquared)
                {
                    continue;
                }
                guidedObjPts.push_back(objPts[index]);
                guidedImgPts.push_back(imgPts[index]);
                guidedOriginalIndices.push_back(static_cast<int>(index));
            }
            const int minCandidates = std::max(4, options.initialPosePrefilterMinCandidates);
            result.prefilterCandidateCount = static_cast<int>(guidedObjPts.size());
            if (static_cast<int>(guidedObjPts.size()) >= minCandidates)
            {
                ransacObjPts = &guidedObjPts;
                ransacImgPts = &guidedImgPts;
                result.usedInitialPosePrefilter = true;
            }
            else
            {
                guidedOriginalIndices.clear();
            }
        }

        // ---- 调用 OpenCV PnP RANSAC ----
        const int minimalMethod = options.useMinimalP3p ? cv::SOLVEPNP_AP3P : cv::SOLVEPNP_ITERATIVE;
        const bool ok = opencv_utils::runDeterministicRansac(options.ransacSeed,
                                                             [&]()
                                                             {
                                                                 return cv::solvePnPRansac(
                                                                     *ransacObjPts,
                                                                     *ransacImgPts,
                                                                     cameraMatrix,
                                                                     distCoeffs,
                                                                     rvec,
                                                                     tvec,
                                                                     useExtrinsicGuess,
                                                                     options.maxIterations,
                                                                     static_cast<float>(options.maxReprojError),
                                                                     options.confidence,
                                                                     inliers,
                                                                     minimalMethod);
                                                             });

        if (!ok || inliers.rows < options.minNumInliers)
        {
            return result;
        }

        // 参考后方交会以 P3P 最小样本产生 RANSAC 假设，再用一致集完成非线性精化。
        // 精化失败时保留已经通过门控的最小解，避免把数值退化误报成注册失败。
        if (options.useMinimalP3p && inliers.rows >= 4)
        {
            std::vector<cv::Point3d> inlierObjPts;
            std::vector<cv::Point2d> inlierImgPts;
            inlierObjPts.reserve(static_cast<std::size_t>(inliers.rows));
            inlierImgPts.reserve(static_cast<std::size_t>(inliers.rows));
            for (int row = 0; row < inliers.rows; ++row)
            {
                const int index = inliers.at<int>(row, 0);
                if (index >= 0 && index < static_cast<int>(ransacObjPts->size()))
                {
                    inlierObjPts.push_back((*ransacObjPts)[static_cast<std::size_t>(index)]);
                    inlierImgPts.push_back((*ransacImgPts)[static_cast<std::size_t>(index)]);
                }
            }
            if (inlierObjPts.size() >= 4)
            {
                cv::Mat refinedRvec = rvec.clone();
                cv::Mat refinedTvec = tvec.clone();
                if (cv::solvePnP(inlierObjPts,
                                 inlierImgPts,
                                 cameraMatrix,
                                 distCoeffs,
                                 refinedRvec,
                                 refinedTvec,
                                 true,
                                 cv::SOLVEPNP_ITERATIVE))
                {
                    rvec = std::move(refinedRvec);
                    tvec = std::move(refinedTvec);
                }
            }
        }

        // 非线性精化后对全体原始对应重新分类。批量后方交会比较的是最终模型
        // 的支持数，不能沿用产生最小假设时的旧 RANSAC 一致集。
        std::vector<cv::Point2d> refinedProjections;
        cv::projectPoints(objPts, rvec, tvec, cameraMatrix, distCoeffs, refinedProjections);
        result.inlierMask.assign(n, 0);
        const double maximumSquaredError = options.maxReprojError * options.maxReprojError;
        for (std::size_t index = 0; index < n; ++index)
        {
            const double dx = refinedProjections[index].x - imgPts[index].x;
            const double dy = refinedProjections[index].y - imgPts[index].y;
            if (std::isfinite(dx) && std::isfinite(dy) && dx * dx + dy * dy <= maximumSquaredError)
            {
                result.inlierMask[index] = 1;
                ++result.numInliers;
            }
        }
        result.inlierRatio = static_cast<double>(result.numInliers) / static_cast<double>(n);
        const double relaxedMinInlierRatio = std::clamp(options.relaxedMinInlierRatio, 0.0, options.minInlierRatio);
        const int relaxedAbsoluteMinInliers =
            options.relaxedMinNumInliers > 0 ? options.relaxedMinNumInliers : std::max(options.minNumInliers, 15);
        const bool passesConfiguredRatio = result.inlierRatio >= options.minInlierRatio;
        const bool passesAbsoluteSupport = options.allowRelaxedInlierRatio &&
                                           result.numInliers >= relaxedAbsoluteMinInliers &&
                                           result.inlierRatio >= relaxedMinInlierRatio;
        const bool isSmallSample =
            options.smallSampleThreshold > 0 && static_cast<int>(n) < options.smallSampleThreshold;
        const bool passesSmallSampleRatio =
            !isSmallSample || result.inlierRatio >= std::clamp(options.smallSampleMinInlierRatio, 0.0, 1.0);
        if (result.numInliers < options.minNumInliers || (!passesConfiguredRatio && !passesAbsoluteSupport) ||
            !passesSmallSampleRatio)
        {
            return result;
        }

        // ---- 将 OpenCV 输出转换为 PlaScan 约定 ----
        // OpenCV 输出的 rvec/tvec 是 world-to-camera 变换：
        //   X_cam = R_cv * X_world + t_cv
        //   相机中心 C = -R_cv^T * t_cv
        //   camera-to-world 旋转 R_c2w = R_cv^T

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv); // 3x3 旋转矩阵（world-to-camera）

        // 若使用正深度约定 K，将 R/t 从正深度坐标系转回物理相机坐标系：
        //   R_physical = D * R_positive * D,  t_physical = D * t_positive
        //   D = diag(1, 1, -1)
        if (depthFlipped)
        {
            R_cv.at<double>(0, 2) *= -1;
            R_cv.at<double>(1, 2) *= -1;
            R_cv.at<double>(2, 0) *= -1;
            R_cv.at<double>(2, 1) *= -1;
            tvec.at<double>(2) *= -1;
        }

        // camera-to-world 旋转 = R_cv^T
        cv::Mat R_c2w = R_cv.t();

        // 相机中心 C = -R_cv^T * t_cv
        cv::Mat C_cv = -R_c2w * tvec;

        // 转换到 std::array
        result.success = true;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                result.R[i * 3 + j] = R_c2w.at<double>(i, j);
            }
            result.C[i] = C_cv.at<double>(i, 0);
        }

        return result;
    }

    PnpResult PnpSolver::solveWithCamera(const std::vector<std::array<double, 3>>& worldPoints,
                                         const std::vector<std::array<double, 2>>& imagePoints,
                                         const FramePinholeCamera& cam,
                                         const PnpOptions& options)
    {
        if (options.useReferenceResection)
        {
            return runReferenceResection(worldPoints, imagePoints, cam, options);
        }
        return solveWithDistortion(worldPoints,
                                   imagePoints,
                                   cam.focalX(),
                                   cam.focalY(),
                                   cam.principalX(),
                                   cam.principalY(),
                                   cam.uAxisSign(),
                                   cam.vAxisSign(),
                                   cam.depthAxisFlipped(),
                                   cam.distortion(),
                                   options);
    }

} // namespace xjw
