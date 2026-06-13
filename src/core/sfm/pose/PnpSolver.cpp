#include "PnpSolver.h"

#include "OpenCvCompat.h"
#include <opencv2/core.hpp>

#include <cmath>

namespace xjw
{

PnpResult PnpSolver::solve(const std::vector<std::array<double, 3>> &worldPoints,
                           const std::vector<std::array<double, 2>> &imagePoints, double fu, double fv, double cu,
                           double cv, int uDir, int vDir, bool depthFlipped, const PnpOptions &options)
{
    PnpResult result;
    const size_t n = worldPoints.size();
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
    const double depthSign = depthFlipped ? -1.0 : 1.0;
    const double fx = depthSign * (uDir < 0 ? -1.0 : 1.0) * fu;
    const double fy = depthSign * (vDir < 0 ? -1.0 : 1.0) * fv;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cu, 0.0, fy, cv, 0.0, 0.0, 1.0);

    // 暂不考虑畸变（假设已去畸变或畸变较小）
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    cv::Mat rvec, tvec;
    cv::Mat inliers;

    // ---- 调用 OpenCV PnP RANSAC ----
    bool ok = cv::solvePnPRansac(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec,
                                 false, // useExtrinsicGuess
                                 options.maxIterations, static_cast<float>(options.maxReprojError), options.confidence,
                                 inliers, cv::SOLVEPNP_ITERATIVE);

    if (!ok || inliers.rows < options.minNumInliers)
    {
        return result;
    }

    double inlierRatio = static_cast<double>(inliers.rows) / static_cast<double>(n);
    if (inlierRatio < options.minInlierRatio)
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

    result.numInliers = inliers.rows;
    result.inlierRatio = inlierRatio;

    // 构建完整内点掩码
    result.inlierMask.resize(n, 0);
    for (int i = 0; i < inliers.rows; ++i)
    {
        int idx = inliers.at<int>(i, 0);
        if (idx >= 0 && static_cast<size_t>(idx) < n)
        {
            result.inlierMask[idx] = 1;
        }
    }

    return result;
}

PnpResult PnpSolver::solveWithCamera(const std::vector<std::array<double, 3>> &worldPoints,
                                     const std::vector<std::array<double, 2>> &imagePoints, const Camera &cam,
                                     const PnpOptions &options)
{
    return solve(worldPoints, imagePoints, cam.focalX(), cam.focalY(), cam.principalX(), cam.principalY(),
                 cam.uAxisSign(), cam.vAxisSign(), cam.depthAxisFlipped(), options);
}

} // namespace xjw
