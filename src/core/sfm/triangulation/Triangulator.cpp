#include "Triangulator.h"

#include "log/Logger.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cmath>
#include <limits>

namespace xjw
{

Triangulator::Triangulator(SfmReconstruction &reconstruction, const CorrespondenceGraph &graph)
    : _reconstruction(reconstruction), _correspondenceGraph(graph)
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

            std::array<double, 3> xyz;
            if (triangulatePair(imageId, fi, corr.imageId, corr.featureIdx, options, xyz))
            {
                // 创建新三维点
                Track track;
                track.elements.push_back({imageId, fi});
                track.elements.push_back({corr.imageId, corr.featureIdx});

                _reconstruction.addPoint3DWithTrack(xyz, track);

                stats.numCreated++;
                break;
            }
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
    int numFiltered = 0;
    auto allIds = _reconstruction.allPoint3DIds();

    for (Point3DId pid : allIds)
    {
        if (!_reconstruction.hasPoint3D(pid))
        {
            continue;
        }
        const auto &pt = _reconstruction.point3D(pid);

        bool shouldFilter = false;

        // 使用已存储的重投影 RMS 误差（BA 后由 recomputeReprojErrors 更新）
        if (!std::isfinite(pt.error) || pt.error > maxReprojError)
        {
            shouldFilter = true;
        }

        // 检查三角化角：遍历所有观测相机对，取最大角
        if (!shouldFilter && pt.track.length() >= 2)
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
                shouldFilter = true;
            }
        }

        if (shouldFilter)
        {
            _reconstruction.deletePoint3D(pid);
            ++numFiltered;
        }
    }

    return numFiltered;
}

// ---- 过滤短轨迹 ----

int Triangulator::filterShortTracks(int minTrackLen)
{
    int numFiltered = 0;
    auto allIds = _reconstruction.allPoint3DIds();

    for (Point3DId pid : allIds)
    {
        if (!_reconstruction.hasPoint3D(pid))
        {
            continue;
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
            _reconstruction.deletePoint3D(pid);
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
    double cameraPoint[3] = {0.0, 0.0, 0.0};
    cam.worldToCamera(world, cameraPoint);
    return cameraPoint[2] > 0.0;
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

        auto R = cam.cameraToWorldRotation();
        auto C = cam.cameraCenter();
        double fu = cam.focalX(), fv = cam.focalY();
        double cu = cam.principalX(), cv = cam.principalY();
        int udir = cam.uAxisSign(), vdir = cam.vAxisSign();
        double fx = (udir < 0 ? -1.0 : 1.0) * fu;
        double fy = (vdir < 0 ? -1.0 : 1.0) * fv;

        // R 是 camera-to-world（行优先），world-to-camera = R^T
        // t_w2c = -R^T * C
        cv::Mat Rw(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Rw.at<double>(i, j) = R[i * 3 + j]; // camera-to-world

        cv::Mat Rc = Rw.t(); // world-to-camera
        cv::Mat Cvec = (cv::Mat_<double>(3, 1) << C[0], C[1], C[2]);
        cv::Mat tvec = -Rc * Cvec;

        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cu, 0.0, fy, cv, 0.0, 0.0, 1.0);

        cv::Mat Rt(3, 4, CV_64F);
        Rc.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(Rt(cv::Rect(3, 0, 1, 3)));

        cv::Mat P = K * Rt; // 3x4

        projMats.push_back(P);
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
    int improved = 0;
    auto allIds = _reconstruction.allPoint3DIds();

    for (Point3DId pid : allIds)
    {
        if (!_reconstruction.hasPoint3D(pid))
            continue;
        auto &pt = _reconstruction.point3D(pid);

        if (pt.track.length() < 2)
            continue;

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
            continue;

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
            continue;

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
            continue;

        // 只在新结果更好时更新
        if (newAvgErr < oldAvgErr || oldAvgErr > maxReprojError)
        {
            pt.xyz = newXyz;
            pt.error = newAvgErr;
            ++improved;
        }
    }

    Logger::instance()->infof("[Triangulator] retriangulatePoints: improved %d / %zu points", improved,
                              allIds.size());
    return improved;
}

// ---- 重算所有点的重投影误差 ----

void Triangulator::recomputeReprojErrors()
{
    auto allIds = _reconstruction.allPoint3DIds();
    for (Point3DId pid : allIds)
    {
        if (!_reconstruction.hasPoint3D(pid))
            continue;
        auto &pt = _reconstruction.point3D(pid);

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
        pt.error = (cnt > 0) ? (sumErr / cnt) : 0.0;
    }
}

} // namespace xjw
