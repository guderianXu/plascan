#include "StereoDenseCloudPipelinePaths.h"

#include "EpipolarRectifier.h"
#include "PatchMatchCUDA.h"
#include "StereoDenseCloudPipelineOutput.h"
#include "DisparityTriangulator.h"
#include "DisparityFilter.h"
#include "PointCloudTifIO.h"
#include "SubpixelRefiner.h"

#include <QDir>

#include "OpenCvCompat.h"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace xjw
{
namespace mvs
{
namespace
{

cv::Mat buildGrayDisparityMask(const cv::Mat &disparity)
{
    cv::Mat validMask(disparity.size(), CV_8U, cv::Scalar(0));
    for (int r = 0; r < disparity.rows; ++r)
    {
        for (int c = 0; c < disparity.cols; ++c)
        {
            if (std::isfinite(disparity.at<float>(r, c)) && disparity.at<float>(r, c) != 0.0f)
            {
                validMask.at<uint8_t>(r, c) = 255;
            }
        }
    }
    return validMask;
}

bool writeOutputs(const cv::Mat &grayL,
                  const Camera &leftCamera,
                  const Camera &rightCamera,
                  const Camera &rectCam,
                  const std::string &outputDir,
                  const StereoPipelineConfig &config,
                  StereoPipelineResult &res,
                  const TriangulationResult &triResult,
                  const cv::Mat &H1inv,
                  StereoDenseCloudPipeline *owner)
{
    Q_UNUSED(leftCamera);
    Q_UNUSED(rightCamera);
    Q_UNUSED(rectCam);

    res.totalPoints = triResult.totalPixels;
    res.validPoints = triResult.validPoints;
    res.coveragePercent = 100.0f * triResult.validPoints / triResult.totalPixels;
    res.medianTriError = triResult.medianError;

    if (triResult.validPoints == 0)
    {
        res.errorMsg = "No valid triangulated points";
        return false;
    }

    if (config.outputTif)
    {
        emit owner->progressChanged("Writing TIF", 0.8f);
        res.tifPath = outputDir + "/PC.tif";
        std::string tifErr;
        if (!PointCloudTifIO::writeTif(res.tifPath, triResult, &tifErr))
        {
            res.errorMsg = "TIF write failed: " + tifErr;
            return false;
        }
    }

    if (config.outputPly)
    {
        emit owner->progressChanged("Writing PLY", 0.9f);
        res.plyPath = outputDir + "/PC.ply";
        std::string plyErr;
        if (!writeStereoPipelinePly(res.plyPath, triResult, grayL, plyErr))
        {
            res.errorMsg = "PLY write failed: " + plyErr;
            return false;
        }
    }

    emit owner->progressChanged("Done", 1.0f);
    std::fprintf(stderr, "[StereoPipeline] Complete: %d valid (%.1f%%) median_err=%.6f\n",
                 res.validPoints, res.coveragePercent, res.medianTriError);
    Q_UNUSED(H1inv);
    return true;
}

} // namespace

bool runOriginalDepthPath(const cv::Mat &grayL,
                          const cv::Mat &grayR,
                          const Camera &leftCamera,
                          const Camera &rightCamera,
                          const std::string &outputDir,
                          const StereoPipelineConfig &config,
                          StereoPipelineResult &res,
                          StereoDenseCloudPipeline *owner)
{
    QDir().mkpath(QString::fromStdString(outputDir));

    const Camera &pdmL = leftCamera;
    const Camera &pdmR = rightCamera;

    auto C1 = leftCamera.cameraCenter();
    auto C2 = rightCamera.cameraCenter();
    double baseline = std::sqrt(
        (C2[0]-C1[0])*(C2[0]-C1[0]) +
        (C2[1]-C1[1])*(C2[1]-C1[1]) +
        (C2[2]-C1[2])*(C2[2]-C1[2]));

    auto R1 = leftCamera.cameraToWorldRotation();
    auto R2 = rightCamera.cameraToWorldRotation();
    double look1[3] = {R1[2], R1[5], R1[8]};
    double look2[3] = {R2[2], R2[5], R2[8]};
    double w0[3] = {C1[0]-C2[0], C1[1]-C2[1], C1[2]-C2[2]};
    double a = look1[0]*look1[0] + look1[1]*look1[1] + look1[2]*look1[2];
    double b = look1[0]*look2[0] + look1[1]*look2[1] + look1[2]*look2[2];
    double cc = look2[0]*look2[0] + look2[1]*look2[1] + look2[2]*look2[2];
    double dd = look1[0]*w0[0] + look1[1]*w0[1] + look1[2]*w0[2];
    double ee = look2[0]*w0[0] + look2[1]*w0[1] + look2[2]*w0[2];
    double den = a * cc - b * b;
    double sceneDepth = baseline * 10.0;
    if (std::abs(den) > 1e-12)
    {
        double s1 = (b*ee - cc*dd) / den;
        double s2 = (a*ee - b*dd) / den;
        double depth = (s1 + s2) * 0.5;
        if (depth > 0)
        {
            sceneDepth = depth;
        }
    }

    float zNear = static_cast<float>(sceneDepth * config.depthRange.nearScale);
    float zFar = static_cast<float>(sceneDepth * config.depthRange.farScale);
    std::fprintf(stderr, "[StereoPipeline] Baseline=%.4f sceneDepth=%.4f zNear=%.4f zFar=%.4f\n",
                 baseline, sceneDepth, zNear, zFar);

    emit owner->progressChanged("PatchMatch L→R", 0.1f);
    PatchMatchConfig pmCfg = config.patchMatch;
    pmCfg.numSourceViews = 1;

    cv::Mat depthMap, confMap;
    std::string pmErr;
    if (!PatchMatchDepthEstimator::estimate(grayL, {grayR}, pdmL, {pdmR}, zNear, zFar, pmCfg,
                                            depthMap, &confMap, &pmErr))
    {
        res.errorMsg = "PatchMatch failed: " + pmErr;
        return false;
    }

    emit owner->progressChanged("PatchMatch R→L", 0.3f);
    cv::Mat depthMapR;
    if (config.filters.enableLeftRightDepthCheck &&
        PatchMatchDepthEstimator::estimate(grayR, {grayL}, pdmR, {pdmL}, zNear, zFar, pmCfg,
                                           depthMapR, nullptr, &pmErr))
    {
        int lrRejected = 0, lrOob = 0, lrNoR = 0, lrMismatch = 0;
        for (int r = 0; r < depthMap.rows; ++r)
        {
            for (int c = 0; c < depthMap.cols; ++c)
            {
                float dL = depthMap.at<float>(r, c);
                if (dL <= 0) continue;
                const double pixelLeft[2] = {
                    static_cast<double>(c),
                    static_cast<double>(r)
                };
                double world[3] = {};
                if (!pdmL.unprojectPixel(pixelLeft, static_cast<double>(dL), world))
                {
                    depthMap.at<float>(r, c) = 0.0f;
                    ++lrRejected;
                    ++lrOob;
                    continue;
                }
                double pixelRight[2] = {};
                double rightDepth = 0.0;
                if (!pdmR.projectWorldPointWithDepth(world, pixelRight, rightDepth))
                {
                    depthMap.at<float>(r, c) = 0.0f;
                    ++lrRejected; ++lrOob;
                    continue;
                }
                int cR = static_cast<int>(std::round(pixelRight[0]));
                int rR = static_cast<int>(std::round(pixelRight[1]));
                if (cR < 0 || cR >= depthMapR.cols || rR < 0 || rR >= depthMapR.rows)
                {
                    depthMap.at<float>(r, c) = 0.0f;
                    ++lrRejected; ++lrOob;
                    continue;
                }
                float dR = depthMapR.at<float>(rR, cR);
                if (dR <= 0)
                {
                    depthMap.at<float>(r, c) = 0.0f;
                    ++lrRejected; ++lrNoR;
                    continue;
                }
                if (std::abs(dL - dR) / dL > config.filters.leftRightDepthRatio)
                {
                    depthMap.at<float>(r, c) = 0.0f;
                    ++lrRejected; ++lrMismatch;
                }
            }
        }
        std::fprintf(stderr, "[StereoPipeline] L-R consistency: rejected %d (oob=%d noR=%d mismatch=%d)\n",
                     lrRejected, lrOob, lrNoR, lrMismatch);
        res.leftRightRejected = lrRejected;
        res.leftRightRejectedOob = lrOob;
        res.leftRightRejectedNoReverse = lrNoR;
        res.leftRightRejectedMismatch = lrMismatch;
    }

    res.depthValidBeforeFiltering = cv::countNonZero(depthMap > 0);
    std::fprintf(stderr, "[StereoPipeline] Depth: %dx%d valid=%d (%.1f%%)\n",
                 depthMap.cols, depthMap.rows, res.depthValidBeforeFiltering,
                 100.0 * res.depthValidBeforeFiltering / (depthMap.rows * depthMap.cols));

    emit owner->progressChanged("Depth filtering", 0.5f);
    cv::Mat validMask(depthMap.size(), CV_8U, cv::Scalar(0));
    for (int r = 0; r < depthMap.rows; ++r)
        for (int c = 0; c < depthMap.cols; ++c)
            if (depthMap.at<float>(r, c) > 0.0f)
                validMask.at<uint8_t>(r, c) = 255;

    if (!confMap.empty() && config.patchMatch.confidenceThresh > 0.0f)
    {
        for (int r = 0; r < depthMap.rows; ++r)
            for (int c = 0; c < depthMap.cols; ++c)
                if (validMask.at<uint8_t>(r, c) != 0 && confMap.at<float>(r, c) < config.patchMatch.confidenceThresh)
                {
                    validMask.at<uint8_t>(r, c) = 0;
                    depthMap.at<float>(r, c) = 0.0f;
                }
    }

    if (config.disparityFilter.medianFilterSize >= 3)
    {
        int kSize = config.disparityFilter.medianFilterSize | 1;
        cv::Mat filtered;
        cv::medianBlur(depthMap, filtered, kSize);
        filtered.copyTo(depthMap, validMask);
    }

    if (config.filters.enableLocalDepthConsistency)
    {
        int localRejected = 0;
        const int radius = std::max(1, config.filters.localWindowRadius);
        cv::Mat depthCopy = depthMap.clone();
        for (int r = radius; r < depthMap.rows - radius; ++r)
        {
            for (int c = radius; c < depthMap.cols - radius; ++c)
            {
                if (validMask.at<uint8_t>(r, c) == 0) continue;
                float d = depthCopy.at<float>(r, c);
                std::vector<float> neighbors;
                neighbors.reserve(static_cast<size_t>((radius * 2 + 1) * (radius * 2 + 1)));
                for (int dr = -radius; dr <= radius; ++dr)
                    for (int dc = -radius; dc <= radius; ++dc)
                    {
                        float nd = depthCopy.at<float>(r + dr, c + dc);
                        if (nd > 0) neighbors.push_back(nd);
                    }
                if (static_cast<int>(neighbors.size()) < config.filters.localMinNeighbors)
                {
                    validMask.at<uint8_t>(r, c) = 0;
                    depthMap.at<float>(r, c) = 0.0f;
                    ++localRejected;
                    continue;
                }
                std::nth_element(neighbors.begin(), neighbors.begin() + neighbors.size() / 2, neighbors.end());
                const float median = neighbors[neighbors.size() / 2];
                if (median <= 0.0f || std::abs(d - median) / median > config.filters.localDepthRatio)
                {
                    validMask.at<uint8_t>(r, c) = 0;
                    depthMap.at<float>(r, c) = 0.0f;
                    ++localRejected;
                }
            }
        }
        std::fprintf(stderr, "[StereoPipeline] Local consistency rejected %d\n", localRejected);
        res.localRejected = localRejected;
    }

    if (config.filters.enableIqrFilter)
    {
        std::vector<float> depths;
        depths.reserve(depthMap.rows * depthMap.cols / 4);
        for (int r = 0; r < depthMap.rows; ++r)
            for (int c = 0; c < depthMap.cols; ++c)
                if (validMask.at<uint8_t>(r, c)) depths.push_back(depthMap.at<float>(r, c));
        if (depths.size() > 100)
        {
            std::sort(depths.begin(), depths.end());
            float q1 = depths[depths.size() / 4];
            float q3 = depths[depths.size() * 3 / 4];
            float iqr = q3 - q1;
            float lo = q1 - config.filters.iqrMultiplier * iqr;
            float hi = q3 + config.filters.iqrMultiplier * iqr;
            int iqrRejected = 0;
            for (int r = 0; r < depthMap.rows; ++r)
                for (int c = 0; c < depthMap.cols; ++c)
                    if (validMask.at<uint8_t>(r, c) != 0)
                    {
                        float d = depthMap.at<float>(r, c);
                        if (d < lo || d > hi)
                        {
                            validMask.at<uint8_t>(r, c) = 0;
                            depthMap.at<float>(r, c) = 0.0f;
                            ++iqrRejected;
                        }
                    }
            std::fprintf(stderr, "[StereoPipeline] IQR filter [%.4f, %.4f]: rejected %d\n", lo, hi, iqrRejected);
            res.iqrRejected = iqrRejected;
        }
    }

    res.validAfterFiltering = cv::countNonZero(validMask);
    std::fprintf(stderr, "[StereoPipeline] After filter: %d valid (%.1f%%)\n",
                 res.validAfterFiltering, 100.0 * res.validAfterFiltering / (depthMap.rows * depthMap.cols));

    emit owner->progressChanged("Triangulation", 0.6f);
    TriangulationResult triResult = DisparityTriangulator::triangulateFromDepth(
        depthMap, validMask, cv::Mat::eye(3, 3, CV_64F), leftCamera, rightCamera, pdmL, config.triangulation);

    return writeOutputs(grayL, leftCamera, rightCamera, pdmL, outputDir, config, res, triResult,
                        cv::Mat::eye(3, 3, CV_64F), owner);
}

bool runRectifiedDisparityPath(const cv::Mat &grayL,
                               const cv::Mat &grayR,
                               const Camera &leftCamera,
                               const Camera &rightCamera,
                               const std::string &outputDir,
                               const StereoPipelineConfig &config,
                               StereoPipelineResult &res,
                               StereoDenseCloudPipeline *owner)
{
    emit owner->progressChanged("Rectified preprocessing", 0.05f);

    EpipolarRectifier::RectifiedPair rect;
    std::string rectErr;
    if (!EpipolarRectifier::rectify(grayL, grayR,
                                    leftCamera,
                                    rightCamera,
                                    rect, &rectErr))
    {
        res.errorMsg = "Rectification failed: " + rectErr;
        return false;
    }

    auto leftRectCam = rect.rectCamLeft;
    auto rightRectCam = rect.rectCamRight;
    const cv::Mat &rectLeftImage = rect.rectLeft;
    const cv::Mat &rectRightImage = rect.rectRight;

    auto C1 = leftCamera.cameraCenter();
    auto C2 = rightCamera.cameraCenter();
    double baseline = std::sqrt(
        (C2[0]-C1[0])*(C2[0]-C1[0]) +
        (C2[1]-C1[1])*(C2[1]-C1[1]) +
        (C2[2]-C1[2])*(C2[2]-C1[2]));
    double avgDepth = baseline * 10.0;
    float zNear = static_cast<float>(avgDepth * config.depthRange.nearScale);
    float zFar = static_cast<float>(avgDepth * config.depthRange.farScale);

    emit owner->progressChanged("Rectified PatchMatch L→R", 0.15f);
    PatchMatchConfig pmCfg = config.patchMatch;
    pmCfg.numSourceViews = 1;
    pmCfg.epipolarRectified = true;

    cv::Mat depthLeft, confLeft;
    std::string pmErr;
    if (!PatchMatchDepthEstimator::estimate(rectLeftImage, {rectRightImage}, leftRectCam, {rightRectCam},
                                            zNear, zFar, pmCfg, depthLeft, &confLeft, &pmErr))
    {
        res.errorMsg = "Rectified PatchMatch failed: " + pmErr;
        return false;
    }

    emit owner->progressChanged("Depth filtering", 0.30f);
    res.depthValidBeforeFiltering = cv::countNonZero(depthLeft > 0);
    std::fprintf(stderr, "[StereoPipeline] Rectified depth: %dx%d valid=%d (%.1f%%)\n",
                 depthLeft.cols, depthLeft.rows,
                 res.depthValidBeforeFiltering,
                 100.0 * res.depthValidBeforeFiltering / (depthLeft.rows * depthLeft.cols));

    cv::Mat validMask(depthLeft.size(), CV_8U, cv::Scalar(0));
    for (int r = 0; r < depthLeft.rows; ++r)
    {
        for (int c = 0; c < depthLeft.cols; ++c)
        {
            if (depthLeft.at<float>(r, c) > 0.0f)
            {
                validMask.at<uint8_t>(r, c) = 255;
            }
        }
    }

    if (!confLeft.empty() && config.patchMatch.confidenceThresh > 0.0f)
    {
        for (int r = 0; r < depthLeft.rows; ++r)
        {
            for (int c = 0; c < depthLeft.cols; ++c)
            {
                if (validMask.at<uint8_t>(r, c) != 0 && confLeft.at<float>(r, c) < config.patchMatch.confidenceThresh)
                {
                    validMask.at<uint8_t>(r, c) = 0;
                    depthLeft.at<float>(r, c) = 0.0f;
                }
            }
        }
    }

    if (config.disparityFilter.medianFilterSize >= 3)
    {
        int kSize = config.disparityFilter.medianFilterSize | 1;
        cv::Mat filtered;
        cv::medianBlur(depthLeft, filtered, kSize);
        filtered.copyTo(depthLeft, validMask);
    }

    if (config.filters.enableLocalDepthConsistency)
    {
        int localRejected = 0;
        const int radius = std::max(1, config.filters.localWindowRadius);
        cv::Mat depthCopy = depthLeft.clone();
        for (int r = radius; r < depthLeft.rows - radius; ++r)
        {
            for (int c = radius; c < depthLeft.cols - radius; ++c)
            {
                if (validMask.at<uint8_t>(r, c) == 0) continue;
                float d = depthCopy.at<float>(r, c);
                std::vector<float> neighbors;
                neighbors.reserve(static_cast<size_t>((radius * 2 + 1) * (radius * 2 + 1)));
                for (int dr = -radius; dr <= radius; ++dr)
                {
                    for (int dc = -radius; dc <= radius; ++dc)
                    {
                        float nd = depthCopy.at<float>(r + dr, c + dc);
                        if (nd > 0.0f)
                        {
                            neighbors.push_back(nd);
                        }
                    }
                }
                if (static_cast<int>(neighbors.size()) < config.filters.localMinNeighbors)
                {
                    validMask.at<uint8_t>(r, c) = 0;
                    depthLeft.at<float>(r, c) = 0.0f;
                    ++localRejected;
                    continue;
                }
                std::nth_element(neighbors.begin(), neighbors.begin() + neighbors.size() / 2, neighbors.end());
                const float median = neighbors[neighbors.size() / 2];
                if (median <= 0.0f || std::abs(d - median) / median > config.filters.localDepthRatio)
                {
                    validMask.at<uint8_t>(r, c) = 0;
                    depthLeft.at<float>(r, c) = 0.0f;
                    ++localRejected;
                }
            }
        }
        std::fprintf(stderr, "[StereoPipeline] Local consistency rejected %d\n", localRejected);
        res.localRejected = localRejected;
    }

    if (config.filters.enableIqrFilter)
    {
        std::vector<float> depths;
        depths.reserve(depthLeft.rows * depthLeft.cols / 4);
        for (int r = 0; r < depthLeft.rows; ++r)
        {
            for (int c = 0; c < depthLeft.cols; ++c)
            {
                if (validMask.at<uint8_t>(r, c) != 0)
                {
                    depths.push_back(depthLeft.at<float>(r, c));
                }
            }
        }
        if (depths.size() > 100)
        {
            std::sort(depths.begin(), depths.end());
            float q1 = depths[depths.size() / 4];
            float q3 = depths[depths.size() * 3 / 4];
            float iqr = q3 - q1;
            float lo = q1 - config.filters.iqrMultiplier * iqr;
            float hi = q3 + config.filters.iqrMultiplier * iqr;
            int iqrRejected = 0;
            for (int r = 0; r < depthLeft.rows; ++r)
            {
                for (int c = 0; c < depthLeft.cols; ++c)
                {
                    if (validMask.at<uint8_t>(r, c) != 0)
                    {
                        float d = depthLeft.at<float>(r, c);
                        if (d < lo || d > hi)
                        {
                            validMask.at<uint8_t>(r, c) = 0;
                            depthLeft.at<float>(r, c) = 0.0f;
                            ++iqrRejected;
                        }
                    }
                }
            }
            std::fprintf(stderr, "[StereoPipeline] IQR filter [%.4f, %.4f]: rejected %d\n", lo, hi, iqrRejected);
            res.iqrRejected = iqrRejected;
        }
    }

    res.validAfterFiltering = cv::countNonZero(validMask);
    std::fprintf(stderr, "[StereoPipeline] After filter: %d valid (%.1f%%)\n",
                 res.validAfterFiltering,
                 100.0 * res.validAfterFiltering / (depthLeft.rows * depthLeft.cols));

    emit owner->progressChanged("Triangulation", 0.65f);
    TriangulationConfig triCfg = config.triangulation;
    triCfg.transposed = rect.transposed;
    TriangulationResult triResult = DisparityTriangulator::triangulateFromDepth(
        depthLeft, validMask, rect.H1inv, leftCamera, rightCamera, leftRectCam, triCfg);

    return writeOutputs(rectLeftImage, leftCamera, rightCamera, leftRectCam, outputDir, config, res,
                        triResult, rect.H1inv, owner);

}

} // namespace mvs
} // namespace xjw
