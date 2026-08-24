#include "EpipolarRectifier.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>

namespace xjw
{
namespace mvs
{

static cv::Mat buildK(const FramePinholeCamera &camera)
{
    const FramePinholeCamera::Intrinsics intrinsics = camera.intrinsics();
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = intrinsics.focalX;
    K.at<double>(0, 2) = intrinsics.principalX;
    K.at<double>(1, 1) = intrinsics.focalY;
    K.at<double>(1, 2) = intrinsics.principalY;
    return K;
}

static cv::Mat buildRcw(const FramePinholeCamera &camera)
{
    const std::array<double, 9> rotation = camera.worldToCameraRotation();
    cv::Mat R(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i)
    {
        R.at<double>(i / 3, i % 3) = rotation[i];
    }
    return R;
}

static cv::Mat buildT(const FramePinholeCamera &camera)
{
    const std::array<double, 3> translation = camera.worldToCameraTranslation();
    cv::Mat t(3, 1, CV_64F);
    t.at<double>(0) = translation[0];
    t.at<double>(1) = translation[1];
    t.at<double>(2) = translation[2];
    return t;
}

static std::array<double, 9> toArray9(const cv::Mat &matrix)
{
    std::array<double, 9> values{};
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            values[r * 3 + c] = matrix.at<double>(r, c);
        }
    }
    return values;
}

static bool hasLensDistortion(const FramePinholeCamera &camera)
{
    const FramePinholeCamera::Distortion distortion = camera.distortion();
    constexpr double epsilon = 1e-15;
    return std::fabs(distortion.radialK1) > epsilon
        || std::fabs(distortion.radialK2) > epsilon
        || std::fabs(distortion.radialK3) > epsilon
        || std::fabs(distortion.tangentialP1) > epsilon
        || std::fabs(distortion.tangentialP2) > epsilon;
}

static FramePinholeCamera cameraFromWorldToCamera(const FramePinholeCamera &source,
                                     const cv::Mat &rotationWorldToCamera,
                                     double focalX,
                                     double focalY,
                                     double principalX,
                                     double principalY)
{
    FramePinholeCamera result;
    result.setIntrinsics(focalX, focalY, principalX, principalY);
    result.setPixelPitch(source.pixelPitch());
    result.setPose(toArray9(rotationWorldToCamera.t()), source.cameraCenter());
    result.setAxisDirections(1, 1);
    result.setDepthAxisFlipped(false);
    result.setDistortion(FramePinholeCamera::Distortion{});
    return result;
}

static FramePinholeCamera buildRectifiedCamera(const FramePinholeCamera &source,
                                   const cv::Mat &rectRotation,
                                   const cv::Mat &projection)
{
    const cv::Mat sourceRcw = buildRcw(source);
    const cv::Mat rectifiedRcw = rectRotation * sourceRcw;
    return cameraFromWorldToCamera(source,
                                   rectifiedRcw,
                                   projection.at<double>(0, 0),
                                   projection.at<double>(1, 1),
                                   projection.at<double>(0, 2),
                                   projection.at<double>(1, 2));
}

static FramePinholeCamera transposeRectifiedCamera(const FramePinholeCamera &source)
{
    cv::Mat rotation = buildRcw(source);
    for (int column = 0; column < 3; ++column)
    {
        std::swap(rotation.at<double>(0, column), rotation.at<double>(1, column));
    }
    const FramePinholeCamera::Intrinsics intrinsics = source.intrinsics();
    return cameraFromWorldToCamera(source,
                                   rotation,
                                   intrinsics.focalY,
                                   intrinsics.focalX,
                                   intrinsics.principalY,
                                   intrinsics.principalX);
}

bool EpipolarRectifier::rectify(
    const cv::Mat &imgLeft,
    const cv::Mat &imgRight,
    const FramePinholeCamera &camLeft,
    const FramePinholeCamera &camRight,
    RectifiedPair &result,
    std::string *errorMsg)
{
    if (imgLeft.empty() || imgRight.empty())
    {
        if (errorMsg) *errorMsg = "Input images are empty";
        return false;
    }

    if (imgLeft.size() != imgRight.size())
    {
        if (errorMsg) *errorMsg = "Input image sizes do not match";
        return false;
    }
    if (!camLeft.isValid() || !camRight.isValid())
    {
        if (errorMsg) *errorMsg = "输入相机无效";
        return false;
    }
    if (hasLensDistortion(camLeft) || hasLensDistortion(camRight))
    {
        if (errorMsg) *errorMsg = "极线校正要求输入影像先完成镜头去畸变";
        return false;
    }
    if (camLeft.uAxisSign() != 1 || camLeft.vAxisSign() != 1 || camLeft.depthAxisFlipped()
        || camRight.uAxisSign() != 1 || camRight.vAxisSign() != 1 || camRight.depthAxisFlipped())
    {
        if (errorMsg) *errorMsg = "极线校正要求正深度规范化 FramePinholeCamera";
        return false;
    }

    result.origW = imgLeft.cols;
    result.origH = imgLeft.rows;
    result.transposed = false;

    const cv::Mat K1 = buildK(camLeft);
    const cv::Mat K2 = buildK(camRight);
    const cv::Mat Rcw1 = buildRcw(camLeft);
    const cv::Mat Rcw2 = buildRcw(camRight);
    const cv::Mat t1 = buildT(camLeft);
    const cv::Mat t2 = buildT(camRight);

    const cv::Mat R = Rcw2 * Rcw1.t();
    const cv::Mat T = t2 - R * t1;

    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;
    cv::Rect roi1;
    cv::Rect roi2;
    const cv::Mat zeroDist = cv::Mat::zeros(5, 1, CV_64F);
    const cv::Size imageSize(imgLeft.cols, imgLeft.rows);

    cv::stereoRectify(K1,
                      zeroDist,
                      K2,
                      zeroDist,
                      imageSize,
                      R,
                      T,
                      R1,
                      R2,
                      P1,
                      P2,
                      Q,
                      cv::STEREO_ZERO_DISPARITY,
                      -1,
                      imageSize,
                      &roi1,
                      &roi2);

    const cv::Rect common_roi = roi1 & roi2;
    const double image_area = static_cast<double>(imageSize.area());
    const double common_coverage = image_area > 0.0
        ? static_cast<double>(common_roi.area()) / image_area
        : 0.0;
    constexpr double minimum_common_coverage = 0.05;
    if (roi1.empty() || roi2.empty() || common_coverage < minimum_common_coverage)
    {
        if (errorMsg)
        {
            *errorMsg = "极线校正后的共同有效区域不足";
        }
        std::fprintf(stderr,
                     "[MVS] Stereo rectification rejected: roiL=%dx%d roiR=%dx%d "
                     "common=%.1f%%\n",
                     roi1.width,
                     roi1.height,
                     roi2.width,
                     roi2.height,
                     common_coverage * 100.0);
        return false;
    }

    const cv::Mat Krect1 = P1(cv::Rect(0, 0, 3, 3)).clone();
    const cv::Mat Krect2 = P2(cv::Rect(0, 0, 3, 3)).clone();

    result.H1 = Krect1 * R1 * K1.inv();
    result.H2 = Krect2 * R2 * K2.inv();
    result.H1inv = result.H1.inv();
    result.H2inv = result.H2.inv();

    cv::warpPerspective(imgLeft,
                        result.rectLeft,
                        result.H1,
                        imageSize,
                        cv::INTER_LINEAR);
    cv::warpPerspective(imgRight,
                        result.rectRight,
                        result.H2,
                        imageSize,
                        cv::INTER_LINEAR);

    result.rectCamLeft = buildRectifiedCamera(camLeft, R1, P1);
    result.rectCamRight = buildRectifiedCamera(camRight, R2, P2);

    const double txRect = P2.at<double>(0, 3);
    const double tyRect = P2.at<double>(1, 3);
    if (std::abs(tyRect) > std::abs(txRect))
    {
        result.transposed = true;

        cv::transpose(result.rectLeft, result.rectLeft);
        cv::transpose(result.rectRight, result.rectRight);

        const cv::Mat transposeH = (cv::Mat_<double>(3, 3) <<
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0);

        result.H1 = transposeH * result.H1;
        result.H2 = transposeH * result.H2;
        result.H1inv = result.H1.inv();
        result.H2inv = result.H2.inv();

        result.rectCamLeft = transposeRectifiedCamera(result.rectCamLeft);
        result.rectCamRight = transposeRectifiedCamera(result.rectCamRight);
    }

    std::fprintf(stderr,
                 "[MVS] Stereo rectification: fxL=%.2f fxR=%.2f cxL=%.2f cxR=%.2f "
                 "tx=%.6f p2_03=%.2f roiL=%dx%d roiR=%dx%d transposed=%d\n",
                 P1.at<double>(0, 0),
                 P2.at<double>(0, 0),
                 P1.at<double>(0, 2),
                 P2.at<double>(0, 2),
                 T.at<double>(0),
                 P2.at<double>(0, 3),
                 roi1.width,
                 roi1.height,
                 roi2.width,
                 roi2.height,
                 result.transposed ? 1 : 0);

    return true;
}

cv::Mat EpipolarRectifier::unrectifyDepth(
    const cv::Mat &rectifiedDepth,
    const RectifiedPair &pair,
    const FramePinholeCamera &originalReferenceCamera,
    int origW, int origH)
{
    const cv::Mat &reference_homography = pair.refIsRight ? pair.H2 : pair.H1;
    const FramePinholeCamera &rectified_reference_camera = pair.refIsRight
        ? pair.rectCamRight
        : pair.rectCamLeft;
    if (rectifiedDepth.empty() || rectifiedDepth.type() != CV_32FC1 ||
        reference_homography.empty() || reference_homography.rows != 3 ||
        reference_homography.cols != 3 || !originalReferenceCamera.isValid() ||
        !rectified_reference_camera.isValid() || origW <= 0 || origH <= 0)
    {
        return cv::Mat();
    }

    const int rW = rectifiedDepth.cols;
    const int rH = rectifiedDepth.rows;
    cv::Mat result(origH, origW, CV_32FC1, cv::Scalar(0.0f));

    cv::Mat homography;
    reference_homography.convertTo(homography, CV_64F);
    const double *h = homography.ptr<double>(0);
    const double h00 = h[0], h01 = h[1], h02 = h[2];
    const double h10 = h[3], h11 = h[4], h12 = h[5];
    const double h20 = h[6], h21 = h[7], h22 = h[8];

    for (int row = 0; row < origH; ++row)
    {
        for (int col = 0; col < origW; ++col)
        {
            const double w = h20 * col + h21 * row + h22;
            if (std::abs(w) < 1e-12)
            {
                continue;
            }
            const double rx = (h00 * col + h01 * row + h02) / w;
            const double ry = (h10 * col + h11 * row + h12) / w;

            const int srcCol = static_cast<int>(std::round(rx));
            const int srcRow = static_cast<int>(std::round(ry));
            if (srcCol < 0 || srcCol >= rW || srcRow < 0 || srcRow >= rH)
            {
                continue;
            }

            const float rectified_depth = rectifiedDepth.at<float>(srcRow, srcCol);
            if (!std::isfinite(rectified_depth) || rectified_depth <= 0.0f)
            {
                continue;
            }

            // A depth sample is axial Z in the rectified camera, not a ray
            // length and not axial Z in the original camera. Reconstruct the
            // point on the exact target ray and express it in the original
            // reference camera before storing it in the unrectified grid.
            const double rectified_pixel[2] = {rx, ry};
            double world[3] = {};
            if (!rectified_reference_camera.unprojectPixel(
                    rectified_pixel, rectified_depth, world))
            {
                continue;
            }
            const double original_depth = originalReferenceCamera.positiveDepth(world);
            if (std::isfinite(original_depth) && original_depth > 0.0 &&
                original_depth <= std::numeric_limits<float>::max())
            {
                result.at<float>(row, col) = static_cast<float>(original_depth);
            }
        }
    }
    return result;
}

bool EpipolarRectifier::rectifiedDepthRange(
    const FramePinholeCamera &originalReferenceCamera,
    const FramePinholeCamera &rectifiedReferenceCamera,
    int originalWidth,
    int originalHeight,
    float originalNear,
    float originalFar,
    float &rectifiedNear,
    float &rectifiedFar)
{
    if (!originalReferenceCamera.isValid() || !rectifiedReferenceCamera.isValid() ||
        hasLensDistortion(originalReferenceCamera) ||
        hasLensDistortion(rectifiedReferenceCamera) ||
        originalWidth <= 0 || originalHeight <= 0 ||
        !std::isfinite(originalNear) || !std::isfinite(originalFar) ||
        originalNear <= 0.0f || originalFar <= originalNear)
    {
        return false;
    }

    const std::array<double, 2> columns{
        0.0, static_cast<double>(std::max(0, originalWidth - 1))};
    const std::array<double, 2> rows{
        0.0, static_cast<double>(std::max(0, originalHeight - 1))};
    const std::array<double, 2> depths{
        static_cast<double>(originalNear), static_cast<double>(originalFar)};
    double minimum_rectified_depth = std::numeric_limits<double>::infinity();
    double maximum_rectified_depth = 0.0;

    // With zero-distortion pinhole cameras, rectified axial Z is affine over
    // the original image ray rectangle and the original depth interval. Its
    // extrema therefore occur at these eight corner/end-point combinations.
    for (const double row : rows)
    {
        for (const double column : columns)
        {
            const double pixel[2] = {column, row};
            for (const double depth : depths)
            {
                double world[3] = {};
                if (!originalReferenceCamera.unprojectPixel(pixel, depth, world))
                {
                    return false;
                }
                const double rectified_depth = rectifiedReferenceCamera.positiveDepth(world);
                if (!std::isfinite(rectified_depth) || rectified_depth <= 0.0)
                {
                    return false;
                }
                minimum_rectified_depth = std::min(
                    minimum_rectified_depth, rectified_depth);
                maximum_rectified_depth = std::max(
                    maximum_rectified_depth, rectified_depth);
            }
        }
    }

    if (!std::isfinite(minimum_rectified_depth) ||
        !std::isfinite(maximum_rectified_depth) ||
        maximum_rectified_depth <= minimum_rectified_depth ||
        maximum_rectified_depth > std::numeric_limits<float>::max())
    {
        return false;
    }

    rectifiedNear = std::nextafter(
        static_cast<float>(minimum_rectified_depth), 0.0f);
    rectifiedFar = std::nextafter(
        static_cast<float>(maximum_rectified_depth),
        std::numeric_limits<float>::infinity());
    return std::isfinite(rectifiedNear) && std::isfinite(rectifiedFar) &&
        rectifiedNear > 0.0f && rectifiedFar > rectifiedNear;
}

cv::Mat EpipolarRectifier::unrectifyNearest(
    const cv::Mat &rectifiedArtifact,
    const RectifiedPair &pair,
    int origW,
    int origH)
{
    const cv::Mat &reference_homography = pair.refIsRight ? pair.H2 : pair.H1;
    if (rectifiedArtifact.empty() || reference_homography.empty() || origW <= 0 || origH <= 0)
    {
        return cv::Mat();
    }

    cv::Mat homography;
    reference_homography.convertTo(homography, CV_64F);
    const double *values = homography.ptr<double>(0);
    cv::Mat result = cv::Mat::zeros(origH, origW, rectifiedArtifact.type());
    const std::size_t element_size = rectifiedArtifact.elemSize();

    for (int row = 0; row < origH; ++row)
    {
        for (int col = 0; col < origW; ++col)
        {
            const double scale = values[6] * col + values[7] * row + values[8];
            if (std::abs(scale) < 1.0e-12)
            {
                continue;
            }
            const int source_col = static_cast<int>(std::lround(
                (values[0] * col + values[1] * row + values[2]) / scale));
            const int source_row = static_cast<int>(std::lround(
                (values[3] * col + values[4] * row + values[5]) / scale));
            if (source_col < 0 || source_col >= rectifiedArtifact.cols ||
                source_row < 0 || source_row >= rectifiedArtifact.rows)
            {
                continue;
            }

            std::memcpy(result.ptr(row) + static_cast<std::size_t>(col) * element_size,
                        rectifiedArtifact.ptr(source_row) +
                            static_cast<std::size_t>(source_col) * element_size,
                        element_size);
        }
    }
    return result;
}

} // namespace mvs
} // namespace xjw
