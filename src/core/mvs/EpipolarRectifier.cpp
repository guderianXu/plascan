#include "EpipolarRectifier.h"

#include "OpenCvCompat.h"
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <cstdio>

namespace xjw
{
namespace mvs
{

static cv::Mat buildK(const PositiveDepthCameraModel &cam)
{
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = cam.fx;
    K.at<double>(0, 2) = cam.cx;
    K.at<double>(1, 1) = cam.fy;
    K.at<double>(1, 2) = cam.cy;
    return K;
}

static cv::Mat buildRcw(const PositiveDepthCameraModel &cam)
{
    cv::Mat R(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i)
    {
        R.at<double>(i / 3, i % 3) = cam.R_cw[i];
    }
    return R;
}

static cv::Mat buildT(const PositiveDepthCameraModel &cam)
{
    cv::Mat t(3, 1, CV_64F);
    t.at<double>(0) = cam.T[0];
    t.at<double>(1) = cam.T[1];
    t.at<double>(2) = cam.T[2];
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

static PositiveDepthCameraModel buildRectifiedCamera(const PositiveDepthCameraModel &source,
                                                     const cv::Mat &rectRotation,
                                                     const cv::Mat &projection)
{
    PositiveDepthCameraModel rectified = source;

    rectified.fx = static_cast<float>(projection.at<double>(0, 0));
    rectified.fy = static_cast<float>(projection.at<double>(1, 1));
    rectified.cx = static_cast<float>(projection.at<double>(0, 2));
    rectified.cy = static_cast<float>(projection.at<double>(1, 2));

    const cv::Mat sourceRcw = buildRcw(source);
    const cv::Mat sourceT = buildT(source);
    const cv::Mat rectifiedRcw = rectRotation * sourceRcw;
    const cv::Mat rectifiedT = rectRotation * sourceT;

    for (int i = 0; i < 9; ++i)
    {
        rectified.R_cw[i] = static_cast<float>(rectifiedRcw.at<double>(i / 3, i % 3));
    }

    for (int i = 0; i < 3; ++i)
    {
        rectified.T[i] = static_cast<float>(rectifiedT.at<double>(i));
        rectified.C[i] = source.C[i];
    }

    return rectified;
}

bool EpipolarRectifier::rectify(
    const cv::Mat &imgLeft,
    const cv::Mat &imgRight,
    const PositiveDepthCameraModel &camLeft,
    const PositiveDepthCameraModel &camRight,
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
                      PLASCAN_OPENCV_ZERO_DISPARITY,
                      -1,
                      imageSize,
                      &roi1,
                      &roi2);

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

        PositiveDepthCameraModel transposedLeft = result.rectCamLeft;
        transposedLeft.fx = result.rectCamLeft.fy;
        transposedLeft.fy = result.rectCamLeft.fx;
        transposedLeft.cx = result.rectCamLeft.cy;
        transposedLeft.cy = result.rectCamLeft.cx;
        for (int c = 0; c < 3; ++c)
        {
            std::swap(transposedLeft.R_cw[c], transposedLeft.R_cw[3 + c]);
        }
        std::swap(transposedLeft.T[0], transposedLeft.T[1]);
        result.rectCamLeft = transposedLeft;

        PositiveDepthCameraModel transposedRight = result.rectCamRight;
        transposedRight.fx = result.rectCamRight.fy;
        transposedRight.fy = result.rectCamRight.fx;
        transposedRight.cx = result.rectCamRight.cy;
        transposedRight.cy = result.rectCamRight.cx;
        for (int c = 0; c < 3; ++c)
        {
            std::swap(transposedRight.R_cw[c], transposedRight.R_cw[3 + c]);
        }
        std::swap(transposedRight.T[0], transposedRight.T[1]);
        result.rectCamRight = transposedRight;
    }

    std::fprintf(stderr,
                 "[MVS] Stereo rectification: fxL=%.2f fxR=%.2f cxL=%.2f cxR=%.2f tx=%.6f p2_03=%.2f roiL=%dx%d roiR=%dx%d transposed=%d\n",
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
    int origW, int origH)
{
    if (rectifiedDepth.empty() || pair.H1inv.empty())
        return cv::Mat();

    const int rW = rectifiedDepth.cols;
    const int rH = rectifiedDepth.rows;
    cv::Mat result(origH, origW, CV_32FC1, cv::Scalar(0.0f));

    const double *h = pair.H1.ptr<double>(0);
    const double h00 = h[0], h01 = h[1], h02 = h[2];
    const double h10 = h[3], h11 = h[4], h12 = h[5];
    const double h20 = h[6], h21 = h[7], h22 = h[8];

    for (int row = 0; row < origH; ++row)
    {
        for (int col = 0; col < origW; ++col)
        {
            double w = h20 * col + h21 * row + h22;
            if (std::abs(w) < 1e-12) continue;
            double rx = (h00 * col + h01 * row + h02) / w;
            double ry = (h10 * col + h11 * row + h12) / w;

            int srcCol = static_cast<int>(std::round(rx));
            int srcRow = static_cast<int>(std::round(ry));
            if (srcCol < 0 || srcCol >= rW || srcRow < 0 || srcRow >= rH)
                continue;

            float d = rectifiedDepth.at<float>(srcRow, srcCol);
            if (d > 0.0f)
                result.at<float>(row, col) = d;
        }
    }
    return result;
}

} // namespace mvs
} // namespace xjw
