#include "SubpixelRefiner.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>

namespace xjw
{
namespace mvs
{

static float computeNCC(const cv::Mat &left, const cv::Mat &right,
                        int row, int colL, int colR, int halfK)
{
    const int rows = left.rows;
    const int cols = left.cols;
    double sumL = 0, sumR = 0, sumLL = 0, sumRR = 0, sumLR = 0;
    int count = 0;

    for (int dr = -halfK; dr <= halfK; ++dr)
    {
        int r = row + dr;
        if (r < 0 || r >= rows) continue;
        for (int dc = -halfK; dc <= halfK; ++dc)
        {
            int cL = colL + dc;
            int cR = colR + dc;
            if (cL < 0 || cL >= cols || cR < 0 || cR >= cols) continue;
            float vL = left.at<float>(r, cL);
            float vR = right.at<float>(r, cR);
            sumL += vL;
            sumR += vR;
            sumLL += vL * vL;
            sumRR += vR * vR;
            sumLR += vL * vR;
            ++count;
        }
    }

    if (count < 4) return -1.0f;
    double meanL = sumL / count;
    double meanR = sumR / count;
    double varL = sumLL / count - meanL * meanL;
    double varR = sumRR / count - meanR * meanR;
    double cov = sumLR / count - meanL * meanR;
    double denom = std::sqrt(varL * varR);
    if (denom < 1e-12) return -1.0f;
    return static_cast<float>(cov / denom);
}

void SubpixelRefiner::refine(cv::Mat &disparity,
                             const cv::Mat &leftImg,
                             const cv::Mat &rightImg,
                             const SubpixelConfig &cfg)
{
    if (cfg.mode == 0 || disparity.empty()) return;

    cv::Mat leftF, rightF;
    if (leftImg.type() == CV_32F)
        leftF = leftImg;
    else
        leftImg.convertTo(leftF, CV_32F);
    if (rightImg.type() == CV_32F)
        rightF = rightImg;
    else
        rightImg.convertTo(rightF, CV_32F);

    const int halfK = cfg.kernelSize / 2;
    const int rows = disparity.rows;
    const int cols = disparity.cols;
    int refined = 0;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            float d = disparity.at<float>(r, c);
            if (!std::isfinite(d) || d == 0.0f) continue;

            int dInt = static_cast<int>(std::round(d));
            float costM = computeNCC(leftF, rightF, r, c, c - dInt, halfK);
            float costP = computeNCC(leftF, rightF, r, c, c - (dInt + 1), halfK);
            float costN = computeNCC(leftF, rightF, r, c, c - (dInt - 1), halfK);

            if (costM < 0 || costP < 0 || costN < 0) continue;

            // NCC is similarity (higher=better), convert to cost (lower=better)
            float cM = 1.0f - costM;
            float cP = 1.0f - costP;
            float cN = 1.0f - costN;

            float denom = cP - 2.0f * cM + cN;
            if (std::abs(denom) < 1e-6f) continue;

            float offset = 0.5f * (cN - cP) / denom;
            if (std::abs(offset) > 0.5f) continue;

            disparity.at<float>(r, c) = static_cast<float>(dInt) + offset;
            ++refined;
        }
    }

    fprintf(stderr, "[SubpixelRefiner] refined %d / %d pixels\n", refined, rows * cols);
}

} // namespace mvs
} // namespace xjw
