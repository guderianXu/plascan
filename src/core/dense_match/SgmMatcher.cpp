// =============================================================================
// 文件: SgmMatcher.cpp
// 功能: 半全局/多全局立体匹配器 CPU 实现
// =============================================================================
#include "SgmMatcher.h"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw::dense_match
{

static const int SGM_DIRS[8][2] = {
    {1, 0}, {0, 1}, {1, 1}, {1, -1},
    {-1, 0}, {0, -1}, {-1, -1}, {-1, 1}
};

SgmMatcher::SgmMatcher(const DenseMatchConfig &cfg) : m_cfg(cfg)
{
}

void SgmMatcher::aggregatePath(CostVolume &L, const CostVolume &C,
                               int imgW, int imgH, int numDisp,
                               int dirX, int dirY) const
{
    int startY = (dirY > 0) ? 0 : imgH - 1;
    int endY   = (dirY > 0) ? imgH : -1;
    int stepY  = (dirY > 0) ? 1 : -1;
    int startX = (dirX > 0) ? 0 : imgW - 1;
    int endX   = (dirX > 0) ? imgW : -1;
    int stepX  = (dirX > 0) ? 1 : -1;

    float p1 = static_cast<float>(m_cfg.p1);
    float p2 = static_cast<float>(m_cfg.p2);

    for (int y = startY; y != endY; y += stepY)
    {
        for (int x = startX; x != endX; x += stepX)
        {
            int px = x - dirX;
            int py = y - dirY;

            if (px >= 0 && px < imgW && py >= 0 && py < imgH)
            {
                float minPrev = 1e20f;
                for (int d = 0; d < numDisp; ++d)
                {
                    float v = L[d].at<float>(py, px);
                    if (v < minPrev)
                    {
                        minPrev = v;
                    }
                }

                for (int d = 0; d < numDisp; ++d)
                {
                    float l0 = L[d].at<float>(py, px);
                    float l1 = (d > 0)
                        ? L[d - 1].at<float>(py, px) + p1 : 1e20f;
                    float l2 = (d < numDisp - 1)
                        ? L[d + 1].at<float>(py, px) + p1 : 1e20f;
                    float l3 = minPrev + p2;
                    float minL = std::min({l0, l1, l2, l3});
                    L[d].at<float>(y, x) +=
                        C[d].at<float>(y, x) + minL - minPrev;
                }
            }
            else
            {
                for (int d = 0; d < numDisp; ++d)
                {
                    L[d].at<float>(y, x) += C[d].at<float>(y, x);
                }
            }
        }
    }
}

DisparityResult SgmMatcher::compute(const cv::Mat &left, const cv::Mat &right)
{
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
    CV_Assert(left.size() == right.size());

    int imgW = left.cols;
    int imgH = left.rows;
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;

    if (numDisp <= 0)
    {
        DisparityResult empty;
        empty.disparity = cv::Mat();
        empty.confidence = cv::Mat();
        empty.validMask = cv::Mat();
        return empty;
    }

    CostVolume C;
#ifdef DM_ENABLE_CUDA
    if (m_cfg.useCuda)
    {
        C = computeCostVolumeCUDA(left, right,
            m_cfg.minDisparity, m_cfg.maxDisparity,
            m_cfg.corrKernelW, m_cfg.corrKernelH,
            m_cfg.costFunc, m_cfg.cudaDevice);
    }
    else
#endif
    {
        C = computeCostVolume(left, right,
            m_cfg.minDisparity, m_cfg.maxDisparity,
            m_cfg.corrKernelW, m_cfg.corrKernelH,
            m_cfg.costFunc, m_cfg.numThreads);
    }

    CostVolume L(numDisp);
    for (int d = 0; d < numDisp; ++d)
    {
        L[d] = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));
    }

    int numDirs = m_cfg.sgmDirections;
    if (numDirs < 1)
    {
        numDirs = 1;
    }
    if (numDirs > 8)
    {
        numDirs = 8;
    }

    for (int dir = 0; dir < numDirs; ++dir)
    {
        aggregatePath(L, C, imgW, imgH, numDisp,
                      SGM_DIRS[dir][0], SGM_DIRS[dir][1]);
    }

    DisparityResult result;
    result.disparity  = cv::Mat(imgH, imgW, CV_32FC1);
    result.confidence = cv::Mat(imgH, imgW, CV_32FC1);
    result.validMask  = cv::Mat(imgH, imgW, CV_8UC1);

#ifdef _OPENMP
    #pragma omp parallel for num_threads(m_cfg.numThreads)
#endif
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            float bestCost = 1e20f;
            float secondBest = 1e20f;
            int bestDisp = 0;

            for (int dIdx = 0; dIdx < numDisp; ++dIdx)
            {
                float c = L[dIdx].at<float>(y, x);
                if (c < bestCost)
                {
                    secondBest = bestCost;
                    bestCost = c;
                    bestDisp = m_cfg.minDisparity + dIdx;
                }
                else if (c < secondBest)
                {
                    secondBest = c;
                }
            }

            result.disparity.at<float>(y, x) = static_cast<float>(bestDisp);
            result.confidence.at<float>(y, x) =
                (bestCost > 0) ? (secondBest - bestCost) / bestCost : 0.0f;
            result.validMask.at<uchar>(y, x) = 1;
        }
    }

    return result;
}

} // namespace xjw::dense_match
