// =============================================================================
// 文件: BlockMatcher.cpp
// 功能: WTA块匹配器 CPU 实现
// =============================================================================
#include "BlockMatcher.h"
#include "CostFunctions.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw::dense_match
{

BlockMatcher::BlockMatcher(const DenseMatchConfig &cfg) : m_cfg(cfg)
{
}

DisparityResult BlockMatcher::compute(const cv::Mat &left, const cv::Mat &right)
{
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
    CV_Assert(left.size() == right.size());

    int imgW = left.cols, imgH = left.rows;
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;

    CostVolume volume;
#ifdef DM_ENABLE_CUDA
    if (m_cfg.useCuda)
    {
        volume = computeCostVolumeCUDA(left, right,
            m_cfg.minDisparity, m_cfg.maxDisparity,
            m_cfg.corrKernelW, m_cfg.corrKernelH,
            m_cfg.costFunc, m_cfg.cudaDevice);
    }
    else
#endif
    {
        volume = computeCostVolume(left, right,
            m_cfg.minDisparity, m_cfg.maxDisparity,
            m_cfg.corrKernelW, m_cfg.corrKernelH,
            m_cfg.costFunc, m_cfg.numThreads);
    }

    DisparityResult result;
    result.disparity  = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));
    result.confidence = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));
    result.validMask  = cv::Mat(imgH, imgW, CV_8UC1, cv::Scalar(0));

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(m_cfg.numThreads)
    #endif
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            float bestCost = 1e20f, secondBest = 1e20f;
            int bestDisp = 0;
            for (int dIdx = 0; dIdx < numDisp; ++dIdx)
            {
                float c = volume[dIdx].at<float>(y, x);
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
            if (bestCost > 0)
            {
                result.confidence.at<float>(y, x) = (secondBest - bestCost) / bestCost;
            }
            result.validMask.at<uchar>(y, x) = 1;
        }
    }
    return result;
}

} // namespace xjw::dense_match
