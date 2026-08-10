// =============================================================================
// 文件: BlockMatcher.cpp
// 功能: WTA块匹配器 CPU 实现
// =============================================================================
#include "BlockMatcher.h"
#include "CostFunctions.h"
#include "SubpixelRefiner.h"
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw::dense_match
{

BlockMatcher::BlockMatcher(const DenseMatchConfig &cfg) : _config(cfg)
{
}

DisparityResult BlockMatcher::compute(const cv::Mat &left, const cv::Mat &right)
{
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
    CV_Assert(left.size() == right.size());

    const int imgW = left.cols;
    const int imgH = left.rows;
    if (_config.maxDisparity <= _config.minDisparity || left.empty())
    {
        return {};
    }

    CostVolume volume;
#ifdef DM_ENABLE_CUDA
    if (_config.useCuda)
    {
        volume = computeCostVolumeCUDA(left, right,
            _config.minDisparity, _config.maxDisparity,
            _config.corrKernelW, _config.corrKernelH,
            _config.costFunc, _config.cudaDevice);
    }
    else
#endif
    {
        volume = computeCostVolume(left, right,
            _config.minDisparity, _config.maxDisparity,
            _config.corrKernelW, _config.corrKernelH,
            _config.costFunc, _config.numThreads);
    }

    DisparityResult result;
    result.disparity  = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));
    result.confidence = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));
    result.validMask  = cv::Mat(imgH, imgW, CV_8UC1, cv::Scalar(0));
    const int threadCount = std::max(1, _config.numThreads);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threadCount)
    #endif
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            const BestDisparity selection = selectBestDisparity(volume, y, x);
            if (!selection.valid)
            {
                continue;
            }

            result.disparity.at<float>(y, x) = static_cast<float>(selection.disparity);
            result.confidence.at<float>(y, x) = selection.confidence;
            result.validMask.at<uchar>(y, x) = 1;
        }
    }

    if (_config.subpixel != SubpixelMode::None)
    {
        SubpixelRefiner refiner(_config);
        result.disparity = refiner.refine(
            result.disparity,
            volume,
            _config.minDisparity,
            _config.maxDisparity,
            result.validMask);
    }
    return result;
}

} // namespace xjw::dense_match
