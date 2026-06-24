#include "DenseMatchService.h"
#include "BlockMatcher.h"
#include "SgmMatcher.h"
#include "opencv/OpenCVSgbmWrapper.h"
#include "SubpixelRefiner.h"
#include "DisparityValidator.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <chrono>

namespace xjw::dense_match
{

DenseMatchService::DenseMatchService(const DenseMatchConfig &cfg) : _config(cfg) {}

DisparityResult DenseMatchService::process()
{
    _left  = cv::imread(_config.leftImagePath, cv::IMREAD_GRAYSCALE);
    _right = cv::imread(_config.rightImagePath, cv::IMREAD_GRAYSCALE);
    if (_left.empty() || _right.empty())
    {
        fprintf(stderr, "[DenseMatch] ERROR: failed to load %s or %s\n",
                _config.leftImagePath.c_str(), _config.rightImagePath.c_str());
        return DisparityResult();
    }
    return process(_left, _right);
}

DisparityResult DenseMatchService::process(const cv::Mat &left, const cv::Mat &right)
{
    if (left.size() != right.size() || left.type() != CV_8UC1 || right.type() != CV_8UC1)
    {
        return DisparityResult();
    }

    fprintf(stdout, "[DenseMatch] size=%dx%d algo=%d cost=%d disp=[%d,%d] kernel=%dx%d "
            "cuda=%d device=%d threads=%d\n",
            left.cols, left.rows,
            static_cast<int>(_config.algorithm), static_cast<int>(_config.costFunc),
            _config.minDisparity, _config.maxDisparity,
            _config.corrKernelW, _config.corrKernelH,
            _config.useCuda ? 1 : 0, _config.cudaDevice, _config.numThreads);

#ifdef DM_ENABLE_CUDA
    fprintf(stdout, "[DenseMatch] DM_ENABLE_CUDA=YES useCuda=%d\n", _config.useCuda ? 1 : 0);
#else
    fprintf(stdout, "[DenseMatch] DM_ENABLE_CUDA=NO (CPU only)\n");
#endif

    auto t0 = std::chrono::steady_clock::now();
    DisparityResult result = computeRawDisparity(left, right);

    auto t1 = std::chrono::steady_clock::now();
    fprintf(stdout, "[DenseMatch] matching: %.0f ms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count());

    DisparityValidator validator(_config);
    result = validator.validate(result.disparity, result.confidence);
    if (_config.enableLRCheck && _config.lrCheckThreshold > 0.0f && !result.disparity.empty())
    {
        DisparityResult reverse = computeRawDisparity(right, left);
        reverse = validator.validate(reverse.disparity, reverse.confidence);
        if (!reverse.disparity.empty() && reverse.disparity.size() == result.disparity.size())
        {
            const cv::Mat lrMask = validator.checkLRConsistency(result.disparity, reverse.disparity);
            if (result.validMask.empty())
            {
                result.validMask = lrMask;
            }
            else
            {
                cv::bitwise_and(result.validMask, lrMask, result.validMask);
            }
        }
    }
    validator.applyImageSupportMask(result, left, right);

    auto t2 = std::chrono::steady_clock::now();
    fprintf(stdout, "[DenseMatch] validate: %.0f ms, total: %.0f ms\n",
            std::chrono::duration<double, std::milli>(t2 - t1).count(),
            std::chrono::duration<double, std::milli>(t2 - t0).count());

    return result;
}

DisparityResult DenseMatchService::computeRawDisparity(const cv::Mat &left, const cv::Mat &right) const
{
    if (_config.algorithm == StereoAlgorithm::BlockMatch)
    {
        BlockMatcher bm(_config);
        return bm.compute(left, right);
    }
    if (_config.algorithm == StereoAlgorithm::OpenCV_SGBM)
    {
        OpenCVSgbmWrapper sgbm(_config);
        return sgbm.compute(left, right);
    }

    SgmMatcher sgm(_config);
    return sgm.compute(left, right);
}

bool DenseMatchService::saveDisparity(const DisparityResult &result,
                                      const std::string &filepath)
{
    if (result.disparity.empty())
    {
        return false;
    }
    return cv::imwrite(filepath, result.disparity);
}

} // namespace xjw::dense_match
