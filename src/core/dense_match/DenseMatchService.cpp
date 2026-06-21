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

DenseMatchService::DenseMatchService(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

DisparityResult DenseMatchService::process()
{
    m_left  = cv::imread(m_cfg.leftImagePath, cv::IMREAD_GRAYSCALE);
    m_right = cv::imread(m_cfg.rightImagePath, cv::IMREAD_GRAYSCALE);
    if (m_left.empty() || m_right.empty())
    {
        fprintf(stderr, "[DenseMatch] ERROR: failed to load %s or %s\n",
                m_cfg.leftImagePath.c_str(), m_cfg.rightImagePath.c_str());
        return DisparityResult();
    }
    return process(m_left, m_right);
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
            static_cast<int>(m_cfg.algorithm), static_cast<int>(m_cfg.costFunc),
            m_cfg.minDisparity, m_cfg.maxDisparity,
            m_cfg.corrKernelW, m_cfg.corrKernelH,
            m_cfg.useCuda ? 1 : 0, m_cfg.cudaDevice, m_cfg.numThreads);

#ifdef DM_ENABLE_CUDA
    fprintf(stdout, "[DenseMatch] DM_ENABLE_CUDA=YES useCuda=%d\n", m_cfg.useCuda ? 1 : 0);
#else
    fprintf(stdout, "[DenseMatch] DM_ENABLE_CUDA=NO (CPU only)\n");
#endif

    auto t0 = std::chrono::steady_clock::now();
    DisparityResult result = computeRawDisparity(left, right);

    auto t1 = std::chrono::steady_clock::now();
    fprintf(stdout, "[DenseMatch] matching: %.0f ms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count());

    DisparityValidator validator(m_cfg);
    result = validator.validate(result.disparity, result.confidence);
    if (m_cfg.enableLRCheck && m_cfg.lrCheckThreshold > 0.0f && !result.disparity.empty())
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
    if (m_cfg.algorithm == StereoAlgorithm::BlockMatch)
    {
        BlockMatcher bm(m_cfg);
        return bm.compute(left, right);
    }
    if (m_cfg.algorithm == StereoAlgorithm::OpenCV_SGBM)
    {
        OpenCVSgbmWrapper sgbm(m_cfg);
        return sgbm.compute(left, right);
    }

    SgmMatcher sgm(m_cfg);
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
