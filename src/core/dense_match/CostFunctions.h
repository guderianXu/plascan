// =============================================================================
// 文件: CostFunctions.h
// 功能: 密集匹配代价函数声明（CPU + CUDA）
// =============================================================================
#pragma once

#include "DenseMatchTypes.h"
#include <cstddef>
#include <opencv2/core.hpp>
#include <vector>

namespace xjw::dense_match
{

// Dense disparity is defined on the left/reference image:
//     disparity = x_left - x_right
// so a left pixel x samples the right image at x - disparity.  Search ranges
// are always [minDisparity, maxDisparity), with an inclusive lower bound and
// an exclusive upper bound.
inline constexpr float kInvalidCost = 1.0e20f;

struct DisparityIndexRange
{
    int begin = 0;
    int end = 0;

    [[nodiscard]] bool empty() const
    {
        return begin >= end;
    }
};

struct CostVolumeBufferLayout
{
    int numDisparities = 0;
    std::size_t planeElementCount = 0;
    std::size_t imageBytes = 0;
    std::size_t planeBytes = 0;
    std::size_t volumeElementCount = 0;
    std::size_t volumeBytes = 0;
};

// Validates the signed disparity span and every size_t multiplication needed
// by the CUDA cost-volume path without allocating image or device memory.
// Invalid dimensions raise std::invalid_argument; unrepresentable layouts
// raise std::overflow_error or std::length_error with a specific reason.
[[nodiscard]] CostVolumeBufferLayout checkedCostVolumeBufferLayout(
    int imageWidth,
    int imageHeight,
    int minDisparity,
    int maxDisparity);

// Returns the valid half-open range of disparity *indices* for a left-image
// column.  Index zero corresponds to minDisparity.
DisparityIndexRange validDisparityIndexRangeForLeftX(
    int leftX,
    int imageWidth,
    int minDisparity,
    int maxDisparity);

class CostVolume
{
public:
    CostVolume() = default;
    CostVolume(int minDisparity, int maxDisparity, cv::Size imageSize);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] int minDisparity() const;
    [[nodiscard]] int maxDisparity() const;
    [[nodiscard]] const cv::Mat &pixelValidMask() const;

    cv::Mat &operator[](std::size_t index);
    const cv::Mat &operator[](std::size_t index) const;
    const cv::Mat &hypothesisValidMask(std::size_t index) const;
    [[nodiscard]] bool isValid(std::size_t index, int y, int x) const;

    auto begin()
    {
        return _costs.begin();
    }

    auto end()
    {
        return _costs.end();
    }

    auto begin() const
    {
        return _costs.begin();
    }

    auto end() const
    {
        return _costs.end();
    }

private:
    int _minDisparity = 0;
    int _maxDisparity = 0;
    std::vector<cv::Mat> _costs;
    std::vector<cv::Mat> _hypothesisValidMasks;
    cv::Mat _pixelValidMask;
};

struct BestDisparity
{
    int disparity = 0;
    float confidence = 0.0f;
    bool valid = false;
};

BestDisparity selectBestDisparity(const CostVolume &volume, int y, int x);

float computeCost(const uchar *left, const uchar *right,
                  int x, int y, int d, int kernelW, int kernelH,
                  int imgW, int imgH, CostFunction func);

CostVolume computeCostVolume(const cv::Mat &left, const cv::Mat &right,
                             int minDisp, int maxDisp,
                             int kernelW, int kernelH,
                             CostFunction func, int numThreads = 1);

#ifdef DM_ENABLE_CUDA
bool isCostVolumeCUDAAvailable(int cudaDevice = 0);

CostVolume computeCostVolumeCUDA(const cv::Mat &left, const cv::Mat &right,
                                 int minDisp, int maxDisp,
                                 int kernelW, int kernelH,
                                 CostFunction func, int cudaDevice = 0);
#endif

} // namespace xjw::dense_match
