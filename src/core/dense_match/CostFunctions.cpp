// =============================================================================
// 文件: CostFunctions.cpp
// 功能: 密集匹配代价函数 CPU 实现
// =============================================================================
#include "CostFunctions.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace xjw::dense_match
{

namespace
{

std::size_t checkedSizeProduct(
    std::size_t left,
    std::size_t right,
    const char *description)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(
            std::string("Dense cost-volume ") + description + " exceeds size_t");
    }
    return left * right;
}

} // namespace

CostVolumeBufferLayout checkedCostVolumeBufferLayout(
    int imageWidth,
    int imageHeight,
    int minDisparity,
    int maxDisparity)
{
    if (imageWidth <= 0 || imageHeight <= 0)
    {
        throw std::invalid_argument("Dense cost-volume image dimensions must be positive");
    }

    const std::int64_t disparityCount = static_cast<std::int64_t>(maxDisparity)
        - static_cast<std::int64_t>(minDisparity);
    if (disparityCount <= 0)
    {
        throw std::invalid_argument(
            "Dense cost-volume disparity range must be non-empty and half-open");
    }
    if (disparityCount > std::numeric_limits<int>::max())
    {
        throw std::overflow_error(
            "Dense cost-volume disparity count cannot be represented by int");
    }

    CostVolumeBufferLayout layout;
    layout.numDisparities = static_cast<int>(disparityCount);
    layout.planeElementCount = checkedSizeProduct(
        static_cast<std::size_t>(imageWidth),
        static_cast<std::size_t>(imageHeight),
        "image element count");
    layout.imageBytes = checkedSizeProduct(
        layout.planeElementCount,
        sizeof(uchar),
        "image byte count");
    layout.planeBytes = checkedSizeProduct(
        layout.planeElementCount,
        sizeof(float),
        "cost-plane byte count");
    layout.volumeElementCount = checkedSizeProduct(
        layout.planeElementCount,
        static_cast<std::size_t>(layout.numDisparities),
        "element count");
    layout.volumeBytes = checkedSizeProduct(
        layout.volumeElementCount,
        sizeof(float),
        "byte count");

    const std::vector<cv::Mat> emptyCostPlanes;
    if (static_cast<std::size_t>(layout.numDisparities) > emptyCostPlanes.max_size())
    {
        throw std::length_error(
            "Dense cost-volume disparity count exceeds the host plane-container limit");
    }
    return layout;
}

DisparityIndexRange validDisparityIndexRangeForLeftX(
    int leftX,
    int imageWidth,
    int minDisparity,
    int maxDisparity)
{
    const std::int64_t disparityCount = static_cast<std::int64_t>(maxDisparity)
        - static_cast<std::int64_t>(minDisparity);
    if (leftX < 0 || leftX >= imageWidth || imageWidth <= 0
        || disparityCount <= 0
        || disparityCount > std::numeric_limits<int>::max())
    {
        return {};
    }

    // 0 <= leftX - disparity < imageWidth
    const std::int64_t validMinDisparity = std::max(
        static_cast<std::int64_t>(minDisparity),
        static_cast<std::int64_t>(leftX) - imageWidth + 1);
    const std::int64_t validMaxDisparity = std::min(
        static_cast<std::int64_t>(maxDisparity),
        static_cast<std::int64_t>(leftX) + 1);
    if (validMinDisparity >= validMaxDisparity)
    {
        return {};
    }

    return {
        static_cast<int>(validMinDisparity - minDisparity),
        static_cast<int>(validMaxDisparity - minDisparity)};
}

CostVolume::CostVolume(int minDisparity, int maxDisparity, cv::Size imageSize)
    : _minDisparity(minDisparity),
      _maxDisparity(maxDisparity)
{
    if (maxDisparity <= minDisparity || imageSize.width <= 0 || imageSize.height <= 0)
    {
        return;
    }

    const CostVolumeBufferLayout layout = checkedCostVolumeBufferLayout(
        imageSize.width,
        imageSize.height,
        minDisparity,
        maxDisparity);
    const int numDisparities = layout.numDisparities;

    _costs.reserve(static_cast<std::size_t>(numDisparities));
    _hypothesisValidMasks.reserve(static_cast<std::size_t>(numDisparities));
    for (int disparityIndex = 0; disparityIndex < numDisparities; ++disparityIndex)
    {
        _costs.emplace_back(imageSize, CV_32FC1, cv::Scalar(kInvalidCost));
        _hypothesisValidMasks.emplace_back(imageSize, CV_8UC1, cv::Scalar(0));
    }
    _pixelValidMask = cv::Mat(imageSize, CV_8UC1, cv::Scalar(0));

    for (int x = 0; x < imageSize.width; ++x)
    {
        const DisparityIndexRange range = validDisparityIndexRangeForLeftX(
            x,
            imageSize.width,
            minDisparity,
            maxDisparity);
        if (range.empty())
        {
            continue;
        }

        _pixelValidMask.col(x).setTo(1);
        for (int disparityIndex = range.begin; disparityIndex < range.end; ++disparityIndex)
        {
            _hypothesisValidMasks[static_cast<std::size_t>(disparityIndex)].col(x).setTo(1);
        }
    }
}

std::size_t CostVolume::size() const
{
    return _costs.size();
}

bool CostVolume::empty() const
{
    return _costs.empty();
}

int CostVolume::minDisparity() const
{
    return _minDisparity;
}

int CostVolume::maxDisparity() const
{
    return _maxDisparity;
}

const cv::Mat &CostVolume::pixelValidMask() const
{
    return _pixelValidMask;
}

cv::Mat &CostVolume::operator[](std::size_t index)
{
    return _costs[index];
}

const cv::Mat &CostVolume::operator[](std::size_t index) const
{
    return _costs[index];
}

const cv::Mat &CostVolume::hypothesisValidMask(std::size_t index) const
{
    return _hypothesisValidMasks[index];
}

bool CostVolume::isValid(std::size_t index, int y, int x) const
{
    return index < _hypothesisValidMasks.size()
        && !_hypothesisValidMasks[index].empty()
        && _hypothesisValidMasks[index].at<uchar>(y, x) != 0;
}

BestDisparity selectBestDisparity(const CostVolume &volume, int y, int x)
{
    float bestCost = kInvalidCost;
    float secondBestCost = kInvalidCost;
    int bestIndex = -1;
    int candidateCount = 0;

    for (std::size_t disparityIndex = 0; disparityIndex < volume.size(); ++disparityIndex)
    {
        if (!volume.isValid(disparityIndex, y, x))
        {
            continue;
        }

        const float cost = volume[disparityIndex].at<float>(y, x);
        if (!std::isfinite(cost) || cost >= kInvalidCost)
        {
            continue;
        }
        ++candidateCount;

        if (cost < bestCost)
        {
            secondBestCost = bestCost;
            bestCost = cost;
            bestIndex = static_cast<int>(disparityIndex);
        }
        else if (cost < secondBestCost)
        {
            secondBestCost = cost;
        }
    }

    if (bestIndex < 0)
    {
        return {};
    }

    BestDisparity selection;
    selection.disparity = volume.minDisparity() + bestIndex;
    if (candidateCount == 1)
    {
        selection.confidence = 1.0f;
        selection.valid = true;
        return selection;
    }

    const float scale = std::max({1.0f, std::fabs(bestCost), std::fabs(secondBestCost)});
    const float margin = secondBestCost - bestCost;
    if (margin <= 1.0e-6f * scale)
    {
        return selection;
    }

    selection.confidence = std::clamp(
        margin / std::max(std::fabs(secondBestCost), 1.0e-6f),
        0.0f,
        1.0f);
    selection.valid = true;
    return selection;
}

namespace
{

// 1) Absolute Difference (AD)
static float adCost(const uchar *left, const uchar *right,
                    int x, int y, int d, int kw, int kh, int imgW, int imgH)
{
    float sum = 0.0f;
    int count = 0;
    int halfKW = kw / 2;
    int halfKH = kh / 2;

    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH)
        {
            continue;
        }
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            const int lx = x + dx;
            const int rx = lx - d;
            if (lx < 0 || lx >= imgW)
            {
                continue;
            }
            if (rx < 0 || rx >= imgW)
            {
                continue;
            }
            float diff = static_cast<float>(left[ry * imgW + lx])
                       - static_cast<float>(right[ry * imgW + rx]);
            sum += std::fabs(diff);
            ++count;
        }
    }

    return count > 0 ? sum / count : 0.0f;
}

// 2) Squared Difference (SD)
static float sdCost(const uchar *left, const uchar *right,
                    int x, int y, int d, int kw, int kh, int imgW, int imgH)
{
    float sum = 0.0f;
    int count = 0;
    int halfKW = kw / 2;
    int halfKH = kh / 2;

    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH)
        {
            continue;
        }
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            const int lx = x + dx;
            const int rx = lx - d;
            if (lx < 0 || lx >= imgW)
            {
                continue;
            }
            if (rx < 0 || rx >= imgW)
            {
                continue;
            }
            float diff = static_cast<float>(left[ry * imgW + lx])
                       - static_cast<float>(right[ry * imgW + rx]);
            sum += diff * diff;
            ++count;
        }
    }

    return count > 0 ? sum / count : 0.0f;
}

// 3) Normalized Cross-Correlation (NCC)  cost = 1 - NCC
static float nccCost(const uchar *left, const uchar *right,
                     int x, int y, int d, int kw, int kh, int imgW, int imgH)
{
    int halfKW = kw / 2;
    int halfKH = kh / 2;

    // First pass: compute mean of each window
    float meanL = 0.0f;
    float meanR = 0.0f;
    int count = 0;

    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH)
        {
            continue;
        }
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            const int lx = x + dx;
            const int rx = lx - d;
            if (lx < 0 || lx >= imgW)
            {
                continue;
            }
            if (rx < 0 || rx >= imgW)
            {
                continue;
            }
            meanL += static_cast<float>(left[ry * imgW + lx]);
            meanR += static_cast<float>(right[ry * imgW + rx]);
            ++count;
        }
    }

    if (count == 0)
    {
        return 0.0f;
    }
    meanL /= count;
    meanR /= count;

    // Second pass: compute variances and covariance
    float varL = 0.0f;
    float varR = 0.0f;
    float cov  = 0.0f;

    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH)
        {
            continue;
        }
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            const int lx = x + dx;
            const int rx = lx - d;
            if (lx < 0 || lx >= imgW)
            {
                continue;
            }
            if (rx < 0 || rx >= imgW)
            {
                continue;
            }
            float vl = static_cast<float>(left[ry * imgW + lx]) - meanL;
            float vr = static_cast<float>(right[ry * imgW + rx]) - meanR;
            varL += vl * vl;
            varR += vr * vr;
            cov  += vl * vr;
        }
    }

    // Handle flat patches (zero variance)
    float stdL = std::sqrt(varL);
    float stdR = std::sqrt(varR);
    const float eps = 1e-8f;

    if (stdL < eps && stdR < eps)
    {
        // Both windows are flat: identical means => cost 0, different => cost 2
        return std::fabs(meanL - meanR) < eps ? 0.0f : 2.0f;
    }
    if (stdL < eps || stdR < eps)
    {
        // One window is flat, the other is not => uncorrelated => cost 1
        return 1.0f;
    }

    float ncc = cov / (stdL * stdR);
    // Clamp to [-1, 1] to handle floating-point rounding
    ncc = std::max(-1.0f, std::min(1.0f, ncc));
    return 1.0f - ncc;
}

// 4) Census Transform (Hamming distance of binary descriptors)
static float censusCost(const uchar *left, const uchar *right,
                        int x, int y, int d, int kw, int kh, int imgW, int imgH)
{
    int halfKW = kw / 2;
    int halfKH = kh / 2;

    const int lxCenter = x;
    const int rxCenter = x - d;

    if (lxCenter < 0 || lxCenter >= imgW || rxCenter < 0 || rxCenter >= imgW
        || y < 0 || y >= imgH)
    {
        return kInvalidCost;
    }

    uchar lCenterVal = left[y * imgW + lxCenter];
    uchar rCenterVal = right[y * imgW + rxCenter];

    int hamming = 0;
    int total   = 0;

    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH)
        {
            continue;
        }
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }

            const int lx = x + dx;
            const int rx = lx - d;
            if (rx < 0 || rx >= imgW || lx < 0 || lx >= imgW)
            {
                continue;
            }

            int lBit = (left[ry * imgW + lx] > lCenterVal) ? 1 : 0;
            int rBit = (right[ry * imgW + rx] > rCenterVal) ? 1 : 0;
            if (lBit != rBit)
            {
                ++hamming;
            }
            ++total;
        }
    }

    return total > 0 ? static_cast<float>(hamming) / total : kInvalidCost;
}

// 5) Ternary Census Transform (tolerance τ = 5)
static float ternaryCensusCost(const uchar *left, const uchar *right,
                               int x, int y, int d, int kw, int kh,
                               int imgW, int imgH)
{
    const int tau = 5;
    int halfKW = kw / 2;
    int halfKH = kh / 2;

    const int lxCenter = x;
    const int rxCenter = x - d;

    if (lxCenter < 0 || lxCenter >= imgW || rxCenter < 0 || rxCenter >= imgW
        || y < 0 || y >= imgH)
    {
        return kInvalidCost;
    }

    int lCenterVal = static_cast<int>(left[y * imgW + lxCenter]);
    int rCenterVal = static_cast<int>(right[y * imgW + rxCenter]);

    int hamming   = 0;
    int validCount = 0;

    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH)
        {
            continue;
        }
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }

            const int lx = x + dx;
            const int rx = lx - d;
            if (rx < 0 || rx >= imgW || lx < 0 || lx >= imgW)
            {
                continue;
            }

            int lDiff = static_cast<int>(left[ry * imgW + lx]) - lCenterVal;
            int rDiff = static_cast<int>(right[ry * imgW + rx]) - rCenterVal;

            // Ternary state: 0 = less, 1 = greater, 2 = masked (within tau)
            int lState = (lDiff > tau) ? 1 : ((lDiff < -tau) ? 0 : 2);
            int rState = (rDiff > tau) ? 1 : ((rDiff < -tau) ? 0 : 2);

            if (lState != 2 && rState != 2)
            {
                ++validCount;
                if (lState != rState)
                {
                    ++hamming;
                }
            }
        }
    }

    return validCount > 0
        ? static_cast<float>(hamming) / validCount
        : kInvalidCost;
}

} // anonymous namespace

// Dispatcher: single-pixel cost
float computeCost(const uchar *left, const uchar *right,
                  int x, int y, int d, int kernelW, int kernelH,
                  int imgW, int imgH, CostFunction func)
{
    const int rightX = x - d;
    if (left == nullptr || right == nullptr
        || x < 0 || x >= imgW || rightX < 0 || rightX >= imgW
        || y < 0 || y >= imgH || kernelW <= 0 || kernelH <= 0)
    {
        return kInvalidCost;
    }

    switch (func)
    {
    case CostFunction::AbsoluteDifference:
        return adCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    case CostFunction::SquaredDifference:
        return sdCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    case CostFunction::NormalizedCrossCorr:
        return nccCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    case CostFunction::CensusTransform:
        return censusCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    case CostFunction::TernaryCensusTransform:
        return ternaryCensusCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    }
    return kInvalidCost;
}

// Full cost volume computation (CPU, OpenMP-parallel)
CostVolume computeCostVolume(const cv::Mat &left, const cv::Mat &right,
                             int minDisp, int maxDisp,
                             int kernelW, int kernelH,
                             CostFunction func, int numThreads)
{
    CV_Assert(!left.empty() && !right.empty());
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
    CV_Assert(left.size() == right.size());

    CV_Assert(maxDisp > minDisp);
    CV_Assert(kernelW > 0 && kernelH > 0);

    const int imgW = left.cols;
    const int imgH = left.rows;
    const int threadCount = std::max(1, numThreads);

    CostVolume volume(minDisp, maxDisp, left.size());
    const cv::Mat contiguousLeft = left.isContinuous() ? left : left.clone();
    const cv::Mat contiguousRight = right.isContinuous() ? right : right.clone();
    const uchar *lPtr = contiguousLeft.ptr<uchar>();
    const uchar *rPtr = contiguousRight.ptr<uchar>();

#if defined(_OPENMP) && _OPENMP >= 200805
    #pragma omp parallel for num_threads(threadCount) collapse(2)
#elif defined(_OPENMP)
    // OpenMP 2.0 (including MSVC's current frontend) has no collapse clause.
    // Parallelizing image rows preserves the same result without warning C4849.
    #pragma omp parallel for num_threads(threadCount)
#endif
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            const DisparityIndexRange range = validDisparityIndexRangeForLeftX(
                x,
                imgW,
                minDisp,
                maxDisp);
            for (int dIdx = range.begin; dIdx < range.end; ++dIdx)
            {
                const int disparity = minDisp + dIdx;
                volume[static_cast<std::size_t>(dIdx)].at<float>(y, x) = computeCost(
                    lPtr, rPtr, x, y, disparity,
                    kernelW, kernelH, imgW, imgH, func);
            }
        }
    }

    return volume;
}

} // namespace xjw::dense_match
