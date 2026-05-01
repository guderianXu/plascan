// =============================================================================
// 文件: CostFunctions.cpp
// 功能: 密集匹配代价函数 CPU 实现
// =============================================================================
#include "CostFunctions.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace xjw::dense_match
{

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
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW)
            {
                continue;
            }
            if (lx < 0 || lx >= imgW)
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
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW)
            {
                continue;
            }
            if (lx < 0 || lx >= imgW)
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
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW)
            {
                continue;
            }
            if (lx < 0 || lx >= imgW)
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
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW)
            {
                continue;
            }
            if (lx < 0 || lx >= imgW)
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

    int lxCenter = x + d;
    int rxCenter = x;

    if (lxCenter < 0 || lxCenter >= imgW || rxCenter < 0 || rxCenter >= imgW
        || y < 0 || y >= imgH)
    {
        return 0.0f;
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

            int rx = x + dx;
            int lx = rx + d;
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

    return total > 0 ? static_cast<float>(hamming) / total : 0.0f;
}

// 5) Ternary Census Transform (tolerance τ = 5)
static float ternaryCensusCost(const uchar *left, const uchar *right,
                               int x, int y, int d, int kw, int kh,
                               int imgW, int imgH)
{
    const int tau = 5;
    int halfKW = kw / 2;
    int halfKH = kh / 2;

    int lxCenter = x + d;
    int rxCenter = x;

    if (lxCenter < 0 || lxCenter >= imgW || rxCenter < 0 || rxCenter >= imgW
        || y < 0 || y >= imgH)
    {
        return 0.0f;
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

            int rx = x + dx;
            int lx = rx + d;
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

    return validCount > 0 ? static_cast<float>(hamming) / validCount : 0.0f;
}

} // anonymous namespace

// Dispatcher: single-pixel cost
float computeCost(const uchar *left, const uchar *right,
                  int x, int y, int d, int kernelW, int kernelH,
                  int imgW, int imgH, CostFunction func)
{
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
    return 0.0f;
}

// Full cost volume computation (CPU, OpenMP-parallel)
CostVolume computeCostVolume(const cv::Mat &left, const cv::Mat &right,
                             int minDisp, int maxDisp,
                             int kernelW, int kernelH,
                             CostFunction func, int numThreads)
{
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
    CV_Assert(left.size() == right.size());

    int numDisp = maxDisp - minDisp;
    int imgW = left.cols;
    int imgH = left.rows;

    CostVolume volume(static_cast<size_t>(numDisp));
    for (int dIdx = 0; dIdx < numDisp; ++dIdx)
    {
        volume[dIdx] = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0.0f));
    }

    const uchar *lPtr = left.ptr<uchar>();
    const uchar *rPtr = right.ptr<uchar>();

#ifdef _OPENMP
    #pragma omp parallel for num_threads(numThreads) collapse(2)
#endif
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            int validDispStart = std::max(minDisp, -x);
            int validDispEnd   = std::min(maxDisp, imgW - x);

            for (int d = validDispStart; d < validDispEnd; ++d)
            {
                int dIdx = d - minDisp;
                volume[dIdx].at<float>(y, x) = computeCost(
                    lPtr, rPtr, x, y, d,
                    kernelW, kernelH, imgW, imgH, func);
            }
        }
    }

    return volume;
}

} // namespace xjw::dense_match
