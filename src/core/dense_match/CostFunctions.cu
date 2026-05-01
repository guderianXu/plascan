// =============================================================================
// 文件: CostFunctions.cu
// 功能: 密集匹配代价函数 CUDA kernel 实现
// =============================================================================
#ifdef DM_ENABLE_CUDA

#include "CostFunctions.h"
#include <cuda_runtime.h>
#include <vector>

namespace xjw::dense_match
{

namespace
{

__device__ float adCostDev(const uchar *left, const uchar *right,
                           int x, int y, int d, int kw, int kh,
                           int imgW, int imgH)
{
    float sum = 0.0f;
    int count = 0;
    int halfKW = kw / 2, halfKH = kh / 2;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx, lx = rx + d;
            if (rx < 0 || rx >= imgW || lx < 0 || lx >= imgW) continue;
            sum += fabsf((float)left[ry * imgW + lx] - (float)right[ry * imgW + rx]);
            ++count;
        }
    }
    return count > 0 ? sum / count : 0.0f;
}

__device__ float sdCostDev(const uchar *left, const uchar *right,
                           int x, int y, int d, int kw, int kh,
                           int imgW, int imgH)
{
    float sum = 0.0f;
    int count = 0;
    int halfKW = kw / 2, halfKH = kh / 2;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx, lx = rx + d;
            if (rx < 0 || rx >= imgW || lx < 0 || lx >= imgW) continue;
            float diff = (float)left[ry * imgW + lx] - (float)right[ry * imgW + rx];
            sum += diff * diff;
            ++count;
        }
    }
    return count > 0 ? sum / count : 0.0f;
}

__device__ float nccCostDev(const uchar *left, const uchar *right,
                            int x, int y, int d, int kw, int kh,
                            int imgW, int imgH)
{
    int halfKW = kw / 2, halfKH = kh / 2;
    double meanL = 0.0, meanR = 0.0;
    int count = 0;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx, lx = rx + d;
            if (rx < 0 || rx >= imgW || lx < 0 || lx >= imgW) continue;
            meanL += left[ry * imgW + lx];
            meanR += right[ry * imgW + rx];
            ++count;
        }
    }
    if (count < 2) return 0.0f;
    meanL /= count;
    meanR /= count;

    double num = 0.0, denL = 0.0, denR = 0.0;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx, lx = rx + d;
            if (rx < 0 || rx >= imgW || lx < 0 || lx >= imgW) continue;
            double dl = left[ry * imgW + lx] - meanL;
            double dr = right[ry * imgW + rx] - meanR;
            num += dl * dr;
            denL += dl * dl;
            denR += dr * dr;
        }
    }
    double denom = sqrt(denL * denR);
    if (denom < 1e-10) return 0.0f;
    return (float)(1.0 - num / denom);
}

__device__ float censusCostDev(const uchar *left, const uchar *right,
                               int x, int y, int d, int kw, int kh,
                               int imgW, int imgH)
{
    int halfKW = kw / 2, halfKH = kh / 2;
    int hamming = 0, count = 0;
    uchar centerL = left[y * imgW + (x + d)];
    uchar centerR = right[y * imgW + x];
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            int rx = x + dx, lx = rx + d;
            if (rx < 0 || rx >= imgW || lx < 0 || lx >= imgW) continue;
            int bitL = (left[ry * imgW + lx] > centerL) ? 1 : 0;
            int bitR = (right[ry * imgW + rx] > centerR) ? 1 : 0;
            hamming += (bitL != bitR);
            ++count;
        }
    }
    return count > 0 ? (float)hamming / count : 0.0f;
}

__device__ float ternaryCensusCostDev(const uchar *left, const uchar *right,
                                      int x, int y, int d, int kw, int kh,
                                      int imgW, int imgH)
{
    const int tau = 5;
    int halfKW = kw / 2, halfKH = kh / 2;
    int hamming = 0, count = 0;
    uchar centerL = left[y * imgW + (x + d)];
    uchar centerR = right[y * imgW + x];
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            int rx = x + dx, lx = rx + d;
            if (rx < 0 || rx >= imgW || lx < 0 || lx >= imgW) continue;
            uchar v = left[ry * imgW + lx];
            int vL = (v > centerL + tau) ? 1 : (v < centerL - tau) ? 0 : 2;
            uchar w = right[ry * imgW + rx];
            int vR = (w > centerR + tau) ? 1 : (w < centerR - tau) ? 0 : 2;
            if (vL != 2 && vR != 2)
                hamming += (vL != vR);
            ++count;
        }
    }
    return count > 0 ? (float)hamming / count : 0.0f;
}

__global__ void computeCostVolumeKernel(const uchar *left, const uchar *right,
                                        float *costVolume, int imgW, int imgH,
                                        int minDisp, int numDisp,
                                        int kernelW, int kernelH,
                                        int costFunc)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imgW || y >= imgH) return;

    int validStart = max(minDisp, -x);
    int validEnd   = min(minDisp + numDisp, imgW - x);

    for (int dIdx = 0; dIdx < numDisp; ++dIdx)
    {
        int d = minDisp + dIdx;
        float c = 0.0f;
        if (d >= validStart && d < validEnd)
        {
            switch (costFunc)
            {
            case 0: c = adCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH); break;
            case 1: c = sdCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH); break;
            case 2: c = nccCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH); break;
            case 3: c = censusCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH); break;
            case 4: c = ternaryCensusCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH); break;
            }
        }
        costVolume[dIdx * imgH * imgW + y * imgW + x] = c;
    }
}

} // anonymous namespace

CostVolume computeCostVolumeCUDA(const cv::Mat &left, const cv::Mat &right,
                                 int minDisp, int maxDisp,
                                 int kernelW, int kernelH,
                                 CostFunction func, int cudaDevice)
{
    cudaSetDevice(cudaDevice);

    int numDisp = maxDisp - minDisp;
    int imgW = left.cols;
    int imgH = left.rows;
    size_t imgBytes = static_cast<size_t>(imgW) * imgH * sizeof(uchar);
    size_t volBytes = static_cast<size_t>(numDisp) * imgW * imgH * sizeof(float);

    uchar *d_left = nullptr;
    uchar *d_right = nullptr;
    float *d_volume = nullptr;
    cudaMalloc(&d_left, imgBytes);
    cudaMalloc(&d_right, imgBytes);
    cudaMalloc(&d_volume, volBytes);

    cudaMemcpy(d_left, left.data, imgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data, imgBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_volume, 0, volBytes);

    dim3 block(16, 16);
    dim3 grid((imgW + 15) / 16, (imgH + 15) / 16);
    computeCostVolumeKernel<<<grid, block>>>(
        d_left, d_right, d_volume, imgW, imgH, minDisp, numDisp,
        kernelW, kernelH, static_cast<int>(func));
    cudaDeviceSynchronize();

    CostVolume volume(static_cast<size_t>(numDisp));
    for (int dIdx = 0; dIdx < numDisp; ++dIdx)
    {
        volume[dIdx] = cv::Mat(imgH, imgW, CV_32FC1);
    }

    std::vector<float> h_volume(static_cast<size_t>(numDisp) * imgW * imgH);
    cudaMemcpy(h_volume.data(), d_volume, volBytes, cudaMemcpyDeviceToHost);

    for (int dIdx = 0; dIdx < numDisp; ++dIdx)
    {
        size_t offset = static_cast<size_t>(dIdx) * imgW * imgH;
        for (int y = 0; y < imgH; ++y)
        {
            for (int x = 0; x < imgW; ++x)
            {
                volume[dIdx].at<float>(y, x) = h_volume[offset + y * imgW + x];
            }
        }
    }

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_volume);
    return volume;
}

} // namespace xjw::dense_match

#endif // DM_ENABLE_CUDA
