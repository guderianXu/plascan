// =============================================================================
// 文件: PatchMatchCUDA.cu
// 模块: MVS - GPU PatchMatch 深度图估计
// 说明:
//   实现带法向量假设的 PatchMatch 深度估计（参考 COLMAP patch_match_cuda.cu）。
//   关键改进：每像素维护 (depth, normal) 平面假设，使用平面单应 NCC，
//   彻底消除只用标量深度时产生的辐射状条纹伪影。
// =============================================================================

#include "PatchMatchCUDA.h"
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <thread>
#include <atomic>
#include <array>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr,"[CUDA] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
        return false; \
    } } while(0)

namespace xjw 
{
namespace mvs 
{

namespace
{

using CpuNormal = std::array<float, 3>;

inline float cpuDot3(const CpuNormal &a, const CpuNormal &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline void cpuNormalize3(CpuNormal &v)
{
    const float norm = std::sqrt(std::max(1e-12f, cpuDot3(v, v)));
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
}

inline float cpuUniformSigned(std::mt19937 &rng)
{
    static thread_local std::uniform_real_distribution<float> dist(-1.f, 1.f);
    return dist(rng);
}

inline float cpuUniformUnit(std::mt19937 &rng)
{
    static thread_local std::uniform_real_distribution<float> dist(0.f, 1.f);
    return dist(rng);
}

inline CpuNormal cpuRayFromPixel(int row, int col, const float invK[4])
{
    return CpuNormal{
        invK[0] * static_cast<float>(col) + invK[1],
        invK[2] * static_cast<float>(row) + invK[3],
        1.f
    };
}

CpuNormal cpuGenerateRandomNormal(int row, int col, const float invK[4], std::mt19937 &rng)
{
    CpuNormal normal{};
    float v1 = 0.f;
    float v2 = 0.f;
    float s = 0.f;
    do
    {
        v1 = cpuUniformSigned(rng);
        v2 = cpuUniformSigned(rng);
        s = v1 * v1 + v2 * v2;
    } while (s >= 1.f || s == 0.f);

    const float sn = std::sqrt(1.f - s);
    normal[0] = 2.f * v1 * sn;
    normal[1] = 2.f * v2 * sn;
    normal[2] = 1.f - 2.f * s;

    const CpuNormal ray = cpuRayFromPixel(row, col, invK);
    if (cpuDot3(normal, ray) > 0.f)
    {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
    }
    cpuNormalize3(normal);
    return normal;
}

CpuNormal cpuRotateNormal(const CpuNormal &normal, float a1, float a2, float a3)
{
    const float sa1 = std::sin(a1);
    const float ca1 = std::cos(a1);
    const float sa2 = std::sin(a2);
    const float ca2 = std::cos(a2);
    const float sa3 = std::sin(a3);
    const float ca3 = std::cos(a3);

    const float R[9] = {
        ca2 * ca3, -ca2 * sa3, sa2,
        ca1 * sa3 + ca3 * sa1 * sa2, ca1 * ca3 - sa1 * sa2 * sa3, -ca2 * sa1,
        sa1 * sa3 - ca1 * ca3 * sa2, ca3 * sa1 + ca1 * sa2 * sa3, ca1 * ca2
    };

    CpuNormal out{
        R[0] * normal[0] + R[1] * normal[1] + R[2] * normal[2],
        R[3] * normal[0] + R[4] * normal[1] + R[5] * normal[2],
        R[6] * normal[0] + R[7] * normal[1] + R[8] * normal[2]
    };
    cpuNormalize3(out);
    return out;
}

CpuNormal cpuPerturbNormal(int row,
                           int col,
                           float perturbation,
                           const CpuNormal &normal,
                           const float invK[4],
                           std::mt19937 &rng,
                           int trial = 0)
{
    const float a1 = 0.5f * perturbation * cpuUniformSigned(rng);
    const float a2 = 0.5f * perturbation * cpuUniformSigned(rng);
    const float a3 = 0.5f * perturbation * cpuUniformSigned(rng);
    CpuNormal out = cpuRotateNormal(normal, a1, a2, a3);

    const CpuNormal ray = cpuRayFromPixel(row, col, invK);
    if (cpuDot3(out, ray) >= 0.f)
    {
        if (trial < 3)
        {
            return cpuPerturbNormal(row, col, 0.5f * perturbation, normal, invK, rng, trial + 1);
        }
        return normal;
    }
    return out;
}

inline float cpuPerturbDepth(float perturbation, float depth, std::mt19937 &rng)
{
    const float dMin = (1.f - perturbation) * depth;
    const float dMax = (1.f + perturbation) * depth;
    return dMin + cpuUniformUnit(rng) * (dMax - dMin);
}

inline float cpuPropagateDepth(float depth,
                               const CpuNormal &normal,
                               float row1,
                               float col1,
                               float row2,
                               float col2,
                               const float invK[4])
{
    const CpuNormal X1{
        depth * (invK[0] * col1 + invK[1]),
        depth * (invK[2] * row1 + invK[3]),
        depth
    };
    const CpuNormal ray2{
        invK[0] * col2 + invK[1],
        invK[2] * row2 + invK[3],
        1.f
    };
    const float denom = cpuDot3(normal, ray2);
    if (std::fabs(denom) < 1e-6f)
    {
        return depth;
    }
    const float t = cpuDot3(normal, X1) / denom;
    return t > 0.f ? t : depth;
}

inline float cpuBilinear(const cv::Mat &image, float u, float v)
{
    if (u < 0.f || v < 0.f || u >= static_cast<float>(image.cols - 1) || v >= static_cast<float>(image.rows - 1))
    {
        return -1.f;
    }

    const int ix = static_cast<int>(u);
    const int iy = static_cast<int>(v);
    const float fx = u - static_cast<float>(ix);
    const float fy = v - static_cast<float>(iy);
    const float v00 = image.at<float>(iy, ix);
    const float v10 = image.at<float>(iy, ix + 1);
    const float v01 = image.at<float>(iy + 1, ix);
    const float v11 = image.at<float>(iy + 1, ix + 1);
    return (1.f - fy) * ((1.f - fx) * v00 + fx * v10)
         + fy * ((1.f - fx) * v01 + fx * v11);
}

void cpuComposeHomography(int row,
                          int col,
                          float depth,
                          const CpuNormal &normal,
                          const float *srcData,
                          const float invK[4],
                          float H[9])
{
    const float fxS = srcData[0];
    const float cxS = srcData[1];
    const float fyS = srcData[2];
    const float cyS = srcData[3];
    const float *R = srcData + 4;
    const float *T = srcData + 13;

    float dist = depth * (normal[0] * (invK[0] * static_cast<float>(col) + invK[1])
                        + normal[1] * (invK[2] * static_cast<float>(row) + invK[3])
                        + normal[2]);
    if (std::fabs(dist) < 1e-6f)
    {
        dist = 1e-6f;
    }
    const float invD = 1.f / dist;

    const float iN0 = invD * normal[0];
    const float iN1 = invD * normal[1];
    const float iN2 = invD * normal[2];

    H[0] = invK[0] * (fxS * (R[0] + iN0 * T[0]) + cxS * (R[6] + iN0 * T[2]));
    H[1] = invK[2] * (fxS * (R[1] + iN1 * T[0]) + cxS * (R[7] + iN1 * T[2]));
    H[2] = fxS * (R[2] + iN2 * T[0]) + cxS * (R[8] + iN2 * T[2])
         + invK[1] * (fxS * (R[0] + iN0 * T[0]) + cxS * (R[6] + iN0 * T[2]))
         + invK[3] * (fxS * (R[1] + iN1 * T[0]) + cxS * (R[7] + iN1 * T[2]));
    H[3] = invK[0] * (fyS * (R[3] + iN0 * T[1]) + cyS * (R[6] + iN0 * T[2]));
    H[4] = invK[2] * (fyS * (R[4] + iN1 * T[1]) + cyS * (R[7] + iN1 * T[2]));
    H[5] = fyS * (R[5] + iN2 * T[1]) + cyS * (R[8] + iN2 * T[2])
         + invK[1] * (fyS * (R[3] + iN0 * T[1]) + cyS * (R[6] + iN0 * T[2]))
         + invK[3] * (fyS * (R[4] + iN1 * T[1]) + cyS * (R[7] + iN1 * T[2]));
    H[6] = invK[0] * (R[6] + iN0 * T[2]);
    H[7] = invK[2] * (R[7] + iN1 * T[2]);
    H[8] = R[8] + invK[1] * (R[6] + iN0 * T[2])
                + invK[3] * (R[7] + iN1 * T[2]) + iN2 * T[2];
}

float cpuComputeHomographyNcc(int refCol,
                              int refRow,
                              float depth,
                              const CpuNormal &normal,
                              const cv::Mat &refImage,
                              const cv::Mat &srcImage,
                              const float *srcData,
                              const float invK[4],
                              int patchHalf)
{
    float H[9];
    cpuComposeHomography(refRow, refCol, depth, normal, srcData, invK, H);

    float sumRef = 0.f;
    float sumSrc = 0.f;
    float sumRef2 = 0.f;
    float sumSrc2 = 0.f;
    float sumRefSrc = 0.f;
    int count = 0;

    for (int dv = -patchHalf; dv <= patchHalf; ++dv)
    {
        for (int du = -patchHalf; du <= patchHalf; ++du)
        {
            const int pu = refCol + du;
            const int pv = refRow + dv;
            if (pu < 0 || pu >= refImage.cols || pv < 0 || pv >= refImage.rows)
            {
                continue;
            }

            const float refValue = refImage.at<float>(pv, pu);
            const float wsC = H[0] * static_cast<float>(pu) + H[1] * static_cast<float>(pv) + H[2];
            const float wsR = H[3] * static_cast<float>(pu) + H[4] * static_cast<float>(pv) + H[5];
            const float wsZ = H[6] * static_cast<float>(pu) + H[7] * static_cast<float>(pv) + H[8];
            if (std::fabs(wsZ) < 1e-6f)
            {
                continue;
            }

            const float srcU = wsC / wsZ;
            const float srcV = wsR / wsZ;
            const float srcValue = cpuBilinear(srcImage, srcU, srcV);
            if (srcValue < 0.f)
            {
                continue;
            }

            sumRef += refValue;
            sumSrc += srcValue;
            sumRef2 += refValue * refValue;
            sumSrc2 += srcValue * srcValue;
            sumRefSrc += refValue * srcValue;
            ++count;
        }
    }

    if (count < 4)
    {
        return 0.f;
    }

    const float invCount = 1.f / static_cast<float>(count);
    const float meanRef = sumRef * invCount;
    const float meanSrc = sumSrc * invCount;
    const float varRef = sumRef2 * invCount - meanRef * meanRef;
    const float varSrc = sumSrc2 * invCount - meanSrc * meanSrc;
    const float cov = sumRefSrc * invCount - meanRef * meanSrc;
    const float denom = std::sqrt(std::max(0.f, varRef * varSrc));
    if (denom < 1e-5f)
    {
        return 0.f;
    }
    return ((cov / denom) + 1.f) * 0.5f;
}

float cpuEvalHypCost(int col,
                     int row,
                     float depth,
                     const CpuNormal &normal,
                     const cv::Mat &refImage,
                     const std::vector<cv::Mat> &srcImages,
                     const std::vector<float> &srcDatas,
                     int patchHalf,
                     const float invK[4])
{
    if (depth <= 0.f)
    {
        return 2.f;
    }

    float sumScore = 0.f;
    int goodSrc = 0;
    for (int srcIndex = 0; srcIndex < static_cast<int>(srcImages.size()); ++srcIndex)
    {
        const float *srcData = srcDatas.data() + srcIndex * 16;
        const float ncc = cpuComputeHomographyNcc(col, row, depth, normal,
                                                  refImage, srcImages[srcIndex], srcData,
                                                  invK, patchHalf);
        if (ncc > 0.05f)
        {
            sumScore += ncc;
            ++goodSrc;
        }
    }

    if (goodSrc == 0)
    {
        return 2.f;
    }
    return 2.f - 2.f * (sumScore / static_cast<float>(goodSrc));
}

template <typename Fn>
void cpuParallelForLines(int lineCount, int workerCount, Fn &&fn)
{
    if (lineCount <= 0)
    {
        return;
    }
    if (workerCount <= 1)
    {
        for (int line = 0; line < lineCount; ++line)
        {
            fn(line);
        }
        return;
    }

    std::atomic<int> nextLine{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(workerCount));
    for (int workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        workers.emplace_back([&]() {
            while (true)
            {
                const int line = nextLine.fetch_add(1);
                if (line >= lineCount)
                {
                    break;
                }
                fn(line);
            }
        });
    }
    for (std::thread &worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

struct SrcImageCacheKey
{
    uintptr_t hostData = 0;
    int rows = 0;
    int cols = 0;
    size_t step = 0;
    int ds = 1;

    bool operator==(const SrcImageCacheKey &other) const
    {
        return hostData == other.hostData
            && rows == other.rows
            && cols == other.cols
            && step == other.step
            && ds == other.ds;
    }
};

struct SrcImageCacheKeyHash
{
    size_t operator()(const SrcImageCacheKey &key) const
    {
        size_t hash = std::hash<uintptr_t>{}(key.hostData);
        hash ^= std::hash<int>{}(key.rows) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.cols) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<size_t>{}(key.step) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(key.ds) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct SrcImageCacheEntry
{
    float *devicePtr = nullptr;
    int scaledW = 0;
    int scaledH = 0;
    size_t bytes = 0;
    uint64_t lastUseTick = 0;
};

std::unordered_map<SrcImageCacheKey, SrcImageCacheEntry, SrcImageCacheKeyHash> g_srcImageGpuCache;
std::mutex g_srcImageGpuCacheMutex;
uint64_t g_srcImageGpuCacheTick = 0;
size_t g_srcImageGpuCacheBytes = 0;

size_t getGrayImageGpuCacheLimitBytes()
{
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess || totalBytes == 0)
    {
        return 1024ULL * 1024ULL * 1024ULL;
    }

    const size_t oneGiB = 1024ULL * 1024ULL * 1024ULL;
    const size_t sixGiB = 6ULL * oneGiB;
    const size_t reserveBytes = 768ULL * 1024ULL * 1024ULL;

    size_t limitByTotal = totalBytes / 2;
    size_t limitByFree = freeBytes > reserveBytes ? (freeBytes - reserveBytes) : (freeBytes / 2);
    size_t limit = std::min(limitByTotal, limitByFree);
    limit = std::max(limit, oneGiB);
    limit = std::min(limit, sixGiB);
    return limit;
}

void evictSrcImageGpuCacheIfNeeded(size_t needBytes)
{
    const size_t cacheLimitBytes = getGrayImageGpuCacheLimitBytes();
    while (g_srcImageGpuCacheBytes + needBytes > cacheLimitBytes && !g_srcImageGpuCache.empty())
    {
        auto lruIt = g_srcImageGpuCache.begin();
        for (auto it = g_srcImageGpuCache.begin(); it != g_srcImageGpuCache.end(); ++it)
        {
            if (it->second.lastUseTick < lruIt->second.lastUseTick)
            {
                lruIt = it;
            }
        }

        if (lruIt->second.devicePtr)
        {
            cudaFree(lruIt->second.devicePtr);
        }
        g_srcImageGpuCacheBytes -= lruIt->second.bytes;
        g_srcImageGpuCache.erase(lruIt);
    }
}

bool getOrUploadGrayImageGpu(
    const cv::Mat &srcGray,
    int scaledW,
    int scaledH,
    int ds,
    float **devicePtrOut,
    bool *cacheHitOut,
    std::string *errorMsg)
{
    if (devicePtrOut == nullptr)
    {
        if (errorMsg)
        {
            *errorMsg = "devicePtrOut 为空";
        }
        return false;
    }
    if (srcGray.empty() || srcGray.data == nullptr)
    {
        if (errorMsg)
        {
            *errorMsg = "源图为空，无法上传 GPU";
        }
        return false;
    }

    SrcImageCacheKey key;
    key.hostData = reinterpret_cast<uintptr_t>(srcGray.data);
    key.rows = srcGray.rows;
    key.cols = srcGray.cols;
    key.step = srcGray.step;
    key.ds = ds;

    {
        std::lock_guard<std::mutex> lock(g_srcImageGpuCacheMutex);
        auto it = g_srcImageGpuCache.find(key);
        if (it != g_srcImageGpuCache.end() && it->second.scaledW == scaledW && it->second.scaledH == scaledH)
        {
            it->second.lastUseTick = ++g_srcImageGpuCacheTick;
            *devicePtrOut = it->second.devicePtr;
            if (cacheHitOut)
            {
                *cacheHitOut = true;
            }
            return true;
        }
    }

    cv::Mat srcScaled;
    cv::resize(srcGray, srcScaled, cv::Size(scaledW, scaledH), 0, 0, cv::INTER_AREA);
    cv::Mat srcFloat;
    srcScaled.convertTo(srcFloat, CV_32F, 1.f / 255.f);

    const size_t bytes = static_cast<size_t>(scaledW) * static_cast<size_t>(scaledH) * sizeof(float);
    float *newDevicePtr = nullptr;
    cudaError_t allocErr = cudaMalloc(&newDevicePtr, bytes);
    if (allocErr != cudaSuccess)
    {
        if (errorMsg)
        {
            *errorMsg = std::string("cudaMalloc 失败: ") + cudaGetErrorString(allocErr);
        }
        return false;
    }

    cudaError_t copyErr = cudaMemcpy(newDevicePtr, srcFloat.ptr<float>(), bytes, cudaMemcpyHostToDevice);
    if (copyErr != cudaSuccess)
    {
        cudaFree(newDevicePtr);
        if (errorMsg)
        {
            *errorMsg = std::string("cudaMemcpy 失败: ") + cudaGetErrorString(copyErr);
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_srcImageGpuCacheMutex);

        auto existing = g_srcImageGpuCache.find(key);
        if (existing != g_srcImageGpuCache.end() && existing->second.scaledW == scaledW && existing->second.scaledH == scaledH)
        {
            existing->second.lastUseTick = ++g_srcImageGpuCacheTick;
            *devicePtrOut = existing->second.devicePtr;
            if (cacheHitOut)
            {
                *cacheHitOut = true;
            }
            cudaFree(newDevicePtr);
            return true;
        }

        evictSrcImageGpuCacheIfNeeded(bytes);

        SrcImageCacheEntry entry;
        entry.devicePtr = newDevicePtr;
        entry.scaledW = scaledW;
        entry.scaledH = scaledH;
        entry.bytes = bytes;
        entry.lastUseTick = ++g_srcImageGpuCacheTick;

        g_srcImageGpuCache[key] = entry;
        g_srcImageGpuCacheBytes += bytes;

        *devicePtrOut = newDevicePtr;
        if (cacheHitOut)
        {
            *cacheHitOut = false;
        }
    }

    return true;
}

} // namespace

// =============================================================================
// GPU 常量内存：参考相机内参逆矩阵 = {1/fx, -cx/fx, 1/fy, -cy/fy}
// =============================================================================
__constant__ float c_ref_inv_K[4];

// =============================================================================
// 设备端辅助函数
// =============================================================================

__device__ inline void mat33MulVec3(const float M[9], const float v[3], float out[3]) 
{
    out[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
    out[1] = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
    out[2] = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
}

__device__ inline float dot3(const float a[3], const float b[3]) 
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// 双线性插值（边界外返回 -1）
__device__ float bilinear(const float *img, int W, int H, float u, float v)
{
    if (u < 0 || v < 0 || u >= W-1 || v >= H-1) return -1.f;
    int ix = (int)u, iy = (int)v;
    float fx = u - ix, fy = v - iy;
    float v00 = img[iy*W+ix],     v10 = img[iy*W+ix+1];
    float v01 = img[(iy+1)*W+ix], v11 = img[(iy+1)*W+ix+1];
    return (1-fy)*((1-fx)*v00 + fx*v10) + fy*((1-fx)*v01 + fx*v11);
}

// =============================================================================
// 深度扰动（COLMAP PerturbDepth）
// =============================================================================
__device__ inline float perturbDepth(float perturbation, float depth, curandState *rs)
{
    float d_min = (1.f - perturbation) * depth;
    float d_max = (1.f + perturbation) * depth;
    return d_min + curand_uniform(rs) * (d_max - d_min);
}

// =============================================================================
// 法向量处理
// =============================================================================

/// 生成随机单位法向量（Marsaglia 方法），确保朝向相机
__device__ void generateRandomNormal(int row, int col,
                                     curandState *rs, float normal[3])
{
    float v1, v2, s;
    do 
    {
        v1 = 2.f * curand_uniform(rs) - 1.f;
        v2 = 2.f * curand_uniform(rs) - 1.f;
        s  = v1*v1 + v2*v2;
    } while (s >= 1.f || s == 0.f);

    float sn = sqrtf(1.f - s);
    normal[0] = 2.f * v1 * sn;
    normal[1] = 2.f * v2 * sn;
    normal[2] = 1.f - 2.f * s;

    float ray[3] = { c_ref_inv_K[0]*col + c_ref_inv_K[1],
                     c_ref_inv_K[2]*row + c_ref_inv_K[3],
                     1.f };
    if (dot3(normal, ray) > 0) 
    {
        normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2];
    }
}

/// 扰动法向量（小角度随机旋转）
__device__ void perturbNormal(int row, int col,
                               float perturbation,
                               const float n[3], curandState *rs,
                               float out[3], int trial = 0)
{
    float a1 = (curand_uniform(rs) - 0.5f) * perturbation;
    float a2 = (curand_uniform(rs) - 0.5f) * perturbation;
    float a3 = (curand_uniform(rs) - 0.5f) * perturbation;

    float sa1=sinf(a1), ca1=cosf(a1);
    float sa2=sinf(a2), ca2=cosf(a2);
    float sa3=sinf(a3), ca3=cosf(a3);

    float R[9];
    R[0]=ca2*ca3;  R[1]=-ca2*sa3; R[2]=sa2;
    R[3]=ca1*sa3+ca3*sa1*sa2; R[4]=ca1*ca3-sa1*sa2*sa3; R[5]=-ca2*sa1;
    R[6]=sa1*sa3-ca1*ca3*sa2; R[7]=ca3*sa1+ca1*sa2*sa3; R[8]=ca1*ca2;

    mat33MulVec3(R, n, out);

    float ray[3] = { c_ref_inv_K[0]*col + c_ref_inv_K[1],
                     c_ref_inv_K[2]*row + c_ref_inv_K[3],
                     1.f };
    if (dot3(out, ray) >= 0.f) 
    {
        if (trial < 3) { perturbNormal(row, col, 0.5f*perturbation, n, rs, out, trial+1); return; }
        out[0]=n[0]; out[1]=n[1]; out[2]=n[2];
        return;
    }
    float inv = rsqrtf(dot3(out, out));
    out[0]*=inv; out[1]*=inv; out[2]*=inv;
}

// =============================================================================
// 深度沿平面传播：将 (row1,col1) 处的平面假设 (d1,n1) 传播到 (row2,col2) 射线
// =============================================================================
__device__ float propagateDepth(float d1, const float n1[3],
                                  float row1, float col1,
                                  float row2, float col2)
{
    float X1[3] = { d1*(c_ref_inv_K[0]*col1 + c_ref_inv_K[1]),
                    d1*(c_ref_inv_K[2]*row1 + c_ref_inv_K[3]),
                    d1 };
    float ray2[3] = { c_ref_inv_K[0]*col2 + c_ref_inv_K[1],
                      c_ref_inv_K[2]*row2 + c_ref_inv_K[3],
                      1.f };
    float denom = dot3(n1, ray2);
    if (fabsf(denom) < 1e-6f) return d1;
    float t = dot3(n1, X1) / denom;
    return (t > 0.f) ? t : d1;
}

// =============================================================================
// 平面单应性合成（参考 COLMAP ComposeHomography）
// srcData: {fx_s,cx_s,fy_s,cy_s, R_rel[9], T_rel[3]}  共 16 floats
// H 将参考帧像素 [col,row,1] 映射到源帧像素（齐次坐标）
// =============================================================================
__device__ void composeHomography(int row, int col,
                                   float depth, const float normal[3],
                                   const float *srcData,
                                   float H[9])
{
    float fx_s=srcData[0], cx_s=srcData[1];
    float fy_s=srcData[2], cy_s=srcData[3];
    const float *R = srcData + 4;
    const float *T = srcData + 13;

    float dist = depth * (normal[0]*(c_ref_inv_K[0]*col + c_ref_inv_K[1]) +
                          normal[1]*(c_ref_inv_K[2]*row + c_ref_inv_K[3]) +
                          normal[2]);
    if (fabsf(dist) < 1e-6f) dist = 1e-6f;
    float inv_d = 1.f / dist;

    float iN0 = inv_d*normal[0], iN1 = inv_d*normal[1], iN2 = inv_d*normal[2];

    H[0] = c_ref_inv_K[0] * (fx_s*(R[0]+iN0*T[0]) + cx_s*(R[6]+iN0*T[2]));
    H[1] = c_ref_inv_K[2] * (fx_s*(R[1]+iN1*T[0]) + cx_s*(R[7]+iN1*T[2]));
    H[2] = fx_s*(R[2]+iN2*T[0]) + cx_s*(R[8]+iN2*T[2])
         + c_ref_inv_K[1]*(fx_s*(R[0]+iN0*T[0]) + cx_s*(R[6]+iN0*T[2]))
         + c_ref_inv_K[3]*(fx_s*(R[1]+iN1*T[0]) + cx_s*(R[7]+iN1*T[2]));
    H[3] = c_ref_inv_K[0] * (fy_s*(R[3]+iN0*T[1]) + cy_s*(R[6]+iN0*T[2]));
    H[4] = c_ref_inv_K[2] * (fy_s*(R[4]+iN1*T[1]) + cy_s*(R[7]+iN1*T[2]));
    H[5] = fy_s*(R[5]+iN2*T[1]) + cy_s*(R[8]+iN2*T[2])
         + c_ref_inv_K[1]*(fy_s*(R[3]+iN0*T[1]) + cy_s*(R[6]+iN0*T[2]))
         + c_ref_inv_K[3]*(fy_s*(R[4]+iN1*T[1]) + cy_s*(R[7]+iN1*T[2]));
    H[6] = c_ref_inv_K[0] * (R[6]+iN0*T[2]);
    H[7] = c_ref_inv_K[2] * (R[7]+iN1*T[2]);
    H[8] = R[8] + c_ref_inv_K[1]*(R[6]+iN0*T[2])
                + c_ref_inv_K[3]*(R[7]+iN1*T[2]) + iN2*T[2];
}

// =============================================================================
// 平面单应 NCC：用 H 映射 patch，计算 NCC（范围 [0,1]）
// =============================================================================
__device__ float computeHomographyNCC(
    int u_r, int v_r,
    float depth, const float normal[3],
    const float *refImg, int refW, int refH,
    const float *srcImg, int srcW, int srcH,
    const float *srcData,
    int patchHalf)
{
    float H[9];
    composeHomography(v_r, u_r, depth, normal, srcData, H);

    float sumRef=0, sumSrc=0, sumRef2=0, sumSrc2=0, sumRefSrc=0;
    int cnt = 0;
    const int r = patchHalf;

    for (int dv = -r; dv <= r; ++dv) 
    {
        for (int du = -r; du <= r; ++du) 
        {
            int pu = u_r + du, pv = v_r + dv;
            if (pu<0||pu>=refW||pv<0||pv>=refH) continue;

            float valRef = refImg[pv*refW + pu];

            float ws_c = H[0]*pu + H[1]*pv + H[2];
            float ws_r = H[3]*pu + H[4]*pv + H[5];
            float ws_z = H[6]*pu + H[7]*pv + H[8];
            if (fabsf(ws_z) < 1e-6f) continue;
            float u_s = ws_c / ws_z;
            float v_s = ws_r / ws_z;

            float valSrc = bilinear(srcImg, srcW, srcH, u_s, v_s);
            if (valSrc < 0) continue;

            sumRef   += valRef;   sumSrc   += valSrc;
            sumRef2  += valRef*valRef; sumSrc2  += valSrc*valSrc;
            sumRefSrc += valRef*valSrc;
            ++cnt;
        }
    }

    if (cnt < 4) return 0.f;
    float n = (float)cnt;
    float mR=sumRef/n, mS=sumSrc/n;
    float vR=sumRef2/n-mR*mR, vS=sumSrc2/n-mS*mS;
    float cov=sumRefSrc/n-mR*mS;
    float denom=sqrtf(vR*vS);
    if (denom < 1e-5f) return 0.f;
    return ((cov/denom) + 1.f) * 0.5f;
}

// =============================================================================
// CUDA Kernels
// =============================================================================

/// 初始化深度图和法向量图
__global__ void kernelInitDepthNormal(
    float *depthMap, float *normalMap,
    const float *hintDepth,
    int W, int H,
    float zNear, float zFar,
    unsigned long long seed)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;
    int idx = v * W + u;

    curandState rs;
    curand_init(seed + idx, 0, 0, &rs);

    float hint = (hintDepth != nullptr) ? hintDepth[idx] : 0.f;
    float z;
    if (hint > 0.f && hint >= zNear && hint <= zFar) 
    {
        float range = hint * 0.3f;
        z = hint + (curand_uniform(&rs) * 2.f - 1.f) * range;
        z = fmaxf(zNear, fminf(zFar, z));
    } 
    else 
    {
        z = zNear + curand_uniform(&rs) * (zFar - zNear);
    }
    depthMap[idx] = z;

    float normal[3];
    generateRandomNormal(v, u, &rs, normal);
    normalMap[idx*3]   = normal[0];
    normalMap[idx*3+1] = normal[1];
    normalMap[idx*3+2] = normal[2];
}

// =============================================================================
// 单假设多源 NCC 代价评估（代价 = 2 - 2*NCC，范围 [0,2]，越小越好）
// =============================================================================
__device__ float evalHypCost(
    int col, int row, float depth, const float normal[3],
    const float *refImg, int refW, int refH,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    int patchHalf)
{
    if (depth <= 0.f) return 2.f;
    float sumScore = 0.f;
    int   goodSrc  = 0;
    for (int si = 0; si < numSrc; ++si) 
    {
        const float *srcImg = srcImgs[si];
        float ncc = computeHomographyNCC(
            col, row, depth, normal,
            refImg, refW, refH,
            srcImg, srcW, srcH,
            srcDatas + si * 16,
            patchHalf);
        if (ncc > 0.05f) { sumScore += ncc; ++goodSrc; }
    }
    if (goodSrc == 0) return 2.f;
    return 2.f - 2.f * (sumScore / goodSrc);  // [0,2]
}

// =============================================================================
// Sweep 核（自上而下）：一列一线程，顺序迭代各行
// 仿照 COLMAP SweepFromTopToBottom，实现无竞争的列扫描正向传播
// =============================================================================
__global__ void kernelSweepTB(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    float zNear, float zFar, int patchHalf,
    float perturbation, unsigned long long seed)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= W) return;

    curandState rs;
    curand_init(seed + (unsigned long long)col, 0, 0, &rs);

    float prevDepth = depthMap[col]; // row=0
    float prevNormal[3] = { normalMap[col*3], normalMap[col*3+1], normalMap[col*3+2] };

    for (int row = 0; row < H; ++row) 
    {
        int idx = row * W + col;

        float currDepth     = depthMap[idx];
        float currNormal[3] = { normalMap[idx*3], normalMap[idx*3+1], normalMap[idx*3+2] };

        // 将上一行平面假设传播到当前行（平面-射线交点）
        float propDepth = (row > 0)
            ? propagateDepth(prevDepth, prevNormal, (float)(row-1), (float)col, (float)row, (float)col)
            : currDepth;
        propDepth = fmaxf(zNear, fminf(zFar, propDepth));

        // 随机扰动（退火）
        float randDepth = perturbDepth(perturbation, currDepth, &rs);
        randDepth = fmaxf(zNear, fminf(zFar, randDepth));
        float randNormal[3];
        perturbNormal(row, col, perturbation * (float)M_PI, currNormal, &rs, randNormal);

        // 5 个假设（仿照 COLMAP）:
        //  0: (curr, curr)  1: (prop, prev)  2: (rand, rand)
        //  3: (curr, rand)  4: (rand, curr)
        float dep[5] = { currDepth, propDepth, randDepth, currDepth, randDepth };
        float nor[5][3] = 
        {
            {currNormal[0],currNormal[1],currNormal[2]},
            {prevNormal[0],prevNormal[1],prevNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {currNormal[0],currNormal[1],currNormal[2]}
        };

        float bestCost = 2.f;
        int   bestIdx  = 0;
        for (int hi = 0; hi < 5; ++hi)
        {
            float cost = evalHypCost(col, row, dep[hi], nor[hi],
                                     refImg, W, H,
                                     srcImgs, srcDatas, srcW, srcH, numSrc,
                                     patchHalf);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestIdx = hi;
            }
        }

        depthMap[idx]      = dep[bestIdx];
        normalMap[idx*3]   = nor[bestIdx][0];
        normalMap[idx*3+1] = nor[bestIdx][1];
        normalMap[idx*3+2] = nor[bestIdx][2];
        confMap[idx]       = 1.f - bestCost * 0.5f; // [0,1]

        prevDepth      = dep[bestIdx];
        prevNormal[0]  = nor[bestIdx][0];
        prevNormal[1]  = nor[bestIdx][1];
        prevNormal[2]  = nor[bestIdx][2];
    }
}

// =============================================================================
// Sweep 核（自下而上）
// =============================================================================
__global__ void kernelSweepBT(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    float zNear, float zFar, int patchHalf,
    float perturbation, unsigned long long seed)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= W) return;

    curandState rs;
    curand_init(seed + (unsigned long long)col, 0, 0, &rs);

    int lastRow = H - 1;
    float prevDepth     = depthMap[lastRow * W + col];
    float prevNormal[3] = { normalMap[(lastRow*W+col)*3],
                             normalMap[(lastRow*W+col)*3+1],
                             normalMap[(lastRow*W+col)*3+2] };

    for (int row = H - 1; row >= 0; --row) 
    {
        int idx = row * W + col;

        float currDepth     = depthMap[idx];
        float currNormal[3] = { normalMap[idx*3], normalMap[idx*3+1], normalMap[idx*3+2] };

        float propDepth = (row < H - 1)
            ? propagateDepth(prevDepth, prevNormal, (float)(row+1), (float)col, (float)row, (float)col)
            : currDepth;
        propDepth = fmaxf(zNear, fminf(zFar, propDepth));

        float randDepth = perturbDepth(perturbation, currDepth, &rs);
        randDepth = fmaxf(zNear, fminf(zFar, randDepth));
        float randNormal[3];
        perturbNormal(row, col, perturbation * (float)M_PI, currNormal, &rs, randNormal);

        float dep[5] = { currDepth, propDepth, randDepth, currDepth, randDepth };
        float nor[5][3] = {

            {currNormal[0],currNormal[1],currNormal[2]},
            {prevNormal[0],prevNormal[1],prevNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {currNormal[0],currNormal[1],currNormal[2]}
        };

        float bestCost = 2.f;
        int   bestIdx  = 0;
        for (int hi = 0; hi < 5; ++hi) 
        {            
            float cost = evalHypCost(col, row, dep[hi], nor[hi],
                                     refImg, W, H,
                                     srcImgs, srcDatas, srcW, srcH, numSrc,
                                     patchHalf);
            if (cost < bestCost) { bestCost = cost; bestIdx = hi; }
        }

        depthMap[idx]      = dep[bestIdx];
        normalMap[idx*3]   = nor[bestIdx][0];
        normalMap[idx*3+1] = nor[bestIdx][1];
        normalMap[idx*3+2] = nor[bestIdx][2];
        confMap[idx]       = 1.f - bestCost * 0.5f;

        prevDepth     = dep[bestIdx];
        prevNormal[0] = nor[bestIdx][0];
        prevNormal[1] = nor[bestIdx][1];
        prevNormal[2] = nor[bestIdx][2];
    }
}

// =============================================================================
// Sweep 核（自左向右）：一行一线程，顺序迭代各列
// =============================================================================
__global__ void kernelSweepLR(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    float zNear, float zFar, int patchHalf,
    float perturbation, unsigned long long seed)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H) return;

    curandState rs;
    curand_init(seed + (unsigned long long)row, 0, 0, &rs);

    float prevDepth     = depthMap[row * W];
    float prevNormal[3] = { normalMap[(row*W)*3], normalMap[(row*W)*3+1], normalMap[(row*W)*3+2] };

    for (int col = 0; col < W; ++col) 
    {
        int idx = row * W + col;

        float currDepth     = depthMap[idx];
        float currNormal[3] = { normalMap[idx*3], normalMap[idx*3+1], normalMap[idx*3+2] };

        float propDepth = (col > 0)
            ? propagateDepth(prevDepth, prevNormal, (float)row, (float)(col-1), (float)row, (float)col)
            : currDepth;
        propDepth = fmaxf(zNear, fminf(zFar, propDepth));

        float randDepth = perturbDepth(perturbation, currDepth, &rs);
        randDepth = fmaxf(zNear, fminf(zFar, randDepth));
        float randNormal[3];
        perturbNormal(row, col, perturbation * (float)M_PI, currNormal, &rs, randNormal);

        float dep[5] = { currDepth, propDepth, randDepth, currDepth, randDepth };
        float nor[5][3] = 
        {
            {currNormal[0],currNormal[1],currNormal[2]},
            {prevNormal[0],prevNormal[1],prevNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {currNormal[0],currNormal[1],currNormal[2]}
        };

        float bestCost = 2.f;
        int   bestIdx  = 0;
        for (int hi = 0; hi < 5; ++hi) 
        {
            float cost = evalHypCost(col, row, dep[hi], nor[hi],
                                     refImg, W, H,
                                     srcImgs, srcDatas, srcW, srcH, numSrc,
                                     patchHalf);
            if (cost < bestCost) { bestCost = cost; bestIdx = hi; }
        }

        depthMap[idx]      = dep[bestIdx];
        normalMap[idx*3]   = nor[bestIdx][0];
        normalMap[idx*3+1] = nor[bestIdx][1];
        normalMap[idx*3+2] = nor[bestIdx][2];
        confMap[idx]       = 1.f - bestCost * 0.5f;

        prevDepth     = dep[bestIdx];
        prevNormal[0] = nor[bestIdx][0];
        prevNormal[1] = nor[bestIdx][1];
        prevNormal[2] = nor[bestIdx][2];
    }
}

// =============================================================================
// Sweep 核（自右向左）
// =============================================================================
__global__ void kernelSweepRL(
    float *depthMap, float *normalMap, float *confMap,
    const float *refImg, int W, int H,
    const float *const *srcImgs, const float *srcDatas, int srcW, int srcH, int numSrc,
    float zNear, float zFar, int patchHalf,
    float perturbation, unsigned long long seed)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H) return;

    curandState rs;
    curand_init(seed + (unsigned long long)row, 0, 0, &rs);

    int lastCol = W - 1;
    float prevDepth     = depthMap[row * W + lastCol];
    float prevNormal[3] = { normalMap[(row*W+lastCol)*3],
                             normalMap[(row*W+lastCol)*3+1],
                             normalMap[(row*W+lastCol)*3+2] };

    for (int col = W - 1; col >= 0; --col) 
    {
        int idx = row * W + col;

        float currDepth     = depthMap[idx];
        float currNormal[3] = { normalMap[idx*3], normalMap[idx*3+1], normalMap[idx*3+2] };

        float propDepth = (col < W - 1)
            ? propagateDepth(prevDepth, prevNormal, (float)row, (float)(col+1), (float)row, (float)col)
            : currDepth;
        propDepth = fmaxf(zNear, fminf(zFar, propDepth));

        float randDepth = perturbDepth(perturbation, currDepth, &rs);
        randDepth = fmaxf(zNear, fminf(zFar, randDepth));
        float randNormal[3];
        perturbNormal(row, col, perturbation * (float)M_PI, currNormal, &rs, randNormal);

        float dep[5] = { currDepth, propDepth, randDepth, currDepth, randDepth };
        float nor[5][3] = 
        {
            {currNormal[0],currNormal[1],currNormal[2]},
            {prevNormal[0],prevNormal[1],prevNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {randNormal[0],randNormal[1],randNormal[2]},
            {currNormal[0],currNormal[1],currNormal[2]}
        };

        float bestCost = 2.f;
        int   bestIdx  = 0;
        for (int hi = 0; hi < 5; ++hi) 
        {
            float cost = evalHypCost(col, row, dep[hi], nor[hi],
                                     refImg, W, H,
                                     srcImgs, srcDatas, srcW, srcH, numSrc,
                                     patchHalf);
            if (cost < bestCost) { bestCost = cost; bestIdx = hi; }
        }

        depthMap[idx]      = dep[bestIdx];
        normalMap[idx*3]   = nor[bestIdx][0];
        normalMap[idx*3+1] = nor[bestIdx][1];
        normalMap[idx*3+2] = nor[bestIdx][2];
        confMap[idx]       = 1.f - bestCost * 0.5f;

        prevDepth     = dep[bestIdx];
        prevNormal[0] = nor[bestIdx][0];
        prevNormal[1] = nor[bestIdx][1];
        prevNormal[2] = nor[bestIdx][2];
    }
}

/// 置信度阈值过滤
__global__ void kernelFinalizeDepth(
    float *depthMap, const float *confMap, int W, int H, float confThresh)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;
    int idx = v * W + u;
    if (confMap[idx] < confThresh) depthMap[idx] = 0.f;
}

// =============================================================================
// GPU 实现
// =============================================================================
bool PatchMatchDepthEstimator::estimateGPU(
    const cv::Mat                &refGray,
    const std::vector<cv::Mat>   &srcGrays,
    const PositiveDepthCameraModel &refCam,
    const std::vector<PositiveDepthCameraModel> &srcCams,
    float zNear, float zFar,
    const PatchMatchConfig       &config,
    cv::Mat                      &depthOut,
    cv::Mat                      *confOut,
    std::string                  *errorMsg,
    const cv::Mat                *hintDepth)
{
    const int refW = refGray.cols, refH = refGray.rows;
    const int N    = (int)srcGrays.size();
    const int ds   = config.downsampleFactor > 0 ? config.downsampleFactor : 1;

    // ── 降采样 ──────────────────────────────────────────────────────
    cv::Mat refScaled, hintScaled;
    cv::resize(refGray, refScaled, cv::Size(refW/ds, refH/ds), 0, 0, cv::INTER_AREA);
    const int sW = refScaled.cols, sH = refScaled.rows;

    PositiveDepthCameraModel refCamS = refCam;
    if (ds > 1) { refCamS.fx/=ds; refCamS.fy/=ds; refCamS.cx/=ds; refCamS.cy/=ds; }

    if (hintDepth && !hintDepth->empty())
        cv::resize(*hintDepth, hintScaled, cv::Size(sW, sH), 0, 0, cv::INTER_NEAREST);

    // ── 设置常量内存 c_ref_inv_K ─────────────────────────────────
    float h_inv_K[4] = 
    {
        1.f / refCamS.fx, -refCamS.cx / refCamS.fx,
        1.f / refCamS.fy, -refCamS.cy / refCamS.fy
    };
    CUDA_CHECK(cudaMemcpyToSymbol(c_ref_inv_K, h_inv_K, sizeof(float)*4));

    // ── 构建源图 srcData（每源 16 floats）───────────────────────
    std::vector<float>   srcDatas(N * 16, 0.f);
    int srcW = sW, srcH2 = sH;

    for (int si = 0; si < N; ++si) 
    {
        PositiveDepthCameraModel sc = srcCams[si];
        if (ds > 1) { sc.fx/=ds; sc.fy/=ds; sc.cx/=ds; sc.cy/=ds; }

        float *sd = srcDatas.data() + si * 16;
        sd[0]=sc.fx; sd[1]=sc.cx; sd[2]=sc.fy; sd[3]=sc.cy;

        float R_rel[9], T_rel[3];
        for (int r=0; r<3; ++r)
            for (int c=0; c<3; ++c) 
            {
                float v=0;
                for (int k=0; k<3; ++k) v += sc.R_cw[r*3+k] * refCamS.R_cw[c*3+k];
                R_rel[r*3+c] = v;
            }
        for (int r=0; r<3; ++r) 
        {
            float v=sc.T[r];
            for (int c=0; c<3; ++c) v -= R_rel[r*3+c] * refCamS.T[c];
            T_rel[r]=v;
        }
        for (int k=0; k<9; ++k) sd[4+k]=R_rel[k];
        sd[13]=T_rel[0]; sd[14]=T_rel[1]; sd[15]=T_rel[2];
    }

    // ── GPU 参数摘要 ──────────────────────────────────────────
    fprintf(stderr, "[GPU] 降采样尺寸: %d×%d  ds=%d  N源=%d  iters=%d  patch=%d\n",
            sW, sH, ds, N, config.numIterations, config.patchHalf*2+1);
    for (int si = 0; si < N; ++si) 
    {
        const float *sd = srcDatas.data() + si * 16;
        const float *T_rel = sd + 13;
        float baseline = sqrtf(T_rel[0]*T_rel[0]+T_rel[1]*T_rel[1]+T_rel[2]*T_rel[2]);
        if (baseline < 1e-4f)
            fprintf(stderr, "[GPU SRC %d] ★★ 警告: 基线≈0! ★★\n", si);
    }

    const int refPx = sW * sH;

    float *d_ref=nullptr, *d_srcData=nullptr;
    float *d_depth=nullptr, *d_normal=nullptr, *d_conf=nullptr, *d_hint=nullptr;
    float **d_srcPtrs = nullptr;

    CUDA_CHECK(cudaMalloc(&d_srcData, N*16    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depth,   refPx   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normal,  refPx*3 * sizeof(float)));  // 法向量图
    CUDA_CHECK(cudaMalloc(&d_conf,    refPx   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_srcPtrs, N * sizeof(float*)));

    CUDA_CHECK(cudaMemcpy(d_srcData, srcDatas.data(),     N*16*sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_conf, 0, refPx * sizeof(float)));

    std::vector<float*> hSrcPtrs(static_cast<size_t>(N), nullptr);
    bool refCacheHit = false;
    int cacheHits = 0;
    int cacheMisses = 0;
    if (!getOrUploadGrayImageGpu(refGray, sW, sH, ds, &d_ref, &refCacheHit, errorMsg))
    {
        if (d_srcData) cudaFree(d_srcData);
        if (d_depth) cudaFree(d_depth);
        if (d_normal) cudaFree(d_normal);
        if (d_conf) cudaFree(d_conf);
        if (d_srcPtrs) cudaFree(d_srcPtrs);
        return false;
    }

    for (int si = 0; si < N; ++si)
    {
        bool cacheHit = false;
        if (!getOrUploadGrayImageGpu(srcGrays[si], sW, sH, ds, &hSrcPtrs[si], &cacheHit, errorMsg))
        {
            if (d_srcData) cudaFree(d_srcData);
            if (d_depth) cudaFree(d_depth);
            if (d_normal) cudaFree(d_normal);
            if (d_conf) cudaFree(d_conf);
            if (d_srcPtrs) cudaFree(d_srcPtrs);
            return false;
        }
        if (cacheHit)
        {
            ++cacheHits;
        }
        else
        {
            ++cacheMisses;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_srcPtrs, hSrcPtrs.data(), N * sizeof(float*), cudaMemcpyHostToDevice));

    bool hasHint = !hintScaled.empty();
    if (hasHint) 
    {
        cv::Mat hF; hintScaled.convertTo(hF, CV_32F);
        CUDA_CHECK(cudaMalloc(&d_hint, refPx*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_hint, hF.ptr<float>(), refPx*sizeof(float), cudaMemcpyHostToDevice));
    }

    // ── 初始化深度 + 法向量 ───────────────────────────────────────
    const int bW = config.cudaBlockW;
    const int bH = config.cudaBlockH;
    const int bS = config.cudaBlockSweep;
    dim3 block(bW, bH);
    dim3 grid((sW + bW - 1) / bW, (sH + bH - 1) / bH);

    kernelInitDepthNormal<<<grid, block>>>(
        d_depth, d_normal, d_hint, sW, sH, zNear, zFar, 42ULL);
    CUDA_CHECK(cudaGetLastError());

    // ── 迭代传播 + 精化（4 向 sweep，仿照 COLMAP）───────────────────
    // 每次迭代做 4 次方向扫描：TtoB → BtoT → LtoR → RtoL
    // perturbation 指数退火：1.0 → 0.5 → 0.25 ...
    dim3 blockSweep(bS);
    dim3 gridCols((sW + bS - 1) / bS);
    dim3 gridRows((sH + bS - 1) / bS);

    float perturbation = 1.0f;
    for (int iter = 0; iter < config.numIterations; ++iter) 
    {
        unsigned long long baseSeed = (unsigned long long)(iter + 1) * 999983ULL;

        // 自上而下
        kernelSweepTB<<<gridCols, blockSweep>>>(
            d_depth, d_normal, d_conf,
            d_ref, sW, sH,
            d_srcPtrs, d_srcData, srcW, srcH2, N,
            zNear, zFar, config.patchHalf,
            perturbation, baseSeed);
        CUDA_CHECK(cudaGetLastError());

        // 自下而上
        kernelSweepBT<<<gridCols, blockSweep>>>(
            d_depth, d_normal, d_conf,
            d_ref, sW, sH,
            d_srcPtrs, d_srcData, srcW, srcH2, N,
            zNear, zFar, config.patchHalf,
            perturbation, baseSeed + 111111ULL);
        CUDA_CHECK(cudaGetLastError());

        // 自左向右
        kernelSweepLR<<<gridRows, blockSweep>>>(
            d_depth, d_normal, d_conf,
            d_ref, sW, sH,
            d_srcPtrs, d_srcData, srcW, srcH2, N,
            zNear, zFar, config.patchHalf,
            perturbation, baseSeed + 222222ULL);
        CUDA_CHECK(cudaGetLastError());

        // 自右向左
        kernelSweepRL<<<gridRows, blockSweep>>>(
            d_depth, d_normal, d_conf,
            d_ref, sW, sH,
            d_srcPtrs, d_srcData, srcW, srcH2, N,
            zNear, zFar, config.patchHalf,
            perturbation, baseSeed + 333333ULL);
        CUDA_CHECK(cudaGetLastError());

        perturbation = fmaxf(perturbation * 0.5f, 0.02f);
    }

    // ── 置信度过滤 ────────────────────────────────────────────────
    kernelFinalizeDepth<<<grid, block>>>(d_depth, d_conf, sW, sH, config.confidenceThresh);
    CUDA_CHECK(cudaGetLastError());

    // ── 拷回 CPU ─────────────────────────────────────────────────
    cv::Mat depthS(sH, sW, CV_32F), confS(sH, sW, CV_32F);
    CUDA_CHECK(cudaMemcpy(depthS.ptr<float>(), d_depth, refPx*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(confS.ptr<float>(),  d_conf,  refPx*sizeof(float), cudaMemcpyDeviceToHost));

    // ── 后处理结果统计诊断 ────────────────────────────────────────
    {
        cv::Mat validMask = (depthS > 0);
        int validCnt = cv::countNonZero(validMask);
        int totalPx  = sW * sH;
        double dMin=0, dMax=0, dMean=0;
        if (validCnt > 0) 
        {
            cv::minMaxLoc(depthS, &dMin, &dMax, nullptr, nullptr, validMask);
            dMean = cv::mean(depthS, validMask)[0];
        }
        double cMean = validCnt > 0 ? cv::mean(confS, validMask)[0] : 0;
        fprintf(stderr, "[GPU RESULT] 有效像素: %d/%d (%.1f%%)  深度: [%.4f, %.4f] mean=%.4f  conf=%.4f\n",
                validCnt, totalPx, 100.f*validCnt/totalPx,
                (float)dMin, (float)dMax, (float)dMean, (float)cMean);

        if (validCnt == 0) 
        {
            fprintf(stderr, "[GPU RESULT] ★★ 警告: 0 个有效深度像素! ★★\n");
        }
    }


    cudaFree(d_srcData); cudaFree(d_srcPtrs);
    cudaFree(d_depth); cudaFree(d_normal); cudaFree(d_conf);
    if (d_hint) cudaFree(d_hint);

    // ── 后处理（在降采样分辨率上做，再 upscale → 更快 ×4+）───────
    if (config.doMedianBlur && config.medianKernelSize > 1) 
    {
        cv::Mat validMask = (depthS > 0);
        cv::Mat tmp; cv::medianBlur(depthS, tmp, config.medianKernelSize);
        tmp.copyTo(depthS, validMask);
    }
    if (config.doBilateralFilter) 
    {
        cv::Mat validMask = (depthS > 0);
        cv::Mat tmp;
        cv::bilateralFilter(depthS, tmp, config.bilateralD,
                            config.bilateralSigmaColor, config.bilateralSigmaSpace);
        depthS = cv::Mat::zeros(depthS.rows, depthS.cols, CV_32F);
        tmp.copyTo(depthS, validMask);
    }

    // ── 上采样到原始分辨率 ───────────────────────────────────────
    cv::Mat depthFull = depthS, confFull = confS;
    if (ds > 1) 
    {
        // 深度图和置信度图必须使用最近邻插值上采样，避免在有效/无效像素
        // 边界产生错误的中间值（双线性插值会把 0 和 valid 混合出
        // 远低于 zNear 的伪深度值，导致大量点云离群点）
        cv::resize(depthS, depthFull, cv::Size(refW, refH), 0, 0, cv::INTER_NEAREST);
        cv::resize(confS,  confFull,  cv::Size(refW, refH), 0, 0, cv::INTER_NEAREST);
    }

    depthOut = depthFull;
    if (confOut) *confOut = confFull;

    int valid = cv::countNonZero(depthFull > 0);
    float mean = (valid>0) ? (float)cv::mean(depthFull, depthFull>0)[0] : 0.f;
    fprintf(stderr,
        "[PatchMatchGPU] %dx%d zNear=%.2f zFar=%.2f valid=%d/%d mean=%.2f "
        "conf>=%.2f hint=%d iters=%d\n",
        sW, sH, zNear, zFar, valid, sW*sH, mean,
        config.confidenceThresh, hasHint?1:0, config.numIterations);
    fprintf(stderr,
        "[PatchMatchGPU] gray cache refHit=%d srcHit=%d srcMiss=%d usage=%.1f/%.1f MB\n",
        refCacheHit ? 1 : 0,
        cacheHits,
        cacheMisses,
        static_cast<float>(g_srcImageGpuCacheBytes) / (1024.0f * 1024.0f),
        static_cast<float>(getGrayImageGpuCacheLimitBytes()) / (1024.0f * 1024.0f));
    return true;
}

// =============================================================================
// CPU fallback（平面扫描）
// =============================================================================
bool PatchMatchDepthEstimator::estimateCPU(
    const cv::Mat                &refGray,
    const std::vector<cv::Mat>   &srcGrays,
    const PositiveDepthCameraModel &refCam,
    const std::vector<PositiveDepthCameraModel> &srcCams,
    float zNear, float zFar,
    const PatchMatchConfig       &config,
    cv::Mat                      &depthOut,
    cv::Mat                      *confOut,
    std::string                  */*errorMsg*/,
    const cv::Mat                *hintDepth)
{
    const int refW = refGray.cols;
    const int refH = refGray.rows;
    const int N = static_cast<int>(srcGrays.size());
    const int ds = config.downsampleFactor > 0 ? config.downsampleFactor : 1;
    const int cpuThreadCount = std::max(1, config.cpuThreadCount);

    cv::Mat refScaled;
    cv::resize(refGray, refScaled,
               cv::Size(std::max(1, refW / ds), std::max(1, refH / ds)),
               0, 0, cv::INTER_AREA);
    const int W = refScaled.cols;
    const int H = refScaled.rows;

    PositiveDepthCameraModel refCamS = refCam;
    if (ds > 1)
    {
        refCamS.fx /= ds;
        refCamS.fy /= ds;
        refCamS.cx /= ds;
        refCamS.cy /= ds;
    }

    cv::Mat hintScaled;
    if (hintDepth && !hintDepth->empty())
    {
        cv::resize(*hintDepth, hintScaled, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
    }

    const float invK[4] = {
        1.f / refCamS.fx, -refCamS.cx / refCamS.fx,
        1.f / refCamS.fy, -refCamS.cy / refCamS.fy
    };

    // float 化
    cv::Mat refF;
    refScaled.convertTo(refF, CV_32F, 1.f / 255.f);
    std::vector<cv::Mat> srcF(N);
    for (int si = 0; si < N; ++si)
    {
        cv::Mat srcScaled;
        cv::resize(srcGrays[si], srcScaled, cv::Size(W, H), 0, 0, cv::INTER_AREA);
        srcScaled.convertTo(srcF[si], CV_32F, 1.f / 255.f);
    }

    // 预计算 srcData
    std::vector<float> srcDatas(N * 16, 0.f);
    for (int si = 0; si < N; ++si)
    {
        float *sd = srcDatas.data() + si * 16;
        PositiveDepthCameraModel sc = srcCams[si];
        if (ds > 1)
        {
            sc.fx /= ds;
            sc.fy /= ds;
            sc.cx /= ds;
            sc.cy /= ds;
        }
        sd[0] = sc.fx; sd[1] = sc.cx; sd[2] = sc.fy; sd[3] = sc.cy;
        float R_rel[9], T_rel[3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
            {
                float v = 0;
                for (int k = 0; k < 3; ++k)
                    v += sc.R_cw[r * 3 + k] * refCamS.R_cw[c * 3 + k];
                R_rel[r * 3 + c] = v;
            }
        for (int r = 0; r < 3; ++r)
        {
            float v = sc.T[r];
            for (int c = 0; c < 3; ++c)
            {
                v -= R_rel[r * 3 + c] * refCamS.T[c];
            }
            T_rel[r] = v;
        }
        for (int k = 0; k < 9; ++k) sd[4+k] = R_rel[k];
        sd[13] = T_rel[0]; sd[14] = T_rel[1]; sd[15] = T_rel[2];
    }

    cv::Mat depthS(H, W, CV_32F, cv::Scalar(0.f));
    cv::Mat confS(H, W, CV_32F, cv::Scalar(0.f));
    cv::Mat normalS(H, W, CV_32FC3, cv::Scalar(0.f, 0.f, -1.f));

    cpuParallelForLines(H, cpuThreadCount, [&](int row) {
        std::mt19937 rng(static_cast<uint32_t>(42ULL + static_cast<unsigned long long>(row)));
        for (int col = 0; col < W; ++col)
        {
            const float hint = !hintScaled.empty() ? hintScaled.at<float>(row, col) : 0.f;
            float depth = 0.f;
            if (hint > 0.f && hint >= zNear && hint <= zFar)
            {
                const float range = hint * 0.3f;
                depth = hint + cpuUniformSigned(rng) * range;
                depth = std::max(zNear, std::min(zFar, depth));
            }
            else
            {
                depth = zNear + cpuUniformUnit(rng) * (zFar - zNear);
            }
            depthS.at<float>(row, col) = depth;
            const CpuNormal normal = cpuGenerateRandomNormal(row, col, invK, rng);
            normalS.at<cv::Vec3f>(row, col) = cv::Vec3f(normal[0], normal[1], normal[2]);
        }
    });

    auto clampDepth = [&](float depth) {
        return std::max(zNear, std::min(zFar, depth));
    };

    auto runColumnSweep = [&](bool topToBottom, unsigned long long seed, float perturbation) {
        cpuParallelForLines(W, cpuThreadCount, [&](int col) {
            std::mt19937 rng(static_cast<uint32_t>(seed + static_cast<unsigned long long>(col)));
            const int rowStart = topToBottom ? 0 : (H - 1);
            const int rowEnd = topToBottom ? H : -1;
            const int rowStep = topToBottom ? 1 : -1;

            int prevRow = rowStart;
            float prevDepth = depthS.at<float>(prevRow, col);
            cv::Vec3f prevNormalVec = normalS.at<cv::Vec3f>(prevRow, col);
            CpuNormal prevNormal{prevNormalVec[0], prevNormalVec[1], prevNormalVec[2]};

            for (int row = rowStart; row != rowEnd; row += rowStep)
            {
                const int idxRow = row;
                const float currDepth = depthS.at<float>(idxRow, col);
                const cv::Vec3f currNormalVec = normalS.at<cv::Vec3f>(idxRow, col);
                const CpuNormal currNormal{currNormalVec[0], currNormalVec[1], currNormalVec[2]};

                float propDepth = currDepth;
                if (row != rowStart)
                {
                    propDepth = cpuPropagateDepth(prevDepth, prevNormal,
                                                  static_cast<float>(prevRow), static_cast<float>(col),
                                                  static_cast<float>(row), static_cast<float>(col), invK);
                    propDepth = clampDepth(propDepth);
                }

                float randDepth = clampDepth(cpuPerturbDepth(perturbation, currDepth, rng));
                CpuNormal randNormal = cpuPerturbNormal(row, col, perturbation * static_cast<float>(M_PI),
                                                        currNormal, invK, rng);

                const std::array<float, 5> dep{currDepth, propDepth, randDepth, currDepth, randDepth};
                const std::array<CpuNormal, 5> nor{{
                    currNormal,
                    prevNormal,
                    randNormal,
                    randNormal,
                    currNormal
                }};

                float bestCost = 2.f;
                int bestIndex = 0;
                for (int hi = 0; hi < 5; ++hi)
                {
                    const float cost = cpuEvalHypCost(col, row, dep[hi], nor[hi],
                                                      refF, srcF, srcDatas,
                                                      config.patchHalf, invK);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestIndex = hi;
                    }
                }

                depthS.at<float>(idxRow, col) = dep[bestIndex];
                const CpuNormal &bestNormal = nor[bestIndex];
                normalS.at<cv::Vec3f>(idxRow, col) = cv::Vec3f(bestNormal[0], bestNormal[1], bestNormal[2]);
                confS.at<float>(idxRow, col) = 1.f - bestCost * 0.5f;

                prevRow = row;
                prevDepth = dep[bestIndex];
                prevNormal = bestNormal;
            }
        });
    };

    auto runRowSweep = [&](bool leftToRight, unsigned long long seed, float perturbation) {
        cpuParallelForLines(H, cpuThreadCount, [&](int row) {
            std::mt19937 rng(static_cast<uint32_t>(seed + static_cast<unsigned long long>(row)));
            const int colStart = leftToRight ? 0 : (W - 1);
            const int colEnd = leftToRight ? W : -1;
            const int colStep = leftToRight ? 1 : -1;

            int prevCol = colStart;
            float prevDepth = depthS.at<float>(row, prevCol);
            cv::Vec3f prevNormalVec = normalS.at<cv::Vec3f>(row, prevCol);
            CpuNormal prevNormal{prevNormalVec[0], prevNormalVec[1], prevNormalVec[2]};

            for (int col = colStart; col != colEnd; col += colStep)
            {
                const float currDepth = depthS.at<float>(row, col);
                const cv::Vec3f currNormalVec = normalS.at<cv::Vec3f>(row, col);
                const CpuNormal currNormal{currNormalVec[0], currNormalVec[1], currNormalVec[2]};

                float propDepth = currDepth;
                if (col != colStart)
                {
                    propDepth = cpuPropagateDepth(prevDepth, prevNormal,
                                                  static_cast<float>(row), static_cast<float>(prevCol),
                                                  static_cast<float>(row), static_cast<float>(col), invK);
                    propDepth = clampDepth(propDepth);
                }

                float randDepth = clampDepth(cpuPerturbDepth(perturbation, currDepth, rng));
                CpuNormal randNormal = cpuPerturbNormal(row, col, perturbation * static_cast<float>(M_PI),
                                                        currNormal, invK, rng);

                const std::array<float, 5> dep{currDepth, propDepth, randDepth, currDepth, randDepth};
                const std::array<CpuNormal, 5> nor{{
                    currNormal,
                    prevNormal,
                    randNormal,
                    randNormal,
                    currNormal
                }};

                float bestCost = 2.f;
                int bestIndex = 0;
                for (int hi = 0; hi < 5; ++hi)
                {
                    const float cost = cpuEvalHypCost(col, row, dep[hi], nor[hi],
                                                      refF, srcF, srcDatas,
                                                      config.patchHalf, invK);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestIndex = hi;
                    }
                }

                depthS.at<float>(row, col) = dep[bestIndex];
                const CpuNormal &bestNormal = nor[bestIndex];
                normalS.at<cv::Vec3f>(row, col) = cv::Vec3f(bestNormal[0], bestNormal[1], bestNormal[2]);
                confS.at<float>(row, col) = 1.f - bestCost * 0.5f;

                prevCol = col;
                prevDepth = dep[bestIndex];
                prevNormal = bestNormal;
            }
        });
    };

    float perturbation = 1.0f;
    for (int iter = 0; iter < config.numIterations; ++iter)
    {
        const unsigned long long baseSeed = static_cast<unsigned long long>(iter + 1) * 999983ULL;
        runColumnSweep(true, baseSeed, perturbation);
        runColumnSweep(false, baseSeed + 111111ULL, perturbation);
        runRowSweep(true, baseSeed + 222222ULL, perturbation);
        runRowSweep(false, baseSeed + 333333ULL, perturbation);
        perturbation = std::max(perturbation * 0.5f, 0.02f);
    }

    for (int row = 0; row < H; ++row)
    {
        float *depthPtr = depthS.ptr<float>(row);
        const float *confPtr = confS.ptr<float>(row);
        for (int col = 0; col < W; ++col)
        {
            if (confPtr[col] < config.confidenceThresh)
            {
                depthPtr[col] = 0.f;
            }
        }
    }

        fprintf(stderr, "[PatchMatchCPU] %dx%d ds=%d threads=%d validRows=%d\n",
            W, H, ds, cpuThreadCount, H);

    if (config.doMedianBlur && config.medianKernelSize > 1)
    {
        cv::Mat tmp;
        cv::Mat validMask = (depthS > 0);
        cv::medianBlur(depthS, tmp, config.medianKernelSize);
        tmp.copyTo(depthS, validMask);
    }

    if (config.doBilateralFilter)
    {
        cv::Mat validMask = (depthS > 0);
        cv::Mat tmp;
        cv::bilateralFilter(depthS, tmp, config.bilateralD,
                            config.bilateralSigmaColor, config.bilateralSigmaSpace);
        depthS = cv::Mat::zeros(depthS.rows, depthS.cols, CV_32F);
        tmp.copyTo(depthS, validMask);
    }

    cv::Mat depthFull = depthS;
    cv::Mat confFull = confS;
    if (ds > 1)
    {
        cv::resize(depthS, depthFull, cv::Size(refW, refH), 0, 0, cv::INTER_NEAREST);
        cv::resize(confS, confFull, cv::Size(refW, refH), 0, 0, cv::INTER_NEAREST);
    }

    depthOut = depthFull;
    if (confOut)
    {
        *confOut = confFull;
    }
    return true;
}

// =============================================================================
// 公有接口
// =============================================================================
bool PatchMatchDepthEstimator::isCudaAvailable()
{
    int cnt = 0;
    cudaGetDeviceCount(&cnt);
    return cnt > 0;
}

void PatchMatchDepthEstimator::cleanupGpuImageCache()
{
    std::lock_guard<std::mutex> lock(g_srcImageGpuCacheMutex);
    for (auto &kv : g_srcImageGpuCache)
    {
        if (kv.second.devicePtr)
        {
            cudaFree(kv.second.devicePtr);
            kv.second.devicePtr = nullptr;
        }
    }
    g_srcImageGpuCache.clear();
    g_srcImageGpuCacheBytes = 0;
}

bool PatchMatchDepthEstimator::estimate(
    const cv::Mat                &refGray,
    const std::vector<cv::Mat>   &srcGrays,
    const PositiveDepthCameraModel &refCam,
    const std::vector<PositiveDepthCameraModel> &srcCams,
    float zNear, float zFar,
    const PatchMatchConfig       &config,
    cv::Mat                      &depthOut,
    cv::Mat                      *confOut,
    std::string                  *errorMsg,
    const cv::Mat                *hintDepth)
{
    if (srcGrays.empty() || srcCams.empty() || srcGrays.size() != srcCams.size()) 
    {
        if (errorMsg) *errorMsg = "源帧数量不匹配或为空";
        return false;
    }
    if (!refCam.valid()) 
    {
        if (errorMsg) *errorMsg = "参考相机参数无效";
        return false;
    }
    zNear = std::max(zNear, 0.01f);
    zFar  = std::max(zFar, zNear + 0.1f);

    if (config.useCuda && isCudaAvailable()) 
    {
        if (estimateGPU(refGray, srcGrays, refCam, srcCams,
                        zNear, zFar, config, depthOut, confOut, errorMsg, hintDepth))
            return true;
        fprintf(stderr, "[PatchMatch] GPU 失败，回退 CPU\n");
    }
    return estimateCPU(refGray, srcGrays, refCam, srcCams,
                       zNear, zFar, config, depthOut, confOut, errorMsg, hintDepth);
}

} // namespace mvs
} // namespace xjw
