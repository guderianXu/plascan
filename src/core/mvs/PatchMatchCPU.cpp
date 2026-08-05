// =============================================================================
// 文件: PatchMatchCPU.cpp
// 模块: MVS - CPU PatchMatch 深度图估计
// =============================================================================

#include "PatchMatchCUDA.h"
#include "PatchMatchPhotometricCost.h"

#include "Logger.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace xjw
{
namespace mvs
{
namespace
{

using CpuNormal = std::array<float, 3>;


struct HostPinholeCamera
{
    float focalX = 0.0f;
    float focalY = 0.0f;
    float principalX = 0.0f;
    float principalY = 0.0f;
    std::array<float, 9> rotationWorldToCamera{};
    std::array<float, 3> translationWorldToCamera{};
};

HostPinholeCamera makeHostPinholeCamera(const Camera &camera, int downsampleFactor)
{
    const Camera::Intrinsics intrinsics = camera.intrinsics();
    const std::array<double, 9> rotation = camera.worldToCameraRotation();
    const std::array<double, 3> translation = camera.worldToCameraTranslation();
    const float scale = 1.0f / static_cast<float>(std::max(1, downsampleFactor));

    HostPinholeCamera result;
    result.focalX = static_cast<float>(intrinsics.focalX) * scale;
    result.focalY = static_cast<float>(intrinsics.focalY) * scale;
    result.principalX = static_cast<float>(intrinsics.principalX) * scale;
    result.principalY = static_cast<float>(intrinsics.principalY) * scale;
    for (int index = 0; index < 9; ++index)
    {
        result.rotationWorldToCamera[index] = static_cast<float>(rotation[index]);
    }
    for (int index = 0; index < 3; ++index)
    {
        result.translationWorldToCamera[index] = static_cast<float>(translation[index]);
    }
    return result;
}


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

inline bool cpuBilinearMaskValid(const cv::Mat &mask, float u, float v)
{
    if (mask.empty())
    {
        return true;
    }
    if (u < 0.f || v < 0.f || u >= static_cast<float>(mask.cols - 1) ||
        v >= static_cast<float>(mask.rows - 1))
    {
        return false;
    }

    const int column = static_cast<int>(u);
    const int row = static_cast<int>(v);
    return mask.at<std::uint8_t>(row, column) != 0 &&
           mask.at<std::uint8_t>(row, column + 1) != 0 &&
           mask.at<std::uint8_t>(row + 1, column) != 0 &&
           mask.at<std::uint8_t>(row + 1, column + 1) != 0;
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
                              const cv::Mat &refMask,
                              const cv::Mat &srcMask,
                              const float *srcData,
                              const float invK[4],
                              int patchHalf,
                              float minimumMaskedSupportRatio)
{
    float H[9];
    cpuComposeHomography(refRow, refCol, depth, normal, srcData, invK, H);

    PatchNccAccumulator accumulator;
    const bool mask_aware = !refMask.empty() || !srcMask.empty();

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

            if (!refMask.empty() && refMask.at<std::uint8_t>(pv, pu) == 0)
            {
                // Reference-mask exclusions are outside the photometric
                // support domain. Do not count them against the paired
                // support ratio, otherwise every foreground silhouette is
                // penalized merely because its patch overlaps background.
                continue;
            }
            const float refValue = refImage.at<float>(pv, pu);
            const float wsC = H[0] * static_cast<float>(pu) + H[1] * static_cast<float>(pv) + H[2];
            const float wsR = H[3] * static_cast<float>(pu) + H[4] * static_cast<float>(pv) + H[5];
            const float wsZ = H[6] * static_cast<float>(pu) + H[7] * static_cast<float>(pv) + H[8];
            if (std::fabs(wsZ) < 1e-6f)
            {
                accumulator.addCandidate(false);
                continue;
            }

            const float srcU = wsC / wsZ;
            const float srcV = wsR / wsZ;
            const float srcValue = cpuBilinear(srcImage, srcU, srcV);
            if (srcValue < 0.f || !cpuBilinearMaskValid(srcMask, srcU, srcV))
            {
                accumulator.addCandidate(false);
                continue;
            }

            accumulator.addCandidate(true, refValue, srcValue);
        }
    }

    return accumulator.score(mask_aware, minimumMaskedSupportRatio);
}

float cpuEvalHypCost(int col,
                     int row,
                     float depth,
                     const CpuNormal &normal,
                     const cv::Mat &refImage,
                     const std::vector<cv::Mat> &srcImages,
                     const cv::Mat &refMask,
                     const std::vector<cv::Mat> &srcMasks,
                     const std::vector<float> &srcDatas,
                     int patchHalf,
                     const float invK[4],
                     float minimumMaskedSupportRatio)
{
    if (depth <= 0.f)
    {
        return 2.f;
    }

    std::array<float, kMaxPatchMatchSourceViews> scores{};
    const int source_count = std::min(static_cast<int>(srcImages.size()),
                                      kMaxPatchMatchSourceViews);
    for (int srcIndex = 0; srcIndex < source_count; ++srcIndex)
    {
        const float *srcData = srcDatas.data() + srcIndex * 16;
        const float ncc = cpuComputeHomographyNcc(col, row, depth, normal,
                                                  refImage, srcImages[srcIndex],
                                                  refMask, srcMasks[srcIndex], srcData,
                                                  invK, patchHalf,
                                                  minimumMaskedSupportRatio);
        scores[static_cast<size_t>(srcIndex)] = ncc;
    }

    const float robust_ncc = robustMultiSourceNcc(scores.data(), source_count);
    return 2.f - 2.f * robust_ncc;
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


cv::Mat resizedBinaryMask(const cv::Mat *mask, const cv::Size &targetSize)
{
    if (mask == nullptr || mask->empty())
    {
        return cv::Mat();
    }

    cv::Mat binary;
    cv::compare(*mask, 0, binary, cv::CMP_GT);
    if (binary.size() == targetSize)
    {
        return binary.isContinuous() ? binary : binary.clone();
    }

    cv::Mat resized;
    cv::resize(binary, resized, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
    return resized.isContinuous() ? resized : resized.clone();
}

} // namespace


bool PatchMatchDepthEstimator::estimateCPU(
    const cv::Mat                &refGray,
    const std::vector<cv::Mat>   &srcGrays,
    const Camera                   &refCam,
    const std::vector<Camera>      &srcCams,
    float zNear, float zFar,
    const PatchMatchConfig       &config,
    cv::Mat                      &depthOut,
    cv::Mat                      *confOut,
    std::string                  *errorMsg,
    const cv::Mat                *hintDepth,
    const cv::Mat                *hintRadius,
    const cv::Mat                *refValidMask,
    const std::vector<cv::Mat>   *srcValidMasks)
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
    const cv::Size scaled_size(W, H);
    const cv::Mat ref_mask_scaled = resizedBinaryMask(refValidMask, scaled_size);
    std::vector<cv::Mat> source_masks_scaled(static_cast<std::size_t>(N));
    if (srcValidMasks)
    {
        for (int source_index = 0; source_index < N; ++source_index)
        {
            source_masks_scaled[static_cast<std::size_t>(source_index)] =
                resizedBinaryMask(&(*srcValidMasks)[static_cast<std::size_t>(source_index)],
                                  scaled_size);
        }
    }

    const HostPinholeCamera refCamS = makeHostPinholeCamera(refCam, ds);

    cv::Mat hintScaled;
    cv::Mat hintRadiusScaled;
    if (hintDepth && !hintDepth->empty())
    {
        if (hintDepth->cols == W && hintDepth->rows == H)
        {
            hintScaled = *hintDepth;
        }
        else
        {
            cv::resize(*hintDepth, hintScaled, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
        }
    }
    if (!hintScaled.empty() && hintRadius && !hintRadius->empty())
    {
        if (hintRadius->cols == W && hintRadius->rows == H)
        {
            hintRadiusScaled = *hintRadius;
        }
        else
        {
            cv::resize(*hintRadius,
                       hintRadiusScaled,
                       cv::Size(W, H),
                       0,
                       0,
                       cv::INTER_NEAREST);
        }
    }

    auto depthBounds = [&](int row, int column)
    {
        float local_near = zNear;
        float local_far = zFar;
        if (!hintScaled.empty())
        {
            const float center = hintScaled.at<float>(row, column);
            if (std::isfinite(center) && center > 0.0f && center >= zNear && center <= zFar)
            {
                float radius = !hintRadiusScaled.empty()
                    ? hintRadiusScaled.at<float>(row, column)
                    : 0.0f;
                if (!std::isfinite(radius) || radius <= 0.0f)
                {
                    radius = std::max(center * 0.3f, (zFar - zNear) * 0.01f);
                }
                local_near = std::max(zNear, center - radius);
                local_far = std::min(zFar, center + radius);
                if (local_far <= local_near)
                {
                    local_near = zNear;
                    local_far = zFar;
                }
            }
        }
        return std::make_pair(local_near, local_far);
    };

    const float invK[4] = {
        1.f / refCamS.focalX, -refCamS.principalX / refCamS.focalX,
        1.f / refCamS.focalY, -refCamS.principalY / refCamS.focalY
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
        const HostPinholeCamera sc = makeHostPinholeCamera(srcCams[si], ds);
        sd[0] = sc.focalX; sd[1] = sc.principalX;
        sd[2] = sc.focalY; sd[3] = sc.principalY;
        float R_rel[9], T_rel[3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
            {
                float v = 0;
                for (int k = 0; k < 3; ++k)
                    v += sc.rotationWorldToCamera[r * 3 + k]
                       * refCamS.rotationWorldToCamera[c * 3 + k];
                R_rel[r * 3 + c] = v;
            }
        for (int r = 0; r < 3; ++r)
        {
            float v = sc.translationWorldToCamera[r];
            for (int c = 0; c < 3; ++c)
            {
                v -= R_rel[r * 3 + c] * refCamS.translationWorldToCamera[c];
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
            const auto [local_near, local_far] = depthBounds(row, col);
            const float depth = local_near + cpuUniformUnit(rng) * (local_far - local_near);
            depthS.at<float>(row, col) = depth;
            const CpuNormal normal = cpuGenerateRandomNormal(row, col, invK, rng);
            normalS.at<cv::Vec3f>(row, col) = cv::Vec3f(normal[0], normal[1], normal[2]);
        }
    });

    auto clampDepth = [&](int row, int column, float depth) {
        const auto [local_near, local_far] = depthBounds(row, column);
        return std::max(local_near, std::min(local_far, depth));
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
                    propDepth = clampDepth(row, col, propDepth);
                }

                const float boundedCurrDepth = clampDepth(row, col, currDepth);
                float randDepth = clampDepth(
                    row,
                    col,
                    cpuPerturbDepth(perturbation, boundedCurrDepth, rng));
                CpuNormal randNormal = cpuPerturbNormal(row, col, perturbation * static_cast<float>(M_PI),
                                                        currNormal, invK, rng);

                const std::array<float, 5> dep{
                    boundedCurrDepth, propDepth, randDepth, boundedCurrDepth, randDepth};
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
                                                      refF, srcF,
                                                      ref_mask_scaled, source_masks_scaled,
                                                      srcDatas, config.patchHalf, invK,
                                                      config.minimumMaskedPatchSupportRatio);
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
                    propDepth = clampDepth(row, col, propDepth);
                }

                const float boundedCurrDepth = clampDepth(row, col, currDepth);
                float randDepth = clampDepth(
                    row,
                    col,
                    cpuPerturbDepth(perturbation, boundedCurrDepth, rng));
                CpuNormal randNormal = cpuPerturbNormal(row, col, perturbation * static_cast<float>(M_PI),
                                                        currNormal, invK, rng);

                const std::array<float, 5> dep{
                    boundedCurrDepth, propDepth, randDepth, boundedCurrDepth, randDepth};
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
                                                      refF, srcF,
                                                      ref_mask_scaled, source_masks_scaled,
                                                      srcDatas, config.patchHalf, invK,
                                                      config.minimumMaskedPatchSupportRatio);
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
        if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
        {
            if (errorMsg) *errorMsg = "PatchMatch cancelled";
            return false;
        }

        const unsigned long long baseSeed = static_cast<unsigned long long>(iter + 1) * 999983ULL;
        runColumnSweep(true, baseSeed, perturbation);
        runColumnSweep(false, baseSeed + 111111ULL, perturbation);
        runRowSweep(true, baseSeed + 222222ULL, perturbation);
        runRowSweep(false, baseSeed + 333333ULL, perturbation);
        perturbation = std::max(perturbation * 0.5f, 0.02f);
    }

    if (config.enablePhotometricUniqueness && N >= 3)
    {
        cpuParallelForLines(H, cpuThreadCount, [&](int row)
        {
            float *confidence_row = confS.ptr<float>(row);
            const float *depth_row = depthS.ptr<float>(row);
            const cv::Vec3f *normal_row = normalS.ptr<cv::Vec3f>(row);
            for (int col = 0; col < W; ++col)
            {
                const float depth = depth_row[col];
                const float best_ncc = confidence_row[col];
                if (!(depth > 0.0f) || !(best_ncc > 0.0f))
                {
                    continue;
                }

                const CpuNormal normal{
                    normal_row[col][0], normal_row[col][1], normal_row[col][2]};
                float competing_ncc = 0.0f;
                const float relative_step =
                    config.photometricUniquenessRelativeDepthStep;
                const float lower_depth = std::max(
                    zNear, depth * (1.0f - relative_step));
                const float upper_depth = std::min(
                    zFar, depth * (1.0f + relative_step));
                const float minimum_distinct_depth = std::max(
                    1e-6f, depth * relative_step * 0.25f);
                if (depth - lower_depth >= minimum_distinct_depth)
                {
                    const float cost = cpuEvalHypCost(
                        col, row, lower_depth, normal,
                        refF, srcF,
                        ref_mask_scaled, source_masks_scaled,
                        srcDatas, config.patchHalf, invK,
                        config.minimumMaskedPatchSupportRatio);
                    competing_ncc = std::max(
                        competing_ncc, 1.0f - cost * 0.5f);
                }
                if (upper_depth - depth >= minimum_distinct_depth)
                {
                    const float cost = cpuEvalHypCost(
                        col, row, upper_depth, normal,
                        refF, srcF,
                        ref_mask_scaled, source_masks_scaled,
                        srcDatas, config.patchHalf, invK,
                        config.minimumMaskedPatchSupportRatio);
                    competing_ncc = std::max(
                        competing_ncc, 1.0f - cost * 0.5f);
                }
                confidence_row[col] = best_ncc *
                    photometricUniquenessConfidenceScale(
                        best_ncc,
                        competing_ncc,
                        config.photometricUniquenessMinimumMargin,
                        config.photometricUniquenessMinimumConfidenceScale);
            }
        });
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

    if (!ref_mask_scaled.empty())
    {
        const cv::Mat invalid_reference = ref_mask_scaled == 0;
        depthS.setTo(cv::Scalar(0.0f), invalid_reference);
        confS.setTo(cv::Scalar(0.0f), invalid_reference);
    }

    LOG_DEBUG("[MVS][PatchMatch][CPU] size=%dx%d ds=%d threads=%d valid_rows=%d",
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

} // namespace mvs
} // namespace xjw
