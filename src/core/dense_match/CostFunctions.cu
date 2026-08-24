// =============================================================================
// 文件: CostFunctions.cu
// 功能: 密集匹配代价函数 CUDA kernel 实现
// =============================================================================
#ifdef DM_ENABLE_CUDA

#include "CostFunctions.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace xjw::dense_match
{

    namespace
    {

        void checkCuda(cudaError_t status, const char* operation)
        {
            if (status != cudaSuccess)
            {
                CV_Error_(cv::Error::GpuApiCallError, ("%s failed: %s", operation, cudaGetErrorString(status)));
            }
        }

        class CudaStream
        {
        public:
            CudaStream()
            {
                checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
            }

            ~CudaStream()
            {
                if (_stream != nullptr)
                {
                    cudaStreamDestroy(_stream);
                }
            }

            CudaStream(const CudaStream&) = delete;
            CudaStream& operator=(const CudaStream&) = delete;
            CudaStream(CudaStream&&) = delete;
            CudaStream& operator=(CudaStream&&) = delete;

            [[nodiscard]] cudaStream_t get() const
            {
                return _stream;
            }

        private:
            cudaStream_t _stream = nullptr;
        };

        template <typename T> class CudaDeviceBuffer
        {
        public:
            CudaDeviceBuffer(std::size_t byteCount, const char* operation)
            {
                checkCuda(cudaMalloc(reinterpret_cast<void**>(&_data), byteCount), operation);
            }

            ~CudaDeviceBuffer()
            {
                if (_data != nullptr)
                {
                    cudaFree(_data);
                }
            }

            CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
            CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
            CudaDeviceBuffer(CudaDeviceBuffer&&) = delete;
            CudaDeviceBuffer& operator=(CudaDeviceBuffer&&) = delete;

            [[nodiscard]] T* get() const
            {
                return _data;
            }

        private:
            T* _data = nullptr;
        };

        __device__ __forceinline__ std::size_t imageOffset(int x, int y, int imageWidth)
        {
            return static_cast<std::size_t>(y) * static_cast<std::size_t>(imageWidth) + static_cast<std::size_t>(x);
        }

        __device__ float
        adCostDev(const uchar* left, const uchar* right, int x, int y, int d, int kw, int kh, int imgW, int imgH)
        {
            float sum = 0.0f;
            int count = 0;
            int halfKW = kw / 2, halfKH = kh / 2;
            for (int dy = -halfKH; dy <= halfKH; ++dy)
            {
                int ry = y + dy;
                if (ry < 0 || ry >= imgH)
                    continue;
                for (int dx = -halfKW; dx <= halfKW; ++dx)
                {
                    int lx = x + dx, rx = lx - d;
                    if (lx < 0 || lx >= imgW || rx < 0 || rx >= imgW)
                        continue;
                    sum += fabsf(static_cast<float>(left[imageOffset(lx, ry, imgW)]) -
                                 static_cast<float>(right[imageOffset(rx, ry, imgW)]));
                    ++count;
                }
            }
            return count > 0 ? sum / count : 0.0f;
        }

        __device__ float
        sdCostDev(const uchar* left, const uchar* right, int x, int y, int d, int kw, int kh, int imgW, int imgH)
        {
            float sum = 0.0f;
            int count = 0;
            int halfKW = kw / 2, halfKH = kh / 2;
            for (int dy = -halfKH; dy <= halfKH; ++dy)
            {
                int ry = y + dy;
                if (ry < 0 || ry >= imgH)
                    continue;
                for (int dx = -halfKW; dx <= halfKW; ++dx)
                {
                    int lx = x + dx, rx = lx - d;
                    if (lx < 0 || lx >= imgW || rx < 0 || rx >= imgW)
                        continue;
                    float diff = static_cast<float>(left[imageOffset(lx, ry, imgW)]) -
                                 static_cast<float>(right[imageOffset(rx, ry, imgW)]);
                    sum += diff * diff;
                    ++count;
                }
            }
            return count > 0 ? sum / count : 0.0f;
        }

        __device__ float
        nccCostDev(const uchar* left, const uchar* right, int x, int y, int d, int kw, int kh, int imgW, int imgH)
        {
            int halfKW = kw / 2, halfKH = kh / 2;
            float meanL = 0.0f, meanR = 0.0f;
            int count = 0;
            for (int dy = -halfKH; dy <= halfKH; ++dy)
            {
                int ry = y + dy;
                if (ry < 0 || ry >= imgH)
                    continue;
                for (int dx = -halfKW; dx <= halfKW; ++dx)
                {
                    int lx = x + dx, rx = lx - d;
                    if (lx < 0 || lx >= imgW || rx < 0 || rx >= imgW)
                        continue;
                    meanL += left[imageOffset(lx, ry, imgW)];
                    meanR += right[imageOffset(rx, ry, imgW)];
                    ++count;
                }
            }
            if (count == 0)
                return 0.0f;
            meanL /= count;
            meanR /= count;

            float covariance = 0.0f, varianceL = 0.0f, varianceR = 0.0f;
            for (int dy = -halfKH; dy <= halfKH; ++dy)
            {
                int ry = y + dy;
                if (ry < 0 || ry >= imgH)
                    continue;
                for (int dx = -halfKW; dx <= halfKW; ++dx)
                {
                    int lx = x + dx, rx = lx - d;
                    if (lx < 0 || lx >= imgW || rx < 0 || rx >= imgW)
                        continue;
                    float dl = static_cast<float>(left[imageOffset(lx, ry, imgW)]) - meanL;
                    float dr = static_cast<float>(right[imageOffset(rx, ry, imgW)]) - meanR;
                    covariance += dl * dr;
                    varianceL += dl * dl;
                    varianceR += dr * dr;
                }
            }
            const float stdL = sqrtf(varianceL);
            const float stdR = sqrtf(varianceR);
            constexpr float epsilon = 1.0e-8f;
            if (stdL < epsilon && stdR < epsilon)
            {
                return fabsf(meanL - meanR) < epsilon ? 0.0f : 2.0f;
            }
            if (stdL < epsilon || stdR < epsilon)
            {
                return 1.0f;
            }
            const float ncc = fmaxf(-1.0f, fminf(1.0f, covariance / (stdL * stdR)));
            return 1.0f - ncc;
        }

        __device__ float
        censusCostDev(const uchar* left, const uchar* right, int x, int y, int d, int kw, int kh, int imgW, int imgH)
        {
            int halfKW = kw / 2, halfKH = kh / 2;
            int hamming = 0, count = 0;
            uchar centerL = left[imageOffset(x, y, imgW)];
            uchar centerR = right[imageOffset(x - d, y, imgW)];
            for (int dy = -halfKH; dy <= halfKH; ++dy)
            {
                int ry = y + dy;
                if (ry < 0 || ry >= imgH)
                    continue;
                for (int dx = -halfKW; dx <= halfKW; ++dx)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int lx = x + dx, rx = lx - d;
                    if (lx < 0 || lx >= imgW || rx < 0 || rx >= imgW)
                        continue;
                    int bitL = (left[imageOffset(lx, ry, imgW)] > centerL) ? 1 : 0;
                    int bitR = (right[imageOffset(rx, ry, imgW)] > centerR) ? 1 : 0;
                    hamming += (bitL != bitR);
                    ++count;
                }
            }
            return count > 0 ? (float)hamming / count : kInvalidCost;
        }

        __device__ float ternaryCensusCostDev(
            const uchar* left, const uchar* right, int x, int y, int d, int kw, int kh, int imgW, int imgH)
        {
            const int tau = 5;
            int halfKW = kw / 2, halfKH = kh / 2;
            int hamming = 0, validCount = 0;
            int centerL = static_cast<int>(left[imageOffset(x, y, imgW)]);
            int centerR = static_cast<int>(right[imageOffset(x - d, y, imgW)]);
            for (int dy = -halfKH; dy <= halfKH; ++dy)
            {
                int ry = y + dy;
                if (ry < 0 || ry >= imgH)
                    continue;
                for (int dx = -halfKW; dx <= halfKW; ++dx)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int lx = x + dx, rx = lx - d;
                    if (lx < 0 || lx >= imgW || rx < 0 || rx >= imgW)
                        continue;
                    int leftDifference = static_cast<int>(left[imageOffset(lx, ry, imgW)]) - centerL;
                    int rightDifference = static_cast<int>(right[imageOffset(rx, ry, imgW)]) - centerR;
                    int vL = (leftDifference > tau) ? 1 : (leftDifference < -tau) ? 0 : 2;
                    int vR = (rightDifference > tau) ? 1 : (rightDifference < -tau) ? 0 : 2;
                    if (vL != 2 && vR != 2)
                    {
                        hamming += (vL != vR);
                        ++validCount;
                    }
                }
            }
            return validCount > 0 ? static_cast<float>(hamming) / validCount : kInvalidCost;
        }

        __global__ void computeCostVolumeKernel(const uchar* left,
                                                const uchar* right,
                                                float* costVolume,
                                                int imgW,
                                                int imgH,
                                                int minDisp,
                                                int maxDisp,
                                                int numDisp,
                                                int kernelW,
                                                int kernelH,
                                                int costFunc)
        {
            const std::size_t xIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t yIndex = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
            if (xIndex >= static_cast<std::size_t>(imgW) || yIndex >= static_cast<std::size_t>(imgH))
            {
                return;
            }
            const int x = static_cast<int>(xIndex);
            const int y = static_cast<int>(yIndex);

            int validStart = max(minDisp, x - imgW + 1);
            int validEnd = min(maxDisp, x + 1);
            const std::size_t planeStride = static_cast<std::size_t>(imgW) * static_cast<std::size_t>(imgH);
            const std::size_t pixelOffset = imageOffset(x, y, imgW);

            for (int dIdx = 0; dIdx < numDisp; ++dIdx)
            {
                int d = minDisp + dIdx;
                float c = kInvalidCost;
                if (d >= validStart && d < validEnd)
                {
                    switch (costFunc)
                    {
                    case 0:
                        c = adCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
                        break;
                    case 1:
                        c = sdCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
                        break;
                    case 2:
                        c = nccCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
                        break;
                    case 3:
                        c = censusCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
                        break;
                    case 4:
                        c = ternaryCensusCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
                        break;
                    }
                }
                const std::size_t volumeOffset = static_cast<std::size_t>(dIdx) * planeStride + pixelOffset;
                costVolume[volumeOffset] = c;
            }
        }

        __global__ void selectCostVolumeKernel(const float* costVolume,
                                               float* disparity,
                                               float* confidence,
                                               uchar* validMask,
                                               int imgW,
                                               int imgH,
                                               int minDisp,
                                               int numDisp,
                                               int subpixelMode)
        {
            const std::size_t xIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t yIndex = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
            if (xIndex >= static_cast<std::size_t>(imgW) || yIndex >= static_cast<std::size_t>(imgH))
            {
                return;
            }

            const std::size_t planeStride = static_cast<std::size_t>(imgW) * static_cast<std::size_t>(imgH);
            const std::size_t pixelOffset = yIndex * static_cast<std::size_t>(imgW) + xIndex;
            float bestCost = kInvalidCost;
            float secondBestCost = kInvalidCost;
            int bestIndex = -1;
            int candidateCount = 0;

            for (int disparityIndex = 0; disparityIndex < numDisp; ++disparityIndex)
            {
                const float cost = costVolume[static_cast<std::size_t>(disparityIndex) * planeStride + pixelOffset];
                if (!isfinite(cost) || cost >= kInvalidCost)
                {
                    continue;
                }
                ++candidateCount;
                if (cost < bestCost)
                {
                    secondBestCost = bestCost;
                    bestCost = cost;
                    bestIndex = disparityIndex;
                }
                else if (cost < secondBestCost)
                {
                    secondBestCost = cost;
                }
            }

            disparity[pixelOffset] = 0.0f;
            confidence[pixelOffset] = 0.0f;
            validMask[pixelOffset] = 0;
            if (bestIndex < 0)
            {
                return;
            }

            float selectedConfidence = 1.0f;
            if (candidateCount > 1)
            {
                const float scale = fmaxf(1.0f, fmaxf(fabsf(bestCost), fabsf(secondBestCost)));
                const float margin = secondBestCost - bestCost;
                if (margin <= 1.0e-6f * scale)
                {
                    return;
                }
                selectedConfidence = fminf(1.0f, fmaxf(0.0f, margin / fmaxf(fabsf(secondBestCost), 1.0e-6f)));
            }

            float selectedDisparity = static_cast<float>(minDisp + bestIndex);
            if (subpixelMode == static_cast<int>(SubpixelMode::Parabola) && bestIndex > 0 && bestIndex + 1 < numDisp)
            {
                const float previousCost =
                    costVolume[static_cast<std::size_t>(bestIndex - 1) * planeStride + pixelOffset];
                const float nextCost = costVolume[static_cast<std::size_t>(bestIndex + 1) * planeStride + pixelOffset];
                if (isfinite(previousCost) && previousCost < kInvalidCost && isfinite(nextCost) &&
                    nextCost < kInvalidCost)
                {
                    const float denominator = 2.0f * (previousCost + nextCost - 2.0f * bestCost);
                    if (denominator > 1.0e-10f)
                    {
                        const float delta = fminf(1.0f, fmaxf(-1.0f, (previousCost - nextCost) / denominator));
                        selectedDisparity += delta;
                    }
                }
            }

            disparity[pixelOffset] = selectedDisparity;
            confidence[pixelOffset] = selectedConfidence;
            validMask[pixelOffset] = 1;
        }

        dim3 selectionGrid(int imageWidth, int imageHeight, dim3 block)
        {
            return dim3((static_cast<unsigned int>(imageWidth) + block.x - 1U) / block.x,
                        (static_cast<unsigned int>(imageHeight) + block.y - 1U) / block.y);
        }

        void launchCostVolume(const uchar* deviceLeft,
                              const uchar* deviceRight,
                              float* deviceVolume,
                              int imageWidth,
                              int imageHeight,
                              int minDisparity,
                              int maxDisparity,
                              int numDisparities,
                              int kernelWidth,
                              int kernelHeight,
                              CostFunction function,
                              cudaStream_t stream)
        {
            const dim3 block(16, 16);
            computeCostVolumeKernel<<<selectionGrid(imageWidth, imageHeight, block), block, 0, stream>>>(
                deviceLeft,
                deviceRight,
                deviceVolume,
                imageWidth,
                imageHeight,
                minDisparity,
                maxDisparity,
                numDisparities,
                kernelWidth,
                kernelHeight,
                static_cast<int>(function));
            checkCuda(cudaGetLastError(), "computeCostVolumeKernel launch");
        }

        void launchSelection(const float* deviceVolume,
                             float* deviceDisparity,
                             float* deviceConfidence,
                             uchar* deviceValidMask,
                             int imageWidth,
                             int imageHeight,
                             int minDisparity,
                             int numDisparities,
                             SubpixelMode subpixel,
                             cudaStream_t stream)
        {
            const dim3 block(16, 16);
            selectCostVolumeKernel<<<selectionGrid(imageWidth, imageHeight, block), block, 0, stream>>>(
                deviceVolume,
                deviceDisparity,
                deviceConfidence,
                deviceValidMask,
                imageWidth,
                imageHeight,
                minDisparity,
                numDisparities,
                static_cast<int>(subpixel));
            checkCuda(cudaGetLastError(), "selectCostVolumeKernel launch");
        }

        DisparityResult downloadSelection(const CudaDeviceBuffer<float>& deviceDisparity,
                                          const CudaDeviceBuffer<float>& deviceConfidence,
                                          const CudaDeviceBuffer<uchar>& deviceValidMask,
                                          cv::Size imageSize,
                                          const CostVolumeBufferLayout& layout,
                                          cudaStream_t stream)
        {
            DisparityResult result;
            result.disparity = cv::Mat(imageSize, CV_32FC1, cv::Scalar(0));
            result.confidence = cv::Mat(imageSize, CV_32FC1, cv::Scalar(0));
            result.validMask = cv::Mat(imageSize, CV_8UC1, cv::Scalar(0));
            checkCuda(cudaMemcpyAsync(result.disparity.ptr<float>(),
                                      deviceDisparity.get(),
                                      layout.planeBytes,
                                      cudaMemcpyDeviceToHost,
                                      stream),
                      "cudaMemcpyAsync(disparity)");
            checkCuda(cudaMemcpyAsync(result.confidence.ptr<float>(),
                                      deviceConfidence.get(),
                                      layout.planeBytes,
                                      cudaMemcpyDeviceToHost,
                                      stream),
                      "cudaMemcpyAsync(confidence)");
            checkCuda(cudaMemcpyAsync(result.validMask.ptr<uchar>(),
                                      deviceValidMask.get(),
                                      layout.imageBytes,
                                      cudaMemcpyDeviceToHost,
                                      stream),
                      "cudaMemcpyAsync(valid mask)");
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(selection download)");
            return result;
        }

    } // anonymous namespace

    bool isCostVolumeCUDAAvailable(int cudaDevice)
    {
        int deviceCount = 0;
        const cudaError_t status = cudaGetDeviceCount(&deviceCount);
        if (status != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        return cudaDevice >= 0 && cudaDevice < deviceCount;
    }

    CostVolume computeCostVolumeCUDA(const cv::Mat& left,
                                     const cv::Mat& right,
                                     int minDisp,
                                     int maxDisp,
                                     int kernelW,
                                     int kernelH,
                                     CostFunction func,
                                     int cudaDevice)
    {
        CV_Assert(!left.empty() && !right.empty());
        CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
        CV_Assert(left.size() == right.size());
        CV_Assert(maxDisp > minDisp);
        CV_Assert(kernelW > 0 && kernelH > 0);

        const int imgW = left.cols;
        const int imgH = left.rows;
        const CostVolumeBufferLayout layout = checkedCostVolumeBufferLayout(imgW, imgH, minDisp, maxDisp);
        const int numDisp = layout.numDisparities;
        const cv::Mat contiguousLeft = left.isContinuous() ? left : left.clone();
        const cv::Mat contiguousRight = right.isContinuous() ? right : right.clone();
        CostVolume volume(minDisp, maxDisp, left.size());

        {
            checkCuda(cudaSetDevice(cudaDevice), "cudaSetDevice");
            CudaStream stream;
            CudaDeviceBuffer<uchar> deviceLeft(layout.imageBytes, "cudaMalloc(left)");
            CudaDeviceBuffer<uchar> deviceRight(layout.imageBytes, "cudaMalloc(right)");
            CudaDeviceBuffer<float> deviceVolume(layout.volumeBytes, "cudaMalloc(cost volume)");

            checkCuda(
                cudaMemcpyAsync(
                    deviceLeft.get(), contiguousLeft.data, layout.imageBytes, cudaMemcpyHostToDevice, stream.get()),
                "cudaMemcpyAsync(left)");
            checkCuda(
                cudaMemcpyAsync(
                    deviceRight.get(), contiguousRight.data, layout.imageBytes, cudaMemcpyHostToDevice, stream.get()),
                "cudaMemcpyAsync(right)");

            launchCostVolume(deviceLeft.get(),
                             deviceRight.get(),
                             deviceVolume.get(),
                             imgW,
                             imgH,
                             minDisp,
                             maxDisp,
                             numDisp,
                             kernelW,
                             kernelH,
                             func,
                             stream.get());
            for (int dIdx = 0; dIdx < numDisp; ++dIdx)
            {
                const std::size_t offset = static_cast<std::size_t>(dIdx) * layout.planeElementCount;
                checkCuda(cudaMemcpyAsync(volume[static_cast<std::size_t>(dIdx)].ptr<float>(),
                                          deviceVolume.get() + offset,
                                          layout.planeBytes,
                                          cudaMemcpyDeviceToHost,
                                          stream.get()),
                          "cudaMemcpyAsync(cost-volume plane)");
            }
            checkCuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize(cost volume)");
        }

        return volume;
    }

    DisparityResult selectCostVolumeCUDA(const CostVolume& volume, SubpixelMode subpixel, int cudaDevice)
    {
        if (volume.empty())
        {
            return {};
        }

        const cv::Size imageSize = volume[0].size();
        const CostVolumeBufferLayout layout = checkedCostVolumeBufferLayout(
            imageSize.width, imageSize.height, volume.minDisparity(), volume.maxDisparity());
        if (volume.size() != static_cast<std::size_t>(layout.numDisparities))
        {
            CV_Error(cv::Error::StsBadArg, "CUDA selection received an inconsistent cost volume");
        }

        checkCuda(cudaSetDevice(cudaDevice), "cudaSetDevice");
        CudaStream stream;
        CudaDeviceBuffer<float> deviceVolume(layout.volumeBytes, "cudaMalloc(cost volume)");
        CudaDeviceBuffer<float> deviceDisparity(layout.planeBytes, "cudaMalloc(disparity)");
        CudaDeviceBuffer<float> deviceConfidence(layout.planeBytes, "cudaMalloc(confidence)");
        CudaDeviceBuffer<uchar> deviceValidMask(layout.imageBytes, "cudaMalloc(valid mask)");

        for (int disparityIndex = 0; disparityIndex < layout.numDisparities; ++disparityIndex)
        {
            const cv::Mat& plane = volume[static_cast<std::size_t>(disparityIndex)];
            if (plane.type() != CV_32FC1 || plane.size() != imageSize || !plane.isContinuous())
            {
                CV_Error(cv::Error::StsBadArg, "CUDA selection requires continuous CV_32FC1 cost planes");
            }
            checkCuda(cudaMemcpyAsync(deviceVolume.get() +
                                          static_cast<std::size_t>(disparityIndex) * layout.planeElementCount,
                                      plane.ptr<float>(),
                                      layout.planeBytes,
                                      cudaMemcpyHostToDevice,
                                      stream.get()),
                      "cudaMemcpyAsync(cost-volume plane)");
        }

        launchSelection(deviceVolume.get(),
                        deviceDisparity.get(),
                        deviceConfidence.get(),
                        deviceValidMask.get(),
                        imageSize.width,
                        imageSize.height,
                        volume.minDisparity(),
                        layout.numDisparities,
                        subpixel,
                        stream.get());
        return downloadSelection(deviceDisparity, deviceConfidence, deviceValidMask, imageSize, layout, stream.get());
    }

    DisparityResult computeBlockMatchCUDA(const cv::Mat& left,
                                          const cv::Mat& right,
                                          int minDisp,
                                          int maxDisp,
                                          int kernelW,
                                          int kernelH,
                                          CostFunction func,
                                          SubpixelMode subpixel,
                                          int cudaDevice)
    {
        CV_Assert(!left.empty() && !right.empty());
        CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
        CV_Assert(left.size() == right.size());
        CV_Assert(maxDisp > minDisp);
        CV_Assert(kernelW > 0 && kernelH > 0);

        const cv::Mat contiguousLeft = left.isContinuous() ? left : left.clone();
        const cv::Mat contiguousRight = right.isContinuous() ? right : right.clone();
        const CostVolumeBufferLayout layout = checkedCostVolumeBufferLayout(left.cols, left.rows, minDisp, maxDisp);

        checkCuda(cudaSetDevice(cudaDevice), "cudaSetDevice");
        CudaStream stream;
        CudaDeviceBuffer<uchar> deviceLeft(layout.imageBytes, "cudaMalloc(left)");
        CudaDeviceBuffer<uchar> deviceRight(layout.imageBytes, "cudaMalloc(right)");
        CudaDeviceBuffer<float> deviceVolume(layout.volumeBytes, "cudaMalloc(cost volume)");
        CudaDeviceBuffer<float> deviceDisparity(layout.planeBytes, "cudaMalloc(disparity)");
        CudaDeviceBuffer<float> deviceConfidence(layout.planeBytes, "cudaMalloc(confidence)");
        CudaDeviceBuffer<uchar> deviceValidMask(layout.imageBytes, "cudaMalloc(valid mask)");
        checkCuda(cudaMemcpyAsync(
                      deviceLeft.get(), contiguousLeft.data, layout.imageBytes, cudaMemcpyHostToDevice, stream.get()),
                  "cudaMemcpyAsync(left)");
        checkCuda(cudaMemcpyAsync(
                      deviceRight.get(), contiguousRight.data, layout.imageBytes, cudaMemcpyHostToDevice, stream.get()),
                  "cudaMemcpyAsync(right)");

        launchCostVolume(deviceLeft.get(),
                         deviceRight.get(),
                         deviceVolume.get(),
                         left.cols,
                         left.rows,
                         minDisp,
                         maxDisp,
                         layout.numDisparities,
                         kernelW,
                         kernelH,
                         func,
                         stream.get());
        launchSelection(deviceVolume.get(),
                        deviceDisparity.get(),
                        deviceConfidence.get(),
                        deviceValidMask.get(),
                        left.cols,
                        left.rows,
                        minDisp,
                        layout.numDisparities,
                        subpixel,
                        stream.get());
        return downloadSelection(deviceDisparity, deviceConfidence, deviceValidMask, left.size(), layout, stream.get());
    }

} // namespace xjw::dense_match

#endif // DM_ENABLE_CUDA
