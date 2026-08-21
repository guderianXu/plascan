#include "SiftComputeBackend.h"

#include "cudaImage.h"
#include "cudaSift.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace xjw::image_matching
{
    namespace
    {

        void checkCuda(cudaError_t status, const char* operation)
        {
            if (status != cudaSuccess)
            {
                throw std::runtime_error(std::string("SIFT CUDA backend ") + operation + ": " +
                                         cudaGetErrorString(status));
            }
        }

        class SiftDataOwner
        {
        public:
            SiftDataOwner(int maximumPoints, bool populate_host)
            {
                InitSiftData(_data, std::max(1, maximumPoints), populate_host, true);
            }

            explicit SiftDataOwner(const cv::Mat& descriptors) : SiftDataOwner(descriptors.rows, true)
            {
                _data.numPts = descriptors.rows;
#ifdef MANAGEDMEM
                SiftPoint* points = _data.m_data;
#else
                SiftPoint* points = _data.h_data;
#endif
                std::memset(points, 0, sizeof(SiftPoint) * static_cast<std::size_t>(descriptors.rows));
                for (int index = 0; index < descriptors.rows; ++index)
                {
                    std::copy(descriptors.ptr<float>(index), descriptors.ptr<float>(index) + 128, points[index].data);
                }
#ifndef MANAGEDMEM
                checkCuda(cudaMemcpy(_data.d_data,
                                     _data.h_data,
                                     sizeof(SiftPoint) * static_cast<std::size_t>(descriptors.rows),
                                     cudaMemcpyHostToDevice),
                          "descriptor upload failed");
#endif
            }

            ~SiftDataOwner() noexcept
            {
                FreeSiftData(_data);
            }

            SiftDataOwner(const SiftDataOwner&) = delete;
            SiftDataOwner& operator=(const SiftDataOwner&) = delete;

            SiftData& get()
            {
                return _data;
            }

            const SiftPoint* hostPoints() const
            {
#ifdef MANAGEDMEM
                return _data.m_data;
#else
                return _data.h_data;
#endif
            }

        private:
            SiftData _data{};
        };

        class SiftTemporaryMemory
        {
        public:
            SiftTemporaryMemory(int width, int height, int octaves)
                : _data(AllocSiftTempMemory(width, height, octaves, false))
            {
                if (!_data)
                {
                    throw std::runtime_error("SIFT CUDA temporary memory allocation failed");
                }
            }

            ~SiftTemporaryMemory() noexcept
            {
                FreeSiftTempMemory(_data);
            }

            float* get() const
            {
                return _data;
            }

        private:
            float* _data = nullptr;
        };

        std::vector<SiftNearestMatch> nearestMatches(const SiftDataOwner& owner, int count)
        {
            const SiftPoint* points = owner.hostPoints();
            std::vector<SiftNearestMatch> matches(static_cast<std::size_t>(count));
            for (int index = 0; index < count; ++index)
            {
                matches[static_cast<std::size_t>(index)] = {
                    points[index].match, points[index].score, points[index].ambiguity};
            }
            return matches;
        }

        bool isUsableExtractedPoint(const SiftPoint& point)
        {
            if (!std::isfinite(point.xpos) ||
                !std::isfinite(point.ypos) ||
                !std::isfinite(point.scale) ||
                point.scale <= 0.0f)
            {
                return false;
            }

            double squared_norm = 0.0;
            for (float value : point.data)
            {
                if (!std::isfinite(value) || value < 0.0f)
                {
                    return false;
                }
                squared_norm += static_cast<double>(value) * value;
            }
            return std::isfinite(squared_norm) && squared_norm > 1.0e-12;
        }

    } // namespace

    bool isCudaSiftBackendAvailable(int deviceIndex)
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || deviceIndex < 0 || deviceIndex >= count)
        {
            return false;
        }
        int previousDevice = 0;
        const bool restoreDevice = cudaGetDevice(&previousDevice) == cudaSuccess;
        const cudaError_t selectStatus = cudaSetDevice(deviceIndex);
        const cudaError_t initializeStatus = selectStatus == cudaSuccess ? cudaFree(nullptr) : selectStatus;
        if (restoreDevice && previousDevice != deviceIndex)
        {
            cudaSetDevice(previousDevice);
        }
        return initializeStatus == cudaSuccess;
    }

    SiftRawFeatures extractCudaSift(const SiftExtractionRequest& request)
    {
        checkCuda(cudaSetDevice(std::max(0, request.deviceIndex)), "device selection failed");
        InitCuda(std::max(0, request.deviceIndex));

        cv::Mat floatImage;
        request.image.convertTo(floatImage, CV_32F);
        if (!floatImage.isContinuous())
        {
            floatImage = floatImage.clone();
        }

        CudaImage cudaImage;
        cudaImage.Allocate(request.image.cols,
                           request.image.rows,
                           iAlignUp(request.image.cols, 128),
                           false,
                           nullptr,
                           reinterpret_cast<float*>(floatImage.data));
        cudaImage.Download();

        const int maximumPoints = std::max(1024, request.maximumFeatures);
        constexpr int octaveCount = 5;
        SiftDataOwner siftData(maximumPoints, true);
        SiftTemporaryMemory temporary(request.image.cols, request.image.rows, octaveCount);
        const float threshold = request.contrastThreshold > 1.0f
                                    ? request.contrastThreshold
                                    : std::clamp(request.contrastThreshold * 1000.0f, 0.1f, 20.0f);
        ExtractSift(siftData.get(), cudaImage, octaveCount, 1.0, threshold, 0.0f, false, temporary.get());

        const int pointCount = std::clamp(siftData.get().numPts, 0, siftData.get().maxPts);
        const SiftPoint* points = siftData.hostPoints();
        std::vector<int> usable_indices;
        usable_indices.reserve(static_cast<std::size_t>(pointCount));
        for (int index = 0; index < pointCount; ++index)
        {
            if (isUsableExtractedPoint(points[index]))
            {
                usable_indices.push_back(index);
            }
        }

        SiftRawFeatures result;
        result.keypoints.resize(usable_indices.size());
        result.descriptors.create(static_cast<int>(usable_indices.size()), 128, CV_32F);
        for (int output_index = 0;
             output_index < static_cast<int>(usable_indices.size());
             ++output_index)
        {
            const SiftPoint& point = points[usable_indices[static_cast<std::size_t>(output_index)]];
            cv::KeyPoint& keypoint = result.keypoints[static_cast<std::size_t>(output_index)];
            keypoint.pt.x = point.xpos;
            keypoint.pt.y = point.ypos;
            keypoint.size = std::max(1.0f, point.scale);
            keypoint.angle = std::isfinite(point.orientation) ? point.orientation : -1.0f;
            keypoint.response = std::isfinite(point.sharpness) ? std::abs(point.sharpness) : 0.0f;
            std::copy(point.data,
                      point.data + 128,
                      result.descriptors.ptr<float>(output_index));
        }
        return result;
    }

    std::vector<SiftNearestMatch>
    matchCudaSift(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, int deviceIndex)
    {
        checkCuda(cudaSetDevice(std::max(0, deviceIndex)), "device selection failed");
        SiftDataOwner query(queryDescriptors);
        SiftDataOwner train(trainDescriptors);
        MatchSiftData(query.get(), train.get());
        return nearestMatches(query, queryDescriptors.rows);
    }

} // namespace xjw::image_matching
