#include "SiftComputeBackend.h"

#include "cudaImage.h"
#include "cudaSift.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace xjw::image_matching
{
    namespace
    {

        static_assert(sizeof(SiftPoint) == 576, "Update the CUDA SIFT worker memory model when SiftPoint ABI changes");

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
            SiftDataOwner(int maximumPoints, bool populateHost)
            {
                InitSiftData(_data, std::max(1, maximumPoints), populateHost, true);
#ifndef MANAGEDMEM
                if (populateHost && _data.h_data)
                {
                    const std::size_t bytes = sizeof(SiftPoint) * static_cast<std::size_t>(_data.maxPts);
                    const cudaError_t status = cudaHostRegister(_data.h_data, bytes, cudaHostRegisterDefault);
                    if (status != cudaSuccess)
                    {
                        FreeSiftData(_data);
                        checkCuda(status, "pinned descriptor workspace registration failed");
                    }
                    _hostRegistered = true;
                }
#endif
            }

            void upload(const cv::Mat& descriptors, cudaStream_t stream)
            {
                if (descriptors.rows > _data.maxPts)
                {
                    throw std::invalid_argument("SIFT CUDA descriptor workspace is too small");
                }
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
                checkCuda(cudaMemcpyAsync(_data.d_data,
                                          _data.h_data,
                                          sizeof(SiftPoint) * static_cast<std::size_t>(descriptors.rows),
                                          cudaMemcpyHostToDevice,
                                          stream),
                          "asynchronous descriptor upload failed");
#endif
            }

            ~SiftDataOwner() noexcept
            {
#ifndef MANAGEDMEM
                if (_hostRegistered && _data.h_data)
                {
                    cudaHostUnregister(_data.h_data);
                }
#endif
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

            int capacity() const
            {
                return _data.maxPts;
            }

        private:
            SiftData _data{};
#ifndef MANAGEDMEM
            bool _hostRegistered = false;
#endif
        };

        std::vector<SiftNearestMatch> nearestMatches(const SiftDataOwner& owner, int count);

        class CudaSiftMatchWorkspace
        {
        public:
            explicit CudaSiftMatchWorkspace(int deviceIndex) : _deviceIndex(deviceIndex)
            {
                checkCuda(cudaSetDevice(_deviceIndex), "match workspace device selection failed");
                checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking), "match stream creation failed");
            }

            ~CudaSiftMatchWorkspace() noexcept
            {
                cudaSetDevice(_deviceIndex);
                _data0.reset();
                _data1.reset();
                if (_stream)
                {
                    cudaStreamDestroy(_stream);
                }
            }

            CudaSiftMatchWorkspace(const CudaSiftMatchWorkspace&) = delete;
            CudaSiftMatchWorkspace& operator=(const CudaSiftMatchWorkspace&) = delete;

            SiftBidirectionalMatches match(const cv::Mat& descriptors0, const cv::Mat& descriptors1)
            {
                checkCuda(cudaSetDevice(_deviceIndex), "match workspace activation failed");
                ensureCapacity(&_data0, descriptors0.rows);
                ensureCapacity(&_data1, descriptors1.rows);
                _data0->upload(descriptors0, _stream);
                _data1->upload(descriptors1, _stream);

                MatchSiftData(_data0->get(), _data1->get(), _stream);
                SiftBidirectionalMatches result;
                result.forward = nearestMatches(*_data0, descriptors0.rows);
                MatchSiftData(_data1->get(), _data0->get(), _stream);
                result.reverse = nearestMatches(*_data1, descriptors1.rows);
                return result;
            }

        private:
            static void ensureCapacity(std::unique_ptr<SiftDataOwner>* owner, int required)
            {
                if (!*owner || (*owner)->capacity() < required)
                {
                    *owner = std::make_unique<SiftDataOwner>(std::max(1, required), true);
                }
            }

            int _deviceIndex = 0;
            cudaStream_t _stream = nullptr;
            std::unique_ptr<SiftDataOwner> _data0;
            std::unique_ptr<SiftDataOwner> _data1;
        };

        class CudaSiftWorkspace
        {
        public:
            explicit CudaSiftWorkspace(int deviceIndex) : _deviceIndex(deviceIndex)
            {
                checkCuda(cudaSetDevice(_deviceIndex), "workspace device selection failed");
                InitCuda(_deviceIndex);
            }

            ~CudaSiftWorkspace() noexcept
            {
                cudaSetDevice(_deviceIndex);
                if (_siftDataInitialized)
                {
                    FreeSiftData(_siftData);
                }
                FreeSiftTempMemory(_temporaryMemory);
                if (_deviceImage)
                {
                    cudaFree(_deviceImage);
                }
                if (_hostImage)
                {
                    cudaFreeHost(_hostImage);
                }
            }

            CudaSiftWorkspace(const CudaSiftWorkspace&) = delete;
            CudaSiftWorkspace& operator=(const CudaSiftWorkspace&) = delete;

            void prepare(int width, int height, int maximumPoints)
            {
                checkCuda(cudaSetDevice(_deviceIndex), "workspace activation failed");
                if (width > _capacityWidth || height > _capacityHeight || !_deviceImage || !_hostImage ||
                    !_temporaryMemory)
                {
                    resizeImageStorage(std::max(width, _capacityWidth), std::max(height, _capacityHeight));
                }
                if (!_siftDataInitialized || maximumPoints > _siftData.maxPts)
                {
                    if (_siftDataInitialized)
                    {
                        FreeSiftData(_siftData);
                        _siftData = {};
                    }
                    InitSiftData(_siftData, std::max(1, maximumPoints), true, true);
                    _siftDataInitialized = true;
                }
            }

            cv::Mat hostImage(int width, int height) const
            {
                return cv::Mat(height, width, CV_32F, _hostImage, static_cast<std::size_t>(_pitch) * sizeof(float));
            }

            void upload(int width, int height)
            {
                checkCuda(cudaMemcpy2DAsync(_deviceImage,
                                            static_cast<std::size_t>(_pitch) * sizeof(float),
                                            _hostImage,
                                            static_cast<std::size_t>(_pitch) * sizeof(float),
                                            static_cast<std::size_t>(width) * sizeof(float),
                                            static_cast<std::size_t>(height),
                                            cudaMemcpyHostToDevice,
                                            nullptr),
                          "asynchronous image upload failed");
            }

            CudaImage cudaImage(int width, int height) const
            {
                CudaImage image;
                image.Allocate(width, height, _pitch, false, _deviceImage, _hostImage);
                return image;
            }

            SiftData& siftData()
            {
                return _siftData;
            }

            float* temporaryMemory() const
            {
                return _temporaryMemory;
            }

        private:
            void resizeImageStorage(int width, int height)
            {
                if (_temporaryMemory)
                {
                    FreeSiftTempMemory(_temporaryMemory);
                    _temporaryMemory = nullptr;
                }
                if (_deviceImage)
                {
                    checkCuda(cudaFree(_deviceImage), "old image storage release failed");
                    _deviceImage = nullptr;
                }
                if (_hostImage)
                {
                    checkCuda(cudaFreeHost(_hostImage), "old host image storage release failed");
                    _hostImage = nullptr;
                }

                _capacityWidth = width;
                _capacityHeight = height;
                _pitch = iAlignUp(_capacityWidth, 128);
                const std::size_t imageBytes =
                    static_cast<std::size_t>(_pitch) * static_cast<std::size_t>(_capacityHeight) * sizeof(float);
                checkCuda(cudaMalloc(reinterpret_cast<void**>(&_deviceImage), imageBytes),
                          "image workspace allocation failed");
                checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&_hostImage), imageBytes, cudaHostAllocDefault),
                          "pinned host workspace allocation failed");
                _temporaryMemory = AllocSiftTempMemory(_capacityWidth, _capacityHeight, 5, false);
                if (!_temporaryMemory)
                {
                    throw std::runtime_error("SIFT CUDA temporary memory allocation failed");
                }
            }

            int _deviceIndex = 0;
            int _capacityWidth = 0;
            int _capacityHeight = 0;
            int _pitch = 0;
            float* _deviceImage = nullptr;
            float* _hostImage = nullptr;
            float* _temporaryMemory = nullptr;
            SiftData _siftData{};
            bool _siftDataInitialized = false;
        };

        thread_local int extractionWorkspaceDevice = -1;
        thread_local std::unique_ptr<CudaSiftWorkspace> threadExtractionWorkspace;
        thread_local int matchWorkspaceDevice = -1;
        thread_local std::unique_ptr<CudaSiftMatchWorkspace> threadMatchWorkspace;

        CudaSiftWorkspace& extractionWorkspace(int deviceIndex)
        {
            if (!threadExtractionWorkspace || extractionWorkspaceDevice != deviceIndex)
            {
                threadExtractionWorkspace = std::make_unique<CudaSiftWorkspace>(deviceIndex);
                extractionWorkspaceDevice = deviceIndex;
            }
            return *threadExtractionWorkspace;
        }

        CudaSiftMatchWorkspace& matchWorkspace(int deviceIndex)
        {
            if (!threadMatchWorkspace || matchWorkspaceDevice != deviceIndex)
            {
                threadMatchWorkspace = std::make_unique<CudaSiftMatchWorkspace>(deviceIndex);
                matchWorkspaceDevice = deviceIndex;
            }
            return *threadMatchWorkspace;
        }

        std::mutex& extractionMutex(int deviceIndex)
        {
            constexpr int maximumTrackedDevices = 16;
            static std::array<std::mutex, maximumTrackedDevices> mutexes;
            return mutexes[static_cast<std::size_t>(std::clamp(deviceIndex, 0, maximumTrackedDevices - 1))];
        }

        std::vector<SiftNearestMatch> nearestMatches(const SiftDataOwner& owner, int count)
        {
            const SiftPoint* points = owner.hostPoints();
            std::vector<SiftNearestMatch> matches(static_cast<std::size_t>(count));
            for (int index = 0; index < count; ++index)
            {
                // CUDA Sift stores ambiguity as second-best cosine similarity divided by
                // the best similarity. The shared filter expects Lowe's Euclidean
                // distance ratio, as produced by the CPU, OpenCL and Metal backends.
                const float bestSimilarity = std::clamp(points[index].score, -1.0f, 1.0f);
                const float secondSimilarity =
                    std::clamp(points[index].ambiguity * points[index].score, -1.0f, bestSimilarity);
                const float bestDistance = std::sqrt(std::max(0.0f, 2.0f - 2.0f * bestSimilarity));
                const float secondDistance = std::sqrt(std::max(1.0e-12f, 2.0f - 2.0f * secondSimilarity));
                const float distanceRatio = std::clamp(bestDistance / secondDistance, 0.0f, 1.0f);
                matches[static_cast<std::size_t>(index)] = {points[index].match, bestSimilarity, distanceRatio};
            }
            return matches;
        }

        bool isUsableExtractedPoint(const SiftPoint& point)
        {
            if (!std::isfinite(point.xpos) || !std::isfinite(point.ypos) || !std::isfinite(point.scale) ||
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

    QString cudaSiftDeviceName(int deviceIndex)
    {
        cudaDeviceProp properties{};
        if (deviceIndex < 0 || cudaGetDeviceProperties(&properties, deviceIndex) != cudaSuccess)
        {
            return QString();
        }
        return QString::fromUtf8(properties.name).trimmed();
    }

    SiftRawFeatures extractCudaSift(const SiftExtractionRequest& request)
    {
        const int deviceIndex = std::max(0, request.deviceIndex);
        // 上游 CUDA SIFT 使用设备全局的计数器和常量。多影像任务可以并行执行
        // CPU 前后处理，但同一设备上的原始提取段必须串行，直到这些状态迁入
        // workspace，避免不同影像互相覆盖关键点计数。
        std::lock_guard extractionLock(extractionMutex(deviceIndex));
        const int maximumPoints = std::max(1024, request.maximumFeatures);
        constexpr int octaveCount = 5;
        CudaSiftWorkspace& workspace = extractionWorkspace(deviceIndex);
        workspace.prepare(request.image.cols, request.image.rows, maximumPoints);
        cv::Mat floatImage = workspace.hostImage(request.image.cols, request.image.rows);
        request.image.convertTo(floatImage, CV_32F);
        workspace.upload(request.image.cols, request.image.rows);
        CudaImage cudaImage = workspace.cudaImage(request.image.cols, request.image.rows);
        const float threshold = request.contrastThreshold > 1.0f
                                    ? request.contrastThreshold
                                    : std::clamp(request.contrastThreshold * 1000.0f, 0.1f, 20.0f);
        ExtractSift(
            workspace.siftData(), cudaImage, octaveCount, 1.0, threshold, 0.0f, false, workspace.temporaryMemory());

        const SiftData& siftData = workspace.siftData();
        const int pointCount = std::clamp(siftData.numPts, 0, siftData.maxPts);
#ifdef MANAGEDMEM
        const SiftPoint* points = siftData.m_data;
#else
        const SiftPoint* points = siftData.h_data;
#endif
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
        for (int output_index = 0; output_index < static_cast<int>(usable_indices.size()); ++output_index)
        {
            const SiftPoint& point = points[usable_indices[static_cast<std::size_t>(output_index)]];
            cv::KeyPoint& keypoint = result.keypoints[static_cast<std::size_t>(output_index)];
            keypoint.pt.x = point.xpos;
            keypoint.pt.y = point.ypos;
            keypoint.size = std::max(1.0f, point.scale);
            keypoint.angle = std::isfinite(point.orientation) ? point.orientation : -1.0f;
            keypoint.response = std::isfinite(point.sharpness) ? std::abs(point.sharpness) : 0.0f;
            std::copy(point.data, point.data + 128, result.descriptors.ptr<float>(output_index));
        }
        return result;
    }

    SiftBidirectionalMatches
    matchCudaSiftBidirectionally(const cv::Mat& descriptors0, const cv::Mat& descriptors1, int deviceIndex)
    {
        return matchWorkspace(std::max(0, deviceIndex)).match(descriptors0, descriptors1);
    }

    std::vector<SiftNearestMatch>
    matchCudaSift(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, int deviceIndex)
    {
        return matchCudaSiftBidirectionally(queryDescriptors, trainDescriptors, deviceIndex).forward;
    }

    void releaseCudaSiftThreadWorkspaces()
    {
        threadMatchWorkspace.reset();
        matchWorkspaceDevice = -1;
        threadExtractionWorkspace.reset();
        extractionWorkspaceDevice = -1;
    }

} // namespace xjw::image_matching
