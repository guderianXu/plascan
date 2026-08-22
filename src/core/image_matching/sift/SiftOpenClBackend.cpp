#define CL_TARGET_OPENCL_VERSION 120

#include "SiftComputeBackend.h"

#include "SiftOpenClKernels.h"

#include <CL/cl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace xjw::image_matching
{
    namespace
    {

        struct OpenClCandidate
        {
            float x;
            float y;
            float size;
            float angle;
            float response;
            std::uint32_t octave;
            std::uint32_t layer;
            std::uint32_t padding;
        };

        struct OpenClNearestMatch
        {
            std::int32_t index;
            float similarity;
            float ambiguity;
            float padding;
        };

        static_assert(sizeof(OpenClCandidate) == 32);
        static_assert(sizeof(OpenClNearestMatch) == 16);

        void checkOpenCl(cl_int status, const char* operation)
        {
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error(std::string("SIFT OpenCL backend ") + operation + " failed with error " +
                                         std::to_string(status));
            }
        }

        std::vector<cl_device_id> openClGpuDevices()
        {
            cl_uint platformCount = 0;
            if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0)
            {
                return {};
            }
            std::vector<cl_platform_id> platforms(platformCount);
            checkOpenCl(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "platform enumeration");
            std::vector<cl_device_id> devices;
            for (cl_platform_id platform : platforms)
            {
                cl_uint count = 0;
                const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
                if (status == CL_DEVICE_NOT_FOUND || count == 0)
                {
                    continue;
                }
                checkOpenCl(status, "GPU enumeration");
                const std::size_t first = devices.size();
                devices.resize(first + count);
                checkOpenCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data() + first, nullptr),
                            "GPU enumeration");
            }
            return devices;
        }

        class OpenClBuffer
        {
        public:
            OpenClBuffer() = default;

            OpenClBuffer(cl_context context,
                         std::size_t bytes,
                         cl_mem_flags flags = CL_MEM_READ_WRITE,
                         void* data = nullptr)
            {
                cl_int status = CL_SUCCESS;
                _memory = clCreateBuffer(context, flags, std::max<std::size_t>(bytes, 4U), data, &status);
                checkOpenCl(status, "buffer allocation");
            }

            ~OpenClBuffer()
            {
                if (_memory)
                {
                    clReleaseMemObject(_memory);
                }
            }

            OpenClBuffer(OpenClBuffer&& other) noexcept : _memory(other._memory)
            {
                other._memory = nullptr;
            }

            OpenClBuffer& operator=(OpenClBuffer&& other) noexcept
            {
                std::swap(_memory, other._memory);
                return *this;
            }

            OpenClBuffer(const OpenClBuffer&) = delete;
            OpenClBuffer& operator=(const OpenClBuffer&) = delete;

            cl_mem get() const
            {
                return _memory;
            }

        private:
            cl_mem _memory = nullptr;
        };

        class OpenClRuntime
        {
        public:
            explicit OpenClRuntime(int deviceIndex)
            {
                const std::vector<cl_device_id> devices = openClGpuDevices();
                if (deviceIndex < 0 || deviceIndex >= static_cast<int>(devices.size()))
                {
                    throw std::runtime_error("SIFT OpenCL device index is unavailable");
                }
                _device = devices[static_cast<std::size_t>(deviceIndex)];
                cl_int status = CL_SUCCESS;
                _context = clCreateContext(nullptr, 1, &_device, nullptr, nullptr, &status);
                checkOpenCl(status, "context creation");
                _queue = clCreateCommandQueue(_context, _device, 0, &status);
                checkOpenCl(status, "command queue creation");
                const char* source = kSiftOpenClSource;
                const std::size_t length = std::strlen(source);
                _program = clCreateProgramWithSource(_context, 1, &source, &length, &status);
                checkOpenCl(status, "program creation");
                status = clBuildProgram(_program, 1, &_device, "-cl-fast-relaxed-math", nullptr, nullptr);
                if (status != CL_SUCCESS)
                {
                    std::size_t logSize = 0;
                    clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                    std::string log(logSize, '\0');
                    clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
                    throw std::runtime_error("SIFT OpenCL kernel compilation failed: " + log);
                }
            }

            ~OpenClRuntime()
            {
                for (auto& [name, kernel] : _kernels)
                {
                    (void)name;
                    clReleaseKernel(kernel);
                }
                if (_program)
                {
                    clReleaseProgram(_program);
                }
                if (_queue)
                {
                    clReleaseCommandQueue(_queue);
                }
                if (_context)
                {
                    clReleaseContext(_context);
                }
            }

            cl_context context() const
            {
                return _context;
            }

            cl_command_queue queue() const
            {
                return _queue;
            }

            cl_kernel kernel(const char* name)
            {
                const auto existing = _kernels.find(name);
                if (existing != _kernels.end())
                {
                    return existing->second;
                }
                cl_int status = CL_SUCCESS;
                cl_kernel result = clCreateKernel(_program, name, &status);
                checkOpenCl(status, name);
                _kernels.emplace(name, result);
                return result;
            }

        private:
            cl_device_id _device = nullptr;
            cl_context _context = nullptr;
            cl_command_queue _queue = nullptr;
            cl_program _program = nullptr;
            std::unordered_map<std::string, cl_kernel> _kernels;
        };

        OpenClRuntime& openClRuntime(int deviceIndex)
        {
            static std::mutex mutex;
            static std::unordered_map<int, std::unique_ptr<OpenClRuntime>> runtimes;
            std::scoped_lock lock(mutex);
            auto& runtime = runtimes[deviceIndex];
            if (!runtime)
            {
                runtime = std::make_unique<OpenClRuntime>(deviceIndex);
            }
            return *runtime;
        }

        template <typename Value> void setKernelValue(cl_kernel kernel, cl_uint index, const Value& value)
        {
            checkOpenCl(clSetKernelArg(kernel, index, sizeof(Value), &value), "kernel argument binding");
        }

        void setKernelBuffer(cl_kernel kernel, cl_uint index, const OpenClBuffer& buffer)
        {
            const cl_mem memory = buffer.get();
            checkOpenCl(clSetKernelArg(kernel, index, sizeof(memory), &memory), "kernel buffer binding");
        }

        std::size_t rounded(std::size_t value, std::size_t block)
        {
            return ((value + block - 1U) / block) * block;
        }

        void enqueue1d(OpenClRuntime& runtime, cl_kernel kernel, std::size_t count)
        {
            const std::size_t local = 64;
            const std::size_t global = rounded(count, local);
            checkOpenCl(
                clEnqueueNDRangeKernel(runtime.queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "one-dimensional kernel launch");
        }

        void enqueue2d(OpenClRuntime& runtime, cl_kernel kernel, std::size_t width, std::size_t height)
        {
            const std::size_t local[2] = {16, 16};
            const std::size_t global[2] = {rounded(width, local[0]), rounded(height, local[1])};
            checkOpenCl(clEnqueueNDRangeKernel(runtime.queue(), kernel, 2, nullptr, global, local, 0, nullptr, nullptr),
                        "two-dimensional kernel launch");
        }

        OpenClBuffer convertImage(OpenClRuntime& runtime, const cv::Mat& image)
        {
            const cv::Mat contiguous = image.isContinuous() ? image : image.clone();
            const std::size_t count = contiguous.total();
            OpenClBuffer input(runtime.context(), count, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, contiguous.data);
            OpenClBuffer output(runtime.context(), count * sizeof(float));
            cl_kernel kernel = runtime.kernel("convert_u8");
            const cl_uint pixelCount = static_cast<cl_uint>(count);
            setKernelBuffer(kernel, 0, input);
            setKernelBuffer(kernel, 1, output);
            setKernelValue(kernel, 2, pixelCount);
            enqueue1d(runtime, kernel, count);
            return output;
        }

        OpenClBuffer
        gaussianBlur(OpenClRuntime& runtime, const OpenClBuffer& input, cl_uint width, cl_uint height, float sigma)
        {
            const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);
            OpenClBuffer temporary(runtime.context(), bytes);
            OpenClBuffer output(runtime.context(), bytes);
            const int radius = std::clamp(static_cast<int>(std::ceil(3.0f * sigma)), 1, 12);
            cl_kernel horizontal = runtime.kernel("gaussian_horizontal");
            setKernelBuffer(horizontal, 0, input);
            setKernelBuffer(horizontal, 1, temporary);
            setKernelValue(horizontal, 2, width);
            setKernelValue(horizontal, 3, height);
            setKernelValue(horizontal, 4, sigma);
            setKernelValue(horizontal, 5, radius);
            enqueue2d(runtime, horizontal, width, height);
            cl_kernel vertical = runtime.kernel("gaussian_vertical");
            setKernelBuffer(vertical, 0, temporary);
            setKernelBuffer(vertical, 1, output);
            setKernelValue(vertical, 2, width);
            setKernelValue(vertical, 3, height);
            setKernelValue(vertical, 4, sigma);
            setKernelValue(vertical, 5, radius);
            enqueue2d(runtime, vertical, width, height);
            return output;
        }

        OpenClBuffer
        difference(OpenClRuntime& runtime, const OpenClBuffer& low, const OpenClBuffer& high, cl_uint count)
        {
            OpenClBuffer output(runtime.context(), static_cast<std::size_t>(count) * sizeof(float));
            cl_kernel kernel = runtime.kernel("difference");
            setKernelBuffer(kernel, 0, low);
            setKernelBuffer(kernel, 1, high);
            setKernelBuffer(kernel, 2, output);
            setKernelValue(kernel, 3, count);
            enqueue1d(runtime, kernel, count);
            return output;
        }

        OpenClBuffer downsample(OpenClRuntime& runtime,
                                const OpenClBuffer& input,
                                cl_uint inputWidth,
                                cl_uint outputWidth,
                                cl_uint outputHeight)
        {
            OpenClBuffer output(runtime.context(),
                                static_cast<std::size_t>(outputWidth) * outputHeight * sizeof(float));
            cl_kernel kernel = runtime.kernel("downsample_half");
            setKernelBuffer(kernel, 0, input);
            setKernelBuffer(kernel, 1, output);
            setKernelValue(kernel, 2, inputWidth);
            setKernelValue(kernel, 3, outputWidth);
            setKernelValue(kernel, 4, outputHeight);
            enqueue2d(runtime, kernel, outputWidth, outputHeight);
            return output;
        }

        void appendDescriptors(SiftRawFeatures* result,
                               const std::vector<OpenClCandidate>& candidates,
                               const std::vector<float>& descriptors,
                               float octaveScale)
        {
            const int previousRows = result->descriptors.rows;
            cv::Mat expanded(previousRows + static_cast<int>(candidates.size()), 128, CV_32F);
            if (previousRows > 0)
            {
                result->descriptors.copyTo(expanded.rowRange(0, previousRows));
            }
            for (int index = 0; index < static_cast<int>(candidates.size()); ++index)
            {
                const OpenClCandidate& candidate = candidates[static_cast<std::size_t>(index)];
                cv::KeyPoint keypoint;
                keypoint.pt = cv::Point2f(candidate.x * octaveScale, candidate.y * octaveScale);
                keypoint.size = candidate.size * octaveScale;
                keypoint.angle = candidate.angle;
                keypoint.response = candidate.response;
                keypoint.octave = static_cast<int>(candidate.octave);
                result->keypoints.push_back(keypoint);
                std::copy(descriptors.data() + static_cast<std::size_t>(index) * 128U,
                          descriptors.data() + static_cast<std::size_t>(index + 1) * 128U,
                          expanded.ptr<float>(previousRows + index));
            }
            result->descriptors = std::move(expanded);
        }

        void appendLayerFeatures(OpenClRuntime& runtime,
                                 const OpenClBuffer& previousDog,
                                 const OpenClBuffer& currentDog,
                                 const OpenClBuffer& nextDog,
                                 const OpenClBuffer& gaussian,
                                 cl_uint width,
                                 cl_uint height,
                                 cl_uint octave,
                                 cl_uint layer,
                                 float threshold,
                                 cl_uint capacity,
                                 SiftRawFeatures* result)
        {
            OpenClBuffer candidates(runtime.context(), static_cast<std::size_t>(capacity) * sizeof(OpenClCandidate));
            OpenClBuffer descriptors(runtime.context(), static_cast<std::size_t>(capacity) * 128U * sizeof(float));
            const cl_uint zero = 0;
            OpenClBuffer count(
                runtime.context(), sizeof(zero), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, const_cast<cl_uint*>(&zero));
            cl_kernel detect = runtime.kernel("detect_extrema");
            setKernelBuffer(detect, 0, previousDog);
            setKernelBuffer(detect, 1, currentDog);
            setKernelBuffer(detect, 2, nextDog);
            setKernelBuffer(detect, 3, candidates);
            setKernelBuffer(detect, 4, count);
            setKernelValue(detect, 5, width);
            setKernelValue(detect, 6, height);
            setKernelValue(detect, 7, octave);
            setKernelValue(detect, 8, layer);
            setKernelValue(detect, 9, threshold);
            setKernelValue(detect, 10, capacity);
            enqueue2d(runtime, detect, width, height);
            cl_kernel orientation = runtime.kernel("assign_orientation");
            setKernelBuffer(orientation, 0, gaussian);
            setKernelBuffer(orientation, 1, candidates);
            setKernelBuffer(orientation, 2, count);
            setKernelValue(orientation, 3, width);
            setKernelValue(orientation, 4, height);
            setKernelValue(orientation, 5, capacity);
            enqueue1d(runtime, orientation, capacity);
            cl_kernel descriptor = runtime.kernel("make_descriptor");
            setKernelBuffer(descriptor, 0, gaussian);
            setKernelBuffer(descriptor, 1, candidates);
            setKernelBuffer(descriptor, 2, count);
            setKernelBuffer(descriptor, 3, descriptors);
            setKernelValue(descriptor, 4, width);
            setKernelValue(descriptor, 5, height);
            setKernelValue(descriptor, 6, capacity);
            enqueue1d(runtime, descriptor, capacity);

            cl_uint countValue = 0;
            checkOpenCl(
                clEnqueueReadBuffer(
                    runtime.queue(), count.get(), CL_TRUE, 0, sizeof(countValue), &countValue, 0, nullptr, nullptr),
                "feature count download");
            countValue = std::min(countValue, capacity);
            if (countValue == 0)
            {
                return;
            }
            std::vector<OpenClCandidate> hostCandidates(countValue);
            std::vector<float> hostDescriptors(static_cast<std::size_t>(countValue) * 128U);
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            candidates.get(),
                                            CL_TRUE,
                                            0,
                                            hostCandidates.size() * sizeof(OpenClCandidate),
                                            hostCandidates.data(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "keypoint download");
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            descriptors.get(),
                                            CL_TRUE,
                                            0,
                                            hostDescriptors.size() * sizeof(float),
                                            hostDescriptors.data(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "descriptor download");
            appendDescriptors(result, hostCandidates, hostDescriptors, std::ldexp(1.0f, static_cast<int>(octave)));
        }

    } // namespace

    bool isOpenClSiftBackendAvailable(int deviceIndex)
    {
        try
        {
            return deviceIndex >= 0 && deviceIndex < static_cast<int>(openClGpuDevices().size()) &&
                   openClRuntime(deviceIndex).context() != nullptr;
        }
        catch (...)
        {
            return false;
        }
    }

    QString openClSiftDeviceName(int deviceIndex)
    {
        try
        {
            const std::vector<cl_device_id> devices = openClGpuDevices();
            if (deviceIndex < 0 || deviceIndex >= static_cast<int>(devices.size()))
            {
                return QString();
            }
            std::size_t size = 0;
            if (clGetDeviceInfo(devices[static_cast<std::size_t>(deviceIndex)],
                                CL_DEVICE_NAME,
                                0,
                                nullptr,
                                &size) != CL_SUCCESS || size == 0)
            {
                return QString();
            }
            std::string name(size, '\0');
            if (clGetDeviceInfo(devices[static_cast<std::size_t>(deviceIndex)],
                                CL_DEVICE_NAME,
                                name.size(),
                                name.data(),
                                nullptr) != CL_SUCCESS)
            {
                return QString();
            }
            if (!name.empty() && name.back() == '\0')
            {
                name.pop_back();
            }
            return QString::fromStdString(name).trimmed();
        }
        catch (...)
        {
            return QString();
        }
    }

    SiftRawFeatures extractOpenClSift(const SiftExtractionRequest& request)
    {
        if (request.image.empty() || request.image.type() != CV_8U)
        {
            throw std::invalid_argument("SIFT OpenCL backend requires a CV_8U grayscale image");
        }
        OpenClRuntime& runtime = openClRuntime(request.deviceIndex);
        constexpr int scalesPerOctave = 3;
        constexpr int gaussianLevelCount = scalesPerOctave + 3;
        constexpr int dogLevelCount = gaussianLevelCount - 1;
        const cl_uint capacity = static_cast<cl_uint>(
            std::clamp(request.maximumFeatures > 0 ? request.maximumFeatures : 32768, 1024, 100000));
        const float threshold = std::clamp(request.contrastThreshold / scalesPerOctave, 0.0001f, 0.1f);
        cl_uint width = static_cast<cl_uint>(request.image.cols);
        cl_uint height = static_cast<cl_uint>(request.image.rows);
        OpenClBuffer octaveBase = convertImage(runtime, request.image);
        SiftRawFeatures result;
        for (cl_uint octave = 0; octave < 6U && std::min(width, height) >= 32U && result.keypoints.size() < capacity;
             ++octave)
        {
            std::vector<OpenClBuffer> gaussianLevels;
            gaussianLevels.reserve(gaussianLevelCount);
            if (octave == 0U)
            {
                constexpr float sourceSigma = 0.5f;
                constexpr float baseSigma = 1.6f;
                const float incremental = std::sqrt(baseSigma * baseSigma - sourceSigma * sourceSigma);
                gaussianLevels.push_back(gaussianBlur(runtime, octaveBase, width, height, incremental));
            }
            else
            {
                // 第 3 个尺度下采样后已经是下一 octave 的 sigma=1.6 基准层。
                gaussianLevels.push_back(std::move(octaveBase));
            }
            float previousSigma = 1.6f;
            for (int level = 1; level < gaussianLevelCount; ++level)
            {
                const float sigma = 1.6f * std::pow(2.0f, static_cast<float>(level) / scalesPerOctave);
                const float incremental = std::sqrt(std::max(0.01f, sigma * sigma - previousSigma * previousSigma));
                gaussianLevels.push_back(gaussianBlur(runtime, gaussianLevels.back(), width, height, incremental));
                previousSigma = sigma;
            }
            const cl_uint pixelCount = width * height;
            std::vector<OpenClBuffer> dogLevels;
            dogLevels.reserve(dogLevelCount);
            for (int level = 0; level < dogLevelCount; ++level)
            {
                dogLevels.push_back(difference(runtime, gaussianLevels[level], gaussianLevels[level + 1], pixelCount));
            }
            for (cl_uint layer = 1; layer + 1 < dogLevels.size(); ++layer)
            {
                const cl_uint remaining = capacity - static_cast<cl_uint>(result.keypoints.size());
                if (remaining == 0U)
                {
                    break;
                }
                appendLayerFeatures(runtime,
                                    dogLevels[layer - 1],
                                    dogLevels[layer],
                                    dogLevels[layer + 1],
                                    gaussianLevels[layer],
                                    width,
                                    height,
                                    octave,
                                    layer,
                                    threshold,
                                    remaining,
                                    &result);
            }
            const cl_uint nextWidth = width / 2U;
            const cl_uint nextHeight = height / 2U;
            if (std::min(nextWidth, nextHeight) < 16U)
            {
                break;
            }
            octaveBase = downsample(runtime, gaussianLevels[scalesPerOctave], width, nextWidth, nextHeight);
            width = nextWidth;
            height = nextHeight;
        }
        checkOpenCl(clFinish(runtime.queue()), "extraction completion");
        return result;
    }

    std::vector<SiftNearestMatch>
    matchOpenClSift(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, int deviceIndex)
    {
        if (queryDescriptors.type() != CV_32F || trainDescriptors.type() != CV_32F || queryDescriptors.cols != 128 ||
            trainDescriptors.cols != 128)
        {
            throw std::invalid_argument("SIFT OpenCL matcher requires CV_32F descriptors with 128 columns");
        }
        if (queryDescriptors.empty() || trainDescriptors.empty())
        {
            return std::vector<SiftNearestMatch>(static_cast<std::size_t>(queryDescriptors.rows));
        }
        OpenClRuntime& runtime = openClRuntime(deviceIndex);
        const cv::Mat query = queryDescriptors.isContinuous() ? queryDescriptors : queryDescriptors.clone();
        const cv::Mat train = trainDescriptors.isContinuous() ? trainDescriptors : trainDescriptors.clone();
        OpenClBuffer queryBuffer(
            runtime.context(), query.total() * query.elemSize(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, query.data);
        OpenClBuffer trainBuffer(
            runtime.context(), train.total() * train.elemSize(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, train.data);
        OpenClBuffer outputBuffer(runtime.context(), static_cast<std::size_t>(query.rows) * sizeof(OpenClNearestMatch));
        const cl_uint queryCount = static_cast<cl_uint>(query.rows);
        const cl_uint trainCount = static_cast<cl_uint>(train.rows);
        cl_kernel kernel = runtime.kernel("nearest_match");
        setKernelBuffer(kernel, 0, queryBuffer);
        setKernelBuffer(kernel, 1, trainBuffer);
        setKernelBuffer(kernel, 2, outputBuffer);
        setKernelValue(kernel, 3, queryCount);
        setKernelValue(kernel, 4, trainCount);
        enqueue1d(runtime, kernel, queryCount);
        std::vector<OpenClNearestMatch> hostOutput(static_cast<std::size_t>(query.rows));
        checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                        outputBuffer.get(),
                                        CL_TRUE,
                                        0,
                                        hostOutput.size() * sizeof(OpenClNearestMatch),
                                        hostOutput.data(),
                                        0,
                                        nullptr,
                                        nullptr),
                    "matching result download");
        std::vector<SiftNearestMatch> result(hostOutput.size());
        for (std::size_t index = 0; index < hostOutput.size(); ++index)
        {
            result[index] = {hostOutput[index].index, hostOutput[index].similarity, hostOutput[index].ambiguity};
        }
        return result;
    }

} // namespace xjw::image_matching
