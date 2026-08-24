#define CL_TARGET_OPENCL_VERSION 120

#ifdef DM_ENABLE_OPENCL

#include "CostFunctions.h"
#include "OpenClCostKernels.h"

#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xjw::dense_match
{

    namespace
    {

        void checkOpenCl(cl_int status, const char* operation)
        {
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error(std::string("Dense-match OpenCL ") + operation + " failed with error " +
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
                cl_uint deviceCount = 0;
                const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
                if (status == CL_DEVICE_NOT_FOUND || deviceCount == 0)
                {
                    continue;
                }
                checkOpenCl(status, "GPU enumeration");
                const std::size_t first = devices.size();
                devices.resize(first + deviceCount);
                checkOpenCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data() + first, nullptr),
                            "GPU enumeration");
            }
            return devices;
        }

        class OpenClBuffer
        {
        public:
            OpenClBuffer(cl_context context,
                         std::size_t byteCount,
                         cl_mem_flags flags = CL_MEM_READ_WRITE,
                         void* hostData = nullptr)
            {
                cl_int status = CL_SUCCESS;
                _memory = clCreateBuffer(context, flags, std::max<std::size_t>(byteCount, 4U), hostData, &status);
                checkOpenCl(status, "buffer allocation");
            }

            ~OpenClBuffer()
            {
                if (_memory != nullptr)
                {
                    clReleaseMemObject(_memory);
                }
            }

            OpenClBuffer(const OpenClBuffer&) = delete;
            OpenClBuffer& operator=(const OpenClBuffer&) = delete;

            [[nodiscard]] cl_mem get() const
            {
                return _memory;
            }

        private:
            cl_mem _memory = nullptr;
        };

        class OpenClKernel
        {
        public:
            OpenClKernel(cl_program program, const char* name)
            {
                cl_int status = CL_SUCCESS;
                _kernel = clCreateKernel(program, name, &status);
                checkOpenCl(status, name);
            }

            ~OpenClKernel()
            {
                if (_kernel != nullptr)
                {
                    clReleaseKernel(_kernel);
                }
            }

            OpenClKernel(const OpenClKernel&) = delete;
            OpenClKernel& operator=(const OpenClKernel&) = delete;

            [[nodiscard]] cl_kernel get() const
            {
                return _kernel;
            }

        private:
            cl_kernel _kernel = nullptr;
        };

        class OpenClRuntime
        {
        public:
            explicit OpenClRuntime(int deviceIndex)
            {
                try
                {
                    const std::vector<cl_device_id> devices = openClGpuDevices();
                    if (deviceIndex < 0 || deviceIndex >= static_cast<int>(devices.size()))
                    {
                        throw std::runtime_error("Dense-match OpenCL device index is unavailable: " +
                                                 std::to_string(deviceIndex));
                    }
                    _device = devices[static_cast<std::size_t>(deviceIndex)];

                    cl_int status = CL_SUCCESS;
                    _context = clCreateContext(nullptr, 1, &_device, nullptr, nullptr, &status);
                    checkOpenCl(status, "context creation");
                    _queue = clCreateCommandQueue(_context, _device, 0, &status);
                    checkOpenCl(status, "command queue creation");

                    const char* source = kDenseMatchOpenClSource;
                    const std::size_t sourceLength = std::strlen(source);
                    _program = clCreateProgramWithSource(_context, 1, &source, &sourceLength, &status);
                    checkOpenCl(status, "program creation");
                    status = clBuildProgram(_program, 1, &_device, "-cl-std=CL1.2", nullptr, nullptr);
                    if (status != CL_SUCCESS)
                    {
                        std::size_t logSize = 0;
                        clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                        std::string log(std::max<std::size_t>(logSize, 1U), '\0');
                        clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
                        throw std::runtime_error("Dense-match OpenCL program build failed: " + log);
                    }
                }
                catch (...)
                {
                    release();
                    throw;
                }
            }

            ~OpenClRuntime()
            {
                release();
            }

            OpenClRuntime(const OpenClRuntime&) = delete;
            OpenClRuntime& operator=(const OpenClRuntime&) = delete;

            [[nodiscard]] cl_context context() const
            {
                return _context;
            }

            [[nodiscard]] cl_command_queue queue() const
            {
                return _queue;
            }

            [[nodiscard]] cl_program program() const
            {
                return _program;
            }

        private:
            void release()
            {
                if (_program != nullptr)
                {
                    clReleaseProgram(_program);
                    _program = nullptr;
                }
                if (_queue != nullptr)
                {
                    clReleaseCommandQueue(_queue);
                    _queue = nullptr;
                }
                if (_context != nullptr)
                {
                    clReleaseContext(_context);
                    _context = nullptr;
                }
            }

            cl_device_id _device = nullptr;
            cl_context _context = nullptr;
            cl_command_queue _queue = nullptr;
            cl_program _program = nullptr;
        };

        OpenClRuntime& openClRuntime(int deviceIndex)
        {
            // Contexts, queues and compiled kernels are expensive to create.  A
            // thread-local runtime avoids rebuilding the OpenCL program for every
            // stereo pair while keeping command queues isolated between workers.
            thread_local std::unordered_map<int, std::unique_ptr<OpenClRuntime>> runtimes;
            std::unique_ptr<OpenClRuntime>& runtime = runtimes[deviceIndex];
            if (!runtime)
            {
                runtime = std::make_unique<OpenClRuntime>(deviceIndex);
            }
            return *runtime;
        }

        template <typename T> void setKernelArgument(cl_kernel kernel, cl_uint index, const T& value)
        {
            checkOpenCl(clSetKernelArg(kernel, index, sizeof(T), &value), "kernel argument setup");
        }

        void enqueueCostVolume(OpenClRuntime& runtime,
                               const OpenClBuffer& left,
                               const OpenClBuffer& right,
                               const OpenClBuffer& volume,
                               int imageWidth,
                               int imageHeight,
                               int minDisparity,
                               int numDisparities,
                               int kernelWidth,
                               int kernelHeight,
                               CostFunction function)
        {
            OpenClKernel kernel(runtime.program(), "compute_cost_volume");
            const cl_mem leftMemory = left.get();
            const cl_mem rightMemory = right.get();
            const cl_mem volumeMemory = volume.get();
            const int functionValue = static_cast<int>(function);
            setKernelArgument(kernel.get(), 0, leftMemory);
            setKernelArgument(kernel.get(), 1, rightMemory);
            setKernelArgument(kernel.get(), 2, volumeMemory);
            setKernelArgument(kernel.get(), 3, imageWidth);
            setKernelArgument(kernel.get(), 4, imageHeight);
            setKernelArgument(kernel.get(), 5, minDisparity);
            setKernelArgument(kernel.get(), 6, numDisparities);
            setKernelArgument(kernel.get(), 7, kernelWidth);
            setKernelArgument(kernel.get(), 8, kernelHeight);
            setKernelArgument(kernel.get(), 9, functionValue);
            const std::size_t globalSize[3] = {static_cast<std::size_t>(imageWidth),
                                               static_cast<std::size_t>(imageHeight),
                                               static_cast<std::size_t>(numDisparities)};
            checkOpenCl(clEnqueueNDRangeKernel(
                            runtime.queue(), kernel.get(), 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr),
                        "cost-volume kernel launch");
        }

        void enqueueSelection(OpenClRuntime& runtime,
                              const OpenClBuffer& volume,
                              const OpenClBuffer& disparity,
                              const OpenClBuffer& confidence,
                              const OpenClBuffer& validMask,
                              int imageWidth,
                              int imageHeight,
                              int minDisparity,
                              int numDisparities,
                              SubpixelMode subpixel)
        {
            OpenClKernel kernel(runtime.program(), "select_cost_volume");
            const cl_mem volumeMemory = volume.get();
            const cl_mem disparityMemory = disparity.get();
            const cl_mem confidenceMemory = confidence.get();
            const cl_mem validMaskMemory = validMask.get();
            const int subpixelValue = static_cast<int>(subpixel);
            setKernelArgument(kernel.get(), 0, volumeMemory);
            setKernelArgument(kernel.get(), 1, disparityMemory);
            setKernelArgument(kernel.get(), 2, confidenceMemory);
            setKernelArgument(kernel.get(), 3, validMaskMemory);
            setKernelArgument(kernel.get(), 4, imageWidth);
            setKernelArgument(kernel.get(), 5, imageHeight);
            setKernelArgument(kernel.get(), 6, minDisparity);
            setKernelArgument(kernel.get(), 7, numDisparities);
            setKernelArgument(kernel.get(), 8, subpixelValue);
            const std::size_t globalSize[2] = {static_cast<std::size_t>(imageWidth),
                                               static_cast<std::size_t>(imageHeight)};
            checkOpenCl(clEnqueueNDRangeKernel(
                            runtime.queue(), kernel.get(), 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr),
                        "selection kernel launch");
        }

        DisparityResult downloadSelection(OpenClRuntime& runtime,
                                          const OpenClBuffer& disparity,
                                          const OpenClBuffer& confidence,
                                          const OpenClBuffer& validMask,
                                          cv::Size imageSize,
                                          const CostVolumeBufferLayout& layout)
        {
            DisparityResult result;
            result.disparity = cv::Mat(imageSize, CV_32FC1, cv::Scalar(0));
            result.confidence = cv::Mat(imageSize, CV_32FC1, cv::Scalar(0));
            result.validMask = cv::Mat(imageSize, CV_8UC1, cv::Scalar(0));
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            disparity.get(),
                                            CL_TRUE,
                                            0,
                                            layout.planeBytes,
                                            result.disparity.ptr<float>(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "disparity download");
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            confidence.get(),
                                            CL_TRUE,
                                            0,
                                            layout.planeBytes,
                                            result.confidence.ptr<float>(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "confidence download");
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            validMask.get(),
                                            CL_TRUE,
                                            0,
                                            layout.imageBytes,
                                            result.validMask.ptr<uchar>(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "valid-mask download");
            return result;
        }

        void uploadCostVolume(OpenClRuntime& runtime,
                              const CostVolume& volume,
                              const OpenClBuffer& deviceVolume,
                              const CostVolumeBufferLayout& layout,
                              cv::Size imageSize)
        {
            for (int disparityIndex = 0; disparityIndex < layout.numDisparities; ++disparityIndex)
            {
                const cv::Mat& plane = volume[static_cast<std::size_t>(disparityIndex)];
                if (plane.type() != CV_32FC1 || plane.size() != imageSize || !plane.isContinuous())
                {
                    throw std::invalid_argument("OpenCL selection requires continuous CV_32FC1 cost planes");
                }
                checkOpenCl(clEnqueueWriteBuffer(runtime.queue(),
                                                 deviceVolume.get(),
                                                 CL_TRUE,
                                                 static_cast<std::size_t>(disparityIndex) * layout.planeBytes,
                                                 layout.planeBytes,
                                                 plane.ptr<float>(),
                                                 0,
                                                 nullptr,
                                                 nullptr),
                            "cost-volume upload");
            }
        }

    } // namespace

    bool isCostVolumeOpenCLAvailable(int openClDevice)
    {
        try
        {
            const std::vector<cl_device_id> devices = openClGpuDevices();
            return openClDevice >= 0 && openClDevice < static_cast<int>(devices.size());
        }
        catch (...)
        {
            return false;
        }
    }

    CostVolume computeCostVolumeOpenCL(const cv::Mat& left,
                                       const cv::Mat& right,
                                       int minDisp,
                                       int maxDisp,
                                       int kernelW,
                                       int kernelH,
                                       CostFunction func,
                                       int openClDevice)
    {
        CV_Assert(!left.empty() && !right.empty());
        CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
        CV_Assert(left.size() == right.size());
        CV_Assert(maxDisp > minDisp);
        CV_Assert(kernelW > 0 && kernelH > 0);

        const cv::Mat contiguousLeft = left.isContinuous() ? left : left.clone();
        const cv::Mat contiguousRight = right.isContinuous() ? right : right.clone();
        const CostVolumeBufferLayout layout = checkedCostVolumeBufferLayout(left.cols, left.rows, minDisp, maxDisp);
        OpenClRuntime& runtime = openClRuntime(openClDevice);
        OpenClBuffer deviceLeft(
            runtime.context(), layout.imageBytes, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, contiguousLeft.data);
        OpenClBuffer deviceRight(
            runtime.context(), layout.imageBytes, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, contiguousRight.data);
        OpenClBuffer deviceVolume(runtime.context(), layout.volumeBytes);
        enqueueCostVolume(runtime,
                          deviceLeft,
                          deviceRight,
                          deviceVolume,
                          left.cols,
                          left.rows,
                          minDisp,
                          layout.numDisparities,
                          kernelW,
                          kernelH,
                          func);

        CostVolume volume(minDisp, maxDisp, left.size());
        for (int disparityIndex = 0; disparityIndex < layout.numDisparities; ++disparityIndex)
        {
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            deviceVolume.get(),
                                            CL_TRUE,
                                            static_cast<std::size_t>(disparityIndex) * layout.planeBytes,
                                            layout.planeBytes,
                                            volume[static_cast<std::size_t>(disparityIndex)].ptr<float>(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "cost-volume download");
        }
        return volume;
    }

    DisparityResult selectCostVolumeOpenCL(const CostVolume& volume, SubpixelMode subpixel, int openClDevice)
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
            throw std::invalid_argument("OpenCL selection received an inconsistent cost volume");
        }

        OpenClRuntime& runtime = openClRuntime(openClDevice);
        OpenClBuffer deviceVolume(runtime.context(), layout.volumeBytes);
        OpenClBuffer deviceDisparity(runtime.context(), layout.planeBytes);
        OpenClBuffer deviceConfidence(runtime.context(), layout.planeBytes);
        OpenClBuffer deviceValidMask(runtime.context(), layout.imageBytes);
        uploadCostVolume(runtime, volume, deviceVolume, layout, imageSize);
        enqueueSelection(runtime,
                         deviceVolume,
                         deviceDisparity,
                         deviceConfidence,
                         deviceValidMask,
                         imageSize.width,
                         imageSize.height,
                         volume.minDisparity(),
                         layout.numDisparities,
                         subpixel);
        return downloadSelection(runtime, deviceDisparity, deviceConfidence, deviceValidMask, imageSize, layout);
    }

    DisparityResult computeBlockMatchOpenCL(const cv::Mat& left,
                                            const cv::Mat& right,
                                            int minDisp,
                                            int maxDisp,
                                            int kernelW,
                                            int kernelH,
                                            CostFunction func,
                                            SubpixelMode subpixel,
                                            int openClDevice)
    {
        CV_Assert(!left.empty() && !right.empty());
        CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
        CV_Assert(left.size() == right.size());
        CV_Assert(maxDisp > minDisp);
        CV_Assert(kernelW > 0 && kernelH > 0);

        const cv::Mat contiguousLeft = left.isContinuous() ? left : left.clone();
        const cv::Mat contiguousRight = right.isContinuous() ? right : right.clone();
        const CostVolumeBufferLayout layout = checkedCostVolumeBufferLayout(left.cols, left.rows, minDisp, maxDisp);
        OpenClRuntime& runtime = openClRuntime(openClDevice);
        OpenClBuffer deviceLeft(
            runtime.context(), layout.imageBytes, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, contiguousLeft.data);
        OpenClBuffer deviceRight(
            runtime.context(), layout.imageBytes, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, contiguousRight.data);
        OpenClBuffer deviceVolume(runtime.context(), layout.volumeBytes);
        OpenClBuffer deviceDisparity(runtime.context(), layout.planeBytes);
        OpenClBuffer deviceConfidence(runtime.context(), layout.planeBytes);
        OpenClBuffer deviceValidMask(runtime.context(), layout.imageBytes);
        enqueueCostVolume(runtime,
                          deviceLeft,
                          deviceRight,
                          deviceVolume,
                          left.cols,
                          left.rows,
                          minDisp,
                          layout.numDisparities,
                          kernelW,
                          kernelH,
                          func);
        enqueueSelection(runtime,
                         deviceVolume,
                         deviceDisparity,
                         deviceConfidence,
                         deviceValidMask,
                         left.cols,
                         left.rows,
                         minDisp,
                         layout.numDisparities,
                         subpixel);
        return downloadSelection(runtime, deviceDisparity, deviceConfidence, deviceValidMask, left.size(), layout);
    }

} // namespace xjw::dense_match

#endif // DM_ENABLE_OPENCL
