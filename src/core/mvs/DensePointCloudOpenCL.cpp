#define CL_TARGET_OPENCL_VERSION 120

#include "DensePointCloudOpenCL.h"

#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xjw::mvs
{
    namespace
    {

        constexpr const char* kUnprojectKernelSource = R"CLC(
__kernel void unproject_dense_cloud(
    __global const float *depth,
    __global const uchar *mask,
    int has_mask,
    __global const uchar *color,
    int channels,
    int width,
    int height,
    float minimum_depth,
    float maximum_depth,
    int subsample,
    int clip_aabb,
    float minimum_x,
    float maximum_x,
    float minimum_y,
    float maximum_y,
    float minimum_z,
    float maximum_z,
    float focal_x,
    float focal_y,
    float principal_x,
    float principal_y,
    int u_axis_sign,
    int v_axis_sign,
    int depth_sign,
    __global const float *camera_to_world,
    __global const float *camera_center,
    __global float *xyz,
    __global uchar *rgb,
    __global int *valid)
{
    const int column = (int)get_global_id(0);
    const int row = (int)get_global_id(1);
    if (column >= width || row >= height)
    {
        return;
    }
    const int index = row * width + column;
    valid[index] = 0;
    if ((row % subsample) != 0 || (column % subsample) != 0 ||
        (has_mask != 0 && mask[index] == 0))
    {
        return;
    }

    const float positive_depth = depth[index];
    if (!isfinite(positive_depth) || positive_depth < minimum_depth ||
        positive_depth > maximum_depth)
    {
        return;
    }
    const float normalized_x = (float)u_axis_sign *
        ((float)column - principal_x) / focal_x;
    const float normalized_y = (float)v_axis_sign *
        ((float)row - principal_y) / focal_y;
    const float camera_z = (float)depth_sign * positive_depth;
    const float camera_x = normalized_x * camera_z;
    const float camera_y = normalized_y * camera_z;
    const float world_x = camera_center[0] +
        camera_to_world[0] * camera_x +
        camera_to_world[1] * camera_y +
        camera_to_world[2] * camera_z;
    const float world_y = camera_center[1] +
        camera_to_world[3] * camera_x +
        camera_to_world[4] * camera_y +
        camera_to_world[5] * camera_z;
    const float world_z = camera_center[2] +
        camera_to_world[6] * camera_x +
        camera_to_world[7] * camera_y +
        camera_to_world[8] * camera_z;
    if (!isfinite(world_x) || !isfinite(world_y) || !isfinite(world_z) ||
        (clip_aabb != 0 &&
         (world_x < minimum_x || world_x > maximum_x ||
          world_y < minimum_y || world_y > maximum_y ||
          world_z < minimum_z || world_z > maximum_z)))
    {
        return;
    }

    xyz[index * 3] = world_x;
    xyz[index * 3 + 1] = world_y;
    xyz[index * 3 + 2] = world_z;
    if (channels == 3)
    {
        rgb[index * 3] = color[index * 3 + 2];
        rgb[index * 3 + 1] = color[index * 3 + 1];
        rgb[index * 3 + 2] = color[index * 3];
    }
    else if (channels == 1)
    {
        const uchar gray = color[index];
        rgb[index * 3] = gray;
        rgb[index * 3 + 1] = gray;
        rgb[index * 3 + 2] = gray;
    }
    else
    {
        rgb[index * 3] = (uchar)128;
        rgb[index * 3 + 1] = (uchar)128;
        rgb[index * 3 + 2] = (uchar)128;
    }
    valid[index] = 1;
}
)CLC";

        void checkOpenCl(cl_int status, const char* operation)
        {
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error(std::string("dense-cloud OpenCL ") + operation + " failed with error " +
                                         std::to_string(status));
            }
        }

        std::vector<cl_device_id> openClGpuDevices()
        {
            cl_uint platformCount = 0;
            const cl_int platformStatus = clGetPlatformIDs(0, nullptr, &platformCount);
            if (platformStatus != CL_SUCCESS || platformCount == 0)
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
                const std::size_t offset = devices.size();
                devices.resize(offset + deviceCount);
                checkOpenCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data() + offset, nullptr),
                            "GPU enumeration");
            }
            return devices;
        }

        std::string deviceString(cl_device_id device, cl_device_info property)
        {
            std::size_t size = 0;
            if (clGetDeviceInfo(device, property, 0, nullptr, &size) != CL_SUCCESS || size == 0)
            {
                return {};
            }
            std::string value(size, '\0');
            if (clGetDeviceInfo(device, property, value.size(), value.data(), nullptr) != CL_SUCCESS)
            {
                return {};
            }
            while (!value.empty() && value.back() == '\0')
            {
                value.pop_back();
            }
            return value;
        }

        class OpenClBuffer
        {
        public:
            OpenClBuffer(cl_context context,
                         std::size_t size,
                         cl_mem_flags flags = CL_MEM_READ_WRITE,
                         void* hostData = nullptr)
            {
                cl_int status = CL_SUCCESS;
                _memory = clCreateBuffer(context, flags, std::max<std::size_t>(size, 1), hostData, &status);
                checkOpenCl(status, "buffer allocation");
            }

            ~OpenClBuffer()
            {
                if (_memory)
                {
                    clReleaseMemObject(_memory);
                }
            }

            OpenClBuffer(const OpenClBuffer&) = delete;
            OpenClBuffer& operator=(const OpenClBuffer&) = delete;

            cl_mem get() const noexcept
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
                try
                {
                    const std::vector<cl_device_id> devices = openClGpuDevices();
                    if (deviceIndex < 0 || deviceIndex >= static_cast<int>(devices.size()))
                    {
                        throw std::runtime_error("dense-cloud OpenCL device index is unavailable");
                    }
                    _device = devices[static_cast<std::size_t>(deviceIndex)];
                    cl_int status = CL_SUCCESS;
                    _context = clCreateContext(nullptr, 1, &_device, nullptr, nullptr, &status);
                    checkOpenCl(status, "context creation");
                    _queue = clCreateCommandQueue(_context, _device, 0, &status);
                    checkOpenCl(status, "command queue creation");
                    const char* source = kUnprojectKernelSource;
                    const std::size_t sourceLength = std::strlen(kUnprojectKernelSource);
                    _program = clCreateProgramWithSource(_context, 1, &source, &sourceLength, &status);
                    checkOpenCl(status, "program creation");
                    status = clBuildProgram(_program, 1, &_device, "-cl-std=CL1.2", nullptr, nullptr);
                    if (status != CL_SUCCESS)
                    {
                        std::size_t logSize = 0;
                        clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                        std::string log(logSize, '\0');
                        clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
                        throw std::runtime_error("dense-cloud OpenCL kernel compilation failed: " + log);
                    }
                    _kernel = clCreateKernel(_program, "unproject_dense_cloud", &status);
                    checkOpenCl(status, "kernel creation");
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

            cl_context context() const noexcept
            {
                return _context;
            }

            cl_command_queue queue() const noexcept
            {
                return _queue;
            }

            cl_kernel kernel() const noexcept
            {
                return _kernel;
            }

            std::mutex& mutex() noexcept
            {
                return _mutex;
            }

        private:
            void release() noexcept
            {
                if (_kernel)
                {
                    clReleaseKernel(_kernel);
                    _kernel = nullptr;
                }
                if (_program)
                {
                    clReleaseProgram(_program);
                    _program = nullptr;
                }
                if (_queue)
                {
                    clReleaseCommandQueue(_queue);
                    _queue = nullptr;
                }
                if (_context)
                {
                    clReleaseContext(_context);
                    _context = nullptr;
                }
            }
            cl_device_id _device = nullptr;
            cl_context _context = nullptr;
            cl_command_queue _queue = nullptr;
            cl_program _program = nullptr;
            cl_kernel _kernel = nullptr;
            std::mutex _mutex;
        };

        OpenClRuntime& openClRuntime(int deviceIndex)
        {
            static std::mutex runtimesMutex;
            static std::unordered_map<int, std::unique_ptr<OpenClRuntime>> runtimes;
            std::scoped_lock lock(runtimesMutex);
            std::unique_ptr<OpenClRuntime>& runtime = runtimes[deviceIndex];
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

        bool hasZeroDistortion(const FramePinholeCamera& camera)
        {
            const FramePinholeCamera::Distortion distortion = camera.distortion();
            return distortion.radialK1 == 0.0 && distortion.radialK2 == 0.0 && distortion.radialK3 == 0.0 &&
                   distortion.tangentialP1 == 0.0 && distortion.tangentialP2 == 0.0;
        }

    } // namespace

    bool DensePointCloudOpenCL::isAvailable(int deviceIndex, std::string* errorMsg)
    {
        if (errorMsg)
        {
            errorMsg->clear();
        }
        try
        {
            (void)openClRuntime(deviceIndex);
            return true;
        }
        catch (const std::exception& error)
        {
            if (errorMsg)
            {
                *errorMsg = error.what();
            }
            return false;
        }
    }

    std::string DensePointCloudOpenCL::deviceName(int deviceIndex)
    {
        try
        {
            const std::vector<cl_device_id> devices = openClGpuDevices();
            if (deviceIndex < 0 || deviceIndex >= static_cast<int>(devices.size()))
            {
                return {};
            }
            return deviceString(devices[static_cast<std::size_t>(deviceIndex)], CL_DEVICE_NAME);
        }
        catch (...)
        {
            return {};
        }
    }

    std::vector<DensePoint> DensePointCloudOpenCL::unproject(const cv::Mat& depth,
                                                             const cv::Mat& mask,
                                                             const FramePinholeCamera& camera,
                                                             const cv::Mat& colorImage,
                                                             float minimumDepth,
                                                             float maximumDepth,
                                                             std::string* errorMsg,
                                                             const DenseCloudOptions* options)
    {
        if (errorMsg)
        {
            errorMsg->clear();
        }
        DenseCloudOptions effectiveOptions;
        if (options)
        {
            effectiveOptions = *options;
        }
        effectiveOptions.minDepth = minimumDepth;
        effectiveOptions.maxDepth = maximumDepth;
        if (!hasZeroDistortion(camera))
        {
            if (errorMsg)
            {
                *errorMsg = "OpenCL dense-cloud unprojection requires a prepared zero-distortion camera";
            }
            return {};
        }
        if (depth.empty())
        {
            return {};
        }

        try
        {
            OpenClRuntime& runtime = openClRuntime(effectiveOptions.openClDeviceIndex);
            std::scoped_lock runtimeLock(runtime.mutex());
            const std::int64_t elementCount64 = static_cast<std::int64_t>(depth.cols) * depth.rows;
            if (elementCount64 <= 0 || elementCount64 > std::numeric_limits<int>::max() / 3)
            {
                throw std::runtime_error("OpenCL dense-cloud image is too large");
            }
            const int elementCount = static_cast<int>(elementCount64);
            const int channels = colorImage.empty() ? 0 : colorImage.channels();
            const int hasMask = mask.empty() ? 0 : 1;
            const cv::Mat continuousDepth = depth.isContinuous() ? depth : depth.clone();
            const cv::Mat continuousMask = mask.empty() || mask.isContinuous() ? mask : mask.clone();
            const cv::Mat continuousColor =
                colorImage.empty() || colorImage.isContinuous() ? colorImage : colorImage.clone();
            std::uint8_t dummy = 0;

            OpenClBuffer depthBuffer(runtime.context(),
                                     static_cast<std::size_t>(elementCount) * sizeof(float),
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     const_cast<float*>(continuousDepth.ptr<float>()));
            OpenClBuffer maskBuffer(
                runtime.context(),
                continuousMask.empty() ? sizeof(dummy) : static_cast<std::size_t>(elementCount) * sizeof(std::uint8_t),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                continuousMask.empty() ? static_cast<void*>(&dummy) : static_cast<void*>(continuousMask.data));
            OpenClBuffer colorBuffer(
                runtime.context(),
                continuousColor.empty() ? sizeof(dummy)
                                        : static_cast<std::size_t>(elementCount) * channels * sizeof(std::uint8_t),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                continuousColor.empty() ? static_cast<void*>(&dummy) : static_cast<void*>(continuousColor.data));
            OpenClBuffer xyzBuffer(runtime.context(), static_cast<std::size_t>(elementCount) * 3 * sizeof(float));
            OpenClBuffer rgbBuffer(runtime.context(),
                                   static_cast<std::size_t>(elementCount) * 3 * sizeof(std::uint8_t));
            OpenClBuffer validBuffer(runtime.context(), static_cast<std::size_t>(elementCount) * sizeof(std::int32_t));

            const FramePinholeCamera::Intrinsics intrinsics = camera.intrinsics();
            const FramePinholeCamera::Pose pose = camera.pose();
            std::array<float, 9> rotation{};
            std::array<float, 3> center{};
            for (int index = 0; index < 9; ++index)
            {
                rotation[static_cast<std::size_t>(index)] =
                    static_cast<float>(pose.cameraToWorldRotation[static_cast<std::size_t>(index)]);
            }
            for (int index = 0; index < 3; ++index)
            {
                center[static_cast<std::size_t>(index)] =
                    static_cast<float>(pose.cameraCenter[static_cast<std::size_t>(index)]);
            }
            OpenClBuffer rotationBuffer(
                runtime.context(), sizeof(rotation), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rotation.data());
            OpenClBuffer centerBuffer(
                runtime.context(), sizeof(center), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, center.data());

            cl_kernel kernel = runtime.kernel();
            cl_uint argument = 0;
            setKernelBuffer(kernel, argument++, depthBuffer);
            setKernelBuffer(kernel, argument++, maskBuffer);
            setKernelValue(kernel, argument++, hasMask);
            setKernelBuffer(kernel, argument++, colorBuffer);
            setKernelValue(kernel, argument++, channels);
            setKernelValue(kernel, argument++, depth.cols);
            setKernelValue(kernel, argument++, depth.rows);
            setKernelValue(kernel, argument++, minimumDepth);
            setKernelValue(kernel, argument++, maximumDepth);
            const int subsample = std::max(1, effectiveOptions.subsample);
            setKernelValue(kernel, argument++, subsample);
            const int clipAabb = effectiveOptions.clipAABB ? 1 : 0;
            setKernelValue(kernel, argument++, clipAabb);
            setKernelValue(kernel, argument++, effectiveOptions.minX);
            setKernelValue(kernel, argument++, effectiveOptions.maxX);
            setKernelValue(kernel, argument++, effectiveOptions.minY);
            setKernelValue(kernel, argument++, effectiveOptions.maxY);
            setKernelValue(kernel, argument++, effectiveOptions.minZ);
            setKernelValue(kernel, argument++, effectiveOptions.maxZ);
            const float focalX = static_cast<float>(intrinsics.focalX);
            const float focalY = static_cast<float>(intrinsics.focalY);
            const float principalX = static_cast<float>(intrinsics.principalX);
            const float principalY = static_cast<float>(intrinsics.principalY);
            setKernelValue(kernel, argument++, focalX);
            setKernelValue(kernel, argument++, focalY);
            setKernelValue(kernel, argument++, principalX);
            setKernelValue(kernel, argument++, principalY);
            setKernelValue(kernel, argument++, intrinsics.uAxisSign);
            setKernelValue(kernel, argument++, intrinsics.vAxisSign);
            const int depthSign = pose.depthAxisFlipped ? -1 : 1;
            setKernelValue(kernel, argument++, depthSign);
            setKernelBuffer(kernel, argument++, rotationBuffer);
            setKernelBuffer(kernel, argument++, centerBuffer);
            setKernelBuffer(kernel, argument++, xyzBuffer);
            setKernelBuffer(kernel, argument++, rgbBuffer);
            setKernelBuffer(kernel, argument++, validBuffer);

            const std::size_t global[2] = {static_cast<std::size_t>(depth.cols), static_cast<std::size_t>(depth.rows)};
            checkOpenCl(
                clEnqueueNDRangeKernel(runtime.queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "kernel launch");

            std::vector<float> hostXyz(static_cast<std::size_t>(elementCount) * 3);
            std::vector<std::uint8_t> hostRgb(static_cast<std::size_t>(elementCount) * 3);
            std::vector<std::int32_t> hostValid(static_cast<std::size_t>(elementCount));
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            xyzBuffer.get(),
                                            CL_TRUE,
                                            0,
                                            hostXyz.size() * sizeof(float),
                                            hostXyz.data(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "point download");
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            rgbBuffer.get(),
                                            CL_TRUE,
                                            0,
                                            hostRgb.size() * sizeof(std::uint8_t),
                                            hostRgb.data(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "color download");
            checkOpenCl(clEnqueueReadBuffer(runtime.queue(),
                                            validBuffer.get(),
                                            CL_TRUE,
                                            0,
                                            hostValid.size() * sizeof(std::int32_t),
                                            hostValid.data(),
                                            0,
                                            nullptr,
                                            nullptr),
                        "validity download");

            std::vector<DensePoint> result;
            const std::size_t sampledColumns =
                (static_cast<std::size_t>(depth.cols) + static_cast<std::size_t>(subsample) - 1) /
                static_cast<std::size_t>(subsample);
            const std::size_t sampledRows =
                (static_cast<std::size_t>(depth.rows) + static_cast<std::size_t>(subsample) - 1) /
                static_cast<std::size_t>(subsample);
            result.reserve(sampledColumns * sampledRows);
            for (int index = 0; index < elementCount; ++index)
            {
                if (hostValid[static_cast<std::size_t>(index)] == 0)
                {
                    continue;
                }
                DensePoint point;
                point.x = hostXyz[static_cast<std::size_t>(index) * 3];
                point.y = hostXyz[static_cast<std::size_t>(index) * 3 + 1];
                point.z = hostXyz[static_cast<std::size_t>(index) * 3 + 2];
                point.r = hostRgb[static_cast<std::size_t>(index) * 3];
                point.g = hostRgb[static_cast<std::size_t>(index) * 3 + 1];
                point.b = hostRgb[static_cast<std::size_t>(index) * 3 + 2];
                result.push_back(point);
            }
            return result;
        }
        catch (const std::exception& error)
        {
            if (errorMsg)
            {
                *errorMsg = error.what();
            }
            return {};
        }
    }

} // namespace xjw::mvs
