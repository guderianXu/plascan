#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace
{

struct KeyFunctor
{
    float voxel = 1.0f;

    __host__ __device__
    int64_t operator()(const thrust::tuple<float, float, float> &point) const
    {
        const float x = thrust::get<0>(point);
        const float y = thrust::get<1>(point);
        const float z = thrust::get<2>(point);

        const int64_t ix = static_cast<int64_t>(floorf(x / voxel));
        const int64_t iy = static_cast<int64_t>(floorf(y / voxel));
        const int64_t iz = static_cast<int64_t>(floorf(z / voxel));

        uint64_t hash = 1469598103934665603ULL;
        hash ^= static_cast<uint64_t>(ix + 0x9e3779b97f4a7c15ULL);
        hash *= 1099511628211ULL;
        hash ^= static_cast<uint64_t>(iy + 0x9e3779b97f4a7c15ULL);
        hash *= 1099511628211ULL;
        hash ^= static_cast<uint64_t>(iz + 0x9e3779b97f4a7c15ULL);
        hash *= 1099511628211ULL;
        return static_cast<int64_t>(hash);
    }
};

struct TuplePlus
{
    __host__ __device__
    thrust::tuple<float, float, float, int> operator()(const thrust::tuple<float, float, float, int> &a,
                                                        const thrust::tuple<float, float, float, int> &b) const
    {
        return thrust::make_tuple(
            thrust::get<0>(a) + thrust::get<0>(b),
            thrust::get<1>(a) + thrust::get<1>(b),
            thrust::get<2>(a) + thrust::get<2>(b),
            thrust::get<3>(a) + thrust::get<3>(b));
    }
};

} // namespace

bool pointcloud_processor_cuda_runtime_available()
{
    int deviceCount = 0;
    const cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
}

bool pointcloud_processor_cuda_voxel_downsample(const float *xyz,
                                                int pointCount,
                                                float voxelSize,
                                                std::vector<float> *outXyz,
                                                std::string *detail)
{
    if (!xyz || !outXyz || pointCount <= 0 || voxelSize <= 0.0f)
    {
        return false;
    }

    outXyz->clear();

    try
    {
        thrust::host_vector<float> hostX(static_cast<std::size_t>(pointCount));
        thrust::host_vector<float> hostY(static_cast<std::size_t>(pointCount));
        thrust::host_vector<float> hostZ(static_cast<std::size_t>(pointCount));
        for (int index = 0; index < pointCount; ++index)
        {
            hostX[static_cast<std::size_t>(index)] = xyz[static_cast<std::size_t>(index) * 3 + 0];
            hostY[static_cast<std::size_t>(index)] = xyz[static_cast<std::size_t>(index) * 3 + 1];
            hostZ[static_cast<std::size_t>(index)] = xyz[static_cast<std::size_t>(index) * 3 + 2];
        }

        thrust::device_vector<float> deviceX = hostX;
        thrust::device_vector<float> deviceY = hostY;
        thrust::device_vector<float> deviceZ = hostZ;
        thrust::device_vector<int64_t> keys(static_cast<std::size_t>(pointCount));

        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(deviceX.begin(), deviceY.begin(), deviceZ.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(deviceX.end(), deviceY.end(), deviceZ.end())),
            keys.begin(),
            KeyFunctor{voxelSize});

        thrust::sort_by_key(
            keys.begin(),
            keys.end(),
            thrust::make_zip_iterator(thrust::make_tuple(deviceX.begin(), deviceY.begin(), deviceZ.begin())));

        thrust::device_vector<int> ones(static_cast<std::size_t>(pointCount), 1);
        thrust::device_vector<int64_t> outKeys(static_cast<std::size_t>(pointCount));
        thrust::device_vector<float> outSumX(static_cast<std::size_t>(pointCount));
        thrust::device_vector<float> outSumY(static_cast<std::size_t>(pointCount));
        thrust::device_vector<float> outSumZ(static_cast<std::size_t>(pointCount));
        thrust::device_vector<int> outCount(static_cast<std::size_t>(pointCount));

        auto endPair = thrust::reduce_by_key(
            keys.begin(),
            keys.end(),
            thrust::make_zip_iterator(thrust::make_tuple(deviceX.begin(), deviceY.begin(), deviceZ.begin(), ones.begin())),
            outKeys.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(outSumX.begin(), outSumY.begin(), outSumZ.begin(), outCount.begin())),
            thrust::equal_to<int64_t>(),
            TuplePlus());

        const int outputCount = static_cast<int>(endPair.first - outKeys.begin());
        if (outputCount <= 0)
        {
            return false;
        }

        thrust::host_vector<float> hostOutX(outputCount);
        thrust::host_vector<float> hostOutY(outputCount);
        thrust::host_vector<float> hostOutZ(outputCount);
        thrust::host_vector<int> hostOutCount(outputCount);
        thrust::copy(outSumX.begin(), outSumX.begin() + outputCount, hostOutX.begin());
        thrust::copy(outSumY.begin(), outSumY.begin() + outputCount, hostOutY.begin());
        thrust::copy(outSumZ.begin(), outSumZ.begin() + outputCount, hostOutZ.begin());
        thrust::copy(outCount.begin(), outCount.begin() + outputCount, hostOutCount.begin());

        outXyz->reserve(static_cast<std::size_t>(outputCount) * 3);
        for (int index = 0; index < outputCount; ++index)
        {
            const float count = hostOutCount[static_cast<std::size_t>(index)] > 0 ? static_cast<float>(hostOutCount[static_cast<std::size_t>(index)]) : 1.0f;
            outXyz->push_back(hostOutX[static_cast<std::size_t>(index)] / count);
            outXyz->push_back(hostOutY[static_cast<std::size_t>(index)] / count);
            outXyz->push_back(hostOutZ[static_cast<std::size_t>(index)] / count);
        }

        if (detail)
        {
            *detail = "CUDA voxel downsample ok";
        }
        return true;
    }
    catch (const std::exception &exception)
    {
        if (detail)
        {
            *detail = std::string("CUDA exception: ") + exception.what();
        }
        return false;
    }
}