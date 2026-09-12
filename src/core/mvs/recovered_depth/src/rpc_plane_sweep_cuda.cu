#include "metmodel/gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace metmodel
{
namespace
{

struct DeviceRpc
{
    double line_off, sample_off, lat_off, lon_off, height_off;
    double line_scale, sample_scale, lat_scale, lon_scale, height_scale;
    double line_num[20], line_den[20], sample_num[20], sample_den[20];
    double sample_offset, sample_lon, sample_lat, sample_height;
    double line_offset, line_lon, line_lat, line_height;
};

struct DeviceCell
{
    double lon, lat, height;
    float confidence;
    unsigned char valid;
};

DeviceRpc pack(const Rpc00bCamera& rpc)
{
    DeviceRpc result{};
    result.line_off = rpc.line_off; result.sample_off = rpc.sample_off;
    result.lat_off = rpc.lat_off; result.lon_off = rpc.lon_off; result.height_off = rpc.height_off;
    result.line_scale = rpc.line_scale; result.sample_scale = rpc.sample_scale;
    result.lat_scale = rpc.lat_scale; result.lon_scale = rpc.lon_scale; result.height_scale = rpc.height_scale;
    std::copy(rpc.line_num.begin(), rpc.line_num.end(), result.line_num);
    std::copy(rpc.line_den.begin(), rpc.line_den.end(), result.line_den);
    std::copy(rpc.sample_num.begin(), rpc.sample_num.end(), result.sample_num);
    std::copy(rpc.sample_den.begin(), rpc.sample_den.end(), result.sample_den);
    const auto& correction = rpc.affine_correction;
    result.sample_offset = correction.sample_offset; result.sample_lon = correction.sample_lon;
    result.sample_lat = correction.sample_lat; result.sample_height = correction.sample_height;
    result.line_offset = correction.line_offset; result.line_lon = correction.line_lon;
    result.line_lat = correction.line_lat; result.line_height = correction.line_height;
    return result;
}

__device__ double poly20(const double* c, double lon, double lat, double height)
{
    const double lon2 = lon * lon, lat2 = lat * lat, height2 = height * height;
    return c[0] + c[1] * lon + c[2] * lat + c[3] * height + c[4] * lon * lat +
        c[5] * lon * height + c[6] * lat * height + c[7] * lon2 + c[8] * lat2 +
        c[9] * height2 + c[10] * lon * lat * height + c[11] * lon * lon2 +
        c[12] * lon * lat2 + c[13] * lon * height2 + c[14] * lon2 * lat +
        c[15] * lat * lat2 + c[16] * lat * height2 + c[17] * lon2 * height +
        c[18] * lat2 * height + c[19] * height * height2;
}

__device__ bool project_rpc(const DeviceRpc& rpc, double lon, double lat, double height,
                            double& sample, double& line)
{
    const double normalized_lon = (lon - rpc.lon_off) / rpc.lon_scale;
    const double normalized_lat = (lat - rpc.lat_off) / rpc.lat_scale;
    const double normalized_height = (height - rpc.height_off) / rpc.height_scale;
    const double line_den = poly20(rpc.line_den, normalized_lon, normalized_lat, normalized_height);
    const double sample_den = poly20(rpc.sample_den, normalized_lon, normalized_lat, normalized_height);
    if (!isfinite(line_den) || !isfinite(sample_den) || fabs(line_den) < 1.0e-14 ||
        fabs(sample_den) < 1.0e-14) return false;
    line = rpc.line_off + rpc.line_scale * poly20(rpc.line_num, normalized_lon, normalized_lat, normalized_height) / line_den;
    sample = rpc.sample_off + rpc.sample_scale * poly20(rpc.sample_num, normalized_lon, normalized_lat, normalized_height) / sample_den;
    const double delta_lon = lon - rpc.lon_off, delta_lat = lat - rpc.lat_off;
    const double delta_height = height - rpc.height_off;
    sample += rpc.sample_offset + rpc.sample_lon * delta_lon + rpc.sample_lat * delta_lat + rpc.sample_height * delta_height;
    line += rpc.line_offset + rpc.line_lon * delta_lon + rpc.line_lat * delta_lat + rpc.line_height * delta_height;
    return isfinite(sample) && isfinite(line);
}

__device__ bool inverse_rpc_at_height(const DeviceRpc& rpc, double sample, double line,
                                      double height, double& lon, double& lat)
{
    lon = rpc.lon_off; lat = rpc.lat_off;
    const double delta_lon = fmax(fabs(rpc.lon_scale) * 1.0e-6, 1.0e-8);
    const double delta_lat = fmax(fabs(rpc.lat_scale) * 1.0e-6, 1.0e-8);
    for (int iteration = 0; iteration != 24; ++iteration)
    {
        double got_sample, got_line, sample_lon, line_lon, sample_lat, line_lat;
        if (!project_rpc(rpc, lon, lat, height, got_sample, got_line) ||
            !project_rpc(rpc, lon + delta_lon, lat, height, sample_lon, line_lon) ||
            !project_rpc(rpc, lon, lat + delta_lat, height, sample_lat, line_lat)) return false;
        const double residual_sample = sample - got_sample, residual_line = line - got_line;
        if (fmax(fabs(residual_sample), fabs(residual_line)) < 1.0e-7) return true;
        const double a = (sample_lon - got_sample) / delta_lon;
        const double b = (sample_lat - got_sample) / delta_lat;
        const double c = (line_lon - got_line) / delta_lon;
        const double d = (line_lat - got_line) / delta_lat;
        const double determinant = a * d - b * c;
        if (!isfinite(determinant) || fabs(determinant) < 1.0e-18) return false;
        lon += (residual_sample * d - residual_line * b) / determinant;
        lat += (a * residual_line - c * residual_sample) / determinant;
        if (!isfinite(lon) || !isfinite(lat)) return false;
    }
    return false;
}

__device__ float bilinear(const float* image, int width, int height, double x, double y)
{
    if (x < 0.0 || y < 0.0 || x > double(width - 1) || y > double(height - 1)) return NAN;
    const int x0 = int(floor(x)), y0 = int(floor(y));
    const int x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    const float dx = float(x - double(x0)), dy = float(y - double(y0));
    const float top = image[y0 * width + x0] * (1.0F - dx) + image[y0 * width + x1] * dx;
    const float bottom = image[y1 * width + x0] * (1.0F - dx) + image[y1 * width + x1] * dx;
    return top * (1.0F - dy) + bottom * dy;
}

__global__ void rpc_height_plane_sweep_kernel(DeviceRpc reference, DeviceRpc neighbor,
                                                const float* reference_image, const float* neighbor_image,
                                                int reference_width, int reference_height,
                                                int neighbor_width, int neighbor_height, int downscale,
                                                int grid_width, int grid_height, int hypotheses,
                                                double height_min, double height_max, DeviceCell* output)
{
    const int x = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= grid_width || y >= grid_height) return;
    const int index = y * grid_width + x;
    const double reference_sample = (double(x) + 0.5) * double(downscale);
    const double reference_line = (double(y) + 0.5) * double(downscale);
    DeviceCell best{};
    float best_cost = INFINITY, second_cost = INFINITY;
    for (int candidate = 0; candidate < hypotheses; ++candidate)
    {
        const double ratio = hypotheses == 1 ? 0.0 : double(candidate) / double(hypotheses - 1);
        const double height = height_min + (height_max - height_min) * ratio;
        double lon, lat, neighbor_sample, neighbor_line;
        if (!inverse_rpc_at_height(reference, reference_sample, reference_line, height, lon, lat) ||
            !project_rpc(neighbor, lon, lat, height, neighbor_sample, neighbor_line)) continue;
        float sum_a = 0.0F, sum_b = 0.0F, sum_aa = 0.0F, sum_bb = 0.0F, sum_ab = 0.0F;
        bool patch_valid = true;
        for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx)
        {
            const float a = bilinear(reference_image, reference_width, reference_height,
                                     reference_sample + double(dx), reference_line + double(dy));
            const float b = bilinear(neighbor_image, neighbor_width, neighbor_height,
                                     neighbor_sample + double(dx), neighbor_line + double(dy));
            if (!isfinite(a) || !isfinite(b)) { patch_valid = false; continue; }
            sum_a += a; sum_b += b; sum_aa += a * a; sum_bb += b * b; sum_ab += a * b;
        }
        if (!patch_valid) continue;
        constexpr float count = 25.0F;
        const float covariance = sum_ab - sum_a * sum_b / count;
        const float variance_a = sum_aa - sum_a * sum_a / count;
        const float variance_b = sum_bb - sum_b * sum_b / count;
        if (!(variance_a > 1.0e-8F) || !(variance_b > 1.0e-8F)) continue;
        const float cost = 1.0F - covariance / sqrtf(variance_a * variance_b);
        if (cost < best_cost) { second_cost = best_cost; best_cost = cost; best.lon = lon; best.lat = lat; best.height = height; }
        else if (cost < second_cost) second_cost = cost;
    }
    if (isfinite(best_cost) && best_cost <= 1.2F)
    {
        best.confidence = fminf(1.0F, fmaxf(0.0F, second_cost - best_cost));
        best.valid = 1U;
    }
    output[index] = best;
}

} // namespace

bool run_rpc_height_plane_sweep_cuda(const RpcCudaHeightPlaneSweepInput& input,
                                     RpcCudaHeightPlaneSweepResult& result,
                                     std::string& error)
{
    if (!input.reference_gray || !input.neighbor_gray || input.downscale <= 0 || input.hypotheses < 8U ||
        !std::isfinite(input.height_min) || !std::isfinite(input.height_max) || !(input.height_min < input.height_max) ||
        input.reference_gray->size() != input.reference_width * input.reference_height ||
        input.neighbor_gray->size() != input.neighbor_width * input.neighbor_height ||
        !validate_rpc00b(input.reference, error) || !validate_rpc00b(input.neighbor, error))
    {
        if (error.empty()) error = "RPC CUDA plane sweep requires two decoded RPC00B rasters, >=8 hypotheses and a finite height gate";
        return false;
    }
    if (cudaSetDevice(static_cast<int>(input.device_index)) != cudaSuccess)
    {
        error = "cannot select CUDA device";
        return false;
    }
    const int grid_width = int(input.reference_width / static_cast<std::size_t>(input.downscale));
    const int grid_height = int(input.reference_height / static_cast<std::size_t>(input.downscale));
    if (grid_width < 2 || grid_height < 2)
    {
        error = "RPC CUDA height-plane-sweep grid is empty";
        return false;
    }
    float* device_reference = nullptr;
    float* device_neighbor = nullptr;
    DeviceCell* device_output = nullptr;
    const std::size_t reference_bytes = input.reference_gray->size() * sizeof(float);
    const std::size_t neighbor_bytes = input.neighbor_gray->size() * sizeof(float);
    const std::size_t output_bytes = std::size_t(grid_width) * std::size_t(grid_height) * sizeof(DeviceCell);
    cudaError_t status = cudaMalloc(&device_reference, reference_bytes);
    if (status == cudaSuccess) status = cudaMalloc(&device_neighbor, neighbor_bytes);
    if (status == cudaSuccess) status = cudaMalloc(&device_output, output_bytes);
    if (status == cudaSuccess) status = cudaMemcpy(device_reference, input.reference_gray->data(), reference_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemcpy(device_neighbor, input.neighbor_gray->data(), neighbor_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        error = cudaGetErrorString(status);
        cudaFree(device_reference); cudaFree(device_neighbor); cudaFree(device_output);
        return false;
    }
    const dim3 threads(16U, 16U);
    const dim3 blocks((unsigned(grid_width) + 15U) / 16U, (unsigned(grid_height) + 15U) / 16U);
    rpc_height_plane_sweep_kernel<<<blocks, threads>>>(pack(input.reference), pack(input.neighbor), device_reference,
        device_neighbor, int(input.reference_width), int(input.reference_height), int(input.neighbor_width),
        int(input.neighbor_height), input.downscale, grid_width, grid_height, int(input.hypotheses), input.height_min,
        input.height_max, device_output);
    status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    std::vector<DeviceCell> cells(std::size_t(grid_width) * std::size_t(grid_height));
    if (status == cudaSuccess) status = cudaMemcpy(cells.data(), device_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(device_reference); cudaFree(device_neighbor); cudaFree(device_output);
    if (status != cudaSuccess)
    {
        error = cudaGetErrorString(status);
        return false;
    }
    result.width = std::size_t(grid_width); result.height = std::size_t(grid_height);
    result.world.resize(cells.size()); result.confidence.resize(cells.size()); result.valid.resize(cells.size());
    for (std::size_t index = 0; index < cells.size(); ++index)
    {
        result.world[index] = {cells[index].lon, cells[index].lat, cells[index].height};
        result.confidence[index] = cells[index].confidence;
        result.valid[index] = cells[index].valid;
    }
    std::ostringstream dispatch;
    dispatch << "RPC CUDA source dispatch=rpc_height_plane_sweep_kernel camera_type=rpc model=RPC00B_4x20 candidates="
             << input.hypotheses << " height_m=[" << input.height_min << ',' << input.height_max << "] grid="
             << grid_width << 'x' << grid_height << " fallback=none type8=unused";
    result.dispatch = dispatch.str();
    error.clear();
    return true;
}

} // namespace metmodel
