#pragma once

#include "metmodel/options.hpp"
#include "metmodel/rpc.hpp"
#include "metalign/geometry.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace metmodel {

struct CostVolumeInput {
    std::size_t pixels = 0;
    std::size_t hypotheses = 0;
    std::size_t neighbors = 0;
    const std::vector<float>* reference = nullptr;
    const std::vector<float>* samples = nullptr;
};

struct ComputeDeviceInfo {
    ComputeBackend backend = ComputeBackend::CPU;
    std::size_t index = 0;
    std::string name = "CPU";
    std::size_t memory_bytes = 0;
};

class ComputeContext {
public:
    virtual ~ComputeContext() = default;
    virtual const ComputeDeviceInfo& device() const = 0;
    virtual bool compute_cost_volume(const CostVolumeInput& input,
                                     std::vector<float>& costs,
                                     std::string& error) = 0;
};

std::unique_ptr<ComputeContext> create_compute_context(const GPUOptions& options,
                                                       std::string& diagnostic);
std::vector<ComputeDeviceInfo> enumerate_compute_devices();
void compute_cost_volume_cpu(const CostVolumeInput& input, std::vector<float>& costs);

// The standard RPC route is a distinct production surface: it performs a
// geodetic height-plane sweep over two RPC00B rasters and cannot feed the
// recovered signed-depth/OOC input contract.
struct RpcCudaHeightPlaneSweepInput {
    Rpc00bCamera reference;
    Rpc00bCamera neighbor;
    std::size_t reference_width = 0;
    std::size_t reference_height = 0;
    std::size_t neighbor_width = 0;
    std::size_t neighbor_height = 0;
    const std::vector<float>* reference_gray = nullptr;
    const std::vector<float>* neighbor_gray = nullptr;
    int downscale = 4;
    std::size_t hypotheses = 64;
    double height_min = 0.0;
    double height_max = 0.0;
    std::size_t device_index = 0;
};

struct RpcCudaHeightPlaneSweepResult {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<metalign::Vec3> world;
    std::vector<float> confidence;
    std::vector<std::uint8_t> valid;
    std::string dispatch;
};

bool run_rpc_height_plane_sweep_cuda(const RpcCudaHeightPlaneSweepInput& input,
                                     RpcCudaHeightPlaneSweepResult& result,
                                     std::string& error);

#ifdef METMODEL_HAS_CUDA
std::unique_ptr<ComputeContext> create_cuda_context(const GPUOptions& options,
                                                    std::string& diagnostic);
std::vector<ComputeDeviceInfo> enumerate_cuda_devices();
#endif
#ifdef METMODEL_HAS_OPENCL
std::unique_ptr<ComputeContext> create_opencl_context(const GPUOptions& options,
                                                      std::string& diagnostic);
std::vector<ComputeDeviceInfo> enumerate_opencl_devices();
#endif
#ifdef METMODEL_HAS_VULKAN
std::unique_ptr<ComputeContext> create_vulkan_context(const GPUOptions& options,
                                                      std::string& diagnostic);
std::vector<ComputeDeviceInfo> enumerate_vulkan_devices();
#endif

}  // namespace metmodel
