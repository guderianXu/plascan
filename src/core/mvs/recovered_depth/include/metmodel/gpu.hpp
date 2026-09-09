#pragma once

#include "metmodel/options.hpp"

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
