#include "metalign/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace metalign {

std::vector<std::vector<RatioMatchResult>>
DescriptorAccelerator::ratio_match_batches(
    std::span<const RatioMatchBatch> batches, float ratio) {
    std::vector<std::vector<RatioMatchResult>> result;
    result.reserve(batches.size());
    for (const RatioMatchBatch& batch : batches) {
        if (!batch.queries || !batch.targets)
            throw std::runtime_error("invalid descriptor match batch");
        result.push_back(ratio_matches(*batch.queries, *batch.targets, ratio));
    }
    return result;
}

Image DescriptorAccelerator::laplacian_response(const Image&, float) {
    throw std::runtime_error("GPU backend does not provide LoG feature production");
}

Image DescriptorAccelerator::gaussian_blur(const Image&, double) {
    throw std::runtime_error("GPU backend does not provide Gaussian filtering");
}

Image DescriptorAccelerator::grayscale(const Image&) {
    throw std::runtime_error("GPU backend does not provide RGB to grayscale conversion");
}

std::vector<GpuExtremum> DescriptorAccelerator::locate_extrema(
    std::span<const Image>, int) {
    throw std::runtime_error("GPU backend does not provide fused LoG extrema detection");
}

std::vector<std::vector<float>> DescriptorAccelerator::orientation_peaks(
    const Image&, std::span<const FeaturePrimitive>) {
    throw std::runtime_error("GPU backend does not provide orientation production");
}

std::vector<Descriptor> DescriptorAccelerator::mldb_descriptors(
    const Image&, std::span<const FeaturePrimitive>) {
    throw std::runtime_error("GPU backend does not provide MLDB production");
}

std::vector<ResidentFeatureOctave>
DescriptorAccelerator::begin_resident_feature_image(const Image&, int) {
    throw std::runtime_error("GPU backend does not provide a resident feature pipeline");
}

std::vector<std::vector<float>> DescriptorAccelerator::resident_orientation_peaks(
    int, int, std::span<const FeaturePrimitive>) {
    throw std::runtime_error("GPU backend does not provide resident orientation production");
}

std::vector<Descriptor> DescriptorAccelerator::resident_mldb_descriptors(
    int, int, std::span<const FeaturePrimitive>) {
    throw std::runtime_error("GPU backend does not provide resident MLDB production");
}

Image DescriptorAccelerator::resident_feature_level(int, int) {
    throw std::runtime_error("GPU backend cannot download a resident feature level");
}

void DescriptorAccelerator::end_resident_feature_image() {
    throw std::runtime_error("GPU backend does not provide a resident feature pipeline");
}

#ifdef METALIGN_HAS_CUDA
std::vector<GpuDeviceInfo> enumerate_cuda_devices();
std::unique_ptr<DescriptorAccelerator> create_cuda_accelerator(int device_index);
#endif

namespace {

class AcceleratorPool final : public DescriptorAccelerator {
public:
    explicit AcceleratorPool(
        std::vector<std::unique_ptr<DescriptorAccelerator>> devices)
        : devices_(std::move(devices)) {
        if (devices_.empty()) throw std::runtime_error("empty accelerator pool");
    }

    std::string backend_name() const override {
        const std::string first = devices_.front()->backend_name();
        const bool homogeneous = std::all_of(
            devices_.begin(), devices_.end(), [&](const auto& device) {
                return device->backend_name() == first;
            });
        return homogeneous ? first + "-pool" : "cuda-opencl-target-pool";
    }
    std::string device_name() const override {
        std::ostringstream stream;
        for (std::size_t index = 0; index < devices_.size(); ++index) {
            if (index) stream << ", ";
            stream << devices_[index]->device_name();
        }
        return stream.str();
    }
    std::size_t device_count() const override { return devices_.size(); }
    void bind_worker_slot(std::size_t slot) override {
        std::lock_guard lock(sticky_mutex_);
        sticky_[std::this_thread::get_id()] = slot % devices_.size();
    }
    bool supports_feature_extraction() const override {
        return std::all_of(devices_.begin(), devices_.end(), [](const auto& device) {
            return device->supports_feature_extraction();
        });
    }
    bool supports_extrema_detection() const override {
        return std::all_of(devices_.begin(), devices_.end(), [](const auto& device) {
            return device->supports_extrema_detection();
        });
    }
    bool supports_device_gaussian() const override {
        return std::all_of(devices_.begin(), devices_.end(), [](const auto& device) {
            return device->supports_device_gaussian();
        });
    }
    bool supports_device_grayscale() const override {
        return std::all_of(devices_.begin(), devices_.end(), [](const auto& device) {
            return device->supports_device_grayscale();
        });
    }
    bool supports_resident_feature_pipeline() const override {
        return std::all_of(devices_.begin(), devices_.end(), [](const auto& device) {
            return device->supports_resident_feature_pipeline();
        });
    }

    Image grayscale(const Image& image) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.grayscale(image);
        });
    }

    Image gaussian_blur(const Image& image, double sigma) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.gaussian_blur(image, sigma);
        });
    }

    Image laplacian_response(const Image& image, float sigma) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.laplacian_response(image, sigma);
        });
    }
    std::vector<GpuExtremum> locate_extrema(
        std::span<const Image> levels, int octave) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.locate_extrema(levels, octave);
        });
    }
    std::vector<std::vector<float>> orientation_peaks(
        const Image& image, std::span<const FeaturePrimitive> points) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.orientation_peaks(image, points);
        });
    }
    std::vector<Descriptor> mldb_descriptors(
        const Image& image, std::span<const FeaturePrimitive> points) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.mldb_descriptors(image, points);
        });
    }
    std::vector<ResidentFeatureOctave> begin_resident_feature_image(
        const Image& image, int downscale) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.begin_resident_feature_image(image, downscale);
        });
    }
    std::vector<std::vector<float>> resident_orientation_peaks(
        int octave, int level, std::span<const FeaturePrimitive> points) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.resident_orientation_peaks(octave, level, points);
        });
    }
    std::vector<Descriptor> resident_mldb_descriptors(
        int octave, int level, std::span<const FeaturePrimitive> points) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.resident_mldb_descriptors(octave, level, points);
        });
    }
    Image resident_feature_level(int octave, int level) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.resident_feature_level(octave, level);
        });
    }
    void end_resident_feature_image() override {
        invoke([&](DescriptorAccelerator& device) {
            device.end_resident_feature_image();
            return 0;
        });
    }

    std::vector<RatioMatchResult> ratio_matches(
        const std::vector<Keypoint>& queries,
        const std::vector<Keypoint>& targets, float ratio) override {
        if (queries.empty()) return {};
        const std::vector<std::size_t> active = active_devices();
        if (active.size() == 1)
            return devices_[active.front()]->ratio_matches(queries, targets, ratio);
        std::vector<RatioMatchResult> result(queries.size());
        std::vector<std::future<void>> workers;
        workers.reserve(active.size());
        for (std::size_t shard = 0; shard < active.size(); ++shard) {
            const std::size_t begin = queries.size() * shard / active.size();
            const std::size_t end = queries.size() * (shard + 1) / active.size();
            workers.push_back(std::async(std::launch::async, [&, shard, begin, end] {
                std::vector<Keypoint> part(queries.begin() + static_cast<std::ptrdiff_t>(begin),
                                           queries.begin() + static_cast<std::ptrdiff_t>(end));
                auto values = devices_[active[shard]]->ratio_matches(
                    part, targets, ratio);
                std::copy(values.begin(), values.end(),
                          result.begin() + static_cast<std::ptrdiff_t>(begin));
            }));
        }
        for (auto& worker : workers) worker.get();
        return result;
    }
    std::vector<std::vector<RatioMatchResult>> ratio_match_batches(
        std::span<const RatioMatchBatch> batches, float ratio) override {
        return invoke([&](DescriptorAccelerator& device) {
            return device.ratio_match_batches(batches, ratio);
        });
    }

private:
    std::size_t sticky_start() {
        std::lock_guard lock(sticky_mutex_);
        const auto id = std::this_thread::get_id();
        const auto found = sticky_.find(id);
        if (found != sticky_.end()) return found->second;
        const std::size_t selected = next_.fetch_add(1) % devices_.size();
        sticky_.emplace(id, selected);
        return selected;
    }

    std::vector<std::size_t> active_devices() const {
        std::vector<std::size_t> result;
        for (std::size_t index = 0; index < devices_.size(); ++index)
            result.push_back(index);
        return result;
    }

    template <class Operation>
    auto invoke(Operation operation)
        -> std::invoke_result_t<Operation, DescriptorAccelerator&> {
        const std::size_t start = sticky_start();
        // A forced cuLaunchKernel failure in Metashape 2.3.2 aborts the whole
        // MatchPhotos task even when another GPU and cpu_enable are present.
        // Only device-construction failures are skipped by the factory.
        return operation(*devices_[start]);
    }

    std::vector<std::unique_ptr<DescriptorAccelerator>> devices_;
    std::mutex sticky_mutex_;
    std::unordered_map<std::thread::id, std::size_t> sticky_;
    std::atomic<std::size_t> next_{0};
};

}  // namespace
#ifdef METALIGN_HAS_OPENCL
std::vector<GpuDeviceInfo> enumerate_opencl_devices();
std::unique_ptr<DescriptorAccelerator> create_opencl_accelerator(int device_index);
#endif

std::vector<GpuDeviceInfo> enumerate_gpu_devices() {
    std::vector<GpuDeviceInfo> result;
#ifdef METALIGN_HAS_CUDA
    const auto cuda = enumerate_cuda_devices();
    result.insert(result.end(), cuda.begin(), cuda.end());
#endif
#ifdef METALIGN_HAS_OPENCL
    const auto opencl = enumerate_opencl_devices();
    result.insert(result.end(), opencl.begin(), opencl.end());
#endif
    return result;
}

std::unique_ptr<DescriptorAccelerator> create_descriptor_accelerator(
    const std::string& backend, int device_index, std::uint64_t device_mask,
    bool cpu_fallback) {
    auto enabled_indices = [&](const std::vector<GpuDeviceInfo>& devices) {
        std::vector<int> result;
        if (device_index >= 0) {
            result.push_back(device_index);
            return result;
        }
        for (const GpuDeviceInfo& device : devices)
            if (device.index < 64 &&
                (device_mask & (std::uint64_t{1} << device.index)) != 0)
                result.push_back(device.index);
        return result;
    };
    auto try_cuda = [&]() {
        std::vector<std::unique_ptr<DescriptorAccelerator>> result;
#ifdef METALIGN_HAS_CUDA
        for (int index : enabled_indices(enumerate_cuda_devices())) {
            try {
                result.push_back(create_cuda_accelerator(index));
            } catch (...) {
                if (device_index >= 0) throw;
            }
        }
#endif
        return result;
    };
    auto try_opencl = [&]() {
        std::vector<std::unique_ptr<DescriptorAccelerator>> result;
#ifdef METALIGN_HAS_OPENCL
        for (int index : enabled_indices(enumerate_opencl_devices())) {
            try {
                result.push_back(create_opencl_accelerator(index));
            } catch (...) {
                if (device_index >= 0) throw;
            }
        }
#endif
        return result;
    };
    auto finish = [](std::vector<std::unique_ptr<DescriptorAccelerator>> devices)
        -> std::unique_ptr<DescriptorAccelerator> {
        if (devices.empty()) return {};
        if (devices.size() == 1) return std::move(devices.front());
        return std::make_unique<AcceleratorPool>(std::move(devices));
    };
    auto try_auto = [&]() {
        std::vector<std::unique_ptr<DescriptorAccelerator>> result;
        std::exception_ptr last_error;
        std::vector<GpuDeviceInfo> cuda_devices;
        std::vector<GpuDeviceInfo> opencl_devices;
#ifdef METALIGN_HAS_CUDA
        cuda_devices = enumerate_cuda_devices();
#endif
#ifdef METALIGN_HAS_OPENCL
        opencl_devices = enumerate_opencl_devices();
#endif
        const auto duplicates_cuda = [&](const GpuDeviceInfo& opencl) {
            return std::any_of(cuda_devices.begin(), cuda_devices.end(),
                               [&](const GpuDeviceInfo& cuda) {
                                   return cuda.name == opencl.name;
                               });
        };
        const auto selected = [&](std::size_t logical_index) {
            if (device_index >= 0)
                return logical_index == static_cast<std::size_t>(device_index);
            return logical_index < 64 &&
                (device_mask & (std::uint64_t{1} << logical_index)) != 0;
        };

        // Runtime capture on the reference host established the physical mask
        // order Intel OpenCL, Intel Vulkan, NVIDIA CUDA.  Vulkan is deliberately
        // outside this replica, but its bit remains reserved so an original
        // Metashape mask can be passed unchanged.  OpenCL aliases of a CUDA
        // device are not a second physical worker.
        std::size_t logical_index = 0;
        std::size_t non_cuda_opencl_count = 0;
#ifdef METALIGN_HAS_OPENCL
        for (const GpuDeviceInfo& device : opencl_devices) {
            if (duplicates_cuda(device)) continue;
            if (selected(logical_index)) {
                try {
                    result.push_back(create_opencl_accelerator(device.index));
                } catch (...) {
                    last_error = std::current_exception();
                }
            }
            ++logical_index;
            ++non_cuda_opencl_count;
        }
#endif
        if (non_cuda_opencl_count != 0) ++logical_index;  // reserved Vulkan slot
#ifdef METALIGN_HAS_CUDA
        for (const GpuDeviceInfo& device : cuda_devices) {
            if (selected(logical_index)) {
                try {
                    result.push_back(create_cuda_accelerator(device.index));
                } catch (...) {
                    last_error = std::current_exception();
                }
            }
            ++logical_index;
        }
#endif
        if (result.empty() && last_error && !cpu_fallback)
            std::rethrow_exception(last_error);
        return result;
    };
    try {
        if (backend == "cuda") {
            if (auto result = finish(try_cuda())) return result;
        } else if (backend == "opencl") {
            if (auto result = finish(try_opencl())) return result;
        } else if (backend == "auto") {
            if (auto result = finish(try_auto())) return result;
        } else if (backend == "cpu") {
            return {};
        } else {
            throw std::runtime_error("unknown GPU backend: " + backend);
        }
    } catch (const std::exception&) {
        if (!cpu_fallback) throw;
        return {};
    }
    if (!cpu_fallback && backend != "cpu")
        throw std::runtime_error("requested GPU backend has no enabled device: " + backend);
    return {};
}

}  // namespace metalign
