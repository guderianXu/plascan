#pragma once

#include "metalign/features.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalign {

struct GpuDeviceInfo {
    std::string backend;
    int index = 0;
    std::string name;
    std::uint64_t memory_bytes = 0;
};

struct RatioMatchResult {
    std::int32_t target = -1;
    double distance = 0.0;
};

// One directed descriptor search inside a MatchPhotos batch.  The pointed-to
// vectors must outlive ratio_match_batches(); backends use their identity to
// upload each image/sign group once and then issue all pair kernels by offset.
struct RatioMatchBatch {
    const std::vector<Keypoint>* queries = nullptr;
    const std::vector<Keypoint>* targets = nullptr;
};

struct FeaturePrimitive {
    float x = 0.0F;
    float y = 0.0F;
    float scale = 1.0F;
    float orientation = 0.0F;
};

// Native 36-byte record produced by Metashape's CUDA/OpenCL LoG extrema
// kernel before either spatial selector runs.  Keep the unused z slot and the
// pre-orientation -1 value: both are part of the observed device ABI.
struct GpuExtremum {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float scale = 1.0F;
    float sign_or_orientation = -1.0F;
    float response = 0.0F;
    std::uint32_t octave = 0;
    std::uint32_t level = 0;
    std::uint32_t flag = 0;
};
static_assert(sizeof(GpuExtremum) == 36);

// One host-visible row per octave in a device-resident feature session.  Only
// dimensions and extrema cross the PCIe boundary; the five Gaussian planes
// remain owned by the backend until end_resident_feature_image().
struct ResidentFeatureOctave {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<GpuExtremum> extrema;
};

class DescriptorAccelerator {
public:
    virtual ~DescriptorAccelerator() = default;
    virtual std::string backend_name() const = 0;
    virtual std::string device_name() const = 0;
    virtual std::size_t device_count() const { return 1; }
    // Persistent MatchPhotos workers are assigned to physical devices by the
    // scheduler. Pools override this so startup races cannot swap devices.
    virtual void bind_worker_slot(std::size_t) {}
    virtual std::vector<RatioMatchResult> ratio_matches(
        const std::vector<Keypoint>& queries,
        const std::vector<Keypoint>& targets,
        float ratio) = 0;
    virtual std::vector<std::vector<RatioMatchResult>> ratio_match_batches(
        std::span<const RatioMatchBatch> batches,
        float ratio);
    virtual bool supports_feature_extraction() const { return false; }
    virtual bool supports_extrema_detection() const { return false; }
    virtual bool supports_device_gaussian() const { return false; }
    virtual bool supports_device_grayscale() const { return false; }
    virtual bool supports_resident_feature_pipeline() const { return false; }
    virtual Image grayscale(const Image& image);
    virtual Image gaussian_blur(const Image& image, double sigma);
    virtual Image laplacian_response(const Image& image, float sigma);
    virtual std::vector<GpuExtremum> locate_extrema(
        std::span<const Image> gaussian_levels, int octave);
    virtual std::vector<std::vector<float>> orientation_peaks(
        const Image& image, std::span<const FeaturePrimitive> points);
    virtual std::vector<Descriptor> mldb_descriptors(
        const Image& image, std::span<const FeaturePrimitive> points);
    // downscale uses the target MatchPhotos API values. Zero means the
    // Highest 2W-1 lattice expansion; 1/2/4/8 are integer sampling factors.
    virtual std::vector<ResidentFeatureOctave> begin_resident_feature_image(
        const Image& image, int downscale);
    virtual std::vector<std::vector<float>> resident_orientation_peaks(
        int octave, int level, std::span<const FeaturePrimitive> points);
    virtual std::vector<Descriptor> resident_mldb_descriptors(
        int octave, int level, std::span<const FeaturePrimitive> points);
    virtual Image resident_feature_level(int octave, int level);
    virtual void end_resident_feature_image();
};

std::vector<GpuDeviceInfo> enumerate_gpu_devices();
std::unique_ptr<DescriptorAccelerator> create_descriptor_accelerator(
    const std::string& backend, int device_index, std::uint64_t device_mask,
    bool cpu_fallback);

}  // namespace metalign
