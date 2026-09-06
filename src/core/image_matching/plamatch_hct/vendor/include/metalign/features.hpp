#pragma once

#include "metalign/image.hpp"
#include "metalign/options.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace metalign {

class DescriptorAccelerator;

// Metashape 2.3.2's default MLDB payload has 498 meaningful bits in 63 bytes.
// The host/device matcher pads it to 64 bytes, so the replica keeps the same
// padded representation and always leaves the final six bits clear.
constexpr std::size_t kDescriptorPayloadBytes = 63;
constexpr std::size_t kDescriptorSize = 64;
using Descriptor = std::array<std::uint8_t, kDescriptorSize>;
using GlobalDescriptor = std::array<float, 64>;

struct Keypoint {
    double x = 0.0;
    double y = 0.0;
    double scale = 1.0;
    double orientation = 0.0;
    double response = 0.0;
    // Original detector scale-space coordinates retained through orientation
    // expansion.  Metashape's 36-byte selector record carries these at +24
    // and +28 and uses them in the recovered tie comparator.
    int octave = 0;
    int level = 0;
    // Pre-orientation detector identity.  Several descriptor rows can share
    // this value and are consolidated during pair matching.
    std::size_t detector_id = 0;
    // Direction-expanded descriptor-row identity consumed by track building.
    std::size_t source_id = 0;
    Descriptor descriptor{};
    int laplacian_sign = 1;
};

struct FeatureSet {
    std::filesystem::path path;
    std::size_t image_width = 0;
    std::size_t image_height = 0;
    std::optional<double> focal_length_pixels;
    // Metashape optimizes one calibration block per sensor. The default photo
    // CLI assigns every frame to sensor zero; --sensor-csv can provide the
    // persistent sensor identity used by the general sparse BA path.
    std::size_t sensor_id = 0;
    std::vector<Keypoint> keypoints;
    // Generic preselection uses a separately selected 2,048-row descriptor
    // stream.  It is not a prefix of the full keypoint stream.
    std::vector<Keypoint> coarse_keypoints;
    std::size_t source_keypoint_count = 0;
    GlobalDescriptor global_descriptor{};
};

class FeatureExtractor {
public:
    explicit FeatureExtractor(MatchPhotosOptions options,
                              DescriptorAccelerator* accelerator = nullptr,
                              bool parallel_cpu_worker = false)
        : options_(std::move(options)), accelerator_(accelerator),
          parallel_cpu_worker_(parallel_cpu_worker) {}
    FeatureSet extract(const std::filesystem::path& path,
                       const std::filesystem::path* mask_path = nullptr) const;
    FeatureSet extract(const Image& image,
                       const std::filesystem::path& path = {},
                       const Image* valid_mask = nullptr) const;

private:
    MatchPhotosOptions options_;
    DescriptorAccelerator* accelerator_ = nullptr;
    bool parallel_cpu_worker_ = false;
};

std::uint32_t descriptor_hamming_distance(const Descriptor& left, const Descriptor& right);
double descriptor_cosine(const GlobalDescriptor& left, const GlobalDescriptor& right);
Descriptor compute_mldb_descriptor(const Image& image, float x, float y,
                                   float scale, float orientation);
std::vector<float> compute_orientation_peaks(const Image& image, float x, float y,
                                             float scale);

}  // namespace metalign
