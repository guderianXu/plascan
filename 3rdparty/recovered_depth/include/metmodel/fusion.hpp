#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace metmodel {

// Runtime-verified out-of-core variational-fusion layout used by Metashape
// 2.3.2 build 22956.  p/v/q are IEEE-754 binary16 bit patterns.
struct OocFusionState {
    std::vector<std::uint8_t> weights;       // count
    std::vector<std::uint8_t> histogram;     // count * 10
    std::vector<std::uint32_t> neighbors;    // count * 6: -x,+x,-y,+y,-z,+z
    std::vector<std::uint8_t> connectivity;  // six direction bits
    std::vector<std::uint8_t> refinement;    // adaptive four-child direction bits
    std::vector<std::uint8_t> flags;         // bit 2 means excluded
    std::vector<float> u;
    std::vector<float> u_old;
    std::vector<std::uint16_t> p;             // count * 3
    std::vector<std::uint16_t> v;             // count * 3
    std::vector<std::uint16_t> v_old;         // count * 3
    std::vector<std::uint16_t> q;             // count * 9, full 3x3 tensor

    [[nodiscard]] std::size_t size() const noexcept { return weights.size(); }
    void validate() const;
};

struct OocFusionParameters {
    float alpha{1.0F};
    float beta{0.05F};
    float data_weight{0.01F};
    std::size_t iterations{200};
};

struct OocFusionCudaStats {
    std::uint64_t kernel_launches{};
    std::uint64_t stream_synchronizations{};
    std::uint64_t host_to_device_bytes{};
    std::uint64_t device_to_host_bytes{};
};

[[nodiscard]] float half_to_float(std::uint16_t value) noexcept;
[[nodiscard]] std::uint16_t float_to_half(float value) noexcept;

// Exact CPU functional-type-2 update order recovered from sub_1EB56B0 and
// sub_1EB4B80.  In particular, projection uses float->half->float before the
// divisor and a second half quantization afterwards.
void run_ooc_fusion_cpu(OocFusionState& state,
                        const OocFusionParameters& parameters = {});

// Persistent-device implementation of the recovered functional-type-2
// schedule.  All twelve SoA arrays stay resident for the complete iteration
// chain; each update_u/update_p launch is followed by the target's stream
// synchronization boundary.
bool run_ooc_fusion_cuda(OocFusionState& state,
                         const OocFusionParameters& parameters,
                         std::size_t device_index,
                         OocFusionCudaStats& stats,
                         std::string& error);

}  // namespace metmodel
