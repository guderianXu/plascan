#include "metmodel/fusion.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace metmodel
{

#if defined(METMODEL_HAS_CUDA)
    bool run_ooc_fusion_cuda_impl(OocFusionState& state,
                                  const OocFusionParameters& parameters,
                                  std::size_t device_index,
                                  OocFusionCudaStats& stats,
                                  std::string& error);
    bool run_ooc_fusion_cuda_source_impl(OocFusionState& state,
                                         const OocFusionParameters& parameters,
                                         std::size_t device_index,
                                         OocFusionCudaStats& stats,
                                         std::string& error);
#endif

    namespace
    {

        constexpr float kInvSqrt12 = 0.28867528F;
        constexpr std::array<std::array<std::uint32_t, 4>, 3> kChildOffsets{{
            {{0, 1, 2, 3}},
            {{0, 4, 1, 5}},
            {{0, 4, 2, 6}},
        }};

        [[nodiscard]] bool bit(std::uint8_t value, unsigned index) noexcept
        {
            return (value & static_cast<std::uint8_t>(1U << index)) != 0;
        }

        [[nodiscard]] std::size_t neighbor(const OocFusionState& state, std::size_t voxel, unsigned direction)
        {
            const auto value = state.neighbors[voxel * 6 + direction];
            if (value == std::numeric_limits<std::uint32_t>::max() || value >= state.size())
            {
                throw std::runtime_error("fusion neighbor index is invalid for an enabled direction");
            }
            return value;
        }

        [[nodiscard]] float h(const std::vector<std::uint16_t>& values, std::size_t index) noexcept
        {
            return half_to_float(values[index]);
        }

        void update_u_v(OocFusionState& s, float alpha, float beta, float data_weight)
        {
            const auto count = s.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (std::size_t voxel = 0; voxel < count; ++voxel)
            {
                if ((s.flags[voxel] & 4U) != 0)
                    continue;

                const float current_u = s.u[voxel];
                s.u_old[voxel] = current_u;
                float divergence_p = 0.0F;
                std::array<float, 3> local_p{};
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    const unsigned even = 2U * axis;
                    local_p[axis] = h(s.p, voxel * 3 + axis);
                    if (bit(s.connectivity[voxel], even + 1U))
                    {
                        divergence_p = divergence_p - local_p[axis];
                    }
                    if (bit(s.connectivity[voxel], even))
                    {
                        const auto adjacent = neighbor(s, voxel, even);
                        if (bit(s.refinement[voxel], even))
                        {
                            for (const auto offset : kChildOffsets[axis])
                            {
                                const auto index = adjacent + offset;
                                if (index >= count)
                                    throw std::runtime_error("fusion child index is invalid");
                                divergence_p = divergence_p + h(s.p, index * 3 + axis) * 0.25F;
                            }
                        }
                        else
                        {
                            divergence_p = divergence_p + h(s.p, adjacent * 3 + axis);
                        }
                    }
                }

                const float u_bar = current_u - divergence_p * (kInvSqrt12 * alpha);
                float total = 0.0F;
                float hard_value = 256.0F;
                std::array<float, 10> centers{};
                for (unsigned bin_index = 0; bin_index < 10; ++bin_index)
                {
                    const float fraction = static_cast<float>(bin_index) / 9.0F;
                    const float center = (fraction + fraction) - 1.0F;
                    centers[bin_index] = center;
                    const auto vote = s.histogram[voxel * 10 + bin_index];
                    if (vote == 255U)
                        hard_value = center;
                    total = total + static_cast<float>(vote);
                }

                float local_weight =
                    ((static_cast<float>(s.weights[voxel]) / 255.89999F) * 0.75F + 0.75F) * data_weight;
                float histogram_scale = 1.0F;
                if (total != 0.0F)
                {
                    histogram_scale = 5.0F / total;
                    total = total * histogram_scale;
                }
                local_weight = local_weight * kInvSqrt12;
                float offset = -u_bar - total * local_weight;
                float result = 2391.2012F;
                for (unsigned bin_index = 0; bin_index < 10; ++bin_index)
                {
                    const float center = centers[bin_index];
                    const float left = (center - 0.11111111F) + offset;
                    if (left > 0.0F)
                    {
                        result = -offset;
                        break;
                    }
                    const auto vote = static_cast<float>(s.histogram[voxel * 10 + bin_index]);
                    offset = offset + ((local_weight + local_weight) * histogram_scale) * vote;
                    const float right = (center + 0.11111111F) + offset;
                    if (right > 0.0F)
                    {
                        result = ((1.0F - (right + right) / (right - left)) * 0.11111111F) + center;
                        break;
                    }
                }
                if (hard_value != 256.0F)
                {
                    result = hard_value;
                    s.u_old[voxel] = hard_value;
                }
                s.u[voxel] = result;

                for (unsigned component = 0; component < 3; ++component)
                {
                    s.v_old[voxel * 3 + component] = s.v[voxel * 3 + component];
                    float divergence_q = 0.0F;
                    for (unsigned axis = 0; axis < 3; ++axis)
                    {
                        const unsigned even = 2U * axis;
                        const auto q_index = (voxel * 3 + component) * 3 + axis;
                        const float value = h(s.q, q_index);
                        if (bit(s.connectivity[voxel], even + 1U))
                        {
                            divergence_q = divergence_q - value;
                        }
                        if (bit(s.connectivity[voxel], even))
                        {
                            const auto adjacent = neighbor(s, voxel, even);
                            if (bit(s.refinement[voxel], even))
                            {
                                for (const auto child : kChildOffsets[axis])
                                {
                                    const auto index = adjacent + child;
                                    if (index >= count)
                                        throw std::runtime_error("fusion child index is invalid");
                                    divergence_q = divergence_q + h(s.q, (index * 3 + component) * 3 + axis) * 0.25F;
                                }
                            }
                            else
                            {
                                divergence_q = divergence_q + h(s.q, (adjacent * 3 + component) * 3 + axis);
                            }
                        }
                    }
                    const float delta = local_p[component] * alpha - divergence_q * beta;
                    float candidate = delta * kInvSqrt12 + h(s.v, voxel * 3 + component);
                    if (candidate < -2.0F)
                        candidate = -2.0F;
                    if (candidate > 2.0F)
                        candidate = 2.0F;
                    s.v[voxel * 3 + component] = float_to_half(candidate);
                }
            }
        }

        void update_p_q(OocFusionState& s, float alpha, float beta)
        {
            const auto count = s.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (std::size_t voxel = 0; voxel < count; ++voxel)
            {
                if ((s.flags[voxel] & 4U) != 0)
                    continue;
                const float center_u_bar = (0.0F - (s.u[voxel] + s.u[voxel])) + s.u_old[voxel];
                std::array<float, 3> center_v_bar{};
                std::array<float, 3> p_candidate{};
                std::array<std::array<float, 3>, 3> q_candidate{};
                for (unsigned component = 0; component < 3; ++component)
                {
                    const float current = h(s.v, voxel * 3 + component);
                    center_v_bar[component] = (current + current) - h(s.v_old, voxel * 3 + component);
                }

                for (unsigned component = 0; component < 3; ++component)
                {
                    const unsigned positive = 2U * component + 1U;
                    float gradient_u = 0.0F;
                    if (bit(s.connectivity[voxel], positive))
                    {
                        const auto adjacent = neighbor(s, voxel, positive);
                        float neighbor_u_bar = 0.0F;
                        if (bit(s.refinement[voxel], positive))
                        {
                            for (const auto child : kChildOffsets[component])
                            {
                                const auto index = adjacent + child;
                                if (index >= count)
                                    throw std::runtime_error("fusion child index is invalid");
                                neighbor_u_bar = neighbor_u_bar + (s.u[index] + s.u[index]);
                                neighbor_u_bar = neighbor_u_bar - s.u_old[index];
                            }
                            neighbor_u_bar = neighbor_u_bar * 0.25F;
                        }
                        else
                        {
                            neighbor_u_bar = ((s.u[adjacent] + s.u[adjacent]) + 0.0F) - s.u_old[adjacent];
                        }
                        gradient_u = neighbor_u_bar + center_u_bar;
                    }
                    p_candidate[component] =
                        (gradient_u - center_v_bar[component]) * (alpha * kInvSqrt12) + h(s.p, voxel * 3 + component);

                    for (unsigned axis = 0; axis < 3; ++axis)
                    {
                        const unsigned positive_axis = 2U * axis + 1U;
                        float gradient_v = 0.0F;
                        if (bit(s.connectivity[voxel], positive_axis))
                        {
                            const auto adjacent = neighbor(s, voxel, positive_axis);
                            float neighbor_v_bar = 0.0F;
                            if (bit(s.refinement[voxel], positive_axis))
                            {
                                for (const auto child : kChildOffsets[axis])
                                {
                                    const auto index = adjacent + child;
                                    if (index >= count)
                                        throw std::runtime_error("fusion child index is invalid");
                                    const float current = h(s.v, index * 3 + component);
                                    neighbor_v_bar = neighbor_v_bar + (current + current);
                                    neighbor_v_bar = neighbor_v_bar - h(s.v_old, index * 3 + component);
                                }
                                neighbor_v_bar = neighbor_v_bar * 0.25F;
                            }
                            else
                            {
                                const float current = h(s.v, adjacent * 3 + component);
                                neighbor_v_bar = ((current + current) + 0.0F) - h(s.v_old, adjacent * 3 + component);
                            }
                            gradient_v = neighbor_v_bar - center_v_bar[component];
                        }
                        q_candidate[component][axis] =
                            gradient_v * (beta * kInvSqrt12) + h(s.q, (voxel * 3 + component) * 3 + axis);
                    }
                }

                const float p_norm_sq = (p_candidate[1] * p_candidate[1] + p_candidate[0] * p_candidate[0]) +
                                        p_candidate[2] * p_candidate[2];
                float p_divisor = 1.0F;
                if (p_norm_sq > alpha * alpha)
                    p_divisor = std::sqrt(p_norm_sq) / alpha;
                for (unsigned component = 0; component < 3; ++component)
                {
                    const float prequantized = half_to_float(float_to_half(p_candidate[component]));
                    s.p[voxel * 3 + component] = float_to_half(prequantized / p_divisor);
                }

                float q_norm_sq = 0.0F;
                for (unsigned row = 0; row < 3; ++row)
                {
                    q_norm_sq = q_candidate[row][2] * q_candidate[row][2] +
                                (q_candidate[row][1] * q_candidate[row][1] +
                                 (q_norm_sq + q_candidate[row][0] * q_candidate[row][0]));
                }
                float q_divisor = 1.0F;
                if (q_norm_sq > beta * beta)
                    q_divisor = std::sqrt(q_norm_sq) / beta;
                for (unsigned row = 0; row < 3; ++row)
                {
                    for (unsigned column = 0; column < 3; ++column)
                    {
                        const float prequantized = half_to_float(float_to_half(q_candidate[row][column]));
                        s.q[(voxel * 3 + row) * 3 + column] = float_to_half(prequantized / q_divisor);
                    }
                }
            }
        }

    } // namespace

    float half_to_float(std::uint16_t value) noexcept
    {
        const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
        std::uint32_t exponent = (value >> 10U) & 0x1FU;
        std::uint32_t mantissa = value & 0x3FFU;
        std::uint32_t bits = 0;
        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                exponent = 113U;
                while ((mantissa & 0x400U) == 0)
                {
                    mantissa <<= 1U;
                    --exponent;
                }
                mantissa &= 0x3FFU;
                bits = sign | (exponent << 23U) | (mantissa << 13U);
            }
        }
        else if (exponent == 31U)
        {
            bits = sign | 0x7F800000U | (mantissa << 13U);
        }
        else
        {
            bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
        }
        return std::bit_cast<float>(bits);
    }

    std::uint16_t float_to_half(float value) noexcept
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        const std::uint32_t magnitude = bits & 0x7FFFFFFFU;
        std::uint32_t result = (bits >> 16U) & 0x8000U;
        if (magnitude <= 0x387FFFFFU)
        {
            if (magnitude > 0x33000000U)
            {
                const std::uint32_t exponent = magnitude >> 23U;
                const std::uint32_t mantissa = (bits & 0x7FFFFFU) | 0x800000U;
                const std::uint32_t remainder = mantissa << (exponent - 94U);
                result |= mantissa >> (126U - exponent);
                if (remainder > 0x80000000U || (remainder == 0x80000000U && (result & 1U) != 0))
                {
                    ++result;
                }
            }
        }
        else if (magnitude > 0x7F7FFFFFU)
        {
            result |= 0x7C00U;
            if (magnitude != 0x7F800000U)
            {
                const auto payload = (magnitude >> 13U) & 0x3FFU;
                result |= payload == 0 ? 1U : payload;
            }
        }
        else if (magnitude > 0x477FEFFFU)
        {
            result |= 0x7C00U;
        }
        else
        {
            result |= (magnitude + ((magnitude & 0x2000U) != 0 ? 1U : 0U) - 0x37FFF001U) >> 13U;
        }
        return static_cast<std::uint16_t>(result);
    }

    void OocFusionState::validate() const
    {
        const auto count = size();
        const auto require = [](bool condition, const char* message)
        {
            if (!condition)
                throw std::invalid_argument(message);
        };
        require(histogram.size() == count * 10, "fusion histogram size mismatch");
        require(neighbors.size() == count * 6, "fusion neighbors size mismatch");
        require(connectivity.size() == count, "fusion connectivity size mismatch");
        require(refinement.size() == count, "fusion refinement size mismatch");
        require(flags.size() == count, "fusion flags size mismatch");
        require(u.size() == count && u_old.size() == count, "fusion scalar size mismatch");
        require(p.size() == count * 3 && v.size() == count * 3 && v_old.size() == count * 3,
                "fusion vector size mismatch");
        require(q.size() == count * 9, "fusion tensor size mismatch");
    }

    void run_ooc_fusion_cpu(OocFusionState& state, const OocFusionParameters& parameters)
    {
        state.validate();
        if (!(parameters.alpha > 0.0F) || !(parameters.beta > 0.0F) || !(parameters.data_weight >= 0.0F))
        {
            throw std::invalid_argument("invalid fusion parameters");
        }
        for (std::size_t iteration = 0; iteration < parameters.iterations; ++iteration)
        {
            if (parameters.is_cancelled && parameters.is_cancelled())
            {
                throw std::runtime_error("Recovered OOC fusion cancelled");
            }
            update_u_v(state, parameters.alpha, parameters.beta, parameters.data_weight);
            update_p_q(state, parameters.alpha, parameters.beta);
        }
    }

    bool run_ooc_fusion_cuda(OocFusionState& state,
                             const OocFusionParameters& parameters,
                             std::size_t device_index,
                             OocFusionCudaStats& stats,
                             std::string& error)
    {
#if defined(METMODEL_HAS_CUDA)
        return run_ooc_fusion_cuda_source_impl(state, parameters, device_index, stats, error);
#else
        (void)state;
        (void)parameters;
        (void)device_index;
        stats = {};
        error = "recovered OOC fusion CUDA path is unavailable in this build";
        return false;
#endif
    }

    bool run_ooc_fusion_cuda_ptx_oracle(OocFusionState& state,
                                        const OocFusionParameters& parameters,
                                        std::size_t device_index,
                                        OocFusionCudaStats& stats,
                                        std::string& error)
    {
#if defined(METMODEL_HAS_CUDA)
        return run_ooc_fusion_cuda_impl(state, parameters, device_index, stats, error);
#else
        (void)state;
        (void)parameters;
        (void)device_index;
        stats = {};
        error = "recovered OOC fusion PTX oracle is unavailable in this build";
        return false;
#endif
    }

    bool run_ooc_fusion_cuda_source(OocFusionState& state,
                                    const OocFusionParameters& parameters,
                                    std::size_t device_index,
                                    OocFusionCudaStats& stats,
                                    std::string& error)
    {
#if defined(METMODEL_HAS_CUDA)
        return run_ooc_fusion_cuda_source_impl(state, parameters, device_index, stats, error);
#else
        (void)state;
        (void)parameters;
        (void)device_index;
        stats = {};
        error = "reconstructed OOC fusion CUDA source path is unavailable in this build";
        return false;
#endif
    }

} // namespace metmodel
