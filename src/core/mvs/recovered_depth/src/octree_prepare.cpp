#include "metmodel/octree_prepare.hpp"

#include "metalign/math.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace metmodel
{
    namespace
    {

        // Read verbatim from Metashape 2.3.2 build 22956 at 0x507e200 and
        // 0x507dea0.  The first table maps a child-local 3x3x3 position to its parent
        // neighbor; the second selects the corresponding child of that neighbor.
        constexpr std::array<std::uint8_t, 216> kOocMarchNeighborIndex{
            0U,  1U,  1U,  3U,  4U,  4U,  3U,  4U,  4U,  9U,  10U, 10U, 12U, 13U, 13U, 12U, 13U, 13U, 9U,  10U,
            10U, 12U, 13U, 13U, 12U, 13U, 13U, 1U,  1U,  2U,  4U,  4U,  5U,  4U,  4U,  5U,  10U, 10U, 11U, 13U,
            13U, 14U, 13U, 13U, 14U, 10U, 10U, 11U, 13U, 13U, 14U, 13U, 13U, 14U, 3U,  4U,  4U,  3U,  4U,  4U,
            6U,  7U,  7U,  12U, 13U, 13U, 12U, 13U, 13U, 15U, 16U, 16U, 12U, 13U, 13U, 12U, 13U, 13U, 15U, 16U,
            16U, 4U,  4U,  5U,  4U,  4U,  5U,  7U,  7U,  8U,  13U, 13U, 14U, 13U, 13U, 14U, 16U, 16U, 17U, 13U,
            13U, 14U, 13U, 13U, 14U, 16U, 16U, 17U, 9U,  10U, 10U, 12U, 13U, 13U, 12U, 13U, 13U, 9U,  10U, 10U,
            12U, 13U, 13U, 12U, 13U, 13U, 18U, 19U, 19U, 21U, 22U, 22U, 21U, 22U, 22U, 10U, 10U, 11U, 13U, 13U,
            14U, 13U, 13U, 14U, 10U, 10U, 11U, 13U, 13U, 14U, 13U, 13U, 14U, 19U, 19U, 20U, 22U, 22U, 23U, 22U,
            22U, 23U, 12U, 13U, 13U, 12U, 13U, 13U, 15U, 16U, 16U, 12U, 13U, 13U, 12U, 13U, 13U, 15U, 16U, 16U,
            21U, 22U, 22U, 21U, 22U, 22U, 24U, 25U, 25U, 13U, 13U, 14U, 13U, 13U, 14U, 16U, 16U, 17U, 13U, 13U,
            14U, 13U, 13U, 14U, 16U, 16U, 17U, 22U, 22U, 23U, 22U, 22U, 23U, 25U, 25U, 26U,
        };

        constexpr std::array<std::uint8_t, 216> kOocMarchChildSlot{
            7U, 6U, 7U, 5U, 4U, 5U, 7U, 6U, 7U, 3U, 2U, 3U, 1U, 0U, 1U, 3U, 2U, 3U, 7U, 6U, 7U, 5U, 4U, 5U, 7U, 6U, 7U,
            6U, 7U, 6U, 4U, 5U, 4U, 6U, 7U, 6U, 2U, 3U, 2U, 0U, 1U, 0U, 2U, 3U, 2U, 6U, 7U, 6U, 4U, 5U, 4U, 6U, 7U, 6U,
            5U, 4U, 5U, 7U, 6U, 7U, 5U, 4U, 5U, 1U, 0U, 1U, 3U, 2U, 3U, 1U, 0U, 1U, 5U, 4U, 5U, 7U, 6U, 7U, 5U, 4U, 5U,
            4U, 5U, 4U, 6U, 7U, 6U, 4U, 5U, 4U, 0U, 1U, 0U, 2U, 3U, 2U, 0U, 1U, 0U, 4U, 5U, 4U, 6U, 7U, 6U, 4U, 5U, 4U,
            3U, 2U, 3U, 1U, 0U, 1U, 3U, 2U, 3U, 7U, 6U, 7U, 5U, 4U, 5U, 7U, 6U, 7U, 3U, 2U, 3U, 1U, 0U, 1U, 3U, 2U, 3U,
            2U, 3U, 2U, 0U, 1U, 0U, 2U, 3U, 2U, 6U, 7U, 6U, 4U, 5U, 4U, 6U, 7U, 6U, 2U, 3U, 2U, 0U, 1U, 0U, 2U, 3U, 2U,
            1U, 0U, 1U, 3U, 2U, 3U, 1U, 0U, 1U, 5U, 4U, 5U, 7U, 6U, 7U, 5U, 4U, 5U, 1U, 0U, 1U, 3U, 2U, 3U, 1U, 0U, 1U,
            0U, 1U, 0U, 2U, 3U, 2U, 0U, 1U, 0U, 4U, 5U, 4U, 6U, 7U, 6U, 4U, 5U, 4U, 0U, 1U, 0U, 2U, 3U, 2U, 0U, 1U, 0U,
        };

        struct CellKey
        {
            std::uint8_t level{};
            std::uint32_t x{};
            std::uint32_t y{};
            std::uint32_t z{};

            friend bool operator==(const CellKey&, const CellKey&) = default;
        };

        struct CellKeyHash
        {
            std::size_t operator()(const CellKey& key) const noexcept
            {
                std::uint64_t value = key.level;
                for (const auto coordinate : {key.x, key.y, key.z})
                {
                    value ^=
                        static_cast<std::uint64_t>(coordinate) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
                }
                // The flat tables below use a power-of-two capacity, unlike
                // libstdc++'s usual prime-sized unordered_map buckets.  Morton-grid
                // coordinates have highly correlated low bits, so the ordinary
                // hash-combine expression alone creates multi-million-element probe
                // clusters.  Apply the SplitMix64 finalizer before masking the bucket.
                value ^= value >> 30U;
                value *= 0xbf58476d1ce4e5b9ULL;
                value ^= value >> 27U;
                value *= 0x94d049bb133111ebULL;
                value ^= value >> 31U;
                return static_cast<std::size_t>(value);
            }
        };

        // The production OOC tables contain millions of cells.  std::unordered_map
        // allocates one heap node per cell and made the recovered South pipeline spend
        // minutes in allocator/system time.  This open-addressed table preserves the
        // exact CellKey lookup semantics while storing keys and values contiguously.
        // No algorithmic ordering depends on bucket iteration in the users below.
        template <class Value> class FlatCellMap
        {
        public:
            explicit FlatCellMap(std::size_t expected)
            {
                if (expected > (std::numeric_limits<std::size_t>::max() - 1U) / 3U * 2U)
                    throw std::length_error("OOC flat cell table capacity overflows");
                const std::size_t target = expected + expected / 2U + 1U;
                std::size_t capacity = 1U;
                while (capacity < target)
                {
                    if (capacity > std::numeric_limits<std::size_t>::max() / 2U)
                        throw std::length_error("OOC flat cell table capacity overflows");
                    capacity *= 2U;
                }
                keys_.resize(capacity);
                values_.resize(capacity);
                occupied_.assign(capacity, 0U);
                mask_ = capacity - 1U;
            }

            std::pair<Value*, bool> emplace(const CellKey& key, const Value& value)
            {
                std::size_t slot = CellKeyHash{}(key)&mask_;
                while (occupied_[slot] != 0U)
                {
                    if (keys_[slot] == key)
                        return {&values_[slot], false};
                    slot = (slot + 1U) & mask_;
                }
                occupied_[slot] = 1U;
                keys_[slot] = key;
                values_[slot] = value;
                return {&values_[slot], true};
            }

            [[nodiscard]] Value* find(const CellKey& key) noexcept
            {
                return const_cast<Value*>(std::as_const(*this).find(key));
            }

            [[nodiscard]] const Value* find(const CellKey& key) const noexcept
            {
                std::size_t slot = CellKeyHash{}(key)&mask_;
                while (occupied_[slot] != 0U)
                {
                    if (keys_[slot] == key)
                        return &values_[slot];
                    slot = (slot + 1U) & mask_;
                }
                return nullptr;
            }

            [[nodiscard]] bool contains(const CellKey& key) const noexcept
            {
                return find(key) != nullptr;
            }

        private:
            std::vector<CellKey> keys_;
            std::vector<Value> values_;
            std::vector<std::uint8_t> occupied_;
            std::size_t mask_{};
        };

        [[nodiscard]] unsigned morton_bit(const std::array<std::uint32_t, 3>& words, unsigned position) noexcept
        {
            if (position >= 64U)
            {
                return (words[0] >> (position - 64U)) & 1U;
            }
            if (position >= 32U)
            {
                return (words[1] >> (position - 32U)) & 1U;
            }
            return (words[2] >> position) & 1U;
        }

        [[nodiscard]] CellKey decode_cell(const std::array<std::uint32_t, 3>& words, std::uint8_t level)
        {
            if (level > 32U)
            {
                throw std::runtime_error("OOC octree level exceeds 96-bit Morton capacity");
            }
            CellKey result{level, 0U, 0U, 0U};
            for (unsigned depth = 0; depth < level; ++depth)
            {
                const unsigned top = 95U - 3U * depth;
                const unsigned code =
                    (morton_bit(words, top) << 2U) | (morton_bit(words, top - 1U) << 1U) | morton_bit(words, top - 2U);
                result.x = (result.x << 1U) | ((code >> 2U) & 1U);
                result.y = (result.y << 1U) | ((code >> 1U) & 1U);
                result.z = (result.z << 1U) | (code & 1U);
            }
            return result;
        }

        [[nodiscard]] CellKey decode_cell(const OocOctreeRecord& record)
        {
            return decode_cell(record.morton_words, record.level);
        }

        [[nodiscard]] CellKey decode_cell(const OocMarchingNode& record)
        {
            return decode_cell(record.morton_words, record.level);
        }

        [[nodiscard]] std::array<std::uint32_t, 3> encode_cell_words(const CellKey& cell)
        {
            if (cell.level > 32U)
            {
                throw std::runtime_error("OOC octree level exceeds 96-bit Morton capacity");
            }
            std::array<std::uint32_t, 3> words{};
            for (unsigned depth = 0; depth < cell.level; ++depth)
            {
                const unsigned coordinate_bit = static_cast<unsigned>(cell.level) - depth - 1U;
                const unsigned top = 95U - 3U * depth;
                for (unsigned axis = 0; axis < 3U; ++axis)
                {
                    const std::uint32_t coordinate = axis == 0U ? cell.x : (axis == 1U ? cell.y : cell.z);
                    if (((coordinate >> coordinate_bit) & 1U) == 0U)
                        continue;
                    const unsigned position = top - axis;
                    if (position >= 64U)
                    {
                        words[0] |= 1U << (position - 64U);
                    }
                    else if (position >= 32U)
                    {
                        words[1] |= 1U << (position - 32U);
                    }
                    else
                    {
                        words[2] |= 1U << position;
                    }
                }
            }
            return words;
        }

        [[nodiscard]] CellKey child_cell(const CellKey& parent, unsigned child)
        {
            return {static_cast<std::uint8_t>(parent.level + 1U),
                    static_cast<std::uint32_t>((parent.x << 1U) | ((child >> 2U) & 1U)),
                    static_cast<std::uint32_t>((parent.y << 1U) | ((child >> 1U) & 1U)),
                    static_cast<std::uint32_t>((parent.z << 1U) | (child & 1U))};
        }

        [[nodiscard]] std::array<double, 4> transform_double4(const std::array<double, 16>& matrix,
                                                              const std::array<float, 3>& point) noexcept
        {
            std::array<double, 4> result{};
            for (std::size_t row = 0; row != 4U; ++row)
            {
                double value = matrix[4U * row] * static_cast<double>(point[0]);
                value = value + matrix[4U * row + 1U] * static_cast<double>(point[1]);
                value = value + matrix[4U * row + 2U] * static_cast<double>(point[2]);
                value = value + matrix[4U * row + 3U];
                result[row] = value;
            }
            return result;
        }

        [[nodiscard]] std::array<double, 3> unproject_mode0(float projected_x,
                                                            float projected_y,
                                                            float projected_depth,
                                                            const OocSampleScaleCameraMode0& camera) noexcept
        {
            double normalized_y = static_cast<double>(projected_y) - static_cast<double>(camera.height) * 0.5;
            normalized_y = normalized_y - camera.principal_y;
            normalized_y = normalized_y / camera.focal_length;
            double normalized_x = static_cast<double>(projected_x) - static_cast<double>(camera.width) * 0.5;
            normalized_x = normalized_x - camera.principal_x;
            normalized_x = normalized_x - camera.shear * normalized_y;
            normalized_x = normalized_x / (camera.focal_length + camera.additive_focal);
            const double depth = static_cast<double>(projected_depth);
            return {normalized_x * depth, normalized_y * depth, depth};
        }

        [[nodiscard]] float squared_world_distance(const std::array<double, 3>& camera_point,
                                                   const std::array<float, 16>& matrix,
                                                   const std::array<float, 3>& original) noexcept
        {
            const std::array<float, 3> point{
                static_cast<float>(camera_point[0]),
                static_cast<float>(camera_point[1]),
                static_cast<float>(camera_point[2]),
            };
            std::array<float, 4> transformed{};
            for (std::size_t row = 0; row != 4U; ++row)
            {
                float value = matrix[4U * row] * point[0];
                value = value + matrix[4U * row + 1U] * point[1];
                value = value + matrix[4U * row + 2U] * point[2];
                value = value + matrix[4U * row + 3U];
                transformed[row] = value;
            }
            float dx = transformed[0] / transformed[3] - original[0];
            dx = dx * dx;
            float dy = transformed[1] / transformed[3] - original[1];
            dy = dy * dy;
            float dz = transformed[2] / transformed[3] - original[2];
            dz = dz * dz;
            float result = dx + dy;
            result = result + dz;
            return result;
        }

        [[nodiscard]] float ooc_multiply_f32(float left, float right) noexcept
        {
            volatile float result = left * right;
            return result;
        }

        [[nodiscard]] float ooc_add_f32(float left, float right) noexcept
        {
            volatile float result = left + right;
            return result;
        }

        [[nodiscard]] double ooc_multiply_f64(double left, double right) noexcept
        {
            volatile double result = left * right;
            return result;
        }

        [[nodiscard]] double ooc_add_f64(double left, double right) noexcept
        {
            volatile double result = left + right;
            return result;
        }

        [[nodiscard]] double ooc_subtract_f64(double left, double right) noexcept
        {
            volatile double result = left - right;
            return result;
        }

        [[nodiscard]] double ooc_divide_f64(double left, double right) noexcept
        {
            volatile double result = left / right;
            return result;
        }

        [[nodiscard]] float ooc_divide_f32(float numerator, float denominator) noexcept
        {
            volatile float result = numerator / denominator;
            return result;
        }

    } // namespace

#if defined(METMODEL_HAS_CUDA)
    bool run_recovered_ooc_histogram_cuda_chain_impl(const OocHistogramCudaChainInput& input,
                                                     OocHistogramCudaChainOutput& output,
                                                     std::string& error,
                                                     bool use_source_kernel);
#endif

    OocHistogramCudaCameraParameters
    make_ooc_histogram_cuda_camera_parameters_mode0(const OocHistogramProjectionInput& input)
    {
        if (input.camera.width <= 0 || input.camera.height <= 0 ||
            input.camera.width > std::numeric_limits<std::uint32_t>::max() ||
            input.camera.height > std::numeric_limits<std::uint32_t>::max() ||
            !std::isfinite(input.camera.focal_length) || !std::isfinite(input.camera.principal_x) ||
            !std::isfinite(input.camera.principal_y) || !std::isfinite(input.camera.additive_focal) ||
            !std::isfinite(input.camera.shear))
        {
            throw std::invalid_argument("OOC CUDA histogram packer requires a finite ordinary camera");
        }
        const auto store = [](auto& bytes, std::size_t offset, const auto& value)
        {
            if (offset + sizeof(value) > bytes.size())
                throw std::logic_error("OOC CUDA calibration store is out of range");
            std::memcpy(bytes.data() + offset, &value, sizeof(value));
        };

        OocHistogramCudaCameraParameters result;
        store(result.calibration.bytes, 32U, 1.0e9F);
        store(result.calibration.bytes, 64U, std::numeric_limits<float>::infinity());
        store(result.calibration.bytes, 68U, std::numeric_limits<float>::infinity());
        store(result.calibration.bytes, 72U, -std::numeric_limits<float>::infinity());
        store(result.calibration.bytes, 76U, -std::numeric_limits<float>::infinity());
        store(result.calibration.bytes, 80U, std::uint32_t{1});
        store(result.calibration.bytes, 84U, static_cast<std::uint32_t>(input.camera.width));
        store(result.calibration.bytes, 88U, static_cast<std::uint32_t>(input.camera.height));
        store(result.calibration.bytes, 92U, static_cast<float>(input.camera.focal_length));
        store(result.calibration.bytes, 96U, static_cast<float>(input.camera.principal_x));
        store(result.calibration.bytes, 100U, static_cast<float>(input.camera.principal_y));
        store(result.calibration.bytes, 104U, static_cast<float>(input.camera.additive_focal));
        store(result.calibration.bytes, 108U, static_cast<float>(input.camera.shear));
        store(result.calibration.bytes, 112U, std::uint8_t{1});
        store(result.calibration.bytes, 113U, std::uint8_t{0});

        result.before_rotation.values = {
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
        };
        result.after_rotation = result.before_rotation;
        const auto& matrix = input.camera_to_world;
        result.exterior_transform.values = {
            matrix[0],
            matrix[1],
            matrix[2],
            0.0F,
            matrix[4],
            matrix[5],
            matrix[6],
            0.0F,
            matrix[8],
            matrix[9],
            matrix[10],
            0.0F,
            matrix[3],
            matrix[7],
            matrix[11],
            matrix[15],
        };
        return result;
    }

    OocPyramidReductionOutput reduce_ooc_pyramid_level(const OocPyramidReductionInput& input)
    {
        if (input.width <= 1U || input.height <= 1U ||
            input.width > std::numeric_limits<std::size_t>::max() / input.height)
        {
            throw std::invalid_argument("OOC pyramid reduction requires a finite two-dimensional source");
        }
        const std::size_t source_size = static_cast<std::size_t>(input.width * input.height);
        if (input.depth.size() != source_size || input.temporary.size() != source_size ||
            input.sample_scale.size() != source_size)
        {
            throw std::invalid_argument("OOC pyramid reduction source planes have different sizes");
        }

        OocPyramidReductionOutput result;
        result.width = input.width / 2U;
        result.height = input.height / 2U;
        if (result.width > std::numeric_limits<std::size_t>::max() / result.height)
        {
            throw std::length_error("OOC pyramid reduction output is too large");
        }
        const std::size_t output_size = static_cast<std::size_t>(result.width * result.height);
        if (!input.output_gate.empty() && input.output_gate.size() != output_size)
        {
            throw std::invalid_argument("OOC pyramid reduction gate has the wrong size");
        }
        result.depth.assign(output_size, 0.0F);
        result.temporary.assign(output_size, 0.0F);
        result.sample_scale.assign(output_size, 0.0F);
        result.temporary_u8.assign(output_size, 0U);

        constexpr std::uint32_t invalid_depth_bits = 0xD3800000U;
        constexpr float invalid_depth = std::bit_cast<float>(invalid_depth_bits);
        for (std::uint64_t output_y = 0U; output_y != result.height; ++output_y)
        {
            for (std::uint64_t output_x = 0U; output_x != result.width; ++output_x)
            {
                const std::size_t output_index = static_cast<std::size_t>(output_y * result.width + output_x);
                if (!input.output_gate.empty() && input.output_gate[output_index] == 0U)
                {
                    if (input.special_invalid_depth)
                    {
                        result.depth[output_index] = invalid_depth;
                    }
                    continue;
                }

                float sum_weight = 0.0F;
                float sum_depth = 0.0F;
                float sum_scale = 0.0F;
                for (std::uint64_t source_y = output_y * 2U; source_y != output_y * 2U + 2U; ++source_y)
                {
                    for (std::uint64_t source_x = output_x * 2U; source_x != output_x * 2U + 2U; ++source_x)
                    {
                        const std::size_t source_index = static_cast<std::size_t>(source_y * input.width + source_x);
                        const float depth = input.depth[source_index];
                        if (depth == 0.0F || std::bit_cast<std::uint32_t>(depth) == invalid_depth_bits)
                        {
                            continue;
                        }
                        const float weight = input.temporary[source_index];
                        const float weighted_depth = ooc_multiply_f32(depth, weight);
                        sum_weight = ooc_add_f32(sum_weight, weight);
                        const float weighted_scale = ooc_multiply_f32(weight, input.sample_scale[source_index]);
                        sum_depth = ooc_add_f32(sum_depth, weighted_depth);
                        sum_scale = ooc_add_f32(sum_scale, weighted_scale);
                    }
                }
                if (!(sum_weight > 0.0F))
                    continue;
                result.depth[output_index] = ooc_divide_f32(sum_depth, sum_weight);
                result.temporary[output_index] = sum_weight;
                const float average_scale = ooc_divide_f32(sum_scale, sum_weight);
                result.sample_scale[output_index] = ooc_add_f32(average_scale, average_scale);
                // CVTTSS2SI r64 returns the integer-indefinite value on overflow;
                // the target then compares only EAX as an unsigned value.
                std::uint32_t truncated_low = 0U;
                if (std::isfinite(sum_weight) && static_cast<double>(sum_weight) < 0x1p63)
                {
                    truncated_low = static_cast<std::uint32_t>(static_cast<std::int64_t>(sum_weight));
                }
                result.temporary_u8[output_index] =
                    static_cast<std::uint8_t>(truncated_low < 255U ? truncated_low : 255U);
            }
        }
        return result;
    }

    bool run_recovered_ooc_histogram_cuda_chain(const OocHistogramCudaChainInput& input,
                                                OocHistogramCudaChainOutput& output,
                                                std::string& error)
    {
#if defined(METMODEL_HAS_CUDA)
        return run_recovered_ooc_histogram_cuda_chain_impl(input, output, error, true);
#else
        (void)input;
        output = {};
        error = "recovered OOC histogram CUDA chain is unavailable in this build";
        return false;
#endif
    }

    bool run_recovered_ooc_histogram_cuda_chain_ptx_oracle(const OocHistogramCudaChainInput& input,
                                                           OocHistogramCudaChainOutput& output,
                                                           std::string& error)
    {
#if defined(METMODEL_HAS_CUDA)
        return run_recovered_ooc_histogram_cuda_chain_impl(input, output, error, false);
#else
        (void)input;
        output = {};
        error = "recovered OOC histogram CUDA chain is unavailable in this build";
        return false;
#endif
    }

    namespace
    {

        struct OocHistogramPointIndexNode
        {
            std::array<double, 3> minimum{std::numeric_limits<double>::max(),
                                          std::numeric_limits<double>::max(),
                                          std::numeric_limits<double>::max()};
            std::array<double, 3> maximum{-std::numeric_limits<double>::max(),
                                          -std::numeric_limits<double>::max(),
                                          -std::numeric_limits<double>::max()};
            std::uint32_t point_offset{};
            std::uint32_t point_count{};
            // Zero denotes a leaf. Children are allocated consecutively.
            std::uint32_t first_child{};
        };

        struct OocHistogramNeighbor
        {
            double squared_distance{};
            std::uint32_t point_index{};
        };

        // sub_41026F0/sub_13A06A0 maintain a max heap ordered by distance and then
        // point index.  The secondary key is observable for the many equidistant
        // centres in a balanced octree.
        struct OocHistogramNeighborLess
        {
            bool operator()(const OocHistogramNeighbor& left, const OocHistogramNeighbor& right) const noexcept
            {
                if (left.squared_distance != right.squared_distance)
                {
                    return left.squared_distance < right.squared_distance;
                }
                return left.point_index < right.point_index;
            }
        };

        struct OocHistogramPendingNode
        {
            double squared_distance{};
            std::uint32_t node_index{};
        };

        // The target stores negative box distances in a max heap.  Expressing the
        // same order directly gives minimum distance first and, on equality, the
        // larger node index first.
        struct OocHistogramPendingNodeGreater
        {
            bool operator()(const OocHistogramPendingNode& left, const OocHistogramPendingNode& right) const noexcept
            {
                if (left.squared_distance != right.squared_distance)
                {
                    return left.squared_distance > right.squared_distance;
                }
                return left.node_index < right.node_index;
            }
        };

        class OocHistogramPointIndex
        {
        public:
            explicit OocHistogramPointIndex(const std::vector<std::array<double, 3>>& points) : points_(points)
            {
                if (points.empty() || points.size() > std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::invalid_argument("OOC histogram point index requires 1..UINT32_MAX points");
                }
                point_indices_.resize(points.size());
                for (std::size_t index = 0; index != points.size(); ++index)
                {
                    point_indices_[index] = static_cast<std::uint32_t>(index);
                }
                nodes_.reserve(points.size() / 8U + 1U);
                make_node(0U, static_cast<std::uint32_t>(points.size()));
                struct PendingBuild
                {
                    std::uint32_t node{};
                    std::uint32_t depth{};
                };
                std::vector<PendingBuild> pending{{0U, 0U}};
                while (!pending.empty())
                {
                    const PendingBuild current = pending.back();
                    pending.pop_back();
                    auto& node = nodes_[current.node];
                    if (current.depth >= 64U || node.point_count <= 16U)
                        continue;

                    std::size_t axis = node.maximum[1] - node.minimum[1] > node.maximum[0] - node.minimum[0] ? 1U : 0U;
                    if (node.maximum[2] - node.minimum[2] > node.maximum[axis] - node.minimum[axis])
                    {
                        axis = 2U;
                    }
                    const double extent = node.maximum[axis] - node.minimum[axis];
                    if (!(extent > 0.0))
                        continue;
                    const double middle = (node.maximum[axis] + node.minimum[axis]) * 0.5;
                    const std::uint32_t begin = node.point_offset;
                    const std::uint32_t end = begin + node.point_count;
                    std::uint32_t left = begin;
                    std::uint32_t right = end - 1U;
                    while (left <= right)
                    {
                        while (left < end && points_[point_indices_[left]][axis] < middle)
                        {
                            ++left;
                        }
                        while (right >= begin && !(points_[point_indices_[right]][axis] < middle))
                        {
                            if (right == 0U)
                            {
                                right = std::numeric_limits<std::uint32_t>::max();
                                break;
                            }
                            --right;
                        }
                        if (right == std::numeric_limits<std::uint32_t>::max() || right < left)
                        {
                            break;
                        }
                        std::swap(point_indices_[left], point_indices_[right]);
                        ++left;
                        if (right == 0U)
                            break;
                        --right;
                    }
                    const std::uint32_t split = left;
                    if (split <= begin || split >= end)
                        continue;

                    // A vector reallocation would invalidate node, so publish the
                    // child index before making either child.
                    const std::uint32_t first_child = static_cast<std::uint32_t>(nodes_.size());
                    nodes_[current.node].first_child = first_child;
                    make_node(begin, split - begin);
                    make_node(split, end - split);
                    const std::uint32_t child_depth = current.depth + 1U;
                    if (child_depth < 64U)
                    {
                        if (split - begin > 16U)
                        {
                            pending.push_back({first_child, child_depth});
                        }
                        if (end - split > 16U)
                        {
                            pending.push_back({first_child + 1U, child_depth});
                        }
                    }
                }
            }

            struct SevenNearest
            {
                std::array<std::uint32_t, 7> indices{};
                std::size_t count{};
                float maximum_distance{};
            };

            [[nodiscard]] SevenNearest seven_nearest(const std::array<double, 3>& query) const
            {
                std::priority_queue<OocHistogramPendingNode,
                                    std::vector<OocHistogramPendingNode>,
                                    OocHistogramPendingNodeGreater>
                    pending;
                std::priority_queue<OocHistogramNeighbor, std::vector<OocHistogramNeighbor>, OocHistogramNeighborLess>
                    result;
                pending.push({box_squared_distance(query, nodes_.front()), 0U});
                double threshold = std::numeric_limits<double>::max();
                while (!pending.empty())
                {
                    OocHistogramPendingNode next = pending.top();
                    pending.pop();
                    if (next.squared_distance >= threshold)
                        break;
                    std::uint32_t node_index = next.node_index;
                    while (nodes_[node_index].first_child != 0U)
                    {
                        const std::uint32_t left = nodes_[node_index].first_child;
                        const std::uint32_t right = left + 1U;
                        const double left_distance = box_squared_distance(query, nodes_[left]);
                        const double right_distance = box_squared_distance(query, nodes_[right]);
                        std::uint32_t near = left;
                        std::uint32_t far = right;
                        double far_distance = right_distance;
                        if (right_distance <= left_distance)
                        {
                            near = right;
                            far = left;
                            far_distance = left_distance;
                        }
                        if (far_distance < threshold)
                        {
                            pending.push({far_distance, far});
                        }
                        node_index = near;
                    }
                    const auto& node = nodes_[node_index];
                    if (!(box_squared_distance(query, node) < threshold))
                        continue;
                    const std::uint32_t end = node.point_offset + node.point_count;
                    for (std::uint32_t offset = node.point_offset; offset != end; ++offset)
                    {
                        const std::uint32_t point_index = point_indices_[offset];
                        const double distance = point_squared_distance(query, points_[point_index]);
                        if (!(distance < threshold))
                            continue;
                        result.push({distance, point_index});
                        if (result.size() > 7U)
                            result.pop();
                        if (result.size() >= 7U)
                        {
                            threshold = result.top().squared_distance;
                        }
                    }
                }

                SevenNearest output;
                float maximum_squared = 0.0F;
                while (!result.empty())
                {
                    const OocHistogramNeighbor value = result.top();
                    result.pop();
                    output.indices[output.count++] = value.point_index;
                    const float squared = static_cast<float>(value.squared_distance);
                    maximum_squared = std::max(maximum_squared, squared);
                }
                output.maximum_distance = std::sqrt(maximum_squared);
                return output;
            }

            template <class Visitor>
            void visit_radius(const std::array<double, 3>& query, float radius, Visitor&& visitor) const
            {
                const double radius_double = static_cast<double>(radius);
                const double squared_radius = radius_double * radius_double;
                std::vector<std::uint32_t> pending{0U};
                while (!pending.empty())
                {
                    const std::uint32_t node_index = pending.back();
                    pending.pop_back();
                    const auto& node = nodes_[node_index];
                    if (!(box_squared_distance(query, node) < squared_radius))
                        continue;
                    if (node.first_child != 0U)
                    {
                        pending.push_back(node.first_child);
                        pending.push_back(node.first_child + 1U);
                        continue;
                    }
                    const std::uint32_t end = node.point_offset + node.point_count;
                    for (std::uint32_t offset = node.point_offset; offset != end; ++offset)
                    {
                        const std::uint32_t point_index = point_indices_[offset];
                        if (point_squared_distance(query, points_[point_index]) < squared_radius)
                        {
                            visitor(point_index);
                        }
                    }
                }
            }

        private:
            std::uint32_t make_node(std::uint32_t offset, std::uint32_t count)
            {
                OocHistogramPointIndexNode node;
                node.point_offset = offset;
                node.point_count = count;
                const std::uint32_t end = offset + count;
                for (std::uint32_t current = offset; current != end; ++current)
                {
                    const auto& point = points_[point_indices_[current]];
                    for (std::size_t axis = 0U; axis != 3U; ++axis)
                    {
                        node.minimum[axis] = std::fmin(point[axis], node.minimum[axis]);
                        node.maximum[axis] = std::fmax(point[axis], node.maximum[axis]);
                    }
                }
                nodes_.push_back(node);
                return static_cast<std::uint32_t>(nodes_.size() - 1U);
            }

            [[nodiscard]] static double box_squared_distance(const std::array<double, 3>& point,
                                                             const OocHistogramPointIndexNode& node) noexcept
            {
                double result = 0.0;
                for (std::size_t axis = 0U; axis != 3U; ++axis)
                {
                    double delta = point[axis] - node.minimum[axis];
                    if (delta >= 0.0)
                    {
                        delta = node.maximum[axis] - point[axis];
                        if (delta >= 0.0)
                            continue;
                    }
                    result = result + delta * delta;
                }
                return result;
            }

            [[nodiscard]] static double point_squared_distance(const std::array<double, 3>& left,
                                                               const std::array<double, 3>& right) noexcept
            {
                const double dx = left[0] - right[0];
                const double dy = left[1] - right[1];
                const double dz = left[2] - right[2];
                double result = dx * dx;
                result = result + dy * dy;
                result = result + dz * dz;
                return result;
            }

            const std::vector<std::array<double, 3>>& points_;
            std::vector<std::uint32_t> point_indices_;
            std::vector<OocHistogramPointIndexNode> nodes_;
        };

        [[nodiscard]] float ooc_histogram_cell_half(float root_scale, std::uint8_t level) noexcept
        {
            const float half_root = ooc_multiply_f32(0.5F, root_scale);
            const float inverse = level == 32U ? std::bit_cast<float>(0x2F800000U)
                                               : ooc_divide_f32(1.0F, static_cast<float>(std::uint64_t{1} << level));
            return ooc_multiply_f32(inverse, half_root);
        }

        [[nodiscard]] float ooc_histogram_node_radius(const OocWeightedNodeRecord& record, float root_scale) noexcept
        {
            const float cell_half = ooc_histogram_cell_half(root_scale, record.level);
            float factor =
                ooc_divide_f32(static_cast<float>(normalized_ooc_weight(record)), std::bit_cast<float>(0x437FE666U));
            factor = ooc_multiply_f32(factor, 0.75F);
            factor = ooc_add_f32(factor, 0.75F);
            return ooc_multiply_f32(cell_half, factor);
        }

    } // namespace

    std::vector<OocHistogramVoxel> build_ooc_histogram_voxels_mode0(const OocHistogramVoxelBuildInput& input)
    {
        if (input.balanced_records.empty() ||
            input.balanced_records.size() > std::numeric_limits<std::uint32_t>::max() - 1ULL ||
            !std::isfinite(input.root_scale) || !(input.root_scale > 0.0F))
        {
            throw std::invalid_argument("invalid balanced-node table for OOC histogram voxels");
        }
        for (std::size_t axis = 0U; axis != 3U; ++axis)
        {
            if (!std::isfinite(input.region_center[axis]) || !std::isfinite(input.region_size[axis]) ||
                !(input.region_size[axis] > 0.0))
            {
                throw std::invalid_argument("invalid OOC histogram reconstruction region");
            }
        }
        for (const double value : input.region_rotation)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("invalid OOC histogram reconstruction rotation");
            }
        }

        std::vector<OocWeightedNodeRecord> ordered(input.balanced_records.begin(), input.balanced_records.end());
        std::sort(ordered.begin(),
                  ordered.end(),
                  [](const OocWeightedNodeRecord& left, const OocWeightedNodeRecord& right)
                  {
                      if (left.level != right.level)
                          return left.level < right.level;
                      return left.morton_words < right.morton_words;
                  });

        std::vector<std::array<double, 3>> positions;
        positions.reserve(ordered.size());
        std::vector<float> radii;
        radii.reserve(ordered.size());
        std::vector<OocHistogramVoxel> output(ordered.size() + 1U);
        for (std::size_t index = 0U; index != ordered.size(); ++index)
        {
            const auto& source = ordered[index];
            const CellKey cell = decode_cell(source.morton_words, source.level);
            const double denominator = std::ldexp(1.0, static_cast<int>(source.level));
            const std::array<double, 3> local{
                ((static_cast<double>(cell.x) + 0.5) / denominator) * static_cast<double>(input.root_scale) -
                    input.region_size[0] * 0.5,
                ((static_cast<double>(cell.y) + 0.5) / denominator) * static_cast<double>(input.root_scale) -
                    input.region_size[1] * 0.5,
                ((static_cast<double>(cell.z) + 0.5) / denominator) * static_cast<double>(input.root_scale) -
                    input.region_size[2] * 0.5,
            };
            std::array<double, 3> position{};
            for (std::size_t axis = 0U; axis != 3U; ++axis)
            {
                const std::size_t row = axis * 3U;
                double value = local[0] * input.region_rotation[row];
                value = value + local[1] * input.region_rotation[row + 1U];
                value = value + local[2] * input.region_rotation[row + 2U];
                value = value + input.region_center[axis];
                position[axis] = value;
            }
            positions.push_back(position);
            radii.push_back(ooc_histogram_node_radius(source, input.root_scale));

            auto& target = output[index + 1U];
            for (std::size_t axis = 0U; axis != 3U; ++axis)
            {
                target.position[axis] = static_cast<float>(position[axis]);
            }
            target.scalar_lut_index = float_to_half(1.0F);
            target.metadata[2] = source.level;
            target.metadata[3] = normalized_ooc_weight(source);
        }

        // The extra leading record is created by 0x17CF84D..0x17CF950 and uses
        // the root centre with a full level-selection vote.
        output.front().position = output[1].position;
        output.front().scalar_lut_index = float_to_half(1.0F);
        output.front().metadata[2] = 0U;
        output.front().metadata[3] = 255U;

        const OocHistogramPointIndex point_index(positions);
        if (ordered.size() < 7U)
        {
            throw std::runtime_error("OOC histogram seven-neighbour search requires seven nodes");
        }
        std::atomic<bool> incomplete_nearest{false};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(ordered.size());
             ++signed_index)
        {
            const std::size_t index = static_cast<std::size_t>(signed_index);
            const auto nearest = point_index.seven_nearest(positions[index]);
            if (nearest.count != 7U)
            {
                incomplete_nearest.store(true, std::memory_order_relaxed);
                continue;
            }
            std::uint8_t maximum_denominator = 0U;
            for (std::size_t neighbor = 0U; neighbor != nearest.count; ++neighbor)
            {
                maximum_denominator =
                    std::max(maximum_denominator, ordered[nearest.indices[neighbor]].weight_denominator);
            }

            const float three_cells =
                ooc_multiply_f32(ooc_histogram_cell_half(input.root_scale, ordered[index].level), 3.0F);
            const float search_radius = ooc_multiply_f32(std::fmin(nearest.maximum_distance, three_cells), 1.1F);
            float minimum_radius = radii[index];
            point_index.visit_radius(positions[index],
                                     search_radius,
                                     [&](std::uint32_t neighbor)
                                     {
                                         if (neighbor != index)
                                         {
                                             minimum_radius = std::fmin(radii[neighbor], minimum_radius);
                                         }
                                     });

            float coefficient = 1.0F;
            if (maximum_denominator != 0U)
            {
                coefficient = ooc_divide_f32(minimum_radius, radii[index]);
                coefficient = ooc_multiply_f32(coefficient, std::bit_cast<float>(0x3F555555U));
            }
            output[index + 1U].scalar_lut_index = float_to_half(coefficient);
        }
        if (incomplete_nearest.load(std::memory_order_relaxed))
        {
            throw std::runtime_error("OOC histogram seven-neighbour search is incomplete");
        }
        return output;
    }

    std::array<std::vector<OocHistogramVoxel>, 4>
    partition_ooc_histogram_voxels_four_way(std::span<const OocHistogramVoxel> voxels)
    {
        std::array<std::vector<OocHistogramVoxel>, 4> result;
        const std::size_t chunk = voxels.size() / 4U + static_cast<std::size_t>(voxels.size() % 4U != 0U);
        std::size_t begin = 0U;
        for (std::size_t partition = 0; partition != result.size(); ++partition)
        {
            const std::size_t end = begin + std::min(chunk, voxels.size() - begin);
            result[partition].assign(voxels.begin() + static_cast<std::ptrdiff_t>(begin),
                                     voxels.begin() + static_cast<std::ptrdiff_t>(end));
            begin = end;
        }
        return result;
    }

    std::uint32_t select_ooc_pyramid_maximum_level(const std::array<std::uint64_t, 33>& level_counts) noexcept
    {
        std::uint64_t total = 0U;
        for (const std::uint64_t count : level_counts)
        {
            total += count;
        }

        const float threshold = static_cast<float>(total) * std::bit_cast<float>(0x3DCCCCD0U);
        std::uint64_t tail = level_counts[32];
        if (threshold <= static_cast<float>(tail))
        {
            return 32U;
        }

        std::uint32_t selected = 31U;
        for (std::uint32_t level = 31U; level > 1U; --level)
        {
            tail += level_counts[level];
            if (threshold <= static_cast<float>(tail))
            {
                return selected;
            }
            selected = level - 1U;
        }
        return selected;
    }

    OocPyramidRegistryLevelMetadata
    make_ooc_pyramid_registry_level_metadata(std::span<const float> depth,
                                             std::span<const std::uint8_t> level_codes,
                                             std::span<const std::uint8_t> histogram_codes)
    {
        if (depth.size() != level_codes.size() || depth.size() != histogram_codes.size())
        {
            throw std::invalid_argument("OOC pyramid registry metadata images have different sizes");
        }

        constexpr float invalid_depth = std::bit_cast<float>(0xD3800000U);
        std::array<std::uint64_t, 33> level_counts{};
        for (std::size_t index = 0; index != depth.size(); ++index)
        {
            if (depth[index] == invalid_depth || depth[index] == 0.0F)
            {
                continue;
            }
            const std::uint8_t level = level_codes[index];
            if (level >= level_counts.size())
            {
                throw std::invalid_argument("OOC pyramid registry level code exceeds 32");
            }
            ++level_counts[level];
        }

        OocPyramidRegistryLevelMetadata result;
        result.level_counts = level_counts;
        result.maximum_level = select_ooc_pyramid_maximum_level(level_counts);
        std::array<std::uint64_t, 32> selected_counts{};
        for (std::size_t index = 0; index != depth.size(); ++index)
        {
            if (depth[index] == 0.0F || depth[index] == invalid_depth || level_codes[index] != result.maximum_level)
            {
                continue;
            }
            const std::uint8_t code = histogram_codes[index];
            if (code >= selected_counts.size())
            {
                throw std::invalid_argument("OOC pyramid registry histogram code exceeds 31");
            }
            ++selected_counts[code];
        }
        for (std::size_t index = 0; index != selected_counts.size(); ++index)
        {
            result.selected_level_histogram[index] = static_cast<float>(selected_counts[index]);
        }
        return result;
    }

    OocAdaptiveRootMode0 derive_ooc_adaptive_root_mode0(std::span<const OocPyramidRegistryLevelMetadata> metadata,
                                                        const std::array<double, 3>& region_size)
    {
        if (metadata.empty())
        {
            throw std::invalid_argument("OOC adaptive-root reduction requires registry metadata");
        }

        OocAdaptiveRootMode0 result;
        for (const auto& camera : metadata)
        {
            if (camera.maximum_level >= result.level_counts.size())
            {
                throw std::invalid_argument("OOC adaptive-root camera maximum level exceeds 32");
            }
            // The stats file contributes only the 32-bin histogram attached to
            // this camera's selected +0x438 level.  It does not contribute the
            // full per-pixel level-code distribution used one stage earlier to
            // select that per-camera level.
            for (const float count : camera.selected_level_histogram)
            {
                if (!std::isfinite(count) || count < 0.0F)
                {
                    throw std::invalid_argument("OOC adaptive-root histogram count is invalid");
                }
                result.level_counts[camera.maximum_level] += static_cast<double>(count);
                const auto integral_count = static_cast<std::uint64_t>(count);
                if (integral_count > std::numeric_limits<std::uint64_t>::max() - result.total_samples)
                {
                    throw std::overflow_error("OOC adaptive-root sample count overflows");
                }
                result.total_samples += integral_count;
            }
        }
        if (result.total_samples == 0U)
        {
            throw std::invalid_argument("OOC adaptive-root reduction has no valid samples");
        }

        // 0x178E6B3..0x178E71D: accumulate level bins from 32 downward until
        // reaching total * 0.1000000238418579, never descending below level 1.
        constexpr double upper_tail_fraction = 0.1000000238418579;
        const double level_threshold = static_cast<double>(result.total_samples) * upper_tail_fraction;
        double level_tail = 0.0;
        std::uint32_t selected_level = 32U;
        for (;;)
        {
            level_tail += result.level_counts[selected_level];
            if (level_threshold <= level_tail || selected_level == 1U)
                break;
            --selected_level;
        }
        result.maximum_level = selected_level;

        for (const auto& camera : metadata)
        {
            if (camera.maximum_level != selected_level)
                continue;
            for (std::size_t bin = 0U; bin != result.selected_level_histogram.size(); ++bin)
            {
                result.selected_level_histogram[bin] += static_cast<double>(camera.selected_level_histogram[bin]);
            }
        }

        double histogram_total = 0.0;
        for (const double count : result.selected_level_histogram)
            histogram_total += count;
        constexpr double lower_tail_fraction = 0.300000011920929;
        const double histogram_threshold = histogram_total * lower_tail_fraction;
        double histogram_prefix = 0.0;
        std::uint32_t winning_bin = 0U;
        bool selected = false;
        for (; winning_bin != 31U; ++winning_bin)
        {
            histogram_prefix += result.selected_level_histogram[winning_bin];
            if (histogram_threshold <= histogram_prefix)
            {
                selected = true;
                break;
            }
        }
        result.winning_bin = winning_bin;
        result.scale_factor =
            selected ? (((static_cast<double>(winning_bin) + 0.5) * 0.03125 + 1.0) / 1.015625) : 1.953846153846154;

        double maximum_extent = region_size[0];
        if (region_size[1] > maximum_extent)
            maximum_extent = region_size[1];
        if (region_size[2] > maximum_extent)
            maximum_extent = region_size[2];
        const double root = maximum_extent * result.scale_factor;
        result.root_scale = static_cast<float>(root);
        if (!std::isfinite(result.root_scale) || !(result.root_scale > 0.0F))
        {
            throw std::invalid_argument("OOC adaptive-root scale is invalid");
        }
        return result;
    }

    std::vector<std::byte> serialize_ooc_pyramid_registry(std::span<const OocPyramidRegistryRawRecord> records)
    {
        constexpr std::size_t header_size = sizeof(std::uint64_t);
        constexpr std::size_t record_size = sizeof(OocPyramidRegistryRawRecord);
        if (records.size() > (std::numeric_limits<std::size_t>::max() - header_size) / record_size)
        {
            throw std::length_error("OOC pyramid registry is too large");
        }
        std::vector<std::byte> result(header_size + records.size() * record_size);
        const std::uint64_t count = static_cast<std::uint64_t>(records.size());
        for (std::size_t index = 0; index != header_size; ++index)
        {
            result[index] = static_cast<std::byte>(count >> (index * 8U));
        }
        if (!records.empty())
        {
            std::memcpy(result.data() + header_size, records.data(), records.size() * record_size);
        }
        return result;
    }

    OocPyramidRegistryPayloadLocation
    ooc_pyramid_registry_payload_location(const OocPyramidRegistryRawRecord& record) noexcept
    {
        const auto read_u64_le = [&](std::size_t offset)
        {
            std::uint64_t value = 0U;
            for (std::size_t index = 0; index != sizeof(value); ++index)
            {
                value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(record[offset + index]))
                         << (index * 8U);
            }
            return value;
        };
        return {
            read_u64_le(0x4BCU),
            read_u64_le(0x4C4U),
            read_u64_le(0x4CCU),
        };
    }

    OocPyramidPayloadLayout parse_ooc_pyramid_payload_layout(std::span<const std::byte> segment,
                                                             std::uint32_t level_count)
    {
        if (level_count == 0U || level_count > (std::numeric_limits<std::uint32_t>::max() - 2U) / 4U)
        {
            throw std::invalid_argument("invalid OOC pyramid level count");
        }
        const std::uint64_t expected_table_count = 4ULL * static_cast<std::uint64_t>(level_count) - 2ULL;
        std::size_t cursor = 0U;
        auto take_u64_le = [&](const char* label)
        {
            if (cursor > segment.size() || segment.size() - cursor < 8U)
            {
                throw std::invalid_argument(std::string("OOC pyramid segment has truncated ") + label);
            }
            std::uint64_t value = 0U;
            for (std::size_t byte = 0U; byte != 8U; ++byte)
            {
                value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(segment[cursor + byte]))
                         << (byte * 8U);
            }
            cursor += 8U;
            return value;
        };
        const std::uint64_t table_count = take_u64_le("table count");
        if (table_count != expected_table_count)
        {
            throw std::invalid_argument("OOC pyramid segment table count does not match its levels");
        }
        if (table_count > (segment.size() - cursor) / 8U)
        {
            throw std::invalid_argument("OOC pyramid segment table is truncated");
        }
        std::vector<std::uint64_t> table(static_cast<std::size_t>(table_count));
        for (std::uint64_t& value : table)
            value = take_u64_le("table item");
        const std::uint64_t encoded_blob_size = take_u64_le("blob size");
        if (encoded_blob_size != segment.size() - cursor)
        {
            throw std::invalid_argument("OOC pyramid segment blob size does not consume the segment");
        }

        OocPyramidPayloadLayout result;
        result.table_count = table_count;
        result.encoded_blob_offset = cursor;
        result.encoded_blob_size = encoded_blob_size;
        result.levels.resize(level_count);
        std::uint64_t blob_cursor = 0U;
        auto take_blob = [&](std::uint64_t size, const char* label)
        {
            if (size > encoded_blob_size - std::min(encoded_blob_size, blob_cursor))
            {
                throw std::invalid_argument(std::string("OOC pyramid segment has truncated ") + label);
            }
            const std::uint64_t offset = blob_cursor;
            blob_cursor += size;
            return offset;
        };
        for (std::uint32_t level = 0U; level != level_count; ++level)
        {
            auto& output = result.levels[level];
            const std::size_t base = static_cast<std::size_t>(level) * 4U;
            output.first_exr_size = table[base];
            output.first_exr_offset = result.encoded_blob_offset + take_blob(output.first_exr_size, "first EXR");
            output.second_exr_size = table[base + 1U];
            output.second_exr_offset = result.encoded_blob_offset + take_blob(output.second_exr_size, "second EXR");
            if (level + 1U == level_count)
                continue;
            output.next_width = table[base + 2U];
            output.next_height = table[base + 3U];
            if (output.next_width == 0U || output.next_height == 0U ||
                output.next_width > std::numeric_limits<std::uint64_t>::max() / output.next_height)
            {
                throw std::invalid_argument("OOC pyramid segment has invalid next-level dimensions");
            }
            output.interlevel_plane_size = output.next_width * output.next_height;
            output.interlevel_plane_offset =
                result.encoded_blob_offset + take_blob(output.interlevel_plane_size, "interlevel byte plane");
        }
        if (blob_cursor != encoded_blob_size)
        {
            throw std::invalid_argument("OOC pyramid segment layout does not consume its blob");
        }
        return result;
    }

    std::vector<OocPyramidRegistryRawRecord> deserialize_ooc_pyramid_registry(std::span<const std::byte> bytes)
    {
        constexpr std::size_t header_size = sizeof(std::uint64_t);
        constexpr std::size_t record_size = sizeof(OocPyramidRegistryRawRecord);
        if (bytes.size() < header_size)
        {
            throw std::invalid_argument("OOC pyramid registry header is truncated");
        }
        std::uint64_t count = 0U;
        for (std::size_t index = 0; index != header_size; ++index)
        {
            count |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[index])) << (index * 8U);
        }
        if (count > (std::numeric_limits<std::size_t>::max() - header_size) / record_size)
        {
            throw std::invalid_argument("OOC pyramid registry count overflows size");
        }
        const std::size_t expected = header_size + static_cast<std::size_t>(count) * record_size;
        if (bytes.size() != expected)
        {
            throw std::invalid_argument("OOC pyramid registry payload size does not match its count");
        }
        std::vector<OocPyramidRegistryRawRecord> records(static_cast<std::size_t>(count));
        if (!records.empty())
        {
            std::memcpy(records.data(), bytes.data() + header_size, records.size() * record_size);
        }
        return records;
    }

    OocPyramidMetadataCodes quantize_ooc_pyramid_metadata_codes(float value, float base_scale)
    {
        if (!std::isfinite(value) || !std::isfinite(base_scale) || value <= 0.0F || base_scale <= 0.0F)
        {
            throw std::invalid_argument("OOC pyramid metadata quantizer requires finite positive inputs");
        }

        float repeated_scale = base_scale;
        std::uint8_t level = 0U;
        while (repeated_scale * 0.75F > value)
        {
            repeated_scale *= 0.5F;
            ++level;
            if (level == 32U)
            {
                return {32U, 0U};
            }
        }

        const float level_scale = base_scale / static_cast<float>(std::uint32_t{1U} << level);
        float normalized = value / level_scale;
        normalized = (normalized - 0.75F) / 0.75F;
        normalized = std::clamp(normalized, 0.0F, 1.0F);
        const auto raw_code = static_cast<std::uint32_t>(normalized * std::bit_cast<float>(0x437FE666U));
        return {level, static_cast<std::uint8_t>(raw_code >> 3U)};
    }

    OocHistogramProjectionResult prepare_ooc_histogram_pyramid_selection_mode0(const OocHistogramVoxel& voxel,
                                                                               const OocHistogramProjectionInput& input)
    {
        if (input.camera.width <= 0 || input.camera.height <= 0 ||
            input.camera.width > std::numeric_limits<std::int32_t>::max() ||
            input.camera.height > std::numeric_limits<std::int32_t>::max() || input.camera.focal_length == 0.0 ||
            input.camera.focal_length + input.camera.additive_focal == 0.0)
        {
            throw std::runtime_error("invalid mode-0 OOC histogram camera");
        }
        if (input.pyramid_levels == 0U || input.pyramid_levels > 31U)
        {
            throw std::runtime_error("invalid OOC histogram pyramid level count");
        }
        if (voxel.scalar_lut_index >= input.scalar_lut.size())
        {
            throw std::runtime_error("OOC histogram scalar LUT index is out of range");
        }
        const std::uint8_t scale_code = voxel.metadata[2];
        if (scale_code > 32U)
        {
            throw std::runtime_error("unsupported OOC histogram support scale code");
        }

        OocHistogramProjectionResult result;
        const auto camera_point = transform_double4(input.world_to_camera, voxel.position);
        if (!(camera_point[2] > 0.0))
        {
            result.decision = OocHistogramProjectionDecision::BehindCamera;
            return result;
        }

        const double inverse_depth = 1.0 / camera_point[2];
        const double normalized_x = camera_point[0] * inverse_depth;
        const double normalized_y = camera_point[1] * inverse_depth;
        double projected_y = static_cast<double>(input.camera.height) * 0.5;
        projected_y = projected_y + input.camera.principal_y;
        projected_y = projected_y + input.camera.focal_length * normalized_y;
        double projected_delta = input.camera.shear * normalized_y;
        double focal_x = input.camera.focal_length * normalized_x;
        double additive_x = input.camera.additive_focal * normalized_x;
        additive_x = additive_x + focal_x;
        projected_delta = projected_delta + additive_x;
        double projected_x = static_cast<double>(input.camera.width) * 0.5;
        projected_x = projected_x + input.camera.principal_x;
        projected_x = projected_x + projected_delta;

        float x = static_cast<float>(projected_x);
        float y = static_cast<float>(projected_y);
        if (!(x >= 0.0F) || !(y >= 0.0F) || !(x < static_cast<float>(input.camera.width)) ||
            !(y < static_cast<float>(input.camera.height)))
        {
            result.decision = OocHistogramProjectionDecision::OutsideImage;
            return result;
        }

        // Target rounds projection depth through float before both mode-0
        // unprojections. Pixel coordinates are already float and the half-pixel
        // perturbation is also performed in float32.
        const float projected_depth = static_cast<float>(camera_point[2]);
        const auto horizontal_camera = unproject_mode0(x + 0.5F, y, projected_depth, input.camera);
        const auto vertical_camera = unproject_mode0(x, y + 0.5F, projected_depth, input.camera);
        const float horizontal_distance =
            std::sqrt(squared_world_distance(horizontal_camera, input.camera_to_world, voxel.position));
        const float vertical_distance =
            std::sqrt(squared_world_distance(vertical_camera, input.camera_to_world, voxel.position));
        float geometry = std::max(horizontal_distance, vertical_distance);
        if (input.mode_b == 1U)
        {
            constexpr float diagonal = std::bit_cast<float>(std::uint32_t{0x3FB50481U});
            geometry = geometry * diagonal;
        }
        result.base_geometry_distance = geometry;

        constexpr float one = 1.0F;
        constexpr float half = 0.5F;
        constexpr float three_quarters = 0.75F;
        constexpr float weight_denominator = std::bit_cast<float>(std::uint32_t{0x437FE666U});
        result.support_scale = scale_code == 32U ? std::bit_cast<float>(std::uint32_t{0x2F800000U})
                                                 : one / static_cast<float>(1ULL << scale_code);
        result.reference_depth = input.scalar_lut[voxel.scalar_lut_index];
        result.projected_depth = projected_depth;
        result.level_selection_vote = voxel.metadata[3];
        float threshold_factor = half * input.threshold_a;
        threshold_factor = threshold_factor * result.support_scale;
        float vote_factor = static_cast<float>(result.level_selection_vote) / weight_denominator;
        vote_factor = vote_factor * three_quarters;
        vote_factor = vote_factor + three_quarters;
        threshold_factor = vote_factor * threshold_factor;
        result.level_threshold = result.reference_depth * threshold_factor;

        std::int32_t width = static_cast<std::int32_t>(input.camera.width);
        std::int32_t height = static_cast<std::int32_t>(input.camera.height);
        std::uint32_t scale = 1U;
        std::uint32_t offset = 0U;
        float selected_geometry = geometry;
        constexpr float four = 4.0F;
        if (input.pyramid_levels > 1U && result.level_threshold > selected_geometry * four)
        {
            for (std::uint32_t level = 1U; level < input.pyramid_levels; ++level)
            {
                selected_geometry = selected_geometry + selected_geometry;
                x = x * half;
                y = y * half;
                const std::uint64_t next_offset =
                    static_cast<std::uint64_t>(offset) +
                    2ULL * static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
                if (next_offset > std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::runtime_error("OOC histogram pyramid offset overflow");
                }
                offset = static_cast<std::uint32_t>(next_offset);
                width /= 2;
                height /= 2;
                scale += scale;
                if (level + 1U == input.pyramid_levels || !(result.level_threshold > selected_geometry * four))
                {
                    break;
                }
            }
        }
        result.geometry_distance = selected_geometry;
        result.selection = {x, y, width, height, scale, offset};
        result.decision = OocHistogramProjectionDecision::Selected;
        return result;
    }

    OocHistogramBilinearAccumulation
    accumulate_ooc_histogram_pyramid_sample(const OocHistogramPyramidSelection& selection,
                                            std::span<const float> pyramid) noexcept
    {
        float x = selection.projected_x - 0.5F;
        float y = selection.projected_y - 0.5F;
        const auto integer_x = static_cast<std::int32_t>(x);
        const auto integer_y = static_cast<std::int32_t>(y);
        const float fraction_x = x - static_cast<float>(integer_x);
        const float fraction_y = y - static_cast<float>(integer_y);
        const auto clamp_x = [&selection](std::int32_t value) noexcept
        { return std::max<std::int32_t>(0, std::min<std::int32_t>(selection.level_width - 1, value)); };
        const auto clamp_y = [&selection](std::int32_t value) noexcept
        { return std::max<std::int32_t>(0, std::min<std::int32_t>(selection.level_height - 1, value)); };
        const std::int32_t x0 = clamp_x(integer_x);
        const std::int32_t x1 = clamp_x(integer_x + 1);
        const std::int32_t y0 = clamp_y(integer_y);
        const std::int32_t y1 = clamp_y(integer_y + 1);
        const std::array<std::int32_t, 4> pixels{
            x0 + selection.level_width * y0,
            x1 + selection.level_width * y0,
            x0 + selection.level_width * y1,
            x1 + selection.level_width * y1,
        };
        const std::array<float, 4> weights{
            (1.0F - fraction_y) * (1.0F - fraction_x),
            (1.0F - fraction_y) * fraction_x,
            (1.0F - fraction_x) * fraction_y,
            fraction_y * fraction_x,
        };

        OocHistogramBilinearAccumulation result;
        for (std::size_t corner = 0; corner != pixels.size(); ++corner)
        {
            const auto index =
                static_cast<std::size_t>(selection.base_float_offset + 2U * static_cast<std::uint32_t>(pixels[corner]));
            if (index + 1U >= pyramid.size())
                continue;
            const float depth = pyramid[index];
            if (depth == 0.0F || std::bit_cast<std::uint32_t>(depth) == 0xD3800000U)
            {
                continue;
            }
            const float weight = weights[corner];
            result.depth_sum = result.depth_sum + depth * weight;
            result.valid_weight = result.valid_weight + weight;
            result.auxiliary_sum = result.auxiliary_sum + pyramid[index + 1U] * weight;
        }
        return result;
    }

    OocHistogramAcceptedVote evaluate_ooc_histogram_sample(const OocHistogramAccumulatedSample& sample) noexcept
    {
        constexpr float one = 1.0F;
        constexpr float negative_support_multiplier = 8.0F;
        constexpr float auxiliary_multiplier = 1.5F;
        constexpr float geometry_reject_multiplier = std::bit_cast<float>(std::uint32_t{0x3FD9999AU});
        constexpr float geometry_ramp_start = std::bit_cast<float>(std::uint32_t{0x3FA66666U});
        constexpr float geometry_ramp_width = std::bit_cast<float>(std::uint32_t{0x3ECCCCD0U});

        float auxiliary = sample.auxiliary_sum / sample.valid_weight;
        const float pyramid_scale = static_cast<float>(sample.pyramid_scale);
        float negative_limit = negative_support_multiplier * sample.level_threshold;
        float residual = sample.depth_sum / sample.valid_weight;
        auxiliary = auxiliary / pyramid_scale;
        residual = residual - sample.projected_depth;
        auxiliary = auxiliary * sample.threshold_b;
        const float one_and_half_auxiliary = auxiliary_multiplier * auxiliary;
        negative_limit = negative_limit + one_and_half_auxiliary;
        negative_limit = -negative_limit;
        if (negative_limit > residual)
        {
            return {OocHistogramSampleDecision::NegativeDepthGate, 0.0F, 0.0F};
        }

        const float geometry_limit = geometry_reject_multiplier * sample.level_threshold;
        if (sample.geometry_distance > geometry_limit)
        {
            return {OocHistogramSampleDecision::GeometryGate, 0.0F, 0.0F};
        }

        float raw_vote_weight = static_cast<float>(sample.vote_base);
        const float attenuation_start = sample.level_threshold * geometry_ramp_start;
        if (attenuation_start < sample.geometry_distance)
        {
            float ramp = sample.geometry_distance / sample.level_threshold;
            ramp = ramp - geometry_ramp_start;
            ramp = ramp / geometry_ramp_width;
            float attenuation = one - ramp;
            raw_vote_weight = raw_vote_weight * attenuation;
        }

        float normalized_residual = residual / auxiliary;
        if (one > normalized_residual)
        {
            normalized_residual = std::max(normalized_residual, -one);
        }
        else
        {
            normalized_residual = one;
        }

        if (residual > -one_and_half_auxiliary)
        {
            float vote_scale = pyramid_scale;
            if (sample.pyramid_scale > 3U)
                vote_scale = 4.0F;
            raw_vote_weight = raw_vote_weight * vote_scale;
            raw_vote_weight = raw_vote_weight * 3.0F;
        }
        return {OocHistogramSampleDecision::Accepted, normalized_residual, raw_vote_weight};
    }

    void
    accumulate_ooc_histogram_vote(OocHistogramVoxel& voxel, float normalized_residual, float raw_vote_weight) noexcept
    {
        constexpr float one = 1.0F;
        constexpr float half = 0.5F;
        constexpr float bin_count_minus_one = 9.0F;

        float position = one + normalized_residual;
        position = position * half;
        position = position * bin_count_minus_one;
        position = position + half;
        std::int32_t selected = static_cast<std::int32_t>(position);
        selected = std::max<std::int32_t>(0, std::min<std::int32_t>(9, selected));

        const auto center = [](std::int32_t index) noexcept
        {
            float value = static_cast<float>(index);
            value = value / 9.0F;
            value = value + value;
            value = value - 1.0F;
            return value;
        };
        const auto add_saturated = [&voxel](std::int32_t index, std::int32_t increment) noexcept
        {
            std::uint8_t& bin = voxel.histogram[static_cast<std::size_t>(index)];
            if (bin == 255U)
                return;
            const std::uint32_t sum = static_cast<std::uint32_t>(bin) + static_cast<std::uint32_t>(increment);
            bin = static_cast<std::uint8_t>(std::min<std::uint32_t>(254U, sum));
        };

        // COMISS(raw_vote_weight, 1.0f) followed by JBE also sends unordered
        // values to this path. Accepted runtime votes are finite and nonnegative.
        if (!(raw_vote_weight > one))
        {
            std::uint8_t& bin = voxel.histogram[static_cast<std::size_t>(selected)];
            if (bin == 255U)
                return;
            float combined = static_cast<float>(bin);
            combined = combined + raw_vote_weight;
            const auto truncated = static_cast<std::int32_t>(combined);
            bin = static_cast<std::uint8_t>(std::min<std::int32_t>(254, std::max<std::int32_t>(0, truncated)));
            return;
        }

        std::int32_t total = 254;
        if (raw_vote_weight < 254.0F)
        {
            // The target converts to int32, then keeps its low uint8 before
            // converting that byte back to float for the split arithmetic.
            total = static_cast<std::uint8_t>(static_cast<std::int32_t>(raw_vote_weight));
        }
        const float quantized_weight = static_cast<float>(total);

        std::int32_t lower = selected;
        std::int32_t upper = selected + 1;
        if (selected != 0 && (selected == 9 || normalized_residual < center(selected)))
        {
            lower = selected - 1;
            upper = selected;
        }

        const float upper_center = center(upper);
        const float lower_center = center(lower);
        float numerator = upper_center - normalized_residual;
        float denominator = upper_center - lower_center;
        float lower_weight = numerator / denominator;
        lower_weight = lower_weight * quantized_weight;
        const auto lower_increment = static_cast<std::int32_t>(std::round(lower_weight));
        const auto upper_increment = total - lower_increment;
        add_saturated(lower, lower_increment);
        add_saturated(upper, upper_increment);
    }

    OocHistogramCameraVoteResult accumulate_ooc_histogram_camera_vote_mode0(OocHistogramVoxel& voxel,
                                                                            const OocHistogramProjectionInput& input,
                                                                            float threshold_b,
                                                                            std::span<const float> pyramid)
    {
        OocHistogramCameraVoteResult result;
        result.projection = prepare_ooc_histogram_pyramid_selection_mode0(voxel, input);
        if (!result.projection.selected())
        {
            result.decision = result.projection.decision == OocHistogramProjectionDecision::BehindCamera
                                  ? OocHistogramCameraVoteDecision::BehindCamera
                                  : OocHistogramCameraVoteDecision::OutsideImage;
            return result;
        }

        const auto& selection = result.projection.selection;
        if (selection.level_width <= 0 || selection.level_height <= 0)
        {
            throw std::runtime_error("invalid selected OOC histogram pyramid level");
        }
        const std::uint64_t required = static_cast<std::uint64_t>(selection.base_float_offset) +
                                       2ULL * static_cast<std::uint32_t>(selection.level_width) *
                                           static_cast<std::uint32_t>(selection.level_height);
        if (required > pyramid.size())
        {
            throw std::runtime_error("OOC histogram pyramid payload is truncated");
        }

        result.accumulation = accumulate_ooc_histogram_pyramid_sample(selection, pyramid);
        if (!result.accumulation.valid())
        {
            result.decision = OocHistogramCameraVoteDecision::InvalidPyramidSample;
            return result;
        }

        result.vote = evaluate_ooc_histogram_sample({result.accumulation.depth_sum,
                                                     result.accumulation.auxiliary_sum,
                                                     result.accumulation.valid_weight,
                                                     result.projection.level_threshold,
                                                     result.projection.geometry_distance,
                                                     result.projection.selection.pyramid_scale,
                                                     result.projection.projected_depth,
                                                     threshold_b,
                                                     input.mode_b});
        switch (result.vote.decision)
        {
        case OocHistogramSampleDecision::NegativeDepthGate:
            result.decision = OocHistogramCameraVoteDecision::NegativeDepthGate;
            return result;
        case OocHistogramSampleDecision::GeometryGate:
            result.decision = OocHistogramCameraVoteDecision::GeometryGate;
            return result;
        case OocHistogramSampleDecision::Accepted:
            break;
        }
        accumulate_ooc_histogram_vote(voxel, result.vote.normalized_residual, result.vote.raw_vote_weight);
        result.decision = OocHistogramCameraVoteDecision::Accepted;
        return result;
    }

    std::uint16_t ooc_weight_sum(const OocWeightedNodeRecord& record) noexcept
    {
        return static_cast<std::uint16_t>(record.weight_sum_le[0]) |
               static_cast<std::uint16_t>(record.weight_sum_le[1] << 8U);
    }

    void set_ooc_weight_sum(OocWeightedNodeRecord& record, std::uint16_t value) noexcept
    {
        record.weight_sum_le[0] = static_cast<std::uint8_t>(value & 0xFFU);
        record.weight_sum_le[1] = static_cast<std::uint8_t>(value >> 8U);
    }

    std::uint8_t quantize_ooc_initial_weight(float cell_scale, float sample_scale) noexcept
    {
        constexpr float lower_scale = 0.75F;
        constexpr float quantization_scale = 255.899993896484375F;
        float normalized = sample_scale / cell_scale;
        normalized = normalized - lower_scale;
        normalized = normalized / lower_scale;

        // This branch order mirrors COMISS in sub_1EA3AE0. In particular, an
        // unordered comparison takes the saturated branch and returns 255.
        if (!(1.0F > normalized))
            return 255U;
        if (!(normalized > 0.0F))
            return 0U;
        return static_cast<std::uint8_t>(static_cast<std::int32_t>(normalized * quantization_scale));
    }

    OocInitialWeightSelection select_ooc_initial_weight(float root_scale,
                                                        float alternate_scale,
                                                        float raw_sample_scale,
                                                        std::uint32_t maximum_level) noexcept
    {
        constexpr float half = 0.5F;
        constexpr float lower_scale = 0.75F;
        constexpr float upper_scale = 1.5F;

        float alternate_cell = half * alternate_scale;
        std::uint8_t alternate_level = 0U;
        while (alternate_level != 32U && lower_scale * alternate_cell > raw_sample_scale)
        {
            alternate_cell = alternate_cell * half;
            ++alternate_level;
        }

        const float root_cell = half * root_scale;
        float probe_cell = root_cell;
        std::uint8_t root_level = 0U;
        while (root_level != 32U && lower_scale * probe_cell > raw_sample_scale)
        {
            probe_cell = probe_cell * half;
            ++root_level;
        }

        std::uint32_t selected_level = maximum_level;
        bool clamp_sample = false;
        if (maximum_level > root_level)
        {
            selected_level = root_level;
            clamp_sample = true;
        }
        else if (maximum_level >= alternate_level)
        {
            selected_level = alternate_level;
            clamp_sample = true;
        }

        // SHL r32,cl masks the count to five bits. CVTSI2SS then treats the result
        // as signed, exactly as at 0x17A5780..0x17A5795.
        const std::uint32_t shift_bits = 1U << (static_cast<unsigned>(selected_level) & 31U);
        const auto signed_divisor = std::bit_cast<std::int32_t>(shift_bits);
        const float cell_scale = root_cell / static_cast<float>(signed_divisor);
        float sample_scale = raw_sample_scale;
        if (clamp_sample)
        {
            const float lower = lower_scale * cell_scale;
            const float upper = upper_scale * cell_scale;
            // MAXSS/MINSS choose their second operand on equality or unordered.
            sample_scale = sample_scale > lower ? sample_scale : lower;
            sample_scale = sample_scale < upper ? sample_scale : upper;
        }
        return {
            static_cast<std::uint8_t>(selected_level),
            cell_scale,
            sample_scale,
            selected_level == 0U && !clamp_sample,
        };
    }

    void set_initial_ooc_weight(OocWeightedNodeRecord& record,
                                std::uint32_t raw_denominator,
                                std::uint8_t quantized_weight) noexcept
    {
        const std::uint32_t product = raw_denominator * static_cast<std::uint32_t>(quantized_weight);
        set_ooc_weight_sum(record, static_cast<std::uint16_t>(product));
        record.weight_denominator = static_cast<std::uint8_t>(raw_denominator);
    }

    std::array<std::uint32_t, 3>
    encode_ooc_morton_words(const std::array<float, 3>& position, float root_scale, std::uint8_t level) noexcept
    {
        std::array<std::uint32_t, 3> words{};
        if (level == 0U || level > 32U || !(root_scale > 0.0F))
        {
            return words;
        }

        std::array<std::uint32_t, 3> cells{};
        constexpr double coordinate_scale = 4294967296.0;
        constexpr double maximum_coordinate = 4294967295.0;
        for (std::size_t axis = 0; axis < position.size(); ++axis)
        {
            // Keep the division as a distinct float32 operation. The target then
            // promotes that rounded quotient before applying the exact double 2^32.
            const float normalized = position[axis] / root_scale;
            const double scaled = static_cast<double>(normalized) * coordinate_scale;
            std::uint32_t coordinate = 0U;
            if (scaled >= 0.0)
            {
                coordinate = scaled > maximum_coordinate ? std::numeric_limits<std::uint32_t>::max()
                                                         : static_cast<std::uint32_t>(scaled);
            }
            cells[axis] = level == 32U ? coordinate : coordinate >> (32U - level);
        }

        for (unsigned depth = 0; depth < level; ++depth)
        {
            const unsigned coordinate_bit = static_cast<unsigned>(level) - depth - 1U;
            const unsigned top = 95U - 3U * depth;
            for (unsigned axis = 0; axis < 3U; ++axis)
            {
                if (((cells[axis] >> coordinate_bit) & 1U) == 0U)
                    continue;
                const unsigned position_bit = top - axis;
                if (position_bit >= 64U)
                {
                    words[0] |= 1U << (position_bit - 64U);
                }
                else if (position_bit >= 32U)
                {
                    words[1] |= 1U << (position_bit - 32U);
                }
                else
                {
                    words[2] |= 1U << position_bit;
                }
            }
        }
        return words;
    }

    OocWeightedNodeRecord make_initial_ooc_weighted_node(const OocDepthCandidate& candidate,
                                                         float root_scale,
                                                         float alternate_scale,
                                                         std::uint32_t maximum_level,
                                                         std::uint32_t raw_denominator) noexcept
    {
        const OocInitialWeightSelection selection =
            select_ooc_initial_weight(root_scale, alternate_scale, candidate.scale, maximum_level);
        OocWeightedNodeRecord record{};
        record.morton_words = encode_ooc_morton_words(candidate.position, root_scale, selection.level);
        record.level = selection.level;
        const std::uint8_t weight =
            selection.direct_zero ? 0U : quantize_ooc_initial_weight(selection.cell_scale, selection.sample_scale);
        set_initial_ooc_weight(record, raw_denominator, weight);
        return record;
    }

    float quantize_ooc_sample_scale_cache(float value) noexcept
    {
        std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        // Unsigned wraparound mirrors the observed target bit operation.
        bits = (bits + 0x80U) & 0xFFFFFF00U;
        return std::bit_cast<float>(bits);
    }

    void quantize_ooc_sample_scale_cache(std::span<float> values) noexcept
    {
        for (float& value : values)
        {
            value = quantize_ooc_sample_scale_cache(value);
        }
    }

    std::vector<OocSampleScalePointRecord>
    build_ooc_sample_scale_point_records_mode0(std::size_t width,
                                               std::size_t height,
                                               std::span<const float> depth,
                                               const OocSampleScaleCameraMode0& camera,
                                               const std::array<double, 16>& camera_to_record)
    {
        const std::size_t pixels = width * height;
        if (depth.size() != pixels)
        {
            throw std::runtime_error("OOC point-record producer input size mismatch");
        }

        std::vector<OocSampleScalePointRecord> records(pixels);
        // sub_27D1340 first materializes the full 3x3 intrinsic matrix. The mode-0
        // fast path then uses the generic cofactor inverse below, rather than the
        // algebraically shorter triangular inverse. Preserving that operation order
        // is observable at one float32 output among the six captured maps.
        double horizontal_focal = camera.additive_focal;
        horizontal_focal = horizontal_focal + camera.focal_length;
        double absolute_x = static_cast<double>(camera.width) * 0.5;
        absolute_x = absolute_x + camera.principal_x;
        double absolute_y = static_cast<double>(camera.height) * 0.5;
        absolute_y = absolute_y + camera.principal_y;
        const std::array<double, 9> intrinsic{
            horizontal_focal, camera.shear, absolute_x, 0.0, camera.focal_length, absolute_y, 0.0, 0.0, 1.0};
        const double a = intrinsic[0];
        const double b = intrinsic[1];
        const double c = intrinsic[2];
        const double d = intrinsic[3];
        const double e = intrinsic[4];
        const double f = intrinsic[5];
        const double g = intrinsic[6];
        const double h = intrinsic[7];
        const double i = intrinsic[8];
        const double cofactor_00 = e * i - f * h;
        const double minor_01 = d * i - f * g;
        const double cofactor_20 = d * h - e * g;
        double determinant = a * cofactor_00;
        determinant = determinant - b * minor_01;
        determinant = determinant + c * cofactor_20;
        const double inverse_determinant = 1.0 / determinant;
        const std::array<double, 9> inverse_intrinsic{cofactor_00 * inverse_determinant,
                                                      (c * h - b * i) * inverse_determinant,
                                                      (b * f - c * e) * inverse_determinant,
                                                      (-minor_01) * inverse_determinant,
                                                      (a * i - c * g) * inverse_determinant,
                                                      (c * d - a * f) * inverse_determinant,
                                                      cofactor_20 * inverse_determinant,
                                                      (b * g - a * h) * inverse_determinant,
                                                      (a * e - b * d) * inverse_determinant};

        if (camera_to_record[12] != 0.0 || camera_to_record[13] != 0.0 || camera_to_record[14] != 0.0 ||
            camera_to_record[15] == 0.0)
        {
            throw std::runtime_error("OOC mode-0 point-record replay requires the observed affine transform");
        }
        std::array<double, 16> normalized = camera_to_record;
        for (std::size_t column = 0; column < 3U; ++column)
        {
            double squared = normalized[4U + column] * normalized[4U + column];
            squared = squared + normalized[column] * normalized[column];
            squared = squared + normalized[8U + column] * normalized[8U + column];
            double inverse_length = 1.0;
            inverse_length = inverse_length / std::sqrt(squared);
            for (std::size_t row = 0; row < 3U; ++row)
            {
                normalized[row * 4U + column] = normalized[row * 4U + column] * inverse_length;
            }
        }
        std::array<std::array<double, 4>, 3> coefficients{};
        for (std::size_t row = 0; row < 3U; ++row)
        {
            for (std::size_t column = 0; column < 3U; ++column)
            {
                double value = normalized[row * 4U] * inverse_intrinsic[column];
                value = value + normalized[row * 4U + 1U] * inverse_intrinsic[3U + column];
                value = value + normalized[row * 4U + 2U] * inverse_intrinsic[6U + column];
                coefficients[row][column] = value;
            }
            coefficients[row][3] = normalized[row * 4U + 3U] / normalized[15];
        }

        for (std::size_t row = 0; row < height; ++row)
        {
            const double row_coordinate = static_cast<double>(row) + 0.5;
            for (std::size_t column = 0; column < width; ++column)
            {
                const std::size_t index = row * width + column;
                const float value = depth[index];
                // UCOMISS against zero accepts unordered values and rejects only
                // ordered zero. The captured depth domain contains no NaNs.
                if (value == 0.0F)
                    continue;
                const double promoted_depth = static_cast<double>(value);
                const double row_depth = promoted_depth * row_coordinate;
                const double column_coordinate = static_cast<double>(column) + 0.5;
                const double column_depth = column_coordinate * promoted_depth;
                for (std::size_t coordinate = 0; coordinate < 3U; ++coordinate)
                {
                    double result = coefficients[coordinate][1] * row_depth;
                    result = result + coefficients[coordinate][0] * column_depth;
                    result = result + coefficients[coordinate][2] * promoted_depth;
                    result = result + coefficients[coordinate][3];
                    records[index].point[coordinate] = static_cast<float>(result);
                }
            }
        }

        const auto depth_valid = [](float value) noexcept { return value != 0.0F; };
        const auto add_triangle = [&](std::size_t first, std::size_t second, std::size_t third)
        {
            const auto& p0 = records[first].point;
            const auto& p1 = records[second].point;
            const auto& p2 = records[third].point;
            const float e2z = p2[2] - p0[2];
            const float e1z = p1[2] - p0[2];
            const float e2y = p2[1] - p0[1];
            const float e2x = p2[0] - p0[0];
            const float e1y = p1[1] - p0[1];
            const float e1x = p1[0] - p0[0];

            const float normal_x = e2z * e1y - e2y * e1z;
            const float normal_y = e2x * e1z - e1x * e2z;
            const float normal_z = e2y * e1x - e2x * e1y;
            float squared = normal_x * normal_x;
            squared = squared + normal_y * normal_y;
            squared = squared + normal_z * normal_z;
            const float length = std::sqrt(squared);
            float inverse = 0.0F;
            if (!(1.0e-20 > static_cast<double>(length)))
            {
                inverse = 1.0F / length;
            }
            const std::array<float, 3> normal{normal_x * inverse, normal_y * inverse, normal_z * inverse};
            for (const std::size_t index : {first, second, third})
            {
                records[index].direction[0] = records[index].direction[0] + normal[0];
                records[index].direction[1] = records[index].direction[1] + normal[1];
                records[index].direction[2] = records[index].direction[2] + normal[2];
            }
        };

        if (width > 1U && height > 1U)
        {
            for (std::size_t row = 0; row + 1U < height; ++row)
            {
                for (std::size_t column = 0; column + 1U < width; ++column)
                {
                    const std::size_t tl = row * width + column;
                    const std::size_t tr = tl + 1U;
                    const std::size_t bl = tl + width;
                    const std::size_t br = bl + 1U;
                    const bool valid_tl = depth_valid(depth[tl]);
                    const bool valid_tr = depth_valid(depth[tr]);
                    const bool valid_bl = depth_valid(depth[bl]);
                    const bool valid_br = depth_valid(depth[br]);
                    std::uint8_t topology = 0U;
                    if (valid_tl && valid_tr && valid_bl && valid_br)
                    {
                        const float first_difference = std::abs(depth[tl] - depth[br]);
                        const float second_difference = std::abs(depth[tr] - depth[bl]);
                        topology = first_difference > second_difference ? 0x09U : 0x06U;
                    }
                    else if (!valid_tl && valid_tr && valid_bl && valid_br)
                    {
                        topology = 0x08U;
                    }
                    else if (valid_tl && !valid_tr && valid_bl && valid_br)
                    {
                        topology = 0x04U;
                    }
                    else if (valid_tl && valid_tr && !valid_bl && valid_br)
                    {
                        topology = 0x02U;
                    }
                    else if (valid_tl && valid_tr && valid_bl && !valid_br)
                    {
                        topology = 0x01U;
                    }

                    // sub_209F3B0 materializes triangle records in this bit order.
                    if ((topology & 0x01U) != 0U)
                        add_triangle(tl, bl, tr);
                    if ((topology & 0x02U) != 0U)
                        add_triangle(tl, br, tr);
                    if ((topology & 0x04U) != 0U)
                        add_triangle(tl, bl, br);
                    if ((topology & 0x08U) != 0U)
                        add_triangle(tr, bl, br);
                }
            }
        }

        for (auto& record : records)
        {
            const float x = record.direction[0];
            const float y = record.direction[1];
            const float z = record.direction[2];
            float squared = x * x;
            squared = squared + y * y;
            squared = squared + z * z;
            float inverse = 0.0F;
            if (squared != 0.0F)
            {
                double inverse_double = 1.0;
                inverse_double = inverse_double / std::sqrt(static_cast<double>(squared));
                inverse = static_cast<float>(inverse_double);
            }
            record.direction[0] = x * inverse;
            record.direction[1] = y * inverse;
            record.direction[2] = z * inverse;
        }
        return records;
    }

    OocSampleScaleWorkerOutput build_ooc_sample_scale_workspace(const OocSampleScaleWorkerInput& input)
    {
        const std::size_t pixels = input.width * input.height;
        if (input.depth.size() != pixels || input.point_records.size() != pixels)
        {
            throw std::runtime_error("OOC sample-scale worker input size mismatch");
        }

        OocSampleScaleWorkerOutput output;
        output.depth.assign(pixels, 0.0F);
        output.temporary.assign(pixels, 0.0F);
        output.sample_scale.assign(pixels, 0.0F);
        output.level.assign(pixels, 0U);
        output.level_weight.assign(pixels, 0U);

        const auto unproject = [&](double pixel_x, double pixel_y, double depth)
        {
            const double half_height = static_cast<double>(input.camera.height) * 0.5;
            double y = pixel_y - half_height;
            y = y - input.camera.principal_y;
            y = y / input.camera.focal_length;

            const double half_width = static_cast<double>(input.camera.width) * 0.5;
            double x = pixel_x - half_width;
            x = x - input.camera.principal_x;
            const double shear_y = input.camera.shear * y;
            x = x - shear_y;
            double horizontal_focal = input.camera.focal_length;
            horizontal_focal = horizontal_focal + input.camera.additive_focal;
            x = x / horizontal_focal;

            x = x * depth;
            y = y * depth;
            const double z = 1.0 * depth;
            return std::array<double, 3>{x, y, z};
        };
        const auto transform = [&](const std::array<double, 3>& point)
        {
            const auto& matrix = input.camera_to_record;
            double denominator = matrix[12] * point[0];
            denominator = denominator + matrix[13] * point[1];
            denominator = denominator + matrix[14] * point[2];
            denominator = denominator + matrix[15];
            std::array<float, 3> result{};
            for (std::size_t row = 0; row < 3U; ++row)
            {
                double value = matrix[row * 4U] * point[0];
                value = value + matrix[row * 4U + 1U] * point[1];
                value = value + matrix[row * 4U + 2U] * point[2];
                value = value + matrix[row * 4U + 3U];
                value = value / denominator;
                result[row] = static_cast<float>(value);
            }
            return result;
        };
        const auto squared_distance = [](const std::array<float, 3>& first, const std::array<float, 3>& second)
        {
            const float dx = first[0] - second[0];
            const float dy = first[1] - second[1];
            const float dz = first[2] - second[2];
            float result = dx * dx;
            result = result + dy * dy;
            result = result + dz * dz;
            return result;
        };

        constexpr float half = 0.5F;
        constexpr float diagonal = 1.414199948310852F;
        constexpr float radius_scale = 0.8695652484893799F;
        constexpr float incidence_limit = 0.699999988079071F;
        constexpr float incidence_floor = 0.10000000149011612F;
        constexpr float level_ratio = 0.75F;
        for (std::size_t row = 0; row < input.height; ++row)
        {
            for (std::size_t column = 0; column < input.width; ++column)
            {
                const std::size_t index = row * input.width + column;
                const auto& record = input.point_records[index];
                float direction_squared = record.direction[0] * record.direction[0];
                direction_squared = direction_squared + record.direction[1] * record.direction[1];
                direction_squared = direction_squared + record.direction[2] * record.direction[2];
                // UCOMISS/JNP/JNE accepts unordered values and rejects only an
                // ordered zero squared norm.
                if (direction_squared == 0.0F)
                    continue;

                const float depth = input.depth[index];
                const double center_x = static_cast<double>(column) + 0.5;
                const double center_y = static_cast<double>(row) + 0.5;
                const auto horizontal = transform(unproject(center_x + 0.5, center_y, static_cast<double>(depth)));
                const auto vertical = transform(unproject(center_x, center_y + 0.5, static_cast<double>(depth)));
                const auto center_half = transform(unproject(center_x, center_y, static_cast<double>(half * depth)));

                const float horizontal_squared = squared_distance(horizontal, record.point);
                const float vertical_squared = squared_distance(vertical, record.point);
                // MAXSS selects its second operand on equality or unordered.
                const float maximum_squared =
                    horizontal_squared > vertical_squared ? horizontal_squared : vertical_squared;
                float radius = std::sqrt(maximum_squared);
                if (input.diagonal_pixel_scale)
                    radius = radius * diagonal;

                const float dx = record.point[0] - center_half[0];
                const float dy = record.point[1] - center_half[1];
                const float dz = record.point[2] - center_half[2];
                float ray_squared = dx * dx;
                ray_squared = ray_squared + dy * dy;
                ray_squared = ray_squared + dz * dz;
                const float ray_length = std::sqrt(ray_squared);
                float inverse_length = 0.0F;
                if (!(1.0e-20 > static_cast<double>(ray_length)))
                {
                    inverse_length = 1.0F / ray_length;
                }

                float incidence = record.direction[0] * (dx * inverse_length);
                incidence = incidence + record.direction[1] * (dy * inverse_length);
                incidence = incidence + record.direction[2] * (dz * inverse_length);
                incidence = std::bit_cast<float>(std::bit_cast<std::uint32_t>(incidence) & 0x7FFFFFFFU);

                float sample_scale = radius * radius_scale;
                if (incidence_limit > incidence)
                {
                    float correction = incidence_limit - incidence;
                    correction = correction / incidence_limit;
                    correction = correction + 1.0F;
                    sample_scale = sample_scale * correction;
                }
                const float denominator = incidence > incidence_floor ? incidence : incidence_floor;
                sample_scale = sample_scale / denominator;

                // The two COMISS/JAE gates reject ordered non-positive or ordered
                // over-threshold values but let unordered values pass.
                if (!std::isnan(sample_scale) && 0.0F >= sample_scale)
                    continue;
                if (!std::isnan(sample_scale) && !std::isnan(input.maximum_sample_scale) &&
                    sample_scale >= input.maximum_sample_scale)
                {
                    continue;
                }

                output.depth[index] = depth;
                output.temporary[index] = 1.0F;
                output.sample_scale[index] = sample_scale;

                const float base_cell = half * input.maximum_sample_scale;
                float cell = base_cell;
                std::uint8_t level = 0U;
                while (level != 32U && level_ratio * cell > sample_scale)
                {
                    cell = cell * half;
                    ++level;
                }
                const float divisor = level == 32U ? 0.0F : static_cast<float>(1U << level);
                const float quantizer_cell = base_cell / divisor;
                output.level[index] = level;
                output.level_weight[index] =
                    static_cast<std::uint8_t>(quantize_ooc_initial_weight(quantizer_cell, sample_scale) >> 3U);
            }
        }
        return output;
    }

    OocPyramidFinerLevelOutput build_ooc_pyramid_finer_level(const OocPyramidFinerLevelInput& input)
    {
        const std::size_t pixels = input.width * input.height;
        if (input.depth.size() != pixels || input.point_records.size() != pixels)
        {
            throw std::runtime_error("OOC finer-level worker input size mismatch");
        }
        if (!input.gate.empty() && input.gate.size() != pixels)
        {
            throw std::runtime_error("OOC finer-level gate size mismatch");
        }

        OocPyramidFinerLevelOutput output;
        output.depth.assign(pixels, 0.0F);
        output.temporary.assign(pixels, 0.0F);
        output.sample_scale.assign(pixels, 0.0F);
        output.temporary_u8.assign(pixels, 0U);

        const auto unproject = [&](double pixel_x, double pixel_y, double depth)
        {
            const double half_height = static_cast<double>(input.camera.height) * 0.5;
            double y = pixel_y - half_height;
            y = y - input.camera.principal_y;
            y = y / input.camera.focal_length;

            const double half_width = static_cast<double>(input.camera.width) * 0.5;
            double x = pixel_x - half_width;
            x = x - input.camera.principal_x;
            const double shear_y = input.camera.shear * y;
            x = x - shear_y;
            double horizontal_focal = input.camera.focal_length;
            horizontal_focal = horizontal_focal + input.camera.additive_focal;
            x = x / horizontal_focal;

            x = x * depth;
            y = y * depth;
            const double z = 1.0 * depth;
            return std::array<double, 3>{x, y, z};
        };
        const auto transform = [&](const std::array<double, 3>& point)
        {
            const auto& matrix = input.camera_to_record;
            double denominator = matrix[12] * point[0];
            denominator = denominator + matrix[13] * point[1];
            denominator = denominator + matrix[14] * point[2];
            denominator = denominator + matrix[15];
            std::array<float, 3> result{};
            for (std::size_t row = 0; row < 3U; ++row)
            {
                double value = matrix[row * 4U] * point[0];
                value = value + matrix[row * 4U + 1U] * point[1];
                value = value + matrix[row * 4U + 2U] * point[2];
                value = value + matrix[row * 4U + 3U];
                value = value / denominator;
                result[row] = static_cast<float>(value);
            }
            return result;
        };
        const auto squared_distance = [](const std::array<float, 3>& first, const std::array<float, 3>& second)
        {
            const float dx = first[0] - second[0];
            const float dy = first[1] - second[1];
            const float dz = first[2] - second[2];
            float result = dx * dx;
            result = result + dy * dy;
            result = result + dz * dz;
            return result;
        };

        constexpr float half = 0.5F;
        constexpr float diagonal = 1.414199948310852F;
        constexpr float radius_scale = 0.8695652484893799F;
        constexpr float incidence_limit = 0.699999988079071F;
        constexpr float incidence_floor = 0.10000000149011612F;
        constexpr std::uint32_t invalid_depth_bits = 0xD3800000U;
        for (std::size_t row = 0; row < input.height; ++row)
        {
            for (std::size_t column = 0; column < input.width; ++column)
            {
                const std::size_t index = row * input.width + column;
                if (!input.gate.empty() && input.gate[index] == 0U)
                {
                    if (input.special_invalid_depth)
                    {
                        output.depth[index] = std::bit_cast<float>(invalid_depth_bits);
                    }
                    continue;
                }

                const auto& record = input.point_records[index];
                float direction_squared = record.direction[0] * record.direction[0];
                direction_squared = direction_squared + record.direction[1] * record.direction[1];
                direction_squared = direction_squared + record.direction[2] * record.direction[2];
                if (direction_squared == 0.0F)
                    continue;

                const float depth = input.depth[index];
                const double center_x = static_cast<double>(column) + 0.5;
                const double center_y = static_cast<double>(row) + 0.5;
                const auto horizontal = transform(unproject(center_x + 0.5, center_y, static_cast<double>(depth)));
                const auto vertical = transform(unproject(center_x, center_y + 0.5, static_cast<double>(depth)));
                auto center_half_unprojected = unproject(center_x, center_y, static_cast<double>(half * depth));
                // sub_1EBFC60, unlike the full-resolution sub_1EBF020 path,
                // rounds all three coordinates of the half-depth unprojection to
                // float before feeding them to the homogeneous transform.
                for (double& value : center_half_unprojected)
                {
                    value = static_cast<double>(static_cast<float>(value));
                }
                const auto center_half = transform(center_half_unprojected);

                const float horizontal_squared = squared_distance(horizontal, record.point);
                const float vertical_squared = squared_distance(vertical, record.point);
                const float maximum_squared =
                    horizontal_squared > vertical_squared ? horizontal_squared : vertical_squared;
                float radius = std::sqrt(maximum_squared);
                if (input.diagonal_pixel_scale)
                    radius = radius * diagonal;

                const float dx = record.point[0] - center_half[0];
                const float dy = record.point[1] - center_half[1];
                const float dz = record.point[2] - center_half[2];
                float ray_squared = dx * dx;
                ray_squared = ray_squared + dy * dy;
                ray_squared = ray_squared + dz * dz;
                const float ray_length = std::sqrt(ray_squared);
                float inverse_length = 0.0F;
                if (!(1.0e-20 > static_cast<double>(ray_length)))
                {
                    inverse_length = 1.0F / ray_length;
                }

                float incidence = record.direction[0] * (dx * inverse_length);
                incidence = incidence + record.direction[1] * (dy * inverse_length);
                incidence = incidence + record.direction[2] * (dz * inverse_length);
                incidence = std::bit_cast<float>(std::bit_cast<std::uint32_t>(incidence) & 0x7FFFFFFFU);

                float sample_scale = radius * radius_scale;
                if (incidence_limit > incidence)
                {
                    float correction = incidence_limit - incidence;
                    correction = correction / incidence_limit;
                    correction = correction + 1.0F;
                    sample_scale = sample_scale * correction;
                }
                const float denominator = incidence > incidence_floor ? incidence : incidence_floor;
                sample_scale = sample_scale / denominator;
                if (!std::isnan(sample_scale) && 0.0F >= sample_scale)
                    continue;

                output.depth[index] = depth;
                output.temporary[index] = 1.0F;
                output.sample_scale[index] = sample_scale;
                output.temporary_u8[index] = 1U;
                ++output.accepted;
                // MAXSS(depth, local_maximum) returns the second operand on
                // equality or unordered inputs.
                output.maximum_depth = depth > output.maximum_depth ? depth : output.maximum_depth;
            }
        }
        return output;
    }

    std::vector<float> expand_ooc_depth_roi(const OocDepthRoiExpansionInput& input)
    {
        if (input.full_width == 0U || input.full_height == 0U || input.x_begin >= input.x_end ||
            input.y_begin >= input.y_end || input.x_end > input.full_width || input.y_end > input.full_height)
        {
            throw std::invalid_argument("OOC depth ROI bounds are invalid");
        }
        const std::size_t roi_width = input.x_end - input.x_begin;
        const std::size_t roi_height = input.y_end - input.y_begin;
        if (roi_width > std::numeric_limits<std::size_t>::max() / roi_height ||
            input.roi_depth.size() != roi_width * roi_height)
        {
            throw std::invalid_argument("OOC depth ROI payload size mismatch");
        }
        if (input.full_width > std::numeric_limits<std::size_t>::max() / input.full_height)
        {
            throw std::invalid_argument("OOC full depth dimensions overflow");
        }

        std::vector<float> result(input.full_width * input.full_height, 0.0F);
        for (std::size_t row = 0U; row != roi_height; ++row)
        {
            const float* source = input.roi_depth.data() + row * roi_width;
            float* destination = result.data() + (input.y_begin + row) * input.full_width + input.x_begin;
            std::memcpy(destination, source, roi_width * sizeof(float));
        }
        return result;
    }

    std::vector<float> extract_ooc_depth_roi(std::size_t full_width,
                                             std::size_t full_height,
                                             const OocDepthRoiBounds& bounds,
                                             std::span<const float> full_depth)
    {
        if (full_width == 0U || full_height == 0U || !bounds.valid() ||
            static_cast<std::uint64_t>(bounds.x_end) > full_width ||
            static_cast<std::uint64_t>(bounds.y_end) > full_height ||
            full_width > std::numeric_limits<std::size_t>::max() / full_height ||
            full_depth.size() != full_width * full_height)
        {
            throw std::invalid_argument("OOC depth ROI extraction input is invalid");
        }
        const std::size_t x_begin = static_cast<std::size_t>(bounds.x_begin);
        const std::size_t y_begin = static_cast<std::size_t>(bounds.y_begin);
        const std::size_t x_end = static_cast<std::size_t>(bounds.x_end);
        const std::size_t y_end = static_cast<std::size_t>(bounds.y_end);
        const std::size_t roi_width = x_end - x_begin;
        const std::size_t roi_height = y_end - y_begin;
        if (roi_width > std::numeric_limits<std::size_t>::max() / roi_height)
        {
            throw std::invalid_argument("OOC depth ROI extraction size overflows");
        }
        std::vector<float> result(roi_width * roi_height);
        for (std::size_t row = 0U; row != roi_height; ++row)
        {
            const float* source = full_depth.data() + (y_begin + row) * full_width + x_begin;
            std::memcpy(result.data() + row * roi_width, source, roi_width * sizeof(float));
        }
        return result;
    }

    std::array<double, 16> condition_ooc_camera_transform_mode0(const std::array<double, 16>& matrix)
    {
        for (const double value : matrix)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("OOC camera transform conditioning requires finite values");
            }
        }

        std::array<double, 16> normalized = matrix;
        if (normalized[15] != 0.0)
        {
            const double inverse_homogeneous = 1.0 / normalized[15];
            for (double& value : normalized)
                value *= inverse_homogeneous;
        }

        std::vector<double> rotation(9U);
        for (std::size_t row = 0U; row != 3U; ++row)
        {
            for (std::size_t column = 0U; column != 3U; ++column)
            {
                rotation[row * 3U + column] = normalized[row * 4U + column];
            }
        }
        const auto decomposition = metalign::svd_target_golub_reinsch(rotation, 3U, 3U);
        if (decomposition.u.size() != 9U || decomposition.vt.size() != 9U)
        {
            throw std::runtime_error("recovered camera-transform SVD returned an invalid shape");
        }

        std::array<double, 16> result{};
        for (std::size_t row = 0U; row != 3U; ++row)
        {
            for (std::size_t column = 0U; column != 3U; ++column)
            {
                const double first = decomposition.u[row * 3U] * decomposition.vt[column];
                const double second = decomposition.u[row * 3U + 1U] * decomposition.vt[3U + column];
                const double third = decomposition.u[row * 3U + 2U] * decomposition.vt[6U + column];
                result[row * 4U + column] = (first + second) + third;
            }
            result[row * 4U + 3U] = normalized[row * 4U + 3U];
        }
        result[15] = normalized[15];
        return result;
    }

    std::array<double, 16> invert_ooc_projective_matrix4(const std::array<double, 16>& matrix)
    {
        const auto& m = matrix;
        const double s0 = m[0] * m[5] - m[1] * m[4];
        const double s1 = m[0] * m[6] - m[2] * m[4];
        const double s2 = m[0] * m[7] - m[3] * m[4];
        const double s3 = m[1] * m[6] - m[2] * m[5];
        const double s4 = m[1] * m[7] - m[3] * m[5];
        const double s5 = m[2] * m[7] - m[3] * m[6];
        const double c0 = m[8] * m[13] - m[9] * m[12];
        const double c1 = m[8] * m[14] - m[10] * m[12];
        const double c2 = m[8] * m[15] - m[11] * m[12];
        const double c3 = m[9] * m[14] - m[10] * m[13];
        const double c4 = m[9] * m[15] - m[11] * m[13];
        const double c5 = m[10] * m[15] - m[11] * m[14];
        double determinant = s0 * c5;
        determinant = determinant - s1 * c4;
        determinant = determinant + s2 * c3;
        determinant = determinant + s3 * c2;
        determinant = determinant - s4 * c1;
        determinant = determinant + s5 * c0;
        if (determinant == 0.0 || !std::isfinite(determinant))
        {
            throw std::invalid_argument("OOC projective matrix is singular");
        }
        double inverse_determinant = 1.0;
        if (determinant != 1.0)
            inverse_determinant = 1.0 / determinant;

        std::array<double, 16> out{};
        auto store = [&](std::size_t index, double value) { out[index] = value * inverse_determinant; };
        double value = m[5] * c5;
        value = value - m[6] * c4;
        value = value + m[7] * c3;
        store(0, value);
        value = -(m[1] * c5);
        value = value + m[2] * c4;
        value = value - m[3] * c3;
        store(1, value);
        value = m[13] * s5;
        value = value - m[14] * s4;
        value = value + m[15] * s3;
        store(2, value);
        value = -(m[9] * s5);
        value = value + m[10] * s4;
        value = value - m[11] * s3;
        store(3, value);

        value = -(m[4] * c5);
        value = value + m[6] * c2;
        value = value - m[7] * c1;
        store(4, value);
        value = m[0] * c5;
        value = value - m[2] * c2;
        value = value + m[3] * c1;
        store(5, value);
        value = -(m[12] * s5);
        value = value + m[14] * s2;
        value = value - m[15] * s1;
        store(6, value);
        value = m[8] * s5;
        value = value - m[10] * s2;
        value = value + m[11] * s1;
        store(7, value);

        value = m[4] * c4;
        value = value - m[5] * c2;
        value = value + m[7] * c0;
        store(8, value);
        value = -(m[0] * c4);
        value = value + m[1] * c2;
        value = value - m[3] * c0;
        store(9, value);
        value = m[12] * s4;
        value = value - m[13] * s2;
        value = value + m[15] * s0;
        store(10, value);
        value = -(m[8] * s4);
        value = value + m[9] * s2;
        value = value - m[11] * s0;
        store(11, value);

        value = -(m[4] * c3);
        value = value + m[5] * c1;
        value = value - m[6] * c0;
        store(12, value);
        value = m[0] * c3;
        value = value - m[1] * c1;
        value = value + m[2] * c0;
        store(13, value);
        value = -(m[12] * s3);
        value = value + m[13] * s1;
        value = value - m[14] * s0;
        store(14, value);
        value = m[8] * s3;
        value = value - m[9] * s1;
        value = value + m[10] * s0;
        store(15, value);
        return out;
    }

    OocDepthRoiBoundsOutput compute_ooc_depth_roi_bounds_mode0(const OocDepthRoiBoundsMode0Input& input)
    {
        if (input.camera.width <= 0 || input.camera.height <= 0 || input.camera.focal_length == 0.0 ||
            !std::isfinite(input.camera.focal_length) || !std::isfinite(input.camera.principal_x) ||
            !std::isfinite(input.camera.principal_y) || !std::isfinite(input.camera.additive_focal) ||
            !std::isfinite(input.camera.shear))
        {
            throw std::invalid_argument("OOC depth ROI camera is outside the recovered pinhole domain");
        }
        for (const double value : input.world_to_camera)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("OOC depth ROI camera transform is not finite");
            }
        }
        for (const double value : input.region_rotation)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("OOC depth ROI region rotation is not finite");
            }
        }
        for (std::size_t axis = 0U; axis != 3U; ++axis)
        {
            if (!std::isfinite(input.region_center[axis]) || !std::isfinite(input.region_size[axis]) ||
                !(input.region_size[axis] > 0.0))
            {
                throw std::invalid_argument("OOC depth ROI region is not finite and positive");
            }
        }

        OocDepthRoiBoundsOutput result;
        std::int64_t minimum_x = std::numeric_limits<std::int64_t>::max();
        std::int64_t minimum_y = std::numeric_limits<std::int64_t>::max();
        std::int64_t maximum_x = std::numeric_limits<std::int64_t>::min();
        std::int64_t maximum_y = std::numeric_limits<std::int64_t>::min();
        std::size_t sample_index = 0U;

        for (std::int32_t outer = 0; outer != 4; ++outer)
        {
            double normalized_outer = ooc_divide_f64(static_cast<double>(outer), 3.0);
            normalized_outer = ooc_subtract_f64(normalized_outer, 0.5);
            const double region_x = ooc_multiply_f64(normalized_outer, input.region_size[0]);
            for (std::int32_t middle = 0; middle != 4; ++middle)
            {
                double normalized_middle = ooc_divide_f64(static_cast<double>(middle), 3.0);
                normalized_middle = ooc_subtract_f64(normalized_middle, 0.5);
                const double region_y = ooc_multiply_f64(normalized_middle, input.region_size[1]);
                for (std::int32_t inner = 0; inner != 4; ++inner)
                {
                    double normalized_inner = ooc_divide_f64(static_cast<double>(inner), 3.0);
                    normalized_inner = ooc_subtract_f64(normalized_inner, 0.5);
                    const double region_z = ooc_multiply_f64(normalized_inner, input.region_size[2]);

                    auto& sample = result.samples[sample_index++];
                    for (std::size_t row = 0U; row != 3U; ++row)
                    {
                        // Project rotation is stored row-major, while its columns
                        // are the oriented-box axes consumed at 0x1EC8A22.
                        double value = ooc_multiply_f64(input.region_rotation[row], region_x);
                        value = ooc_add_f64(value, ooc_multiply_f64(input.region_rotation[3U + row], region_y));
                        value = ooc_add_f64(value, ooc_multiply_f64(input.region_rotation[6U + row], region_z));
                        sample.world[row] = ooc_add_f64(value, input.region_center[row]);
                    }

                    std::array<double, 3> camera_point{};
                    for (std::size_t row = 0U; row != 3U; ++row)
                    {
                        double value = ooc_multiply_f64(input.world_to_camera[4U * row], sample.world[0]);
                        value =
                            ooc_add_f64(value, ooc_multiply_f64(input.world_to_camera[4U * row + 1U], sample.world[1]));
                        value =
                            ooc_add_f64(value, ooc_multiply_f64(input.world_to_camera[4U * row + 2U], sample.world[2]));
                        camera_point[row] = ooc_add_f64(value, input.world_to_camera[4U * row + 3U]);
                    }

                    double squared_norm = ooc_multiply_f64(camera_point[0], camera_point[0]);
                    squared_norm = ooc_add_f64(squared_norm, ooc_multiply_f64(camera_point[1], camera_point[1]));
                    squared_norm = ooc_add_f64(squared_norm, ooc_multiply_f64(camera_point[2], camera_point[2]));
                    if (!(squared_norm >= 1.4210854715202e-14) || !(camera_point[2] > 0.0))
                    {
                        continue;
                    }

                    const double inverse_z = ooc_divide_f64(1.0, camera_point[2]);
                    const double normalized_x = ooc_multiply_f64(camera_point[0], inverse_z);
                    const double normalized_y = ooc_multiply_f64(camera_point[1], inverse_z);

                    double projected_x = ooc_multiply_f64(input.camera.additive_focal, normalized_x);
                    projected_x = ooc_add_f64(projected_x, ooc_multiply_f64(input.camera.focal_length, normalized_x));
                    projected_x = ooc_add_f64(projected_x, ooc_multiply_f64(input.camera.shear, normalized_y));
                    double projected_y = ooc_multiply_f64(input.camera.focal_length, normalized_y);

                    double center_y = ooc_multiply_f64(static_cast<double>(input.camera.height), 0.5);
                    center_y = ooc_add_f64(center_y, input.camera.principal_y);
                    projected_y = ooc_add_f64(center_y, projected_y);
                    double center_x = ooc_multiply_f64(static_cast<double>(input.camera.width), 0.5);
                    center_x = ooc_add_f64(center_x, input.camera.principal_x);
                    projected_x = ooc_add_f64(center_x, projected_x);
                    sample.projected = {projected_x, projected_y};

                    constexpr double minimum_integer = static_cast<double>(std::numeric_limits<std::int64_t>::min());
                    constexpr double maximum_integer = static_cast<double>(std::numeric_limits<std::int64_t>::max());
                    if (!std::isfinite(projected_x) || !std::isfinite(projected_y) || projected_x < minimum_integer ||
                        projected_x >= maximum_integer || projected_y < minimum_integer ||
                        projected_y >= maximum_integer)
                    {
                        throw std::overflow_error("OOC depth ROI projection exceeds int64 domain");
                    }
                    const auto pixel_x = static_cast<std::int64_t>(projected_x);
                    const auto pixel_y = static_cast<std::int64_t>(projected_y);
                    minimum_x = std::min(minimum_x, pixel_x);
                    minimum_y = std::min(minimum_y, pixel_y);
                    maximum_x = std::max(maximum_x, pixel_x);
                    maximum_y = std::max(maximum_y, pixel_y);
                    sample.valid = true;
                    ++result.valid_samples;
                }
            }
        }

        if (result.valid_samples == 0U)
            return result;
        const std::int64_t margin_x = (maximum_x - minimum_x) / 20;
        const std::int64_t margin_y = (maximum_y - minimum_y) / 20;
        result.bounds.x_begin = std::max<std::int64_t>(0, minimum_x - margin_x);
        result.bounds.y_begin = std::max<std::int64_t>(0, minimum_y - margin_y);
        result.bounds.x_end = std::min<std::int64_t>(input.camera.width, maximum_x + margin_x);
        result.bounds.y_end = std::min<std::int64_t>(input.camera.height, maximum_y + margin_y);
        if (!result.bounds.valid())
            result.bounds = {};
        return result;
    }

    OocDepthRoiBoundsOutput compute_ooc_depth_roi_bounds_from_project_mode0(const OocDepthRoiProjectMode0Input& input)
    {
        OocDepthRoiBoundsMode0Input internal;
        internal.camera = input.camera;
        internal.world_to_camera =
            invert_ooc_projective_matrix4(condition_ooc_camera_transform_mode0(input.camera_to_world));
        internal.region_rotation = input.region_rotation;
        internal.region_center = input.region_center;
        internal.region_size = input.region_size;
        return compute_ooc_depth_roi_bounds_mode0(internal);
    }

    float derive_ooc_maximum_sample_scale(const std::array<double, 3>& region_size) noexcept
    {
        double maximum = region_size[0];
        if (region_size[1] > maximum)
            maximum = region_size[1];
        if (region_size[2] > maximum)
            maximum = region_size[2];
        return static_cast<float>(maximum);
    }

    OocDepthRoiBounds downsample_ooc_depth_roi_bounds(const OocDepthRoiBounds& bounds)
    {
        if (!bounds.valid())
            return {};
        return {
            bounds.x_begin / 2,
            bounds.y_begin / 2,
            bounds.x_end / 2 + bounds.x_end % 2,
            bounds.y_end / 2 + bounds.y_end % 2,
        };
    }

    OocSampleScalePyramidOutput build_ooc_sample_scale_pyramid_mode0(const OocSampleScalePyramidInput& input)
    {
        for (std::size_t level = 0U; level != input.seeds.size(); ++level)
        {
            const auto& seed = input.seeds[level];
            if (seed.width == 0U || seed.height == 0U ||
                seed.width > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
                seed.height > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
                seed.width > std::numeric_limits<std::size_t>::max() / seed.height ||
                seed.depth.size() != seed.width * seed.height)
            {
                throw std::invalid_argument("OOC sample-scale pyramid seed dimensions are invalid");
            }
            if (seed.camera.width != static_cast<std::int64_t>(seed.width) ||
                seed.camera.height != static_cast<std::int64_t>(seed.height))
            {
                throw std::invalid_argument("OOC sample-scale pyramid camera dimensions do not match");
            }
            if (level != 0U && (seed.width != input.seeds[level - 1U].width / 2U ||
                                seed.height != input.seeds[level - 1U].height / 2U))
            {
                throw std::invalid_argument("OOC sample-scale pyramid seeds are not adjacent levels");
            }
            if (!seed.gate.empty() && seed.gate.size() != seed.depth.size())
            {
                throw std::invalid_argument("OOC sample-scale pyramid seed gate size mismatch");
            }
        }
        if (!input.seeds[0].gate.empty())
        {
            throw std::invalid_argument("the recovered full-resolution OOC worker requires an empty gate");
        }

        std::size_t reduction_count = 0U;
        std::size_t reduction_width = input.seeds[2].width;
        std::size_t reduction_height = input.seeds[2].height;
        while (reduction_width > 1U && reduction_height > 1U)
        {
            reduction_width /= 2U;
            reduction_height /= 2U;
            ++reduction_count;
        }
        if (!input.reduction_gates.empty() && input.reduction_gates.size() != reduction_count)
        {
            throw std::invalid_argument("OOC sample-scale pyramid reduction gate count mismatch");
        }

        OocSampleScalePyramidOutput result;
        result.levels.reserve(3U + reduction_count);

        const auto initial_records = build_ooc_sample_scale_point_records_mode0(input.seeds[0].width,
                                                                                input.seeds[0].height,
                                                                                input.seeds[0].depth,
                                                                                input.seeds[0].camera,
                                                                                input.camera_to_record);
        const OocSampleScaleWorkerOutput initial = build_ooc_sample_scale_workspace({input.seeds[0].width,
                                                                                     input.seeds[0].height,
                                                                                     input.seeds[0].depth,
                                                                                     initial_records,
                                                                                     input.seeds[0].camera,
                                                                                     input.camera_to_record,
                                                                                     input.maximum_sample_scale,
                                                                                     input.diagonal_pixel_scale});
        result.levels.push_back(
            {input.seeds[0].width, input.seeds[0].height, initial.depth, initial.temporary, initial.sample_scale, {}});
        result.initial_level = initial.level;
        result.initial_level_weight = initial.level_weight;

        for (std::size_t finer = 0U; finer != 2U; ++finer)
        {
            const auto& seed = input.seeds[finer + 1U];
            const auto records = build_ooc_sample_scale_point_records_mode0(
                seed.width, seed.height, seed.depth, seed.camera, input.camera_to_record);
            OocPyramidFinerLevelOutput output = build_ooc_pyramid_finer_level({seed.width,
                                                                               seed.height,
                                                                               seed.depth,
                                                                               records,
                                                                               seed.camera,
                                                                               input.camera_to_record,
                                                                               seed.gate,
                                                                               input.diagonal_pixel_scale,
                                                                               input.special_invalid_depth});
            result.finer_accepted[finer] = output.accepted;
            result.finer_maximum_depth[finer] = output.maximum_depth;
            result.levels.push_back({seed.width,
                                     seed.height,
                                     std::move(output.depth),
                                     std::move(output.temporary),
                                     std::move(output.sample_scale),
                                     std::move(output.temporary_u8)});
        }

        for (std::size_t reduction = 0U; reduction != reduction_count; ++reduction)
        {
            const auto& source = result.levels.back();
            const std::span<const std::uint8_t> gate =
                input.reduction_gates.empty() ? std::span<const std::uint8_t>{} : input.reduction_gates[reduction];
            OocPyramidReductionOutput output = reduce_ooc_pyramid_level({source.width,
                                                                         source.height,
                                                                         source.depth,
                                                                         source.temporary,
                                                                         source.sample_scale,
                                                                         gate,
                                                                         input.special_invalid_depth});
            result.levels.push_back({static_cast<std::size_t>(output.width),
                                     static_cast<std::size_t>(output.height),
                                     std::move(output.depth),
                                     std::move(output.temporary),
                                     std::move(output.sample_scale),
                                     std::move(output.temporary_u8)});
        }
        return result;
    }

    OocSampleScalePyramidOutput
    build_ooc_sample_scale_pyramid_from_rois_mode0(const OocSampleScalePyramidRoiInput& input)
    {
        std::array<std::vector<float>, 3> expanded;
        OocSampleScalePyramidInput pyramid;
        for (std::size_t seed = 0U; seed != input.seeds.size(); ++seed)
        {
            expanded[seed] = expand_ooc_depth_roi(input.seeds[seed].depth);
            pyramid.seeds[seed] = {
                input.seeds[seed].depth.full_width,
                input.seeds[seed].depth.full_height,
                expanded[seed],
                input.seeds[seed].camera,
                input.seeds[seed].gate,
            };
        }
        pyramid.camera_to_record = input.camera_to_record;
        pyramid.maximum_sample_scale = input.maximum_sample_scale;
        pyramid.diagonal_pixel_scale = input.diagonal_pixel_scale;
        pyramid.special_invalid_depth = input.special_invalid_depth;
        pyramid.reduction_gates = input.reduction_gates;
        return build_ooc_sample_scale_pyramid_mode0(pyramid);
    }

    OocSampleScalePyramidOutput
    build_ooc_sample_scale_pyramid_from_region_mode0(const OocSampleScalePyramidRegionInput& input)
    {
        if (input.cameras[0].width != input.bounds.camera.width ||
            input.cameras[0].height != input.bounds.camera.height ||
            input.cameras[0].focal_length != input.bounds.camera.focal_length ||
            input.cameras[0].principal_x != input.bounds.camera.principal_x ||
            input.cameras[0].principal_y != input.bounds.camera.principal_y ||
            input.cameras[0].additive_focal != input.bounds.camera.additive_focal ||
            input.cameras[0].shear != input.bounds.camera.shear)
        {
            throw std::invalid_argument("OOC autonomous ROI camera does not match the first seed");
        }

        const auto projection = compute_ooc_depth_roi_bounds_from_project_mode0(input.bounds);
        if (!projection.bounds.valid())
        {
            throw std::invalid_argument("OOC autonomous ROI does not intersect the rectified camera");
        }
        std::array<OocDepthRoiBounds, 3> bounds{
            projection.bounds,
            downsample_ooc_depth_roi_bounds(projection.bounds),
            {},
        };
        bounds[2] = downsample_ooc_depth_roi_bounds(bounds[1]);

        OocSampleScalePyramidRoiInput roi;
        for (std::size_t seed = 0U; seed != input.cameras.size(); ++seed)
        {
            const auto& camera = input.cameras[seed];
            const auto& current = bounds[seed];
            if (!current.valid() || camera.width <= 0 || camera.height <= 0)
            {
                throw std::invalid_argument("OOC autonomous ROI finer-level bounds are invalid");
            }
            roi.seeds[seed] = {
                {
                    static_cast<std::size_t>(camera.width),
                    static_cast<std::size_t>(camera.height),
                    static_cast<std::size_t>(current.x_begin),
                    static_cast<std::size_t>(current.y_begin),
                    static_cast<std::size_t>(current.x_end),
                    static_cast<std::size_t>(current.y_end),
                    input.roi_depths[seed],
                },
                camera,
                input.gates[seed],
            };
        }
        roi.camera_to_record = input.camera_to_record;
        roi.maximum_sample_scale = input.maximum_sample_scale;
        roi.diagonal_pixel_scale = input.diagonal_pixel_scale;
        roi.special_invalid_depth = input.special_invalid_depth;
        roi.reduction_gates = input.reduction_gates;
        return build_ooc_sample_scale_pyramid_from_rois_mode0(roi);
    }

    OocSampleScalePyramidOutput
    build_ooc_sample_scale_pyramid_from_voted_depths_mode0(const OocSampleScalePyramidVotedDepthInput& input)
    {
        const auto projection = compute_ooc_depth_roi_bounds_from_project_mode0(input.bounds);
        if (!projection.bounds.valid())
        {
            throw std::invalid_argument("OOC voted-depth ROI does not intersect the rectified camera");
        }
        std::array<OocDepthRoiBounds, 3> bounds{
            projection.bounds,
            downsample_ooc_depth_roi_bounds(projection.bounds),
            {},
        };
        bounds[2] = downsample_ooc_depth_roi_bounds(bounds[1]);

        std::array<std::vector<float>, 3> roi_depths;
        OocSampleScalePyramidRegionInput region;
        region.bounds = input.bounds;
        region.cameras = input.cameras;
        region.gates = input.gates;
        // The six target work items carry the raw project c2w bytes here. This is
        // intentionally distinct from the SVD-conditioned transform used above
        // for ROI projection.
        region.camera_to_record = input.bounds.camera_to_world;
        region.maximum_sample_scale = derive_ooc_maximum_sample_scale(input.bounds.region_size);
        region.diagonal_pixel_scale = input.diagonal_pixel_scale;
        region.special_invalid_depth = input.special_invalid_depth;
        region.reduction_gates = input.reduction_gates;
        for (std::size_t level = 0U; level != input.voted_depths.size(); ++level)
        {
            const auto& camera = input.cameras[level];
            if (camera.width <= 0 || camera.height <= 0)
            {
                throw std::invalid_argument("OOC voted-depth camera dimensions are invalid");
            }
            roi_depths[level] = extract_ooc_depth_roi(static_cast<std::size_t>(camera.width),
                                                      static_cast<std::size_t>(camera.height),
                                                      bounds[level],
                                                      input.voted_depths[level]);
            region.roi_depths[level] = roi_depths[level];
        }
        return build_ooc_sample_scale_pyramid_from_region_mode0(region);
    }

    OocDepthRoiProjectMode0Input
    make_ooc_depth_roi_project_mode0_input(const Scene& scene, std::size_t camera_index, std::uint32_t depth_downscale)
    {
        if (camera_index >= scene.cameras.size())
        {
            throw std::out_of_range("OOC Scene camera index is out of range");
        }
        if (!scene.region.specified || depth_downscale == 0U)
        {
            throw std::invalid_argument("OOC Scene adapter requires a reconstruction region and nonzero "
                                        "depth downscale");
        }
        const Camera& camera = scene.cameras[camera_index];
        if (!camera.aligned || camera.image.width == 0U || camera.image.height == 0U ||
            camera.image.width % depth_downscale != 0U || camera.image.height % depth_downscale != 0U)
        {
            throw std::invalid_argument("OOC Scene camera is unaligned or not divisible by the depth "
                                        "downscale");
        }
        const std::size_t width = camera.image.width / depth_downscale;
        const std::size_t height = camera.image.height / depth_downscale;
        if (width > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
            height > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        {
            throw std::overflow_error("OOC Scene depth dimensions overflow int64");
        }
        const double scale = static_cast<double>(depth_downscale);
        OocDepthRoiProjectMode0Input result;
        result.camera = {
            static_cast<std::int64_t>(width),
            static_cast<std::int64_t>(height),
            camera.model.f / scale,
            camera.model.cx / scale - static_cast<double>(width) * 0.5,
            camera.model.cy / scale - static_cast<double>(height) * 0.5,
            0.0,
            0.0,
        };
        result.camera_to_world = {
            camera.pose.rotation(0, 0),
            camera.pose.rotation(1, 0),
            camera.pose.rotation(2, 0),
            camera.center.x,
            camera.pose.rotation(0, 1),
            camera.pose.rotation(1, 1),
            camera.pose.rotation(2, 1),
            camera.center.y,
            camera.pose.rotation(0, 2),
            camera.pose.rotation(1, 2),
            camera.pose.rotation(2, 2),
            camera.center.z,
            0.0,
            0.0,
            0.0,
            1.0,
        };
        result.region_rotation = scene.region.rotation;
        result.region_center = {scene.region.center.x, scene.region.center.y, scene.region.center.z};
        result.region_size = {scene.region.size.x, scene.region.size.y, scene.region.size.z};
        return result;
    }

    OocSampleScalePyramidOutput build_ooc_sample_scale_pyramid_from_scene_voted_depths_mode0(
        const Scene& scene,
        std::size_t camera_index,
        const std::array<std::span<const float>, 3>& voted_depths,
        std::uint32_t depth_downscale,
        bool captured_diagonal_pixel_scale)
    {
        if (depth_downscale > std::numeric_limits<std::uint32_t>::max() / 4U)
        {
            throw std::overflow_error("OOC Scene pyramid downscale overflows");
        }
        OocSampleScalePyramidVotedDepthInput input;
        input.bounds = make_ooc_depth_roi_project_mode0_input(scene, camera_index, depth_downscale);
        for (std::size_t level = 0U; level != voted_depths.size(); ++level)
        {
            const std::uint32_t level_downscale = depth_downscale << static_cast<std::uint32_t>(level);
            const OocDepthRoiProjectMode0Input level_input =
                make_ooc_depth_roi_project_mode0_input(scene, camera_index, level_downscale);
            input.cameras[level] = level_input.camera;
            input.voted_depths[level] = voted_depths[level];
        }
        input.diagonal_pixel_scale = captured_diagonal_pixel_scale;
        // sub_17E7160 passes BuildModel settings+48 as the worker's invalid-depth
        // sentinel selector. Controlled public-API runs identify that byte as
        // volumetric_masks. This Scene bridge has no mask/gate input and is the
        // exact volumetric_masks=false path; gated replay remains available via
        // build_ooc_sample_scale_pyramid_from_voted_depths_mode0().
        input.special_invalid_depth = false;
        return build_ooc_sample_scale_pyramid_from_voted_depths_mode0(input);
    }

    OocSampleScalePyramidOutput
    build_ooc_sample_scale_pyramid_from_scene_voted_depths_mode0(const Scene& scene,
                                                                 std::size_t camera_index,
                                                                 const std::array<std::vector<float>, 3>& voted_depths,
                                                                 std::uint32_t depth_downscale,
                                                                 bool captured_diagonal_pixel_scale)
    {
        const std::array<std::span<const float>, 3> views{voted_depths[0], voted_depths[1], voted_depths[2]};
        return build_ooc_sample_scale_pyramid_from_scene_voted_depths_mode0(
            scene, camera_index, views, depth_downscale, captured_diagonal_pixel_scale);
    }

    OocSceneVotedDepthBundleMode0Output
    build_ooc_pyramid_bundle_from_scene_voted_depths_mode0(const Scene& scene,
                                                           std::span<const OocSceneVotedDepthMode0View> cameras,
                                                           std::uint32_t depth_downscale,
                                                           std::size_t workitem_size_cameras,
                                                           std::size_t max_workgroup_size)
    {
        OocSceneVotedDepthBundleMode0Output output;
        output.camera_groups =
            partition_ooc_pyramid_camera_groups(cameras.size(), workitem_size_cameras, max_workgroup_size);
        output.items.reserve(cameras.size());
        output.pyramids.reserve(cameras.size());
        std::unordered_set<std::size_t> scene_camera_indices;
        scene_camera_indices.reserve(cameras.size());
        for (const auto& camera : cameras)
        {
            if (camera.camera_index >= scene.cameras.size() || !scene_camera_indices.insert(camera.camera_index).second)
            {
                throw std::invalid_argument("OOC Scene voting bundle camera index is invalid or duplicated");
            }
            const std::size_t stable_id = scene.cameras[camera.camera_index].index;
            if (stable_id > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::overflow_error("OOC Scene voting bundle camera ID exceeds uint32");
            }
            OocDepthRoiProjectMode0Input project =
                make_ooc_depth_roi_project_mode0_input(scene, camera.camera_index, depth_downscale);
            output.items.push_back({static_cast<std::uint32_t>(stable_id), std::move(project)});
            output.pyramids.push_back(
                build_ooc_sample_scale_pyramid_from_scene_voted_depths_mode0(scene,
                                                                             camera.camera_index,
                                                                             camera.voted_depths,
                                                                             depth_downscale,
                                                                             camera.captured_diagonal_pixel_scale));
        }

        std::vector<OocPyramidBundleMode0Group> groups;
        groups.reserve(output.camera_groups.size());
        for (const auto& range : output.camera_groups)
        {
            groups.push_back(
                {std::span<const OocPyramidBundleMode0Item>(output.items).subspan(range.begin_index, range.item_count),
                 std::span<const OocSampleScalePyramidOutput>(output.pyramids)
                     .subspan(range.begin_index, range.item_count)});
        }
        output.bundle = serialize_ooc_pyramid_bundle_mode0(groups);
        return output;
    }

    OocPyramidRegistryRawRecord make_ooc_pyramid_registry_record_mode0(const OocPyramidRegistryMode0Input& input,
                                                                       const OocSampleScalePyramidOutput& pyramid)
    {
        if (input.project.camera.width <= 0 || input.project.camera.height <= 0 || pyramid.levels.empty() ||
            pyramid.levels.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::invalid_argument("OOC registry mode-0 camera or pyramid dimensions are invalid");
        }
        const auto& level0 = pyramid.levels.front();
        const std::size_t width = static_cast<std::size_t>(input.project.camera.width);
        const std::size_t height = static_cast<std::size_t>(input.project.camera.height);
        if (width > std::numeric_limits<std::size_t>::max() / height || level0.width != width ||
            level0.height != height || level0.depth.size() != width * height ||
            pyramid.initial_level.size() != level0.depth.size() ||
            pyramid.initial_level_weight.size() != level0.depth.size())
        {
            throw std::invalid_argument("OOC registry mode-0 pyramid does not match the camera");
        }
        const OocDepthRoiBoundsOutput projected = compute_ooc_depth_roi_bounds_from_project_mode0(input.project);
        if (!projected.bounds.valid())
        {
            throw std::invalid_argument("OOC registry mode-0 camera has no valid region ROI");
        }
        const OocPyramidRegistryLevelMetadata metadata =
            make_ooc_pyramid_registry_level_metadata(level0.depth, pyramid.initial_level, pyramid.initial_level_weight);

        OocPyramidRegistryRawRecord record{};
        const auto put_unsigned = [&](std::size_t offset, std::uint64_t value, std::size_t size)
        {
            if (offset > record.size() || size > record.size() - offset)
            {
                throw std::logic_error("OOC registry field exceeds record");
            }
            for (std::size_t index = 0U; index != size; ++index)
            {
                record[offset + index] = static_cast<std::byte>(value >> (8U * index));
            }
        };
        const auto put_u32 = [&](std::size_t offset, std::uint32_t value)
        { put_unsigned(offset, value, sizeof(value)); };
        const auto put_u64 = [&](std::size_t offset, std::uint64_t value)
        { put_unsigned(offset, value, sizeof(value)); };
        const auto put_float = [&](std::size_t offset, float value)
        { put_u32(offset, std::bit_cast<std::uint32_t>(value)); };
        const auto put_double = [&](std::size_t offset, double value)
        { put_u64(offset, std::bit_cast<std::uint64_t>(value)); };

        put_u32(0x000U, 6U);
        put_u32(0x004U, input.camera_id);
        const auto conditioned = condition_ooc_camera_transform_mode0(input.project.camera_to_world);
        for (std::size_t index = 0U; index != conditioned.size(); ++index)
        {
            put_double(0x008U + index * sizeof(double), conditioned[index]);
        }

        // sub_27CB080 initializes the embedded normal perspective calibration.
        put_u64(0x088U, 1U);
        put_u64(0x090U, static_cast<std::uint64_t>(width));
        put_u64(0x098U, static_cast<std::uint64_t>(height));
        put_double(0x0A0U, input.project.camera.focal_length);
        put_double(0x0A8U, input.project.camera.principal_x);
        put_double(0x0B0U, input.project.camera.principal_y);
        put_double(0x0B8U, input.project.camera.additive_focal);
        put_double(0x0C0U, input.project.camera.shear);
        put_u32(0x12CU, 0x41CDCD65U);

        // The recovered production path always consumes three adjacent voting
        // depth products.  sub_1EC4970 supplies the generated level count.
        put_u32(0x400U, 3U);
        put_u32(0x404U, static_cast<std::uint32_t>(pyramid.levels.size()));
        constexpr float invalid_depth = std::bit_cast<float>(0xD3800000U);
        std::uint64_t valid_count = 0U;
        for (const float depth : level0.depth)
        {
            if (depth != 0.0F && depth != invalid_depth)
                ++valid_count;
        }
        put_u64(0x408U, valid_count);
        put_u32(0x410U, 1U);
        put_float(0x414U, std::numeric_limits<float>::max());
        put_u64(0x418U, static_cast<std::uint64_t>(projected.bounds.x_begin));
        put_u64(0x420U, static_cast<std::uint64_t>(projected.bounds.y_begin));
        put_u64(0x428U, static_cast<std::uint64_t>(projected.bounds.x_end));
        put_u64(0x430U, static_cast<std::uint64_t>(projected.bounds.y_end));
        put_u32(0x438U, metadata.maximum_level);
        for (std::size_t index = 0U; index != metadata.selected_level_histogram.size(); ++index)
        {
            put_float(0x43CU + index * sizeof(float), metadata.selected_level_histogram[index]);
        }
        put_u64(0x4BCU, input.payload.shard_index);
        put_u64(0x4C4U, input.payload.byte_offset);
        put_u64(0x4CCU, input.payload.byte_size);
        return record;
    }

    OocPyramidRegistryLevelMetadata make_ooc_pyramid_registry_level_metadata(const OocSampleScaleWorkerOutput& output)
    {
        return make_ooc_pyramid_registry_level_metadata(output.depth, output.level, output.level_weight);
    }

    std::vector<OocDepthCandidate> build_ooc_depth_candidate_workspace(const OocDepthCandidateInput& input)
    {
        const std::size_t pixels = input.width * input.height;
        if (input.depth.size() != pixels || input.sample_scale.size() != pixels)
        {
            throw std::runtime_error("OOC depth candidate plane size mismatch");
        }
        if (!(input.focal_length > 0.0))
        {
            throw std::runtime_error("OOC depth candidate focal length is invalid");
        }

        // sub_97A4E0 uses the six upper and six lower 2x2 minors below. This is
        // intentionally a full projective inverse: replacing it by a rigid
        // transpose matched one early capture only because that matrix happened to
        // round compatibly, and diverged on the next camera.
        const auto invert4 = [](const std::array<double, 16>& matrix)
        {
            const auto& m = matrix;
            const double s0 = m[0] * m[5] - m[1] * m[4];
            const double s1 = m[0] * m[6] - m[2] * m[4];
            const double s2 = m[0] * m[7] - m[3] * m[4];
            const double s3 = m[1] * m[6] - m[2] * m[5];
            const double s4 = m[1] * m[7] - m[3] * m[5];
            const double s5 = m[2] * m[7] - m[3] * m[6];
            const double c0 = m[8] * m[13] - m[9] * m[12];
            const double c1 = m[8] * m[14] - m[10] * m[12];
            const double c2 = m[8] * m[15] - m[11] * m[12];
            const double c3 = m[9] * m[14] - m[10] * m[13];
            const double c4 = m[9] * m[15] - m[11] * m[13];
            const double c5 = m[10] * m[15] - m[11] * m[14];
            double determinant = s0 * c5;
            determinant = determinant - s1 * c4;
            determinant = determinant + s2 * c3;
            determinant = determinant + s3 * c2;
            determinant = determinant - s4 * c1;
            determinant = determinant + s5 * c0;
            double inverse_determinant = 1.0;
            if (determinant != 1.0)
                inverse_determinant = 1.0 / determinant;

            std::array<double, 16> out{};
            auto store = [&](std::size_t index, double value) { out[index] = value * inverse_determinant; };
            double value = m[5] * c5;
            value = value - m[6] * c4;
            value = value + m[7] * c3;
            store(0, value);
            value = -(m[1] * c5);
            value = value + m[2] * c4;
            value = value - m[3] * c3;
            store(1, value);
            value = m[13] * s5;
            value = value - m[14] * s4;
            value = value + m[15] * s3;
            store(2, value);
            value = -(m[9] * s5);
            value = value + m[10] * s4;
            value = value - m[11] * s3;
            store(3, value);

            value = -(m[4] * c5);
            value = value + m[6] * c2;
            value = value - m[7] * c1;
            store(4, value);
            value = m[0] * c5;
            value = value - m[2] * c2;
            value = value + m[3] * c1;
            store(5, value);
            value = -(m[12] * s5);
            value = value + m[14] * s2;
            value = value - m[15] * s1;
            store(6, value);
            value = m[8] * s5;
            value = value - m[10] * s2;
            value = value + m[11] * s1;
            store(7, value);

            value = m[4] * c4;
            value = value - m[5] * c2;
            value = value + m[7] * c0;
            store(8, value);
            value = -(m[0] * c4);
            value = value + m[1] * c2;
            value = value - m[3] * c0;
            store(9, value);
            value = m[12] * s4;
            value = value - m[13] * s2;
            value = value + m[15] * s0;
            store(10, value);
            value = -(m[8] * s4);
            value = value + m[9] * s2;
            value = value - m[11] * s0;
            store(11, value);

            value = -(m[4] * c3);
            value = value + m[5] * c1;
            value = value - m[6] * c0;
            store(12, value);
            value = m[0] * c3;
            value = value - m[1] * c1;
            value = value + m[2] * c0;
            store(13, value);
            value = -(m[12] * s3);
            value = value + m[13] * s1;
            value = value - m[14] * s0;
            store(14, value);
            value = m[8] * s3;
            value = value - m[9] * s1;
            value = value + m[10] * s0;
            store(15, value);
            return out;
        };

        const std::array<double, 16> world_to_root = invert4(input.root_to_world);
        std::vector<OocDepthCandidate> result(pixels);
        const auto& camera = input.camera_to_world;
        for (std::size_t y = 0; y < input.height; ++y)
        {
            for (std::size_t x = 0; x < input.width; ++x)
            {
                const std::size_t index = y * input.width + x;
                const float depth = input.depth[index];
                const float scale = input.sample_scale[index];
                if (!(depth > 0.0F) || !std::isfinite(depth) || !(scale > 0.0F))
                {
                    continue;
                }

                double local_y = static_cast<double>(y) + 0.5;
                local_y = local_y - static_cast<double>(input.height) * 0.5;
                const double centered_y = input.principal_y + 0.5 - static_cast<double>(input.height) * 0.5;
                local_y = local_y - centered_y;
                local_y = local_y / input.focal_length;
                local_y = local_y * static_cast<double>(depth);
                double local_x = static_cast<double>(x) + 0.5;
                local_x = local_x - static_cast<double>(input.width) * 0.5;
                const double centered_x = input.principal_x + 0.5 - static_cast<double>(input.width) * 0.5;
                local_x = local_x - centered_x;
                local_x = local_x / input.focal_length;
                local_x = local_x * static_cast<double>(depth);
                const double local_z = static_cast<double>(depth);
                double camera_denominator = camera[12] * local_x;
                camera_denominator = camera_denominator + camera[13] * local_y;
                camera_denominator = camera_denominator + camera[14] * local_z;
                camera_denominator = camera_denominator + camera[15];
                const auto camera_coordinate = [&](std::size_t row)
                {
                    double value = camera[row * 4U] * local_x;
                    value = value + camera[row * 4U + 1U] * local_y;
                    value = value + camera[row * 4U + 2U] * local_z;
                    value = value + camera[row * 4U + 3U];
                    return value / camera_denominator;
                };
                const double world_x = camera_coordinate(0U);
                const double world_y = camera_coordinate(1U);
                const double world_z = camera_coordinate(2U);

                double root_denominator = world_to_root[12] * world_x;
                root_denominator = root_denominator + world_to_root[13] * world_y;
                root_denominator = root_denominator + world_to_root[14] * world_z;
                root_denominator = root_denominator + world_to_root[15];
                const auto root_coordinate = [&](std::size_t row)
                {
                    double value = world_to_root[row * 4U] * world_x;
                    value = value + world_to_root[row * 4U + 1U] * world_y;
                    value = value + world_to_root[row * 4U + 2U] * world_z;
                    value = value + world_to_root[row * 4U + 3U];
                    value = value / root_denominator;
                    value = value + input.root_extent[row] * 0.5;
                    return value;
                };
                const double root_x = root_coordinate(0U);
                const double root_y = root_coordinate(1U);
                const double root_z = root_coordinate(2U);
                if (root_x < 0.0 || root_x > input.root_extent[0] || root_y < 0.0 || root_y > input.root_extent[1] ||
                    root_z < 0.0 || root_z > input.root_extent[2])
                {
                    continue;
                }
                result[index] = {{static_cast<float>(root_x), static_cast<float>(root_y), static_cast<float>(root_z)},
                                 scale};
            }
        }
        return result;
    }

    void apply_ooc_depth_candidate_planarity(std::span<OocDepthCandidate> workspace,
                                             std::size_t width,
                                             std::size_t height,
                                             float threshold)
    {
        if (workspace.size() != width * height)
        {
            throw std::runtime_error("OOC planarity workspace size mismatch");
        }
        auto candidate = [&](std::ptrdiff_t x, std::ptrdiff_t y) -> const OocDepthCandidate*
        {
            if (x < 0 || y < 0 || x >= static_cast<std::ptrdiff_t>(width) || y >= static_cast<std::ptrdiff_t>(height))
            {
                return nullptr;
            }
            const auto& value = workspace[static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)];
            return value.scale != 0.0F ? &value : nullptr;
        };
        auto difference = [](const OocDepthCandidate& a, const OocDepthCandidate& b)
        {
            return std::array<float, 3>{
                a.position[0] - b.position[0], a.position[1] - b.position[1], a.position[2] - b.position[2]};
        };

        std::vector<std::uint64_t> selected((workspace.size() + 63U) / 64U, 0U);
        for (std::size_t y = 0; y < height; ++y)
        {
            for (std::size_t x = 0; x < width; ++x)
            {
                const std::size_t index = y * width + x;
                const auto& center = workspace[index];
                if (center.scale == 0.0F)
                    continue;
                const auto* up = candidate(static_cast<std::ptrdiff_t>(x), static_cast<std::ptrdiff_t>(y) - 1);
                const auto* down = candidate(static_cast<std::ptrdiff_t>(x), static_cast<std::ptrdiff_t>(y) + 1);
                const auto* left = candidate(static_cast<std::ptrdiff_t>(x) - 1, static_cast<std::ptrdiff_t>(y));
                const auto* right = candidate(static_cast<std::ptrdiff_t>(x) + 1, static_cast<std::ptrdiff_t>(y));
                std::array<float, 3> vertical{};
                std::array<float, 3> horizontal{};
                if (up && down)
                    vertical = difference(*down, *up);
                else if (up)
                    vertical = difference(center, *up);
                else if (down)
                    vertical = difference(*down, center);
                else
                    continue;
                if (left && right)
                    horizontal = difference(*right, *left);
                else if (left)
                    horizontal = difference(center, *left);
                else if (right)
                    horizontal = difference(*right, center);
                else
                    continue;

                float normal_x = vertical[1] * horizontal[2] - vertical[2] * horizontal[1];
                float normal_y = vertical[2] * horizontal[0] - vertical[0] * horizontal[2];
                float normal_z = vertical[0] * horizontal[1] - vertical[1] * horizontal[0];
                const float normal_length = std::sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
                if (!(normal_length >= 1.0e-20F))
                    continue;
                const float inverse_length = 1.0F / normal_length;
                normal_x *= inverse_length;
                normal_y *= inverse_length;
                normal_z *= inverse_length;

                float sum_x = center.position[0];
                float sum_y = center.position[1];
                float sum_z = center.position[2];
                float centroid_count = 1.0F;
                float distance_sum = 0.0F;
                float distance_count = 0.0F;
                for (const auto* axial : {left, right, up, down})
                {
                    if (!axial)
                        continue;
                    sum_x += axial->position[0];
                    sum_y += axial->position[1];
                    sum_z += axial->position[2];
                    centroid_count += 1.0F;
                    const float dx = axial->position[0] - center.position[0];
                    const float dy = axial->position[1] - center.position[1];
                    const float dz = axial->position[2] - center.position[2];
                    distance_sum += std::sqrt(dx * dx + dy * dy + dz * dz);
                    distance_count += 1.0F;
                }
                if (distance_count == 0.0F)
                    continue;
                const float centroid_x = sum_x / centroid_count;
                const float centroid_y = sum_y / centroid_count;
                const float centroid_z = sum_z / centroid_count;
                const float mean_distance = distance_sum / distance_count;

                float largest = 0.0F;
                float second_largest = 0.0F;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const auto* point =
                            candidate(static_cast<std::ptrdiff_t>(x) + dx, static_cast<std::ptrdiff_t>(y) + dy);
                        if (!point)
                            continue;
                        const float px = point->position[0] - centroid_x;
                        const float py = point->position[1] - centroid_y;
                        const float pz = point->position[2] - centroid_z;
                        const float residual = std::abs(px * normal_x + py * normal_y + pz * normal_z);
                        if (residual >= largest)
                        {
                            second_largest = largest;
                            largest = residual;
                        }
                        else if (residual > second_largest)
                        {
                            second_largest = residual;
                        }
                    }
                }
                if (threshold > second_largest / mean_distance)
                {
                    selected[index / 64U] |= 1ULL << (index & 63U);
                }
            }
        }
        for (std::size_t index = 0; index < workspace.size(); ++index)
        {
            if ((selected[index / 64U] & (1ULL << (index & 63U))) != 0U)
            {
                workspace[index].scale += workspace[index].scale;
            }
        }
    }

    std::vector<OocDepthCandidate> compact_ooc_depth_candidates(std::span<const OocDepthCandidate> workspace)
    {
        std::vector<OocDepthCandidate> result;
        for (const auto& candidate : workspace)
        {
            if (candidate.scale > 0.0F)
                result.push_back(candidate);
        }
        return result;
    }

    float aggregate_ooc_marching_scalar(std::span<const float> scalars, std::span<const float> weights)
    {
        if (scalars.size() != weights.size())
        {
            throw std::runtime_error("OOC marching scalar/weight size mismatch");
        }
        if (scalars.size() > 8U)
        {
            throw std::runtime_error("OOC marching scalar neighborhood exceeds eight");
        }

        std::array<float, 8> weighted_scalars{};
        std::array<float, 8> normalized_weights{};
        std::copy(scalars.begin(), scalars.end(), weighted_scalars.begin());
        std::copy(weights.begin(), weights.end(), normalized_weights.begin());

        if (!weights.empty())
        {
            float minimum_weight = std::numeric_limits<float>::max();
            for (const float weight : weights)
            {
                // MINSS(weight, minimum_weight) retains the second operand for an
                // unordered or equal comparison. This ordered test has the same
                // effect for the target's positive finite weights and for NaNs.
                if (weight < minimum_weight)
                    minimum_weight = weight;
            }
            for (std::size_t index = 0; index < weights.size(); ++index)
            {
                normalized_weights[index] = normalized_weights[index] / minimum_weight;
                weighted_scalars[index] = weighted_scalars[index] * normalized_weights[index];
            }
        }

        const auto insertion_sort = [](float* first, float* last)
        {
            if (first == last)
                return;
            for (float* current = first + 1; current != last; ++current)
            {
                const float value = *current;
                if (*first > value)
                {
                    std::move_backward(first, current, current + 1);
                    *first = value;
                    continue;
                }
                float* position = current;
                while (*(position - 1) > value)
                {
                    *position = *(position - 1);
                    --position;
                }
                *position = value;
            }
        };
        insertion_sort(weighted_scalars.data(), weighted_scalars.data() + scalars.size());
        insertion_sort(normalized_weights.data(), normalized_weights.data() + weights.size());

        float scalar_sum = 0.0F;
        float weight_sum = 0.0F;
        for (std::size_t index = 0; index < scalars.size(); ++index)
        {
            scalar_sum = scalar_sum + weighted_scalars[index];
            weight_sum = weight_sum + normalized_weights[index];
        }
        const float result = scalar_sum / weight_sum;
        return result == 0.0F ? 1.0e-6F : result;
    }

    namespace
    {

        OocMarchingActiveCells
        build_ooc_marching_active_cells_impl(std::span<const OocMarchingExtractNode> nodes,
                                             const std::optional<OocMarchingGridBounds> bounds,
                                             const std::optional<std::uint32_t> validated_maximum_level)
        {
            if (nodes.empty())
            {
                throw std::runtime_error("OOC marching tree is empty");
            }
            if (((nodes.size() - 1U) & 7U) != 0U)
            {
                throw std::runtime_error("OOC marching tree is not root plus child octets");
            }

            const auto child_start = [&](std::size_t node) -> std::size_t
            {
                const auto group = nodes[node].child_group;
                if (group == std::numeric_limits<std::uint32_t>::max())
                    return std::numeric_limits<std::size_t>::max();
                const std::uint64_t start = 1ULL + 8ULL * group;
                if (start + 7ULL >= nodes.size())
                {
                    throw std::runtime_error("OOC marching child group is truncated");
                }
                return static_cast<std::size_t>(start);
            };

            std::uint32_t maximum_level = validated_maximum_level.value_or(0U);
            if (!validated_maximum_level.has_value())
            {
                std::vector<std::uint8_t> reached(nodes.size(), 0U);
                const auto inspect_tree = [&](auto&& self, std::size_t node, std::uint32_t level) -> void
                {
                    if (reached[node]++)
                    {
                        throw std::runtime_error("OOC marching tree shares or cycles a node");
                    }
                    maximum_level = std::max(maximum_level, level);
                    const auto first = child_start(node);
                    if (first == std::numeric_limits<std::size_t>::max())
                        return;
                    for (std::size_t slot = 0U; slot != 8U; ++slot)
                        self(self, first + slot, level + 1U);
                };
                inspect_tree(inspect_tree, 0U, 0U);
                if (std::find(reached.begin(), reached.end(), 0U) != reached.end())
                {
                    throw std::runtime_error("OOC marching tree contains unreachable nodes");
                }
            }
            if (bounds.has_value() && bounds->level != maximum_level)
            {
                throw std::runtime_error("OOC marching bounds must use the tree maximum level");
            }

            const auto overlaps_bounds =
                [&](const std::uint32_t x, const std::uint32_t y, const std::uint32_t z, const std::uint32_t level)
            {
                if (!bounds.has_value())
                    return true;
                const std::uint32_t shift = maximum_level - level;
                const std::uint32_t width = std::uint32_t{1} << shift;
                const std::uint32_t x0 = x << shift;
                const std::uint32_t y0 = y << shift;
                const std::uint32_t z0 = z << shift;
                return bounds->minimum[0] < x0 + width && bounds->minimum[1] < y0 + width &&
                       bounds->minimum[2] < z0 + width && bounds->maximum[0] > x0 && bounds->maximum[1] > y0 &&
                       bounds->maximum[2] > z0;
            };

            using Neighborhood = std::array<std::int64_t, 27>;
            OocMarchingActiveCells output;
            output.bits.assign((nodes.size() + 63U) / 64U, 0ULL);
            output.maximum_level = maximum_level;
            std::vector<std::uint64_t> neighbor_closure(output.bits.size(), 0ULL);

            const auto bit_is_set = [](const std::vector<std::uint64_t>& bits, std::size_t node)
            { return (bits[node >> 6U] & (1ULL << (node & 63U))) != 0U; };
            const auto set_bit = [](std::vector<std::uint64_t>& bits, std::size_t node)
            { bits[node >> 6U] |= 1ULL << (node & 63U); };
            const auto add_active = [&](std::size_t node)
            {
                if (bit_is_set(output.bits, node))
                    return;
                set_bit(output.bits, node);
                output.entries.push_back({node, output.entries.size()});
            };

            const auto make_child_neighborhood = [&](const Neighborhood& parent, std::size_t child_slot)
            {
                Neighborhood child{};
                child.fill(-1);
                for (std::size_t position = 0U; position != 27U; ++position)
                {
                    const std::size_t table = 27U * child_slot + position;
                    const std::int64_t parent_node = parent[kOocMarchNeighborIndex[table]];
                    if (parent_node < 0)
                        continue;
                    const auto first = child_start(static_cast<std::size_t>(parent_node));
                    if (first == std::numeric_limits<std::size_t>::max())
                        continue;
                    child[position] = static_cast<std::int64_t>(first + kOocMarchChildSlot[table]);
                }
                return child;
            };

            struct Sample
            {
                std::size_t node{};
                float weight{};
            };
            const auto sample_node = [&](const Neighborhood& neighborhood,
                                         std::size_t child_slot,
                                         std::size_t corner,
                                         std::size_t selector) -> std::optional<Sample>
            {
                std::array<std::size_t, 3> coordinate{1U, 1U, 1U};
                for (std::size_t axis = 0U; axis != 3U; ++axis)
                {
                    if ((selector & (1U << axis)) != 0U)
                        coordinate[axis] = 2U * ((corner >> axis) & 1U);
                }
                const std::size_t table = 27U * child_slot + coordinate[0] + 3U * coordinate[1] + 9U * coordinate[2];
                std::int64_t node = neighborhood[kOocMarchNeighborIndex[table]];
                if (node < 0)
                    return std::nullopt;

                float scale = 2.0F;
                auto first = child_start(static_cast<std::size_t>(node));
                if (first != std::numeric_limits<std::size_t>::max())
                {
                    node = static_cast<std::int64_t>(first + kOocMarchChildSlot[table]);
                    scale = 1.0F;
                    first = child_start(static_cast<std::size_t>(node));
                    if (first != std::numeric_limits<std::size_t>::max())
                    {
                        node = static_cast<std::int64_t>(first);
                        if (((corner & 1U) != 0U) != (coordinate[0] != 1U))
                            ++node;
                        if (((corner & 2U) != 0U) != (coordinate[1] != 1U))
                            node += 2;
                        if (((corner & 4U) != 0U) != (coordinate[2] != 1U))
                            node += 4;
                        scale = 0.5F;
                    }
                }
                return Sample{static_cast<std::size_t>(node), 1.0F / scale};
            };

            const auto classify_corner =
                [&](const Neighborhood& neighborhood, std::size_t child_slot, std::size_t corner)
            {
                int count = 0;
                int positive = 0;
                int negative = 0;
                for (std::size_t selector = 0U; selector != 8U; ++selector)
                {
                    const auto sample = sample_node(neighborhood, child_slot, corner, selector);
                    if (!sample)
                        continue;
                    const float scalar = nodes[sample->node].scalar;
                    if (scalar > 0.0F)
                        ++positive;
                    else if (scalar < 0.0F)
                        ++negative;
                    ++count;
                }
                if (count == positive)
                    return 1;
                if (count == negative)
                    return -1;
                return 0;
            };

            const auto aggregate_corner =
                [&](const Neighborhood& neighborhood, std::size_t child_slot, std::size_t corner)
            {
                std::array<float, 8> scalars{};
                std::array<float, 8> weights{};
                std::size_t count = 0U;
                for (std::size_t selector = 0U; selector != 8U; ++selector)
                {
                    const auto sample = sample_node(neighborhood, child_slot, corner, selector);
                    if (!sample)
                        continue;
                    scalars[count] = nodes[sample->node].scalar;
                    weights[count] = sample->weight;
                    ++count;
                }
                return aggregate_ooc_marching_scalar(std::span<const float>(scalars.data(), count),
                                                     std::span<const float>(weights.data(), count));
            };

            const auto edge_endpoints = [](std::size_t edge)
            {
                const std::size_t axis = edge >> 2U;
                const std::size_t quadrant = edge & 3U;
                std::size_t first = 0U;
                if (axis == 0U)
                    first = 2U * ((quadrant & 1U) + 2U * (quadrant >> 1U));
                else if (axis == 1U)
                    first = (quadrant & 1U) + 4U * (quadrant >> 1U);
                else
                    first = (quadrant & 1U) + 2U * (quadrant >> 1U);
                return std::array<std::size_t, 2>{first, first + (1U << axis)};
            };

            const auto discover = [&](auto&& self,
                                      std::size_t node,
                                      std::uint32_t level,
                                      std::uint32_t x,
                                      std::uint32_t y,
                                      std::uint32_t z,
                                      const Neighborhood& neighborhood) -> void
            {
                const auto first = child_start(node);
                if (first == std::numeric_limits<std::size_t>::max())
                    return;
                for (std::size_t child_slot = 0U; child_slot != 8U; ++child_slot)
                {
                    const std::size_t child = first + child_slot;
                    const auto grandchild = child_start(child);
                    const std::uint32_t child_level = level + 1U;
                    const std::uint32_t child_x = 2U * x + static_cast<std::uint32_t>(child_slot & 1U);
                    const std::uint32_t child_y = 2U * y + static_cast<std::uint32_t>((child_slot >> 1U) & 1U);
                    const std::uint32_t child_z = 2U * z + static_cast<std::uint32_t>(child_slot >> 2U);
                    if (grandchild != std::numeric_limits<std::size_t>::max() && child_level != maximum_level)
                    {
                        if (!overlaps_bounds(child_x, child_y, child_z, child_level))
                            continue;
                        const auto child_neighborhood = make_child_neighborhood(neighborhood, child_slot);
                        if (child_neighborhood[13] != static_cast<std::int64_t>(child))
                        {
                            throw std::runtime_error("OOC marching child neighborhood lost its center");
                        }
                        self(self, child, child_level, child_x, child_y, child_z, child_neighborhood);
                        continue;
                    }

                    int positive = 0;
                    int negative = 0;
                    bool mixed = false;
                    for (std::size_t corner = 0U; corner != 8U; ++corner)
                    {
                        const int sign = classify_corner(neighborhood, child_slot, corner);
                        if (sign == 0)
                        {
                            mixed = true;
                            break;
                        }
                        if (sign > 0)
                        {
                            ++positive;
                            if (negative != 0)
                            {
                                mixed = true;
                                break;
                            }
                        }
                        else
                        {
                            ++negative;
                            if (positive != 0)
                            {
                                mixed = true;
                                break;
                            }
                        }
                    }
                    if (!mixed)
                        continue;

                    std::array<float, 8> corner_scalar{};
                    std::uint32_t sign_mask = 0U;
                    for (std::size_t corner = 0U; corner != 8U; ++corner)
                    {
                        corner_scalar[corner] = aggregate_corner(neighborhood, child_slot, corner);
                        if (corner_scalar[corner] > 0.0F)
                            sign_mask |= 1U << corner;
                    }
                    if (sign_mask == 0U || sign_mask == 255U)
                        continue;
                    add_active(child);

                    for (std::size_t edge = 0U; edge != 12U; ++edge)
                    {
                        const auto endpoints = edge_endpoints(edge);
                        if (((sign_mask >> endpoints[0]) & 1U) == ((sign_mask >> endpoints[1]) & 1U))
                        {
                            continue;
                        }
                        const std::size_t axis = edge >> 2U;
                        const std::size_t low = edge & 1U;
                        const std::size_t high = (edge & 2U) != 0U;
                        for (std::size_t offset = 0U; offset != 4U; ++offset)
                        {
                            std::size_t position = 0U;
                            if (axis == 0U)
                            {
                                position = 9U * (high + (offset >> 1U)) + 3U * (low + (offset & 1U)) + 1U;
                            }
                            else if (axis == 1U)
                            {
                                position = 9U * (high + (offset >> 1U)) + low + 3U + (offset & 1U);
                            }
                            else
                            {
                                position = 3U * (high + (offset >> 1U)) + low + 9U + (offset & 1U);
                            }
                            const auto adjacent = neighborhood[kOocMarchNeighborIndex[27U * child_slot + position]];
                            if (adjacent >= 0 && child_start(static_cast<std::size_t>(adjacent)) ==
                                                     std::numeric_limits<std::size_t>::max())
                            {
                                set_bit(neighbor_closure, static_cast<std::size_t>(adjacent));
                            }
                        }
                    }
                }
                for (std::size_t slot = 0U; slot != 8U; ++slot)
                {
                    if (bit_is_set(output.bits, first + slot))
                    {
                        add_active(node);
                        break;
                    }
                }
            };

            const auto close_neighbors = [&](auto&& self,
                                             std::size_t node,
                                             std::uint32_t level,
                                             std::uint32_t x,
                                             std::uint32_t y,
                                             std::uint32_t z,
                                             const Neighborhood& neighborhood) -> void
            {
                const auto first = child_start(node);
                if (first == std::numeric_limits<std::size_t>::max())
                    return;
                for (std::size_t child_slot = 0U; child_slot != 8U; ++child_slot)
                {
                    const std::size_t child = first + child_slot;
                    const auto grandchild = child_start(child);
                    const std::uint32_t child_level = level + 1U;
                    const std::uint32_t child_x = 2U * x + static_cast<std::uint32_t>(child_slot & 1U);
                    const std::uint32_t child_y = 2U * y + static_cast<std::uint32_t>((child_slot >> 1U) & 1U);
                    const std::uint32_t child_z = 2U * z + static_cast<std::uint32_t>(child_slot >> 2U);
                    if (grandchild != std::numeric_limits<std::size_t>::max() && child_level != maximum_level)
                    {
                        if (!overlaps_bounds(child_x, child_y, child_z, child_level))
                            continue;
                        self(self,
                             child,
                             child_level,
                             child_x,
                             child_y,
                             child_z,
                             make_child_neighborhood(neighborhood, child_slot));
                    }
                    else if (bit_is_set(neighbor_closure, child))
                    {
                        add_active(child);
                    }
                }
                for (std::size_t slot = 0U; slot != 8U; ++slot)
                {
                    if (bit_is_set(output.bits, first + slot))
                    {
                        add_active(node);
                        break;
                    }
                }
            };

            Neighborhood root_neighborhood{};
            root_neighborhood.fill(-1);
            root_neighborhood[13] = 0;
            discover(discover, 0U, 0U, 0U, 0U, 0U, root_neighborhood);
            close_neighbors(close_neighbors, 0U, 0U, 0U, 0U, 0U, root_neighborhood);
            return output;
        }

    } // namespace

    OocMarchingActiveCells build_ooc_marching_active_cells(std::span<const OocMarchingExtractNode> nodes,
                                                           const std::optional<OocMarchingGridBounds> bounds)
    {
        return build_ooc_marching_active_cells_impl(nodes, bounds, std::nullopt);
    }

    std::vector<OocMarchingWorkCell> plan_ooc_marching_capacity_frontier(std::span<const OocMarchingExtractNode> nodes,
                                                                         const std::uint64_t maximum_subtree_nodes)
    {
        if (nodes.empty())
        {
            throw std::invalid_argument("OOC marching work planner requires a tree");
        }
        if (((nodes.size() - 1U) & 7U) != 0U)
        {
            throw std::invalid_argument("OOC marching work planner requires root-plus-octets layout");
        }
        if (nodes.size() > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::invalid_argument("OOC marching work planner exceeds uint32 node indices");
        }
        if (maximum_subtree_nodes == 0U)
        {
            throw std::invalid_argument("OOC marching work planner capacity must be positive");
        }

        const auto child_start = [&](const std::size_t node)
        {
            const std::uint32_t group = nodes[node].child_group;
            if (group == std::numeric_limits<std::uint32_t>::max())
            {
                return std::numeric_limits<std::size_t>::max();
            }
            const std::uint64_t start64 = 1ULL + 8ULL * group;
            if (start64 + 7ULL >= nodes.size())
            {
                throw std::runtime_error("OOC marching work planner found a truncated child octet");
            }
            return static_cast<std::size_t>(start64);
        };

        std::vector<std::uint64_t> subtree_size(nodes.size(), 0U);
        std::vector<std::uint8_t> visited(nodes.size(), 0U);
        std::function<std::uint64_t(std::size_t)> measure = [&](const std::size_t node) -> std::uint64_t
        {
            if (visited[node] != 0U)
            {
                throw std::runtime_error("OOC marching work planner found a shared or cyclic node");
            }
            visited[node] = 1U;
            std::uint64_t count = 1U;
            const std::size_t first = child_start(node);
            if (first != std::numeric_limits<std::size_t>::max())
            {
                for (std::size_t slot = 0U; slot != 8U; ++slot)
                {
                    count += measure(first + slot);
                }
            }
            subtree_size[node] = count;
            return count;
        };
        measure(0U);
        if (std::find(visited.begin(), visited.end(), std::uint8_t{0U}) != visited.end())
        {
            throw std::runtime_error("OOC marching work planner found unreachable nodes");
        }

        std::vector<OocMarchingWorkCell> result;
        std::function<void(std::size_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)> split =
            [&](const std::size_t node,
                const std::uint32_t x,
                const std::uint32_t y,
                const std::uint32_t z,
                const std::uint32_t level)
        {
            const std::size_t first = child_start(node);
            if (subtree_size[node] <= maximum_subtree_nodes)
            {
                result.push_back({static_cast<std::uint32_t>(node), x, y, z, level, subtree_size[node]});
                return;
            }
            if (first == std::numeric_limits<std::size_t>::max())
            {
                throw std::runtime_error("OOC marching leaf exceeds the work-part capacity");
            }
            if (level == 31U)
            {
                throw std::runtime_error("OOC marching work planner coordinate level overflow");
            }
            for (std::size_t slot = 0U; slot != 8U; ++slot)
            {
                split(first + slot,
                      2U * x + static_cast<std::uint32_t>(slot & 1U),
                      2U * y + static_cast<std::uint32_t>((slot >> 1U) & 1U),
                      2U * z + static_cast<std::uint32_t>((slot >> 2U) & 1U),
                      level + 1U);
            }
        };
        split(0U, 0U, 0U, 0U, 0U);
        std::sort(result.begin(),
                  result.end(),
                  [](const OocMarchingWorkCell& left, const OocMarchingWorkCell& right)
                  { return left.node_index < right.node_index; });
        return result;
    }

    std::vector<OocMarchingCellState>
    build_ooc_marching_initial_cell_states(std::span<const OocMarchingExtractNode> nodes,
                                           const OocMarchingActiveCells& active_cells)
    {
        if (nodes.empty() || active_cells.entries.empty())
        {
            throw std::runtime_error("OOC marching initial state requires an active tree");
        }
        if (nodes.size() > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("OOC marching tree exceeds uint32 node indices");
        }
        if (active_cells.bits.size() != (nodes.size() + 63U) / 64U)
        {
            throw std::runtime_error("OOC marching active bitset size mismatch");
        }

        const auto child_start = [&](std::size_t node) -> std::size_t
        {
            const auto group = nodes[node].child_group;
            if (group == std::numeric_limits<std::uint32_t>::max())
                return std::numeric_limits<std::size_t>::max();
            const std::uint64_t start = 1ULL + 8ULL * group;
            if (start + 7ULL >= nodes.size())
            {
                throw std::runtime_error("OOC marching child group is truncated");
            }
            return static_cast<std::size_t>(start);
        };
        const auto bit_is_set = [](const std::vector<std::uint64_t>& bits, std::size_t node)
        { return (bits[node >> 6U] & (1ULL << (node & 63U))) != 0U; };

        std::vector<std::int64_t> ordinal(nodes.size(), -1);
        for (std::size_t entry_index = 0U; entry_index != active_cells.entries.size(); ++entry_index)
        {
            const auto& entry = active_cells.entries[entry_index];
            if (entry.selected_node_index >= nodes.size() || entry.cell_state_index >= active_cells.entries.size() ||
                entry.cell_state_index != entry_index || ordinal[entry.selected_node_index] != -1 ||
                !bit_is_set(active_cells.bits, static_cast<std::size_t>(entry.selected_node_index)))
            {
                throw std::runtime_error("invalid OOC marching active-cell entry");
            }
            ordinal[entry.selected_node_index] = static_cast<std::int64_t>(entry.cell_state_index);
        }
        if (ordinal[0] < 0)
        {
            throw std::runtime_error("OOC marching active tree has no root");
        }

        using Neighborhood = std::array<std::int64_t, 27>;
        const auto make_child_neighborhood = [&](const Neighborhood& parent, std::size_t child_slot)
        {
            Neighborhood child{};
            child.fill(-1);
            for (std::size_t position = 0U; position != 27U; ++position)
            {
                const std::size_t table = 27U * child_slot + position;
                const std::int64_t parent_node = parent[kOocMarchNeighborIndex[table]];
                if (parent_node < 0)
                    continue;
                const auto first = child_start(static_cast<std::size_t>(parent_node));
                if (first == std::numeric_limits<std::size_t>::max())
                    continue;
                child[position] = static_cast<std::int64_t>(first + kOocMarchChildSlot[table]);
            }
            return child;
        };

        struct Sample
        {
            std::size_t node{};
            float weight{};
        };
        const auto sample_node = [&](const Neighborhood& neighborhood,
                                     std::size_t child_slot,
                                     std::size_t corner,
                                     std::size_t selector) -> std::optional<Sample>
        {
            std::array<std::size_t, 3> coordinate{1U, 1U, 1U};
            for (std::size_t axis = 0U; axis != 3U; ++axis)
            {
                if ((selector & (1U << axis)) != 0U)
                    coordinate[axis] = 2U * ((corner >> axis) & 1U);
            }
            const std::size_t table = 27U * child_slot + coordinate[0] + 3U * coordinate[1] + 9U * coordinate[2];
            std::int64_t node = neighborhood[kOocMarchNeighborIndex[table]];
            if (node < 0)
                return std::nullopt;
            float scale = 2.0F;
            auto first = child_start(static_cast<std::size_t>(node));
            if (first != std::numeric_limits<std::size_t>::max())
            {
                node = static_cast<std::int64_t>(first + kOocMarchChildSlot[table]);
                scale = 1.0F;
                first = child_start(static_cast<std::size_t>(node));
                if (first != std::numeric_limits<std::size_t>::max())
                {
                    node = static_cast<std::int64_t>(first);
                    if (((corner & 1U) != 0U) != (coordinate[0] != 1U))
                        ++node;
                    if (((corner & 2U) != 0U) != (coordinate[1] != 1U))
                        node += 2;
                    if (((corner & 4U) != 0U) != (coordinate[2] != 1U))
                        node += 4;
                    scale = 0.5F;
                }
            }
            return Sample{static_cast<std::size_t>(node), 1.0F / scale};
        };

        const auto aggregate_corner = [&](const Neighborhood& neighborhood, std::size_t child_slot, std::size_t corner)
        {
            std::array<float, 8> scalars{};
            std::array<float, 8> weights{};
            std::size_t count = 0U;
            for (std::size_t selector = 0U; selector != 8U; ++selector)
            {
                const auto sample = sample_node(neighborhood, child_slot, corner, selector);
                if (!sample)
                    continue;
                scalars[count] = nodes[sample->node].scalar;
                weights[count] = sample->weight;
                ++count;
            }
            return aggregate_ooc_marching_scalar(std::span<const float>(scalars.data(), count),
                                                 std::span<const float>(weights.data(), count));
        };

        std::vector<OocMarchingCellState> states(active_cells.entries.size());
        std::vector<std::uint8_t> initialized(states.size(), 0U);
        constexpr float internal_scalar = std::bit_cast<float>(0xC36F199AU);
        std::function<void(std::size_t, std::uint32_t, const Neighborhood&)> initialize =
            [&](std::size_t node, std::uint32_t level, const Neighborhood& neighborhood)
        {
            const auto state_index = ordinal[node];
            if (state_index < 0)
            {
                throw std::runtime_error("OOC marching recursion reached inactive node");
            }
            auto& state = states[static_cast<std::size_t>(state_index)];
            initialized[static_cast<std::size_t>(state_index)] = 1U;
            const auto first = child_start(node);
            if (first == std::numeric_limits<std::size_t>::max())
            {
                throw std::runtime_error("OOC active internal node has no children");
            }
            bool has_active_child = false;
            for (std::size_t slot = 0U; slot != 8U; ++slot)
            {
                state.child_state[slot] = ordinal[first + slot];
                has_active_child |= state.child_state[slot] >= 0;
            }
            if (has_active_child)
            {
                state.corner_scalar.fill(internal_scalar);
                state.edge_vertex.fill(-1);
            }

            for (std::size_t child_slot = 0U; child_slot != 8U; ++child_slot)
            {
                const std::size_t child = first + child_slot;
                const auto child_state_index = ordinal[child];
                if (child_state_index < 0)
                    continue;
                const std::uint32_t child_level = level + 1U;
                const auto grandchild = child_start(child);
                if (grandchild != std::numeric_limits<std::size_t>::max() && child_level != active_cells.maximum_level)
                {
                    const auto child_neighborhood = make_child_neighborhood(neighborhood, child_slot);
                    if (child_neighborhood[13] != static_cast<std::int64_t>(child))
                    {
                        throw std::runtime_error("OOC marching child neighborhood lost its center");
                    }
                    initialize(child, child_level, child_neighborhood);
                    continue;
                }

                auto& child_state = states[static_cast<std::size_t>(child_state_index)];
                initialized[static_cast<std::size_t>(child_state_index)] = 1U;
                child_state.weighted_cell_scale = nodes[child].weighted_cell_scale;
                child_state.source_node_index = static_cast<std::uint32_t>(child);
                for (std::size_t corner = 0U; corner != 8U; ++corner)
                {
                    child_state.corner_scalar[corner] = aggregate_corner(neighborhood, child_slot, corner);
                    if (child_state.corner_scalar[corner] == 0.0F)
                    {
                        throw std::runtime_error("ZERO");
                    }
                }
                child_state.edge_vertex.fill(-1);
                child_state.child_state.fill(-1);
            }
        };

        Neighborhood root_neighborhood{};
        root_neighborhood.fill(-1);
        root_neighborhood[13] = 0;
        initialize(0U, 0U, root_neighborhood);
        if (std::find(initialized.begin(), initialized.end(), 0U) != initialized.end())
        {
            throw std::runtime_error("OOC marching active state was not initialized");
        }
        // sub_1ED4410 applies this final full-vector edge reset after recursion.
        for (auto& state : states)
            state.edge_vertex.fill(-1);
        return states;
    }

    std::vector<OocMarchingCellState> complete_ooc_marching_cell_states(std::vector<OocMarchingCellState> states,
                                                                        const OocMarchingActiveCells& active_cells,
                                                                        std::vector<std::size_t>* pass_state_counts)
    {
        if (states.size() != active_cells.entries.size() || states.empty())
        {
            throw std::runtime_error("OOC marching closure initial-state size mismatch");
        }
        std::int64_t root = -1;
        for (const auto& entry : active_cells.entries)
        {
            if (entry.selected_node_index == 0U)
            {
                root = static_cast<std::int64_t>(entry.cell_state_index);
                break;
            }
        }
        if (root < 0 || static_cast<std::size_t>(root) >= states.size())
        {
            throw std::runtime_error("OOC marching closure has no root state");
        }

        using Neighborhood = std::array<std::int64_t, 27>;
        const auto has_children = [&](std::int64_t state)
        {
            if (state < 0 || static_cast<std::size_t>(state) >= states.size())
                return false;
            const auto& children = states[static_cast<std::size_t>(state)].child_state;
            return std::find_if(children.begin(), children.end(), [](std::int64_t child) { return child != -1; }) !=
                   children.end();
        };
        const auto edge_endpoints = [](std::size_t edge)
        {
            const std::size_t axis = edge >> 2U;
            const std::size_t quadrant = edge & 3U;
            std::size_t first = 0U;
            if (axis == 0U)
                first = 2U * ((quadrant & 1U) + 2U * (quadrant >> 1U));
            else if (axis == 1U)
                first = (quadrant & 1U) + 4U * (quadrant >> 1U);
            else
                first = (quadrant & 1U) + 2U * (quadrant >> 1U);
            return std::array<std::size_t, 2>{first, first + (1U << axis)};
        };
        const auto edge_mask = [&](std::int64_t state)
        {
            if (state < 0 || static_cast<std::size_t>(state) >= states.size())
                throw std::runtime_error("OOC marching closure state is out of range");
            std::uint32_t signs = 0U;
            const auto& scalar = states[static_cast<std::size_t>(state)].corner_scalar;
            for (std::size_t corner = 0U; corner != 8U; ++corner)
                if (scalar[corner] > 0.0F)
                    signs |= 1U << corner;
            std::uint32_t result = 0U;
            for (std::size_t edge = 0U; edge != 12U; ++edge)
            {
                const auto endpoints = edge_endpoints(edge);
                if (((signs >> endpoints[0]) & 1U) != ((signs >> endpoints[1]) & 1U))
                {
                    result |= 1U << edge;
                }
            }
            return result;
        };
        const auto make_child_neighborhood = [&](const Neighborhood& parent, std::size_t child_slot)
        {
            Neighborhood child{};
            child.fill(-1);
            for (std::size_t position = 0U; position != 27U; ++position)
            {
                const std::size_t table = 27U * child_slot + position;
                const std::int64_t parent_state = parent[kOocMarchNeighborIndex[table]];
                if (!has_children(parent_state))
                    continue;
                child[position] = states[static_cast<std::size_t>(parent_state)].child_state[kOocMarchChildSlot[table]];
            }
            return child;
        };

        constexpr std::array<std::array<std::array<std::uint8_t, 2>, 4>, 6> face_edge_pairs{{
            {{{7U, 11U}, {7U, 9U}, {5U, 11U}, {5U, 9U}}},
            {{{6U, 10U}, {6U, 8U}, {4U, 10U}, {4U, 8U}}},
            {{{3U, 11U}, {3U, 10U}, {1U, 11U}, {1U, 10U}}},
            {{{2U, 9U}, {2U, 8U}, {0U, 9U}, {0U, 8U}}},
            {{{3U, 7U}, {3U, 6U}, {2U, 7U}, {2U, 6U}}},
            {{{1U, 5U}, {1U, 4U}, {0U, 5U}, {0U, 4U}}},
        }};

        const auto mode0_edge_trigger =
            [&](std::int64_t current, std::size_t child_slot, const Neighborhood& neighborhood)
        {
            const std::uint32_t current_edges = edge_mask(current);
            for (std::size_t edge = 0U; edge != 12U; ++edge)
            {
                // The target scans the complement of the current edge mask:
                // it is looking for a finer neighbor crossing an edge that
                // the coarse cell itself does not represent.
                if ((current_edges & (1U << edge)) != 0U)
                    continue;
                const std::size_t low = edge & 1U;
                const std::size_t high = (edge & 2U) != 0U;
                const std::size_t axis = edge >> 2U;
                for (std::size_t offset = 0U; offset != 4U; ++offset)
                {
                    std::size_t position = 0U;
                    if (axis == 0U)
                    {
                        position = 9U * (high + (offset >> 1U)) + 3U * (low + (offset & 1U)) + 1U;
                    }
                    else if (axis == 1U)
                    {
                        position = 9U * (high + (offset >> 1U)) + low + (offset & 1U) + 3U;
                    }
                    else
                    {
                        position = 3U * (high + (offset >> 1U)) + low + (offset & 1U) + 9U;
                    }
                    const std::size_t table = 27U * child_slot + position;
                    const std::int64_t adjacent_parent = neighborhood[kOocMarchNeighborIndex[table]];
                    if (!has_children(adjacent_parent))
                        continue;
                    const std::int64_t adjacent_child =
                        states[static_cast<std::size_t>(adjacent_parent)].child_state[kOocMarchChildSlot[table]];
                    if (!has_children(adjacent_child))
                        continue;

                    std::size_t first_axis = 1U - low;
                    if (low + (offset & 1U) == 1U)
                        first_axis = low;
                    std::size_t second_axis = (edge & 2U) == 0U;
                    if (high + (offset >> 1U) == 1U)
                        second_axis = high;
                    const std::size_t combined = first_axis + 2U * second_axis;
                    std::size_t child0 = 0U;
                    std::size_t child1 = 0U;
                    if (axis == 0U)
                    {
                        child0 = 2U * combined;
                        child1 = child0 + 1U;
                    }
                    else if (axis == 1U)
                    {
                        child0 = first_axis + 4U * second_axis;
                        child1 = child0 + 2U;
                    }
                    else
                    {
                        child0 = combined;
                        child1 = child0 + 4U;
                    }
                    const auto& grandchildren = states[static_cast<std::size_t>(adjacent_child)].child_state;
                    const std::int64_t state0 = grandchildren[child0];
                    const std::int64_t state1 = grandchildren[child1];
                    const std::uint32_t shared_edge = 1U << (first_axis + 4U * axis + 2U * second_axis);
                    if (state0 != -1 && state1 != -1 && (edge_mask(state0) & shared_edge) != 0U &&
                        (edge_mask(state1) & shared_edge) != 0U)
                    {
                        return true;
                    }
                }
            }
            return false;
        };

        const auto mode0_face_trigger = [&](std::int64_t, std::size_t child_slot, const Neighborhood& neighborhood)
        {
            for (std::size_t direction = 0U; direction != 6U; ++direction)
            {
                const std::size_t axis = direction >> 1U;
                const std::size_t side = direction & 1U;
                std::array<std::size_t, 3> coordinate{1U, 1U, 1U};
                coordinate[axis] = 2U * side;
                const std::size_t position = coordinate[0] + 3U * coordinate[1] + 9U * coordinate[2];
                const std::size_t table = 27U * child_slot + position;
                const std::int64_t adjacent_parent = neighborhood[kOocMarchNeighborIndex[table]];
                if (!has_children(adjacent_parent))
                    continue;
                const std::int64_t adjacent_child =
                    states[static_cast<std::size_t>(adjacent_parent)].child_state[kOocMarchChildSlot[table]];
                if (!has_children(adjacent_child))
                    continue;

                std::array<std::uint8_t, 4> flags{};
                std::array<std::size_t, 2> other{};
                std::size_t out = 0U;
                for (std::size_t candidate = 0U; candidate != 3U; ++candidate)
                    if (candidate != axis)
                        other[out++] = candidate;
                for (std::size_t quadrant = 0U; quadrant != 4U; ++quadrant)
                {
                    std::array<std::size_t, 3> child_coordinate{};
                    child_coordinate[axis] = 1U - side;
                    child_coordinate[other[0]] = quadrant & 1U;
                    child_coordinate[other[1]] = quadrant >> 1U;
                    const std::int64_t grandchild =
                        states[static_cast<std::size_t>(adjacent_child)]
                            .child_state[child_coordinate[0] + 2U * child_coordinate[1] + 4U * child_coordinate[2]];
                    if (grandchild == -1)
                        continue;
                    const std::uint32_t edges = edge_mask(grandchild);
                    const auto pair = face_edge_pairs[direction][quadrant];
                    if ((edges & (1U << pair[0])) != 0U)
                        flags[quadrant & 1U] = 1U;
                    if ((edges & (1U << pair[1])) != 0U)
                        flags[(quadrant >> 1U) + 2U] = 1U;
                }
                if ((flags[0] != 0U && flags[1] != 0U) || (flags[2] != 0U && flags[3] != 0U))
                {
                    return true;
                }
            }
            return false;
        };

        const auto mode1_face_trigger =
            [&](std::int64_t current, std::size_t child_slot, const Neighborhood& neighborhood)
        {
            for (std::size_t direction = 0U; direction != 6U; ++direction)
            {
                const std::size_t axis = direction >> 1U;
                const std::size_t side = direction & 1U;
                std::array<std::size_t, 2> other{};
                std::size_t out = 0U;
                for (std::size_t candidate = 0U; candidate != 3U; ++candidate)
                    if (candidate != axis)
                        other[out++] = candidate;
                unsigned positive = 0U;
                for (std::size_t quadrant = 0U; quadrant != 4U; ++quadrant)
                {
                    std::array<std::size_t, 3> corner{};
                    corner[axis] = side;
                    corner[other[0]] = quadrant & 1U;
                    corner[other[1]] = quadrant >> 1U;
                    positive += states[static_cast<std::size_t>(current)]
                                    .corner_scalar[corner[0] + 2U * corner[1] + 4U * corner[2]] > 0.0F;
                }
                if (positive == 0U || positive == 4U)
                    continue;
                std::array<std::size_t, 3> coordinate{1U, 1U, 1U};
                coordinate[axis] = 2U * side;
                const std::size_t position = coordinate[0] + 3U * coordinate[1] + 9U * coordinate[2];
                const std::size_t table = 27U * child_slot + position;
                const std::int64_t adjacent_parent = neighborhood[kOocMarchNeighborIndex[table]];
                if (!has_children(adjacent_parent))
                    continue;
                const std::int64_t adjacent_child =
                    states[static_cast<std::size_t>(adjacent_parent)].child_state[kOocMarchChildSlot[table]];
                if (!has_children(adjacent_child))
                    continue;
                const auto& grandchildren = states[static_cast<std::size_t>(adjacent_child)].child_state;
                for (const auto grandchild : grandchildren)
                    if (grandchild != -1 && has_children(grandchild))
                        return true;
            }
            return false;
        };

        const auto nonzero_average = [](std::span<const float> values)
        {
            float sum = 0.0F;
            for (const float value : values)
                sum = sum + value;
            const float result = sum * (1.0F / static_cast<float>(values.size()));
            return result == 0.0F ? 1.0e-6F : result;
        };

        const auto expand = [&](std::int64_t current, std::size_t child_slot, const Neighborhood& neighborhood)
        {
            if (current < 0 || static_cast<std::size_t>(current) >= states.size() || has_children(current))
            {
                throw std::runtime_error("invalid OOC marching closure expansion");
            }
            const Neighborhood child_neighborhood = make_child_neighborhood(neighborhood, child_slot);
            if (child_neighborhood[13] != current)
            {
                throw std::runtime_error("OOC marching closure lost its center");
            }
            const OocMarchingCellState parent = states[static_cast<std::size_t>(current)];
            const float center = nonzero_average(parent.corner_scalar);
            std::array<float, 6> face{};
            for (std::size_t direction = 0U; direction != 6U; ++direction)
            {
                const std::size_t axis = direction >> 1U;
                const std::size_t side = direction & 1U;
                std::array<std::size_t, 2> other{};
                std::size_t out = 0U;
                for (std::size_t candidate = 0U; candidate != 3U; ++candidate)
                    if (candidate != axis)
                        other[out++] = candidate;
                std::array<float, 4> values{};
                for (std::size_t quadrant = 0U; quadrant != 4U; ++quadrant)
                {
                    std::array<std::size_t, 3> corner{};
                    corner[axis] = side;
                    corner[other[0]] = quadrant & 1U;
                    corner[other[1]] = quadrant >> 1U;
                    values[quadrant] = parent.corner_scalar[corner[0] + 2U * corner[1] + 4U * corner[2]];
                }
                face[direction] = nonzero_average(values);
            }
            std::array<float, 12> edge{};
            for (std::size_t index = 0U; index != 12U; ++index)
            {
                const auto endpoints = edge_endpoints(index);
                const std::array<float, 2> values{parent.corner_scalar[endpoints[0]],
                                                  parent.corner_scalar[endpoints[1]]};
                edge[index] = nonzero_average(values);
            }

            const std::size_t first = states.size();
            if (first > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() - 8))
            {
                throw std::runtime_error("OOC marching closure state index overflow");
            }
            for (std::size_t slot = 0U; slot != 8U; ++slot)
                states[static_cast<std::size_t>(current)].child_state[slot] = static_cast<std::int64_t>(first + slot);
            // Do not reserve exactly one child octet at a time.  reserve(first+8)
            // defeated vector's geometric growth and copied the complete
            // 152-byte state table for nearly every expansion (hundreds of
            // millions of page faults on South).  push_back keeps the exact state
            // order while using the standard amortized geometric allocation.
            for (std::size_t child = 0U; child != 8U; ++child)
            {
                OocMarchingCellState next{};
                next.weighted_cell_scale = parent.weighted_cell_scale;
                next.source_node_index = parent.source_node_index;
                next.edge_vertex.fill(-1);
                next.child_state.fill(-1);
                for (std::size_t corner = 0U; corner != 8U; ++corner)
                {
                    float scalar = std::numeric_limits<float>::max();
                    if (corner == child)
                    {
                        scalar = parent.corner_scalar[child];
                    }
                    else if (corner == (child ^ 7U))
                    {
                        scalar = center;
                    }
                    else
                    {
                        for (std::size_t selector = 0U; selector != 8U; ++selector)
                        {
                            std::array<std::size_t, 3> coordinate{};
                            for (std::size_t axis = 0U; axis != 3U; ++axis)
                            {
                                const int direction = ((corner >> axis) & 1U) != 0U ? 1 : -1;
                                coordinate[axis] =
                                    static_cast<std::size_t>(1 + direction * static_cast<int>((selector >> axis) & 1U));
                            }
                            const std::size_t position = coordinate[0] + 3U * coordinate[1] + 9U * coordinate[2];
                            const std::size_t table = 27U * child + position;
                            const std::int64_t neighbor = child_neighborhood[kOocMarchNeighborIndex[table]];
                            if (neighbor == current || !has_children(neighbor))
                                continue;
                            const std::int64_t neighbor_child =
                                states[static_cast<std::size_t>(neighbor)].child_state[kOocMarchChildSlot[table]];
                            if (neighbor_child == -1)
                                continue;
                            std::size_t neighbor_corner = corner;
                            if (coordinate[0] == 2U)
                                --neighbor_corner;
                            else if (coordinate[0] == 0U)
                                ++neighbor_corner;
                            if (coordinate[1] == 2U)
                                neighbor_corner -= 2U;
                            else if (coordinate[1] == 0U)
                                neighbor_corner += 2U;
                            if (coordinate[2] == 2U)
                                neighbor_corner -= 4U;
                            else if (coordinate[2] == 0U)
                                neighbor_corner += 4U;
                            scalar = states[static_cast<std::size_t>(neighbor_child)].corner_scalar[neighbor_corner];
                        }
                        if (scalar == std::numeric_limits<float>::max())
                        {
                            const std::size_t difference = child ^ corner;
                            unsigned difference_count = 0U;
                            unsigned difference_axis_sum = 0U;
                            for (unsigned axis = 0U; axis != 3U; ++axis)
                            {
                                if ((difference & (1U << axis)) == 0U)
                                    continue;
                                ++difference_count;
                                difference_axis_sum += axis;
                            }
                            if (difference_count == 2U)
                            {
                                const unsigned remaining_axis = 3U - difference_axis_sum;
                                scalar = face[2U * remaining_axis + ((child >> remaining_axis) & 1U)];
                            }
                            else if (difference_count == 1U)
                            {
                                std::size_t edge_index = 0U;
                                if (difference_axis_sum == 0U)
                                {
                                    edge_index = ((child >> 1U) & 1U) + 2U * ((child >> 2U) & 1U);
                                }
                                else if (difference_axis_sum == 1U)
                                {
                                    edge_index = 4U + (child & 1U) + 2U * ((child >> 2U) & 1U);
                                }
                                else if (difference_axis_sum == 2U)
                                {
                                    edge_index = 8U + (child & 1U) + 2U * ((child >> 1U) & 1U);
                                }
                                else
                                {
                                    throw std::runtime_error("invalid OOC marching closure edge axis");
                                }
                                scalar = edge[edge_index];
                            }
                            else
                            {
                                throw std::runtime_error("invalid OOC marching closure corner relation");
                            }
                        }
                    }
                    if (scalar == 0.0F)
                        throw std::runtime_error("ZERO");
                    next.corner_scalar[corner] = scalar;
                }
                states.push_back(next);
            }
        };

        const auto run_pass = [&](bool mode1)
        {
            const std::size_t before = states.size();
            std::function<void(std::int64_t, const Neighborhood&)> visit =
                [&](std::int64_t current, const Neighborhood& neighborhood)
            {
                for (std::size_t child_slot = 0U; child_slot != 8U; ++child_slot)
                {
                    const std::int64_t child = states[static_cast<std::size_t>(current)].child_state[child_slot];
                    if (child == -1)
                        continue;
                    if (has_children(child))
                    {
                        visit(child, make_child_neighborhood(neighborhood, child_slot));
                        continue;
                    }
                    // a5 enables the target's extra mode-1 face test; it does
                    // not disable the edge and face tests used by mode 0.
                    // Every repeated closure pass therefore re-evaluates the
                    // two base predicates before the additional mode-1 rule.
                    const bool should_expand = mode0_edge_trigger(child, child_slot, neighborhood) ||
                                               mode0_face_trigger(child, child_slot, neighborhood) ||
                                               (mode1 && mode1_face_trigger(child, child_slot, neighborhood));
                    if (should_expand)
                        expand(child, child_slot, neighborhood);
                }
            };
            Neighborhood neighborhood{};
            neighborhood.fill(-1);
            neighborhood[13] = root;
            visit(root, neighborhood);
            return states.size() - before;
        };

        run_pass(false);
        if (pass_state_counts != nullptr)
            pass_state_counts->push_back(states.size());
        while (run_pass(true) != 0U)
        {
            if (pass_state_counts != nullptr)
                pass_state_counts->push_back(states.size());
        }
        if (pass_state_counts != nullptr)
            pass_state_counts->push_back(states.size());
        return states;
    }

    std::uint8_t normalized_ooc_weight(const OocWeightedNodeRecord& record)
    {
        const std::uint32_t sum = ooc_weight_sum(record);
        const std::uint32_t denominator = record.weight_denominator;
        const std::uint32_t result = denominator == 0U ? sum : sum / denominator;
        if (result > 255U)
        {
            throw std::runtime_error("OOC normalized weight exceeds uint8 range");
        }
        return static_cast<std::uint8_t>(result);
    }

    void merge_ooc_weighted_node(OocWeightedNodeRecord& target, const OocWeightedNodeRecord& source)
    {
        if (target.morton_words != source.morton_words || target.level != source.level)
        {
            throw std::runtime_error("cannot merge different OOC weighted cells");
        }
        if (target.weight_denominator == 0U)
        {
            set_ooc_weight_sum(target, ooc_weight_sum(source));
            target.weight_denominator = source.weight_denominator;
            return;
        }
        if (source.weight_denominator == 0U)
            return;

        const std::uint32_t combined_denominator =
            static_cast<std::uint32_t>(target.weight_denominator) + source.weight_denominator;
        const std::uint32_t combined_sum = static_cast<std::uint32_t>(ooc_weight_sum(target)) + ooc_weight_sum(source);
        if (combined_denominator <= 254U)
        {
            set_ooc_weight_sum(target, static_cast<std::uint16_t>(combined_sum));
            target.weight_denominator = static_cast<std::uint8_t>(combined_denominator);
            return;
        }
        target.weight_denominator = 127U;
        set_ooc_weight_sum(target, static_cast<std::uint16_t>((127U * combined_sum) / combined_denominator));
    }

    bool ooc_weighted_node_less(const OocWeightedNodeRecord& left, const OocWeightedNodeRecord& right) noexcept
    {
        for (std::size_t index = 0; index < left.morton_words.size(); ++index)
        {
            if (left.morton_words[index] < right.morton_words[index])
                return true;
            if (left.morton_words[index] > right.morton_words[index])
                return false;
        }
        return left.level < right.level;
    }

    namespace
    {

        bool same_ooc_weighted_node_key(const OocWeightedNodeRecord& left, const OocWeightedNodeRecord& right) noexcept
        {
            return left.morton_words == right.morton_words && left.level == right.level;
        }

    } // namespace

    std::vector<OocWeightedNodeRecord> reduce_ooc_weighted_nodes_local(std::span<const OocWeightedNodeRecord> records)
    {
        std::vector<OocWeightedNodeRecord> sorted(records.begin(), records.end());
        std::sort(sorted.begin(), sorted.end(), ooc_weighted_node_less);
        std::vector<OocWeightedNodeRecord> result;
        result.reserve(sorted.size());
        for (std::size_t index = 0; index < sorted.size();)
        {
            std::size_t next = index + 1U;
            while (next < sorted.size() && same_ooc_weighted_node_key(sorted[index], sorted[next]))
            {
                OocWeightedNodeRecord& previous = sorted[next - 1U];
                OocWeightedNodeRecord& current = sorted[next];
                if (previous.weight_denominator > current.weight_denominator)
                {
                    set_ooc_weight_sum(current, ooc_weight_sum(previous));
                    current.weight_denominator = previous.weight_denominator;
                }
                else if (previous.weight_denominator == current.weight_denominator)
                {
                    // The target performs signed IDIV. Initial records contain
                    // nonzero low-byte denominators in the captured input domain.
                    const std::int32_t denominator = static_cast<std::int32_t>(previous.weight_denominator);
                    if (denominator == 0)
                    {
                        throw std::runtime_error("local OOC weighted-node reduction has zero denominator");
                    }
                    if (static_cast<std::int32_t>(ooc_weight_sum(previous)) / denominator <
                        static_cast<std::int32_t>(ooc_weight_sum(current)) / denominator)
                    {
                        set_ooc_weight_sum(current, ooc_weight_sum(previous));
                    }
                }
                ++next;
            }
            result.push_back(sorted[next - 1U]);
            index = next;
        }
        return result;
    }

    std::vector<OocWeightedNodeRecord>
    merge_ooc_weighted_nodes_all_cameras(std::span<const OocWeightedNodeRecord> records)
    {
        std::vector<OocWeightedNodeRecord> sorted(records.begin(), records.end());
        std::sort(sorted.begin(), sorted.end(), ooc_weighted_node_less);
        std::vector<OocWeightedNodeRecord> result;
        result.reserve(sorted.size());
        for (const OocWeightedNodeRecord& record : sorted)
        {
            if (result.empty() || !same_ooc_weighted_node_key(result.back(), record))
            {
                result.push_back(record);
            }
            else
            {
                merge_ooc_weighted_node(result.back(), record);
            }
        }
        return result;
    }

    std::vector<OocWeightedNodeRecord>
    merge_ooc_weighted_nodes_all_cameras(std::vector<OocWeightedNodeRecord>&& records)
    {
        std::sort(records.begin(), records.end(), ooc_weighted_node_less);
        std::vector<OocWeightedNodeRecord> result;
        result.reserve(records.size());
        for (const OocWeightedNodeRecord& record : records)
        {
            if (result.empty() || !same_ooc_weighted_node_key(result.back(), record))
            {
                result.push_back(record);
            }
            else
            {
                merge_ooc_weighted_node(result.back(), record);
            }
        }
        return result;
    }

    std::vector<OocWeightedNodeRecord> filter_ooc_multi_camera_nodes(std::span<const OocWeightedNodeRecord> records)
    {
        std::vector<OocWeightedNodeRecord> result;
        result.reserve(records.size());
        for (const OocWeightedNodeRecord& record : records)
        {
            if (record.weight_denominator > 1U)
                result.push_back(record);
        }
        return result;
    }

    std::vector<OocWeightedNodeRecord> balance_ooc_weighted_nodes(std::span<const OocWeightedNodeRecord> records)
    {
        using CellSet = std::unordered_set<CellKey, CellKeyHash>;
        std::uint8_t maximum_level = 0U;
        for (const OocWeightedNodeRecord& record : records)
        {
            maximum_level = std::max(maximum_level, record.level);
        }

        FlatCellMap<OocWeightedNodeRecord> payloads(records.size());
        std::vector<CellSet> triggers(static_cast<std::size_t>(maximum_level) + 1U);
        for (const OocWeightedNodeRecord& record : records)
        {
            const CellKey cell = decode_cell(record.morton_words, record.level);
            const auto [unused, inserted] = payloads.emplace(cell, record);
            if (!inserted)
            {
                throw std::runtime_error("duplicate OOC cell before balancing");
            }
            triggers[cell.level].insert(cell);
        }

        // 0x17A91B3..0x17A9B24 forms a parent for each finer trigger and visits
        // its complete 26-neighbour stencil. Processing levels in descending order
        // makes this a compact recurrence rather than an iterative leaf search.
        for (unsigned level = maximum_level; level > 1U; --level)
        {
            CellSet& coarse = triggers[level - 1U];
            coarse.reserve(coarse.size() + triggers[level].size() * 4U);
            const std::uint64_t limit = std::uint64_t{1} << (level - 1U);
            for (const CellKey& fine : triggers[level])
            {
                const std::int64_t parent_x = fine.x >> 1U;
                const std::int64_t parent_y = fine.y >> 1U;
                const std::int64_t parent_z = fine.z >> 1U;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const std::int64_t x = parent_x + dx;
                    if (x < 0 || static_cast<std::uint64_t>(x) >= limit)
                        continue;
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        const std::int64_t y = parent_y + dy;
                        if (y < 0 || static_cast<std::uint64_t>(y) >= limit)
                            continue;
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            const std::int64_t z = parent_z + dz;
                            if (z < 0 || static_cast<std::uint64_t>(z) >= limit)
                            {
                                continue;
                            }
                            coarse.insert(CellKey{static_cast<std::uint8_t>(level - 1U),
                                                  static_cast<std::uint32_t>(x),
                                                  static_cast<std::uint32_t>(y),
                                                  static_cast<std::uint32_t>(z)});
                        }
                    }
                }
            }
        }

        std::vector<OocWeightedNodeRecord> result;
        result.reserve(records.size() * 4U + 1U);
        if (const auto* root = payloads.find(CellKey{}); root != nullptr)
        {
            result.push_back(*root);
        }
        else
        {
            result.emplace_back();
        }

        for (unsigned level = 1U; level <= maximum_level; ++level)
        {
            CellSet parents;
            parents.reserve(triggers[level].size());
            for (const CellKey& trigger : triggers[level])
            {
                parents.insert(
                    CellKey{static_cast<std::uint8_t>(level - 1U), trigger.x >> 1U, trigger.y >> 1U, trigger.z >> 1U});
            }
            for (const CellKey& parent : parents)
            {
                unsigned last_child = 0U;
                bool have_child = false;
                for (unsigned child = 0U; child < 8U; ++child)
                {
                    if (triggers[level].contains(child_cell(parent, child)))
                    {
                        last_child = child;
                        have_child = true;
                    }
                }
                if (!have_child)
                {
                    throw std::runtime_error("empty OOC trigger sibling group");
                }
                const CellKey last_trigger = child_cell(parent, last_child);
                std::uint16_t inherited_weight = 0U;
                if (const auto* source = payloads.find(last_trigger); source != nullptr)
                {
                    inherited_weight = normalized_ooc_weight(*source);
                }

                for (unsigned child = 0U; child < 8U; ++child)
                {
                    const CellKey cell = child_cell(parent, child);
                    if (const auto* source = payloads.find(cell); source != nullptr)
                    {
                        result.push_back(*source);
                        continue;
                    }
                    OocWeightedNodeRecord generated{};
                    generated.morton_words = encode_cell_words(cell);
                    generated.level = cell.level;
                    // A stencil-created trigger is present before sibling closure
                    // and therefore remains zero. Only a non-trigger sibling gets
                    // the last trigger's normalized byte in the uint16 sum field.
                    if (!triggers[level].contains(cell))
                    {
                        set_ooc_weight_sum(generated, inherited_weight);
                    }
                    result.push_back(generated);
                }
            }
        }
        std::sort(result.begin(), result.end(), ooc_weighted_node_less);
        return result;
    }

    std::vector<OocWeightedNodeRecord>
    balance_ooc_weighted_nodes_partitioned(std::span<const OocWeightedNodeRecord> records,
                                           std::size_t partition_records)
    {
        if (partition_records == 0U)
        {
            throw std::invalid_argument("OOC balance partition size must be non-zero");
        }
        std::vector<OocWeightedNodeRecord> sorted(records.begin(), records.end());
        std::sort(sorted.begin(), sorted.end(), ooc_weighted_node_less);
        const std::size_t partition_count = (sorted.size() + partition_records - 1U) / partition_records;
        if (partition_count <= 1U)
            return balance_ooc_weighted_nodes(sorted);

        std::vector<std::vector<OocWeightedNodeRecord>> partitions(partition_count);
        for (std::size_t partition = 0U; partition != partition_count; ++partition)
        {
            const std::size_t first = partition * partition_records;
            const std::size_t count = std::min(partition_records, sorted.size() - first);
            partitions[partition] =
                balance_ooc_weighted_nodes(std::span<const OocWeightedNodeRecord>(sorted).subspan(first, count));
        }

        // Worker outputs are individually sorted. Merge their key streams without
        // materializing and sorting another multi-million-record tagged table.
        std::vector<std::size_t> cursors(partition_count, 0U);
        std::vector<OocWeightedNodeRecord> result;
        std::size_t upper_bound = 0U;
        for (const auto& partition : partitions)
            upper_bound += partition.size();
        result.reserve(upper_bound);
        for (;;)
        {
            const OocWeightedNodeRecord* first = nullptr;
            for (std::size_t partition = 0U; partition != partition_count; ++partition)
            {
                if (cursors[partition] == partitions[partition].size())
                    continue;
                const auto& candidate = partitions[partition][cursors[partition]];
                if (first == nullptr || ooc_weighted_node_less(candidate, *first))
                {
                    first = &candidate;
                }
            }
            if (first == nullptr)
                break;

            const OocWeightedNodeRecord* original = nullptr;
            const OocWeightedNodeRecord* later_generated = nullptr;
            for (std::size_t partition = 0U; partition != partition_count; ++partition)
            {
                if (cursors[partition] == partitions[partition].size())
                    continue;
                const auto& candidate = partitions[partition][cursors[partition]];
                if (!same_ooc_weighted_node_key(*first, candidate))
                    continue;
                ++cursors[partition];
                if (candidate.weight_denominator != 0U)
                {
                    if (original != nullptr)
                    {
                        throw std::runtime_error("duplicate original OOC balance record across partitions");
                    }
                    original = &candidate;
                }
                else
                {
                    // Iterating in spatial partition order intentionally leaves
                    // the later partition as the owner of generated halo overlap.
                    later_generated = &candidate;
                }
            }
            if (original == nullptr && later_generated == nullptr)
            {
                throw std::runtime_error("empty OOC balance merge group");
            }
            result.push_back(original != nullptr ? *original : *later_generated);
        }
        return result;
    }

    std::vector<OocOctreeRecord> initialize_ooc_octree_records(std::span<const OocWeightedNodeRecord> balanced_records)
    {
        std::vector<OocOctreeRecord> result;
        result.reserve(balanced_records.size());
        for (const OocWeightedNodeRecord& source : balanced_records)
        {
            OocOctreeRecord target{};
            target.morton_words = source.morton_words;
            target.level = source.level;
            target.weight = normalized_ooc_weight(source);
            target.trailing_field = 0xC000U;
            result.push_back(target);
        }
        return result;
    }

    std::vector<OocOctreeRecord>
    build_ooc_octree_records_from_histogram_mode0(std::span<const OocWeightedNodeRecord> balanced_records,
                                                  std::span<const OocHistogramVoxel> histogram_voxels)
    {
        if (balanced_records.empty() || histogram_voxels.size() != balanced_records.size() + 1U)
        {
            throw std::invalid_argument("OOC histogram/persistent-record sizes are inconsistent");
        }
        std::vector<OocWeightedNodeRecord> level_order(balanced_records.begin(), balanced_records.end());
        std::sort(level_order.begin(),
                  level_order.end(),
                  [](const OocWeightedNodeRecord& left, const OocWeightedNodeRecord& right)
                  {
                      if (left.level != right.level)
                          return left.level < right.level;
                      return left.morton_words < right.morton_words;
                  });
        if (level_order.back().level == 0U)
        {
            throw std::invalid_argument("OOC histogram table has no persistent level below its maximum");
        }

        std::unordered_map<CellKey, std::array<std::uint8_t, 10>, CellKeyHash> histograms;
        histograms.reserve(level_order.size());
        for (std::size_t index = 0U; index != level_order.size(); ++index)
        {
            const auto& source = level_order[index];
            const auto& voxel = histogram_voxels[index + 1U];
            if (voxel.metadata[2] != source.level || voxel.metadata[3] != normalized_ooc_weight(source))
            {
                throw std::invalid_argument("OOC histogram voxel metadata does not match its Morton node");
            }
            const auto [unused, inserted] =
                histograms.emplace(decode_cell(source.morton_words, source.level), voxel.histogram);
            if (!inserted)
            {
                throw std::invalid_argument("OOC histogram input contains a duplicate Morton node");
            }
        }

        std::vector<OocOctreeRecord> result;
        result.reserve(balanced_records.size());
        for (const auto& source : balanced_records)
        {
            const auto found = histograms.find(decode_cell(source.morton_words, source.level));
            if (found == histograms.end())
            {
                throw std::runtime_error("OOC persistent node has no accumulated histogram");
            }
            OocOctreeRecord target{};
            target.morton_words = source.morton_words;
            target.level = source.level;
            target.weight = normalized_ooc_weight(source);
            target.histogram = found->second;
            target.scalar_lut_index = 0U;
            target.trailing_field = 0xC000U;
            result.push_back(target);
        }
        if (result.empty())
        {
            throw std::runtime_error("OOC histogram bridge produced no records");
        }
        return result;
    }

    std::vector<std::uint32_t> select_ooc_octree_nodes_breadth_first(std::span<const OocOctreeRecord> records)
    {
        if (records.empty() || records.size() > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::invalid_argument("OOC persistent record table is empty or exceeds uint32");
        }
        const auto record_less = [](const OocOctreeRecord& left, const OocOctreeRecord& right) noexcept
        {
            if (left.morton_words != right.morton_words)
            {
                return left.morton_words < right.morton_words;
            }
            return left.level < right.level;
        };
        if (std::is_sorted(records.begin(), records.end(), record_less))
        {
            std::array<std::vector<std::uint32_t>, 33> levels;
            for (std::size_t index = 0U; index != records.size(); ++index)
            {
                const auto& record = records[index];
                if (record.level > 32U)
                {
                    throw std::invalid_argument("OOC persistent record level exceeds the supported domain");
                }
                if (index != 0U && record.morton_words == records[index - 1U].morton_words &&
                    record.level == records[index - 1U].level)
                {
                    throw std::invalid_argument("OOC persistent record table contains a duplicate cell");
                }
                levels[record.level].push_back(static_cast<std::uint32_t>(index));
            }
            if (levels[0].size() != 1U || decode_cell(records[levels[0][0]]) != CellKey{})
            {
                throw std::invalid_argument("OOC persistent record table has no unique root");
            }

            for (std::size_t level = 1U; level != levels.size(); ++level)
            {
                const auto& children = levels[level];
                if ((children.size() & 7U) != 0U)
                {
                    throw std::invalid_argument("OOC persistent tree contains an incomplete child octet");
                }
                for (std::size_t first = 0U; first != children.size(); first += 8U)
                {
                    const CellKey first_child = decode_cell(records[children[first]]);
                    const CellKey parent{static_cast<std::uint8_t>(level - 1U),
                                         first_child.x >> 1U,
                                         first_child.y >> 1U,
                                         first_child.z >> 1U};
                    for (unsigned slot = 0U; slot != 8U; ++slot)
                    {
                        if (decode_cell(records[children[first + slot]]) != child_cell(parent, slot))
                        {
                            throw std::invalid_argument("OOC persistent tree contains an incomplete child octet");
                        }
                    }
                    OocOctreeRecord parent_record{};
                    parent_record.morton_words = encode_cell_words(parent);
                    parent_record.level = parent.level;
                    const auto found = std::lower_bound(records.begin(), records.end(), parent_record, record_less);
                    if (found == records.end() || found->morton_words != parent_record.morton_words ||
                        found->level != parent_record.level)
                    {
                        throw std::invalid_argument("OOC persistent tree contains an unreachable child octet");
                    }
                }
            }

            std::vector<std::uint32_t> result;
            result.reserve(records.size());
            for (const auto& level : levels)
            {
                result.insert(result.end(), level.begin(), level.end());
            }
            return result;
        }

        FlatCellMap<std::uint32_t> lookup(records.size());
        for (std::size_t index = 0U; index != records.size(); ++index)
        {
            const auto key = decode_cell(records[index]);
            const auto [unused, inserted] = lookup.emplace(key, static_cast<std::uint32_t>(index));
            if (!inserted)
            {
                throw std::invalid_argument("OOC persistent record table contains a duplicate cell");
            }
        }
        const auto root = lookup.find(CellKey{0U, 0U, 0U, 0U});
        if (root == nullptr)
        {
            throw std::invalid_argument("OOC persistent record table has no root");
        }

        std::vector<std::uint32_t> result{*root};
        for (std::size_t cursor = 0U; cursor != result.size(); ++cursor)
        {
            const CellKey parent = decode_cell(records[result[cursor]]);
            if (parent.level == 32U)
                continue;
            std::array<std::uint32_t, 8> children{};
            std::size_t child_count = 0U;
            for (unsigned slot = 0U; slot != 8U; ++slot)
            {
                const auto child = lookup.find(child_cell(parent, slot));
                if (child != nullptr)
                    children[child_count++] = *child;
            }
            if (child_count != 0U && child_count != 8U)
            {
                throw std::invalid_argument("OOC persistent tree contains an incomplete child octet");
            }
            result.insert(result.end(), children.begin(), children.begin() + static_cast<std::ptrdiff_t>(child_count));
        }
        if (result.size() != records.size())
        {
            throw std::invalid_argument("OOC persistent tree contains unreachable records");
        }
        return result;
    }

    std::vector<std::uint8_t> make_ooc_leaf_active_mask(std::span<const OocOctreeRecord> records,
                                                        std::span<const std::uint32_t> selected_indices)
    {
        if (selected_indices.size() != records.size())
        {
            throw std::invalid_argument("OOC selected tree does not cover the record table");
        }
        FlatCellMap<std::uint8_t> cells(records.size());
        for (const auto& record : records)
        {
            if (!cells.emplace(decode_cell(record), std::uint8_t{1U}).second)
            {
                throw std::invalid_argument("OOC active-mask input contains a duplicate cell");
            }
        }
        std::vector<std::uint8_t> result(selected_indices.size(), 0U);
        std::vector<std::uint8_t> visited(records.size(), 0U);
        for (const std::uint32_t record_index : selected_indices)
        {
            if (record_index >= records.size() || visited[record_index]++)
            {
                throw std::invalid_argument("OOC active-mask selection is invalid or duplicated");
            }
        }
        const auto& read_only_cells = std::as_const(cells);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(selected_indices.size());
             ++signed_index)
        {
            const std::size_t index = static_cast<std::size_t>(signed_index);
            const CellKey cell = decode_cell(records[selected_indices[index]]);
            const bool has_children = cell.level != 32U && read_only_cells.contains(child_cell(cell, 0U));
            result[index] = has_children ? 0U : 1U;
        }
        return result;
    }

    OocFusionState prepare_ooc_fusion_state(const std::vector<OocOctreeRecord>& records,
                                            const std::vector<std::uint32_t>& selected_indices,
                                            const std::vector<std::uint8_t>& active,
                                            const std::vector<float>& scalar_lut,
                                            const std::vector<std::uint8_t>& partition_excluded,
                                            std::optional<std::size_t> cuda_device_index)
    {
        const std::size_t count = selected_indices.size();
        if (count > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("OOC selected-node count exceeds uint32 index range");
        }
        if (active.size() != count)
        {
            throw std::runtime_error("OOC active-mask size mismatch");
        }
        if (!partition_excluded.empty() && partition_excluded.size() != count)
        {
            throw std::runtime_error("OOC partition-mask size mismatch");
        }

        OocFusionState state;
        state.weights.resize(count);
        state.histogram.resize(count * 10U);
        state.neighbors.assign(count * 6U, std::numeric_limits<std::uint32_t>::max());
        state.connectivity.assign(count, 0U);
        state.refinement.assign(count, 0U);
        state.flags.assign(count, 0U);
        state.u.resize(count);
        state.u_old.resize(count);
        state.p.assign(count * 3U, 0U);
        state.v.assign(count * 3U, 0U);
        state.v_old.assign(count * 3U, 0U);
        state.q.assign(count * 9U, 0U);

        if (cuda_device_index.has_value())
        {
#if defined(METMODEL_HAS_CUDA)
            std::vector<std::uint32_t> morton_words(count * 3U);
            std::vector<std::uint8_t> levels(count);
            for (std::size_t output_index = 0; output_index < count; ++output_index)
            {
                const auto record_index = selected_indices[output_index];
                if (record_index >= records.size())
                {
                    throw std::runtime_error("OOC selected index is outside the record table");
                }
                const auto& record = records[record_index];
                if (record.scalar_lut_index >= scalar_lut.size())
                {
                    throw std::runtime_error("OOC scalar LUT index is outside the table");
                }
                if (output_index != 0U)
                {
                    const auto& previous = records[selected_indices[output_index - 1U]];
                    if (previous.level > record.level ||
                        (previous.level == record.level && !(previous.morton_words < record.morton_words)))
                    {
                        throw std::runtime_error("OOC CUDA neighbor input is not level/Morton ordered");
                    }
                }
            }
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(count); ++signed_index)
            {
                const std::size_t output_index = static_cast<std::size_t>(signed_index);
                const auto& record = records[selected_indices[output_index]];
                state.weights[output_index] = record.weight;
                for (std::size_t bin = 0; bin < 10U; ++bin)
                {
                    state.histogram[10U * output_index + bin] = record.histogram[bin];
                }
                state.u[output_index] = scalar_lut[record.scalar_lut_index];
                state.u_old[output_index] = state.u[output_index];
                std::uint8_t flags = active[output_index] == 0U ? 4U : 0U;
                if (!partition_excluded.empty() && partition_excluded[output_index] != 0U)
                {
                    flags |= 2U;
                }
                state.flags[output_index] = flags;
                for (std::size_t word = 0U; word < 3U; ++word)
                {
                    morton_words[word * count + output_index] = record.morton_words[word];
                }
                levels[output_index] = record.level;
            }

            std::string cuda_error;
            if (!run_recovered_ooc_neighbors_cuda_source(morton_words,
                                                         levels,
                                                         state.neighbors,
                                                         state.connectivity,
                                                         state.refinement,
                                                         *cuda_device_index,
                                                         cuda_error))
            {
                throw std::runtime_error(cuda_error);
            }
            state.validate();
            return state;
#else
            throw std::runtime_error("OOC CUDA neighbor search requested in a CUDA-disabled build");
#endif
        }

        std::vector<CellKey> cells(count);
        FlatCellMap<std::uint32_t> lookup(count);

        for (std::size_t output_index = 0; output_index < count; ++output_index)
        {
            const auto record_index = selected_indices[output_index];
            if (record_index >= records.size())
            {
                throw std::runtime_error("OOC selected index is outside the record table");
            }
            const auto& record = records[record_index];
            if (record.scalar_lut_index >= scalar_lut.size())
            {
                throw std::runtime_error("OOC scalar LUT index is outside the table");
            }
            const auto cell = decode_cell(record);
            cells[output_index] = cell;
            const auto [unused, inserted] = lookup.emplace(cell, static_cast<std::uint32_t>(output_index));
            if (!inserted)
                throw std::runtime_error("duplicate selected OOC cell");
        }

#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(count); ++signed_index)
        {
            const std::size_t output_index = static_cast<std::size_t>(signed_index);
            const auto& record = records[selected_indices[output_index]];
            state.weights[output_index] = record.weight;
            for (std::size_t bin = 0; bin < 10U; ++bin)
            {
                state.histogram[10U * output_index + bin] = record.histogram[bin];
            }
            state.u[output_index] = scalar_lut[record.scalar_lut_index];
            state.u_old[output_index] = state.u[output_index];
            std::uint8_t flags = active[output_index] == 0U ? 4U : 0U;
            if (!partition_excluded.empty() && partition_excluded[output_index] != 0U)
            {
                flags |= 2U;
            }
            state.flags[output_index] = flags;
        }

        constexpr std::array<int, 6> axes{0, 0, 1, 1, 2, 2};
        constexpr std::array<int, 6> signs{-1, 1, -1, 1, -1, 1};
        const auto& read_only_lookup = std::as_const(lookup);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(count); ++signed_index)
        {
            const std::size_t index = static_cast<std::size_t>(signed_index);
            const auto& cell = cells[index];
            const std::array<std::uint32_t, 3> xyz{cell.x, cell.y, cell.z};
            for (std::size_t direction = 0; direction < 6U; ++direction)
            {
                const int axis = axes[direction];
                const int sign = signs[direction];
                const std::uint32_t* found = nullptr;
                bool refined = false;

                if (cell.level < 32U)
                {
                    std::array<std::uint64_t, 3> fine{2ULL * cell.x, 2ULL * cell.y, 2ULL * cell.z};
                    if (sign < 0)
                    {
                        if (xyz[static_cast<std::size_t>(axis)] != 0U)
                        {
                            fine[static_cast<std::size_t>(axis)] = 2ULL * xyz[static_cast<std::size_t>(axis)] - 1ULL;
                            found = read_only_lookup.find(CellKey{static_cast<std::uint8_t>(cell.level + 1U),
                                                                  static_cast<std::uint32_t>(fine[0]),
                                                                  static_cast<std::uint32_t>(fine[1]),
                                                                  static_cast<std::uint32_t>(fine[2])});
                        }
                    }
                    else
                    {
                        fine[static_cast<std::size_t>(axis)] = 2ULL * xyz[static_cast<std::size_t>(axis)] + 2ULL;
                        const std::uint64_t limit = 1ULL << (cell.level + 1U);
                        if (fine[static_cast<std::size_t>(axis)] < limit)
                        {
                            found = read_only_lookup.find(CellKey{static_cast<std::uint8_t>(cell.level + 1U),
                                                                  static_cast<std::uint32_t>(fine[0]),
                                                                  static_cast<std::uint32_t>(fine[1]),
                                                                  static_cast<std::uint32_t>(fine[2])});
                        }
                    }
                    refined = found != nullptr;
                }

                if (found == nullptr)
                {
                    auto same = xyz;
                    const std::uint64_t limit = 1ULL << cell.level;
                    if (sign < 0)
                    {
                        if (same[static_cast<std::size_t>(axis)] != 0U)
                        {
                            --same[static_cast<std::size_t>(axis)];
                            found = read_only_lookup.find(CellKey{cell.level, same[0], same[1], same[2]});
                        }
                    }
                    else if (static_cast<std::uint64_t>(same[static_cast<std::size_t>(axis)]) + 1ULL < limit)
                    {
                        ++same[static_cast<std::size_t>(axis)];
                        found = read_only_lookup.find(CellKey{cell.level, same[0], same[1], same[2]});
                    }
                }

                const bool crosses_parent = sign < 0 ? (xyz[static_cast<std::size_t>(axis)] & 1U) == 0U
                                                     : (xyz[static_cast<std::size_t>(axis)] & 1U) != 0U;
                const std::uint64_t level_limit = 1ULL << cell.level;
                const bool inside_domain =
                    sign < 0 ? xyz[static_cast<std::size_t>(axis)] != 0U
                             : static_cast<std::uint64_t>(xyz[static_cast<std::size_t>(axis)]) + 1ULL < level_limit;
                if (found == nullptr && cell.level != 0U && crosses_parent && inside_domain)
                {
                    std::array<std::uint32_t, 3> coarse{cell.x >> 1U, cell.y >> 1U, cell.z >> 1U};
                    coarse[static_cast<std::size_t>(axis)] = static_cast<std::uint32_t>(
                        (static_cast<std::int64_t>(xyz[static_cast<std::size_t>(axis)]) + sign) / 2);
                    found = read_only_lookup.find(
                        CellKey{static_cast<std::uint8_t>(cell.level - 1U), coarse[0], coarse[1], coarse[2]});
                }

                if (found != nullptr)
                {
                    state.neighbors[6U * index + direction] = *found;
                    state.connectivity[index] |= static_cast<std::uint8_t>(1U << direction);
                    if (refined)
                    {
                        state.refinement[index] |= static_cast<std::uint8_t>(1U << direction);
                    }
                }
            }
        }

        state.validate();
        return state;
    }

    std::vector<std::uint16_t> pack_ooc_fused_scalars(const std::vector<float>& fused_u)
    {
        std::vector<std::uint16_t> result;
        result.reserve(fused_u.size());
        for (const float value : fused_u)
        {
            // This ordering preserves the target's COMISS/JBE behavior: NaN takes
            // the first branch and is serialized as +1 rather than as half NaN.
            const float clamped = !(value < 1.0F) ? 1.0F : (value <= -1.0F ? -1.0F : value);
            result.push_back(float_to_half(clamped));
        }
        return result;
    }

    void write_ooc_fused_scalars(std::vector<OocOctreeRecord>& records,
                                 const std::vector<std::uint32_t>& selected_indices,
                                 const std::vector<std::uint16_t>& packed_scalars,
                                 const std::vector<std::uint8_t>& partition_included)
    {
        if (packed_scalars.size() != selected_indices.size())
        {
            throw std::runtime_error("OOC packed-scalar size mismatch");
        }
        if (!partition_included.empty() && partition_included.size() != selected_indices.size())
        {
            throw std::runtime_error("OOC partition-inclusion size mismatch");
        }
        for (std::size_t index = 0; index < selected_indices.size(); ++index)
        {
            const auto record_index = selected_indices[index];
            if (record_index >= records.size())
            {
                throw std::runtime_error("OOC scalar write index is outside the record table");
            }
            if (partition_included.empty() || partition_included[index] != 0U)
            {
                const float value = half_to_float(packed_scalars[index]);
                if (!(value >= -1.0F && value <= 1.0F))
                {
                    throw std::runtime_error("OOC packed scalar is outside [-1, 1]");
                }
                records[record_index].trailing_field = packed_scalars[index];
            }
        }
    }

    std::vector<OocMarchingNode> make_ooc_marching_nodes(const std::vector<OocOctreeRecord>& records,
                                                         const std::vector<std::uint32_t>& selected_indices,
                                                         const std::vector<std::uint8_t>& weight_denominator,
                                                         std::uint8_t denominator_cap)
    {
        const std::size_t count = selected_indices.size();
        if (weight_denominator.size() != count)
        {
            throw std::runtime_error("OOC marching weight-denominator size mismatch");
        }
        for (const std::uint32_t record_index : selected_indices)
        {
            if (record_index >= records.size())
            {
                throw std::runtime_error("OOC marching index is outside the record table");
            }
        }
        std::vector<OocMarchingNode> result(count);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(count); ++signed_index)
        {
            const std::size_t index = static_cast<std::size_t>(signed_index);
            const auto record_index = selected_indices[index];
            const auto& source = records[record_index];
            auto& target = result[index];
            target.morton_words = source.morton_words;
            target.level = source.level;
            target.weight_denominator = std::min(weight_denominator[index], denominator_cap);
            std::uint32_t central_histogram_sum = 0U;
            for (std::size_t bin = 3U; bin != 7U; ++bin)
            {
                central_histogram_sum += source.histogram[bin];
            }
            target.central_histogram_sum = std::min(central_histogram_sum, 254U);
            target.scalar = half_to_float(source.trailing_field);
            if (target.scalar == 0.0F)
                target.scalar = 1.0e-6F;
            const float cell_scale = std::ldexp(0.5F, -static_cast<int>(source.level));
            target.weighted_cell_scale = cell_scale * (0.75F + 0.75F * (static_cast<float>(source.weight) / 255.9F));
        }
        return result;
    }

    std::vector<std::uint8_t>
    select_ooc_marching_weight_denominators(const std::vector<OocOctreeRecord>& records,
                                            const std::vector<std::uint32_t>& selected_indices,
                                            std::span<const OocWeightedNodeRecord> balanced_records)
    {
        // The production scheduler preserves the persistent balanced-record
        // index space. Verify that invariant for every record before taking
        // the direct projection; synthetic or reordered callers retain the
        // generic CellKey join below.
        bool same_index_space = records.size() == balanced_records.size();
        if (same_index_space)
        {
            std::atomic<bool> mismatch{false};
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(records.size());
                 ++signed_index)
            {
                if (mismatch.load(std::memory_order_relaxed))
                    continue;
                const std::size_t index = static_cast<std::size_t>(signed_index);
                if (records[index].level != balanced_records[index].level ||
                    records[index].morton_words != balanced_records[index].morton_words)
                {
                    mismatch.store(true, std::memory_order_relaxed);
                }
            }
            same_index_space = !mismatch.load(std::memory_order_relaxed);
        }
        if (same_index_space)
        {
            std::vector<std::uint8_t> result(selected_indices.size(), 0U);
            std::atomic<bool> invalid_index{false};
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(selected_indices.size());
                 ++signed_index)
            {
                const std::size_t index = static_cast<std::size_t>(signed_index);
                const std::uint32_t record_index = selected_indices[index];
                if (record_index >= balanced_records.size())
                {
                    invalid_index.store(true, std::memory_order_relaxed);
                }
                else
                {
                    result[index] = balanced_records[record_index].weight_denominator;
                }
            }
            if (invalid_index.load(std::memory_order_relaxed))
            {
                throw std::runtime_error("OOC marching index is outside the record table");
            }
            return result;
        }

        FlatCellMap<std::uint8_t> denominators(balanced_records.size());
        for (const OocWeightedNodeRecord& source : balanced_records)
        {
            const CellKey key = decode_cell(source.morton_words, source.level);
            const auto [unused, inserted] = denominators.emplace(key, source.weight_denominator);
            if (!inserted)
            {
                throw std::runtime_error("duplicate upstream OOC weighted node for marching");
            }
        }

        std::vector<std::uint8_t> result(selected_indices.size(), 0U);
        for (const std::uint32_t record_index : selected_indices)
        {
            if (record_index >= records.size())
            {
                throw std::runtime_error("OOC marching index is outside the record table");
            }
        }
        const auto& read_only_denominators = std::as_const(denominators);
        std::atomic<bool> missing_denominator{false};
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(selected_indices.size());
             ++signed_index)
        {
            const std::size_t index = static_cast<std::size_t>(signed_index);
            const OocOctreeRecord& record = records[selected_indices[index]];
            const CellKey key = decode_cell(record.morton_words, record.level);
            const auto* source = read_only_denominators.find(key);
            if (source == nullptr)
            {
                missing_denominator.store(true, std::memory_order_relaxed);
            }
            else
            {
                result[index] = *source;
            }
        }
        if (missing_denominator.load(std::memory_order_relaxed))
        {
            throw std::runtime_error("OOC marching node has no upstream weight denominator");
        }
        return result;
    }

    std::vector<OocMarchingNode> make_ooc_marching_nodes(const std::vector<OocOctreeRecord>& records,
                                                         const std::vector<std::uint32_t>& selected_indices,
                                                         std::span<const OocWeightedNodeRecord> balanced_records,
                                                         std::uint8_t denominator_cap)
    {
        return make_ooc_marching_nodes(
            records,
            selected_indices,
            select_ooc_marching_weight_denominators(records, selected_indices, balanced_records),
            denominator_cap);
    }

    std::vector<OocMarchingExtractNode> build_ooc_marching_extract(const std::vector<OocMarchingNode>& nodes)
    {
        if (nodes.size() > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("OOC marching-node count exceeds uint32 index range");
        }
        const auto node_less = [](const OocMarchingNode& left, const OocMarchingNode& right) noexcept
        {
            if (left.level != right.level)
                return left.level < right.level;
            return left.morton_words < right.morton_words;
        };
        if (!std::is_sorted(nodes.begin(), nodes.end(), node_less))
        {
            throw std::runtime_error("OOC marching nodes are not level/Morton ordered");
        }

        std::vector<OocMarchingExtractNode> result(nodes.size());
        std::atomic<int> extract_error{0};
#pragma omp parallel for schedule(dynamic, 1024)
        for (std::ptrdiff_t signed_index = 0; signed_index < static_cast<std::ptrdiff_t>(nodes.size()); ++signed_index)
        {
            const std::size_t index = static_cast<std::size_t>(signed_index);
            auto& target = result[index];
            target.weighted_cell_scale = nodes[index].weighted_cell_scale;
            target.scalar = nodes[index].scalar;
            const CellKey cell = decode_cell(nodes[index]);
            if (cell.level == 32U)
                continue;
            OocMarchingNode child_key{};
            child_key.morton_words = encode_cell_words(child_cell(cell, 0U));
            child_key.level = static_cast<std::uint8_t>(cell.level + 1U);
            const auto found = std::lower_bound(nodes.begin(), nodes.end(), child_key, node_less);
            if (found == nodes.end() || found->level != child_key.level ||
                found->morton_words != child_key.morton_words)
                continue;
            const std::uint32_t first_child = static_cast<std::uint32_t>(found - nodes.begin());
            if ((first_child & 7U) != 1U)
            {
                extract_error.store(1, std::memory_order_relaxed);
                continue;
            }
            if (static_cast<std::uint64_t>(first_child) + 7U >= nodes.size())
            {
                extract_error.store(2, std::memory_order_relaxed);
                continue;
            }
            for (unsigned slot = 1U; slot != 8U; ++slot)
            {
                const auto& candidate = nodes[first_child + slot];
                const CellKey expected = child_cell(cell, slot);
                if (candidate.level != expected.level || candidate.morton_words != encode_cell_words(expected))
                {
                    extract_error.store(3, std::memory_order_relaxed);
                    break;
                }
            }
            target.child_group = (first_child - 1U) >> 3U;
        }
        if (extract_error.load(std::memory_order_relaxed) == 1)
        {
            throw std::runtime_error("OOC children do not start at index 1 modulo 8");
        }
        if (extract_error.load(std::memory_order_relaxed) == 2)
        {
            throw std::runtime_error("truncated OOC child group");
        }
        if (extract_error.load(std::memory_order_relaxed) == 3)
        {
            throw std::runtime_error("non-contiguous OOC child group");
        }
        return result;
    }

    void reorder_ooc_marching_child_groups(std::vector<OocMarchingExtractNode>& nodes)
    {
        if (nodes.empty())
            return;
        if (((nodes.size() - 1U) & 7U) != 0U)
        {
            throw std::runtime_error("OOC marching nodes are not root plus child octets");
        }
        constexpr std::array<std::size_t, 8> source_order{0U, 4U, 2U, 6U, 1U, 5U, 3U, 7U};
        for (std::size_t first = 1U; first < nodes.size(); first += 8U)
        {
            std::array<OocMarchingExtractNode, 8> copy{};
            std::copy_n(nodes.begin() + static_cast<std::ptrdiff_t>(first), 8U, copy.begin());
            for (std::size_t slot = 0; slot < 8U; ++slot)
            {
                nodes[first + slot] = copy[source_order[slot]];
            }
        }
    }

    [[nodiscard]] std::vector<std::array<float, 4>>
    make_ooc_marching_trim_attributes(std::span<const OocMarchingNode> nodes)
    {
        std::vector<std::array<float, 4>> attributes;
        attributes.reserve(nodes.size());
        for (const OocMarchingNode& node : nodes)
        {
            attributes.push_back({static_cast<float>(node.weight_denominator),
                                  static_cast<float>(node.central_histogram_sum),
                                  static_cast<float>(node.level),
                                  1.0F});
        }
        // The marching extractor reorders each child octet only in its compact
        // 12-byte topology view.  Vertex source indices still address the original
        // breadth-first 32-byte node table.  Metashape therefore reads attributes
        // +16/+20/+12 directly at that source index; permuting this parallel array
        // silently associates a sibling's denominator/histogram with the vertex.
        return attributes;
    }

    RecoveredOocContinuousPartOutput
    run_recovered_ooc_continuous_part_cpu(std::vector<OocOctreeRecord> records,
                                          const std::vector<std::uint32_t>& selected_indices,
                                          const std::vector<std::uint8_t>& active,
                                          const std::vector<float>& scalar_lut,
                                          std::span<const OocWeightedNodeRecord> balanced_records,
                                          const std::vector<std::uint8_t>& partition_excluded,
                                          const std::vector<std::uint8_t>& partition_included,
                                          const OocFusionParameters& fusion_parameters)
    {
        if (balanced_records.empty() && !selected_indices.empty())
        {
            throw std::runtime_error("OOC continuous part requires upstream weighted nodes");
        }
        RecoveredOocContinuousPartOutput output;
        output.records = std::move(records);
        output.fusion =
            prepare_ooc_fusion_state(output.records, selected_indices, active, scalar_lut, partition_excluded);
        run_ooc_fusion_cpu(output.fusion, fusion_parameters);
        output.packed_scalars = pack_ooc_fused_scalars(output.fusion.u);
        write_ooc_fused_scalars(output.records, selected_indices, output.packed_scalars, partition_included);
        output.marching_nodes = make_ooc_marching_nodes(output.records, selected_indices, balanced_records);
        output.marching_extract_before_reorder = build_ooc_marching_extract(output.marching_nodes);
        output.marching_extract = output.marching_extract_before_reorder;
        reorder_ooc_marching_child_groups(output.marching_extract);
        output.marching_active_cells = build_ooc_marching_active_cells(output.marching_extract);
        output.marching_initial_cell_states =
            build_ooc_marching_initial_cell_states(output.marching_extract, output.marching_active_cells);
        output.marching_cell_states =
            complete_ooc_marching_cell_states(output.marching_initial_cell_states, output.marching_active_cells);
        return output;
    }

    static RecoveredOocMultilevelModelOutput
    run_recovered_ooc_multilevel_model_impl(std::span<const OocOctreeRecord> histogram_records,
                                            std::span<const OocWeightedNodeRecord> balanced_records,
                                            std::span<const float> scalar_lut,
                                            std::span<const std::uint32_t> support_levels,
                                            double root_scale,
                                            const std::array<double, 16>& root_grid_to_world,
                                            std::optional<std::size_t> cuda_device_index,
                                            OocFusionCudaStats* cuda_stats,
                                            const OocFusionParameters& fusion_parameters)
    {
        if (histogram_records.empty() || balanced_records.empty() || scalar_lut.size() != 65536U ||
            !std::isfinite(root_scale) || !(root_scale > 0.0))
        {
            throw std::invalid_argument("invalid recovered OOC multilevel model input");
        }
        for (const double value : root_grid_to_world)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("OOC root-grid transform contains a non-finite value");
            }
        }

        std::uint32_t maximum_level = 0U;
        for (const auto& record : balanced_records)
            maximum_level = std::max(maximum_level, static_cast<std::uint32_t>(record.level));
        if (maximum_level == 0U || maximum_level >= 31U)
        {
            throw std::invalid_argument("OOC balanced tree maximum level is outside marching domain");
        }
        std::vector<std::uint32_t> stages(support_levels.size());
        std::copy(support_levels.begin(), support_levels.end(), stages.begin());
        if (stages.empty())
        {
            if (maximum_level > 6U)
            {
                throw std::invalid_argument("OOC multilevel schedule must be explicit above level six");
            }
            stages.push_back(maximum_level);
        }
        std::uint32_t previous_level = 0U;
        for (const std::uint32_t level : stages)
        {
            if (level == 0U || level > maximum_level || (previous_level != 0U && level <= previous_level))
            {
                throw std::invalid_argument("OOC multilevel support levels are invalid or unordered");
            }
            if (previous_level != 0U && level - previous_level != 2U)
            {
                throw std::invalid_argument("OOC mode-0 support levels must advance by two");
            }
            previous_level = level;
        }
        if (stages.back() != maximum_level)
        {
            throw std::invalid_argument("OOC multilevel schedule does not reach the tree maximum");
        }

        // Each stage includes the current support-level leaves as well as all
        // ancestors. Target fusion-entry captures have exactly the cumulative
        // balanced-node counts for levels 6/8/10/12; omitting the current leaves
        // produces a much smaller, observably wrong scalar field.
        if (cuda_stats != nullptr)
            *cuda_stats = {};
        std::vector<OocOctreeRecord> global_records(histogram_records.begin(), histogram_records.end());
        std::vector<std::uint8_t> has_previous_scalar(global_records.size(), 0U);
        const std::vector<float> scalar_values(scalar_lut.begin(), scalar_lut.end());
        RecoveredOocMultilevelModelOutput output;
        output.stages.reserve(stages.size());

        std::vector<OocOctreeRecord> final_records;
        std::vector<std::uint32_t> final_selected;
        for (const std::uint32_t support_level : stages)
        {
            if (fusion_parameters.is_cancelled && fusion_parameters.is_cancelled())
            {
                throw std::runtime_error("Recovered OOC model cancelled");
            }
            if (fusion_parameters.stage_progress)
                fusion_parameters.stage_progress(support_level);
            const auto stage_started = std::chrono::steady_clock::now();
            FlatCellMap<std::uint16_t> previous_scalars(global_records.size());
            std::size_t previous_scalar_count = 0U;
            for (std::size_t index = 0U; index != global_records.size(); ++index)
            {
                if (has_previous_scalar[index] == 0U)
                    continue;
                const auto [unused, inserted] =
                    previous_scalars.emplace(decode_cell(global_records[index]), global_records[index].trailing_field);
                if (!inserted)
                {
                    throw std::runtime_error("OOC prior scalar table contains a duplicate cell");
                }
                ++previous_scalar_count;
            }
            std::vector<std::size_t> local_to_global;
            local_to_global.reserve(global_records.size());
            std::vector<OocOctreeRecord> local_records;
            local_records.reserve(global_records.size());
            for (std::size_t index = 0U; index != global_records.size(); ++index)
            {
                if (global_records[index].level > support_level)
                    continue;
                OocOctreeRecord record = global_records[index];
                if (has_previous_scalar[index] != 0U)
                {
                    // +26 is the prior stage's binary16 output and +24 is the
                    // binary16 LUT index consumed by sub_1EAC320.
                    record.scalar_lut_index = record.trailing_field;
                }
                else if (previous_scalar_count != 0U)
                {
                    // Target level-8 entry records prove that newly introduced
                    // nodes inherit the binary16 scalar of their nearest node in
                    // the preceding support level: all level-7 child octets equal
                    // their level-6 parent, and all level-8 descendants equal the
                    // same nearest solved ancestor.  Walk upward rather than
                    // interpolating; the observed transfer is bit-exact.
                    CellKey ancestor = decode_cell(record);
                    const std::uint16_t* inherited = nullptr;
                    while (ancestor.level != 0U && inherited == nullptr)
                    {
                        --ancestor.level;
                        ancestor.x >>= 1U;
                        ancestor.y >>= 1U;
                        ancestor.z >>= 1U;
                        inherited = previous_scalars.find(ancestor);
                    }
                    if (inherited == nullptr)
                    {
                        throw std::runtime_error("OOC new multilevel node has no solved ancestor");
                    }
                    record.scalar_lut_index = *inherited;
                }
                local_to_global.push_back(index);
                local_records.push_back(record);
            }
            if (local_records.empty())
            {
                throw std::runtime_error("OOC multilevel stage has no persistent nodes");
            }
            const auto filtered_at = std::chrono::steady_clock::now();
            auto selected = select_ooc_octree_nodes_breadth_first(local_records);
            auto active = make_ooc_leaf_active_mask(local_records, selected);
            const auto topology_at = std::chrono::steady_clock::now();
            const std::uint64_t active_count =
                static_cast<std::uint64_t>(std::count(active.begin(), active.end(), std::uint8_t{1U}));
            std::uint64_t balanced_count = 0U;
            for (const auto& record : balanced_records)
            {
                if (record.level <= support_level)
                    ++balanced_count;
            }
            output.stages.push_back({support_level,
                                     balanced_count,
                                     static_cast<std::uint64_t>(local_records.size()),
                                     active_count,
                                     std::chrono::duration<double>(filtered_at - stage_started).count(),
                                     std::chrono::duration<double>(topology_at - filtered_at).count()});

            {
                OocFusionState fusion =
                    prepare_ooc_fusion_state(local_records, selected, active, scalar_values, {}, cuda_device_index);
                const auto prepared_at = std::chrono::steady_clock::now();
                output.stages.back().preparation_seconds =
                    std::chrono::duration<double>(prepared_at - topology_at).count();
                if (cuda_device_index.has_value())
                {
                    OocFusionCudaStats stage_stats;
                    std::string error;
                    if (!run_ooc_fusion_cuda(fusion, fusion_parameters, *cuda_device_index, stage_stats, error))
                    {
                        throw std::runtime_error("recovered OOC multilevel CUDA fusion failed: " + error);
                    }
                    if (cuda_stats != nullptr)
                    {
                        cuda_stats->kernel_launches += stage_stats.kernel_launches;
                        cuda_stats->stream_synchronizations += stage_stats.stream_synchronizations;
                        cuda_stats->host_to_device_bytes += stage_stats.host_to_device_bytes;
                        cuda_stats->device_to_host_bytes += stage_stats.device_to_host_bytes;
                    }
                }
                else
                {
                    run_ooc_fusion_cpu(fusion, fusion_parameters);
                }
                const auto fused_at = std::chrono::steady_clock::now();
                output.stages.back().fusion_seconds = std::chrono::duration<double>(fused_at - prepared_at).count();
                auto packed = pack_ooc_fused_scalars(fusion.u);
                write_ooc_fused_scalars(local_records, selected, packed);
            }
            for (std::size_t local = 0U; local != local_records.size(); ++local)
            {
                const std::size_t global = local_to_global[local];
                global_records[global].trailing_field = local_records[local].trailing_field;
                has_previous_scalar[global] = 1U;
            }
            output.stages.back().commit_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - topology_at).count() -
                output.stages.back().preparation_seconds - output.stages.back().fusion_seconds;
            if (support_level == maximum_level)
            {
                final_records = std::move(local_records);
                final_selected = std::move(selected);
            }
        }

        if (final_records.size() != global_records.size() || final_selected.size() != final_records.size())
        {
            throw std::runtime_error("OOC final multilevel stage does not cover persistent records");
        }
        output.records = std::move(final_records);
        const auto marching_started = std::chrono::steady_clock::now();
        auto marching_nodes = make_ooc_marching_nodes(output.records, final_selected, balanced_records);
        auto node_trim_attributes = make_ooc_marching_trim_attributes(marching_nodes);
        const auto marching_nodes_at = std::chrono::steady_clock::now();
        output.marching_nodes_seconds = std::chrono::duration<double>(marching_nodes_at - marching_started).count();
        auto marching_extract = build_ooc_marching_extract(marching_nodes);
        marching_nodes.clear();
        marching_nodes.shrink_to_fit();
        reorder_ooc_marching_child_groups(marching_extract);
        output.marching_work_cells = plan_ooc_marching_capacity_frontier(marching_extract);
        const auto marching_extract_at = std::chrono::steady_clock::now();
        output.marching_extract_seconds =
            std::chrono::duration<double>(marching_extract_at - marching_nodes_at).count();
        struct MarchingSourceGridCell
        {
            std::uint32_t level{};
            std::array<std::uint32_t, 3> coordinate{};
        };
        std::vector<MarchingSourceGridCell> source_grid(marching_extract.size());
        std::uint32_t marching_maximum_level = 0U;
        const auto map_source_grid = [&](auto&& self,
                                         const std::size_t node,
                                         const std::uint32_t level,
                                         const std::array<std::uint32_t, 3>& coordinate) -> void
        {
            source_grid[node] = {level, coordinate};
            marching_maximum_level = std::max(marching_maximum_level, level);
            const auto group = marching_extract[node].child_group;
            if (group == std::numeric_limits<std::uint32_t>::max())
                return;
            const std::size_t first = 1U + 8U * group;
            if (first + 7U >= marching_extract.size())
            {
                throw std::runtime_error("OOC bounded marching source tree is truncated");
            }
            for (std::size_t slot = 0U; slot != 8U; ++slot)
            {
                std::array<std::uint32_t, 3> child{};
                for (std::size_t axis = 0U; axis != 3U; ++axis)
                {
                    child[axis] = 2U * coordinate[axis] + static_cast<std::uint32_t>((slot >> axis) & 1U);
                }
                self(self, first + slot, level + 1U, child);
            }
        };
        map_source_grid(map_source_grid, 0U, 0U, {0U, 0U, 0U});
        if (marching_maximum_level == 0U || marching_maximum_level >= 31U)
        {
            throw std::runtime_error("OOC bounded marching tree maximum level is invalid");
        }
        output.marching_maximum_level = marching_maximum_level;
        const std::uint32_t grid_domain = std::uint32_t{1} << marching_maximum_level;
        double active_seconds = 0.0;
        double closure_seconds = 0.0;
        double edge_seconds = 0.0;
        double raw_seconds = 0.0;
        struct MarchingSeamPart
        {
            std::array<std::uint32_t, 3> lower{};
            std::array<std::uint32_t, 3> upper{};
            std::size_t vertex_begin{};
            std::size_t vertex_end{};
            std::size_t face_begin{};
            std::size_t face_end{};
        };
        std::vector<MarchingSeamPart> seam_parts;
        seam_parts.reserve(output.marching_work_cells.size());
        std::vector<std::array<double, 3>> raw_root_grid_positions;

        const auto expanded_bounds_for = [&](const OocMarchingWorkCell& work_cell)
        {
            const std::uint32_t shift = marching_maximum_level - work_cell.level;
            const std::array<std::uint32_t, 3> ownership_lower{
                work_cell.x << shift, work_cell.y << shift, work_cell.z << shift};
            const std::array<std::uint32_t, 3> ownership_upper{
                (work_cell.x + 1U) << shift, (work_cell.y + 1U) << shift, (work_cell.z + 1U) << shift};
            OocMarchingGridBounds expanded_bounds;
            expanded_bounds.level = marching_maximum_level;
            for (std::size_t axis = 0U; axis != 3U; ++axis)
            {
                expanded_bounds.minimum[axis] = ownership_lower[axis] == 0U ? 0U : ownership_lower[axis] - 1U;
                expanded_bounds.maximum[axis] =
                    ownership_upper[axis] == grid_domain ? grid_domain : ownership_upper[axis] + 1U;
            }
            return expanded_bounds;
        };
        const auto parallel_indices = [&](const std::size_t count, auto&& function)
        {
            if (count == 0U)
                return;
            std::atomic<std::size_t> next{0U};
            std::atomic<bool> stopped{false};
            std::exception_ptr failure;
            std::mutex failure_mutex;
            const std::size_t worker_count = std::min<std::size_t>(8U, count);
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (std::size_t worker = 0U; worker != worker_count; ++worker)
            {
                workers.emplace_back(
                    [&]
                    {
                        while (!stopped.load(std::memory_order_relaxed))
                        {
                            const std::size_t index = next.fetch_add(1U, std::memory_order_relaxed);
                            if (index >= count)
                                return;
                            try
                            {
                                function(index);
                            }
                            catch (...)
                            {
                                {
                                    std::lock_guard<std::mutex> lock(failure_mutex);
                                    if (!failure)
                                        failure = std::current_exception();
                                }
                                stopped.store(true, std::memory_order_relaxed);
                                return;
                            }
                        }
                    });
            }
            for (auto& worker : workers)
                worker.join();
            if (failure)
                std::rethrow_exception(failure);
        };
        const auto parallel_parts = [&](auto&& function)
        { parallel_indices(output.marching_work_cells.size(), function); };

        std::vector<OocMarchingActiveCells> active_parts(output.marching_work_cells.size());
        const auto active_started = std::chrono::steady_clock::now();
        parallel_parts(
            [&](const std::size_t index)
            {
                const auto expanded_bounds = expanded_bounds_for(output.marching_work_cells[index]);
                // plan_ooc_marching_capacity_frontier has already traversed and
                // validated this immutable tree in full, and map_source_grid above
                // established the same maximum level.  Repeating the O(N) structural
                // validation for every bounded work cell changed no active-cell
                // semantics and dominated South's 43-part active stage.
                active_parts[index] =
                    build_ooc_marching_active_cells_impl(marching_extract, expanded_bounds, marching_maximum_level);
            });
        active_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - active_started).count();

        std::vector<std::vector<OocMarchingCellState>> initial_parts(active_parts.size());
        std::vector<std::size_t> initial_counts(active_parts.size(), 0U);
        parallel_parts(
            [&](const std::size_t index)
            {
                if (!active_parts[index].entries.empty())
                {
                    initial_parts[index] =
                        build_ooc_marching_initial_cell_states(marching_extract, active_parts[index]);
                    initial_counts[index] = initial_parts[index].size();
                }
            });
        std::vector<std::vector<OocMarchingCellState>> closed_parts(active_parts.size());
        std::vector<std::uint64_t> root_states(active_parts.size(), std::numeric_limits<std::uint64_t>::max());
        const auto closure_started = std::chrono::steady_clock::now();
        parallel_parts(
            [&](const std::size_t index)
            {
                if (active_parts[index].entries.empty())
                    return;
                closed_parts[index] =
                    complete_ooc_marching_cell_states(std::move(initial_parts[index]), active_parts[index]);
                for (const auto& entry : active_parts[index].entries)
                {
                    if (entry.selected_node_index == 0U)
                    {
                        root_states[index] = entry.cell_state_index;
                        break;
                    }
                }
                if (root_states[index] == std::numeric_limits<std::uint64_t>::max())
                {
                    throw std::runtime_error("OOC bounded marching active set has no root state");
                }
            });
        closure_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - closure_started).count();
        for (std::size_t index = 0U; index != active_parts.size(); ++index)
        {
            output.marching_initial_cell_count += initial_counts[index];
            output.marching_closed_cell_count += closed_parts[index].size();
        }
        active_parts.clear();
        active_parts.shrink_to_fit();
        initial_parts.clear();
        initial_parts.shrink_to_fit();

        struct LocalMarchingRawPart
        {
            OocMarchingRawOutput raw;
            std::vector<std::array<double, 3>> root_grid_positions;
            std::array<std::uint32_t, 3> ownership_lower{};
            std::array<std::uint32_t, 3> ownership_upper{};
            double edge_seconds{};
            double raw_seconds{};
        };
        std::vector<std::optional<LocalMarchingRawPart>> raw_parts(output.marching_work_cells.size());
        parallel_parts(
            [&](const std::size_t work_index)
            {
                const auto& work_cell = output.marching_work_cells[work_index];
                const std::uint32_t shift = marching_maximum_level - work_cell.level;
                const std::array<std::uint32_t, 3> ownership_lower{
                    work_cell.x << shift, work_cell.y << shift, work_cell.z << shift};
                const std::array<std::uint32_t, 3> ownership_upper{
                    (work_cell.x + 1U) << shift, (work_cell.y + 1U) << shift, (work_cell.z + 1U) << shift};
                LocalMarchingRawPart local;
                local.ownership_lower = ownership_lower;
                local.ownership_upper = ownership_upper;

                if (root_states[work_index] == std::numeric_limits<std::uint64_t>::max())
                {
                    raw_parts[work_index] = std::move(local);
                    return;
                }
                auto closed_cells = std::move(closed_parts[work_index]);
                const std::uint64_t root_state = root_states[work_index];
                const auto edges_started = std::chrono::steady_clock::now();
                auto edges = build_ooc_marching_edge_mesh(
                    std::move(closed_cells), root_state, marching_maximum_level, root_scale, root_grid_to_world);
                local.edge_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - edges_started).count();
                const auto raw_started = std::chrono::steady_clock::now();
                const auto part = build_ooc_marching_raw_mesh(edges, root_state, marching_maximum_level, true);
                if (edges.vertex_root_grid_position.size() != part.vertices.size())
                {
                    throw std::runtime_error("OOC bounded marching lost root-grid vertex coordinates");
                }
                const auto source_overlaps_ownership = [&](const std::uint32_t source)
                {
                    if (source >= source_grid.size())
                    {
                        throw std::runtime_error("OOC bounded marching source exceeds source tree");
                    }
                    const auto& cell = source_grid[source];
                    const std::uint32_t source_shift = marching_maximum_level - cell.level;
                    const std::uint32_t width = std::uint32_t{1} << source_shift;
                    for (std::size_t axis = 0U; axis != 3U; ++axis)
                    {
                        const std::uint32_t lower = cell.coordinate[axis] << source_shift;
                        if (ownership_lower[axis] >= lower + width || ownership_upper[axis] <= lower)
                            return false;
                    }
                    return true;
                };
                std::vector<std::uint32_t> remap(part.vertices.size(), std::numeric_limits<std::uint32_t>::max());
                const double root_grid_unit = root_scale / static_cast<double>(grid_domain);
                for (std::size_t index = 0U; index != part.vertices.size(); ++index)
                {
                    bool owned = true;
                    for (std::size_t axis = 0U; axis != 3U; ++axis)
                    {
                        const double coordinate = edges.vertex_root_grid_position[index][axis];
                        owned = owned && coordinate >= root_grid_unit * ownership_lower[axis] &&
                                coordinate <= root_grid_unit * ownership_upper[axis];
                    }
                    if (!owned)
                        continue;
                    if (local.raw.vertices.size() >= std::numeric_limits<std::uint32_t>::max())
                    {
                        throw std::runtime_error("OOC bounded marching vertex index exceeds uint32");
                    }
                    remap[index] = static_cast<std::uint32_t>(local.raw.vertices.size());
                    local.raw.vertices.push_back(part.vertices[index]);
                    local.raw.vertex_scale.push_back(part.vertex_scale[index]);
                    const std::uint32_t source = part.vertex_source[index];
                    local.raw.vertex_source.push_back(source);
                    if (source >= node_trim_attributes.size())
                    {
                        throw std::runtime_error("OOC raw vertex trim source exceeds marching nodes");
                    }
                    local.raw.vertex_trim_attribute.push_back(node_trim_attributes[source]);
                    local.root_grid_positions.push_back(edges.vertex_root_grid_position[index]);
                }
                for (std::size_t index = 0U; index != part.triangles.size(); ++index)
                {
                    const auto& source_triangle = part.triangles[index].vertices;
                    if (remap[source_triangle[0]] == std::numeric_limits<std::uint32_t>::max() ||
                        remap[source_triangle[1]] == std::numeric_limits<std::uint32_t>::max() ||
                        remap[source_triangle[2]] == std::numeric_limits<std::uint32_t>::max() ||
                        !source_overlaps_ownership(part.face_source[index]))
                    {
                        continue;
                    }
                    local.raw.triangles.push_back(
                        {{remap[source_triangle[0]], remap[source_triangle[1]], remap[source_triangle[2]]}});
                    local.raw.face_source.push_back(part.face_source[index]);
                }
                local.raw_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - raw_started).count();
                raw_parts[work_index] = std::move(local);
            });

        std::size_t total_raw_vertices = 0U;
        std::size_t total_raw_faces = 0U;
        for (const auto& local : raw_parts)
        {
            if (!local.has_value())
            {
                throw std::runtime_error("OOC bounded marching worker omitted a raw part");
            }
            total_raw_vertices += local->raw.vertices.size();
            total_raw_faces += local->raw.triangles.size();
        }
        if (total_raw_vertices > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("OOC bounded marching vertex index exceeds uint32");
        }
        output.raw_mesh.vertices.reserve(total_raw_vertices);
        output.raw_mesh.vertex_scale.reserve(total_raw_vertices);
        output.raw_mesh.vertex_source.reserve(total_raw_vertices);
        output.raw_mesh.vertex_trim_attribute.reserve(total_raw_vertices);
        raw_root_grid_positions.reserve(total_raw_vertices);
        output.raw_mesh.triangles.reserve(total_raw_faces);
        output.raw_mesh.face_source.reserve(total_raw_faces);
        for (std::size_t work_index = 0U; work_index != raw_parts.size(); ++work_index)
        {
            auto& local = *raw_parts[work_index];
            const std::size_t part_vertex_begin = output.raw_mesh.vertices.size();
            const std::size_t part_face_begin = output.raw_mesh.triangles.size();
            output.raw_mesh.vertices.insert(output.raw_mesh.vertices.end(),
                                            std::make_move_iterator(local.raw.vertices.begin()),
                                            std::make_move_iterator(local.raw.vertices.end()));
            output.raw_mesh.vertex_scale.insert(
                output.raw_mesh.vertex_scale.end(), local.raw.vertex_scale.begin(), local.raw.vertex_scale.end());
            output.raw_mesh.vertex_source.insert(
                output.raw_mesh.vertex_source.end(), local.raw.vertex_source.begin(), local.raw.vertex_source.end());
            output.raw_mesh.vertex_trim_attribute.insert(
                output.raw_mesh.vertex_trim_attribute.end(),
                std::make_move_iterator(local.raw.vertex_trim_attribute.begin()),
                std::make_move_iterator(local.raw.vertex_trim_attribute.end()));
            raw_root_grid_positions.insert(
                raw_root_grid_positions.end(), local.root_grid_positions.begin(), local.root_grid_positions.end());
            for (auto triangle : local.raw.triangles)
            {
                for (auto& vertex : triangle.vertices)
                    vertex += static_cast<std::uint32_t>(part_vertex_begin);
                output.raw_mesh.triangles.push_back(triangle);
            }
            output.raw_mesh.face_source.insert(
                output.raw_mesh.face_source.end(), local.raw.face_source.begin(), local.raw.face_source.end());
            seam_parts.push_back({local.ownership_lower,
                                  local.ownership_upper,
                                  part_vertex_begin,
                                  output.raw_mesh.vertices.size(),
                                  part_face_begin,
                                  output.raw_mesh.triangles.size()});
            edge_seconds += local.edge_seconds;
            raw_seconds += local.raw_seconds;
            raw_parts[work_index].reset();
        }
        // sub_1EA0680 does not pass the concatenated per-part surface straight to
        // QEM.  It first joins vertices on touching part boundaries.  The worker
        // local worker first marks the endpoints of topology boundary edges
        // (undirected edges referenced by exactly one local triangle).  The
        // worker at sub_1E9E5A0 considers only those vertices, tests whether they
        // lie within one sixteenth of a finest-grid cell from an ownership-box
        // face, then links them to the nearest eligible vertex in every
        // compatible neighboring part.  The
        // connected component's lowest concatenated index is retained.  This is
        // deliberately not a global position hash: coincident vertices inside a
        // single part represent distinct marching topology and must survive.
        if (raw_root_grid_positions.size() != output.raw_mesh.vertices.size())
        {
            throw std::runtime_error("OOC marching seam coordinate count mismatch");
        }
        if (!output.raw_mesh.vertices.empty() && seam_parts.size() > 1U)
        {
            struct SeamNeighbor
            {
                std::size_t part{};
                std::uint8_t boundary_mask{};
            };
            std::vector<std::vector<SeamNeighbor>> neighbors(seam_parts.size());
            for (std::size_t left = 0U; left != seam_parts.size(); ++left)
            {
                for (std::size_t right = left + 1U; right != seam_parts.size(); ++right)
                {
                    std::uint8_t left_mask = 0U;
                    std::uint8_t right_mask = 0U;
                    for (std::size_t axis = 0U; axis != 3U; ++axis)
                    {
                        const std::size_t other_a = (axis + 1U) % 3U;
                        const std::size_t other_b = (axis + 2U) % 3U;
                        const bool overlaps = seam_parts[left].upper[other_a] >= seam_parts[right].lower[other_a] &&
                                              seam_parts[right].upper[other_a] >= seam_parts[left].lower[other_a] &&
                                              seam_parts[left].upper[other_b] >= seam_parts[right].lower[other_b] &&
                                              seam_parts[right].upper[other_b] >= seam_parts[left].lower[other_b];
                        if (!overlaps)
                            continue;
                        if (seam_parts[left].lower[axis] == seam_parts[right].upper[axis])
                        {
                            left_mask |= static_cast<std::uint8_t>(1U << (2U * axis));
                            right_mask |= static_cast<std::uint8_t>(1U << (2U * axis + 1U));
                        }
                        else if (seam_parts[left].upper[axis] == seam_parts[right].lower[axis])
                        {
                            left_mask |= static_cast<std::uint8_t>(1U << (2U * axis + 1U));
                            right_mask |= static_cast<std::uint8_t>(1U << (2U * axis));
                        }
                    }
                    if (left_mask != 0U)
                    {
                        neighbors[left].push_back({right, left_mask});
                        neighbors[right].push_back({left, right_mask});
                    }
                }
            }

            // sub_1E9F160 sorts the three undirected edges of every local face;
            // singleton edge groups are the local surface boundary.  Its bitset
            // is passed to sub_1E9E5A0 as an eligibility mask.  Reconstruct this
            // per part: a global positional test alone over-merges vertices that
            // are close to a partition face but are not on a local boundary edge.
            std::vector<std::uint8_t> seam_eligible(output.raw_mesh.vertices.size(), 0U);
            parallel_indices(seam_parts.size(),
                             [&](const std::size_t part_index)
                             {
                                 const auto& part = seam_parts[part_index];
                                 std::vector<std::uint64_t> edges;
                                 edges.reserve(3U * (part.face_end - part.face_begin));
                                 for (std::size_t face = part.face_begin; face != part.face_end; ++face)
                                 {
                                     const auto& triangle = output.raw_mesh.triangles[face].vertices;
                                     for (std::size_t edge = 0U; edge != 3U; ++edge)
                                     {
                                         const std::uint32_t a = triangle[edge];
                                         const std::uint32_t b = triangle[(edge + 1U) % 3U];
                                         const std::uint32_t lower = std::min(a, b);
                                         const std::uint32_t upper = std::max(a, b);
                                         edges.push_back((static_cast<std::uint64_t>(lower) << 32U) |
                                                         static_cast<std::uint64_t>(upper));
                                     }
                                 }
                                 std::sort(edges.begin(), edges.end());
                                 for (std::size_t begin = 0U; begin != edges.size();)
                                 {
                                     std::size_t end = begin + 1U;
                                     while (end != edges.size() && edges[end] == edges[begin])
                                         ++end;
                                     if (end == begin + 1U)
                                     {
                                         seam_eligible[static_cast<std::uint32_t>(edges[begin] >> 32U)] = 1U;
                                         seam_eligible[static_cast<std::uint32_t>(edges[begin])] = 1U;
                                     }
                                     begin = end;
                                 }
                             });

            std::vector<std::uint8_t> boundary_mask(output.raw_mesh.vertices.size(), 0U);
            const double finest_scale = root_scale / static_cast<double>(std::uint32_t{1} << marching_maximum_level);
            const double boundary_epsilon = finest_scale * 0.0625;
            parallel_indices(seam_parts.size(),
                             [&](const std::size_t part_index)
                             {
                                 const auto& part = seam_parts[part_index];
                                 for (std::size_t vertex = part.vertex_begin; vertex != part.vertex_end; ++vertex)
                                 {
                                     if (seam_eligible[vertex] == 0U)
                                         continue;
                                     std::uint8_t mask = 0U;
                                     for (std::size_t axis = 0U; axis != 3U; ++axis)
                                     {
                                         const double minimum = finest_scale * part.lower[axis];
                                         const double maximum = finest_scale * part.upper[axis];
                                         const double coordinate = raw_root_grid_positions[vertex][axis];
                                         if (minimum + boundary_epsilon > coordinate)
                                         {
                                             mask |= static_cast<std::uint8_t>(1U << (2U * axis));
                                         }
                                         else if (coordinate > maximum - boundary_epsilon)
                                         {
                                             mask |= static_cast<std::uint8_t>(1U << (2U * axis + 1U));
                                         }
                                     }
                                     boundary_mask[vertex] = mask;
                                 }
                             });

            std::vector<std::uint32_t> parent(output.raw_mesh.vertices.size());
            for (std::size_t index = 0U; index != parent.size(); ++index)
                parent[index] = static_cast<std::uint32_t>(index);
            const auto squared_distance = [&](const std::size_t left, const std::size_t right)
            {
                const auto& a = output.raw_mesh.vertices[left].position;
                const auto& b = output.raw_mesh.vertices[right].position;
                const float dx = static_cast<float>(b[0] - a[0]);
                const float dy = static_cast<float>(b[1] - a[1]);
                const float dz = static_cast<float>(b[2] - a[2]);
                return static_cast<float>(static_cast<float>(dx * dx + dy * dy) + dz * dz);
            };
            parallel_indices(seam_parts.size(),
                             [&](const std::size_t part_index)
                             {
                                 const auto& part = seam_parts[part_index];
                                 for (std::size_t vertex = part.vertex_begin; vertex != part.vertex_end; ++vertex)
                                 {
                                     if (boundary_mask[vertex] == 0U)
                                         continue;
                                     float best_distance = std::numeric_limits<float>::max();
                                     std::uint32_t best = static_cast<std::uint32_t>(vertex);
                                     bool found = false;
                                     for (const auto& neighbor : neighbors[part_index])
                                     {
                                         if ((boundary_mask[vertex] & neighbor.boundary_mask) == 0U)
                                             continue;
                                         const auto& candidate_part = seam_parts[neighbor.part];
                                         for (std::size_t candidate = candidate_part.vertex_begin;
                                              candidate != candidate_part.vertex_end;
                                              ++candidate)
                                         {
                                             if (boundary_mask[candidate] == 0U)
                                                 continue;
                                             const float distance = squared_distance(vertex, candidate);
                                             if (!found || best_distance > distance)
                                             {
                                                 best_distance = distance;
                                                 best = static_cast<std::uint32_t>(candidate);
                                                 found = true;
                                             }
                                         }
                                     }
                                     if (found)
                                         parent[vertex] = best;
                                 }
                             });
            const auto nearest = parent;
            for (std::size_t vertex = 0U; vertex != parent.size(); ++vertex)
                parent[vertex] = static_cast<std::uint32_t>(vertex);
            const auto find_root = [&](auto&& self, std::uint32_t vertex) -> std::uint32_t
            {
                if (parent[vertex] == vertex)
                    return vertex;
                parent[vertex] = self(self, parent[vertex]);
                return parent[vertex];
            };
            const auto unite = [&](const std::uint32_t left, const std::uint32_t right)
            {
                const std::uint32_t a = find_root(find_root, left);
                const std::uint32_t b = find_root(find_root, right);
                if (a == b)
                    return;
                const std::uint32_t lower = std::min(a, b);
                const std::uint32_t upper = std::max(a, b);
                parent[upper] = lower;
            };
            for (std::size_t vertex = 0U; vertex != nearest.size(); ++vertex)
            {
                if (nearest[vertex] != vertex)
                {
                    unite(static_cast<std::uint32_t>(vertex), nearest[vertex]);
                }
            }
            for (std::size_t vertex = 0U; vertex != parent.size(); ++vertex)
                parent[vertex] = find_root(find_root, static_cast<std::uint32_t>(vertex));

            OocMarchingRawOutput joined;
            std::vector<std::uint32_t> compact(parent.size(), std::numeric_limits<std::uint32_t>::max());
            for (std::size_t vertex = 0U; vertex != parent.size(); ++vertex)
            {
                if (parent[vertex] != vertex)
                    continue;
                compact[vertex] = static_cast<std::uint32_t>(joined.vertices.size());
                joined.vertices.push_back(output.raw_mesh.vertices[vertex]);
                joined.vertex_scale.push_back(output.raw_mesh.vertex_scale[vertex]);
                joined.vertex_source.push_back(output.raw_mesh.vertex_source[vertex]);
                joined.vertex_trim_attribute.push_back(output.raw_mesh.vertex_trim_attribute[vertex]);
            }
            for (std::size_t vertex = 0U; vertex != parent.size(); ++vertex)
                compact[vertex] = compact[parent[vertex]];
            for (std::size_t face = 0U; face != output.raw_mesh.triangles.size(); ++face)
            {
                auto triangle = output.raw_mesh.triangles[face];
                for (auto& vertex : triangle.vertices)
                    vertex = compact[vertex];
                if (triangle.vertices[0] == triangle.vertices[1] || triangle.vertices[1] == triangle.vertices[2] ||
                    triangle.vertices[2] == triangle.vertices[0])
                {
                    continue;
                }
                joined.triangles.push_back(triangle);
                joined.face_source.push_back(output.raw_mesh.face_source[face]);
            }
            output.raw_mesh = std::move(joined);
        }

        // The caller of sub_1EA0680 does one topology-aware attribute pass before
        // entering mode-3 QEM.  It snapshots channel 1, then visits all three
        // undirected edges of every retained triangle and MAXSS-propagates the
        // frozen value in both directions.  Because the source is frozen this is
        // exactly a one-ring maximum, not an iterative flood fill.  Channels
        // 0/2/3 remain the direct source-node values built above.
        std::vector<float> original_histogram_sum(output.raw_mesh.vertex_trim_attribute.size());
        for (std::size_t vertex = 0U; vertex != output.raw_mesh.vertex_trim_attribute.size(); ++vertex)
        {
            original_histogram_sum[vertex] = output.raw_mesh.vertex_trim_attribute[vertex][1];
        }
        for (const OocMarchingTriangle& triangle : output.raw_mesh.triangles)
        {
            for (std::size_t edge = 0U; edge != 3U; ++edge)
            {
                const std::uint32_t left = triangle.vertices[edge];
                const std::uint32_t right = triangle.vertices[(edge + 1U) % 3U];
                if (left >= original_histogram_sum.size() || right >= original_histogram_sum.size())
                {
                    throw std::runtime_error("OOC trim-attribute edge exceeds vertex table");
                }
                auto& left_value = output.raw_mesh.vertex_trim_attribute[left][1];
                auto& right_value = output.raw_mesh.vertex_trim_attribute[right][1];
                left_value = std::max(left_value, original_histogram_sum[right]);
                right_value = std::max(right_value, original_histogram_sum[left]);
            }
        }
        marching_extract.clear();
        marching_extract.shrink_to_fit();
        source_grid.clear();
        source_grid.shrink_to_fit();
        output.marching_active_seconds = active_seconds;
        output.marching_closure_seconds = closure_seconds;
        output.marching_edges_seconds = edge_seconds;
        output.marching_raw_seconds = raw_seconds;
        validate_ooc_marching_raw_output(output.raw_mesh);
        return output;
    }

    RecoveredOocMultilevelModelOutput
    run_recovered_ooc_multilevel_model_cpu(std::span<const OocOctreeRecord> histogram_records,
                                           std::span<const OocWeightedNodeRecord> balanced_records,
                                           std::span<const float> scalar_lut,
                                           std::span<const std::uint32_t> support_levels,
                                           double root_scale,
                                           const std::array<double, 16>& root_grid_to_world,
                                           const OocFusionParameters& fusion_parameters)
    {
        return run_recovered_ooc_multilevel_model_impl(histogram_records,
                                                       balanced_records,
                                                       scalar_lut,
                                                       support_levels,
                                                       root_scale,
                                                       root_grid_to_world,
                                                       std::nullopt,
                                                       nullptr,
                                                       fusion_parameters);
    }

    RecoveredOocMultilevelModelOutput
    run_recovered_ooc_multilevel_model_cuda(std::span<const OocOctreeRecord> histogram_records,
                                            std::span<const OocWeightedNodeRecord> balanced_records,
                                            std::span<const float> scalar_lut,
                                            std::span<const std::uint32_t> support_levels,
                                            double root_scale,
                                            const std::array<double, 16>& root_grid_to_world,
                                            std::size_t device_index,
                                            OocFusionCudaStats& cuda_stats,
                                            const OocFusionParameters& fusion_parameters)
    {
        return run_recovered_ooc_multilevel_model_impl(histogram_records,
                                                       balanced_records,
                                                       scalar_lut,
                                                       support_levels,
                                                       root_scale,
                                                       root_grid_to_world,
                                                       device_index,
                                                       &cuda_stats,
                                                       fusion_parameters);
    }

    void validate_ooc_marching_raw_output(const OocMarchingRawOutput& output)
    {
        if (output.vertex_scale.size() != output.vertices.size() ||
            output.vertex_source.size() != output.vertices.size())
        {
            throw std::runtime_error("OOC marching per-vertex output size mismatch");
        }
        if (output.face_source.size() != output.triangles.size())
        {
            throw std::runtime_error("OOC marching per-face output size mismatch");
        }
        if (!output.vertex_trim_attribute.empty() && output.vertex_trim_attribute.size() != output.vertices.size())
        {
            throw std::runtime_error("OOC marching trim-attribute output size mismatch");
        }
        for (const auto& vertex : output.vertices)
        {
            if (vertex.trailing_word != 0U)
            {
                throw std::runtime_error("OOC marching vertex trailing word is not zero");
            }
        }
        for (const auto& triangle : output.triangles)
        {
            for (const auto index : triangle.vertices)
            {
                if (index >= output.vertices.size())
                {
                    throw std::runtime_error("OOC marching triangle index is out of range");
                }
            }
        }
    }

    OocMarchingEdgeOutput make_ooc_marching_edge_vertex(const OocMarchingEdgeInput& input)
    {
        if (input.edge >= 12U)
        {
            throw std::runtime_error("OOC marching edge index exceeds twelve-edge cube");
        }
        if (input.child_slot >= 8U)
        {
            throw std::runtime_error("OOC marching child slot exceeds octet");
        }

        const std::uint32_t axis = input.edge >> 2U;
        const std::uint32_t quadrant = input.edge & 3U;
        std::array<std::uint32_t, 3> low{};
        if (axis == 0U)
        {
            low = {0U, quadrant & 1U, quadrant >> 1U};
        }
        else if (axis == 1U)
        {
            low = {quadrant & 1U, 0U, quadrant >> 1U};
        }
        else
        {
            low = {quadrant & 1U, quadrant >> 1U, 0U};
        }
        auto high = low;
        high[axis] = 1U;
        const std::uint32_t low_corner = low[0] | (low[1] << 1U) | (low[2] << 2U);
        const std::uint32_t high_corner = high[0] | (high[1] << 1U) | (high[2] << 2U);

        // SHL r32,cl masks the count to five bits. CVTSI2SD consumes the result as
        // signed int32. Retaining this behavior matters for the target's declared
        // level range even though observed marching samples use levels 3..6.
        const std::uint32_t divisor_bits = 1U << ((input.level + 1U) & 31U);
        const double cell_scale = input.root_scale / static_cast<double>(std::bit_cast<std::int32_t>(divisor_bits));
        const double high_weight = static_cast<double>(std::fabs(input.corner_scalar[high_corner]));
        const double low_weight = static_cast<double>(std::fabs(input.corner_scalar[low_corner]));
        const double weight_sum = low_weight + high_weight;

        std::array<double, 3> grid{};
        for (std::size_t coordinate = 0; coordinate < grid.size(); ++coordinate)
        {
            const std::uint32_t cell_bits = std::bit_cast<std::uint32_t>(input.cell[coordinate]);
            const std::uint32_t child_bit = (input.child_slot >> coordinate) & 1U;
            const std::uint32_t base = cell_bits * 2U + child_bit;
            const auto low_coordinate = std::bit_cast<std::int32_t>(base + low[coordinate]);
            const auto high_coordinate = std::bit_cast<std::int32_t>(base + high[coordinate]);

            double low_term = static_cast<double>(low_coordinate);
            low_term = low_term * cell_scale;
            low_term = low_term * high_weight;
            double high_term = static_cast<double>(high_coordinate);
            high_term = high_term * cell_scale;
            high_term = high_term * low_weight;
            grid[coordinate] = (low_term + high_term) / weight_sum;
        }

        double denominator = input.transform[12] * grid[0];
        denominator = denominator + input.transform[13] * grid[1];
        denominator = denominator + input.transform[14] * grid[2];
        denominator = denominator + input.transform[15];

        OocMarchingEdgeOutput output;
        for (std::size_t row = 0; row < 3U; ++row)
        {
            double numerator = input.transform[4U * row] * grid[0];
            numerator = numerator + input.transform[4U * row + 1U] * grid[1];
            numerator = numerator + input.transform[4U * row + 2U] * grid[2];
            numerator = numerator + input.transform[4U * row + 3U];
            output.vertex.position[row] = static_cast<float>(numerator / denominator);
        }
        output.vertex.trailing_word = 0U;
        output.vertex_scale =
            input.weighted_cell_scale <= 0.0F ? static_cast<float>(cell_scale * 0.5) : input.weighted_cell_scale;
        output.source_node_index = input.source_node_index;
        output.root_grid_position = grid;
        return output;
    }

    OocMarchingEdgeMesh build_ooc_marching_edge_mesh(std::vector<OocMarchingCellState> cell_states,
                                                     std::uint64_t root_state_index,
                                                     std::uint32_t maximum_level,
                                                     double root_scale,
                                                     const std::array<double, 16>& transform)
    {
        if (cell_states.empty() || root_state_index >= cell_states.size())
        {
            throw std::runtime_error("OOC marching edge tree has no valid root");
        }
        if (maximum_level == 0U || maximum_level >= 31U)
        {
            throw std::runtime_error("OOC marching edge level exceeds int32 grid domain");
        }

        struct StateLocation
        {
            std::uint32_t level{};
            std::array<std::uint32_t, 3> cell{};
            std::int64_t parent{-1};
            std::uint32_t child_slot{};
            bool seen{};
        };
        std::vector<StateLocation> locations(cell_states.size());
        std::vector<std::size_t> leaf_order;
        std::function<void(
            std::size_t, std::uint32_t, const std::array<std::uint32_t, 3>&, std::int64_t, std::uint32_t)>
            locate = [&](std::size_t state_index,
                         std::uint32_t level,
                         const std::array<std::uint32_t, 3>& cell,
                         std::int64_t parent,
                         std::uint32_t child_slot)
        {
            if (state_index >= cell_states.size())
            {
                throw std::runtime_error("OOC marching child index is out of range");
            }
            auto& location = locations[state_index];
            if (location.seen)
            {
                throw std::runtime_error("OOC marching edge tree shares or cycles a child");
            }
            if (level > maximum_level)
            {
                throw std::runtime_error("OOC marching edge tree exceeds maximum level");
            }
            location = StateLocation{level, cell, parent, child_slot, true};

            bool has_child = false;
            for (std::size_t slot = 0U; slot != 8U; ++slot)
            {
                const auto child = cell_states[state_index].child_state[slot];
                if (child < 0)
                    continue;
                has_child = true;
                std::array<std::uint32_t, 3> child_cell{};
                for (std::size_t axis = 0U; axis != 3U; ++axis)
                {
                    child_cell[axis] = 2U * cell[axis] + ((slot >> axis) & 1U);
                }
                locate(static_cast<std::size_t>(child),
                       level + 1U,
                       child_cell,
                       static_cast<std::int64_t>(state_index),
                       static_cast<std::uint32_t>(slot));
            }
            if (!has_child)
                leaf_order.push_back(state_index);
        };
        locate(static_cast<std::size_t>(root_state_index), 0U, {0U, 0U, 0U}, -1, 0U);
        if (std::find_if(locations.begin(), locations.end(), [](const StateLocation& value) { return !value.seen; }) !=
            locations.end())
        {
            throw std::runtime_error("OOC marching edge tree has unreachable states");
        }

        struct SegmentKey
        {
            std::uint32_t axis{};
            std::array<std::uint32_t, 3> low{};
            std::array<std::uint32_t, 3> high{};
            bool operator==(const SegmentKey&) const = default;
        };
        struct SegmentKeyHash
        {
            std::size_t operator()(const SegmentKey& key) const noexcept
            {
                std::size_t value = key.axis;
                for (const auto coordinate : key.low)
                {
                    value ^=
                        static_cast<std::size_t>(coordinate) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
                }
                for (const auto coordinate : key.high)
                {
                    value ^=
                        static_cast<std::size_t>(coordinate) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
                }
                return value;
            }
        };
        const auto endpoints = [](std::uint32_t edge)
        {
            const std::uint32_t axis = edge >> 2U;
            const std::uint32_t quadrant = edge & 3U;
            std::array<std::uint32_t, 3> low{};
            if (axis == 0U)
            {
                low = {0U, quadrant & 1U, quadrant >> 1U};
            }
            else if (axis == 1U)
            {
                low = {quadrant & 1U, 0U, quadrant >> 1U};
            }
            else
            {
                low = {quadrant & 1U, quadrant >> 1U, 0U};
            }
            auto high = low;
            high[axis] = 1U;
            return std::array<std::array<std::uint32_t, 3>, 2>{low, high};
        };
        const auto corner_index = [](const std::array<std::uint32_t, 3>& p)
        { return p[0] | (p[1] << 1U) | (p[2] << 2U); };

        std::unordered_map<SegmentKey, std::size_t, SegmentKeyHash> segment_index;
        std::vector<SegmentKey> segments;
        std::vector<std::array<std::int64_t, 12>> leaf_segment(cell_states.size());
        for (auto& item : leaf_segment)
            item.fill(-1);
        for (const auto state_index : leaf_order)
        {
            auto& state = cell_states[state_index];
            state.edge_vertex.fill(-1);
            const auto& location = locations[state_index];
            const std::uint32_t scale = 1U << (maximum_level - location.level);
            for (std::uint32_t edge = 0U; edge != 12U; ++edge)
            {
                const auto points = endpoints(edge);
                const auto low_corner = corner_index(points[0]);
                const auto high_corner = corner_index(points[1]);
                if ((state.corner_scalar[low_corner] > 0.0F) == (state.corner_scalar[high_corner] > 0.0F))
                {
                    continue;
                }
                SegmentKey key;
                key.axis = edge >> 2U;
                for (std::size_t axis = 0U; axis != 3U; ++axis)
                {
                    key.low[axis] = (location.cell[axis] + points[0][axis]) * scale;
                    key.high[axis] = (location.cell[axis] + points[1][axis]) * scale;
                }
                const auto [found, inserted] = segment_index.emplace(key, segments.size());
                if (inserted)
                    segments.push_back(key);
                leaf_segment[state_index][edge] = static_cast<std::int64_t>(found->second);
            }
        }

        std::vector<std::size_t> parent(segments.size());
        for (std::size_t index = 0U; index != parent.size(); ++index)
            parent[index] = index;
        const auto find_root = [&](std::size_t index, const auto& self) -> std::size_t
        {
            if (parent[index] != index)
                parent[index] = self(parent[index], self);
            return parent[index];
        };
        const auto unite = [&](std::size_t left, std::size_t right)
        {
            left = find_root(left, find_root);
            right = find_root(right, find_root);
            if (left != right)
                parent[right] = left;
        };
        for (std::size_t index = 0U; index != segments.size(); ++index)
        {
            const auto& segment = segments[index];
            const std::uint32_t length = segment.high[segment.axis] - segment.low[segment.axis];
            if (length > (1U << (maximum_level - 1U)))
                continue;
            SegmentKey coarse = segment;
            const std::uint32_t coarse_length = 2U * length;
            coarse.low[coarse.axis] = (coarse.low[coarse.axis] / coarse_length) * coarse_length;
            coarse.high = coarse.low;
            coarse.high[coarse.axis] += coarse_length;
            const auto found = segment_index.find(coarse);
            if (found != segment_index.end())
                unite(index, found->second);
        }

        std::vector<std::uint32_t> finest_length(segments.size(), std::numeric_limits<std::uint32_t>::max());
        for (std::size_t index = 0U; index != segments.size(); ++index)
        {
            const auto root = find_root(index, find_root);
            const auto& segment = segments[index];
            finest_length[root] = std::min(finest_length[root], segment.high[segment.axis] - segment.low[segment.axis]);
        }

        OocMarchingEdgeMesh output;
        output.cell_states = std::move(cell_states);
        std::vector<std::int64_t> component_vertex(segments.size(), -1);
        for (const auto state_index : leaf_order)
        {
            const auto& location = locations[state_index];
            for (std::uint32_t edge = 0U; edge != 12U; ++edge)
            {
                const auto segment_value = leaf_segment[state_index][edge];
                if (segment_value < 0)
                    continue;
                const auto segment = static_cast<std::size_t>(segment_value);
                const auto component = find_root(segment, find_root);
                const auto& key = segments[segment];
                const std::uint32_t length = key.high[key.axis] - key.low[key.axis];
                if (length != finest_length[component] || component_vertex[component] >= 0)
                {
                    continue;
                }
                if (location.parent < 0 || location.level == 0U)
                {
                    throw std::runtime_error("OOC marching edge vertex has no parent cell");
                }
                std::array<std::int32_t, 3> parent_cell{};
                for (std::size_t axis = 0U; axis != 3U; ++axis)
                {
                    parent_cell[axis] =
                        static_cast<std::int32_t>(locations[static_cast<std::size_t>(location.parent)].cell[axis]);
                }
                const auto& state = output.cell_states[state_index];
                const auto generated = make_ooc_marching_edge_vertex({
                    edge,
                    location.child_slot,
                    parent_cell,
                    location.level - 1U,
                    state.weighted_cell_scale,
                    state.source_node_index,
                    state.corner_scalar,
                    root_scale,
                    transform,
                });
                if (output.vertices.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
                {
                    throw std::runtime_error("OOC marching vertex count exceeds int32");
                }
                component_vertex[component] = static_cast<std::int64_t>(output.vertices.size());
                output.vertices.push_back(generated.vertex);
                output.vertex_scale.push_back(generated.vertex_scale);
                output.vertex_source.push_back(generated.source_node_index);
                output.vertex_root_grid_position.push_back(generated.root_grid_position);
            }
        }
        for (std::size_t index = 0U; index != component_vertex.size(); ++index)
        {
            if (find_root(index, find_root) == index && component_vertex[index] < 0)
            {
                throw std::runtime_error("OOC marching edge component has no owner at segment " +
                                         std::to_string(index) + " finest " + std::to_string(finest_length[index]));
            }
        }
        for (const auto state_index : leaf_order)
        {
            for (std::size_t edge = 0U; edge != 12U; ++edge)
            {
                const auto segment = leaf_segment[state_index][edge];
                if (segment < 0)
                    continue;
                const auto component = find_root(static_cast<std::size_t>(segment), find_root);
                output.cell_states[state_index].edge_vertex[edge] =
                    static_cast<std::int32_t>(component_vertex[component]);
            }
        }
        return output;
    }

    OocMarchingContourTrace trace_ooc_marching_local_contour(const OocMarchingCellState& cell, std::uint32_t start_edge)
    {
        if (start_edge >= 12U)
        {
            throw std::runtime_error("OOC contour start edge exceeds twelve-edge cube");
        }
        if (cell.edge_vertex[start_edge] < 0)
        {
            throw std::runtime_error("OOC contour starts on a missing edge vertex");
        }

        constexpr std::array<std::uint32_t, 12> positive_face{2U, 4U, 5U, 3U, 4U, 1U, 0U, 5U, 0U, 2U, 3U, 1U};
        constexpr std::array<std::uint32_t, 12> negative_face{4U, 3U, 2U, 5U, 0U, 4U, 5U, 1U, 2U, 1U, 0U, 3U};

        const auto edge_coordinates = [](std::uint32_t edge)
        {
            const std::uint32_t quadrant = edge & 3U;
            if (edge <= 3U)
            {
                return std::array<std::int32_t, 3>{1,
                                                   static_cast<std::int32_t>(2U * (quadrant & 1U)),
                                                   static_cast<std::int32_t>(2U * (quadrant >> 1U))};
            }
            if (edge <= 7U)
            {
                return std::array<std::int32_t, 3>{static_cast<std::int32_t>(2U * (quadrant & 1U)),
                                                   1,
                                                   static_cast<std::int32_t>(2U * (quadrant >> 1U))};
            }
            return std::array<std::int32_t, 3>{
                static_cast<std::int32_t>(2U * (quadrant & 1U)), static_cast<std::int32_t>(2U * (quadrant >> 1U)), 1};
        };
        const auto edge_from_coordinates = [](const std::array<std::int32_t, 3>& p)
        {
            if (p[0] == 1)
            {
                return static_cast<std::uint32_t>(p[1] / 2 + p[2]);
            }
            if (p[1] == 1)
            {
                return static_cast<std::uint32_t>(4 + p[0] / 2 + p[2]);
            }
            if (p[2] == 1)
            {
                return static_cast<std::uint32_t>(8 + p[0] / 2 + p[1]);
            }
            throw std::runtime_error("OOC contour perimeter point is not an edge");
        };

        OocMarchingContourTrace result;
        std::uint32_t edge = start_edge;
        do
        {
            if (result.edges.size() >= 12U)
            {
                throw std::runtime_error("OOC local contour did not close within twelve edges");
            }
            result.edges.push_back(edge);
            result.vertex_indices.push_back(static_cast<std::uint32_t>(cell.edge_vertex[edge]));

            const std::uint32_t axis = edge >> 2U;
            const std::uint32_t quadrant = edge & 3U;
            std::uint32_t low_corner = 0U;
            if (axis == 0U)
            {
                low_corner = 2U * (quadrant & 1U) + 4U * (quadrant >> 1U);
            }
            else if (axis == 1U)
            {
                low_corner = (quadrant & 1U) + 4U * (quadrant >> 1U);
            }
            else
            {
                low_corner = (quadrant & 1U) + 2U * (quadrant >> 1U);
            }
            const std::uint32_t face =
                cell.corner_scalar[low_corner] >= 0.0F ? positive_face[edge] : negative_face[edge];
            const std::uint32_t face_axis = face >> 1U;
            const std::uint32_t face_side = face & 1U;
            auto coordinates = edge_coordinates(edge);
            coordinates[face_axis] = static_cast<std::int32_t>(2U * face_side);

            std::array<std::uint32_t, 2> other_axes{};
            std::size_t other_count = 0U;
            for (std::uint32_t coordinate = 0U; coordinate != 3U; ++coordinate)
            {
                if (coordinate != face_axis)
                    other_axes[other_count++] = coordinate;
            }
            const auto first_axis = other_axes[0];
            const auto second_axis = other_axes[1];
            const std::int32_t first_value = coordinates[first_axis];
            std::int32_t perimeter = 3 - first_value;
            if (first_value == 1)
                perimeter = coordinates[second_axis];

            std::int32_t direction = 0;
            std::int32_t cursor = 0;
            if (face_axis == 2U)
            {
                cursor = perimeter + (face_side == 0U ? 5 : 3);
                direction = face_side == 0U ? 1 : -1;
            }
            else
            {
                cursor = perimeter + (face_side == 0U ? 3 : 5);
                direction = face_side == 0U ? -1 : 1;
            }

            bool found = false;
            const std::int32_t stop_delta = ~perimeter;
            while (true)
            {
                std::int32_t phase = cursor % 4;
                std::int32_t first = 1;
                std::int32_t second = phase;
                if (phase % 2 == 1)
                {
                    first = 2 * (phase / -2) + 2;
                    second = 1;
                }
                auto candidate_coordinates = coordinates;
                candidate_coordinates[first_axis] = first;
                candidate_coordinates[second_axis] = second;
                const auto candidate = edge_from_coordinates(candidate_coordinates);
                if (cell.edge_vertex[candidate] >= 0)
                {
                    edge = candidate;
                    found = true;
                    break;
                }
                cursor += direction;
                if (static_cast<std::uint32_t>(stop_delta + cursor) > 6U)
                    break;
            }
            if (!found)
            {
                throw std::runtime_error("OOC local contour face has no outgoing edge from " + std::to_string(edge) +
                                         " through face " + std::to_string(face));
            }
        } while (edge != start_edge);
        return result;
    }

    OocMarchingAdaptiveGridStep trace_ooc_marching_adaptive_grid_step(const OocMarchingCellState& cell,
                                                                      std::uint32_t edge,
                                                                      const std::array<std::int32_t, 25>& grid)
    {
        if (edge >= 12U)
        {
            throw std::runtime_error("OOC adaptive grid edge exceeds twelve-edge cube");
        }
        constexpr std::array<std::uint32_t, 12> positive_face{2U, 4U, 5U, 3U, 4U, 1U, 0U, 5U, 0U, 2U, 3U, 1U};
        constexpr std::array<std::uint32_t, 12> negative_face{4U, 3U, 2U, 5U, 0U, 4U, 5U, 1U, 2U, 1U, 0U, 3U};
        const auto edge_coordinates = [](std::uint32_t value)
        {
            const auto q = value & 3U;
            if (value < 4U)
                return std::array<int, 3>{1, int(2U * (q & 1U)), int(2U * (q >> 1U))};
            if (value < 8U)
                return std::array<int, 3>{int(2U * (q & 1U)), 1, int(2U * (q >> 1U))};
            return std::array<int, 3>{int(2U * (q & 1U)), int(2U * (q >> 1U)), 1};
        };
        const auto edge_from_coordinates = [](const std::array<int, 3>& p)
        {
            if (p[0] == 1)
                return std::uint32_t(p[1] / 2 + p[2]);
            if (p[1] == 1)
                return std::uint32_t(4 + p[0] / 2 + p[2]);
            if (p[2] == 1)
                return std::uint32_t(8 + p[0] / 2 + p[1]);
            throw std::runtime_error("OOC adaptive exit coordinate is not an edge");
        };

        const auto axis = edge >> 2U;
        const auto q = edge & 3U;
        std::uint32_t low_corner = 0U;
        if (axis == 0U)
            low_corner = 2U * (q & 1U) + 4U * (q >> 1U);
        else if (axis == 1U)
            low_corner = (q & 1U) + 4U * (q >> 1U);
        else
            low_corner = (q & 1U) + 2U * (q >> 1U);
        const std::uint32_t face = cell.corner_scalar[low_corner] >= 0.0F ? positive_face[edge] : negative_face[edge];
        const std::uint32_t face_axis = face >> 1U;
        const std::uint32_t face_side = face & 1U;
        auto coordinates = edge_coordinates(edge);
        coordinates[face_axis] = int(2U * face_side);
        std::array<std::uint32_t, 2> other{};
        std::size_t other_count = 0U;
        for (std::uint32_t a = 0U; a != 3U; ++a)
            if (a != face_axis)
                other[other_count++] = a;

        const int first_coordinate = coordinates[other[0]];
        const int center = 2 * (first_coordinate + 5 * coordinates[other[1]]);
        int grid_index = first_coordinate == 1 ? center - 1 : center - 5;
        if (grid[static_cast<std::size_t>(grid_index)] == -1)
            grid_index = first_coordinate == 1 ? center + 1 : center + 5;
        if (grid_index < 0 || grid_index >= 25 || grid[static_cast<std::size_t>(grid_index)] == -1)
        {
            throw std::runtime_error("OOC adaptive grid has no entering vertex");
        }

        int tile_x = (grid_index % 5) / 3;
        int tile_y = grid_index / 15;
        int local_x = grid_index % 5 - 2 * tile_x;
        int local_y = grid_index / 5 - 2 * tile_y;
        OocMarchingAdaptiveGridStep result;
        result.vertex_indices.push_back(static_cast<std::uint32_t>(grid[static_cast<std::size_t>(grid_index)]));

        while (true)
        {
            int perimeter = local_x == 1 ? local_y : 3 - local_x;
            int cursor = 0;
            int direction = 0;
            if (face_axis == 2U)
            {
                cursor = perimeter + (face_side == 0U ? 5 : 3);
                direction = face_side == 0U ? 1 : -1;
            }
            else
            {
                cursor = perimeter + (face_side == 0U ? 3 : 5);
                direction = face_side == 0U ? -1 : 1;
            }
            const int stop_delta = ~perimeter;
            std::array<int, 2> candidate{};
            std::int32_t vertex = -1;
            while (true)
            {
                int second = cursor % 4;
                int first = 1;
                if (second % 2 == 1)
                {
                    first = 2 * (second / -2) + 2;
                    second = 1;
                }
                candidate = {first, second};
                vertex = grid[static_cast<std::size_t>(2 * tile_x + first + 5 * (2 * tile_y + second))];
                if (vertex != -1)
                    break;
                cursor += direction;
                if (static_cast<std::uint32_t>(stop_delta + cursor) > 6U)
                    break;
            }
            const bool exits_x = static_cast<std::uint32_t>(tile_x + candidate[0] - 1) > 1U;
            const bool exits_y = static_cast<std::uint32_t>(tile_y + candidate[1] - 1) > 1U;
            if (exits_x || exits_y)
            {
                coordinates[other[1]] = candidate[0] != 1 ? 1 : 2 * tile_y;
                coordinates[other[0]] = candidate[1] != 1 ? 1 : 2 * tile_x;
                result.next_edge = edge_from_coordinates(coordinates);
                return result;
            }
            result.vertex_indices.push_back(static_cast<std::uint32_t>(vertex));
            tile_x += candidate[0] - 1;
            tile_y += candidate[1] - 1;
            local_x = 2 - candidate[0];
            local_y = 2 - candidate[1];
        }
    }

    OocMarchingRawOutput build_ooc_marching_raw_mesh(const OocMarchingEdgeMesh& edges,
                                                     std::uint64_t root_state_index,
                                                     std::uint32_t maximum_level,
                                                     bool penalize_axis_aligned)
    {
        const auto& states = edges.cell_states;
        if (states.empty() || root_state_index >= states.size())
        {
            throw std::runtime_error("OOC marching face tree has no valid root");
        }
        if (maximum_level == 0U || maximum_level >= 31U)
        {
            throw std::runtime_error("OOC marching face level exceeds int32 grid domain");
        }

        struct Location
        {
            std::uint32_t level{};
            std::array<std::uint32_t, 3> cell{};
            bool seen{};
        };
        struct LocationKey
        {
            std::uint32_t level{};
            std::array<std::uint32_t, 3> cell{};
            bool operator==(const LocationKey&) const = default;
        };
        struct LocationKeyHash
        {
            std::size_t operator()(const LocationKey& key) const noexcept
            {
                std::size_t value = key.level;
                for (const auto coordinate : key.cell)
                {
                    value ^=
                        static_cast<std::size_t>(coordinate) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
                }
                return value;
            }
        };
        std::vector<Location> locations(states.size());
        std::vector<std::size_t> leaf_order;
        std::unordered_map<LocationKey, std::size_t, LocationKeyHash> leaf_at;
        std::function<void(std::size_t, std::uint32_t, const std::array<std::uint32_t, 3>&)> locate =
            [&](std::size_t state_index, std::uint32_t level, const std::array<std::uint32_t, 3>& cell)
        {
            if (state_index >= states.size() || locations[state_index].seen)
            {
                throw std::runtime_error("OOC marching face tree has invalid child ownership");
            }
            if (level > maximum_level)
            {
                throw std::runtime_error("OOC marching face tree exceeds maximum level");
            }
            locations[state_index] = Location{level, cell, true};
            bool has_child = false;
            for (std::size_t slot = 0U; slot != 8U; ++slot)
            {
                const auto child = states[state_index].child_state[slot];
                if (child < 0)
                    continue;
                has_child = true;
                std::array<std::uint32_t, 3> child_cell{};
                for (std::size_t axis = 0U; axis != 3U; ++axis)
                {
                    child_cell[axis] = 2U * cell[axis] + ((slot >> axis) & 1U);
                }
                locate(static_cast<std::size_t>(child), level + 1U, child_cell);
            }
            if (!has_child)
            {
                leaf_order.push_back(state_index);
                if (!leaf_at.emplace(LocationKey{level, cell}, state_index).second)
                {
                    throw std::runtime_error("OOC marching face tree duplicates a cell");
                }
            }
        };
        locate(static_cast<std::size_t>(root_state_index), 0U, {0U, 0U, 0U});
        if (std::find_if(locations.begin(), locations.end(), [](const Location& value) { return !value.seen; }) !=
            locations.end())
        {
            throw std::runtime_error("OOC marching face tree has unreachable states");
        }

        constexpr std::array<std::uint32_t, 12> positive_face{2U, 4U, 5U, 3U, 4U, 1U, 0U, 5U, 0U, 2U, 3U, 1U};
        constexpr std::array<std::uint32_t, 12> negative_face{4U, 3U, 2U, 5U, 0U, 4U, 5U, 1U, 2U, 1U, 0U, 3U};
        const auto edge_coordinates = [](std::uint32_t edge)
        {
            const auto quadrant = edge & 3U;
            if (edge < 4U)
            {
                return std::array<int, 3>{
                    1, static_cast<int>(2U * (quadrant & 1U)), static_cast<int>(2U * (quadrant >> 1U))};
            }
            if (edge < 8U)
            {
                return std::array<int, 3>{
                    static_cast<int>(2U * (quadrant & 1U)), 1, static_cast<int>(2U * (quadrant >> 1U))};
            }
            return std::array<int, 3>{
                static_cast<int>(2U * (quadrant & 1U)), static_cast<int>(2U * (quadrant >> 1U)), 1};
        };
        const auto edge_from_coordinates = [](const std::array<int, 3>& point)
        {
            if (point[0] == 1)
                return static_cast<std::uint32_t>(point[1] / 2 + point[2]);
            if (point[1] == 1)
                return static_cast<std::uint32_t>(4 + point[0] / 2 + point[2]);
            if (point[2] == 1)
                return static_cast<std::uint32_t>(8 + point[0] / 2 + point[1]);
            throw std::runtime_error("OOC face perimeter point is not a cube edge");
        };
        const auto edge_endpoints = [](std::uint32_t edge)
        {
            const auto axis = edge >> 2U;
            const auto quadrant = edge & 3U;
            std::array<std::uint32_t, 3> low{};
            if (axis == 0U)
                low = {0U, quadrant & 1U, quadrant >> 1U};
            else if (axis == 1U)
                low = {quadrant & 1U, 0U, quadrant >> 1U};
            else
                low = {quadrant & 1U, quadrant >> 1U, 0U};
            auto high = low;
            high[axis] = 1U;
            return std::array<std::array<std::uint32_t, 3>, 2>{low, high};
        };
        const auto low_corner = [&](std::uint32_t edge)
        {
            const auto points = edge_endpoints(edge);
            return points[0][0] | (points[0][1] << 1U) | (points[0][2] << 2U);
        };

        const auto local_next = [&](const OocMarchingCellState& state, std::uint32_t edge, std::uint32_t face)
        {
            const auto face_axis = face >> 1U;
            const auto face_side = face & 1U;
            auto coordinates = edge_coordinates(edge);
            coordinates[face_axis] = static_cast<int>(2U * face_side);
            std::array<std::uint32_t, 2> other{};
            std::size_t count = 0U;
            for (std::uint32_t axis = 0U; axis != 3U; ++axis)
                if (axis != face_axis)
                    other[count++] = axis;
            const int first = coordinates[other[0]];
            const int perimeter = first == 1 ? coordinates[other[1]] : 3 - first;
            int cursor = 0;
            int direction = 0;
            if (face_axis == 2U)
            {
                cursor = perimeter + (face_side == 0U ? 5 : 3);
                direction = face_side == 0U ? 1 : -1;
            }
            else
            {
                cursor = perimeter + (face_side == 0U ? 3 : 5);
                direction = face_side == 0U ? -1 : 1;
            }
            const int stop_delta = ~perimeter;
            while (true)
            {
                int second = cursor % 4;
                int first_coordinate = 1;
                if (second % 2 == 1)
                {
                    first_coordinate = 2 * (second / -2) + 2;
                    second = 1;
                }
                auto candidate = coordinates;
                candidate[other[0]] = first_coordinate;
                candidate[other[1]] = second;
                const auto next = edge_from_coordinates(candidate);
                if (state.edge_vertex[next] >= 0)
                    return next;
                cursor += direction;
                if (static_cast<std::uint32_t>(stop_delta + cursor) > 6U)
                {
                    throw std::runtime_error("OOC local face has no outgoing sign-changing edge");
                }
            }
        };

        const auto finer_face_grid = [&](std::size_t state_index,
                                         std::uint32_t face) -> std::optional<std::array<std::int32_t, 25>>
        {
            const auto& location = locations[state_index];
            if (location.level >= maximum_level)
                return std::nullopt;
            const std::uint32_t face_axis = face >> 1U;
            const std::uint32_t face_side = face & 1U;
            std::array<std::uint32_t, 2> other{};
            std::size_t other_count = 0U;
            for (std::uint32_t axis = 0U; axis != 3U; ++axis)
                if (axis != face_axis)
                    other[other_count++] = axis;

            std::array<std::size_t, 4> fine_cells{};
            fine_cells.fill(std::numeric_limits<std::size_t>::max());
            bool has_fine_cell = false;
            for (std::size_t tile = 0U; tile != 4U; ++tile)
            {
                std::array<std::uint32_t, 3> fine{};
                fine[face_axis] =
                    face_side == 0U ? 2U * location.cell[face_axis] - 1U : 2U * (location.cell[face_axis] + 1U);
                fine[other[0]] = 2U * location.cell[other[0]] + (tile & 1U);
                fine[other[1]] = 2U * location.cell[other[1]] + static_cast<std::uint32_t>(tile >> 1U);
                const auto found = leaf_at.find(LocationKey{location.level + 1U, fine});
                if (found != leaf_at.end())
                {
                    fine_cells[tile] = found->second;
                    has_fine_cell = true;
                }
            }
            if (!has_fine_cell)
                return std::nullopt;

            std::array<std::int32_t, 25> grid{};
            grid.fill(-1);
            const std::uint32_t current_scale = 1U << (maximum_level - location.level);
            std::array<std::uint32_t, 3> current_low{};
            for (std::size_t axis = 0U; axis != 3U; ++axis)
                current_low[axis] = location.cell[axis] * current_scale;
            const std::uint32_t plane = current_low[face_axis] + face_side * current_scale;
            for (const auto fine_index : fine_cells)
            {
                if (fine_index == std::numeric_limits<std::size_t>::max())
                    continue;
                const auto& fine_location = locations[fine_index];
                const std::uint32_t fine_scale = 1U << (maximum_level - fine_location.level);
                for (std::uint32_t edge = 0U; edge != 12U; ++edge)
                {
                    const auto vertex = states[fine_index].edge_vertex[edge];
                    if (vertex < 0)
                        continue;
                    const auto points = edge_endpoints(edge);
                    std::array<std::uint32_t, 3> low{};
                    std::array<std::uint32_t, 3> high{};
                    for (std::size_t axis = 0U; axis != 3U; ++axis)
                    {
                        low[axis] = (fine_location.cell[axis] + points[0][axis]) * fine_scale;
                        high[axis] = (fine_location.cell[axis] + points[1][axis]) * fine_scale;
                    }
                    if (low[face_axis] != plane || high[face_axis] != plane)
                        continue;
                    std::array<int, 2> grid_coordinate{};
                    for (std::size_t item = 0U; item != 2U; ++item)
                    {
                        const auto axis = other[item];
                        const std::uint32_t twice_midpoint = low[axis] + high[axis];
                        const std::uint32_t twice_origin = 2U * current_low[axis];
                        const std::uint32_t numerator = 2U * (twice_midpoint - twice_origin);
                        if (numerator % current_scale != 0U)
                        {
                            throw std::runtime_error("OOC adaptive face vertex is off the 5x5 grid");
                        }
                        grid_coordinate[item] = static_cast<int>(numerator / current_scale);
                    }
                    const int position = grid_coordinate[0] + 5 * grid_coordinate[1];
                    if (position < 0 || position >= 25)
                    {
                        throw std::runtime_error("OOC adaptive face vertex exceeds the 5x5 grid");
                    }
                    auto& destination = grid[static_cast<std::size_t>(position)];
                    if (destination != -1 && destination != vertex)
                    {
                        throw std::runtime_error("OOC adaptive face grid has conflicting vertices");
                    }
                    destination = vertex;
                }
            }
            return grid;
        };

        OocMarchingRawOutput output;
        output.vertices = edges.vertices;
        output.vertex_scale = edges.vertex_scale;
        output.vertex_source = edges.vertex_source;
        for (const auto state_index : leaf_order)
        {
            const auto& state = states[state_index];
            std::array<std::uint8_t, 12> visited{};
            for (std::uint32_t start_edge = 0U; start_edge != 12U; ++start_edge)
            {
                if (state.edge_vertex[start_edge] < 0 || visited[start_edge] != 0U)
                    continue;
                std::vector<std::uint32_t> polygon;
                std::uint32_t edge = start_edge;
                do
                {
                    if (edge >= 12U || visited[edge] != 0U)
                    {
                        throw std::runtime_error("OOC adaptive contour failed to close cleanly");
                    }
                    visited[edge] = 1U;
                    const auto corner = low_corner(edge);
                    const std::uint32_t face =
                        state.corner_scalar[corner] >= 0.0F ? positive_face[edge] : negative_face[edge];
                    const auto grid = finer_face_grid(state_index, face);
                    if (grid)
                    {
                        const auto step = trace_ooc_marching_adaptive_grid_step(state, edge, *grid);
                        polygon.insert(polygon.end(), step.vertex_indices.begin(), step.vertex_indices.end());
                        edge = step.next_edge;
                    }
                    else
                    {
                        polygon.push_back(static_cast<std::uint32_t>(state.edge_vertex[edge]));
                        edge = local_next(state, edge, face);
                    }
                    if (polygon.size() > 256U)
                    {
                        throw std::runtime_error("OOC adaptive contour exceeds target stack capacity");
                    }
                } while (edge != start_edge);
                std::vector<OocMarchingVertex> polygon_vertices;
                polygon_vertices.reserve(polygon.size());
                for (const auto vertex : polygon)
                {
                    if (vertex >= output.vertices.size())
                    {
                        throw std::runtime_error("OOC contour references an invalid raw vertex");
                    }
                    polygon_vertices.push_back(output.vertices[vertex]);
                }
                const auto triangulation =
                    triangulate_ooc_marching_polygon(polygon_vertices, polygon, penalize_axis_aligned);
                output.triangles.insert(
                    output.triangles.end(), triangulation.triangles.begin(), triangulation.triangles.end());
                output.face_source.insert(
                    output.face_source.end(), triangulation.triangles.size(), state.source_node_index);
            }
        }
        validate_ooc_marching_raw_output(output);
        return output;
    }

    OocMarchingPolygonDpResult triangulate_ooc_marching_polygon(std::span<const OocMarchingVertex> polygon_vertices,
                                                                std::span<const std::uint32_t> vertex_indices,
                                                                bool penalize_axis_aligned)
    {
        const std::size_t count = polygon_vertices.size();
        if (count != vertex_indices.size())
        {
            throw std::runtime_error("OOC marching polygon vertex/index size mismatch");
        }
        if (count < 3U)
        {
            throw std::runtime_error("OOC marching polygon has fewer than three vertices");
        }
        if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        {
            throw std::runtime_error("OOC marching polygon exceeds int32 DP range");
        }

        const auto triangle_area =
            [](const OocMarchingVertex& a, const OocMarchingVertex& b, const OocMarchingVertex& c)
        {
            // Exact scalar instruction order of sub_2BA5630.
            double c_z = static_cast<double>(c.position[2]);
            double c_y = static_cast<double>(c.position[1]);
            const double a_y = static_cast<double>(a.position[1]);
            const double a_z = static_cast<double>(a.position[2]);
            const double a_x = static_cast<double>(a.position[0]);
            double c_x = static_cast<double>(c.position[0]);
            double b_y = static_cast<double>(b.position[1]);
            double b_z = static_cast<double>(b.position[2]);
            double b_x = static_cast<double>(b.position[0]);

            c_y = c_y - a_y;
            c_z = c_z - a_z;
            c_x = c_x - a_x;
            b_y = b_y - a_y;
            b_z = b_z - a_z;
            b_x = b_x - a_x;
            double component_0 = b_y * c_x;
            b_y = b_y * c_z;
            double component_1 = b_x * c_y;
            c_x = c_x * b_z;
            b_x = b_x * c_z;
            b_z = b_z * c_y;
            c_x = c_x - b_x;
            b_y = b_y - b_z;
            component_1 = component_1 - component_0;
            c_x = c_x * c_x;
            b_y = b_y * b_y;
            component_1 = component_1 * component_1;
            b_y = b_y + c_x;
            component_1 = component_1 + b_y;
            const double length = std::sqrt(component_1);
            return 0.5 * length;
        };

        OocMarchingPolygonDpResult result;
        result.cost.assign(count * count, -1.0);
        result.split.assign(count * count, -1);
        const auto at = [count](std::size_t row, std::size_t column) { return row * count + column; };
        for (std::size_t index = 0; index + 1U < count; ++index)
        {
            result.cost[at(index, index + 1U)] = 0.0;
        }

        double maximum_initial_area = 0.0;
        for (std::size_t index = 0; index + 2U < count; ++index)
        {
            const double area =
                triangle_area(polygon_vertices[index], polygon_vertices[index + 1U], polygon_vertices[index + 2U]);
            result.cost[at(index, index + 2U)] = area;
            // MAXSD(area, old) chooses old for equality or unordered operands.
            maximum_initial_area = area > maximum_initial_area ? area : maximum_initial_area;
        }

        if (penalize_axis_aligned)
        {
            for (std::size_t index = 0; index + 2U < count; ++index)
            {
                bool shares_coordinate = false;
                for (std::size_t coordinate = 0; coordinate < 3U; ++coordinate)
                {
                    const float value = polygon_vertices[index].position[coordinate];
                    if (value == polygon_vertices[index + 1U].position[coordinate] &&
                        value == polygon_vertices[index + 2U].position[coordinate])
                    {
                        shares_coordinate = true;
                    }
                }
                if (shares_coordinate)
                {
                    double penalty = maximum_initial_area;
                    penalty = penalty + maximum_initial_area;
                    result.cost[at(index, index + 2U)] = penalty + result.cost[at(index, index + 2U)];
                }
            }
        }

        for (std::size_t length = 3U; length < count; ++length)
        {
            for (std::size_t begin = 0; begin + length < count; ++begin)
            {
                const std::size_t end = begin + length;
                double best = -1.0;
                std::int32_t best_split = -1;
                for (std::size_t middle = begin + 1U; middle < end; ++middle)
                {
                    double subcost = result.cost[at(begin, middle)];
                    subcost = subcost + result.cost[at(middle, end)];
                    double candidate =
                        triangle_area(polygon_vertices[begin], polygon_vertices[middle], polygon_vertices[end]);
                    candidate = candidate + subcost;
                    if (best_split == -1 || best > candidate)
                    {
                        best = candidate;
                        best_split = static_cast<std::int32_t>(middle);
                    }
                    else
                    {
                        // MINSD(candidate, best) returns best on equality/unordered.
                        best = candidate < best ? candidate : best;
                    }
                }
                result.cost[at(begin, end)] = best;
                result.split[at(begin, end)] = best_split;
            }
        }

        const auto backtrack = [&](auto&& self, std::size_t begin, std::size_t end) -> void
        {
            if (begin + 2U == end)
            {
                result.triangles.push_back(
                    {{{vertex_indices[begin], vertex_indices[begin + 1U], vertex_indices[end]}}});
                return;
            }
            const auto split_value = result.split[at(begin, end)];
            if (split_value <= static_cast<std::int32_t>(begin) || split_value >= static_cast<std::int32_t>(end))
            {
                throw std::runtime_error("OOC marching DP contains invalid split");
            }
            const auto middle = static_cast<std::size_t>(split_value);
            if (middle != begin + 1U)
                self(self, begin, middle);
            result.triangles.push_back({{{vertex_indices[begin], vertex_indices[middle], vertex_indices[end]}}});
            if (middle != end - 1U)
                self(self, middle, end);
        };
        backtrack(backtrack, 0U, count - 1U);
        return result;
    }

} // namespace metmodel
