#include "metmodel/octree_prepare.hpp"
#include "recovered_cuda_source.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace metmodel
{
    namespace
    {

        struct DeviceCell
        {
            std::uint32_t x;
            std::uint32_t y;
            std::uint32_t z;
            std::uint32_t level;
        };

        __device__ __forceinline__ std::uint64_t compact_morton_21(std::uint64_t value)
        {
            value &= 0x1249249249249249ULL;
            value = (value ^ (value >> 2U)) & 0x10c30c30c30c30c3ULL;
            value = (value ^ (value >> 4U)) & 0x100f00f00f00f00fULL;
            value = (value ^ (value >> 8U)) & 0x1f0000ff0000ffULL;
            value = (value ^ (value >> 16U)) & 0x1f00000000ffffULL;
            value = (value ^ (value >> 32U)) & 0x1fffffULL;
            return value;
        }

        __device__ __forceinline__ std::uint64_t spread_morton_21(std::uint32_t value)
        {
            std::uint64_t result = static_cast<std::uint64_t>(value & 0x1fffffU);
            result = (result | (result << 32U)) & 0x1f00000000ffffULL;
            result = (result | (result << 16U)) & 0x1f0000ff0000ffULL;
            result = (result | (result << 8U)) & 0x100f00f00f00f00fULL;
            result = (result | (result << 4U)) & 0x10c30c30c30c30c3ULL;
            result = (result | (result << 2U)) & 0x1249249249249249ULL;
            return result;
        }

        __device__ __forceinline__ DeviceCell decode_cell(const std::uint32_t* words,
                                                          const std::uint8_t* levels,
                                                          std::uint32_t count,
                                                          std::uint32_t index)
        {
            const std::uint32_t level = levels[index];
            DeviceCell cell{0U, 0U, 0U, level};
            if (level <= 21U)
            {
                const std::uint32_t bits = 3U * level;
                std::uint64_t packed = 0U;
                if (bits != 0U && bits <= 32U)
                {
                    packed = static_cast<std::uint64_t>(words[index]) >> (32U - bits);
                }
                else if (bits != 0U)
                {
                    packed = (static_cast<std::uint64_t>(words[index]) << 32U) | words[count + index];
                    packed >>= 64U - bits;
                }
                cell.x = static_cast<std::uint32_t>(compact_morton_21(packed >> 2U));
                cell.y = static_cast<std::uint32_t>(compact_morton_21(packed >> 1U));
                cell.z = static_cast<std::uint32_t>(compact_morton_21(packed));
                return cell;
            }
            for (std::uint32_t depth = 0U; depth < level; ++depth)
            {
                const std::uint32_t top = 95U - 3U * depth;
                std::uint32_t code = 0U;
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                {
                    const std::uint32_t position = top - axis;
                    const std::uint32_t word_index = position >= 64U ? 0U : (position >= 32U ? 1U : 2U);
                    const std::uint32_t bit =
                        position >= 64U ? position - 64U : (position >= 32U ? position - 32U : position);
                    code |= ((words[word_index * count + index] >> bit) & 1U) << (2U - axis);
                }
                cell.x = (cell.x << 1U) | ((code >> 2U) & 1U);
                cell.y = (cell.y << 1U) | ((code >> 1U) & 1U);
                cell.z = (cell.z << 1U) | (code & 1U);
            }
            return cell;
        }

        __device__ __forceinline__ void encode_cell(const DeviceCell& cell, std::uint32_t (&words)[3])
        {
            words[0] = 0U;
            words[1] = 0U;
            words[2] = 0U;
            if (cell.level <= 21U)
            {
                const std::uint64_t packed =
                    (spread_morton_21(cell.x) << 2U) | (spread_morton_21(cell.y) << 1U) | spread_morton_21(cell.z);
                const std::uint32_t bits = 3U * cell.level;
                if (bits != 0U && bits <= 32U)
                {
                    words[0] = static_cast<std::uint32_t>(packed << (32U - bits));
                }
                else if (bits != 0U)
                {
                    const std::uint64_t aligned = packed << (64U - bits);
                    words[0] = static_cast<std::uint32_t>(aligned >> 32U);
                    words[1] = static_cast<std::uint32_t>(aligned);
                }
                return;
            }
            for (std::uint32_t depth = 0U; depth < cell.level; ++depth)
            {
                const std::uint32_t coordinate_bit = cell.level - depth - 1U;
                const std::uint32_t top = 95U - 3U * depth;
                const std::uint32_t coordinates[3] = {cell.x, cell.y, cell.z};
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                {
                    if (((coordinates[axis] >> coordinate_bit) & 1U) == 0U)
                        continue;
                    const std::uint32_t position = top - axis;
                    const std::uint32_t word = position >= 64U ? 0U : (position >= 32U ? 1U : 2U);
                    const std::uint32_t bit =
                        position >= 64U ? position - 64U : (position >= 32U ? position - 32U : position);
                    words[word] |= 1U << bit;
                }
            }
        }

        __device__ __forceinline__ int compare_key(const std::uint32_t* morton_words,
                                                   const std::uint8_t* levels,
                                                   std::uint32_t count,
                                                   std::uint32_t index,
                                                   std::uint32_t level,
                                                   const std::uint32_t (&words)[3])
        {
            const std::uint32_t candidate_level = levels[index];
            if (candidate_level < level)
                return -1;
            if (candidate_level > level)
                return 1;
            for (std::uint32_t word = 0U; word < 3U; ++word)
            {
                const std::uint32_t value = morton_words[word * count + index];
                if (value < words[word])
                    return -1;
                if (value > words[word])
                    return 1;
            }
            return 0;
        }

        __device__ __forceinline__ std::uint32_t binary_search_cell(const std::uint32_t* morton_words,
                                                                    const std::uint8_t* levels,
                                                                    std::uint32_t count,
                                                                    const DeviceCell& cell)
        {
            std::uint32_t words[3];
            encode_cell(cell, words);
            std::uint32_t begin = 0U;
            std::uint32_t end = count;
            while (begin < end)
            {
                const std::uint32_t middle = begin + (end - begin) / 2U;
                if (compare_key(morton_words, levels, count, middle, cell.level, words) < 0)
                {
                    begin = middle + 1U;
                }
                else
                {
                    end = middle;
                }
            }
            return begin < count && compare_key(morton_words, levels, count, begin, cell.level, words) == 0
                       ? begin
                       : 0xffffffffU;
        }

        __global__ __launch_bounds__(128, 1) void neighbors_binary_search_kernel(const std::uint32_t* morton_words,
                                                                                 const std::uint8_t* levels,
                                                                                 std::uint32_t* neighbors,
                                                                                 std::uint8_t* connectivity,
                                                                                 std::uint8_t* refinement,
                                                                                 std::uint32_t count,
                                                                                 std::uint32_t offset)
        {
            const std::uint32_t index = offset + blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count)
                return;
            const DeviceCell cell = decode_cell(morton_words, levels, count, index);
            std::uint8_t connection_bits = 0U;
            std::uint8_t refinement_bits = 0U;
            for (std::uint32_t direction = 0U; direction < 6U; ++direction)
            {
                const std::uint32_t axis = direction / 2U;
                const bool positive = (direction & 1U) != 0U;
                const std::uint32_t coordinate = axis == 0U ? cell.x : (axis == 1U ? cell.y : cell.z);
                std::uint32_t found = 0xffffffffU;
                bool found_refined = false;

                if (cell.level < 32U)
                {
                    DeviceCell fine{cell.x << 1U, cell.y << 1U, cell.z << 1U, cell.level + 1U};
                    std::uint32_t* value = axis == 0U ? &fine.x : (axis == 1U ? &fine.y : &fine.z);
                    if (positive)
                    {
                        *value = 2U * coordinate + 2U;
                        if (static_cast<std::uint64_t>(*value) < (1ULL << fine.level))
                            found = binary_search_cell(morton_words, levels, count, fine);
                    }
                    else if (coordinate != 0U)
                    {
                        *value = 2U * coordinate - 1U;
                        found = binary_search_cell(morton_words, levels, count, fine);
                    }
                    found_refined = found != 0xffffffffU;
                }

                if (found == 0xffffffffU)
                {
                    DeviceCell same = cell;
                    std::uint32_t* value = axis == 0U ? &same.x : (axis == 1U ? &same.y : &same.z);
                    if (positive)
                    {
                        if (static_cast<std::uint64_t>(*value) + 1ULL < (1ULL << cell.level))
                        {
                            ++*value;
                            found = binary_search_cell(morton_words, levels, count, same);
                        }
                    }
                    else if (*value != 0U)
                    {
                        --*value;
                        found = binary_search_cell(morton_words, levels, count, same);
                    }
                }

                const bool crosses_parent = positive ? (coordinate & 1U) != 0U : (coordinate & 1U) == 0U;
                const bool inside =
                    positive ? static_cast<std::uint64_t>(coordinate) + 1ULL < (1ULL << cell.level) : coordinate != 0U;
                if (found == 0xffffffffU && cell.level != 0U && crosses_parent && inside)
                {
                    DeviceCell coarse{cell.x >> 1U, cell.y >> 1U, cell.z >> 1U, cell.level - 1U};
                    std::uint32_t* value = axis == 0U ? &coarse.x : (axis == 1U ? &coarse.y : &coarse.z);
                    *value = positive ? (coordinate + 1U) / 2U : (coordinate - 1U) / 2U;
                    found = binary_search_cell(morton_words, levels, count, coarse);
                }

                neighbors[6U * index + direction] = found;
                if (found != 0xffffffffU)
                {
                    connection_bits |= static_cast<std::uint8_t>(1U << direction);
                    if (found_refined)
                        refinement_bits |= static_cast<std::uint8_t>(1U << direction);
                }
            }
            connectivity[index] = connection_bits;
            refinement[index] = refinement_bits;
        }

    } // namespace

    cudaError_t launch_recovered_ooc_neighbors_binary_search_source(const std::uint32_t* morton_words_soa,
                                                                    const std::uint8_t* levels,
                                                                    std::uint32_t* neighbors,
                                                                    std::uint8_t* connectivity,
                                                                    std::uint8_t* refinement,
                                                                    std::uint32_t count,
                                                                    std::uint32_t offset,
                                                                    std::size_t work_items,
                                                                    cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        neighbors_binary_search_kernel<<<blocks, threads, 0U, stream>>>(
            morton_words_soa, levels, neighbors, connectivity, refinement, count, offset);
        return cudaGetLastError();
    }

    bool run_recovered_ooc_neighbors_cuda_source(const std::vector<std::uint32_t>& morton_words_soa,
                                                 const std::vector<std::uint8_t>& levels,
                                                 std::vector<std::uint32_t>& neighbors,
                                                 std::vector<std::uint8_t>& connectivity,
                                                 std::vector<std::uint8_t>& refinement,
                                                 std::size_t device_index,
                                                 std::string& error)
    {
        const std::size_t count = levels.size();
        if (count == 0U || count > std::numeric_limits<std::uint32_t>::max() || morton_words_soa.size() != 3U * count ||
            neighbors.size() != 6U * count || connectivity.size() != count || refinement.size() != count ||
            device_index > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            error = "invalid OOC CUDA neighbor-search arguments";
            return false;
        }

        std::uint32_t* device_morton = nullptr;
        std::uint8_t* device_levels = nullptr;
        std::uint32_t* device_neighbors = nullptr;
        std::uint8_t* device_connectivity = nullptr;
        std::uint8_t* device_refinement = nullptr;
        auto release = [&]()
        {
            cudaFree(device_morton);
            cudaFree(device_levels);
            cudaFree(device_neighbors);
            cudaFree(device_connectivity);
            cudaFree(device_refinement);
        };

        const std::size_t morton_bytes = morton_words_soa.size() * sizeof(std::uint32_t);
        const std::size_t level_bytes = levels.size();
        const std::size_t neighbor_bytes = neighbors.size() * sizeof(std::uint32_t);
        cudaError_t status = cudaSetDevice(static_cast<int>(device_index));
        if (status == cudaSuccess)
            status = cudaMalloc(&device_morton, morton_bytes);
        if (status == cudaSuccess)
            status = cudaMalloc(&device_levels, level_bytes);
        if (status == cudaSuccess)
            status = cudaMalloc(&device_neighbors, neighbor_bytes);
        if (status == cudaSuccess)
            status = cudaMalloc(&device_connectivity, level_bytes);
        if (status == cudaSuccess)
            status = cudaMalloc(&device_refinement, level_bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(device_morton, morton_words_soa.data(), morton_bytes, cudaMemcpyHostToDevice);
        if (status == cudaSuccess)
            status = cudaMemcpy(device_levels, levels.data(), level_bytes, cudaMemcpyHostToDevice);

        constexpr std::size_t batch_capacity = 1'000'000U;
        for (std::size_t offset = 0U; status == cudaSuccess && offset < count; offset += batch_capacity)
        {
            const std::size_t work_items = std::min(batch_capacity, count - offset);
            status = launch_recovered_ooc_neighbors_binary_search_source(device_morton,
                                                                         device_levels,
                                                                         device_neighbors,
                                                                         device_connectivity,
                                                                         device_refinement,
                                                                         static_cast<std::uint32_t>(count),
                                                                         static_cast<std::uint32_t>(offset),
                                                                         work_items);
        }
        if (status == cudaSuccess)
            status = cudaDeviceSynchronize();
        if (status == cudaSuccess)
            status = cudaMemcpy(neighbors.data(), device_neighbors, neighbor_bytes, cudaMemcpyDeviceToHost);
        if (status == cudaSuccess)
            status = cudaMemcpy(connectivity.data(), device_connectivity, level_bytes, cudaMemcpyDeviceToHost);
        if (status == cudaSuccess)
            status = cudaMemcpy(refinement.data(), device_refinement, level_bytes, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
        {
            error = std::string("OOC CUDA neighbor search failed: ") + cudaGetErrorString(status);
            release();
            return false;
        }
        release();
        error.clear();
        return true;
    }

} // namespace metmodel
