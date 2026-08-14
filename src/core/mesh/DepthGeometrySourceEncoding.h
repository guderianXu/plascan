#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace xjw::mesh
{

constexpr std::size_t kDepthGeometrySourceSlotCount = 256;
constexpr std::size_t kDepthGeometrySourceWordCount = 4;
constexpr std::size_t kDepthGeometryLocalSourceSlotCount = 16;

class DepthGeometrySourceMask
{
public:
    constexpr DepthGeometrySourceMask() noexcept = default;

    constexpr DepthGeometrySourceMask(std::uint64_t lowBits) noexcept
        : _words{lowBits, 0, 0, 0}
    {
    }

    static constexpr DepthGeometrySourceMask singleBit(
        std::size_t slot) noexcept
    {
        DepthGeometrySourceMask mask;
        if (slot < kDepthGeometrySourceSlotCount)
        {
            mask._words[slot / 64U] =
                std::uint64_t{1} << static_cast<unsigned>(slot % 64U);
        }
        return mask;
    }

    constexpr bool any() const noexcept
    {
        return _words[0] != 0 || _words[1] != 0 ||
            _words[2] != 0 || _words[3] != 0;
    }

    constexpr bool none() const noexcept
    {
        return !any();
    }

    constexpr bool testBit(std::size_t slot) const noexcept
    {
        return slot < kDepthGeometrySourceSlotCount &&
            (_words[slot / 64U] &
             (std::uint64_t{1} << static_cast<unsigned>(slot % 64U))) != 0;
    }

    constexpr std::size_t count() const noexcept
    {
        std::size_t result = 0;
        for (const std::uint64_t word : _words)
        {
            result += static_cast<std::size_t>(std::popcount(word));
        }
        return result;
    }

    constexpr const std::array<std::uint64_t,
                               kDepthGeometrySourceWordCount> &words()
        const noexcept
    {
        return _words;
    }

    constexpr DepthGeometrySourceMask &operator|=(
        const DepthGeometrySourceMask &other) noexcept
    {
        for (std::size_t index = 0; index < _words.size(); ++index)
        {
            _words[index] |= other._words[index];
        }
        return *this;
    }

    constexpr DepthGeometrySourceMask &operator&=(
        const DepthGeometrySourceMask &other) noexcept
    {
        for (std::size_t index = 0; index < _words.size(); ++index)
        {
            _words[index] &= other._words[index];
        }
        return *this;
    }

    constexpr bool operator==(
        const DepthGeometrySourceMask &other) const noexcept = default;

    constexpr bool operator!=(
        const DepthGeometrySourceMask &other) const noexcept
    {
        return !(*this == other);
    }

    friend constexpr DepthGeometrySourceMask operator|(
        DepthGeometrySourceMask left,
        const DepthGeometrySourceMask &right) noexcept
    {
        left |= right;
        return left;
    }

    friend constexpr DepthGeometrySourceMask operator&(
        DepthGeometrySourceMask left,
        const DepthGeometrySourceMask &right) noexcept
    {
        left &= right;
        return left;
    }

    friend constexpr DepthGeometrySourceMask operator<<(
        const DepthGeometrySourceMask &mask,
        std::size_t shift) noexcept
    {
        if (shift >= kDepthGeometrySourceSlotCount)
        {
            return {};
        }

        DepthGeometrySourceMask shifted;
        const std::size_t word_shift = shift / 64U;
        const unsigned bit_shift = static_cast<unsigned>(shift % 64U);
        for (std::size_t destination = word_shift;
             destination < shifted._words.size();
             ++destination)
        {
            const std::size_t source = destination - word_shift;
            shifted._words[destination] = mask._words[source] << bit_shift;
            if (bit_shift != 0U && source > 0)
            {
                shifted._words[destination] |=
                    mask._words[source - 1] >> (64U - bit_shift);
            }
        }
        return shifted;
    }

private:
    std::array<std::uint64_t, kDepthGeometrySourceWordCount> _words{};
};

static_assert(
    kDepthGeometrySourceSlotCount == kDepthGeometrySourceWordCount * 64U);
static_assert(
    sizeof(DepthGeometrySourceMask) ==
    kDepthGeometrySourceWordCount * sizeof(std::uint64_t));

struct DepthGeometrySourceReferenceCount
{
    int sourceIndex = -1;
    std::uint64_t referenceCount = 0;
    int mappedSlot = -1;
};

struct DepthGeometrySourceEncodingStatistics
{
    std::uint64_t frameCount = 0;
    std::uint64_t validReferenceCount = 0;
    std::uint64_t ignoredNegativeReferenceCount = 0;
    std::uint64_t mappedReferenceCount = 0;
    std::uint64_t unmappedReferenceCount = 0;
    std::size_t distinctSourceCount = 0;
    std::size_t mappedSourceCount = 0;
    std::size_t unmappedSourceCount = 0;
};

/**
 * @brief Cached mapping from a frame's 16 local source bits to global bits.
 *
 * An ordinal contains either one exact global source bit or zero when the
 * source is negative, missing, or outside the globally selected top 256.
 */
struct DepthGeometryLocalSourceEncoding
{
    std::array<DepthGeometrySourceMask,
               kDepthGeometryLocalSourceSlotCount> ordinalMasks{};

    DepthGeometrySourceMask encode(std::uint16_t localMask) const noexcept;

    std::uint16_t mappedLocalMask(std::uint16_t localMask) const noexcept;
};

/**
 * @brief Deterministic, collision-free encoding of global geometry sources.
 *
 * Sources are ranked by descending reference count, with ascending source ID
 * as the deterministic tie-break. The first 256 receive unique bits.
 * Every other source is conservatively left unmapped and can never alias a
 * selected source or another unmapped source.
 */
class DepthGeometrySourceEncoding
{
public:
    static DepthGeometrySourceEncoding build(
        const std::vector<std::vector<int>> &frameSourceIndices);

    const std::vector<int> &slotToSourceIndex() const noexcept;
    const std::vector<DepthGeometrySourceReferenceCount> &sourceReferenceCounts()
        const noexcept;
    const DepthGeometrySourceEncodingStatistics &statistics() const noexcept;

    int slotForSourceIndex(int sourceIndex) const noexcept;

    DepthGeometryLocalSourceEncoding makeLocalEncoding(
        const std::vector<int> &sourceIndices) const;

    DepthGeometrySourceMask encodeLocalMask(
        const std::vector<int> &sourceIndices,
        std::uint16_t localMask) const;

private:
    std::vector<int> _slotToSourceIndex;
    std::vector<DepthGeometrySourceReferenceCount> _sourceReferenceCounts;
    DepthGeometrySourceEncodingStatistics _statistics;
    std::unordered_map<int, std::size_t> _sourceIndexToSlot;
};

} // namespace xjw::mesh
