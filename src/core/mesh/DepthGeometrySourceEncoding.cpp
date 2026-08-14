#include "DepthGeometrySourceEncoding.h"

#include <algorithm>
#include <limits>

namespace xjw::mesh
{

DepthGeometrySourceMask DepthGeometryLocalSourceEncoding::encode(
    std::uint16_t localMask) const noexcept
{
    DepthGeometrySourceMask encoded;
    for (std::size_t ordinal = 0;
         ordinal < ordinalMasks.size();
         ++ordinal)
    {
        const std::uint16_t local_bit = static_cast<std::uint16_t>(
            std::uint16_t{1} << ordinal);
        if ((localMask & local_bit) != 0)
        {
            encoded |= ordinalMasks[ordinal];
        }
    }
    return encoded;
}

std::uint16_t DepthGeometryLocalSourceEncoding::mappedLocalMask(
    std::uint16_t localMask) const noexcept
{
    std::uint16_t mapped = 0;
    for (std::size_t ordinal = 0;
         ordinal < ordinalMasks.size();
         ++ordinal)
    {
        const std::uint16_t local_bit = static_cast<std::uint16_t>(
            std::uint16_t{1} << ordinal);
        if ((localMask & local_bit) != 0 && ordinalMasks[ordinal].any())
        {
            mapped = static_cast<std::uint16_t>(mapped | local_bit);
        }
    }
    return mapped;
}

DepthGeometrySourceEncoding DepthGeometrySourceEncoding::build(
    const std::vector<std::vector<int>> &frameSourceIndices)
{
    DepthGeometrySourceEncoding encoding;
    encoding._statistics.frameCount = static_cast<std::uint64_t>(
        frameSourceIndices.size());

    std::unordered_map<int, std::uint64_t> reference_counts;
    for (const std::vector<int> &source_indices : frameSourceIndices)
    {
        for (const int source_index : source_indices)
        {
            if (source_index < 0)
            {
                ++encoding._statistics.ignoredNegativeReferenceCount;
                continue;
            }

            ++encoding._statistics.validReferenceCount;
            std::uint64_t &count = reference_counts[source_index];
            if (count < std::numeric_limits<std::uint64_t>::max())
            {
                ++count;
            }
        }
    }

    encoding._sourceReferenceCounts.reserve(reference_counts.size());
    for (const auto &[source_index, reference_count] : reference_counts)
    {
        encoding._sourceReferenceCounts.push_back(
            {source_index, reference_count, -1});
    }
    std::sort(
        encoding._sourceReferenceCounts.begin(),
        encoding._sourceReferenceCounts.end(),
        [](const DepthGeometrySourceReferenceCount &left,
           const DepthGeometrySourceReferenceCount &right)
        {
            if (left.referenceCount != right.referenceCount)
            {
                return left.referenceCount > right.referenceCount;
            }
            return left.sourceIndex < right.sourceIndex;
        });

    encoding._statistics.distinctSourceCount =
        encoding._sourceReferenceCounts.size();
    encoding._statistics.mappedSourceCount = std::min(
        kDepthGeometrySourceSlotCount,
        encoding._sourceReferenceCounts.size());
    encoding._statistics.unmappedSourceCount =
        encoding._statistics.distinctSourceCount -
        encoding._statistics.mappedSourceCount;
    encoding._slotToSourceIndex.reserve(
        encoding._statistics.mappedSourceCount);
    encoding._sourceIndexToSlot.reserve(
        encoding._statistics.mappedSourceCount);

    for (std::size_t slot = 0;
         slot < encoding._sourceReferenceCounts.size();
         ++slot)
    {
        DepthGeometrySourceReferenceCount &source =
            encoding._sourceReferenceCounts[slot];
        if (slot < kDepthGeometrySourceSlotCount)
        {
            source.mappedSlot = static_cast<int>(slot);
            encoding._slotToSourceIndex.push_back(source.sourceIndex);
            encoding._sourceIndexToSlot.emplace(source.sourceIndex, slot);
            encoding._statistics.mappedReferenceCount +=
                source.referenceCount;
        }
        else
        {
            encoding._statistics.unmappedReferenceCount +=
                source.referenceCount;
        }
    }
    return encoding;
}

const std::vector<int> &DepthGeometrySourceEncoding::slotToSourceIndex()
    const noexcept
{
    return _slotToSourceIndex;
}

const std::vector<DepthGeometrySourceReferenceCount> &
DepthGeometrySourceEncoding::sourceReferenceCounts() const noexcept
{
    return _sourceReferenceCounts;
}

const DepthGeometrySourceEncodingStatistics &
DepthGeometrySourceEncoding::statistics() const noexcept
{
    return _statistics;
}

int DepthGeometrySourceEncoding::slotForSourceIndex(
    int sourceIndex) const noexcept
{
    const auto found = _sourceIndexToSlot.find(sourceIndex);
    return found == _sourceIndexToSlot.end()
        ? -1
        : static_cast<int>(found->second);
}

DepthGeometryLocalSourceEncoding
DepthGeometrySourceEncoding::makeLocalEncoding(
    const std::vector<int> &sourceIndices) const
{
    DepthGeometryLocalSourceEncoding local_encoding;
    const std::size_t ordinal_count = std::min(
        sourceIndices.size(), kDepthGeometryLocalSourceSlotCount);
    for (std::size_t ordinal = 0; ordinal < ordinal_count; ++ordinal)
    {
        const int slot = slotForSourceIndex(sourceIndices[ordinal]);
        if (slot >= 0)
        {
            local_encoding.ordinalMasks[ordinal] =
                DepthGeometrySourceMask::singleBit(
                    static_cast<std::size_t>(slot));
        }
    }
    return local_encoding;
}

DepthGeometrySourceMask DepthGeometrySourceEncoding::encodeLocalMask(
    const std::vector<int> &sourceIndices,
    std::uint16_t localMask) const
{
    return makeLocalEncoding(sourceIndices).encode(localMask);
}

} // namespace xjw::mesh
