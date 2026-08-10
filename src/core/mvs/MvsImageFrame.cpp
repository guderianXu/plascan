#include "MvsImageCache.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace xjw::mvs
{
namespace
{

struct AllocationSpan
{
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

void appendAllocation(const cv::Mat &matrix,
                      std::array<AllocationSpan, 3> *spans,
                      std::size_t *span_count) noexcept
{
    if (!spans || !span_count || *span_count >= spans->size()
        || matrix.empty() || !matrix.datastart || !matrix.dataend)
    {
        return;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(matrix.datastart);
    const auto end = reinterpret_cast<std::uintptr_t>(matrix.dataend);
    if (end > begin)
    {
        (*spans)[*span_count] = {begin, end};
        ++(*span_count);
    }
}

} // namespace

std::uint64_t MvsImageFrame::residentBytes() const noexcept
{
    std::array<AllocationSpan, 3> spans{};
    std::size_t span_count = 0;
    appendAllocation(gray, &spans, &span_count);
    appendAllocation(preparedGray, &spans, &span_count);
    appendAllocation(validMask, &spans, &span_count);
    if (span_count == 0)
    {
        return 0;
    }

    const auto less = [](const AllocationSpan &left, const AllocationSpan &right) noexcept
    {
        return left.begin < right.begin ||
            (left.begin == right.begin && left.end < right.end);
    };
    for (std::size_t index = 1; index < span_count; ++index)
    {
        AllocationSpan current = spans[index];
        std::size_t insertion_index = index;
        while (insertion_index > 0 && less(current, spans[insertion_index - 1]))
        {
            spans[insertion_index] = spans[insertion_index - 1];
            --insertion_index;
        }
        spans[insertion_index] = current;
    }
    std::uint64_t bytes = 0;
    std::uintptr_t allocation_begin = spans.front().begin;
    std::uintptr_t allocation_end = spans.front().end;
    for (std::size_t index = 1; index < span_count; ++index)
    {
        if (spans[index].begin <= allocation_end)
        {
            allocation_end = std::max(allocation_end, spans[index].end);
            continue;
        }
        bytes += static_cast<std::uint64_t>(allocation_end - allocation_begin);
        allocation_begin = spans[index].begin;
        allocation_end = spans[index].end;
    }
    return bytes + static_cast<std::uint64_t>(allocation_end - allocation_begin);
}

} // namespace xjw::mvs
