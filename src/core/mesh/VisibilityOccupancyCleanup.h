#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace xjw::mesh::detail
{

std::uint64_t retainLargestVisibilityFullComponent(
    const std::array<int, 3> &dimensions,
    std::vector<std::uint8_t> *occupied);

std::uint64_t fillInteriorVisibilityEmptyBubbles(
    const std::array<int, 3> &dimensions,
    std::vector<std::uint8_t> *occupied,
    const std::vector<std::uint8_t> *protectedEmpty = nullptr);

std::uint64_t closeVisibilityOccupancySixConnected(
    const std::array<int, 3> &dimensions,
    int iterations,
    std::vector<std::uint8_t> *occupied,
    const std::vector<std::uint8_t> *protectedEmpty = nullptr);

std::vector<float> visibilitySignedDistanceSamples(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    int maximumDistance);

} // namespace xjw::mesh::detail
