#include "metmodel/patchmatch_store.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <type_traits>
#include <unordered_set>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace metmodel {
namespace {

constexpr std::array<char, 8> magic{'M', 'M', 'P', 'M', 'D', '4', '0', '1'};
constexpr std::uint32_t format_version = 1U;
constexpr std::uint64_t maximum_elements = 1ULL << 32U;
std::atomic<std::uint64_t> temporary_sequence{0U};

template <class T>
std::uint64_t hash_values(std::span<const T> values) {
    const auto bytes = std::as_bytes(values);
    std::uint64_t hash = 14695981039346656037ULL;
    for (const std::byte value : bytes) {
        hash ^= std::to_integer<std::uint8_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

template <class T>
bool write_scalar(std::ostream& stream, T value) {
    static_assert(std::is_unsigned_v<T>);
    std::array<char, sizeof(T)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<char>((value >> (index * 8U)) & 0xffU);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(stream);
}

template <class T>
bool read_scalar(std::istream& stream, T& value) {
    static_assert(std::is_unsigned_v<T>);
    std::array<unsigned char, sizeof(T)> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!stream) return false;
    value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= static_cast<T>(bytes[index]) << (index * 8U);
    return true;
}

std::filesystem::path camera_path(const std::filesystem::path& root,
                                  std::size_t camera_index) {
    return root / ("camera_" + std::to_string(camera_index) + ".mpmd4");
}

bool sync_path(const std::filesystem::path& path, bool directory) {
#if defined(__linux__)
    const int flags = O_RDONLY | (directory ? O_DIRECTORY : 0);
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0) return false;
    const bool ok = ::fsync(descriptor) == 0;
    const bool closed = ::close(descriptor) == 0;
    return ok && closed;
#else
    (void)path;
    (void)directory;
    return true;
#endif
}

template <class T>
bool write_vector(std::ostream& stream, std::span<const T> values) {
    if (!write_scalar(stream, static_cast<std::uint64_t>(values.size())) ||
        !write_scalar(stream, hash_values(values))) return false;
    const auto bytes = std::as_bytes(values);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(stream);
}

template <class T>
bool read_vector(std::istream& stream, std::vector<T>& values) {
    std::uint64_t count = 0, expected_hash = 0;
    if (!read_scalar(stream, count) || !read_scalar(stream, expected_hash) ||
        count > maximum_elements ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) return false;
    values.resize(static_cast<std::size_t>(count));
    const auto bytes = std::as_writable_bytes(std::span<T>(values));
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    return stream && hash_values<T>(values) == expected_hash;
}

}  // namespace

bool write_recovered_patchmatch_store_camera(
    const std::filesystem::path& root,
    const RecoveredPatchMatchD4PyramidOutput& camera,
    std::string& error) {
    try {
        if (camera.ranked_neighbor_camera_indices.empty()) {
            error = "PatchMatch store camera has no ranked neighbours";
            return false;
        }
        const std::size_t groups =
            (camera.ranked_neighbor_camera_indices.size() + 7U) / 8U;
        for (std::size_t level = 0; level < 3U; ++level) {
            if (camera.depth_levels[level].empty() ||
                camera.depth_levels[level].size() >
                    std::numeric_limits<std::size_t>::max() / groups ||
                camera.packed_inlier_masks[level].size() !=
                    groups * camera.depth_levels[level].size()) {
                error = "PatchMatch store level dimensions are invalid";
                return false;
            }
        }
        std::filesystem::create_directories(root);
        const auto destination = camera_path(root, camera.camera_index);
        if (std::filesystem::exists(destination)) {
            error = "PatchMatch store camera already exists";
            return false;
        }
        const auto temporary = destination.string() + ".tmp-" +
            std::to_string(temporary_sequence.fetch_add(1U));
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            error = "cannot create PatchMatch store temporary file";
            return false;
        }
        stream.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        bool ok = stream && write_scalar(stream, format_version) &&
            write_scalar(stream, static_cast<std::uint64_t>(camera.camera_index)) &&
            write_scalar(stream, static_cast<std::uint64_t>(
                camera.ranked_neighbor_camera_indices.size()));
        for (const std::size_t neighbor : camera.ranked_neighbor_camera_indices)
            ok = ok && write_scalar(stream, static_cast<std::uint64_t>(neighbor));
        for (std::size_t level = 0; ok && level < 3U; ++level) {
            ok = write_vector(stream, std::span<const float>(camera.depth_levels[level])) &&
                 write_vector(stream, std::span<const std::uint8_t>(
                     camera.packed_inlier_masks[level]));
        }
        stream.flush();
        ok = ok && static_cast<bool>(stream);
        stream.close();
        if (!ok) {
            std::filesystem::remove(temporary);
            error = "short write to PatchMatch store camera";
            return false;
        }
        if (!sync_path(temporary, false)) {
            std::filesystem::remove(temporary);
            error = "cannot sync PatchMatch store temporary file";
            return false;
        }
        std::filesystem::rename(temporary, destination);
        if (!sync_path(root, true)) {
            error = "cannot sync PatchMatch store directory";
            return false;
        }
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool read_recovered_patchmatch_store_camera(
    const std::filesystem::path& root,
    std::size_t expected_camera_index,
    RecoveredPatchMatchD4PyramidOutput& camera,
    std::string& error) {
    try {
        std::ifstream stream(camera_path(root, expected_camera_index),
                             std::ios::binary);
        std::array<char, 8> actual_magic{};
        std::uint32_t version = 0;
        std::uint64_t camera_index = 0;
        stream.read(actual_magic.data(),
                    static_cast<std::streamsize>(actual_magic.size()));
        RecoveredPatchMatchD4PyramidOutput result;
        std::uint64_t neighbor_count = 0;
        if (!stream || actual_magic != magic ||
            !read_scalar(stream, version) || version != format_version ||
            !read_scalar(stream, camera_index) ||
            camera_index != expected_camera_index ||
            !read_scalar(stream, neighbor_count) ||
            neighbor_count > maximum_elements) {
            error = "PatchMatch store header or identity is invalid";
            return false;
        }
        result.camera_index = expected_camera_index;
        result.ranked_neighbor_camera_indices.resize(
            static_cast<std::size_t>(neighbor_count));
        for (std::size_t& neighbor : result.ranked_neighbor_camera_indices) {
            std::uint64_t encoded = 0;
            if (!read_scalar(stream, encoded) ||
                encoded > std::numeric_limits<std::size_t>::max()) {
                error = "PatchMatch store neighbour identity is invalid";
                return false;
            }
            neighbor = static_cast<std::size_t>(encoded);
        }
        for (std::size_t level = 0; level < 3U; ++level) {
            if (!read_vector(stream, result.depth_levels[level]) ||
                !read_vector(stream, result.packed_inlier_masks[level])) {
                error = "PatchMatch store payload or checksum is invalid";
                return false;
            }
        }
        const std::size_t groups =
            (result.ranked_neighbor_camera_indices.size() + 7U) / 8U;
        if (groups == 0U) {
            error = "PatchMatch store has no ranked neighbours";
            return false;
        }
        for (std::size_t level = 0; level < 3U; ++level) {
            if (result.depth_levels[level].empty() ||
                result.depth_levels[level].size() >
                    std::numeric_limits<std::size_t>::max() / groups ||
                result.packed_inlier_masks[level].size() !=
                    groups * result.depth_levels[level].size()) {
                error = "PatchMatch store level dimensions are invalid";
                return false;
            }
        }
        char trailing = 0;
        if (stream.read(&trailing, 1)) {
            error = "PatchMatch store file has trailing bytes";
            return false;
        }
        if (!stream.eof()) {
            error = "PatchMatch store file ended unexpectedly";
            return false;
        }
        camera = std::move(result);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool plan_recovered_patchmatch_store_batches(
    std::span<const std::size_t> references,
    std::span<const std::vector<std::size_t>> ranked_neighbors_by_camera,
    std::size_t maximum_references_per_batch,
    std::vector<RecoveredPatchMatchStoreBatch>& batches,
    std::string& error) {
    if (references.empty() || maximum_references_per_batch == 0U) {
        error = "PatchMatch store batch domain is empty";
        return false;
    }
    std::unordered_set<std::size_t> reference_set(references.begin(), references.end());
    if (reference_set.size() != references.size()) {
        error = "PatchMatch store references are duplicated";
        return false;
    }
    std::vector<RecoveredPatchMatchStoreBatch> result;
    for (std::size_t begin = 0; begin < references.size();
         begin += maximum_references_per_batch) {
        const std::size_t end = std::min(references.size(),
                                        begin + maximum_references_per_batch);
        RecoveredPatchMatchStoreBatch batch;
        batch.references.assign(references.begin() + static_cast<std::ptrdiff_t>(begin),
                                references.begin() + static_cast<std::ptrdiff_t>(end));
        std::unordered_set<std::size_t> closure;
        for (const std::size_t reference : batch.references) {
            if (reference >= ranked_neighbors_by_camera.size()) {
                error = "PatchMatch store reference has no neighbour row";
                return false;
            }
            closure.insert(reference);
            for (const std::size_t neighbor : ranked_neighbors_by_camera[reference]) {
                if (!reference_set.contains(neighbor)) {
                    error = "PatchMatch store voting closure omits a neighbour product";
                    return false;
                }
                closure.insert(neighbor);
            }
        }
        batch.closure.assign(closure.begin(), closure.end());
        std::sort(batch.closure.begin(), batch.closure.end());
        result.push_back(std::move(batch));
    }
    batches = std::move(result);
    error.clear();
    return true;
}

}  // namespace metmodel
