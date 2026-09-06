#include "metalign/features.hpp"
#include "metalign/gpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(METALIGN_HAS_OPENMP)
#include <omp.h>
#endif

namespace metalign {
namespace {

constexpr float kPi = 3.1415927410125732F;
constexpr float kTwoPi = 2.0F * kPi;
constexpr int kIntervals = 3;
constexpr int kGaussianLevels = 5;
constexpr int kOctaves = 6;
constexpr float kSigma0 = 1.6F;
constexpr float kScaleStep = 1.2599210498948732F;  // 2^(1/3)

struct DetectorReplayInvocation {
    std::size_t iteration_count = 0;
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> ordered_worker_ranges;
};

using DetectorReplaySchedule = std::array<DetectorReplayInvocation, kOctaves>;

bool environment_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

const std::filesystem::path* gomp_capture_directory() {
    static const std::optional<std::filesystem::path> directory = [] {
        const char* value = std::getenv("METALIGN_CAPTURE_GOMP_SCHEDULES");
        if (!value || *value == '\0')
            return std::optional<std::filesystem::path>{};
        return std::optional<std::filesystem::path>{value};
    }();
    return directory ? &*directory : nullptr;
}

std::mutex& gomp_capture_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::pair<std::size_t, std::size_t>> compress_iterations(
    const std::vector<std::size_t>& iterations) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    for (std::size_t iteration : iterations) {
        if (!ranges.empty() && ranges.back().second == iteration)
            ranges.back().second = iteration + 1;
        else
            ranges.emplace_back(iteration, iteration + 1);
    }
    return ranges;
}

void capture_detector_schedule(
    const std::filesystem::path& image, std::size_t octave,
    std::size_t iteration_count,
    const std::vector<std::vector<std::size_t>>& worker_iterations,
    const std::vector<std::size_t>& critical_order) {
    const auto* directory = gomp_capture_directory();
    if (!directory) return;
    std::lock_guard lock(gomp_capture_mutex());
    std::ofstream stream(*directory / "detector_schedule.txt", std::ios::app);
    if (!stream) throw std::runtime_error("cannot append detector schedule capture");
    stream << image.filename().string() << ' ' << octave << ' '
           << iteration_count << ' ' << critical_order.size();
    for (std::size_t worker : critical_order) {
        if (worker >= worker_iterations.size())
            throw std::runtime_error("invalid detector critical order");
        const auto ranges = compress_iterations(worker_iterations[worker]);
        stream << ' ' << ranges.size();
        for (const auto& [begin, end] : ranges)
            stream << ' ' << begin << ' ' << end;
    }
    stream << '\n';
}

void capture_orientation_schedule(const std::filesystem::path& image,
                                  const char* stream_name,
                                  const std::vector<std::size_t>& order) {
    const auto* directory = gomp_capture_directory();
    if (!directory) return;
    std::lock_guard lock(gomp_capture_mutex());
    std::ofstream stream(*directory /
        (std::string(stream_name) + "_orientation_schedule.txt"), std::ios::app);
    if (!stream) throw std::runtime_error("cannot append orientation schedule capture");
    stream << image.filename().string();
    for (std::size_t worker : order) stream << ' ' << worker;
    stream << '\n';
}

const std::map<std::string, DetectorReplaySchedule>& target_detector_replay_schedule() {
    static const std::map<std::string, DetectorReplaySchedule> schedule = [] {
        const char* filename = std::getenv("METALIGN_REPLAY_TARGET_DETECTOR");
        if (!filename || *filename == '\0')
            return std::map<std::string, DetectorReplaySchedule>{};
        std::ifstream stream(filename);
        if (!stream)
            throw std::runtime_error(
                std::string("cannot open target detector schedule: ") + filename);
        std::map<std::string, DetectorReplaySchedule> result;
        std::map<std::string, std::array<bool, kOctaves>> seen;
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty() || line.front() == '#') continue;
            std::istringstream parser(line);
            std::string image;
            int octave = -1;
            std::size_t iteration_count = 0;
            std::size_t worker_count = 0;
            if (!(parser >> image >> octave >> iteration_count >> worker_count) ||
                octave < 0 || octave >= kOctaves || worker_count != 20)
                throw std::runtime_error("invalid target detector schedule header");
            DetectorReplayInvocation invocation;
            invocation.iteration_count = iteration_count;
            invocation.ordered_worker_ranges.resize(worker_count);
            std::vector<unsigned char> coverage(iteration_count, 0);
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                std::size_t range_count = 0;
                if (!(parser >> range_count))
                    throw std::runtime_error("truncated target detector worker schedule");
                auto& ranges = invocation.ordered_worker_ranges[worker];
                ranges.reserve(range_count);
                for (std::size_t range = 0; range < range_count; ++range) {
                    std::size_t begin = 0;
                    std::size_t end = 0;
                    if (!(parser >> begin >> end) || begin >= end || end > iteration_count)
                        throw std::runtime_error("invalid target detector dynamic range");
                    for (std::size_t index = begin; index < end; ++index) {
                        if (coverage[index] != 0)
                            throw std::runtime_error("overlapping target detector ranges");
                        coverage[index] = 1;
                    }
                    ranges.emplace_back(begin, end);
                }
            }
            parser >> std::ws;
            if (!parser.eof() ||
                std::find(coverage.begin(), coverage.end(), 0) != coverage.end())
                throw std::runtime_error("incomplete target detector schedule row");
            auto& image_seen = seen[image];
            if (image_seen[static_cast<std::size_t>(octave)])
                throw std::runtime_error("duplicate target detector schedule row");
            image_seen[static_cast<std::size_t>(octave)] = true;
            result[image][static_cast<std::size_t>(octave)] = std::move(invocation);
        }
        if (result.empty())
            throw std::runtime_error("target detector schedule is empty");
        for (const auto& [image, octaves] : seen) {
            static_cast<void>(image);
            if (std::find(octaves.begin(), octaves.end(), false) != octaves.end())
                throw std::runtime_error("target detector image lacks an octave");
        }
        return result;
    }();
    return schedule;
}

const DetectorReplaySchedule* target_detector_replay(const std::filesystem::path& path) {
    const auto& schedules = target_detector_replay_schedule();
    if (schedules.empty()) return nullptr;
    const auto found = schedules.find(path.filename().string());
    if (found == schedules.end())
        throw std::runtime_error(
            "target detector schedule lacks image " + path.filename().string());
    return &found->second;
}

// Analysis-only replay of an *observed* unnamed-GOMP critical-entry order.
// The production path deliberately leaves that order to libgomp: the target
// itself varies it between processes.  A Phase27 manifest carries one
// filename and one worker-id permutation per selected orientation stream.
std::map<std::string, std::vector<std::size_t>>
load_target_orientation_replay_schedule(const char* environment,
                                        const char* stream_name) {
    const char* filename = std::getenv(environment);
        if (!filename || *filename == '\0')
            return std::map<std::string, std::vector<std::size_t>>{};
        std::ifstream stream(filename);
        if (!stream)
            throw std::runtime_error(
                std::string("cannot open target ") + stream_name +
                "-orientation schedule: " + filename);
        std::map<std::string, std::vector<std::size_t>> result;
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty() || line.front() == '#') continue;
            std::istringstream parser(line);
            std::string image_name;
            if (!(parser >> image_name)) continue;
            std::vector<std::size_t> order;
            std::size_t worker = 0;
            while (parser >> worker) order.push_back(worker);
            if (order.empty() || !parser.eof())
                throw std::runtime_error(
                    std::string("invalid target ") + stream_name +
                    "-orientation schedule row for " + image_name);
            if (!result.emplace(std::move(image_name), std::move(order)).second)
                throw std::runtime_error(std::string("duplicate target ") + stream_name +
                                         "-orientation schedule row");
        }
        if (result.empty())
            throw std::runtime_error(std::string("target ") + stream_name +
                                     "-orientation schedule is empty");
        return result;
}

const std::map<std::string, std::vector<std::size_t>>&
target_coarse_orientation_replay_schedule() {
    static const std::map<std::string, std::vector<std::size_t>> schedule =
        load_target_orientation_replay_schedule(
            "METALIGN_REPLAY_TARGET_COARSE_ORIENTATION", "coarse");
    return schedule;
}

const std::map<std::string, std::vector<std::size_t>>&
target_full_orientation_replay_schedule() {
    static const std::map<std::string, std::vector<std::size_t>> schedule =
        load_target_orientation_replay_schedule(
            "METALIGN_REPLAY_TARGET_FULL_ORIENTATION", "full");
    return schedule;
}

const std::vector<std::size_t>* target_orientation_replay_order(
    const std::filesystem::path& image_path, std::size_t selected_rows) {
    const std::map<std::string, std::vector<std::size_t>>* schedule = nullptr;
    const char* stream_name = nullptr;
    if (selected_rows == 2048) {
        schedule = &target_coarse_orientation_replay_schedule();
        stream_name = "coarse";
    } else {
        schedule = &target_full_orientation_replay_schedule();
        stream_name = "full";
    }
    if (schedule->empty()) return nullptr;
    const auto found = schedule->find(image_path.filename().string());
    if (found == schedule->end())
        throw std::runtime_error(
            std::string("target ") + stream_name + "-orientation schedule lacks " +
            image_path.filename().string());
    return &found->second;
}

struct Candidate {
    float x = 0.0F;
    float y = 0.0F;
    float scale = 1.0F;
    float response = 0.0F;
    int octave = 0;
    int level = 0;
    int laplacian_sign = 1;
};

struct SelectorCandidateRecord {
    float x;
    float y;
    float z;
    float scale;
    float sign_or_orientation;
    float response;
    std::uint32_t octave;
    std::uint32_t level;
    std::uint32_t flag;
};
static_assert(sizeof(SelectorCandidateRecord) == 36);

using CandidateIdentity = std::array<std::uint32_t, 7>;

CandidateIdentity candidate_identity(const Candidate& candidate) {
    return {std::bit_cast<std::uint32_t>(candidate.x),
            std::bit_cast<std::uint32_t>(candidate.y),
            std::bit_cast<std::uint32_t>(candidate.scale),
            std::bit_cast<std::uint32_t>(candidate.response),
            static_cast<std::uint32_t>(candidate.octave),
            static_cast<std::uint32_t>(candidate.level),
            candidate.laplacian_sign > 0 ? 1U : 0U};
}

CandidateIdentity candidate_identity(const SelectorCandidateRecord& record) {
    return {std::bit_cast<std::uint32_t>(record.x),
            std::bit_cast<std::uint32_t>(record.y),
            std::bit_cast<std::uint32_t>(record.scale),
            std::bit_cast<std::uint32_t>(record.response),
            record.octave, record.level, record.flag};
}

void replay_target_candidate_order(const std::filesystem::path& image_path,
                                   std::vector<Candidate>& candidates,
                                   const char* environment,
                                   const char* suffix,
                                   const char* stream_name) {
    const char* directory = std::getenv(environment);
    if (!directory || *directory == '\0') return;
    const std::filesystem::path source = std::filesystem::path(directory) /
        (image_path.filename().string() + suffix);
    std::ifstream stream(source, std::ios::binary);
    if (!stream)
        throw std::runtime_error(std::string("cannot open target ") + stream_name +
                                 " candidate order: " + source.string());
    std::vector<SelectorCandidateRecord> target_rows;
    SelectorCandidateRecord record{};
    while (stream.read(reinterpret_cast<char*>(&record), sizeof(record)))
        target_rows.push_back(record);
    if (!stream.eof())
        throw std::runtime_error(std::string("truncated target ") + stream_name +
                                 " candidate order: " + source.string());
    if (target_rows.size() != candidates.size())
        throw std::runtime_error(std::string("target ") + stream_name +
                                 " candidate count differs for " +
                                 image_path.filename().string());
    std::map<CandidateIdentity, std::vector<Candidate>> available;
    for (Candidate& candidate : candidates)
        available[candidate_identity(candidate)].push_back(std::move(candidate));
    std::vector<Candidate> ordered;
    ordered.reserve(candidates.size());
    for (const SelectorCandidateRecord& target : target_rows) {
        auto found = available.find(candidate_identity(target));
        if (found == available.end() || found->second.empty())
            throw std::runtime_error(std::string("target ") + stream_name +
                                     " candidate is absent from replica: " +
                                     image_path.filename().string());
        ordered.push_back(std::move(found->second.back()));
        found->second.pop_back();
    }
    for (const auto& [identity, remaining] : available) {
        static_cast<void>(identity);
        if (!remaining.empty())
            throw std::runtime_error(std::string("replica ") + stream_name +
                                     " candidate is absent from target: " +
                                     image_path.filename().string());
    }
    candidates = std::move(ordered);
}

void dump_raw_selector_candidates(const std::filesystem::path& image_path,
                                  const std::vector<Candidate>& candidates,
                                  const char* stage) {
    const char* directory = std::getenv("METALIGN_DUMP_RAW_SELECTOR_CANDIDATES");
    if (!directory || *directory == '\0') return;
    if (const char* filter = std::getenv("METALIGN_DUMP_RAW_SELECTOR_STAGE");
        filter != nullptr && *filter != '\0' && std::strcmp(filter, stage) != 0)
        return;
    if (const char* filter = std::getenv("METALIGN_DUMP_RAW_SELECTOR_IMAGE");
        filter != nullptr && *filter != '\0' &&
        std::strstr(filter, image_path.filename().string().c_str()) == nullptr)
        return;
    std::filesystem::create_directories(directory);
    const std::filesystem::path output =
        std::filesystem::path(directory) /
        (image_path.filename().string() + "." + stage + ".bin");
    std::ofstream stream(output, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot write raw selector dump: " + output.string());
    for (const Candidate& candidate : candidates) {
        // Captured rows prove +16 is the pre-orientation -1 sentinel and +32
        // is the response/Laplacian-sign class: positive -> 1, negative -> 0.
        const SelectorCandidateRecord record{
            candidate.x, candidate.y, 0.0F, candidate.scale,
            -1.0F, candidate.response,
            static_cast<std::uint32_t>(candidate.octave),
            static_cast<std::uint32_t>(candidate.level),
            candidate.laplacian_sign > 0 ? 1U : 0U};
        stream.write(reinterpret_cast<const char*>(&record), sizeof(record));
    }
}

void dump_phase24_keypoints(const std::filesystem::path& image_path,
                            const std::vector<Keypoint>& keypoints,
                            const char* stage,
                            bool include_descriptors) {
    const char* directory = std::getenv("METALIGN_DUMP_PHASE24");
    if (!directory || *directory == '\0') return;
    const std::filesystem::path root(directory);
    std::filesystem::create_directories(root);
    const std::filesystem::path records_path =
        root / (image_path.filename().string() + "." + stage + ".records36.bin");
    std::ofstream records(records_path, std::ios::binary);
    if (!records)
        throw std::runtime_error("cannot write Phase 24 row dump: " + records_path.string());
    std::ofstream descriptors;
    std::filesystem::path descriptors_path;
    if (include_descriptors) {
        descriptors_path =
            root / (image_path.filename().string() + "." + stage + ".descriptors64.bin");
        descriptors.open(descriptors_path, std::ios::binary);
        if (!descriptors)
            throw std::runtime_error(
                "cannot write Phase 24 descriptor dump: " + descriptors_path.string());
    }
    for (const Keypoint& keypoint : keypoints) {
        const SelectorCandidateRecord record{
            static_cast<float>(keypoint.x), static_cast<float>(keypoint.y), 0.0F,
            static_cast<float>(keypoint.scale), static_cast<float>(keypoint.orientation),
            static_cast<float>(keypoint.response),
            static_cast<std::uint32_t>(keypoint.octave),
            static_cast<std::uint32_t>(keypoint.level),
            keypoint.laplacian_sign > 0 ? 1U : 0U};
        records.write(reinterpret_cast<const char*>(&record), sizeof(record));
        if (include_descriptors) {
            descriptors.write(reinterpret_cast<const char*>(keypoint.descriptor.data()),
                              static_cast<std::streamsize>(keypoint.descriptor.size()));
        }
    }
    if (!records || (include_descriptors && !descriptors))
        throw std::runtime_error("failed to write Phase 24 feature stream for " +
                                 image_path.string());
    if (include_descriptors &&
        (std::strcmp(stage, "coarse_hctree_order") == 0 ||
         std::strcmp(stage, "full_hctree_order") == 0)) {
        for (const int sign : {-1, 1}) {
            const char* sign_name = sign < 0 ? "negative" : "positive";
            const std::filesystem::path sign_path = root /
                (image_path.filename().string() + "." + stage + "." + sign_name +
                 ".descriptors64.bin");
            std::ofstream sign_stream(sign_path, std::ios::binary);
            if (!sign_stream)
                throw std::runtime_error(
                    "cannot write Phase 24 sign descriptor dump: " + sign_path.string());
            for (const Keypoint& keypoint : keypoints) {
                if (keypoint.laplacian_sign != sign) continue;
                sign_stream.write(
                    reinterpret_cast<const char*>(keypoint.descriptor.data()),
                    static_cast<std::streamsize>(keypoint.descriptor.size()));
            }
            if (!sign_stream)
                throw std::runtime_error(
                    "failed to write Phase 24 sign descriptor dump: " + sign_path.string());
        }
    }
}

struct Octave {
    std::array<Image, kGaussianLevels> gaussian;
    std::array<Image, kGaussianLevels> log;
};

void dump_gaussian_pyramid(const std::filesystem::path& image_path,
                           const std::vector<Octave>& pyramid) {
    const char* directory = std::getenv("METALIGN_DUMP_GAUSSIAN_PYRAMID");
    if (directory == nullptr || *directory == '\0') return;
    if (const char* filter = std::getenv("METALIGN_DUMP_GAUSSIAN_PYRAMID_IMAGE");
        filter != nullptr && *filter != '\0' && image_path.filename() != filter)
        return;
    const std::filesystem::path root(directory);
    std::filesystem::create_directories(root);
    std::ofstream manifest(root / "manifest.json");
    if (!manifest) throw std::runtime_error("cannot write Gaussian pyramid manifest");
    manifest << "{\n  \"image\": \"" << image_path.filename().string()
             << "\",\n  \"octaves\": [\n";
    for (std::size_t octave_index = 0; octave_index < pyramid.size(); ++octave_index) {
        const Octave& octave = pyramid[octave_index];
        const Image& shape = octave.gaussian[0];
        manifest << "    {\"octave\": " << octave_index
                 << ", \"width\": " << shape.width
                 << ", \"height\": " << shape.height
                 << ", \"stride\": " << shape.width << ", \"levels\": [";
        for (int level = 0; level < kGaussianLevels; ++level) {
            char name[64]{};
            std::snprintf(name, sizeof(name), "octave_%02zu_level_%02d.f32",
                          octave_index, level);
            const std::filesystem::path output = root / name;
            std::ofstream stream(output, std::ios::binary);
            if (!stream)
                throw std::runtime_error("cannot write Gaussian pyramid level: " +
                                         output.string());
            const Image& image = octave.gaussian[static_cast<std::size_t>(level)];
            stream.write(reinterpret_cast<const char*>(image.gray.data()),
                         static_cast<std::streamsize>(image.gray.size() * sizeof(float)));
            if (!stream)
                throw std::runtime_error("failed to write Gaussian pyramid level: " +
                                         output.string());
            if (level != 0) manifest << ", ";
            manifest << "{\"level\": " << level << ", \"file\": \"" << name << "\"}";
        }
        manifest << "]}" << (octave_index + 1 == pyramid.size() ? "\n" : ",\n");
    }
    manifest << "  ]\n}\n";
}

Image laplacian_response(const Image& image, float sigma, bool parallel_rows = false) {
    Image result;
    result.width = image.width;
    result.height = image.height;
    result.gray.assign(image.width * image.height, 0.0F);
    const float normalization = sigma * sigma;
#if defined(METALIGN_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(parallel_rows)
#endif
    for (std::ptrdiff_t row = 1;
         row < static_cast<std::ptrdiff_t>(image.height) - 1; ++row) {
        const std::size_t y = static_cast<std::size_t>(row);
        for (std::size_t x = 1; x + 1 < image.width; ++x) {
            const float center = image.at(x, y);
            result.at(x, y) = (4.0F * center - image.at(x - 1, y) - image.at(x + 1, y) -
                               image.at(x, y - 1) - image.at(x, y + 1)) * normalization;
        }
    }
    return result;
}

bool solve3x3(const float h[3][3], const float gradient[3], float offset[3]) {
    // Exact scalar cofactor expansion and grouping used by 0x26E5003--
    // 0x26E518B.  A generic inverse is algebraically equivalent, but its
    // rounding changes a small number of refined extrema and descriptor bits.
    const float cofactor_xz = h[0][1] * h[1][2] - h[1][1] * h[0][2];
    const float cofactor_xx = h[1][1] * h[2][2] - h[1][2] * h[1][2];
    const float cofactor_xy = h[0][2] * h[1][2] - h[2][2] * h[0][1];
    const float determinant =
        h[0][0] * cofactor_xx -
        (h[2][2] * h[0][1] - h[0][2] * h[1][2]) * h[0][1] +
        cofactor_xz * h[0][2];
    if (determinant == 0.0F) return false;
    const float inverse_determinant = 1.0F / determinant;
    offset[0] = -(cofactor_xz * gradient[2] +
                  (cofactor_xy * gradient[1] + cofactor_xx * gradient[0])) *
                inverse_determinant;
    const float cofactor_yz = h[0][1] * h[0][2] - h[1][2] * h[0][0];
    offset[1] = -((cofactor_xy * gradient[0] +
                   (h[2][2] * h[0][0] - h[0][2] * h[0][2]) * gradient[1]) +
                  cofactor_yz * gradient[2]) * inverse_determinant;
    const float cofactor_zz = h[0][0] * h[1][1] - h[0][1] * h[0][1];
    offset[2] = -(cofactor_yz * gradient[1] + cofactor_xz * gradient[0] +
                  cofactor_zz * gradient[2]) * inverse_determinant;
    return true;
}

bool refine_extremum(const Octave& octave, int level, int x, int y, Candidate& candidate) {
    const auto value = [&](int dl, int dx, int dy) {
        return octave.log[static_cast<std::size_t>(level + dl)].at(
            static_cast<std::size_t>(x + dx), static_cast<std::size_t>(y + dy));
    };
    const float center = value(0, 0, 0);
    const float gradient[3] = {
        0.5F * (value(0, 1, 0) - value(0, -1, 0)),
        0.5F * (value(0, 0, 1) - value(0, 0, -1)),
        0.5F * (value(1, 0, 0) - value(-1, 0, 0))};
    float hessian[3][3]{};
    hessian[0][0] = value(0, 1, 0) + value(0, -1, 0) - 2.0F * center;
    hessian[1][1] = value(0, 0, 1) + value(0, 0, -1) - 2.0F * center;
    hessian[2][2] = value(1, 0, 0) + value(-1, 0, 0) - 2.0F * center;
    // 0x26E4E16--0x26E4F2D does not evaluate the algebraically equivalent
    // left-associated `a-b-c+d`.  Its scalar SSE order is
    // up_left + ((down_right - down_left) - up_right), followed by *0.25.
    // South Building has real cases where those associations differ by one
    // ULP and move the refined response/selector rank.
    const float hxy_difference = value(0, -1, -1) +
        ((value(0, 1, 1) - value(0, -1, 1)) - value(0, 1, -1));
    hessian[0][1] = hessian[1][0] = hxy_difference * 0.25F;
    hessian[0][2] = hessian[2][0] = 0.25F *
        (value(1, 1, 0) - value(1, -1, 0) - value(-1, 1, 0) + value(-1, -1, 0));
    hessian[1][2] = hessian[2][1] = 0.25F *
        (value(1, 0, 1) - value(1, 0, -1) - value(-1, 0, 1) + value(-1, 0, -1));
    float offset[3]{};
    if (!solve3x3(hessian, gradient, offset)) return false;
    if (std::abs(offset[0]) > 1.0F || std::abs(offset[1]) > 1.0F ||
        std::abs(offset[2]) > 1.0F) return false;

    const float det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[0][1];
    if (det <= 0.0F) return false;
    const float trace = hessian[0][0] + hessian[1][1];
    if (trace * trace / det >= 12.1F) return false;

    // 0x26E52C2/0x26E52D9 first adds the sub-pixel offset to the integer
    // coordinate and only then adds the 0.5 pixel-centre shift.  Reassociating
    // these additions is not bit-equivalent for coordinates around 2^11.
    candidate.x = (offset[0] + static_cast<float>(x)) + 0.5F;
    candidate.y = (offset[1] + static_cast<float>(y)) + 0.5F;
    const float scale_exponent =
        (static_cast<float>(level) + offset[2]) / static_cast<float>(kIntervals);
    candidate.scale = static_cast<float>(
        std::pow(2.0, static_cast<double>(scale_exponent)) *
        static_cast<double>(kSigma0));
    const float refined_response = center +
        (gradient[2] * offset[2] +
         (gradient[1] * offset[1] + gradient[0] * offset[0])) * 0.5F;
    candidate.response = std::abs(refined_response) * static_cast<float>(kIntervals);
    candidate.laplacian_sign = refined_response < 0.0F ? -1 : 1;
    candidate.level = level;
    // The recovered refiner performs this check after sub-pixel/scale
    // interpolation.  It is not a scan-loop border based on the discrete
    // level sigma: border = refined_scale * 7 (0x26E4AA0).
    const float border = candidate.scale * 7.0F;
    if (border > candidate.x || border > candidate.y ||
        candidate.x > static_cast<float>(octave.gaussian[0].width) - border ||
        candidate.y > static_cast<float>(octave.gaussian[0].height) - border)
        return false;
    return true;
}

std::vector<Candidate> detect_candidates(const std::vector<Octave>& pyramid,
                                         const std::filesystem::path& path,
                                         bool parallel_cpu_worker) {
    std::vector<Candidate> result;
    const DetectorReplaySchedule* replay = target_detector_replay(path);
    for (std::size_t octave_index = 0; octave_index < pyramid.size(); ++octave_index) {
        const Octave& octave = pyramid[octave_index];
        const int width = static_cast<int>(octave.gaussian[0].width);
        const int height = static_cast<int>(octave.gaussian[0].height);
        // 0x26E5470 evaluates LoG samples in 2x2x2 blocks.  The eight sample
        // order is level, then y, then x.  The max/min index is updated only
        // on a strict comparison, so the first sample wins an exact tie.
        // Neighbours outside that winning sample's source block are then
        // compared strictly.  This is observably different from testing each
        // sample against all 26 neighbours: flat two-pixel LoG plateaus retain
        // the first sample in the block.
        const auto process_block_row = [&](int block_level, int block_y,
                                           std::vector<Candidate>& output) {
            for (int block_x = 2; block_x + 2 < width; block_x += 2) {
                    struct BlockSample {
                        float value;
                        int level;
                        int x;
                        int y;
                    };
                    std::array<BlockSample, 8> samples{};
                    std::size_t sample_index = 0;
                    for (int level_offset = 0; level_offset < 2; ++level_offset) {
                        const int level = block_level + level_offset;
                        const Image& image = octave.log[static_cast<std::size_t>(level)];
                        for (int y_offset = 0; y_offset < 2; ++y_offset) {
                            for (int x_offset = 0; x_offset < 2; ++x_offset) {
                                const int x = block_x + x_offset;
                                const int y = block_y + y_offset;
                                samples[sample_index++] = {
                                    image.at(static_cast<std::size_t>(x),
                                             static_cast<std::size_t>(y)),
                                    level, x, y};
                            }
                        }
                    }

                    std::size_t maximum_index = 0;
                    std::size_t minimum_index = 0;
                    for (std::size_t index = 1; index < samples.size(); ++index) {
                        if (samples[index].value > samples[maximum_index].value)
                            maximum_index = index;
                        if (samples[index].value < samples[minimum_index].value)
                            minimum_index = index;
                    }

                    const auto emit_if_extremum = [&](std::size_t index, bool maximum) {
                        const BlockSample& sample = samples[index];
                        if (sample.level > kIntervals || sample.x + 2 >= width ||
                            sample.y + 2 >= height)
                            return;
                        for (int dl = -1; dl <= 1; ++dl) {
                            const Image& neighbor = octave.log[
                                static_cast<std::size_t>(sample.level + dl)];
                            for (int dy = -1; dy <= 1; ++dy) {
                                for (int dx = -1; dx <= 1; ++dx) {
                                    if (dl == 0 && dx == 0 && dy == 0) continue;
                                    const int neighbor_level = sample.level + dl;
                                    const int neighbor_x = sample.x + dx;
                                    const int neighbor_y = sample.y + dy;
                                    const bool inside_source_block =
                                        neighbor_level >= block_level &&
                                        neighbor_level < block_level + 2 &&
                                        neighbor_x >= block_x && neighbor_x < block_x + 2 &&
                                        neighbor_y >= block_y && neighbor_y < block_y + 2;
                                    if (inside_source_block) continue;
                                    const float other = neighbor.at(
                                        static_cast<std::size_t>(neighbor_x),
                                        static_cast<std::size_t>(neighbor_y));
                                    if (maximum ? sample.value <= other : sample.value >= other)
                                        return;
                                }
                            }
                        }
                        Candidate candidate;
                        if (!refine_extremum(octave, sample.level, sample.x, sample.y,
                                             candidate))
                            return;
                        candidate.octave = static_cast<int>(octave_index);
                        output.push_back(candidate);
                    };

                    // 0x26E5A2A gates the block maximum against the configured
                    // +response threshold and 0x26E620D gates the minimum
                    // against its negation.  The public default captured at
                    // 0x26E4810 is exactly +0.0, so an all-positive block must
                    // not emit its (positive) minimum, and an all-negative
                    // block must not emit its maximum.
                    if (samples[maximum_index].value >= 0.0F)
                        emit_if_extremum(maximum_index, true);
                    if (minimum_index != maximum_index &&
                        samples[minimum_index].value <= -0.0F)
                        emit_if_extremum(minimum_index, false);
            }
        };

        const int block_row_count = std::max(0, (height - 4) / 2);
        const int block_level_count = (kIntervals + 1) / 2;
        const int iteration_count = block_row_count * block_level_count;
        if (replay != nullptr) {
            if (octave_index >= replay->size())
                throw std::runtime_error("target detector schedule has too few octaves");
            const DetectorReplayInvocation& invocation = (*replay)[octave_index];
            if (invocation.iteration_count != static_cast<std::size_t>(iteration_count))
                throw std::runtime_error("target detector iteration count mismatch");
            // sub_26E5470 appends a complete private vector after the barrier.
            // The manifest lists workers in actual critical-entry order, and
            // each worker's dynamic ranges in the order returned by libgomp.
            std::vector<std::vector<Candidate>> worker_candidates(
                invocation.ordered_worker_ranges.size());
#if defined(METALIGN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::ptrdiff_t worker = 0;
                 worker < static_cast<std::ptrdiff_t>(
                     invocation.ordered_worker_ranges.size()); ++worker) {
                const auto& ranges = invocation.ordered_worker_ranges[
                    static_cast<std::size_t>(worker)];
                auto& local = worker_candidates[static_cast<std::size_t>(worker)];
                for (const auto& [begin, end] : ranges) {
                    for (std::size_t iteration = begin; iteration < end; ++iteration) {
                        const int block_level = 1 + 2 * static_cast<int>(
                            iteration / static_cast<std::size_t>(block_row_count));
                        const int block_y = 2 + 2 * static_cast<int>(
                            iteration % static_cast<std::size_t>(block_row_count));
                        process_block_row(block_level, block_y, local);
                    }
                }
            }
            for (auto& local : worker_candidates) {
                result.insert(result.end(), local.begin(), local.end());
            }
            continue;
        }
#if defined(METALIGN_HAS_OPENMP)
        // sub_26E49F0 calls sub_26E48B0 serially for each octave.  That wrapper
        // invokes GOMP_parallel(sub_26E5470, ..., num_threads=0).  The worker
        // uses dynamic chunk 1, a private vector, GOMP_loop_end's barrier and
        // then one critical append of the complete vector.  Enable this exact
        // scheduling shape only for a differential run: the main pipeline
        // currently parallelizes photos with std::async, so enabling it there
        // without --threads 1 would oversubscribe the machine.
        if (parallel_cpu_worker || environment_enabled("METALIGN_TARGET_GOMP_DETECTOR")) {
            std::vector<std::vector<std::size_t>> worker_iterations;
            std::vector<std::size_t> critical_order;
#pragma omp parallel
            {
                const std::size_t worker =
                    static_cast<std::size_t>(omp_get_thread_num());
#pragma omp single
                worker_iterations.resize(
                    static_cast<std::size_t>(omp_get_num_threads()));
                std::vector<Candidate> local;
                // 0x26E5543/0x26E6205 call the original libgomp
                // GOMP_loop_dynamic_start/next ABI.  With current GCC a bare
                // schedule(dynamic, 1) lowers to the distinct
                // GOMP_loop_nonmonotonic_dynamic_* entry points, changing the
                // detector's private-vector membership and therefore the
                // critical-append/HCTree row order.
#pragma omp for schedule(monotonic : dynamic, 1)
                for (int iteration = 0; iteration < iteration_count; ++iteration) {
                    worker_iterations[worker].push_back(
                        static_cast<std::size_t>(iteration));
                    const int block_level = 1 + 2 * (iteration / block_row_count);
                    const int block_y = 2 + 2 * (iteration % block_row_count);
                    process_block_row(block_level, block_y, local);
                }
#pragma omp critical
                {
                    critical_order.push_back(worker);
                    result.insert(result.end(), local.begin(), local.end());
                }
            }
            capture_detector_schedule(
                path, octave_index, static_cast<std::size_t>(iteration_count),
                worker_iterations, critical_order);
            continue;
        }
#endif
        for (int block_level = 1; block_level <= kIntervals; block_level += 2)
            for (int block_y = 2; block_y + 2 < height; block_y += 2)
                process_block_row(block_level, block_y, result);
    }
    return result;
}

std::vector<Candidate> detect_candidates_gpu(
    const std::vector<Octave>& pyramid, DescriptorAccelerator& accelerator) {
    std::vector<Candidate> result;
    for (std::size_t octave = 0; octave < pyramid.size(); ++octave) {
        const auto extrema = accelerator.locate_extrema(
            std::span<const Image>(pyramid[octave].gaussian.data(),
                                   pyramid[octave].gaussian.size()),
            static_cast<int>(octave));
        result.reserve(result.size() + extrema.size());
        for (const GpuExtremum& point : extrema) {
            Candidate candidate;
            candidate.x = point.x;
            candidate.y = point.y;
            candidate.scale = point.scale;
            candidate.response = point.response;
            candidate.octave = static_cast<int>(point.octave);
            candidate.level = static_cast<int>(point.level);
            candidate.laplacian_sign = point.flag != 0 ? 1 : -1;
            result.push_back(candidate);
        }
    }
    return result;
}

template <typename Point, typename Geometry>
std::vector<Point> spatial_select_exact(const std::vector<Point>& points,
                                        std::size_t response_quota,
                                        std::size_t scale_quota,
                                        std::size_t width,
                                        std::size_t height,
                                        Geometry geometry) {
    const std::size_t limit = response_quota + scale_quota;
    if (limit == 0 || points.size() <= limit) return points;

    // sub_26843C0: floor(sqrt(limit / 8)), with a minimum of one.  The
    // following guard is literal from sub_2687810 (normally a no-op after the
    // floor conversion).  Thus 40,000 points use 70x70 cells and the generic
    // 2,048-point stream uses 16x16 cells.
    std::size_t side = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::sqrt(static_cast<double>(limit) * 0.125)));
    while (side * side > limit) side >>= 1;
    const std::size_t cell_count = side * side;

    struct RankedPoint {
        float primary = 0.0F;
        float secondary = 0.0F;
        std::size_t index = 0;
    };
    std::vector<std::vector<RankedPoint>> response_bins(cell_count);
    std::vector<std::vector<RankedPoint>> scale_bins(cell_count);
    for (std::size_t index = 0; index < points.size(); ++index) {
        const auto [x, y, scale, response] = geometry(points[index]);
        // sub_2685190 performs every operation in binary32 and truncates the
        // final quotient toward zero.
        const int cell_x = static_cast<int>(
            (static_cast<float>(x) * static_cast<float>(side)) /
            static_cast<float>(width));
        const int cell_y = static_cast<int>(
            (static_cast<float>(y) * static_cast<float>(side)) /
            static_cast<float>(height));
        if (cell_x < 0 || cell_y < 0 || cell_x >= static_cast<int>(side) ||
            cell_y >= static_cast<int>(side))
            continue;
        const std::size_t cell = static_cast<std::size_t>(cell_y) * side +
                                 static_cast<std::size_t>(cell_x);
        response_bins[cell].push_back({response, scale, index});
        scale_bins[cell].push_back({scale, response, index});
    }
    const auto rank_less = [&](const RankedPoint& left, const RankedPoint& right) {
        if (left.primary != right.primary) return left.primary < right.primary;
        if (left.secondary != right.secondary) return left.secondary < right.secondary;
        // sub_2689260 does not use the input position as its final tie break.
        // After the two 16-byte rank-record floats it dereferences the source
        // 36-byte keypoint and compares +32,+24,+28,+4,+0,+8,+12,+16,+20.
        // Using index here happened to replay a captured target input exactly,
        // but selected different direction rows as soon as GOMP permuted its
        // twenty orientation chunks.
        const Point& left_point = points[left.index];
        const Point& right_point = points[right.index];
        const auto tie_less = [](const auto& a, const auto& b) -> int {
            if (a < b) return -1;
            if (b < a) return 1;
            return 0;
        };
        int order = tie_less(left_point.laplacian_sign > 0 ? 1 : 0,
                             right_point.laplacian_sign > 0 ? 1 : 0);
        if (order != 0) return order < 0;
        order = tie_less(left_point.octave, right_point.octave);
        if (order != 0) return order < 0;
        order = tie_less(left_point.level, right_point.level);
        if (order != 0) return order < 0;
        order = tie_less(left_point.y, right_point.y);
        if (order != 0) return order < 0;
        order = tie_less(left_point.x, right_point.x);
        if (order != 0) return order < 0;
        // Selector record +8 is z and is zero for every photo keypoint.
        order = tie_less(left_point.scale, right_point.scale);
        if (order != 0) return order < 0;
        if constexpr (std::is_same_v<Point, Keypoint>) {
            order = tie_less(left_point.orientation, right_point.orientation);
            if (order != 0) return order < 0;
        }
        order = tie_less(left_point.response, right_point.response);
        if (order != 0) return order < 0;
        return false;
    };
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        std::sort(response_bins[cell].begin(), response_bins[cell].end(), rank_less);
        std::sort(scale_bins[cell].begin(), scale_bins[cell].end(), rank_less);
    }

    // sub_26846F0 water-fills one common per-cell cap, then assigns the
    // residual one-by-one in original cell order.
    std::vector<std::size_t> sorted_counts;
    sorted_counts.reserve(cell_count);
    for (const auto& bin : response_bins) sorted_counts.push_back(bin.size());
    std::sort(sorted_counts.begin(), sorted_counts.end());
    std::size_t cap = 0;
    std::size_t consumed = 0;
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        const std::size_t remaining = cell_count - cell;
        if (consumed + remaining * sorted_counts[cell] > limit) {
            cap = (limit - consumed) / remaining;
            break;
        }
        consumed += sorted_counts[cell];
        cap = sorted_counts[cell];
    }
    std::vector<std::size_t> quotas(cell_count, 0);
    std::size_t assigned = 0;
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        quotas[cell] = std::min(cap, response_bins[cell].size());
        assigned += quotas[cell];
    }
    for (std::size_t cell = 0; assigned < limit && cell < cell_count; ++cell) {
        if (quotas[cell] < response_bins[cell].size()) {
            ++quotas[cell];
            ++assigned;
        }
    }

    std::vector<unsigned char> selected(points.size(), 0);
    const float response_fraction = static_cast<float>(response_quota) /
                                    static_cast<float>(limit);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        const std::size_t response_count = static_cast<std::size_t>(
            static_cast<float>(quotas[cell]) * response_fraction);
        std::size_t remaining_response = response_count;
        for (auto it = response_bins[cell].rbegin(); it != response_bins[cell].rend() &&
             remaining_response != 0; ++it) {
            if (selected[it->index] == 0) {
                selected[it->index] = 1;
                --remaining_response;
            }
        }
        std::size_t remaining_scale = quotas[cell] - response_count;
        for (auto it = scale_bins[cell].rbegin(); it != scale_bins[cell].rend() &&
             remaining_scale != 0; ++it) {
            if (selected[it->index] == 0) {
                selected[it->index] = 1;
                --remaining_scale;
            }
        }
    }

    // sub_26CEE20 copies selected records in their original input order.
    std::vector<Point> result;
    result.reserve(std::min(limit, points.size()));
    for (std::size_t index = 0; index < points.size(); ++index)
        if (selected[index] != 0) result.push_back(points[index]);
    return result;
}

std::vector<float> orientation_peaks(const Image& image, float x, float y, float scale) {
    std::array<float, 36> histogram{};
    const float sigma = 1.5F * scale;
    const int radius = std::min(20, static_cast<int>(3.0F * sigma + 0.5F));
    const int center_x = static_cast<int>(x);
    const int center_y = static_cast<int>(y);
    if (center_x <= radius || center_y <= radius ||
        center_x + radius + 1 >= static_cast<int>(image.width) ||
        center_y + radius + 1 >= static_cast<int>(image.height)) return {};
    std::vector<float> weights(static_cast<std::size_t>(2 * radius + 1));
    const float inverse_weight_variance = 1.0F / (sigma * (sigma + sigma));
    for (int i = -radius; i <= radius; ++i) {
        // 0x26E7A66--0x26E7AAA forms the exponent with float mulss, calls the
        // double exp entry point, then rounds the result back to float.
        const float exponent = static_cast<float>(i * -i) * inverse_weight_variance;
        weights[static_cast<std::size_t>(i + radius)] =
            static_cast<float>(std::exp(static_cast<double>(exponent)));
    }
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) continue;
            const int px = std::clamp(center_x + dx, 1, static_cast<int>(image.width) - 2);
            const int py = std::clamp(center_y + dy, 1, static_cast<int>(image.height) - 2);
            const float gx = image.at(static_cast<std::size_t>(px + 1), static_cast<std::size_t>(py)) -
                             image.at(static_cast<std::size_t>(px - 1), static_cast<std::size_t>(py));
            const float gy = image.at(static_cast<std::size_t>(px), static_cast<std::size_t>(py - 1)) -
                             image.at(static_cast<std::size_t>(px), static_cast<std::size_t>(py + 1));
            const float magnitude = std::sqrt(gx * gx + gy * gy);
            // The target promotes the two float gradients to double for
            // atan2, adds the double pi constant, and only then rounds to
            // float before histogram binning.
            const float angle = static_cast<float>(
                std::atan2(static_cast<double>(gy), static_cast<double>(gx)) +
                static_cast<double>(kPi));
            unsigned bin = static_cast<unsigned>(angle * 5.729578F + 0.5F);
            if (bin >= 36) bin = 0;
            histogram[bin] += weights[static_cast<std::size_t>(dx + radius)] *
                              weights[static_cast<std::size_t>(dy + radius)] * magnitude;
        }
    }
    for (int pass = 0; pass < 2; ++pass) {
        const auto previous = histogram;
        for (std::size_t i = 0; i < histogram.size(); ++i)
            histogram[i] = previous[(i + 35) % 36] * 0.25F + previous[i] * 0.5F +
                           previous[(i + 1) % 36] * 0.25F;
    }
    const float threshold = *std::max_element(histogram.begin(), histogram.end()) * 0.8F;
    std::vector<float> result;
    result.reserve(10);
    for (int bin = 0; bin < 36 && result.size() < 10; ++bin) {
        const float left = histogram[static_cast<std::size_t>((bin + 35) % 36)];
        const float center = histogram[static_cast<std::size_t>(bin)];
        const float right = histogram[static_cast<std::size_t>((bin + 1) % 36)];
        if (center <= left || center <= right || center < threshold) continue;
        float refined = static_cast<float>(bin) +
            0.5F * (left - right) / (left + right - 2.0F * center);
        if (refined < 0.0F) refined += 36.0F;
        if (refined >= 36.0F) refined -= 36.0F;
        result.push_back(refined * 0.17453292F - kPi);
    }
    return result;
}

void compare_cells(const std::vector<std::array<float, 3>>& values,
                   Descriptor& descriptor, int& bit) {
    for (int channel = 0; channel < 3; ++channel) {
        for (std::size_t first = 0; first < values.size(); ++first) {
            for (std::size_t second = first + 1; second < values.size(); ++second, ++bit) {
                if (values[first][static_cast<std::size_t>(channel)] >
                    values[second][static_cast<std::size_t>(channel)])
                    descriptor[static_cast<std::size_t>(bit >> 3)] |=
                        static_cast<std::uint8_t>(1U << (bit & 7));
            }
        }
    }
}

void describe_level(const Image& image, float x, float y, float scale, float orientation,
                    int grid, const std::vector<int>& selected, Descriptor& descriptor, int& bit) {
    constexpr int channels = 3;
    constexpr int radius = 10;
    constexpr float scale_multiplier = 1.1F;
    const int cell_size = (grid + 2 * radius - 1) / grid;
    const int border = static_cast<int>(scale + 0.5F);
    const int maximum_x = static_cast<int>(image.width) - 1 - border;
    const int maximum_y = static_cast<int>(image.height) - 1 - border;
    // 0x26E6810 calls the double-precision sincos entry point with the float
    // angle promoted to double, then rounds both results back to float.  A
    // direct sinf/cosf call changes a few integer sampling coordinates exactly
    // at cell boundaries.
    const float sine = static_cast<float>(std::sin(static_cast<double>(orientation)));
    const float cosine = static_cast<float>(std::cos(static_cast<double>(orientation)));
    const float sampling_scale = scale_multiplier * scale;
    std::vector<std::array<float, channels>> values(selected.size());
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const int cell = selected[index];
        const int row_start = cell_size * (cell / grid) - radius;
        const int column_start = cell_size * (cell % grid) - radius;
        std::array<float, channels> sum{};
        for (int row = row_start; row < row_start + cell_size; ++row) {
            for (int column = column_start; column < column_start + cell_size; ++column) {
                int py = static_cast<int>(((static_cast<float>(column) * sine +
                                            static_cast<float>(row) * cosine) * sampling_scale) + y);
                int px = static_cast<int>(((static_cast<float>(column) * cosine -
                                            static_cast<float>(row) * sine) * sampling_scale) + x);
                py = std::clamp(py, border, maximum_y);
                px = std::clamp(px, border, maximum_x);
                const float center = image.at(static_cast<std::size_t>(px), static_cast<std::size_t>(py));
                const float dx = image.at(static_cast<std::size_t>(px + border), static_cast<std::size_t>(py)) -
                                 image.at(static_cast<std::size_t>(px - border), static_cast<std::size_t>(py));
                const float dy = image.at(static_cast<std::size_t>(px), static_cast<std::size_t>(py + border)) -
                                 image.at(static_cast<std::size_t>(px), static_cast<std::size_t>(py - border));
                sum[0] += center;
                sum[1] += cosine * dx + sine * dy;
                sum[2] += cosine * dy - sine * dx;
            }
        }
        values[index] = sum;
    }
    compare_cells(values, descriptor, bit);
}

Descriptor make_mldb_descriptor(const Image& image, float x, float y, float scale,
                                float orientation) {
    static const std::vector<int> level0{0, 2, 4, 6, 8};
    static const std::vector<int> level1{1, 3, 5, 9, 12, 15, 19, 21, 23};
    static const std::vector<int> level2{0, 2, 4, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 20, 22, 24};
    Descriptor result{};
    int bit = 0;
    describe_level(image, x, y, scale, orientation, 3, level0, result, bit);
    describe_level(image, x, y, scale, orientation, 5, level1, result, bit);
    describe_level(image, x, y, scale, orientation, 5, level2, result, bit);
    return result;
}

bool mask_allows(const Image* mask, double original_x, double original_y) {
    return !mask || sample_bilinear(*mask, original_x, original_y) > 0.5F;
}

Image integer_decimate(const Image& source, int factor) {
    if (factor <= 1) return source;
    Image result;
    result.width = (source.width + static_cast<std::size_t>(factor) - 1) /
                   static_cast<std::size_t>(factor);
    result.height = (source.height + static_cast<std::size_t>(factor) - 1) /
                    static_cast<std::size_t>(factor);
    result.gray.resize(result.width * result.height);
    for (std::size_t y = 0; y < result.height; ++y)
        for (std::size_t x = 0; x < result.width; ++x)
            result.at(x, y) = source.at(x * static_cast<std::size_t>(factor),
                                        y * static_cast<std::size_t>(factor));
    return result;
}

Image target_device_grayscale(const Image& source) {
    if (source.rgb.size() != source.width * source.height * 3U) return source;
    Image result = source;
    result.gray.resize(source.width * source.height);
    // CUDA module 126, convert_rgb2gray<uchar,float>: G is multiplied first,
    // then R and B are accumulated by two round-to-nearest FMAs.  The result
    // is truncated to uint8 and only then divided by 255.  This differs from
    // the target CPU conversion's double-precision BT.601 expression.
    constexpr float red = std::bit_cast<float>(std::uint32_t{0x3e991687U});
    constexpr float green = std::bit_cast<float>(std::uint32_t{0x3f1645a2U});
    constexpr float blue = std::bit_cast<float>(std::uint32_t{0x3de978d5U});
    for (std::size_t index = 0; index < result.gray.size(); ++index) {
        const float r = static_cast<float>(source.rgb[index * 3U]);
        const float g = static_cast<float>(source.rgb[index * 3U + 1U]);
        const float b = static_cast<float>(source.rgb[index * 3U + 2U]);
        const float luminance = std::fma(b, blue, std::fma(r, red, g * green));
        const auto code = static_cast<std::uint8_t>(
            static_cast<std::uint32_t>(luminance) & 0xffU);
        result.gray[index] = static_cast<float>(code) / 255.0F;
    }
    return result;
}

Image target_device_gaussian_blur(const Image& source, double sigma) {
    if (source.empty() || sigma <= 0.0) return source;
    const int radius = std::max(1, static_cast<int>(4.0 * sigma + 1.0));
    std::vector<float> half_kernel(static_cast<std::size_t>(radius + 1));
    const float sigma_squared = static_cast<float>(sigma * sigma);
    const double exponent = -0.5 / static_cast<double>(sigma_squared);
    double sum = -1.0;
    for (int index = 0; index <= radius; ++index) {
        const float value = static_cast<float>(std::exp(exponent * index * index));
        half_kernel[static_cast<std::size_t>(index)] = value;
        sum += static_cast<double>(value + value);
    }
    for (float& value : half_kernel)
        value = static_cast<float>(static_cast<double>(value) / sum);

    // CUDA module 128's linearRow/ColumnFilter<N,float> specializations
    // expand the symmetric kernel and issue one round-to-nearest FMA per tap
    // from the negative edge to the positive edge.  The CPU path instead
    // forms (left+right)*weight; that algebraic shortcut is not bit-equivalent.
    Image result = source;
    result.gray.assign(source.width * source.height, 0.0F);
    std::vector<float> temporary(source.width * source.height, 0.0F);
    for (std::size_t y = 0; y < source.height; ++y) {
        for (std::size_t x = 0; x < source.width; ++x) {
            float value = 0.0F;
            for (int tap = -radius; tap <= radius; ++tap) {
                const std::size_t sample = static_cast<std::size_t>(std::clamp(
                    static_cast<std::ptrdiff_t>(x) + tap,
                    std::ptrdiff_t{0},
                    static_cast<std::ptrdiff_t>(source.width - 1)));
                value = std::fma(source.at(sample, y),
                                 half_kernel[static_cast<std::size_t>(std::abs(tap))],
                                 value);
            }
            temporary[y * source.width + x] = value;
        }
    }
    for (std::size_t y = 0; y < source.height; ++y) {
        for (std::size_t x = 0; x < source.width; ++x) {
            float value = 0.0F;
            for (int tap = -radius; tap <= radius; ++tap) {
                const std::size_t sample = static_cast<std::size_t>(std::clamp(
                    static_cast<std::ptrdiff_t>(y) + tap,
                    std::ptrdiff_t{0},
                    static_cast<std::ptrdiff_t>(source.height - 1)));
                value = std::fma(temporary[sample * source.width + x],
                                 half_kernel[static_cast<std::size_t>(std::abs(tap))],
                                 value);
            }
            result.at(x, y) = value;
        }
    }
    return result;
}

std::vector<Octave> build_pyramid(const Image& input, int downscale,
                                  DescriptorAccelerator* accelerator,
                                  bool parallel_cpu_worker = false) {
    std::vector<Octave> result;
    result.reserve(kOctaves);
    // 1/2/4/8 use the recovered integer stride sampler. Zero selects the
    // Highest (2W-1)x(2H-1) interpolation lattice.
    Image base = downscale == 0
        ? upsample_highest(input, parallel_cpu_worker)
        : integer_decimate(input, downscale);
    const bool device_arithmetic =
        accelerator && accelerator->supports_extrema_detection();
    const auto blur = [device_arithmetic, accelerator, parallel_cpu_worker](
                          const Image& image, double sigma) {
        if (!device_arithmetic)
            return gaussian_blur(image, sigma, parallel_cpu_worker);
        if (accelerator->supports_device_gaussian())
            return accelerator->gaussian_blur(image, sigma);
        return target_device_gaussian_blur(image, sigma);
    };
    const float inherited_sigma = downscale == 0 ? 1.0F : 0.5F;
    const double sigma0 = static_cast<double>(kSigma0);
    base = blur(base, std::sqrt(
        sigma0 * sigma0 -
        static_cast<double>(inherited_sigma * inherited_sigma)));
    // The target stores an absolute-sigma table built by repeated float
    // multiplication (config+0x30), rather than evaluating powf separately
    // for each level.  Runtime bits for the first five entries are
    // 3fcccccd,40010413,40228cc4,404ccccd,40810413.
    std::array<float, kGaussianLevels> level_sigma{};
    level_sigma[0] = kSigma0;
    for (int level = 1; level < kGaussianLevels; ++level)
        level_sigma[static_cast<std::size_t>(level)] =
            level_sigma[static_cast<std::size_t>(level - 1)] * kScaleStep;
    for (int octave_index = 0; octave_index < kOctaves && base.width >= 32 && base.height >= 32;
         ++octave_index) {
        Octave octave;
        octave.gaussian[0] = base;
        for (int level = 1; level < kGaussianLevels; ++level) {
            const float previous_sigma = level_sigma[static_cast<std::size_t>(level - 1)];
            const float sigma = level_sigma[static_cast<std::size_t>(level)];
            octave.gaussian[static_cast<std::size_t>(level)] = blur(
                octave.gaussian[static_cast<std::size_t>(level - 1)],
                std::sqrt(sigma * sigma - previous_sigma * previous_sigma));
        }
        for (int level = 0; level < kGaussianLevels; ++level) {
            if (accelerator && accelerator->supports_extrema_detection()) break;
            const float sigma = level_sigma[static_cast<std::size_t>(level)];
            octave.log[static_cast<std::size_t>(level)] = accelerator
                ? accelerator->laplacian_response(
                      octave.gaussian[static_cast<std::size_t>(level)], sigma)
                : laplacian_response(
                      octave.gaussian[static_cast<std::size_t>(level)], sigma,
                      parallel_cpu_worker);
        }
        Image next = integer_decimate(octave.gaussian[3], 2);
        result.push_back(std::move(octave));
        base = std::move(next);
    }
    return result;
}

}  // namespace

FeatureSet FeatureExtractor::extract(const std::filesystem::path& path,
                                     const std::filesystem::path* mask_path) const {
    const Image original = load_image(path, parallel_cpu_worker_);
    Image mask;
    const Image* mask_pointer = nullptr;
    if (mask_path) {
        mask = load_image(*mask_path);
        if (mask.width != original.width || mask.height != original.height)
            mask = resize_bilinear(mask, original.width, original.height);
        mask_pointer = &mask;
    }
    return extract(original, path, mask_pointer);
}

FeatureSet FeatureExtractor::extract(const Image& original,
                                     const std::filesystem::path& path,
                                     const Image* mask_pointer) const {
    if (original.width < 32 || original.height < 32)
        throw std::runtime_error("image is too small for alignment: " + path.string());

    const int downscale = options_.downscale;
    const float source_scale = alignment_accuracy_source_scale(downscale);
    DescriptorAccelerator* feature_accelerator =
        accelerator_ && accelerator_->supports_feature_extraction()
            ? accelerator_ : nullptr;
    // A Gaussian dump intentionally selects the legacy host-visible path: its
    // purpose is to materialize every plane for differential analysis. Normal
    // production keeps all 30 planes resident until orientation and MLDB are
    // complete and transfers only compact outputs.
    const bool dump_resident_gaussian =
        environment_enabled("METALIGN_DUMP_RESIDENT_GAUSSIAN");
    const bool resident_features = feature_accelerator &&
        feature_accelerator->supports_resident_feature_pipeline() &&
        (std::getenv("METALIGN_DUMP_GAUSSIAN_PYRAMID") == nullptr ||
         dump_resident_gaussian) &&
        !environment_enabled("METALIGN_DISABLE_RESIDENT_FEATURES");
    struct ResidentFeatureGuard {
        DescriptorAccelerator* accelerator = nullptr;
        ~ResidentFeatureGuard() {
            if (accelerator) accelerator->end_resident_feature_image();
        }
    } resident_guard;
    std::vector<Octave> pyramid;
    std::vector<Candidate> candidates;
    if (resident_features) {
        const auto resident_octaves =
            feature_accelerator->begin_resident_feature_image(original, downscale);
        resident_guard.accelerator = feature_accelerator;
        pyramid.resize(resident_octaves.size());
        for (std::size_t octave_index = 0; octave_index < resident_octaves.size();
             ++octave_index) {
            const ResidentFeatureOctave& resident = resident_octaves[octave_index];
            for (Image& level : pyramid[octave_index].gaussian) {
                level.width = resident.width;
                level.height = resident.height;
            }
            if (dump_resident_gaussian) {
                for (int level = 0; level < kGaussianLevels; ++level)
                    pyramid[octave_index].gaussian[static_cast<std::size_t>(level)] =
                        feature_accelerator->resident_feature_level(
                            static_cast<int>(octave_index), level);
            }
            candidates.reserve(candidates.size() + resident.extrema.size());
            for (const GpuExtremum& point : resident.extrema) {
                Candidate candidate;
                candidate.x = point.x;
                candidate.y = point.y;
                candidate.scale = point.scale;
                candidate.response = point.response;
                candidate.octave = static_cast<int>(point.octave);
                candidate.level = static_cast<int>(point.level);
                candidate.laplacian_sign = point.flag != 0 ? 1 : -1;
                candidates.push_back(candidate);
            }
        }
        if (dump_resident_gaussian) dump_gaussian_pyramid(path, pyramid);
    } else {
        Image input = feature_accelerator
            ? (feature_accelerator->supports_device_grayscale()
                   ? feature_accelerator->grayscale(original)
                   : target_device_grayscale(original))
            : original;
        input.rgb.clear();
        pyramid = build_pyramid(
            input, downscale, feature_accelerator, parallel_cpu_worker_);
        dump_gaussian_pyramid(path, pyramid);
        candidates = feature_accelerator &&
                feature_accelerator->supports_extrema_detection()
            ? detect_candidates_gpu(pyramid, *feature_accelerator)
            : detect_candidates(pyramid, path, parallel_cpu_worker_);
    }
    dump_raw_selector_candidates(path, candidates, "raw_input36");

    const double megapixels = static_cast<double>(original.width * original.height) / 1'000'000.0;
    std::size_t limit = options_.keypoint_limit == 0
        ? std::numeric_limits<std::size_t>::max() : options_.keypoint_limit;
    static_cast<void>(megapixels);
    const auto candidate_geometry = [source_scale](const Candidate& candidate) {
        const float factor = source_scale *
                             static_cast<float>(1 << candidate.octave);
        const float x = ((candidate.x - 0.5F) * factor) + 0.5F;
        const float y = ((candidate.y - 0.5F) * factor) + 0.5F;
        return std::array<float, 4>{x, y, candidate.scale * factor,
                                    candidate.response};
    };
    // The detection worker calls sub_2687810 twice on the raw extrema:
    // (0, 2048, false) for generic preselection and (keypoint_limit, 0,
    // false) for the full matching stream.  The two sets are independently
    // selected; the coarse set is not a prefix/subset assumption.
    std::vector<Candidate> coarse_candidates = spatial_select_exact(
        candidates, 0, 2048, original.width, original.height, candidate_geometry);
    // A Phase27 trace can optionally provide the target selector's *ordered*
    // 2,048-row coarse output.  The ordinary selector remains untouched;
    // this branch exists solely to replay one captured GOMP/HCTree boundary.
    replay_target_candidate_order(path, coarse_candidates,
                                  "METALIGN_REPLAY_TARGET_COARSE_CANDIDATES",
                                  ".coarse_selected36.bin", "coarse");
    candidates = spatial_select_exact(
        candidates, limit, 0, original.width, original.height, candidate_geometry);
    // The full stream has an independently selected, per-photo count (not a
    // universal 40k row count).  The replay requires exact count and bitwise
    // candidate identities before it permits an observed-order substitution.
    replay_target_candidate_order(path, candidates,
                                  "METALIGN_REPLAY_TARGET_FULL_CANDIDATES",
                                  ".full_selected36.bin", "full");
    dump_raw_selector_candidates(path, coarse_candidates, "coarse_selected36");
    dump_raw_selector_candidates(path, candidates, "full_selected36");

    FeatureSet result;
    result.path = path;
    result.image_width = original.width;
    result.image_height = original.height;
    if (original.focal_length_35mm) {
        // Runtime calibration capture gives 3688.6320165945194 px for the
        // 4928x3264, 27-mm-equivalent Nikon sample.  This is exactly the ratio
        // of image diagonal to the 36x24-mm still-frame diagonal.
        result.focal_length_pixels = *original.focal_length_35mm *
            std::hypot(static_cast<double>(original.width),
                       static_cast<double>(original.height)) /
            std::hypot(36.0, 24.0);
    }
    // filter_mask participates before direction expansion and the second
    // selector. mask_tiepoints is a later gate: it removes already-selected
    // oriented rows without refilling their spatial quota. Keeping the two
    // stages distinct follows the two independent task bytes at +48/+49.
    const Image* detector_mask = options_.filter_mask ? mask_pointer : nullptr;
    const auto build_descriptor_rows = [&](const std::vector<Candidate>& selected_candidates,
                                           const char* stream_name) {
        struct PendingRow {
            Keypoint keypoint;
            const Image* level_image = nullptr;
            int octave = 0;
            int level = 0;
            float local_x = 0.0F;
            float local_y = 0.0F;
            float local_scale = 1.0F;
        };
        std::vector<std::vector<float>> accelerated_orientations;
        if (feature_accelerator) {
            accelerated_orientations.resize(selected_candidates.size());
            for (int octave_index = 0; octave_index < static_cast<int>(pyramid.size());
                 ++octave_index) {
                for (int level = 0; level < kGaussianLevels; ++level) {
                    std::vector<std::size_t> indices;
                    std::vector<FeaturePrimitive> points;
                    for (std::size_t index = 0; index < selected_candidates.size(); ++index) {
                        const Candidate& candidate = selected_candidates[index];
                        if (candidate.octave != octave_index || candidate.level != level)
                            continue;
                        indices.push_back(index);
                        points.push_back({candidate.x, candidate.y, candidate.scale, 0.0F});
                    }
                    if (points.empty()) continue;
                    auto orientations = resident_features
                        ? feature_accelerator->resident_orientation_peaks(
                              octave_index, level, points)
                        : feature_accelerator->orientation_peaks(
                              pyramid[static_cast<std::size_t>(octave_index)]
                                  .gaussian[static_cast<std::size_t>(level)], points);
                    if (orientations.size() != points.size())
                        throw std::runtime_error("GPU orientation output count mismatch");
                    for (std::size_t index = 0; index < indices.size(); ++index)
                        accelerated_orientations[indices[index]] =
                            std::move(orientations[index]);
                }
            }
        }
        const auto expand_range = [&](std::size_t begin, std::size_t end,
                                      std::vector<PendingRow>& output) {
          for (std::size_t candidate_index = begin; candidate_index < end;
               ++candidate_index) {
            const Candidate& candidate = selected_candidates[candidate_index];
            const Octave& octave = pyramid[static_cast<std::size_t>(candidate.octave)];
            const Image& level_image =
                octave.gaussian[static_cast<std::size_t>(candidate.level)];
            const float coordinate_factor =
                static_cast<float>(1 << candidate.octave) * source_scale;
            const float original_x = (candidate.x - 0.5F) * coordinate_factor + 0.5F;
            const float original_y = (candidate.y - 0.5F) * coordinate_factor + 0.5F;
            if (!mask_allows(detector_mask, original_x, original_y)) continue;
            const std::vector<float> orientations = feature_accelerator
                ? accelerated_orientations[candidate_index]
                : orientation_peaks(
                      level_image, candidate.x, candidate.y, candidate.scale);
            for (float orientation : orientations) {
                PendingRow pending;
                Keypoint& keypoint = pending.keypoint;
                keypoint.x = original_x;
                keypoint.y = original_y;
                keypoint.scale = candidate.scale * coordinate_factor;
                keypoint.orientation = orientation;
                keypoint.response = candidate.response;
                keypoint.octave = candidate.octave;
                keypoint.level = candidate.level;
                keypoint.detector_id = candidate_index;
                keypoint.source_id = candidate_index;
                keypoint.laplacian_sign = candidate.laplacian_sign;
                pending.level_image = &level_image;
                pending.octave = candidate.octave;
                pending.level = candidate.level;
                pending.local_x = candidate.x;
                pending.local_y = candidate.y;
                pending.local_scale = candidate.scale;
                output.push_back(std::move(pending));
            }
          }
        };

        std::vector<PendingRow> pending_rows;
#if defined(METALIGN_HAS_OPENMP)
        const std::vector<std::size_t>* replay_append_order =
            target_orientation_replay_order(path, selected_candidates.size());
        const bool reproduce_target_gomp =
            parallel_cpu_worker_ ||
            environment_enabled("METALIGN_TARGET_GOMP_ORIENTATION") ||
            environment_enabled("METALIGN_TARGET_GOMP_DETECTOR") ||
            replay_append_order != nullptr;
        if (reproduce_target_gomp) {
            // sub_26E7860 invokes GOMP_parallel(sub_26E7EC0, ..., 0, 0).
            // The worker manually assigns one contiguous quotient/remainder
            // slice to each thread, builds a private direction-expanded
            // vector, reaches GOMP_barrier, then appends that complete vector
            // under GOMP_critical.  Normal operation deliberately leaves that
            // order to libgomp instead of hard-coding one captured process.
            if (replay_append_order != nullptr) {
                // A target Phase27 capture provides this direction region's
                // completed critical-entry permutation. Recreate the workers'
                // private vectors first, then concatenate them in that
                // captured order. This is a diagnostic replay, not a
                // replacement scheduler for ordinary alignment.
                std::vector<std::vector<PendingRow>> private_rows(
                    replay_append_order->size());
                bool team_matches_schedule = true;
#pragma omp parallel shared(team_matches_schedule, private_rows)
                {
                    const std::size_t thread_count =
                        static_cast<std::size_t>(omp_get_num_threads());
                    const std::size_t thread_index =
                        static_cast<std::size_t>(omp_get_thread_num());
                    if (thread_count != replay_append_order->size()) {
#pragma omp critical
                        team_matches_schedule = false;
                    } else {
                        const std::size_t quotient = selected_candidates.size() / thread_count;
                        const std::size_t remainder = selected_candidates.size() % thread_count;
                        const std::size_t count =
                            quotient + (thread_index < remainder ? 1U : 0U);
                        const std::size_t begin = thread_index < remainder
                            ? thread_index * (quotient + 1U)
                            : remainder + thread_index * quotient;
                        expand_range(begin, begin + count, private_rows[thread_index]);
                    }
#pragma omp barrier
                }
                if (!team_matches_schedule)
                    throw std::runtime_error(
                        "target orientation schedule team size does not match OpenMP team");
                std::vector<unsigned char> seen(private_rows.size(), 0);
                for (const std::size_t thread_index : *replay_append_order) {
                    if (thread_index >= private_rows.size() || seen[thread_index] != 0)
                        throw std::runtime_error(
                            "invalid target orientation append permutation");
                    seen[thread_index] = 1;
                    std::vector<PendingRow>& local = private_rows[thread_index];
                    pending_rows.insert(pending_rows.end(),
                                        std::make_move_iterator(local.begin()),
                                        std::make_move_iterator(local.end()));
                }
                if (std::find(seen.begin(), seen.end(), 0) != seen.end())
                    throw std::runtime_error(
                        "incomplete target orientation append permutation");
            } else {
                std::vector<std::size_t> physical_append_order;
#pragma omp parallel
                {
                    const std::size_t thread_count =
                        static_cast<std::size_t>(omp_get_num_threads());
                    const std::size_t thread_index =
                        static_cast<std::size_t>(omp_get_thread_num());
                    const std::size_t quotient = selected_candidates.size() / thread_count;
                    const std::size_t remainder = selected_candidates.size() % thread_count;
                    const std::size_t count = quotient + (thread_index < remainder ? 1U : 0U);
                    const std::size_t begin = thread_index < remainder
                        ? thread_index * (quotient + 1U)
                        : remainder + thread_index * quotient;
                    std::vector<PendingRow> local;
                    expand_range(begin, begin + count, local);
#pragma omp barrier
#pragma omp critical
                    {
                        physical_append_order.push_back(thread_index);
                        pending_rows.insert(pending_rows.end(),
                                            std::make_move_iterator(local.begin()),
                                            std::make_move_iterator(local.end()));
                    }
                }
                capture_orientation_schedule(path, stream_name,
                                             physical_append_order);
            }
        } else
#endif
        {
            expand_range(0, selected_candidates.size(), pending_rows);
        }

        // sub_26E6EA0 is a second worker: it statically partitions the final
        // orientation-row array and writes descriptors to their global slots.
        // Keeping this work out of the orientation worker is important both
        // for the target's critical-append order and for throughput.
#if defined(METALIGN_HAS_OPENMP)
        if (!feature_accelerator && reproduce_target_gomp) {
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t row = 0;
                 row < static_cast<std::ptrdiff_t>(pending_rows.size()); ++row) {
                PendingRow& pending = pending_rows[static_cast<std::size_t>(row)];
                pending.keypoint.descriptor = make_mldb_descriptor(
                    *pending.level_image, pending.local_x, pending.local_y,
                    pending.local_scale,
                    -static_cast<float>(pending.keypoint.orientation));
            }
        } else
#endif
        {
            if (feature_accelerator) {
                std::map<std::pair<int, int>, std::vector<std::size_t>> groups;
                for (std::size_t row = 0; row < pending_rows.size(); ++row) {
                    const PendingRow& pending = pending_rows[row];
                    groups[{pending.octave, pending.level}].push_back(row);
                }
                for (const auto& [layer, rows] : groups) {
                    std::vector<FeaturePrimitive> points;
                    points.reserve(rows.size());
                    for (std::size_t row : rows) {
                        const PendingRow& pending = pending_rows[row];
                        points.push_back({pending.local_x, pending.local_y,
                                          pending.local_scale,
                                          -static_cast<float>(
                                              pending.keypoint.orientation)});
                    }
                    const auto [octave_index, level] = layer;
                    auto descriptors = resident_features
                        ? feature_accelerator->resident_mldb_descriptors(
                              octave_index, level, points)
                        : feature_accelerator->mldb_descriptors(
                              pyramid[static_cast<std::size_t>(octave_index)]
                                  .gaussian[static_cast<std::size_t>(level)], points);
                    if (descriptors.size() != rows.size())
                        throw std::runtime_error("GPU MLDB output count mismatch");
                    for (std::size_t index = 0; index < rows.size(); ++index)
                        pending_rows[rows[index]].keypoint.descriptor =
                            descriptors[index];
                }
            } else {
                for (PendingRow& pending : pending_rows) {
                    pending.keypoint.descriptor = make_mldb_descriptor(
                        *pending.level_image, pending.local_x, pending.local_y,
                        pending.local_scale,
                        -static_cast<float>(pending.keypoint.orientation));
                }
            }
        }
        std::vector<Keypoint> rows;
        rows.reserve(pending_rows.size());
        for (PendingRow& pending : pending_rows)
            rows.push_back(std::move(pending.keypoint));
        return rows;
    };
    result.keypoints = build_descriptor_rows(candidates, "full");
    result.coarse_keypoints = build_descriptor_rows(coarse_candidates, "coarse");
    dump_phase24_keypoints(path, result.keypoints, "full_pre_selector", true);
    dump_phase24_keypoints(path, result.coarse_keypoints, "coarse_pre_selector", true);

    // After orientation expansion, the target applies both selectors once
    // more. Runtime captures show full inputs around 50,000 rows reduced by
    // (keypoint_limit, 0, false) to exactly 40,000, and coarse inputs around
    // 2,804/2,821 reduced by (0, 2048, false) to exactly 2,048. Omitting the
    // full second pass was the reason South Building exceeded 40,000 rows per
    // image and retained hundreds of thousands of extra full matches.
    const auto keypoint_geometry = [](const Keypoint& keypoint) {
        return std::array<float, 4>{static_cast<float>(keypoint.x),
                                    static_cast<float>(keypoint.y),
                                    static_cast<float>(keypoint.scale),
                                    static_cast<float>(keypoint.response)};
    };
    if (limit != std::numeric_limits<std::size_t>::max()) {
        result.keypoints = spatial_select_exact(
            result.keypoints, limit, 0, original.width, original.height,
            keypoint_geometry);
    }
    dump_phase24_keypoints(path, result.keypoints, "full_post_selector", true);
    result.coarse_keypoints = spatial_select_exact(
        result.coarse_keypoints, 0, 2048, original.width, original.height,
        keypoint_geometry);
    if (mask_pointer && options_.mask_tiepoints) {
        const auto remove_masked = [&](std::vector<Keypoint>& rows) {
            rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const Keypoint& row) {
                return !mask_allows(mask_pointer, row.x, row.y);
            }), rows.end());
        };
        remove_masked(result.keypoints);
        remove_masked(result.coarse_keypoints);
    }
    dump_phase24_keypoints(path, result.coarse_keypoints, "coarse_post_selector", true);

    const auto finalize_rows = [](std::vector<Keypoint>& rows,
                                  std::size_t detector_count) {
        std::stable_partition(rows.begin(), rows.end(), [](const Keypoint& keypoint) {
            return keypoint.laplacian_sign < 0;
        });
        std::vector<std::size_t> first_row(detector_count, rows.size());
        for (std::size_t row = 0; row < rows.size(); ++row) {
            Keypoint& keypoint = rows[row];
            std::size_t& canonical = first_row[keypoint.detector_id];
            if (canonical == rows.size()) canonical = row;
            keypoint.source_id = canonical;
        }
    };
    finalize_rows(result.keypoints, candidates.size());
    finalize_rows(result.coarse_keypoints, coarse_candidates.size());
    dump_phase24_keypoints(path, result.keypoints, "full_hctree_order", true);
    dump_phase24_keypoints(path, result.coarse_keypoints, "coarse_hctree_order", true);
    result.source_keypoint_count = result.keypoints.size();

    // Retain the compact bit-density signature only as a diagnostic/ABI field.
    // The recovered production generic-preselection path no longer consumes it;
    // both coarse and full matching use the 64-byte MLDB descriptor stream.
    for (const Keypoint& keypoint : result.keypoints)
        for (std::size_t i = 0; i < result.global_descriptor.size(); ++i)
            result.global_descriptor[i] += static_cast<float>(std::popcount(keypoint.descriptor[i]));
    double length = 0.0;
    for (float value : result.global_descriptor) length += value * value;
    length = std::sqrt(length) + 1e-12;
    for (float& value : result.global_descriptor) value = static_cast<float>(value / length);
    return result;
}

std::uint32_t descriptor_hamming_distance(const Descriptor& left, const Descriptor& right) {
    std::uint32_t result = 0;
    for (std::size_t offset = 0; offset < kDescriptorSize; offset += sizeof(std::uint64_t)) {
        std::uint64_t a = 0;
        std::uint64_t b = 0;
        std::memcpy(&a, left.data() + offset, sizeof(a));
        std::memcpy(&b, right.data() + offset, sizeof(b));
        // 0x26F3AE0--0x26F3B4C uses this 64-bit SWAR popcount in the HCTree
        // hot loop.  The standalone fixed-size loop lets GCC combine several
        // lanes with SSE2 while retaining the target's exact integer result.
        std::uint64_t difference = a ^ b;
        difference -= (difference >> 1U) & 0x5555555555555555ULL;
        difference = (difference & 0x3333333333333333ULL) +
                     ((difference >> 2U) & 0x3333333333333333ULL);
        difference = (difference + (difference >> 4U)) & 0x0F0F0F0F0F0F0F0FULL;
        result += static_cast<std::uint32_t>(
            (difference * 0x0101010101010101ULL) >> 56U);
    }
    return result;
}

Descriptor compute_mldb_descriptor(const Image& image, float x, float y,
                                   float scale, float orientation) {
    return make_mldb_descriptor(image, x, y, scale, orientation);
}

std::vector<float> compute_orientation_peaks(const Image& image, float x, float y,
                                             float scale) {
    return orientation_peaks(image, x, y, scale);
}

double descriptor_cosine(const GlobalDescriptor& left, const GlobalDescriptor& right) {
    double result = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) result += left[i] * right[i];
    return result;
}

}  // namespace metalign
