#pragma once

#include "metalign/features.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace metalign {

class DescriptorAccelerator;
struct FeatureMatch;
struct ImagePair;
struct PairMatches;

class CpuDescriptorIndex {
public:
    explicit CpuDescriptorIndex(const FeatureSet& features);
    ~CpuDescriptorIndex();
    CpuDescriptorIndex(CpuDescriptorIndex&&) noexcept;
    CpuDescriptorIndex& operator=(CpuDescriptorIndex&&) noexcept;
    CpuDescriptorIndex(const CpuDescriptorIndex&) = delete;
    CpuDescriptorIndex& operator=(const CpuDescriptorIndex&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend std::vector<FeatureMatch> match_feature_sets(
        const FeatureSet&, const FeatureSet&, const MatchPhotosOptions&,
        DescriptorAccelerator*, const CpuDescriptorIndex*, const CpuDescriptorIndex*,
        std::size_t*);
    friend std::vector<PairMatches> match_feature_pairs_focal_batches(
        const std::vector<FeatureSet>&, const std::vector<ImagePair>&,
        const MatchPhotosOptions&);
};

struct ImagePair {
    std::size_t first = 0;
    std::size_t second = 0;
    ImagePair ordered() const { return first < second ? *this : ImagePair{second, first}; }
    auto operator<=>(const ImagePair&) const = default;
};

struct FeatureMatch {
    std::size_t first = 0;
    std::size_t second = 0;
    double distance = 0.0;
};

struct PairMatches {
    ImagePair pair;
    std::vector<FeatureMatch> matches;
    std::size_t directional_count = 0;
};

struct ReferencePosition {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

std::map<std::string, ReferencePosition> read_reference_csv(
    const std::filesystem::path& path);
std::set<ImagePair> select_image_pairs(
    const std::vector<FeatureSet>& features,
    const ProgramOptions& options,
    const std::map<std::string, ReferencePosition>& reference = {},
    std::size_t thread_count = 1,
    DescriptorAccelerator* accelerator = nullptr);
std::set<ImagePair> select_generic_image_pairs_from_coarse(
    const std::vector<FeatureSet>& coarse_features,
    const std::vector<ImagePair>& candidates,
    const MatchPhotosOptions& options,
    std::size_t thread_count = 1,
    DescriptorAccelerator* accelerator = nullptr,
    const std::vector<const CpuDescriptorIndex*>* indices = nullptr);
std::set<ImagePair> select_generic_image_pairs_from_coarse(
    const std::vector<FeatureSet>& coarse_features,
    const MatchPhotosOptions& options,
    std::size_t thread_count = 1,
    DescriptorAccelerator* accelerator = nullptr,
    const std::vector<const CpuDescriptorIndex*>* indices = nullptr);
std::vector<FeatureMatch> match_feature_sets(
    const FeatureSet& first,
    const FeatureSet& second,
    const MatchPhotosOptions& options,
    DescriptorAccelerator* accelerator = nullptr,
    const CpuDescriptorIndex* first_index = nullptr,
    const CpuDescriptorIndex* second_index = nullptr,
    std::size_t* raw_directional_count = nullptr);
std::vector<PairMatches> match_feature_pairs_focal_batches(
    const std::vector<FeatureSet>& features,
    const std::vector<ImagePair>& pairs,
    const MatchPhotosOptions& options);
std::vector<PairMatches> match_feature_pairs_accelerated_batches(
    const std::vector<FeatureSet>& features,
    const std::vector<ImagePair>& pairs,
    const MatchPhotosOptions& options,
    DescriptorAccelerator& accelerator,
    const std::function<bool()>& should_cancel = {},
    const std::function<void(std::size_t, std::size_t)>& progress_callback = {});
std::vector<PairMatches> match_feature_pairs_accelerated_batches(
    std::span<const FeatureSet* const> features,
    const std::vector<ImagePair>& pairs,
    const MatchPhotosOptions& options,
    DescriptorAccelerator& accelerator,
    const std::function<bool()>& should_cancel = {},
    const std::function<void(std::size_t, std::size_t)>& progress_callback = {});
std::vector<std::size_t> local_consistency_inliers(
    const FeatureSet& first,
    const FeatureSet& second,
    const std::vector<FeatureMatch>& matches,
    const ImagePair* pair = nullptr);

}  // namespace metalign
