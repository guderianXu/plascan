#include "ReferenceTrackSpatialSelector.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace xjw::detail
{
    namespace
    {

        struct SpatialCandidate
        {
            std::uint32_t track = 0;
            float weight = 0.0f;
        };

        float usableScale(const FeatureKeypoint& keypoint)
        {
            return std::isfinite(keypoint.scale) && keypoint.scale > 0.0f ? keypoint.scale : 1.0f;
        }

        bool spatialLess(const SpatialCandidate& left, const SpatialCandidate& right)
        {
            return left.weight < right.weight;
        }

        void spatialMoveMedianToFirst(SpatialCandidate* result,
                                      SpatialCandidate* first,
                                      SpatialCandidate* middle,
                                      SpatialCandidate* last)
        {
            if (spatialLess(*first, *middle))
            {
                if (spatialLess(*middle, *last))
                {
                    std::iter_swap(result, middle);
                }
                else if (spatialLess(*first, *last))
                {
                    std::iter_swap(result, last);
                }
                else
                {
                    std::iter_swap(result, first);
                }
            }
            else if (spatialLess(*first, *last))
            {
                std::iter_swap(result, first);
            }
            else if (spatialLess(*middle, *last))
            {
                std::iter_swap(result, last);
            }
            else
            {
                std::iter_swap(result, middle);
            }
        }

        SpatialCandidate*
        spatialUnguardedPartition(SpatialCandidate* first, SpatialCandidate* last, SpatialCandidate* pivot)
        {
            while (true)
            {
                while (spatialLess(*first, *pivot))
                {
                    ++first;
                }
                --last;
                while (spatialLess(*pivot, *last))
                {
                    --last;
                }
                if (!(first < last))
                {
                    return first;
                }
                std::iter_swap(first, last);
                ++first;
            }
        }

        void spatialIntrosortLoop(SpatialCandidate* first, SpatialCandidate* last, std::size_t depthLimit)
        {
            constexpr std::ptrdiff_t threshold = 16;
            while (last - first > threshold)
            {
                if (depthLimit == 0)
                {
                    std::partial_sort(first, last, last, spatialLess);
                    return;
                }
                --depthLimit;
                SpatialCandidate* middle = first + (last - first) / 2;
                spatialMoveMedianToFirst(first, first + 1, middle, last - 1);
                SpatialCandidate* cut = spatialUnguardedPartition(first + 1, last, first);
                spatialIntrosortLoop(cut, last, depthLimit);
                last = cut;
            }
        }

        void spatialUnguardedLinearInsert(SpatialCandidate* last)
        {
            const SpatialCandidate value = *last;
            SpatialCandidate* previous = last - 1;
            while (spatialLess(value, *previous))
            {
                *last = *previous;
                last = previous;
                --previous;
            }
            *last = value;
        }

        void spatialInsertionSort(SpatialCandidate* first, SpatialCandidate* last)
        {
            if (first == last)
            {
                return;
            }
            for (SpatialCandidate* current = first + 1; current != last; ++current)
            {
                if (spatialLess(*current, *first))
                {
                    const SpatialCandidate value = *current;
                    std::move_backward(first, current, current + 1);
                    *first = value;
                }
                else
                {
                    spatialUnguardedLinearInsert(current);
                }
            }
        }

        void referenceSpatialSort(std::vector<SpatialCandidate>* values)
        {
            if (!values || values->size() <= 1)
            {
                return;
            }
            std::size_t depth = 0;
            for (std::size_t count = values->size(); count > 1; count >>= 1U)
            {
                ++depth;
            }
            SpatialCandidate* first = values->data();
            SpatialCandidate* last = first + values->size();
            spatialIntrosortLoop(first, last, depth * 2);
            constexpr std::ptrdiff_t threshold = 16;
            if (last - first > threshold)
            {
                spatialInsertionSort(first, first + threshold);
                for (SpatialCandidate* current = first + threshold; current != last; ++current)
                {
                    spatialUnguardedLinearInsert(current);
                }
            }
            else
            {
                spatialInsertionSort(first, last);
            }
        }

        std::size_t waterFillLevel(const std::vector<std::size_t>& selectedPerBin,
                                   const std::vector<std::vector<SpatialCandidate>>& bins,
                                   std::size_t limit)
        {
            std::vector<unsigned char> fixed(bins.size(), 0);
            std::size_t level = 0;
            for (std::size_t bin = 0; bin < bins.size(); ++bin)
            {
                level += selectedPerBin[bin] + bins[bin].size();
            }
            while (true)
            {
                std::size_t fixedTotal = 0;
                std::vector<std::size_t> totals;
                totals.reserve(bins.size());
                for (std::size_t bin = 0; bin < bins.size(); ++bin)
                {
                    if (fixed[bin])
                    {
                        fixedTotal += selectedPerBin[bin];
                    }
                    else
                    {
                        totals.push_back(selectedPerBin[bin] + bins[bin].size());
                    }
                }
                if (totals.empty() || fixedTotal > limit)
                {
                    return level;
                }
                std::sort(totals.begin(), totals.end());
                const std::size_t budget = limit - fixedTotal;
                std::size_t consumed = 0;
                for (std::size_t index = 0; index < totals.size(); ++index)
                {
                    const std::size_t remaining = totals.size() - index;
                    if (consumed + totals[index] * remaining > budget)
                    {
                        level = (budget - consumed) / remaining;
                        break;
                    }
                    consumed += totals[index];
                }
                bool changed = false;
                for (std::size_t bin = 0; bin < bins.size(); ++bin)
                {
                    if (!fixed[bin] && selectedPerBin[bin] > level)
                    {
                        fixed[bin] = 1;
                        changed = true;
                    }
                }
                if (!changed)
                {
                    return level;
                }
            }
        }

        class ParkMillerEngine
        {
        public:
            explicit ParkMillerEngine(std::uint64_t seed) : _state(seed)
            {
            }

            std::uint64_t bounded(std::uint64_t inclusiveUpper)
            {
                const std::uint64_t range = inclusiveUpper + 1;
                const std::uint64_t scale = 0x7FFFFFFDULL / range;
                const std::uint64_t accepted = scale * range;
                std::uint64_t value = 0;
                do
                {
                    _state = (16807ULL * _state) % 0x7FFFFFFFULL;
                    value = _state - 1;
                } while (value >= accepted);
                return value / scale;
            }

        private:
            std::uint64_t _state;
        };

        void referenceShuffle(std::vector<std::size_t>* values)
        {
            if (!values || values->size() <= 1)
            {
                return;
            }
            ParkMillerEngine engine(1);
            std::size_t index = 1;
            if ((values->size() & 1U) == 0U)
            {
                std::swap((*values)[1], (*values)[engine.bounded(1)]);
                index = 2;
            }
            for (; index + 1 < values->size(); index += 2)
            {
                const std::uint64_t secondRange = index + 2;
                const std::uint64_t combined = engine.bounded((index + 2) * (index + 1) - 1);
                std::swap((*values)[index], (*values)[combined / secondRange]);
                std::swap((*values)[index + 1], (*values)[combined % secondRange]);
            }
        }

        const FeatureKeypoint* findKeypoint(const std::map<ImageId, ReferenceTrackImageFeatures>& images,
                                            const TrackElement& observation)
        {
            const auto image = images.find(observation.imageId);
            if (image == images.end() || observation.featureIdx >= image->second.keypoints.size())
            {
                return nullptr;
            }
            return &image->second.keypoints[observation.featureIdx];
        }

    } // namespace

    std::vector<Track> selectReferenceSpatialTracks(const std::map<ImageId, ReferenceTrackImageFeatures>& images,
                                                    std::vector<Track> tracks,
                                                    std::size_t tiePointLimit)
    {
        if (tiePointLimit == 0 || tracks.empty())
        {
            return tracks;
        }

        std::size_t gridSize = 16;
        for (int attempt = 0; attempt < 5 && gridSize * gridSize > tiePointLimit; ++attempt)
        {
            gridSize >>= 1U;
        }
        if (gridSize == 0)
        {
            return {};
        }
        const std::size_t binCount = gridSize * gridSize;

        std::vector<float> trackWeights(tracks.size(), 0.0f);
        std::map<ImageId, std::size_t> cameraObservations;
        for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
        {
            for (const TrackElement& observation : tracks[trackIndex].elements)
            {
                const FeatureKeypoint* keypoint = findKeypoint(images, observation);
                if (!keypoint)
                {
                    continue;
                }
                trackWeights[trackIndex] += 1.0f / usableScale(*keypoint);
                ++cameraObservations[observation.imageId];
            }
        }

        std::vector<ImageId> cameraOrder;
        cameraOrder.reserve(images.size());
        for (const auto& [imageId, image] : images)
        {
            (void)image;
            cameraOrder.push_back(imageId);
        }
        std::sort(cameraOrder.begin(),
                  cameraOrder.end(),
                  [&](ImageId left, ImageId right)
                  { return std::tie(cameraObservations[left], left) < std::tie(cameraObservations[right], right); });

        std::vector<unsigned char> selected(tracks.size(), 0);
        std::vector<std::size_t> binPermutation(binCount);
        std::iota(binPermutation.begin(), binPermutation.end(), 0);
        referenceShuffle(&binPermutation);

        for (ImageId imageId : cameraOrder)
        {
            const ReferenceTrackImageFeatures& image = images.at(imageId);
            if (!(image.width > 0.0f) || !(image.height > 0.0f))
            {
                continue;
            }
            std::vector<std::vector<SpatialCandidate>> bins(binCount);
            std::vector<std::size_t> selectedPerBin(binCount, 0);
            for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
            {
                const auto observation =
                    std::find_if(tracks[trackIndex].elements.begin(),
                                 tracks[trackIndex].elements.end(),
                                 [imageId](const auto& value) { return value.imageId == imageId; });
                if (observation == tracks[trackIndex].elements.end() ||
                    observation->featureIdx >= image.keypoints.size())
                {
                    continue;
                }
                const FeatureKeypoint& keypoint = image.keypoints[observation->featureIdx];
                const int binX = static_cast<int>(keypoint.x * static_cast<float>(gridSize) / image.width);
                const int binY = static_cast<int>(keypoint.y * static_cast<float>(gridSize) / image.height);
                if (binX < 0 || binY < 0 || binX >= static_cast<int>(gridSize) || binY >= static_cast<int>(gridSize))
                {
                    continue;
                }
                const std::size_t bin = static_cast<std::size_t>(binX) + gridSize * static_cast<std::size_t>(binY);
                if (selected[trackIndex])
                {
                    ++selectedPerBin[bin];
                }
                else
                {
                    if (trackIndex > std::numeric_limits<std::uint32_t>::max())
                    {
                        throw std::overflow_error("too many tracks for reference spatial selector");
                    }
                    bins[bin].push_back({static_cast<std::uint32_t>(trackIndex), trackWeights[trackIndex]});
                }
            }

            for (auto& bin : bins)
            {
                referenceSpatialSort(&bin);
            }
            const std::size_t level = waterFillLevel(selectedPerBin, bins, tiePointLimit);
            std::size_t total = 0;
            for (std::size_t bin = 0; bin < binCount; ++bin)
            {
                while (!bins[bin].empty() && selectedPerBin[bin] < level)
                {
                    selected[bins[bin].back().track] = 1;
                    bins[bin].pop_back();
                    ++selectedPerBin[bin];
                }
                total += selectedPerBin[bin];
            }
            for (std::size_t bin : binPermutation)
            {
                if (total >= tiePointLimit)
                {
                    break;
                }
                if (!bins[bin].empty() && selectedPerBin[bin] <= level)
                {
                    selected[bins[bin].back().track] = 1;
                    bins[bin].pop_back();
                    ++total;
                }
            }
        }

        std::vector<Track> retained;
        retained.reserve(tracks.size());
        for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
        {
            if (selected[trackIndex])
            {
                retained.push_back(std::move(tracks[trackIndex]));
            }
        }
        return retained;
    }

} // namespace xjw::detail
