#include "ReferenceTrackBuilder.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <utility>

namespace xjw
{
    namespace
    {

        using ObservationKey = std::pair<ImageId, FeatureIdx>;

        class MinimumRootDisjointSet
        {
        public:
            explicit MinimumRootDisjointSet(std::size_t size) : _parent(size)
            {
                std::iota(_parent.begin(), _parent.end(), 0);
            }

            std::size_t find(std::size_t value)
            {
                if (_parent[value] != value)
                {
                    _parent[value] = find(_parent[value]);
                }
                return _parent[value];
            }

            void join(std::size_t left, std::size_t right)
            {
                left = find(left);
                right = find(right);
                if (left == right)
                {
                    return;
                }
                if (left < right)
                {
                    _parent[right] = left;
                }
                else
                {
                    _parent[left] = right;
                }
            }

        private:
            std::vector<std::size_t> _parent;
        };

        float usableScale(const FeatureKeypoint& keypoint)
        {
            return std::isfinite(keypoint.scale) && keypoint.scale > 0.0f ? keypoint.scale : 1.0f;
        }

        bool isStationaryTrack(const Track& track, const std::map<ImageId, detail::ReferenceTrackImageFeatures>& images)
        {
            const std::size_t observationCount = track.elements.size();
            if (observationCount < 2)
            {
                return false;
            }
            const int requiredNeighbours = observationCount <= 4 ? (observationCount > 2 ? 2 : 1) : 3;
            for (std::size_t first = 0; first < observationCount; ++first)
            {
                const TrackElement& leftObservation = track.elements[first];
                const auto leftImage = images.find(leftObservation.imageId);
                if (leftImage == images.end() || leftObservation.featureIdx >= leftImage->second.keypoints.size())
                {
                    continue;
                }
                const FeatureKeypoint& left = leftImage->second.keypoints[leftObservation.featureIdx];
                const float leftScale = usableScale(left);
                int closeNeighbours = 0;
                for (std::size_t second = 0; second < observationCount; ++second)
                {
                    if (first == second)
                    {
                        continue;
                    }
                    const TrackElement& rightObservation = track.elements[second];
                    const auto rightImage = images.find(rightObservation.imageId);
                    if (rightImage == images.end() ||
                        rightObservation.featureIdx >= rightImage->second.keypoints.size())
                    {
                        continue;
                    }
                    const FeatureKeypoint& right = rightImage->second.keypoints[rightObservation.featureIdx];
                    const float rightScale = usableScale(right);
                    if (leftScale > 2.0f * rightScale || rightScale > 2.0f * leftScale)
                    {
                        continue;
                    }
                    const double meanScale = 0.5 * static_cast<double>(leftScale + rightScale);
                    const double dx = static_cast<double>(left.x) - right.x;
                    const double dy = static_cast<double>(left.y) - right.y;
                    if (std::sqrt(dx * dx + dy * dy) <= 4.0 * meanScale)
                    {
                        ++closeNeighbours;
                    }
                }
                if (closeNeighbours >= requiredNeighbours)
                {
                    return true;
                }
            }
            return false;
        }

        void removeStationaryTracks(const std::map<ImageId, detail::ReferenceTrackImageFeatures>& images,
                                    ReferenceTrackBuildResult* result)
        {
            std::vector<Track> retained;
            retained.reserve(result->tracks.size());
            for (Track& track : result->tracks)
            {
                if (isStationaryTrack(track, images))
                {
                    ++result->prunedStationaryTracks;
                }
                else
                {
                    retained.push_back(std::move(track));
                }
            }
            result->tracks = std::move(retained);
        }

    } // namespace

    void ReferenceTrackBuilder::addMatchPair(ImageId imageA, ImageId imageB, const std::vector<MatchIndexPair>& matches)
    {
        if (imageA == imageB || imageA == kInvalidImageId || imageB == kInvalidImageId)
        {
            return;
        }
        _edges.reserve(_edges.size() + matches.size());
        for (const MatchIndexPair& match : matches)
        {
            _edges.push_back({imageA, imageB, match.first, match.second});
        }
    }

    void ReferenceTrackBuilder::setImageKeypoints(ImageId imageId,
                                                  const std::vector<FeatureKeypoint>& keypoints,
                                                  float imageWidth,
                                                  float imageHeight)
    {
        detail::ReferenceTrackImageFeatures image;
        image.keypoints = keypoints;
        image.width = imageWidth;
        image.height = imageHeight;
        if (!(image.width > 0.0f) || !(image.height > 0.0f))
        {
            float maximumX = 0.0f;
            float maximumY = 0.0f;
            for (const FeatureKeypoint& keypoint : keypoints)
            {
                if (std::isfinite(keypoint.x))
                {
                    maximumX = std::max(maximumX, keypoint.x);
                }
                if (std::isfinite(keypoint.y))
                {
                    maximumY = std::max(maximumY, keypoint.y);
                }
            }
            image.width = std::max(1.0f, maximumX + 1.0f);
            image.height = std::max(1.0f, maximumY + 1.0f);
        }
        _images[imageId] = std::move(image);
    }

    ReferenceTrackBuildResult ReferenceTrackBuilder::build(const ReferenceTrackBuildOptions& options) const
    {
        ReferenceTrackBuildResult result;
        std::map<ObservationKey, std::size_t> provisionalByObservation;
        std::vector<Edge> validEdges;
        validEdges.reserve(_edges.size());

        std::size_t provisionalCount = 0;
        for (const Edge& edge : _edges)
        {
            const auto firstImage = _images.find(edge.imageA);
            const auto secondImage = _images.find(edge.imageB);
            if (edge.featureA == kInvalidFeatureIdx || edge.featureB == kInvalidFeatureIdx ||
                firstImage == _images.end() || secondImage == _images.end() ||
                edge.featureA >= firstImage->second.keypoints.size() ||
                edge.featureB >= secondImage->second.keypoints.size())
            {
                ++result.invalidMatchCount;
                continue;
            }

            const ObservationKey firstKey{edge.imageA, edge.featureA};
            const ObservationKey secondKey{edge.imageB, edge.featureB};
            auto first = provisionalByObservation.find(firstKey);
            auto second = provisionalByObservation.find(secondKey);
            if (first != provisionalByObservation.end() && second == provisionalByObservation.end())
            {
                provisionalByObservation.emplace(secondKey, first->second);
            }
            else if (first == provisionalByObservation.end() && second != provisionalByObservation.end())
            {
                provisionalByObservation.emplace(firstKey, second->second);
            }
            else if (first == provisionalByObservation.end() && second == provisionalByObservation.end())
            {
                provisionalByObservation.emplace(firstKey, provisionalCount);
                provisionalByObservation.emplace(secondKey, provisionalCount);
                ++provisionalCount;
            }
            validEdges.push_back(edge);
        }

        MinimumRootDisjointSet sets(provisionalCount);
        for (const Edge& edge : validEdges)
        {
            sets.join(provisionalByObservation.at({edge.imageA, edge.featureA}),
                      provisionalByObservation.at({edge.imageB, edge.featureB}));
        }

        std::vector<std::size_t> compact(provisionalCount, 0);
        std::size_t trackCount = 0;
        for (std::size_t point = 0; point < provisionalCount; ++point)
        {
            const std::size_t root = sets.find(point);
            if (root == point)
            {
                compact[point] = trackCount++;
            }
            else
            {
                compact[point] = compact[root];
            }
        }
        result.generatedTrackCount = static_cast<int>(trackCount);
        result.tracks.resize(trackCount);
        for (const auto& [imageId, image] : _images)
        {
            for (std::size_t featureIndex = 0; featureIndex < image.keypoints.size(); ++featureIndex)
            {
                const auto provisional =
                    provisionalByObservation.find({imageId, static_cast<FeatureIdx>(featureIndex)});
                if (provisional == provisionalByObservation.end())
                {
                    continue;
                }
                const std::size_t trackIndex = compact[sets.find(provisional->second)];
                result.tracks[trackIndex].elements.push_back({imageId, static_cast<FeatureIdx>(featureIndex)});
                result.tracks[trackIndex].confidence = 1.0;
            }
        }

        for (Track& track : result.tracks)
        {
            std::map<ImageId, std::size_t> observationsPerImage;
            for (const TrackElement& observation : track.elements)
            {
                ++observationsPerImage[observation.imageId];
            }
            std::vector<TrackElement> retained;
            retained.reserve(track.elements.size());
            for (const TrackElement& observation : track.elements)
            {
                if (observationsPerImage[observation.imageId] == 1)
                {
                    retained.push_back(observation);
                }
                else
                {
                    ++result.removedDuplicateObservations;
                }
            }
            track.elements = std::move(retained);
        }

        const std::size_t beforeShortRemoval = result.tracks.size();
        result.tracks.erase(std::remove_if(result.tracks.begin(),
                                           result.tracks.end(),
                                           [](const Track& track) { return track.length() <= 1; }),
                            result.tracks.end());
        result.removedShortTracks = static_cast<int>(beforeShortRemoval - result.tracks.size());

        if (options.excludeStationaryTracks)
        {
            removeStationaryTracks(_images, &result);
        }

        result.tracksBeforeSpatialSelection = static_cast<int>(result.tracks.size());
        result.tracks = detail::selectReferenceSpatialTracks(_images, std::move(result.tracks), options.tiePointLimit);
        result.prunedBySpatialSelection = result.tracksBeforeSpatialSelection - static_cast<int>(result.tracks.size());
        for (const Track& track : result.tracks)
        {
            ++result.trackLengthHistogram[static_cast<int>(track.length())];
        }
        return result;
    }

} // namespace xjw
