#include "PlaMatchHctFeaturePayload.h"

#include <utility>

namespace xjw::image_matching
{
    namespace
    {

        std::size_t estimatedFeatureBytes(const metalign::FeatureSet& features)
        {
            return features.keypoints.size() * sizeof(metalign::Keypoint);
        }

        std::size_t estimatedIndexBytes(const metalign::FeatureSet& features)
        {
            constexpr std::size_t treeCount = 8;
            constexpr std::size_t bytesPerTreeRow = sizeof(std::uint32_t) + sizeof(std::uint32_t) * 2;
            return features.keypoints.size() * (sizeof(metalign::Descriptor) + treeCount * bytesPerTreeRow);
        }

        bool validFeatureRows(const metalign::FeatureSet& features)
        {
            for (const metalign::Keypoint& keypoint : features.keypoints)
            {
                if (keypoint.source_id >= features.keypoints.size())
                {
                    return false;
                }
            }
            return true;
        }

    } // namespace

    PlaMatchHctFeaturePayload::PlaMatchHctFeaturePayload(metalign::FeatureSet features)
        : _fullFeatures(std::move(features))
    {
        _coarseFeatures.path = _fullFeatures.path;
        _coarseFeatures.image_width = _fullFeatures.image_width;
        _coarseFeatures.image_height = _fullFeatures.image_height;
        _coarseFeatures.focal_length_pixels = _fullFeatures.focal_length_pixels;
        _coarseFeatures.sensor_id = _fullFeatures.sensor_id;
        _coarseFeatures.keypoints = std::move(_fullFeatures.coarse_keypoints);
        _coarseFeatures.source_keypoint_count = _coarseFeatures.keypoints.size();
        _coarseFeatures.global_descriptor = _fullFeatures.global_descriptor;
        _fullFeatures.coarse_keypoints.clear();
    }

    PlaMatchHctFeaturePayload::~PlaMatchHctFeaturePayload() = default;

    std::string PlaMatchHctFeaturePayload::schemaId() const
    {
        return "plamatch_hct:mldb64:v1";
    }

    std::size_t PlaMatchHctFeaturePayload::approximateBytes() const
    {
        std::scoped_lock lock(_fullIndexMutex, _coarseIndexMutex);
        return estimatedFeatureBytes(_fullFeatures) + estimatedFeatureBytes(_coarseFeatures) +
               (_fullIndex ? estimatedIndexBytes(_fullFeatures) : 0) +
               (_coarseIndex ? estimatedIndexBytes(_coarseFeatures) : 0);
    }

    bool PlaMatchHctFeaturePayload::isConsistent(std::size_t featureCount) const
    {
        return featureCount == _fullFeatures.keypoints.size() && !_fullFeatures.keypoints.empty() &&
               _fullFeatures.image_width > 0 && _fullFeatures.image_height > 0 &&
               _coarseFeatures.image_width == _fullFeatures.image_width &&
               _coarseFeatures.image_height == _fullFeatures.image_height && validFeatureRows(_fullFeatures) &&
               validFeatureRows(_coarseFeatures);
    }

    const metalign::FeatureSet& PlaMatchHctFeaturePayload::fullFeatures() const
    {
        return _fullFeatures;
    }

    const metalign::FeatureSet& PlaMatchHctFeaturePayload::coarseFeatures() const
    {
        return _coarseFeatures;
    }

    const metalign::CpuDescriptorIndex& PlaMatchHctFeaturePayload::fullIndex() const
    {
        std::lock_guard lock(_fullIndexMutex);
        if (!_fullIndex)
        {
            _fullIndex = std::make_unique<metalign::CpuDescriptorIndex>(_fullFeatures);
        }
        return *_fullIndex;
    }

    const metalign::CpuDescriptorIndex& PlaMatchHctFeaturePayload::coarseIndex() const
    {
        std::lock_guard lock(_coarseIndexMutex);
        if (!_coarseIndex)
        {
            _coarseIndex = std::make_unique<metalign::CpuDescriptorIndex>(_coarseFeatures);
        }
        return *_coarseIndex;
    }

} // namespace xjw::image_matching
