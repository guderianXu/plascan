#pragma once

#include "../FeatureSet.h"
#include "metalign/matching.hpp"

#include <memory>
#include <mutex>

namespace xjw::image_matching
{

    class PlaMatchHctFeaturePayload final : public IFeaturePayload
    {
    public:
        explicit PlaMatchHctFeaturePayload(metalign::FeatureSet features);
        ~PlaMatchHctFeaturePayload() override;

        std::string schemaId() const override;
        std::size_t approximateBytes() const override;
        bool isConsistent(std::size_t featureCount) const override;

        const metalign::FeatureSet& fullFeatures() const;
        const metalign::FeatureSet& coarseFeatures() const;
        const metalign::CpuDescriptorIndex& fullIndex() const;
        const metalign::CpuDescriptorIndex& coarseIndex() const;

    private:
        metalign::FeatureSet _fullFeatures;
        metalign::FeatureSet _coarseFeatures;
        mutable std::unique_ptr<metalign::CpuDescriptorIndex> _fullIndex;
        mutable std::unique_ptr<metalign::CpuDescriptorIndex> _coarseIndex;
        mutable std::mutex _fullIndexMutex;
        mutable std::mutex _coarseIndexMutex;
    };

} // namespace xjw::image_matching
