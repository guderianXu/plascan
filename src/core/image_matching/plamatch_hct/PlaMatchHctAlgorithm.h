#pragma once

#include "../ImageMatchingAlgorithm.h"

#include <memory>

namespace metalign
{
    class DescriptorAccelerator;
}

namespace xjw::image_matching
{

    inline constexpr const char* kPlaMatchHctAlgorithmId = "plamatch_hct";
    inline constexpr std::uint32_t kPlaMatchHctAlgorithmVersion = 2;

    struct PlaMatchHctBackendResolution
    {
        bool valid = false;
        SiftComputeBackend backend = SiftComputeBackend::Cpu;
        int deviceIndex = 0;
        QString deviceName;
        QString displayName;
        QString errorMessage;
    };

    PlaMatchHctBackendResolution resolvePlaMatchHctBackend(SiftComputeBackend requestedBackend, int deviceIndex);

    class PlaMatchHctAlgorithm final : public IImageMatchingAlgorithm
    {
    public:
        explicit PlaMatchHctAlgorithm(ImageMatchingRuntimeConfig config);
        ~PlaMatchHctAlgorithm() override;

        ImageMatchingAlgorithmDescriptor descriptor() const override;
        FeatureSet extract(const ImageFeatureInput& input) const override;
        MatchResult matchFeatures(const FeatureSet& features0, const FeatureSet& features1) override;
        std::vector<MatchResult> matchFeatureBatch(std::span<const FeaturePairInput> pairs,
                                                   const std::function<bool()>& shouldCancel,
                                                   const BatchMatchProgressCallback& progressCallback = {}) override;

    private:
        ImageMatchingRuntimeConfig _config;
        SiftComputeBackend _resolvedBackend = SiftComputeBackend::Cpu;
        QString _computeBackend;
        std::unique_ptr<metalign::DescriptorAccelerator> _accelerator;
    };

    void registerPlaMatchHctAlgorithm();

} // namespace xjw::image_matching
