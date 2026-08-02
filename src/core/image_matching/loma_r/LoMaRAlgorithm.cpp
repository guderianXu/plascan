#include "LoMaRAlgorithm.h"

#include "ImageMatchingRegistry.h"
#include "LoMaRTensorRtBackend.h"

#include <stdexcept>
#include <utility>

namespace xjw::image_matching
{

class LoMaRAlgorithm::Impl
{
public:
    explicit Impl(ImageMatchingRuntimeConfig config)
        : _config(std::move(config))
    {
        if (_config.tensorRtFeatureEnginePath.trimmed().isEmpty() ||
            _config.tensorRtMatcherEnginePath.trimmed().isEmpty())
        {
            throw std::invalid_argument("LoMa-R requires feature and matcher TensorRT engines");
        }
    }

    LoMaRTensorRtBackend &backend() const
    {
        if (!_backend)
        {
            LoMaRTensorRtConfig config;
            config.featureEnginePath = _config.tensorRtFeatureEnginePath;
            config.matcherEnginePath = _config.tensorRtMatcherEnginePath;
            config.cudaDevice = _config.cudaDevice;
            config.inputWidth = _config.modelInputWidth;
            config.inputHeight = _config.modelInputHeight;
            config.keypointCount = _config.maxMatcherKeypoints;
            config.descriptorDimension = _config.descriptorDimension;
            config.maxKeypoints = _config.maxKeypoints;
            config.matchThreshold = _config.matchThreshold;
            _backend = std::make_unique<LoMaRTensorRtBackend>(std::move(config));
        }
        return *_backend;
    }

private:
    ImageMatchingRuntimeConfig _config;
    mutable std::unique_ptr<LoMaRTensorRtBackend> _backend;
};

LoMaRAlgorithm::LoMaRAlgorithm(ImageMatchingRuntimeConfig config)
    : _impl(std::make_unique<Impl>(std::move(config)))
{
}

LoMaRAlgorithm::~LoMaRAlgorithm() = default;

ImageMatchingAlgorithmDescriptor LoMaRAlgorithm::descriptor() const
{
    ImageMatchingAlgorithmDescriptor value;
    value.id = QString::fromLatin1(kLoMaRAlgorithmId);
    value.displayName = QStringLiteral("LoMa-R (TensorRT)");
    value.version = kLoMaRAlgorithmVersion;
    value.inputModel = AlgorithmInputModel::ReusableFeatures;
    value.requiresCuda = true;
    value.suppliesStableFeatureIds = true;
    value.requiresColorInput = true;
    return value;
}

FeatureSet LoMaRAlgorithm::extract(const ImageFeatureInput &input) const
{
    return _impl->backend().extract(input);
}

MatchResult LoMaRAlgorithm::matchFeatures(const FeatureSet &features0,
                                          const FeatureSet &features1)
{
    return _impl->backend().match(features0, features1);
}

void registerLoMaRAlgorithm()
{
    ImageMatchingAlgorithmDescriptor descriptor;
    descriptor.id = QString::fromLatin1(kLoMaRAlgorithmId);
    descriptor.displayName = QStringLiteral("LoMa-R (TensorRT)");
    descriptor.version = kLoMaRAlgorithmVersion;
    descriptor.inputModel = AlgorithmInputModel::ReusableFeatures;
    descriptor.requiresCuda = true;
    descriptor.suppliesStableFeatureIds = true;
    descriptor.requiresColorInput = true;

    QString ignoredError;
    ImageMatchingRegistry::registerAlgorithm(
        descriptor,
        [](const ImageMatchingRuntimeConfig &config)
        {
            return std::make_unique<LoMaRAlgorithm>(config);
        },
        &ignoredError);
}

} // namespace xjw::image_matching
