#include "SiftLightGlueAlgorithm.h"

#include "ImageMatchingRegistry.h"
#include "lightglue/LightGlueFeatureBudget.h"
#include "sift/SiftFeatureExtractor.h"

#include <algorithm>
#include <stdexcept>

namespace xjw::image_matching
{

SiftLightGlueAlgorithm::SiftLightGlueAlgorithm(ImageMatchingRuntimeConfig config)
    : _config(std::move(config))
{
}

void SiftLightGlueAlgorithm::ensureMatcher()
{
    if (_matcher)
    {
        return;
    }
    if (_config.tensorRtEnginePath.trimmed().isEmpty())
    {
        throw std::invalid_argument("SIFT + LightGlue requires a TensorRT engine path");
    }
    TensorRtLightGlueConfig matcherConfig;
    matcherConfig.enginePath = _config.tensorRtEnginePath.toStdString();
    matcherConfig.cudaDevice = _config.cudaDevice;
    matcherConfig.scoreThreshold = _config.matchThreshold;
    _matcher = std::make_unique<TensorRtLightGlueMatcher>(matcherConfig);
    if (!_matcher->isLoaded())
    {
        throw std::runtime_error("failed to load the TensorRT LightGlue engine");
    }
}

SiftLightGlueAlgorithm::~SiftLightGlueAlgorithm() = default;

ImageMatchingAlgorithmDescriptor SiftLightGlueAlgorithm::descriptor() const
{
    ImageMatchingAlgorithmDescriptor value;
    value.id = QString::fromLatin1(kSiftLightGlueAlgorithmId);
    value.displayName = QStringLiteral("CUDA SIFT + TensorRT LightGlue");
    value.version = kSiftLightGlueAlgorithmVersion;
    value.inputModel = AlgorithmInputModel::ReusableFeatures;
    value.requiresCuda = true;
    value.suppliesStableFeatureIds = true;
    return value;
}

FeatureSet SiftLightGlueAlgorithm::extract(const ImageFeatureInput &input) const
{
    return SiftFeatureExtractor::extract(input, _config);
}

MatchResult SiftLightGlueAlgorithm::matchFeatures(const FeatureSet &features0,
                                                  const FeatureSet &features1)
{
    // 特征阶段通过同一注册接口创建算法对象，但不应为每幅影像提前加载一次
    // TensorRT engine；matcher 在真正进入像对匹配时按 worker 延迟初始化。
    ensureMatcher();
    if (!features0.isConsistent() || !features1.isConsistent())
    {
        throw std::invalid_argument("LightGlue received an inconsistent SIFT feature set");
    }

    int keypointBudget = _config.maxMatcherKeypoints;
    if (keypointBudget <= 0)
    {
        keypointBudget = _matcher->bucketKeypoints();
    }
    if (keypointBudget <= 0)
    {
        keypointBudget = std::max(features0.size(), features1.size());
    }

    const BudgetedFeatureData budgeted0 =
        budgetFeatureDataForLightGlue(features0, keypointBudget);
    const BudgetedFeatureData budgeted1 =
        budgetFeatureDataForLightGlue(features1, keypointBudget);
    const MatchResult limited = _matcher->match(budgeted0.features, budgeted1.features);
    return remapLightGlueMatchResultToOriginal(
        limited, budgeted0, features0.size(), budgeted1, features1.size());
}

void registerSiftLightGlueAlgorithm()
{
    ImageMatchingAlgorithmDescriptor descriptor;
    descriptor.id = QString::fromLatin1(kSiftLightGlueAlgorithmId);
    descriptor.displayName = QStringLiteral("CUDA SIFT + TensorRT LightGlue");
    descriptor.version = kSiftLightGlueAlgorithmVersion;
    descriptor.inputModel = AlgorithmInputModel::ReusableFeatures;
    descriptor.requiresCuda = true;
    descriptor.suppliesStableFeatureIds = true;

    QString ignoredError;
    ImageMatchingRegistry::registerAlgorithm(
        descriptor,
        [](const ImageMatchingRuntimeConfig &config)
        {
            return std::make_unique<SiftLightGlueAlgorithm>(config);
        },
        &ignoredError);
}

} // namespace xjw::image_matching
