#include "MatchPhotosAlgorithmSelector.h"

#include <gtest/gtest.h>

#include <array>
#include <utility>

TEST(AlignmentAccuracyTest, MapsNamesAndApiValuesExactly)
{
    using xjw::matchphotos::AlignmentAccuracy;
    using xjw::matchphotos::alignmentAccuracyDownscale;
    using xjw::matchphotos::alignmentAccuracyFromName;
    using xjw::matchphotos::alignmentAccuracyName;

    const std::array<std::pair<const char*, int>, 5> expected{
        {{"highest", 0}, {"high", 1}, {"medium", 2}, {"low", 4}, {"lowest", 8}}};
    for (const auto& [name, downscale] : expected)
    {
        const AlignmentAccuracy accuracy = alignmentAccuracyFromName(QString::fromLatin1(name));
        EXPECT_EQ(alignmentAccuracyName(accuracy), QString::fromLatin1(name));
        EXPECT_EQ(alignmentAccuracyDownscale(accuracy), downscale);
    }
    EXPECT_EQ(alignmentAccuracyFromName(QStringLiteral("standard")), AlignmentAccuracy::Medium);
    EXPECT_EQ(alignmentAccuracyFromName(QStringLiteral("fast")), AlignmentAccuracy::Lowest);
    EXPECT_EQ(alignmentAccuracyFromName(QStringLiteral("unknown")), AlignmentAccuracy::High);
}

TEST(MatchPhotosAlgorithmSelectorTest, DefaultUsesRegisteredPlaMatchHct)
{
    xjw::matchphotos::MatchPhotosOptions options;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.algorithmId, QStringLiteral("plamatch_hct"));
    EXPECT_GT(plan.algorithmVersion, 0u);
    EXPECT_EQ(plan.displayName, QStringLiteral("PlaMatch-HCT（空间一致性二进制匹配）"));
    EXPECT_TRUE(plan.extractsFeaturesInMemory);
    EXPECT_FALSE(plan.requiresCuda);
    EXPECT_TRUE(plan.rotationRobust);
    EXPECT_TRUE(plan.requiresColorInput);
    EXPECT_TRUE(plan.suppliesCoarsePairPreselection);
    EXPECT_TRUE(plan.supportsBatchFeatureMatching);
    EXPECT_FALSE(plan.lowTextureRecovery);
    EXPECT_TRUE(plan.reason.contains(QStringLiteral(".pimatch")));
}

TEST(MatchPhotosAlgorithmSelectorTest, FastAutoSiftDisablesLowTextureRecovery)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("auto_sift");
    options.profile = xjw::matchphotos::MatchPhotosProfile::Fast;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    ASSERT_TRUE(plan.valid);
    EXPECT_FALSE(plan.lowTextureRecovery);
}

TEST(MatchPhotosAlgorithmSelectorTest, RejectsUnknownAlgorithm)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("removed_matcher");

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_FALSE(plan.valid);
    EXPECT_TRUE(plan.validationError.contains(QStringLiteral("未注册")));
}

TEST(MatchPhotosAlgorithmSelectorTest, SelectsRegisteredLoMaR)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("loma_r");

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.algorithmId, QStringLiteral("loma_r"));
    EXPECT_EQ(plan.displayName, QStringLiteral("LoMa-R (TensorRT)"));
    EXPECT_TRUE(plan.extractsFeaturesInMemory);
    EXPECT_TRUE(plan.requiresCuda);
    EXPECT_TRUE(plan.rotationRobust);
}

TEST(MatchPhotosAlgorithmSelectorTest, SelectsRegisteredAutoSift)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("auto_sift");

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.algorithmId, QStringLiteral("auto_sift"));
    EXPECT_EQ(plan.displayName, QStringLiteral("Auto SIFT（CUDA / Metal / OpenCL / CPU）"));
    EXPECT_TRUE(plan.extractsFeaturesInMemory);
    EXPECT_FALSE(plan.requiresCuda);
    EXPECT_TRUE(plan.rotationRobust);
}

TEST(MatchPhotosAlgorithmSelectorTest, SelectsRegisteredPlaMatchHct)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("plamatch_hct");
    options.device = xjw::matchphotos::ComputeDevice::Cpu;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.algorithmId, QStringLiteral("plamatch_hct"));
    EXPECT_EQ(plan.displayName, QStringLiteral("PlaMatch-HCT（空间一致性二进制匹配）"));
    EXPECT_TRUE(plan.extractsFeaturesInMemory);
    EXPECT_FALSE(plan.requiresCuda);
    EXPECT_TRUE(plan.requiresColorInput);
    EXPECT_EQ(plan.featureSchemaVersion, 2);
    EXPECT_EQ(plan.alignmentDownscale, 1);
    EXPECT_TRUE(plan.suppliesCoarsePairPreselection);
    EXPECT_TRUE(plan.supportsBatchFeatureMatching);
    EXPECT_EQ(plan.executionBackend, xjw::image_matching::SiftComputeBackend::Cpu);
    EXPECT_FALSE(plan.preferCuda);
    EXPECT_TRUE(plan.backendReason.contains(QStringLiteral("CPU HCTree")));
}

TEST(MatchPhotosAlgorithmSelectorTest, CpuCompatibleProfileKeepsDefaultPlaMatchHctOnCpu)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.profile = xjw::matchphotos::MatchPhotosProfile::CpuCompatible;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    ASSERT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.algorithmId, QStringLiteral("plamatch_hct"));
    EXPECT_EQ(plan.executionBackend, xjw::image_matching::SiftComputeBackend::Cpu);
    EXPECT_EQ(plan.computeDeviceDisplayName, QStringLiteral("CPU HCTree"));
}

TEST(MatchPhotosAlgorithmSelectorTest, RejectsUnsupportedMetalPlaMatchHctDevice)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("plamatch_hct");
    options.device = xjw::matchphotos::ComputeDevice::Metal;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_FALSE(plan.valid);
    EXPECT_TRUE(plan.validationError.contains(QStringLiteral("不提供 Metal")));
}

TEST(MatchPhotosAlgorithmSelectorTest, AutoUsesResolvedCpuBackend)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("auto_sift");

    auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);
    plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::resolveExecutionBackend(
        options, std::move(plan), xjw::image_matching::SiftComputeBackend::Cpu);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.executionBackend, xjw::image_matching::SiftComputeBackend::Cpu);
    EXPECT_TRUE(plan.backendFallback);
    EXPECT_FALSE(plan.requiresCuda);
    EXPECT_TRUE(plan.backendReason.contains(QStringLiteral("OpenCV CPU")));
    EXPECT_TRUE(plan.computeDeviceName.isEmpty());
    EXPECT_EQ(plan.computeDeviceDisplayName, QStringLiteral("OpenCV CPU"));
}

TEST(MatchPhotosAlgorithmSelectorTest, ExplicitCudaUsesResolvedCudaBackend)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("auto_sift");
    options.device = xjw::matchphotos::ComputeDevice::Cuda;

    auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);
    plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::resolveExecutionBackend(
        options, std::move(plan), xjw::image_matching::SiftComputeBackend::Cuda);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.executionBackend, xjw::image_matching::SiftComputeBackend::Cuda);
}

TEST(MatchPhotosAlgorithmSelectorTest, ExplicitCpuUsesCpuSiftBackend)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("auto_sift");
    options.device = xjw::matchphotos::ComputeDevice::Cpu;

    auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);
    plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::resolveExecutionBackend(
        options, std::move(plan), xjw::image_matching::SiftComputeBackend::Cpu);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.executionBackend, xjw::image_matching::SiftComputeBackend::Cpu);
    EXPECT_FALSE(plan.backendFallback);
}

TEST(MatchPhotosAlgorithmSelectorTest, ExplicitMetalUsesMetalBackend)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("auto_sift");
    options.device = xjw::matchphotos::ComputeDevice::Metal;

    auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);
    plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::resolveExecutionBackend(
        options, std::move(plan), xjw::image_matching::SiftComputeBackend::Metal);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.executionBackend, xjw::image_matching::SiftComputeBackend::Metal);
}

TEST(MatchPhotosAlgorithmSelectorTest, RejectsCpuForCudaOnlyAlgorithm)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("sift_lightglue");
    options.device = xjw::matchphotos::ComputeDevice::Cpu;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_FALSE(plan.valid);
    EXPECT_TRUE(plan.validationError.contains(QStringLiteral("CUDA")));
}

TEST(MatchPhotosAlgorithmSelectorTest, GuidedMatchingRemainsExplicit)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.profile = xjw::matchphotos::MatchPhotosProfile::DifficultTexture;
    options.guidedMatchingMode = xjw::matchphotos::GuidedMatchingMode::Automatic;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid);
    EXPECT_EQ(plan.guidedMatchingMode, xjw::matchphotos::GuidedMatchingMode::Automatic);
    EXPECT_GE(plan.maxKeypoints, 12000);
}

TEST(MatchPhotosAlgorithmSelectorTest, ExplicitZeroKeypointLimitMeansUnlimited)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.useExplicitKeypointLimit = true;
    options.maxKeypoints = 0;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid);
    EXPECT_EQ(plan.maxKeypoints, 0);
}
