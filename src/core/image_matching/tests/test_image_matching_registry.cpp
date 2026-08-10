#include "ImageMatchingRegistry.h"
#include "cuda_sift/CudaSiftAlgorithm.h"
#include "loma_r/LoMaRAlgorithm.h"
#include "sift_lightglue/SiftLightGlueAlgorithm.h"

#include <gtest/gtest.h>

namespace xjw::image_matching
{
namespace
{

TEST(ImageMatchingRegistryTest, ExposesBuiltInAlgorithmCapabilities)
{
    const std::vector<ImageMatchingAlgorithmDescriptor> algorithms =
        ImageMatchingRegistry::descriptors();
    ASSERT_EQ(algorithms.size(), 3U);

    const auto findAlgorithm = [&](const QString &id)
    {
        return std::find_if(algorithms.cbegin(), algorithms.cend(),
                            [&](const auto &algorithm) { return algorithm.id == id; });
    };
    const auto sift = findAlgorithm(QString::fromLatin1(kSiftLightGlueAlgorithmId));
    const auto cudaSift = findAlgorithm(QString::fromLatin1(kCudaSiftAlgorithmId));
    const auto loma = findAlgorithm(QString::fromLatin1(kLoMaRAlgorithmId));
    ASSERT_NE(sift, algorithms.cend());
    ASSERT_NE(cudaSift, algorithms.cend());
    ASSERT_NE(loma, algorithms.cend());
    EXPECT_EQ(sift->version, kSiftLightGlueAlgorithmVersion);
    EXPECT_EQ(cudaSift->version, kCudaSiftAlgorithmVersion);
    EXPECT_EQ(cudaSift->inputModel, AlgorithmInputModel::ReusableFeatures);
    EXPECT_FALSE(cudaSift->requiresCuda);
    EXPECT_TRUE(cudaSift->suppliesStableFeatureIds);
    EXPECT_FALSE(cudaSift->requiresColorInput);
    EXPECT_EQ(loma->version, kLoMaRAlgorithmVersion);
    EXPECT_EQ(loma->inputModel, AlgorithmInputModel::ReusableFeatures);
    EXPECT_TRUE(loma->requiresCuda);
    EXPECT_TRUE(loma->suppliesStableFeatureIds);
    EXPECT_TRUE(loma->requiresColorInput);
}

#if !defined(PLASCAN_HAS_TENSORRT)
TEST(ImageMatchingRegistryTest, ReportsTensorRtRequirementWhenCreatingUnavailableAlgorithm)
{
    ASSERT_TRUE(ImageMatchingRegistry::contains(QStringLiteral("sift_lightglue")));
    QString error;
    const std::unique_ptr<IImageMatchingAlgorithm> algorithm =
        ImageMatchingRegistry::create(QStringLiteral("sift_lightglue"),
                                      ImageMatchingRuntimeConfig{},
                                      &error);
    EXPECT_EQ(algorithm, nullptr);
    EXPECT_TRUE(error.contains(QStringLiteral("TensorRT"))) << error.toStdString();
}
#endif

TEST(ImageMatchingRegistryTest, CreatesCudaSiftForCpuFallbackWithoutExternalModel)
{
    ImageMatchingRuntimeConfig config;
    config.forceCpuSift = true;
    config.allowCpuSiftFallback = true;
    QString error;
    const std::unique_ptr<IImageMatchingAlgorithm> algorithm =
        ImageMatchingRegistry::create(QStringLiteral("cuda_sift"), config,
                                      &error);
    EXPECT_NE(algorithm, nullptr) << error.toStdString();
}

TEST(ImageMatchingRegistryTest, RejectsUnknownAlgorithmWithoutFallback)
{
    QString error;
    const std::unique_ptr<IImageMatchingAlgorithm> algorithm =
        ImageMatchingRegistry::create(QStringLiteral("removed_algorithm"),
                                      ImageMatchingRuntimeConfig{},
                                      &error);
    EXPECT_EQ(algorithm, nullptr);
    EXPECT_TRUE(error.contains(QStringLiteral("未注册")));
}

} // namespace
} // namespace xjw::image_matching
