#include "ImageMatchingRegistry.h"
#include "loma_r/LoMaRAlgorithm.h"
#include "sift_lightglue/SiftLightGlueAlgorithm.h"

#include <gtest/gtest.h>

namespace xjw::image_matching
{
namespace
{

TEST(ImageMatchingRegistryTest, ExposesOnlyAvailableBuiltInAlgorithms)
{
    const std::vector<ImageMatchingAlgorithmDescriptor> algorithms =
        ImageMatchingRegistry::descriptors();
#if defined(PLASCAN_HAS_TENSORRT)
    ASSERT_EQ(algorithms.size(), 2U);

    const auto findAlgorithm = [&](const QString &id)
    {
        return std::find_if(algorithms.cbegin(), algorithms.cend(),
                            [&](const auto &algorithm) { return algorithm.id == id; });
    };
    const auto sift = findAlgorithm(QString::fromLatin1(kSiftLightGlueAlgorithmId));
    const auto loma = findAlgorithm(QString::fromLatin1(kLoMaRAlgorithmId));
    ASSERT_NE(sift, algorithms.cend());
    ASSERT_NE(loma, algorithms.cend());
    EXPECT_EQ(sift->version, kSiftLightGlueAlgorithmVersion);
    EXPECT_EQ(loma->version, kLoMaRAlgorithmVersion);
    EXPECT_EQ(loma->inputModel, AlgorithmInputModel::ReusableFeatures);
    EXPECT_TRUE(loma->requiresCuda);
    EXPECT_TRUE(loma->suppliesStableFeatureIds);
    EXPECT_TRUE(loma->requiresColorInput);
#else
    EXPECT_TRUE(algorithms.empty());
#endif
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
