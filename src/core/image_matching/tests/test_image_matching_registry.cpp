#include "ImageMatchingRegistry.h"
#include "sift_lightglue/SiftLightGlueAlgorithm.h"

#include <gtest/gtest.h>

namespace xjw::image_matching
{
namespace
{

TEST(ImageMatchingRegistryTest, ExposesOnlyTheSupportedBuiltInAlgorithm)
{
    const std::vector<ImageMatchingAlgorithmDescriptor> algorithms =
        ImageMatchingRegistry::descriptors();
    ASSERT_EQ(algorithms.size(), 1U);
    EXPECT_EQ(algorithms.front().id, QString::fromLatin1(kSiftLightGlueAlgorithmId));
    EXPECT_EQ(algorithms.front().version, kSiftLightGlueAlgorithmVersion);
    EXPECT_EQ(algorithms.front().inputModel, AlgorithmInputModel::ReusableFeatures);
    EXPECT_TRUE(algorithms.front().requiresCuda);
    EXPECT_TRUE(algorithms.front().suppliesStableFeatureIds);
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
