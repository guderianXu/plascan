#include "ImageMatchingRegistry.h"
#include "loma_r/LoMaRAlgorithm.h"
#include "plamatch_hct/PlaMatchHctAlgorithm.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift_lightglue/SiftLightGlueAlgorithm.h"

#include <gtest/gtest.h>

namespace xjw::image_matching
{
    namespace
    {

        TEST(ImageMatchingRegistryTest, ExposesBuiltInAlgorithmCapabilities)
        {
            const std::vector<ImageMatchingAlgorithmDescriptor> algorithms = ImageMatchingRegistry::descriptors();
            ASSERT_EQ(algorithms.size(), 4U);

            const auto findAlgorithm = [&](const QString& id)
            {
                return std::find_if(
                    algorithms.cbegin(), algorithms.cend(), [&](const auto& algorithm) { return algorithm.id == id; });
            };
            const auto sift = findAlgorithm(QString::fromLatin1(kSiftLightGlueAlgorithmId));
            const auto autoSift = findAlgorithm(QString::fromLatin1(kAutoSiftAlgorithmId));
            const auto loma = findAlgorithm(QString::fromLatin1(kLoMaRAlgorithmId));
            const auto plaMatch = findAlgorithm(QString::fromLatin1(kPlaMatchHctAlgorithmId));
            ASSERT_NE(sift, algorithms.cend());
            ASSERT_NE(autoSift, algorithms.cend());
            ASSERT_NE(loma, algorithms.cend());
            ASSERT_NE(plaMatch, algorithms.cend());
            EXPECT_EQ(sift->version, kSiftLightGlueAlgorithmVersion);
            EXPECT_EQ(autoSift->version, kAutoSiftAlgorithmVersion);
            EXPECT_EQ(autoSift->inputModel, AlgorithmInputModel::ReusableFeatures);
            EXPECT_FALSE(autoSift->requiresCuda);
            EXPECT_TRUE(autoSift->suppliesStableFeatureIds);
            EXPECT_FALSE(autoSift->requiresColorInput);
            EXPECT_EQ(loma->version, kLoMaRAlgorithmVersion);
            EXPECT_EQ(loma->inputModel, AlgorithmInputModel::ReusableFeatures);
            EXPECT_TRUE(loma->requiresCuda);
            EXPECT_TRUE(loma->suppliesStableFeatureIds);
            EXPECT_TRUE(loma->requiresColorInput);
            EXPECT_EQ(plaMatch->version, kPlaMatchHctAlgorithmVersion);
            EXPECT_EQ(plaMatch->inputModel, AlgorithmInputModel::ReusableFeatures);
            EXPECT_FALSE(plaMatch->requiresCuda);
            EXPECT_TRUE(plaMatch->suppliesStableFeatureIds);
            EXPECT_TRUE(plaMatch->requiresColorInput);
            EXPECT_TRUE(plaMatch->suppliesCoarsePairPreselection);
        }

#if !defined(PLASCAN_HAS_TENSORRT)
        TEST(ImageMatchingRegistryTest, ReportsTensorRtRequirementWhenCreatingUnavailableAlgorithm)
        {
            ASSERT_TRUE(ImageMatchingRegistry::contains(QStringLiteral("sift_lightglue")));
            QString error;
            const std::unique_ptr<IImageMatchingAlgorithm> algorithm =
                ImageMatchingRegistry::create(QStringLiteral("sift_lightglue"), ImageMatchingRuntimeConfig{}, &error);
            EXPECT_EQ(algorithm, nullptr);
            EXPECT_TRUE(error.contains(QStringLiteral("TensorRT"))) << error.toStdString();
        }
#endif

        TEST(ImageMatchingRegistryTest, CreatesAutoSiftForCpuFallbackWithoutExternalModel)
        {
            ImageMatchingRuntimeConfig config;
            config.siftBackend = SiftComputeBackend::Cpu;
            QString error;
            const std::unique_ptr<IImageMatchingAlgorithm> algorithm =
                ImageMatchingRegistry::create(QStringLiteral("auto_sift"), config, &error);
            EXPECT_NE(algorithm, nullptr) << error.toStdString();
        }

        TEST(ImageMatchingRegistryTest, CreatesPlaMatchHctWithoutExternalModel)
        {
            QString error;
            const std::unique_ptr<IImageMatchingAlgorithm> algorithm = ImageMatchingRegistry::create(
                QString::fromLatin1(kPlaMatchHctAlgorithmId), ImageMatchingRuntimeConfig{}, &error);
            EXPECT_NE(algorithm, nullptr) << error.toStdString();
        }

        TEST(ImageMatchingRegistryTest, RejectsUnknownAlgorithmWithoutFallback)
        {
            QString error;
            const std::unique_ptr<IImageMatchingAlgorithm> algorithm = ImageMatchingRegistry::create(
                QStringLiteral("removed_algorithm"), ImageMatchingRuntimeConfig{}, &error);
            EXPECT_EQ(algorithm, nullptr);
            EXPECT_TRUE(error.contains(QStringLiteral("未注册")));
        }

    } // namespace
} // namespace xjw::image_matching
