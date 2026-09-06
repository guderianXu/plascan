#include "ImageMatchingRegistry.h"
#include "plamatch_hct/PlaMatchHctAlgorithm.h"
#include "plamatch_hct/PlaMatchHctFeatureCacheFile.h"
#include "plamatch_hct/PlaMatchHctFeaturePayload.h"
#include "metalign/image.hpp"

#include <QFile>
#include <QTemporaryDir>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace xjw::image_matching
{
    namespace
    {

        cv::Mat texturedImage()
        {
            cv::Mat image(256, 256, CV_8UC3);
            cv::RNG random(0x504C414D);
            random.fill(image, cv::RNG::UNIFORM, 0, 256);
            return image;
        }

        TEST(PlaMatchHctAccuracyTest, HighestBuildsRecoveredHalfStepLattice)
        {
            metalign::Image source;
            source.width = 3;
            source.height = 2;
            source.gray = {0.0F, 2.0F, 4.0F, 6.0F, 8.0F, 10.0F};

            const metalign::Image highest = metalign::upsample_highest(source);

            EXPECT_EQ(highest.width, 5U);
            EXPECT_EQ(highest.height, 3U);
            const std::vector<float> expected{
                0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
            EXPECT_EQ(highest.gray, expected);
        }

        TEST(PlaMatchHctAlgorithmTest, IsRegisteredAsReusableCpuAlgorithm)
        {
            ASSERT_TRUE(ImageMatchingRegistry::contains(QStringLiteral("plamatch_hct")));
            const auto descriptors = ImageMatchingRegistry::descriptors();
            const auto found = std::find_if(descriptors.cbegin(),
                                            descriptors.cend(),
                                            [](const ImageMatchingAlgorithmDescriptor& descriptor)
                                            { return descriptor.id == QLatin1String(kPlaMatchHctAlgorithmId); });
            ASSERT_NE(found, descriptors.cend());
            EXPECT_EQ(found->version, kPlaMatchHctAlgorithmVersion);
            EXPECT_EQ(found->inputModel, AlgorithmInputModel::ReusableFeatures);
            EXPECT_FALSE(found->requiresCuda);
            EXPECT_TRUE(found->suppliesStableFeatureIds);
            EXPECT_TRUE(found->requiresColorInput);
            EXPECT_TRUE(found->suppliesCoarsePairPreselection);
            EXPECT_TRUE(found->supportsBatchFeatureMatching);
        }

        TEST(PlaMatchHctAlgorithmTest, ExtractsMldbAndMatchesIdenticalImage)
        {
            ImageMatchingRuntimeConfig config;
            config.maxKeypoints = 256;
            config.siftBackend = SiftComputeBackend::Cpu;
            PlaMatchHctAlgorithm algorithm(config);

            const cv::Mat image = texturedImage();
            ImageFeatureInput input;
            input.imagePath = QStringLiteral("synthetic.png");
            input.colorImage = image;
            input.originalWidth = image.cols;
            input.originalHeight = image.rows;

            const FeatureSet features = algorithm.extract(input);
            ASSERT_TRUE(features.isConsistent());
            ASSERT_GT(features.size(), 8);
            EXPECT_EQ(features.descriptors.type(), CV_8U);
            EXPECT_EQ(features.descriptors.cols, 64);
            EXPECT_EQ(features.sourceAlgorithm, kPlaMatchHctAlgorithmId);
            EXPECT_EQ(features.computeBackend, "plamatch_hct_cpu");
            EXPECT_NE(std::dynamic_pointer_cast<const PlaMatchHctFeaturePayload>(features.payload), nullptr);

            const MatchResult matches = algorithm.matchFeatures(features, features);
            EXPECT_GT(matches.numMatches, 7);
            EXPECT_EQ(matches.numMatches, static_cast<int>(matches.cvMatches.size()));
            for (const cv::DMatch& match : matches.cvMatches)
            {
                EXPECT_EQ(match.queryIdx, match.trainIdx);
                EXPECT_FLOAT_EQ(match.distance, 0.0F);
            }
        }

        TEST(PlaMatchHctFeatureCacheTest, RoundTripsCompleteFeaturesAndRejectsChangedSource)
        {
            QTemporaryDir temporaryDirectory;
            ASSERT_TRUE(temporaryDirectory.isValid());
            const QString imagePath = temporaryDirectory.filePath(QStringLiteral("source-image.bin"));
            QFile sourceFile(imagePath);
            ASSERT_TRUE(sourceFile.open(QIODevice::WriteOnly));
            ASSERT_EQ(sourceFile.write("source-v1"), 9);
            sourceFile.close();

            ImageMatchingRuntimeConfig config;
            config.maxKeypoints = 128;
            config.siftBackend = SiftComputeBackend::Cpu;
            PlaMatchHctAlgorithm algorithm(config);
            const cv::Mat image = texturedImage();
            ImageFeatureInput input;
            input.imagePath = imagePath;
            input.colorImage = image;
            input.originalWidth = image.cols;
            input.originalHeight = image.rows;
            const FeatureSet features = algorithm.extract(input);
            ASSERT_TRUE(features.isConsistent());

            const QString cachePath =
                PlaMatchHctFeatureCacheFile::filePathForImage(temporaryDirectory.path(), imagePath);
            const QString signature = QStringLiteral("unit-test-producer-v1");
            QString error;
            ASSERT_TRUE(PlaMatchHctFeatureCacheFile::write(cachePath, imagePath, signature, features, &error))
                << error.toStdString();

            const std::shared_ptr<FeatureSet> cached =
                PlaMatchHctFeatureCacheFile::read(cachePath, imagePath, signature, &error);
            ASSERT_NE(cached, nullptr) << error.toStdString();
            ASSERT_TRUE(cached->isConsistent());
            EXPECT_EQ(cached->imageWidth, features.imageWidth);
            EXPECT_EQ(cached->imageHeight, features.imageHeight);
            EXPECT_EQ(cached->computeBackend, features.computeBackend);
            EXPECT_EQ(cached->keypoints.size(), features.keypoints.size());
            EXPECT_EQ(cached->descriptors.rows, features.descriptors.rows);
            EXPECT_EQ(cached->descriptors.cols, features.descriptors.cols);
            EXPECT_EQ(cached->descriptors.type(), features.descriptors.type());
            EXPECT_EQ(cv::countNonZero(cached->descriptors != features.descriptors), 0);

            const auto cachedPayload = std::dynamic_pointer_cast<const PlaMatchHctFeaturePayload>(cached->payload);
            const auto sourcePayload = std::dynamic_pointer_cast<const PlaMatchHctFeaturePayload>(features.payload);
            ASSERT_NE(cachedPayload, nullptr);
            ASSERT_NE(sourcePayload, nullptr);
            EXPECT_EQ(cachedPayload->fullFeatures().global_descriptor, sourcePayload->fullFeatures().global_descriptor);
            EXPECT_EQ(cachedPayload->coarseFeatures().keypoints.size(),
                      sourcePayload->coarseFeatures().keypoints.size());

            ASSERT_TRUE(sourceFile.open(QIODevice::Append));
            ASSERT_EQ(sourceFile.write("-changed"), 8);
            sourceFile.close();
            EXPECT_EQ(PlaMatchHctFeatureCacheFile::read(cachePath, imagePath, signature, &error), nullptr);
            EXPECT_FALSE(error.isEmpty());
        }

        TEST(PlaMatchHctAccuracyTest, AllFiveLevelsExtractOnCpu)
        {
            const cv::Mat image = texturedImage();
            for (const int downscale : {0, 1, 2, 4, 8})
            {
                ImageMatchingRuntimeConfig config;
                config.maxKeypoints = 128;
                config.siftBackend = SiftComputeBackend::Cpu;
                config.alignmentDownscale = downscale;
                PlaMatchHctAlgorithm algorithm(config);

                ImageFeatureInput input;
                input.imagePath = QStringLiteral("synthetic.png");
                input.colorImage = image;
                input.originalWidth = image.cols;
                input.originalHeight = image.rows;

                const FeatureSet features = algorithm.extract(input);
                EXPECT_TRUE(features.isConsistent()) << "downscale=" << downscale;
                EXPECT_EQ(features.imageWidth, image.cols);
                EXPECT_EQ(features.imageHeight, image.rows);
            }
        }

        TEST(PlaMatchHctAccuracyTest, HighestExtractsOnResolvedAccelerator)
        {
            const PlaMatchHctBackendResolution resolution = resolvePlaMatchHctBackend(SiftComputeBackend::Automatic, 0);
            ASSERT_TRUE(resolution.valid);
            if (resolution.backend == SiftComputeBackend::Cpu)
            {
                GTEST_SKIP() << "No PlaMatch-HCT accelerator is available";
            }

            ImageMatchingRuntimeConfig config;
            config.maxKeypoints = 128;
            config.siftBackend = resolution.backend;
            config.alignmentDownscale = 0;
            PlaMatchHctAlgorithm algorithm(config);

            const cv::Mat image = texturedImage();
            ImageFeatureInput input;
            input.imagePath = QStringLiteral("synthetic.png");
            input.colorImage = image;
            input.originalWidth = image.cols;
            input.originalHeight = image.rows;

            const FeatureSet features = algorithm.extract(input);
            EXPECT_TRUE(features.isConsistent());
            EXPECT_FALSE(features.computeBackend.empty());
        }

        TEST(PlaMatchHctAlgorithmTest, AcceleratedBatchHonorsCancellation)
        {
            const PlaMatchHctBackendResolution resolution = resolvePlaMatchHctBackend(SiftComputeBackend::Automatic, 0);
            ASSERT_TRUE(resolution.valid);
            if (resolution.backend == SiftComputeBackend::Cpu)
            {
                GTEST_SKIP() << "No PlaMatch-HCT accelerator is available";
            }

            ImageMatchingRuntimeConfig config;
            config.siftBackend = resolution.backend;
            PlaMatchHctAlgorithm algorithm(config);
            EXPECT_THROW(algorithm.matchFeatureBatch(std::span<const FeaturePairInput>{}, []() { return true; }),
                         std::runtime_error);
        }

        TEST(PlaMatchHctAlgorithmTest, AcceleratedBatchReportsSubBatchProgress)
        {
            const PlaMatchHctBackendResolution resolution = resolvePlaMatchHctBackend(SiftComputeBackend::Automatic, 0);
            ASSERT_TRUE(resolution.valid);
            if (resolution.backend == SiftComputeBackend::Cpu)
            {
                GTEST_SKIP() << "No PlaMatch-HCT accelerator is available";
            }

            ImageMatchingRuntimeConfig config;
            config.maxKeypoints = 256;
            config.siftBackend = resolution.backend;
            PlaMatchHctAlgorithm algorithm(config);

            const cv::Mat image = texturedImage();
            ImageFeatureInput input;
            input.imagePath = QStringLiteral("synthetic.png");
            input.colorImage = image;
            input.originalWidth = image.cols;
            input.originalHeight = image.rows;
            const FeatureSet features = algorithm.extract(input);

            std::vector<FeaturePairInput> pairs(81, {&features, &features});
            std::vector<std::pair<std::size_t, std::size_t>> progress;
            const std::vector<MatchResult> matches = algorithm.matchFeatureBatch(
                pairs,
                {},
                [&progress](std::size_t completed, std::size_t total) { progress.emplace_back(completed, total); });

            ASSERT_EQ(matches.size(), pairs.size());
            ASSERT_EQ(progress.size(), 2U);
            EXPECT_EQ(progress[0].first, 80U);
            EXPECT_EQ(progress[0].second, 81U);
            EXPECT_EQ(progress[1].first, 81U);
            EXPECT_EQ(progress[1].second, 81U);
        }

    } // namespace
} // namespace xjw::image_matching
