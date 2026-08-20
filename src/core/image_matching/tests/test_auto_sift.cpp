#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftFeatureExtractor.h"
#include "sift/SiftGuidedMatcher.h"
#include "sift/SiftMatchFilter.h"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

namespace xjw::image_matching
{
    namespace
    {

        FeatureSet makeSiftFeatures(int count)
        {
            FeatureSet features;
            features.imageWidth = 1024;
            features.imageHeight = 1024;
            features.sourceAlgorithm = "sift";
            features.keypoints.reserve(static_cast<std::size_t>(count));
            features.scores.assign(static_cast<std::size_t>(count), 1.0f);
            features.descriptors = cv::Mat::zeros(count, 128, CV_32F);
            for (int index = 0; index < count; ++index)
            {
                features.keypoints.emplace_back(
                    static_cast<float>(index % 1024), static_cast<float>(index / 1024), 1.0f);
                float* descriptor = features.descriptors.ptr<float>(index);
                std::uint32_t state = static_cast<std::uint32_t>(index + 1);
                float squaredNorm = 0.0f;
                for (int dimension = 0; dimension < 128; ++dimension)
                {
                    state = state * 1664525U + 1013904223U;
                    descriptor[dimension] = static_cast<float>((state >> 8U) & 0xffffU) / 65535.0f;
                    squaredNorm += descriptor[dimension] * descriptor[dimension];
                }
                const float inverseNorm = 1.0f / std::sqrt(squaredNorm);
                for (int dimension = 0; dimension < 128; ++dimension)
                {
                    descriptor[dimension] *= inverseNorm;
                }
            }
            return features;
        }

        TEST(AutoSiftMatchFilterTest, KeepsOnlyMutualMatchesAboveConfidenceThreshold)
        {
            const std::vector<SiftNearestMatch> forward = {{1, 0.90f, 0.60f}, {0, 0.95f, 0.90f}, {2, 0.80f, 0.70f}};
            const std::vector<SiftNearestMatch> reverse = {{2, 0.90f, 0.60f}, {0, 0.85f, 0.50f}, {2, 0.90f, 0.75f}};

            const MatchResult result = filterSiftMutualMatches(forward, reverse, 0.15f);

            ASSERT_EQ(result.matches0.size(), 3U);
            EXPECT_EQ(result.matches0[0], 1);
            EXPECT_EQ(result.matches0[1], -1);
            EXPECT_EQ(result.matches0[2], 2);
            ASSERT_EQ(result.cvMatches.size(), 2U);
            EXPECT_EQ(result.sourceAlgorithm, "auto_sift");
        }

        TEST(AutoSiftMatchFilterTest, RejectsNonFiniteLowSimilarityAndExtremeAmbiguity)
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            const std::vector<SiftNearestMatch> forward = {{0, nan, 0.1f}, {1, 0.90f, 0.995f}, {2, 0.10f, 0.1f}};
            const std::vector<SiftNearestMatch> reverse = {{0, 0.90f, 0.1f}, {1, 0.90f, 0.995f}, {2, 0.10f, 0.1f}};

            const MatchResult result = filterSiftMutualMatches(forward, reverse, 0.15f);

            EXPECT_TRUE(result.cvMatches.empty());
        }

        TEST(AutoSiftAlgorithmTest, CpuBackendMatchesNormalizedDescriptors)
        {
            FeatureSet features0;
            features0.imageWidth = 100;
            features0.imageHeight = 100;
            features0.keypoints = {cv::KeyPoint(10.0f, 10.0f, 1.0f), cv::KeyPoint(20.0f, 20.0f, 1.0f)};
            features0.scores = {1.0f, 1.0f};
            features0.descriptors = cv::Mat::zeros(2, 128, CV_32F);
            features0.descriptors.at<float>(0, 0) = 512.0f;
            features0.descriptors.at<float>(1, 1) = 256.0f;

            FeatureSet features1;
            features1.imageWidth = features0.imageWidth;
            features1.imageHeight = features0.imageHeight;
            features1.keypoints = features0.keypoints;
            features1.scores = features0.scores;
            features1.descriptors = cv::Mat::zeros(2, 128, CV_32F);
            features1.descriptors.at<float>(0, 1) = 1.0f;
            features1.descriptors.at<float>(1, 0) = 1.0f;

            ImageMatchingRuntimeConfig config;
            config.siftBackend = SiftComputeBackend::Cpu;
            AutoSiftAlgorithm algorithm(config);

            const MatchResult result = algorithm.matchFeatures(features0, features1);

            ASSERT_EQ(result.cvMatches.size(), 2U);
            EXPECT_EQ(result.matches0[0], 1);
            EXPECT_EQ(result.matches0[1], 0);
        }

        TEST(AutoSiftGuidedMatcherTest, RecoversMutualDescriptorsInsideEpipolarBand)
        {
            FeatureSet features0;
            FeatureSet features1;
            features0.imageWidth = features1.imageWidth = 200;
            features0.imageHeight = features1.imageHeight = 100;
            features0.keypoints = {cv::KeyPoint(20.0f, 25.0f, 1.0f), cv::KeyPoint(40.0f, 60.0f, 1.0f)};
            features1.keypoints = {cv::KeyPoint(24.0f, 25.5f, 1.0f), cv::KeyPoint(45.0f, 60.5f, 1.0f)};
            features0.scores = features1.scores = {1.0f, 1.0f};
            features0.descriptors = cv::Mat::zeros(2, 128, CV_32F);
            features1.descriptors = cv::Mat::zeros(2, 128, CV_32F);
            features0.descriptors.at<float>(0, 0) = 1.0f;
            features0.descriptors.at<float>(1, 1) = 1.0f;
            features1.descriptors.at<float>(0, 0) = 1.0f;
            features1.descriptors.at<float>(1, 1) = 1.0f;
            const std::array<double, 9> fundamental{{0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0}};

            const auto matches = findGuidedSiftMatches(features0, features1, fundamental, {}, {});

            ASSERT_EQ(matches.size(), 2U);
            EXPECT_EQ(matches[0].index0, matches[0].index1);
            EXPECT_EQ(matches[1].index0, matches[1].index1);
        }

        TEST(AutoSiftFeatureExtractorTest, AdaptiveCpuPathUpscalesSmallImages)
        {
            cv::Mat image(240, 320, CV_8U, cv::Scalar(32));
            for (int y = 12; y < image.rows - 12; y += 16)
            {
                for (int x = 12; x < image.cols - 12; x += 16)
                {
                    cv::circle(
                        image, cv::Point(x, y), 4, cv::Scalar(((x + y) / 16) % 2 == 0 ? 230 : 90), -1, cv::LINE_AA);
                }
            }

            ImageFeatureInput input;
            input.grayImage = image;
            input.originalWidth = image.cols;
            input.originalHeight = image.rows;
            ImageMatchingRuntimeConfig config;
            config.siftBackend = SiftComputeBackend::Cpu;
            config.adaptiveSift = true;
            config.rootSift = true;
            config.maxKeypoints = 600;
            config.siftContrastThreshold = 0.02f;

            const FeatureSet features = SiftFeatureExtractor::extract(input, config);

            EXPECT_FALSE(features.empty());
            EXPECT_LE(features.size(), 600);
            EXPECT_EQ(features.descriptors.cols, 128);
            for (const cv::KeyPoint& keypoint : features.keypoints)
            {
                EXPECT_GE(keypoint.pt.x, 0.0f);
                EXPECT_LT(keypoint.pt.x, static_cast<float>(image.cols));
                EXPECT_GE(keypoint.pt.y, 0.0f);
                EXPECT_LT(keypoint.pt.y, static_cast<float>(image.rows));
            }
        }

        TEST(AutoSiftFeatureExtractorTest, LargeImagesMergeCoarseAndNativeTileCoordinates)
        {
            cv::Mat image(900, 1200, CV_8U, cv::Scalar(24));
            for (int y = 24; y < image.rows - 24; y += 48)
            {
                for (int x = 24; x < image.cols - 24; x += 48)
                {
                    cv::rectangle(image,
                                  cv::Rect(x - 8, y - 8, 16, 16),
                                  cv::Scalar(((x + y) / 48) % 2 == 0 ? 225 : 96),
                                  -1,
                                  cv::LINE_AA);
                }
            }

            ImageFeatureInput input;
            input.grayImage = image;
            input.originalWidth = image.cols;
            input.originalHeight = image.rows;
            ImageMatchingRuntimeConfig config;
            config.siftBackend = SiftComputeBackend::Cpu;
            config.adaptiveSift = true;
            config.rootSift = true;
            config.maxImageDimension = 512;
            config.maxKeypoints = 300;

            const FeatureSet features = SiftFeatureExtractor::extract(input, config);

            EXPECT_FALSE(features.empty());
            EXPECT_LE(features.size(), 300);
            EXPECT_TRUE(std::any_of(features.keypoints.cbegin(),
                                    features.keypoints.cend(),
                                    [](const cv::KeyPoint& keypoint) { return keypoint.pt.x > 900.0f; }));
        }

        TEST(AutoSiftAlgorithmTest, CudaBackendHandlesPartialTiles)
        {
            if (!SiftFeatureExtractor::isBackendAvailable(SiftComputeBackend::Cuda, 0))
            {
                GTEST_SKIP() << "CUDA SIFT device is unavailable";
            }

            ImageMatchingRuntimeConfig config;
            config.cudaDevice = 0;
            config.siftBackend = SiftComputeBackend::Cuda;
            AutoSiftAlgorithm algorithm(config);

            const MatchResult result = algorithm.matchFeatures(makeSiftFeatures(607), makeSiftFeatures(531));

            EXPECT_EQ(result.matches0.size(), 607U);
            EXPECT_EQ(result.matchingScores0.size(), 607U);
            EXPECT_EQ(result.numMatches, 531);
        }

        void expectGpuExtractionWorks(SiftComputeBackend backend)
        {
            if (!SiftFeatureExtractor::isBackendAvailable(backend, 0))
            {
                GTEST_SKIP() << siftBackendName(backend) << " SIFT device is unavailable";
            }
            cv::Mat image(256, 320, CV_8U, cv::Scalar(24));
            for (int y = 20; y < image.rows - 20; y += 24)
            {
                for (int x = 20; x < image.cols - 20; x += 24)
                {
                    cv::circle(
                        image, cv::Point(x, y), 6, cv::Scalar(((x + y) / 24) % 2 == 0 ? 235 : 80), -1, cv::LINE_AA);
                }
            }

            ImageFeatureInput input;
            input.grayImage = image;
            input.originalWidth = image.cols;
            input.originalHeight = image.rows;
            ImageMatchingRuntimeConfig config;
            config.siftBackend = backend;
            config.maxKeypoints = 500;
            config.siftContrastThreshold = 0.01f;

            const FeatureSet features = SiftFeatureExtractor::extract(input, config);

            ASSERT_FALSE(features.empty());
            EXPECT_EQ(features.computeBackend, siftBackendName(backend));
            EXPECT_EQ(features.descriptors.type(), CV_32F);
            EXPECT_EQ(features.descriptors.cols, 128);
            EXPECT_TRUE(cv::checkRange(features.descriptors));
        }

        TEST(AutoSiftFeatureExtractorTest, MetalBuildsPyramidAndDescriptorsOnGpu)
        {
            expectGpuExtractionWorks(SiftComputeBackend::Metal);
        }

        TEST(AutoSiftFeatureExtractorTest, OpenClBuildsPyramidAndDescriptorsOnGpu)
        {
            expectGpuExtractionWorks(SiftComputeBackend::OpenCl);
        }

        TEST(AutoSiftBackendTest, AutomaticUsesMetalWhenCudaIsUnavailable)
        {
            if (SiftFeatureExtractor::isBackendAvailable(SiftComputeBackend::Cuda, 0) ||
                !SiftFeatureExtractor::isBackendAvailable(SiftComputeBackend::Metal, 0))
            {
                GTEST_SKIP() << "Test requires Metal without CUDA";
            }
            EXPECT_EQ(SiftFeatureExtractor::resolveBackend(SiftComputeBackend::Automatic, 0),
                      SiftComputeBackend::Metal);
        }

        TEST(AutoSiftBackendTest, ExplicitUnavailableDeviceDoesNotChangeBackend)
        {
            EXPECT_THROW(SiftFeatureExtractor::resolveBackend(SiftComputeBackend::Metal, 99), std::runtime_error);
        }

        TEST(AutoSiftAlgorithmTest, MetalMatchesDescriptorsOnGpu)
        {
            if (!SiftFeatureExtractor::isBackendAvailable(SiftComputeBackend::Metal, 0))
            {
                GTEST_SKIP() << "Metal SIFT device is unavailable";
            }
            ImageMatchingRuntimeConfig config;
            config.siftBackend = SiftComputeBackend::Metal;
            AutoSiftAlgorithm algorithm(config);

            const MatchResult result = algorithm.matchFeatures(makeSiftFeatures(67), makeSiftFeatures(53));

            EXPECT_EQ(result.matches0.size(), 67U);
            EXPECT_EQ(result.numMatches, 53);
        }

        TEST(AutoSiftAlgorithmTest, OpenClMatchesDescriptorsOnGpu)
        {
            if (!SiftFeatureExtractor::isBackendAvailable(SiftComputeBackend::OpenCl, 0))
            {
                GTEST_SKIP() << "OpenCL SIFT device is unavailable";
            }
            ImageMatchingRuntimeConfig config;
            config.siftBackend = SiftComputeBackend::OpenCl;
            AutoSiftAlgorithm algorithm(config);

            const MatchResult result = algorithm.matchFeatures(makeSiftFeatures(67), makeSiftFeatures(53));

            EXPECT_EQ(result.matches0.size(), 67U);
            EXPECT_EQ(result.numMatches, 53);
        }

    } // namespace
} // namespace xjw::image_matching
