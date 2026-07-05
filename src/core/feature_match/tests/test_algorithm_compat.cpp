#include <gtest/gtest.h>
#include "AlgorithmCompat.h"

using namespace xjw::feature_match;

TEST(AlgorithmCompatTest, SuperGlueOnlySupportsSP)
{
    auto suffixes = compatibleFeatureSuffixes("superglue");
    ASSERT_EQ(suffixes.size(), 1);
    EXPECT_EQ(suffixes[0], ".sp");
}

TEST(AlgorithmCompatTest, LightGlueSupportsMultiple)
{
    auto suffixes = compatibleFeatureSuffixes("lightglue");
    EXPECT_GE(suffixes.size(), 4);
    EXPECT_TRUE(suffixes.contains(".sp"));
    EXPECT_TRUE(suffixes.contains(".dsk"));
    EXPECT_TRUE(suffixes.contains(".alk"));
    EXPECT_TRUE(suffixes.contains(".sift"));
}

TEST(AlgorithmCompatTest, DefaultMatcherForFeatureSuffix)
{
    EXPECT_EQ(defaultMatcherForFeatureSuffix("image.sp"), "superglue");
    EXPECT_EQ(defaultMatcherForFeatureSuffix("image.dsk"), "lightglue");
    EXPECT_EQ(defaultMatcherForFeatureSuffix("image.alk"), "lightglue");
    EXPECT_EQ(defaultMatcherForFeatureSuffix("image.sift"), "lightglue");
    EXPECT_EQ(defaultMatcherForFeatureSuffix("image.orb"), "orb_bf_hamming");
    EXPECT_EQ(defaultMatcherForFeatureSuffix("image.akz"), "orb_bf_hamming");
    EXPECT_EQ(defaultMatcherForFeatureSuffix("image.surf"), "sift_bf_l2");
    EXPECT_EQ(defaultMatcherForFeatureSuffix("image.dedode"), "dedode");
}

TEST(AlgorithmCompatTest, EndToEndHasNoFeatures)
{
    for (const auto &algo : {"loftr", "roma"})
    {
        auto suffixes = compatibleFeatureSuffixes(algo);
        EXPECT_TRUE(suffixes.isEmpty()) << algo << " should have no feature deps";
    }
}

TEST(AlgorithmCompatTest, BfHammingOnlyOrb)
{
    auto suffixes = compatibleFeatureSuffixes("orb_bf_hamming");
    ASSERT_EQ(suffixes.size(), 2);
    EXPECT_TRUE(suffixes.contains(".orb"));
    EXPECT_TRUE(suffixes.contains(".akz"));
}

TEST(AlgorithmCompatTest, BfL2AndFlannOnlySift)
{
    for (const auto &algo : {"sift_bf_l2", "sift_flann"})
    {
        auto suffixes = compatibleFeatureSuffixes(algo);
        ASSERT_EQ(suffixes.size(), 2) << algo;
        EXPECT_TRUE(suffixes.contains(".sift")) << algo;
        EXPECT_TRUE(suffixes.contains(".surf")) << algo;
    }
}

TEST(AlgorithmCompatTest, DedodeUsesDedodeFeatures)
{
    auto suffixes = compatibleFeatureSuffixes("dedode");
    ASSERT_EQ(suffixes.size(), 1);
    EXPECT_EQ(suffixes[0], ".dedode");
}

TEST(AlgorithmCompatTest, IsEndToEnd)
{
    EXPECT_TRUE(isEndToEndAlgorithm("loftr"));
    EXPECT_TRUE(isEndToEndAlgorithm("roma"));
    EXPECT_FALSE(isEndToEndAlgorithm("dedode"));
    EXPECT_FALSE(isEndToEndAlgorithm("superglue"));
    EXPECT_FALSE(isEndToEndAlgorithm("orb_bf_hamming"));
}

TEST(AlgorithmCompatTest, AlgorithmDisplayName)
{
    EXPECT_FALSE(algorithmDisplayName("superglue").isEmpty());
    EXPECT_FALSE(algorithmDisplayName("loftr").isEmpty());
    EXPECT_FALSE(algorithmDisplayName("dedode").isEmpty());
    EXPECT_FALSE(algorithmDisplayName("orb_bf_hamming").isEmpty());
}
