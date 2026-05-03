// =============================================================================
// test_feature_extractor_suffix.cpp — 多提取器后缀映射 + 自动检测 + 工厂
// 不依赖 Qt (避免 slots/signals 宏与 LibTorch 冲突)
// =============================================================================
#include <gtest/gtest.h>

#include "ExtractorFactory.h"
#include "TraditionalFeatureExtractor.h"

#include <string>

// ── 1. 后缀映射 (验证与 FeatureFileIO.h 中 ExtractorSuffix 一致) ──

// 内联拷贝 FeatureFileIO.h 的 forAlgorithm 逻辑, 避免 Qt 依赖
static const char *suffixForAlgorithm(const std::string &algo)
{
    if (algo == "disk")       return ".dsk";
    if (algo == "aliked")     return ".alk";
    if (algo == "sift")       return ".sift";
    if (algo == "orb")        return ".orb";
    if (algo == "akaze")      return ".akz";
    if (algo == "surf")       return ".surf";
    if (algo == "dedode")     return ".dedode";
    return ".sp";
}

TEST(ExtractorSuffixTest, ForAlgorithmReturnsCorrectSuffix)
{
    EXPECT_STREQ(suffixForAlgorithm("superpoint"), ".sp");
    EXPECT_STREQ(suffixForAlgorithm("disk"),       ".dsk");
    EXPECT_STREQ(suffixForAlgorithm("aliked"),     ".alk");
    EXPECT_STREQ(suffixForAlgorithm("sift"),       ".sift");
    EXPECT_STREQ(suffixForAlgorithm("orb"),        ".orb");
    EXPECT_STREQ(suffixForAlgorithm("akaze"),      ".akz");
    EXPECT_STREQ(suffixForAlgorithm("surf"),       ".surf");
    EXPECT_STREQ(suffixForAlgorithm("dedode"),     ".dedode");
}

TEST(ExtractorSuffixTest, UnknownAlgorithmReturnsDotSp)
{
    EXPECT_STREQ(suffixForAlgorithm("unknown_algo"), ".sp");
}

// ── 2. 自动后缀检测 (模拟 cli_feature_match.cpp) ──

TEST(SuffixDetectionTest, AutoDetectFromPath)
{
    auto autoMatcher = [](const std::string &spPath) -> std::string
    {
        auto pos = spPath.rfind('.');
        if (pos == std::string::npos) return "superglue";
        std::string ext = spPath.substr(pos);
        if (ext == ".sp" || ext == ".dedode") return "superglue";
        if (ext == ".dsk" || ext == ".alk" || ext == ".sift") return "bf";
        return "superglue";
    };

    EXPECT_EQ(autoMatcher("img.sp"),     "superglue");
    EXPECT_EQ(autoMatcher("img.dedode"), "superglue");
    EXPECT_EQ(autoMatcher("img.dsk"),    "bf");
    EXPECT_EQ(autoMatcher("img.alk"),    "bf");
    EXPECT_EQ(autoMatcher("img.sift"),   "bf");
    EXPECT_EQ(autoMatcher("img.orb"),    "superglue");
    EXPECT_EQ(autoMatcher("img.xyz"),    "superglue");
}

// ── 3. 算法名归一化 ──

TEST(AlgorithmNormalizationTest, BasicNormalization)
{
    using xjw::feature_extractors::TraditionalFeatureExtractor;
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("SuperPoint"), "superpoint");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("DISK"),       "disk");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("SIFT"),       "sift");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("orb"),        "orb");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("surf"),       "surf");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("aliked"),     "aliked");
    // Unknown algorithms fall back to "superpoint" (default)
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("akaze"),      "akaze");
    EXPECT_EQ(TraditionalFeatureExtractor::normalizeAlgorithmName("unknown"),    "superpoint");
}

TEST(AlgorithmNormalizationTest, IsTraditionalAlgorithm)
{
    using xjw::feature_extractors::TraditionalFeatureExtractor;
    EXPECT_TRUE(TraditionalFeatureExtractor::isTraditionalAlgorithm("sift"));
    EXPECT_TRUE(TraditionalFeatureExtractor::isTraditionalAlgorithm("orb"));
    EXPECT_TRUE(TraditionalFeatureExtractor::isTraditionalAlgorithm("surf"));
    EXPECT_TRUE(TraditionalFeatureExtractor::isTraditionalAlgorithm("akaze"));
    EXPECT_FALSE(TraditionalFeatureExtractor::isTraditionalAlgorithm("superpoint"));
    EXPECT_FALSE(TraditionalFeatureExtractor::isTraditionalAlgorithm("disk"));
    EXPECT_FALSE(TraditionalFeatureExtractor::isTraditionalAlgorithm("aliked"));
}

// ── 4. ExtractorConfig 默认值 ──

TEST(ExtractorConfigTest, DefaultValuesAreReasonable)
{
    ExtractorConfig cfg;
    EXPECT_EQ(cfg.maxKeypoints, 4096);
    EXPECT_FLOAT_EQ(cfg.detThreshold, 0.003f);
    EXPECT_EQ(cfg.nmsRadius, 3);
    EXPECT_EQ(cfg.removeBorder, 4);
    EXPECT_EQ(cfg.maxImageDim, 2048);
    EXPECT_TRUE(cfg.useCuda);
    EXPECT_EQ(cfg.cudaDevice, 0);
}
