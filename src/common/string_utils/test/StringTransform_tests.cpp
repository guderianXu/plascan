#include <string>

#include <gtest/gtest.h>

#include "string_utils/StringTransform.h"

namespace
{

using xjw::common::string_utils::asciiLowerCopy;
using xjw::common::string_utils::endsWithAsciiIgnoreCase;
using xjw::common::string_utils::trimAsciiWhitespace;

TEST(CommonStringTransformTest, LowercasesOnlyAsciiUppercaseLetters)
{
    std::string input;
    input.push_back(static_cast<char>(0xC3));
    input.push_back(static_cast<char>(0x84));
    input += "AZaz-9";

    std::string expected;
    expected.push_back(static_cast<char>(0xC3));
    expected.push_back(static_cast<char>(0x84));
    expected += "azaz-9";
    EXPECT_EQ(asciiLowerCopy(input), expected);
}

TEST(CommonStringTransformTest, TrimsAllAsciiWhitespaceAndPreservesInterior)
{
    EXPECT_EQ(trimAsciiWhitespace("\t\n\r\f\v alpha  beta \t\n\r\f\v"), "alpha  beta");
}

TEST(CommonStringTransformTest, HandlesEmptyAndAllWhitespaceInputs)
{
    EXPECT_EQ(trimAsciiWhitespace(" \t\n\r\f\v"), "");
    EXPECT_EQ(trimAsciiWhitespace(""), "");
}

TEST(CommonStringTransformTest, PreservesHighBytesAtTextBoundaries)
{
    std::string input;
    input.push_back(static_cast<char>(0xC3));
    input += " alpha ";
    input.push_back(static_cast<char>(0x84));

    EXPECT_EQ(trimAsciiWhitespace(input), input);
}

TEST(CommonStringTransformTest, MatchesSuffixIgnoringAsciiCase)
{
    EXPECT_TRUE(endsWithAsciiIgnoreCase("cloud.PLY", ".ply"));
    EXPECT_TRUE(endsWithAsciiIgnoreCase("cloud.ply", ".PLY"));
    EXPECT_TRUE(endsWithAsciiIgnoreCase("cloud.ply", ""));
    EXPECT_FALSE(endsWithAsciiIgnoreCase("ply", ".ply"));
    EXPECT_FALSE(endsWithAsciiIgnoreCase("cloud.xyz", ".ply"));
}

TEST(CommonStringTransformTest, ComparesHighByteSuffixesExactly)
{
    std::string text{"cloud"};
    text.push_back(static_cast<char>(0xC3));
    text.push_back(static_cast<char>(0x84));

    std::string matching_suffix;
    matching_suffix.push_back(static_cast<char>(0xC3));
    matching_suffix.push_back(static_cast<char>(0x84));

    std::string different_suffix;
    different_suffix.push_back(static_cast<char>(0xC3));
    different_suffix.push_back(static_cast<char>(0x85));

    EXPECT_TRUE(endsWithAsciiIgnoreCase(text, matching_suffix));
    EXPECT_FALSE(endsWithAsciiIgnoreCase(text, different_suffix));
}

} // namespace
