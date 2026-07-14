#include <vector>

#include <gtest/gtest.h>

#include "string_utils/StringParsing.h"

namespace
{

TEST(CommonStringParsingTest, ParsesMatrixLine)
{
    std::vector<double> values;

    ASSERT_TRUE(xjw::common::string_utils::extractDoublesFromText(
        "R = 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0", values));

    const std::vector<double> expected{1.0, 0.0, 0.0,
                                       0.0, 1.0, 0.0,
                                       0.0, 0.0, 1.0};
    EXPECT_EQ(values, expected);
}

TEST(CommonStringParsingTest, ParsesSupportedNumericForms)
{
    std::vector<double> values;

    ASSERT_TRUE(xjw::common::string_utils::extractDoublesFromText(
        "values = -2 +3.5 .25 6. 1e-3 -2E+2", values));

    const std::vector<double> expected{-2.0, 3.5, 0.25, 6.0, 0.001, -200.0};
    EXPECT_EQ(values, expected);
}

TEST(CommonStringParsingTest, ClearsOutputWhenNoNumberExists)
{
    std::vector<double> values{42.0};

    EXPECT_FALSE(xjw::common::string_utils::extractDoublesFromText("R = none", values));
    EXPECT_TRUE(values.empty());
}

TEST(CommonStringParsingTest, ReplacesPreexistingOutputWhenNumbersAreExtracted)
{
    std::vector<double> values{42.0, 84.0};

    ASSERT_TRUE(xjw::common::string_utils::extractDoublesFromText("value = 2.5", values));
    const std::vector<double> expected{2.5};
    EXPECT_EQ(values, expected);
}

TEST(CommonStringParsingTest, IgnoresOutOfRangeMatchAndKeepsValidValues)
{
    std::vector<double> values;

    ASSERT_TRUE(xjw::common::string_utils::extractDoublesFromText("1e999 2.5", values));
    ASSERT_EQ(values.size(), 1U);
    EXPECT_DOUBLE_EQ(values.front(), 2.5);
}

TEST(CommonStringParsingTest, ExtractsDigitEmbeddedInIdentifier)
{
    std::vector<double> values;

    ASSERT_TRUE(xjw::common::string_utils::extractDoublesFromText("k1 = 0.25", values));
    const std::vector<double> expected{1.0, 0.25};
    EXPECT_EQ(values, expected);
}

} // namespace
