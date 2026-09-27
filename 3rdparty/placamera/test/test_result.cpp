#include <placamera/result.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <type_traits>

static_assert(!std::is_default_constructible_v<placamera::Result<int>>);
static_assert(!std::is_default_constructible_v<placamera::Result<void>>);
static_assert(!std::is_default_constructible_v<placamera::EvaluationResult<int>>);

TEST(PlaCameraResult, CarriesStructuredDiagnostics)
{
    const auto failed =
        placamera::Result<int>::failure(placamera::CameraErrorCode::ParseFailure, "invalid field", "camera.txt", 17);

    EXPECT_FALSE(failed);
    EXPECT_EQ(failed.errorCode(), placamera::CameraErrorCode::ParseFailure);
    EXPECT_EQ(failed.message(), "invalid field");
    EXPECT_EQ(failed.error().source, "camera.txt");
    ASSERT_TRUE(failed.error().line.has_value());
    EXPECT_EQ(*failed.error().line, 17u);
}

TEST(PlaCameraResult, MovesOwnedValuesWithoutExtraPointerBoilerplate)
{
    auto result = placamera::Result<std::unique_ptr<std::string>>::success(std::make_unique<std::string>("camera"));
    ASSERT_TRUE(result);
    EXPECT_EQ(**result, "camera");

    auto value = result.takeValue();
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, "camera");
}

TEST(PlaCameraResult, UsesTheSameErrorViewForEvaluationResults)
{
    const auto failed = placamera::EvaluationResult<double>::failure(
        placamera::CameraError{placamera::CameraErrorCode::NonConvergence, "iteration limit", "rpc", std::nullopt});

    EXPECT_FALSE(failed);
    EXPECT_EQ(failed.errorCode(), placamera::CameraErrorCode::NonConvergence);
    EXPECT_EQ(failed.error().source, "rpc");
}

TEST(PlaCameraResult, RepresentsSuccessfulVoidOperations)
{
    EXPECT_TRUE(placamera::Result<void>::success());
    EXPECT_FALSE(placamera::Result<void>::failure(placamera::CameraErrorCode::IoFailure, "write failed"));
}
