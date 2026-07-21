#include "preparation/TiePointPreparation.h"

#include <gtest/gtest.h>

TEST(TiePointPreparationTest, DelegatesExactlyOnceToMatchPhotosRunner)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.maxKeypoints = 1234;
    xjw::matchphotos::MatchPhotosContext context;
    context.projectPath = QStringLiteral("sample.plascan");

    int call_count = 0;
    const auto result = xjw::aerial_triangulation::TiePointPreparation::run(
        options,
        context,
        [&](const xjw::matchphotos::MatchPhotosOptions &actual_options,
            const xjw::matchphotos::MatchPhotosContext &actual_context)
        {
            ++call_count;
            EXPECT_EQ(actual_options.maxKeypoints, 1234);
            EXPECT_EQ(actual_context.projectPath, QStringLiteral("sample.plascan"));
            xjw::matchphotos::MatchPhotosResult prepared;
            prepared.success = true;
            prepared.trackCount = 42;
            return prepared;
        });

    EXPECT_EQ(call_count, 1);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.trackCount, 42);
}
