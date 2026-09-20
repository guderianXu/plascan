#include "project/services/ProjectSession.h"

#include <gtest/gtest.h>

namespace
{

    using xjw::gui::project::ProjectSession;

    TEST(ProjectSessionNullTest, ReturnsEmptyReadModels)
    {
        ProjectSession session(nullptr);

        EXPECT_FALSE(session.isDirty());
        EXPECT_TRUE(session.projectPath().isEmpty());
        EXPECT_TRUE(session.activeChunkId().isEmpty());
        EXPECT_TRUE(session.metadata().isEmpty());
        EXPECT_TRUE(session.coreMetadata().isEmpty());
        EXPECT_TRUE(session.allImages().isEmpty());
        EXPECT_TRUE(session.intersectionResults().isEmpty());
    }

    TEST(ProjectSessionNullTest, ExplainsCameraMutationFailure)
    {
        ProjectSession session(nullptr);
        int updatedCount = 7;
        QString errorMessage;

        EXPECT_FALSE(session.setCameraInstances({}, &updatedCount, &errorMessage));
        EXPECT_EQ(updatedCount, 0);
        EXPECT_EQ(errorMessage, QStringLiteral("ProjectData 未初始化"));
    }

    TEST(ProjectSessionNullTest, ResetsBothReplaceCounts)
    {
        ProjectSession session(nullptr);
        int updatedCount = 7;
        int clearedCount = 9;
        QString errorMessage;

        EXPECT_FALSE(session.replaceCameraInstances({}, {}, &updatedCount, &clearedCount, &errorMessage));
        EXPECT_EQ(updatedCount, 0);
        EXPECT_EQ(clearedCount, 0);
        EXPECT_EQ(errorMessage, QStringLiteral("ProjectData 未初始化"));
    }

} // namespace
