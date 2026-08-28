#include "project/services/ProjectSessionFacade.h"

#include <gtest/gtest.h>

namespace
{

using xjw::gui::project::ProjectSessionFacade;

TEST(ProjectSessionFacadeTest, NullSessionReturnsEmptyReadModels)
{
    const ProjectSessionFacade facade;

    EXPECT_FALSE(facade.isDirty());
    EXPECT_TRUE(facade.projectPath().isEmpty());
    EXPECT_TRUE(facade.activeChunkId().isEmpty());
    EXPECT_TRUE(facade.metadata().isEmpty());
    EXPECT_TRUE(facade.coreMetadata().isEmpty());
    EXPECT_TRUE(facade.allImages().isEmpty());
    EXPECT_TRUE(facade.intersectionResults().isEmpty());
}

TEST(ProjectSessionFacadeTest, NullSessionExplainsCameraMutationFailure)
{
    const ProjectSessionFacade facade;
    int updatedCount = 7;
    QString errorMessage;

    EXPECT_FALSE(facade.setImageCameras({}, &updatedCount, &errorMessage));
    EXPECT_EQ(updatedCount, 0);
    EXPECT_EQ(errorMessage, QStringLiteral("ProjectData 未初始化"));
}

TEST(ProjectSessionFacadeTest, NullSessionResetsBothReplaceCounts)
{
    const ProjectSessionFacade facade;
    int updatedCount = 7;
    int clearedCount = 9;
    QString errorMessage;

    EXPECT_FALSE(facade.replaceImageCameras(
        {}, {}, &updatedCount, &clearedCount, &errorMessage));
    EXPECT_EQ(updatedCount, 0);
    EXPECT_EQ(clearedCount, 0);
    EXPECT_EQ(errorMessage, QStringLiteral("ProjectData 未初始化"));
}

} // namespace
