#include "ProjectSessionContext.h"

#include <gtest/gtest.h>

#include <QDir>

namespace
{

using xjw::gui::project::ProjectSessionContext;

TEST(ProjectSessionContextTest, MatchesEquivalentProjectPaths)
{
    const QString project_path = QDir::temp().filePath(QStringLiteral("plascan/session/project.plascan"));
    const QString equivalent_path = QDir::cleanPath(
        QDir::temp().filePath(QStringLiteral("plascan/session/../session/project.plascan")));

    const ProjectSessionContext left{project_path, QStringLiteral("chunk-1"), 7};
    const ProjectSessionContext right{equivalent_path, QStringLiteral("chunk-1"), 7};

    EXPECT_TRUE(left.matches(right));
    EXPECT_TRUE(right.matches(left));
}

TEST(ProjectSessionContextTest, RejectsDifferentChunkOrGeneration)
{
    const QString project_path = QDir::temp().filePath(QStringLiteral("project.plascan"));
    const ProjectSessionContext current{project_path, QStringLiteral("chunk-1"), 7};

    EXPECT_FALSE(current.matches({project_path, QStringLiteral("chunk-2"), 7}));
    EXPECT_FALSE(current.matches({project_path, QStringLiteral("chunk-1"), 8}));
}

TEST(ProjectSessionContextTest, RejectsDifferentProject)
{
    const ProjectSessionContext current{
        QDir::temp().filePath(QStringLiteral("first.plascan")),
        QStringLiteral("chunk-1"),
        7
    };
    const ProjectSessionContext other{
        QDir::temp().filePath(QStringLiteral("second.plascan")),
        QStringLiteral("chunk-1"),
        7
    };

    EXPECT_FALSE(current.matches(other));
}

} // namespace
