#include "ProjectSessionContext.h"
#include "ProjectTerrainRequests.h"

#include <gtest/gtest.h>

#include <QDir>

namespace
{

using xjw::gui::project::ProjectSessionContext;
using xjw::gui::project::DemGenerationRequest;
using xjw::gui::project::DemGenerationMode;

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

TEST(DemGenerationRequestTest, RequiresExistingSourceFieldsAtBoundary)
{
    DemGenerationRequest request;
    QString error;

    EXPECT_FALSE(request.validate(&error));
    EXPECT_FALSE(error.isEmpty());

    request.sourcePointCloudPath = QStringLiteral("cloud.ply");
    request.outputDirectory = QStringLiteral("dem");
    request.resolution = -1.0;
    EXPECT_FALSE(request.validate(&error));

    request.resolution = 0.5;
    request.dataType = QStringLiteral("float32");
    EXPECT_TRUE(request.validate(&error));
    EXPECT_TRUE(error.isEmpty());
}

TEST(DemGenerationRequestTest, ValidatesImageStereoProductMode)
{
    DemGenerationRequest request;
    request.mode = DemGenerationMode::ImageStereo;
    request.imageStereoOptions.sourceImages = {QStringLiteral("left.tif"), QStringLiteral("right.tif")};
    QString error;

    EXPECT_TRUE(request.validate(&error));
    request.imageStereoOptions.sourceImages = {QStringLiteral("left.tif"), QStringLiteral("left.tif")};
    EXPECT_FALSE(request.validate(&error));
    EXPECT_FALSE(error.isEmpty());
}

} // namespace
