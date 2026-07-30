#include <gtest/gtest.h>

#include "ProjectFileIntegration.h"

#include <QDir>
#include <QFileInfo>

using xjw::gui::platform::projectOpenCommand;
using xjw::gui::platform::startupProjectPath;

TEST(ProjectFileIntegrationTest, FindsUnicodeProjectArgument)
{
    const QString project_path =
        QDir::temp().absoluteFilePath(QStringLiteral("行星工程/测试项目.plascan"));
    const QString actual = startupProjectPath(
        {QStringLiteral("plascan.exe"), QStringLiteral("--ignored-option"), project_path});

    EXPECT_EQ(QDir::cleanPath(QFileInfo(project_path).absoluteFilePath()), actual);
}

TEST(ProjectFileIntegrationTest, AcceptsProjectExtensionCaseInsensitively)
{
    const QString project_path = QDir::temp().absoluteFilePath(QStringLiteral("sample.PlaScan"));

    EXPECT_EQ(QDir::cleanPath(QFileInfo(project_path).absoluteFilePath()),
              startupProjectPath({QStringLiteral("plascan.exe"), project_path}));
}

TEST(ProjectFileIntegrationTest, IgnoresUnrelatedArguments)
{
    EXPECT_TRUE(startupProjectPath(
        {QStringLiteral("plascan.exe"), QStringLiteral("--safe-mode"), QStringLiteral("image.tif")}).isEmpty());
}

TEST(ProjectFileIntegrationTest, ProducesQuotedShellOpenCommand)
{
    const QString executable = QDir::temp().absoluteFilePath(QStringLiteral("PlaScan App/plascan.exe"));
    const QString expected_executable =
        QDir::toNativeSeparators(QFileInfo(executable).absoluteFilePath());

    EXPECT_EQ(QStringLiteral("\"%1\" \"%2\"").arg(expected_executable, QStringLiteral("%1")),
              projectOpenCommand(executable));
}
