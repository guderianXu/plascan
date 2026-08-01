#include "MatchPhotosRuntime.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

TEST(MatchPhotosRuntimeTest, DefaultsToAutomaticTensorRtEngineLookup)
{
    const xjw::matchphotos::MatchPhotosOptions options;

    EXPECT_TRUE(options.lightGlueTensorRtEnginePath.isEmpty());
}

TEST(MatchPhotosRuntimeTest, ResolvesExplicitTensorRtEngine)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    const QString enginePath = directory.filePath(QStringLiteral("lightglue.engine"));
    QFile engineFile(enginePath);
    ASSERT_TRUE(engineFile.open(QIODevice::WriteOnly));
    ASSERT_EQ(engineFile.write("test-engine"), 11);
    engineFile.close();

    xjw::matchphotos::MatchPhotosOptions options;
    options.lightGlueTensorRtEnginePath = enginePath;
    QString engineName;

    const QString resolved =
        xjw::matchphotos::resolveLightGlueTensorRtEnginePath(options, &engineName);

    EXPECT_EQ(resolved, QDir::cleanPath(QFileInfo(enginePath).absoluteFilePath()));
    EXPECT_EQ(engineName, QStringLiteral("lightglue.engine"));
}

TEST(MatchPhotosRuntimeTest, RejectsMissingExplicitTensorRtEngine)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.lightGlueTensorRtEnginePath = QStringLiteral("missing-lightglue.engine");
    QString engineName = QStringLiteral("stale");

    const QString resolved =
        xjw::matchphotos::resolveLightGlueTensorRtEnginePath(options, &engineName);

    EXPECT_TRUE(resolved.isEmpty());
    EXPECT_TRUE(engineName.isEmpty());
}
