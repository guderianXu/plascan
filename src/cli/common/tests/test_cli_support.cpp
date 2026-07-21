#include "CliJsonIO.h"
#include "CliOutputPolicy.h"
#include "CliPathUtils.h"
#include "CliTokenUtils.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonObject>
#include <QTemporaryDir>

namespace
{

TEST(CliSupportTest, NormalizesTokensAndUtf8Strings)
{
    EXPECT_EQ(xjw::cli::normalizedToken(std::string(" High-Accuracy ")),
              QStringLiteral("high_accuracy"));
    EXPECT_EQ(xjw::cli::normalizedToken(std::string(), QStringLiteral("auto")),
              QStringLiteral("auto"));
    EXPECT_EQ(xjw::cli::fromStdString(u8"影像"), QStringLiteral("影像"));
}

TEST(CliSupportTest, ReadsAndWritesJsonObjects)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = QDir(directory.path()).filePath(QStringLiteral("nested/report.json"));
    const QJsonObject expected{{QStringLiteral("status"), QStringLiteral("ok")}};
    QString error;

    ASSERT_TRUE(xjw::cli::writeJsonFile(path, expected, &error)) << qUtf8Printable(error);
    QJsonObject actual;
    ASSERT_TRUE(xjw::cli::readJsonFile(path, &actual, &error)) << qUtf8Printable(error);
    EXPECT_EQ(actual, expected);
}

TEST(CliSupportTest, ProtectsNonEmptyOutputDirectoriesUnlessForced)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    QFile marker(QDir(directory.path()).filePath(QStringLiteral("report.json")));
    ASSERT_TRUE(marker.open(QIODevice::WriteOnly));
    marker.close();

    QString error;
    EXPECT_FALSE(xjw::cli::validateOutputDirectory(directory.path(), false, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("拒绝覆盖")));
    EXPECT_TRUE(xjw::cli::validateOutputDirectory(directory.path(), true, &error));
}

} // namespace
