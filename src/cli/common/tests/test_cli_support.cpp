#include "CliJsonIO.h"
#include "CliOutputPolicy.h"
#include "CliPathUtils.h"
#include "CliTokenUtils.h"
#include "cli_common.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

namespace
{

TEST(CliSupportTest, ConfiguresConsistentHelpAndExitCodeGuidance)
{
    CLI::App app{"测试命令"};
    cli::configureApp(app);

    ASSERT_NE(app.get_help_ptr(), nullptr);
    EXPECT_EQ(app.get_help_ptr()->get_description(), "显示帮助信息并退出");
    EXPECT_NE(app.get_footer().find("0 成功"), std::string::npos);
    EXPECT_NE(app.get_footer().find("3 算法执行错误"), std::string::npos);
    EXPECT_EQ(app.get_formatter()->get_column_width(), 36U);
}

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

TEST(CliSupportTest, KeepsLatestJsonObjectForEachNonNegativeIntegerKey)
{
    const QJsonArray events{
        QJsonObject{{QStringLiteral("ref_index"), 2},
                    {QStringLiteral("stage"), QStringLiteral("initial")}},
        QJsonObject{{QStringLiteral("ref_index"), 0},
                    {QStringLiteral("stage"), QStringLiteral("initial")}},
        QJsonObject{{QStringLiteral("ref_index"), 2},
                    {QStringLiteral("stage"), QStringLiteral("filtered")}},
        QJsonObject{{QStringLiteral("status"), QStringLiteral("unkeyed")}},
        QJsonObject{{QStringLiteral("ref_index"), -1},
                    {QStringLiteral("status"), QStringLiteral("invalid")}}
    };

    const QJsonArray latest =
        xjw::cli::latestJsonObjectsByNonNegativeIntegerKey(
            events, QStringLiteral("ref_index"));

    ASSERT_EQ(latest.size(), 4);
    EXPECT_EQ(latest.at(0).toObject().value(QStringLiteral("ref_index")).toInt(), 0);
    EXPECT_EQ(latest.at(1).toObject().value(QStringLiteral("ref_index")).toInt(), 2);
    EXPECT_EQ(latest.at(1).toObject().value(QStringLiteral("stage")).toString(),
              QStringLiteral("filtered"));
    EXPECT_EQ(latest.at(2).toObject().value(QStringLiteral("status")).toString(),
              QStringLiteral("unkeyed"));
    EXPECT_EQ(latest.at(3).toObject().value(QStringLiteral("status")).toString(),
              QStringLiteral("invalid"));
}

} // namespace
