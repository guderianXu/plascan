#include <gtest/gtest.h>

#include <QFile>
#include <QProcess>
#include <QProcessEnvironment>
#include <QTemporaryDir>

#ifndef PLASCAN_MARKER_PRINT_CLI_PATH
#define PLASCAN_MARKER_PRINT_CLI_PATH ""
#endif

namespace
{

struct ProcessResult
{
    int exitCode = -1;
    QString output;
};

ProcessResult runCli(const QStringList &arguments)
{
    QProcess process;
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("QT_QPA_PLATFORM"), QStringLiteral("offscreen"));
    process.setProcessEnvironment(environment);
    process.start(QStringLiteral(PLASCAN_MARKER_PRINT_CLI_PATH), arguments);
    if (!process.waitForFinished(30000))
    {
        process.kill();
        return {-1, process.errorString()};
    }
    return {process.exitCode(),
            QString::fromUtf8(process.readAllStandardOutput())
                + QString::fromUtf8(process.readAllStandardError())};
}

} // namespace

TEST(MarkerPrintCliTest, WritesRequestedAprilTagPdf)
{
    ASSERT_FALSE(QStringLiteral(PLASCAN_MARKER_PRINT_CLI_PATH).isEmpty());
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString output = directory.filePath(QStringLiteral("tags.pdf"));
    const ProcessResult result = runCli({
        QStringLiteral("--family"), QStringLiteral("tag36h11"),
        QStringLiteral("--ids"), QStringLiteral("1,2,3"),
        QStringLiteral("--diameter-mm"), QStringLiteral("30"),
        QStringLiteral("--output"), output,
    });
    ASSERT_EQ(result.exitCode, 0) << qPrintable(result.output);
    QFile file(output);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_TRUE(file.read(5).startsWith("%PDF"));
}

TEST(MarkerPrintCliTest, RejectsUnavailableCircularCompatibilityFamily)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const ProcessResult result = runCli({
        QStringLiteral("--family"), QStringLiteral("circular12"),
        QStringLiteral("--ids"), QStringLiteral("1"),
        QStringLiteral("--output"), directory.filePath(QStringLiteral("target.pdf")),
    });
    EXPECT_NE(result.exitCode, 0);
    EXPECT_TRUE(result.output.contains(QStringLiteral("语料"))
                || result.output.contains(QStringLiteral("corpus"), Qt::CaseInsensitive));
}
