#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QProcess>
#include <QTemporaryDir>

#ifndef PLASCAN_MARKER_DETECT_CLI_PATH
#define PLASCAN_MARKER_DETECT_CLI_PATH ""
#endif

namespace
{

QImage circleImage()
{
    QImage image(240, 200, QImage::Format_Grayscale8);
    image.fill(Qt::white);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::black);
    painter.drawEllipse(QPointF(120.5, 100.5), 24.0, 24.0);
    return image;
}

struct ProcessResult
{
    int exitCode = -1;
    QString output;
};

ProcessResult runCli(const QStringList &arguments)
{
    QProcess process;
    process.start(QStringLiteral(PLASCAN_MARKER_DETECT_CLI_PATH), arguments);
    if (!process.waitForFinished(30000))
    {
        process.kill();
        return {-1, process.errorString()};
    }
    return {process.exitCode(),
            QString::fromUtf8(process.readAllStandardOutput())
                + QString::fromUtf8(process.readAllStandardError())};
}

QJsonObject readJson(const QString &path)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly));
    return QJsonDocument::fromJson(file.readAll()).object();
}

} // namespace

TEST(MarkerDetectCliTest, WritesDeterministicObservationsAndAppliesMask)
{
    ASSERT_FALSE(QStringLiteral(PLASCAN_MARKER_DETECT_CLI_PATH).isEmpty());
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString image_path = directory.filePath(QStringLiteral("circle.png"));
    const QString output_path = directory.filePath(QStringLiteral("detections.json"));
    ASSERT_TRUE(circleImage().save(image_path));

    ProcessResult detected = runCli({QStringLiteral("--image"), image_path,
                                     QStringLiteral("--family"), QStringLiteral("noncoded-circle"),
                                     QStringLiteral("--output"), output_path});
    ASSERT_EQ(detected.exitCode, 0) << qPrintable(detected.output);
    QJsonObject output = readJson(output_path);
    EXPECT_EQ(output.value(QStringLiteral("schema")).toString(),
              QStringLiteral("plascan.marker-detections.v1"));
    ASSERT_EQ(output.value(QStringLiteral("observations")).toArray().size(), 1);

    QImage mask(circleImage().size(), QImage::Format_Grayscale8);
    mask.fill(Qt::black);
    QPainter painter(&mask);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::white);
    painter.drawEllipse(QPointF(120.5, 100.5), 30.0, 30.0);
    painter.end();
    const QString mask_path = directory.filePath(QStringLiteral("mask.png"));
    ASSERT_TRUE(mask.save(mask_path));

    ProcessResult masked = runCli({QStringLiteral("--image"), image_path,
                                   QStringLiteral("--mask"), mask_path,
                                   QStringLiteral("--family"), QStringLiteral("noncoded-circle"),
                                   QStringLiteral("--output"), output_path});
    ASSERT_EQ(masked.exitCode, 0) << qPrintable(masked.output);
    output = readJson(output_path);
    EXPECT_TRUE(output.value(QStringLiteral("observations")).toArray().isEmpty());
}

TEST(MarkerDetectCliTest, RejectsUnavailableCircularCompatibilityFamily)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString image_path = directory.filePath(QStringLiteral("circle.png"));
    ASSERT_TRUE(circleImage().save(image_path));
    const ProcessResult result = runCli({
        QStringLiteral("--image"), image_path,
        QStringLiteral("--family"), QStringLiteral("circular12"),
        QStringLiteral("--output"), directory.filePath(QStringLiteral("output.json"))
    });
    EXPECT_NE(result.exitCode, 0);
    EXPECT_TRUE(result.output.contains(QStringLiteral("corpus"), Qt::CaseInsensitive));
}
