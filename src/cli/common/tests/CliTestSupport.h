#pragma once

#include <gtest/gtest.h>

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QIODevice>
#include <QProcess>
#include <QRegularExpression>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>
#include <QVector>

#include <zip.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>

#ifndef PLASCAN_MESH_RECONSTRUCT_CLI_PATH
#define PLASCAN_MESH_RECONSTRUCT_CLI_PATH ""
#endif

namespace
{

struct CliResult
{
    int exitCode = -1;
    QString stdoutText;
    QString stderrText;
};

struct Point3f
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

QString repoRoot()
{
    return QStringLiteral(PLASCAN_SOURCE_DIR);
}

QString readSourceFile(const QString &relativePath)
{
    QFile file(QDir(repoRoot()).filePath(relativePath));
    EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(file.fileName());
    if (!file.isOpen())
    {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

QString readTextFile(const QString &path)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(path);
    if (!file.isOpen())
    {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

QString utf8(const char *text)
{
    return QString::fromUtf8(text);
}

void expectContainsAll(const QString &text, std::initializer_list<const char *> needles)
{
    for (const char *needle : needles)
    {
        EXPECT_TRUE(text.contains(utf8(needle))) << needle;
    }
}

void expectNotContainsAll(const QString &text, std::initializer_list<const char *> needles)
{
    for (const char *needle : needles)
    {
        EXPECT_FALSE(text.contains(utf8(needle))) << needle;
    }
}

int indexOfOrFail(const QString &text, const char *needle, int from = 0)
{
    const int index = text.indexOf(utf8(needle), from);
    EXPECT_GE(index, 0) << needle;
    return index;
}

QString sectionBetween(const QString &text,
                       const char *startNeedle,
                       const char *endNeedle,
                       int from = 0)
{
    const int start = indexOfOrFail(text, startNeedle, from);
    if (start < 0)
    {
        return QString();
    }
    const int end = indexOfOrFail(text, endNeedle, start);
    if (end < 0)
    {
        return QString();
    }
    EXPECT_GT(end, start) << endNeedle;
    return text.mid(start, end - start);
}

void expectMatches(const QString &text, const char *pattern)
{
    const QRegularExpression expression(
        utf8(pattern),
        QRegularExpression::DotMatchesEverythingOption);
    EXPECT_TRUE(expression.match(text).hasMatch()) << pattern;
}

QString executablePath(const char *path)
{
    return utf8(path).trimmed();
}

#define SKIP_IF_MISSING_EXECUTABLE(path)                                                               \
    do                                                                                                 \
    {                                                                                                  \
        if ((path).isEmpty() || !QFileInfo::exists(path))                                              \
        {                                                                                              \
            GTEST_SKIP() << "CLI executable not available: " << qPrintable(path);                      \
        }                                                                                              \
    } while (false)

CliResult runCli(const QString &program, const QStringList &arguments, int timeoutMs = 60000)
{
    QProcess process;
    process.setProgram(program);
    process.setArguments(arguments);
    process.setWorkingDirectory(repoRoot());
    process.setProcessChannelMode(QProcess::SeparateChannels);
    process.start();

    if (!process.waitForStarted(timeoutMs))
    {
        return CliResult{-1, QString(), process.errorString()};
    }
    if (!process.waitForFinished(timeoutMs))
    {
        process.kill();
        process.waitForFinished(5000);
        return CliResult{-1, QString::fromUtf8(process.readAllStandardOutput()),
                         QStringLiteral("process timeout: %1").arg(program)};
    }

    return CliResult{process.exitCode(),
                     QString::fromUtf8(process.readAllStandardOutput()),
                     QString::fromUtf8(process.readAllStandardError())};
}

QString combinedOutput(const CliResult &result)
{
    return result.stdoutText + result.stderrText;
}

void writeTextFile(const QString &path, const QString &text)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text)) << qPrintable(path);
    file.write(text.toUtf8());
}

void writeBytesFile(const QString &path, const QByteArray &bytes)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << qPrintable(path);
    file.write(bytes);
}

QJsonObject readJsonObject(const QString &path)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(path);
    if (!file.isOpen())
    {
        return QJsonObject();
    }
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    EXPECT_TRUE(doc.isObject()) << qPrintable(path);
    return doc.object();
}

void writeTsaiCamera(const QString &path)
{
    writeTextFile(path,
                  QStringLiteral("VERSION_3\n"
                                 "PINHOLE\n"
                                 "TSAI\n"
                                 "fu = 100\n"
                                 "fv = 100\n"
                                 "cu = 50\n"
                                 "cv = 50\n"
                                 "u_direction = 1 0 0\n"
                                 "v_direction = 0 1 0\n"
                                 "w_direction = 0 0 1\n"
                                 "C = 0 0 0\n"
                                 "R = 1 0 0 0 1 0 0 0 1\n"
                                 "pitch = 1\n"));
}

void writeUInt32LE(QFile *file, quint32 value)
{
    file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeUInt16LE(QFile *file, quint16 value)
{
    file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeFloatLE(QFile *file, float value)
{
    file->write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeBinaryPly(const QString &path, const QVector<Point3f> &points)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << qPrintable(path);
    const QByteArray header = QStringLiteral("ply\n"
                                            "format binary_little_endian 1.0\n"
                                            "element vertex %1\n"
                                            "property float x\n"
                                            "property float y\n"
                                            "property float z\n"
                                            "end_header\n")
                                  .arg(points.size())
                                  .toUtf8();
    file.write(header);
    for (const Point3f &point : points)
    {
        writeFloatLE(&file, point.x);
        writeFloatLE(&file, point.y);
        writeFloatLE(&file, point.z);
    }
}

void writeBinaryPlyWithScalarProperties(const QString &path, const QVector<Point3f> &points)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << qPrintable(path);
    const QByteArray header = QStringLiteral("ply\n"
                                            "format binary_little_endian 1.0\n"
                                            "element vertex %1\n"
                                            "property float x\n"
                                            "property float y\n"
                                            "property float z\n"
                                            "property ushort intensity\n"
                                            "property float confidence\n"
                                            "end_header\n")
                                  .arg(points.size())
                                  .toUtf8();
    file.write(header);
    for (int i = 0; i < points.size(); ++i)
    {
        const Point3f &point = points.at(i);
        writeFloatLE(&file, point.x);
        writeFloatLE(&file, point.y);
        writeFloatLE(&file, point.z);
        writeUInt16LE(&file, static_cast<quint16>(1000 + i));
        writeFloatLE(&file, 0.5f + 0.1f * static_cast<float>(i));
    }
}

QString readPlyHeader(const QString &path)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly)) << qPrintable(path);
    if (!file.isOpen())
    {
        return QString();
    }
    const QByteArray prefix = file.read(4096);
    const int endHeader = prefix.indexOf("end_header\n");
    EXPECT_GE(endHeader, 0) << qPrintable(path);
    if (endHeader < 0)
    {
        return QString::fromUtf8(prefix);
    }
    return QString::fromUtf8(prefix.left(endHeader + static_cast<int>(QByteArray("end_header\n").size())));
}

void writeZipEntry(const QString &zipPath, const QString &entryName, const QByteArray &contents)
{
    int errorCode = 0;
    const QByteArray nativeZipPath = QFile::encodeName(zipPath);
    zip_t *archive = zip_open(nativeZipPath.constData(), ZIP_CREATE | ZIP_TRUNCATE, &errorCode);
    ASSERT_NE(archive, nullptr) << qPrintable(zipPath) << " zip_error=" << errorCode;

    zip_source_t *source = zip_source_buffer(archive, contents.constData(), contents.size(), 0);
    ASSERT_NE(source, nullptr);
    const QByteArray nativeEntryName = entryName.toUtf8();
    ASSERT_GE(zip_file_add(archive, nativeEntryName.constData(), source, ZIP_FL_OVERWRITE), 0);
    ASSERT_EQ(zip_close(archive), 0);
}

#ifndef PLASCAN_RECONSTRUCT_PIPELINE_CLI_PATH
#define PLASCAN_RECONSTRUCT_PIPELINE_CLI_PATH ""
#endif

#ifndef PLASCAN_CAMERA_CONVERT_CLI_PATH
#define PLASCAN_CAMERA_CONVERT_CLI_PATH ""
#endif

#ifndef PLASCAN_BUNDLE_ADJUST_CLI_PATH
#define PLASCAN_BUNDLE_ADJUST_CLI_PATH ""
#endif

#ifndef PLASCAN_FEATURE_MATCH_CLI_PATH
#define PLASCAN_FEATURE_MATCH_CLI_PATH ""
#endif

#ifndef PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH
#define PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH ""
#endif

#ifndef PLASCAN_MATCH_PHOTOS_CLI_PATH
#define PLASCAN_MATCH_PHOTOS_CLI_PATH ""
#endif

#ifndef PLASCAN_AERIAL_TRIANGULATION_CLI_PATH
#define PLASCAN_AERIAL_TRIANGULATION_CLI_PATH ""
#endif

} // namespace
