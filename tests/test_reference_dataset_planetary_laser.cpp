#include <gtest/gtest.h>

#include "ReferenceDatasetWorkflow.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

namespace
{

QString writeJson(const QString &directory,
                  const QString &name,
                  const QJsonObject &object)
{
    const QString path = QDir(directory).filePath(name);
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    EXPECT_GT(file.write(QJsonDocument(object).toJson(QJsonDocument::Compact)), 0);
    return path;
}

} // namespace

TEST(ReferenceDatasetPlanetaryLaserTest, DoesNotClassifyGenericPointArrayAsIsisLidar)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = writeJson(
        directory.path(),
        QStringLiteral("generic_points.json"),
        QJsonObject{{QStringLiteral("points"),
                     QJsonArray{QJsonObject{{QStringLiteral("x"), 1.0},
                                            {QStringLiteral("y"), 2.0},
                                            {QStringLiteral("z"), 3.0}}}}});

    EXPECT_TRUE(xjw::core::project::referenceDatasetTypeForPath(path).isEmpty());
}

TEST(ReferenceDatasetPlanetaryLaserTest, RecognizesIsisLidarPointSignature)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QJsonObject point{
        {QStringLiteral("id"), QStringLiteral("shot-1")},
        {QStringLiteral("time"), 123.0},
        {QStringLiteral("range"), 10.0},
        {QStringLiteral("sigmaRange"), 0.1},
        {QStringLiteral("latitude"), 1.0},
        {QStringLiteral("longitude"), 2.0},
        {QStringLiteral("radius"), 1737.4},
    };
    const QString path = writeJson(
        directory.path(),
        QStringLiteral("isis_lidar.json"),
        QJsonObject{{QStringLiteral("points"), QJsonArray{point}}});

    EXPECT_EQ(xjw::core::project::referenceDatasetTypeForPath(path),
              QStringLiteral("planetary_laser_shots"));
}
