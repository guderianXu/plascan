#include "MatchPhotosRuntime.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

namespace
{

class ScopedEnvironment final
{
public:
    ScopedEnvironment(const char *name, const QByteArray &value)
        : _name(name),
          _previous(qgetenv(name)),
          _wasSet(qEnvironmentVariableIsSet(name))
    {
        qputenv(_name.constData(), value);
    }

    ~ScopedEnvironment()
    {
        if (_wasSet)
        {
            qputenv(_name.constData(), _previous);
        }
        else
        {
            qunsetenv(_name.constData());
        }
    }

private:
    QByteArray _name;
    QByteArray _previous;
    bool _wasSet = false;
};

QString createEngine(const QString &directory, int bucket)
{
    const QString name = QStringLiteral("lightglue_sift_bucket%1_fp32.engine").arg(bucket);
    const QString path = QDir(directory).filePath(name);
    QFile engine(path);
    EXPECT_TRUE(engine.open(QIODevice::WriteOnly));
    EXPECT_EQ(engine.write("test-engine"), 11);
    engine.close();

    QFile metadata(path + QStringLiteral(".json"));
    EXPECT_TRUE(metadata.open(QIODevice::WriteOnly));
    const QJsonObject object{{QStringLiteral("bucket_keypoints"), bucket}};
    metadata.write(QJsonDocument(object).toJson(QJsonDocument::Compact));
    metadata.close();
    return path;
}

} // namespace

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

TEST(MatchPhotosRuntimeTest, SelectsLargestSafeEngineBucketFromModelDirectory)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString bucket2048 = createEngine(directory.path(), 2048);
    const QString bucket4096 = createEngine(directory.path(), 4096);
    createEngine(directory.path(), 8192);

    ScopedEnvironment explicitEngine("PLASCAN_LIGHTGLUE_TENSORRT_ENGINE", QByteArray());
    ScopedEnvironment modelDirectory(
        "PLASCAN_MODEL_DIR", QFile::encodeName(directory.path()));
    const xjw::matchphotos::MatchPhotosOptions options;

    const auto resolved = xjw::matchphotos::resolveLightGlueTensorRtEngine(options, 6000);

    ASSERT_TRUE(resolved.isValid());
    EXPECT_EQ(resolved.path, QDir::cleanPath(QFileInfo(bucket4096).absoluteFilePath()));
    EXPECT_EQ(resolved.bucketKeypoints, 4096);
    EXPECT_NE(resolved.path, QDir::cleanPath(QFileInfo(bucket2048).absoluteFilePath()));
}
