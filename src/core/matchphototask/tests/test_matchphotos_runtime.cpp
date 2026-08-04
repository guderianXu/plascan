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

QString createLoMaRPackage(const QString &directory,
                           int keypoints = 2048,
                           bool includeMatcher = true)
{
    const QString featureName = QStringLiteral("loma_r_features_k%1_fp16.engine")
                                    .arg(keypoints);
    const QString matcherName = QStringLiteral("loma_r_matcher_k%1_fp16.engine")
                                    .arg(keypoints);
    for (const QString &name : QStringList{featureName, matcherName})
    {
        if (!includeMatcher && name == matcherName)
        {
            continue;
        }
        QFile engine(QDir(directory).filePath(name));
        EXPECT_TRUE(engine.open(QIODevice::WriteOnly));
        EXPECT_GT(engine.write("test-engine"), 0);
    }

    const QJsonObject manifest{
        {QStringLiteral("schema_version"), 1},
        {QStringLiteral("algorithm_id"), QStringLiteral("loma_r")},
        {QStringLiteral("algorithm_version"), 1},
        {QStringLiteral("feature_engine"), featureName},
        {QStringLiteral("matcher_engine"), matcherName},
        {QStringLiteral("input_width"), 784},
        {QStringLiteral("input_height"), 784},
        {QStringLiteral("keypoint_count"), keypoints},
        {QStringLiteral("descriptor_dimension"), 256}};
    const QString manifestPath = QDir(directory).filePath(
        QStringLiteral("loma_r_k%1_fp16.json").arg(keypoints));
    QFile file(manifestPath);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly));
    file.write(QJsonDocument(manifest).toJson(QJsonDocument::Compact));
    return manifestPath;
}

QString createPortableLoMaRPackage(const QString &directory,
                                   int keypoints = 1024,
                                   int featureKeypoints = 3840)
{
    const QString featureName = QStringLiteral("loma_r_features_k%1_fp16.onnx")
                                    .arg(featureKeypoints);
    const QString matcherName = QStringLiteral("loma_r_matcher_dynamic_fp16.onnx");
    for (const QString &name : QStringList{featureName, matcherName})
    {
        QFile onnx(QDir(directory).filePath(name));
        EXPECT_TRUE(onnx.open(QIODevice::WriteOnly));
        EXPECT_GT(onnx.write("test-onnx"), 0);
    }

    const QJsonObject manifest{
        {QStringLiteral("schema_version"), 2},
        {QStringLiteral("algorithm_id"), QStringLiteral("loma_r")},
        {QStringLiteral("algorithm_version"), 1},
        {QStringLiteral("precision"), QStringLiteral("fp16")},
        {QStringLiteral("feature_onnx"), featureName},
        {QStringLiteral("matcher_onnx"), matcherName},
        {QStringLiteral("input_width"), 784},
        {QStringLiteral("input_height"), 784},
        {QStringLiteral("keypoint_count"), keypoints},
        {QStringLiteral("feature_keypoint_count"), featureKeypoints},
        {QStringLiteral("descriptor_dimension"), 256}};
    const QString manifestPath = QDir(directory).filePath(
        QStringLiteral("loma_r_k%1_fp16.json").arg(keypoints));
    QFile file(manifestPath);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly));
    file.write(QJsonDocument(manifest).toJson(QJsonDocument::Compact));
    return manifestPath;
}

} // namespace

TEST(MatchPhotosRuntimeTest, DefaultsToAutomaticTensorRtEngineLookup)
{
    const xjw::matchphotos::MatchPhotosOptions options;

    EXPECT_TRUE(options.lightGlueTensorRtEnginePath.isEmpty());
    EXPECT_TRUE(options.lomaRTensorRtPackagePath.isEmpty());
}

TEST(MatchPhotosRuntimeTest, ResolvesExplicitLoMaRTensorRtPackage)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString manifestPath = createLoMaRPackage(directory.path());

    xjw::matchphotos::MatchPhotosOptions options;
    options.lomaRTensorRtPackagePath = manifestPath;
    const auto resolved = xjw::matchphotos::resolveLoMaRTensorRtPackage(options);

    ASSERT_TRUE(resolved.isValid()) << qPrintable(resolved.errorMessage);
    EXPECT_EQ(resolved.manifestPath,
              QDir::cleanPath(QFileInfo(manifestPath).absoluteFilePath()));
    EXPECT_TRUE(QFileInfo::exists(resolved.featureEnginePath));
    EXPECT_TRUE(QFileInfo::exists(resolved.matcherEnginePath));
    EXPECT_EQ(resolved.keypointCount, 2048);
    EXPECT_EQ(resolved.descriptorDimension, 256);
}

TEST(MatchPhotosRuntimeTest, ResolvesPortableLoMaRPackageWithoutBuildingEngine)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString manifestPath = createPortableLoMaRPackage(directory.path());

    xjw::matchphotos::MatchPhotosOptions options;
    options.lomaRTensorRtPackagePath = manifestPath;
    const auto resolved = xjw::matchphotos::resolveLoMaRTensorRtPackage(
        options, 1024, false);

    ASSERT_TRUE(resolved.isValid()) << qPrintable(resolved.errorMessage);
    EXPECT_TRUE(QFileInfo::exists(resolved.featureOnnxPath));
    EXPECT_TRUE(QFileInfo::exists(resolved.matcherOnnxPath));
    EXPECT_TRUE(resolved.featureEnginePath.isEmpty());
    EXPECT_TRUE(resolved.matcherEnginePath.isEmpty());
    EXPECT_EQ(resolved.keypointCount, 1024);
    EXPECT_EQ(resolved.featureKeypointCount, 3840);
}

TEST(MatchPhotosRuntimeTest, RejectsIncompleteLoMaRTensorRtPackage)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    xjw::matchphotos::MatchPhotosOptions options;
    options.lomaRTensorRtPackagePath = createLoMaRPackage(directory.path(), 2048, false);

    const auto resolved = xjw::matchphotos::resolveLoMaRTensorRtPackage(options);

    EXPECT_FALSE(resolved.isValid());
    EXPECT_TRUE(resolved.errorMessage.contains(QStringLiteral("matcher"),
                                               Qt::CaseInsensitive));
}

TEST(MatchPhotosRuntimeTest, SelectsLargestLoMaRPackageWithinPreferredBudget)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    createLoMaRPackage(directory.path(), 1024);
    const QString bucket2048 = createLoMaRPackage(directory.path(), 2048);
    createLoMaRPackage(directory.path(), 3840);

    ScopedEnvironment explicitPackage("PLASCAN_LOMA_R_TENSORRT_PACKAGE", QByteArray());
    ScopedEnvironment modelDirectory(
        "PLASCAN_MODEL_DIR", QFile::encodeName(directory.path()));
    xjw::matchphotos::MatchPhotosOptions options;
    options.lomaRKeypointBudget = 2048;

    const auto resolved = xjw::matchphotos::resolveLoMaRTensorRtPackage(options, 40000);

    ASSERT_TRUE(resolved.isValid()) << qPrintable(resolved.errorMessage);
    EXPECT_EQ(resolved.manifestPath,
              QDir::cleanPath(QFileInfo(bucket2048).absoluteFilePath()));
    EXPECT_EQ(resolved.keypointCount, 2048);
}

TEST(MatchPhotosRuntimeTest, FallsBackToSmallestLoMaRPackageAboveTinyBudget)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString bucket1024 = createLoMaRPackage(directory.path(), 1024);
    createLoMaRPackage(directory.path(), 2048);

    ScopedEnvironment explicitPackage("PLASCAN_LOMA_R_TENSORRT_PACKAGE", QByteArray());
    ScopedEnvironment modelDirectory(
        "PLASCAN_MODEL_DIR", QFile::encodeName(directory.path()));
    xjw::matchphotos::MatchPhotosOptions options;
    options.lomaRKeypointBudget = 500;

    const auto resolved = xjw::matchphotos::resolveLoMaRTensorRtPackage(options, 500);

    ASSERT_TRUE(resolved.isValid()) << qPrintable(resolved.errorMessage);
    EXPECT_EQ(resolved.manifestPath,
              QDir::cleanPath(QFileInfo(bucket1024).absoluteFilePath()));
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

TEST(MatchPhotosRuntimeTest, ResolvesPortableLightGlueOnnxWithoutBuildingEngine)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    const QString onnxPath = directory.filePath(
        QStringLiteral("lightglue_sift_bucket4096.onnx"));
    QFile onnx(onnxPath);
    ASSERT_TRUE(onnx.open(QIODevice::WriteOnly));
    ASSERT_GT(onnx.write("test-onnx"), 0);
    onnx.close();

    xjw::matchphotos::MatchPhotosOptions options;
    options.lightGlueTensorRtEnginePath = onnxPath;
    const auto resolved = xjw::matchphotos::resolveLightGlueTensorRtEngine(
        options, 4096, false);

    ASSERT_TRUE(resolved.isValid()) << qPrintable(resolved.errorMessage);
    EXPECT_EQ(resolved.path, QDir::cleanPath(QFileInfo(onnxPath).absoluteFilePath()));
    EXPECT_EQ(resolved.sourceOnnxPath, resolved.path);
    EXPECT_EQ(resolved.bucketKeypoints, 4096);
}

TEST(MatchPhotosRuntimeTest, ResolvesLightGlueFromReleasePackageSubdirectory)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString packageDirectory = QDir(directory.path()).filePath(
        QStringLiteral("lightglue_tensorrt"));
    ASSERT_TRUE(QDir().mkpath(packageDirectory));
    const QString enginePath = createEngine(packageDirectory, 4096);

    ScopedEnvironment explicitEngine("PLASCAN_LIGHTGLUE_TENSORRT_ENGINE", QByteArray());
    ScopedEnvironment modelDirectory(
        "PLASCAN_MODEL_DIR", QFile::encodeName(directory.path()));
    const auto resolved = xjw::matchphotos::resolveLightGlueTensorRtEngine(
        xjw::matchphotos::MatchPhotosOptions(), 4096);

    ASSERT_TRUE(resolved.isValid());
    EXPECT_EQ(resolved.path, QDir::cleanPath(QFileInfo(enginePath).absoluteFilePath()));
}

TEST(MatchPhotosRuntimeTest, ResolvesLoMaRFromReleasePackageSubdirectory)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString packageDirectory = QDir(directory.path()).filePath(
        QStringLiteral("loma_r_tensorrt"));
    ASSERT_TRUE(QDir().mkpath(packageDirectory));
    const QString manifestPath = createLoMaRPackage(packageDirectory, 2048);

    ScopedEnvironment explicitPackage("PLASCAN_LOMA_R_TENSORRT_PACKAGE", QByteArray());
    ScopedEnvironment modelDirectory(
        "PLASCAN_MODEL_DIR", QFile::encodeName(directory.path()));
    xjw::matchphotos::MatchPhotosOptions options;
    options.lomaRKeypointBudget = 2048;
    const auto resolved = xjw::matchphotos::resolveLoMaRTensorRtPackage(options, 40000);

    ASSERT_TRUE(resolved.isValid()) << qPrintable(resolved.errorMessage);
    EXPECT_EQ(resolved.manifestPath,
              QDir::cleanPath(QFileInfo(manifestPath).absoluteFilePath()));
}
