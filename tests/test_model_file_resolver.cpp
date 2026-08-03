#include "model/ModelFileResolver.h"
#include "model/ModelAssetCatalog.h"
#include "model/U2NetModelCatalog.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

namespace
{

void writeFile(const QString &path)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_EQ(file.write("model"), 5);
}

class ScopedEnvVar
{
public:
    ScopedEnvVar(const char *name, const QString &value)
        : _name(name)
        , _hadPrevious(qEnvironmentVariableIsSet(name))
        , _previous(qgetenv(name))
    {
        qputenv(name, value.toUtf8());
    }

    ~ScopedEnvVar()
    {
        if (_hadPrevious)
        {
            qputenv(_name, _previous);
        }
        else
        {
            qunsetenv(_name);
        }
    }

private:
    const char *_name;
    bool _hadPrevious = false;
    QByteArray _previous;
};

} // namespace

TEST(ModelFileResolverTest, SearchesEnvironmentBeforeSourceAndApplicationDirs)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString env_dir = QDir(temp_dir.path()).filePath(QStringLiteral("env_models"));
    const QString source_root = QDir(temp_dir.path()).filePath(QStringLiteral("source"));
    const QString app_dir = QDir(source_root).filePath(QStringLiteral("build/bin"));
    const QString model_name = QStringLiteral("feature_engine.plan");
    const QString env_path = QDir(env_dir).filePath(model_name);
    writeFile(QDir(source_root).filePath(QStringLiteral("resources/models/%1").arg(model_name)));
    writeFile(QDir(app_dir).filePath(QStringLiteral("resources/models/%1").arg(model_name)));
    writeFile(env_path);
    ScopedEnvVar env("PLASCAN_MODEL_DIR", env_dir);

    xjw::common::model::ModelFileSearchOptions options;
    options.sourceRoot = source_root;
    options.applicationDir = app_dir;
    const xjw::common::model::ModelFileResolver resolver(options);
    EXPECT_EQ(QDir::cleanPath(resolver.findModel(model_name)), QDir::cleanPath(env_path));
}

TEST(ModelFileResolverTest, FindsFirstExistingModelAndReportsPickedName)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString source_root = QDir(temp_dir.path()).filePath(QStringLiteral("source"));
    const QString app_dir = QDir(source_root).filePath(QStringLiteral("build/bin"));
    const QString picked_model = QStringLiteral("matcher_engine.plan");
    const QString picked_path = QDir(source_root).filePath(
        QStringLiteral("resources/models/%1").arg(picked_model));
    writeFile(picked_path);
    writeFile(QDir(source_root).filePath(QStringLiteral("CMakeLists.txt")));
    qunsetenv("PLASCAN_MODEL_DIR");

    xjw::common::model::ModelFileSearchOptions options;
    options.sourceRoot = source_root;
    options.applicationDir = app_dir;
    const xjw::common::model::ModelFileResolver resolver(options);
    QString picked_name;
    const QString path = resolver.findFirstModel(
        QStringList{QStringLiteral("missing.plan"), picked_model}, &picked_name);

    EXPECT_EQ(QDir::cleanPath(path), QDir::cleanPath(picked_path));
    EXPECT_EQ(picked_name, picked_model);
}

TEST(U2NetModelCatalogTest, ResolvesBundledOnnxModelFromResources)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    const QString source_root = QDir(temp_dir.path()).filePath(QStringLiteral("source"));
    const QString model_path = QDir(source_root).filePath(QStringLiteral("resources/models/U2Net_v1.onnx"));
    writeFile(model_path);
    writeFile(QDir(source_root).filePath(QStringLiteral("CMakeLists.txt")));
    qunsetenv("PLASCAN_MODEL_DIR");

    xjw::common::model::ModelFileSearchOptions options;
    options.sourceRoot = source_root;
    options.applicationDir = QDir(source_root).filePath(QStringLiteral("build/bin"));
    const xjw::common::model::ModelFileResolver resolver(options);
    const auto status = xjw::common::model::u2netModelStatus(resolver);
    EXPECT_TRUE(status.isInstalled);
    EXPECT_EQ(QDir::cleanPath(status.modelPath), QDir::cleanPath(model_path));
}

TEST(ModelFileResolverTest, UsesSourceModelsAsWritableRootForSourceBuild)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    const QString source_root = QDir(temp_dir.path()).filePath(QStringLiteral("source"));
    writeFile(QDir(source_root).filePath(QStringLiteral("CMakeLists.txt")));

    xjw::common::model::ModelFileSearchOptions options;
    options.sourceRoot = source_root;
    options.applicationDir = QDir(source_root).filePath(QStringLiteral("build/bin"));
    options.userModelDir = QDir(temp_dir.path()).filePath(QStringLiteral("user_models"));
    options.environmentVariable = QStringLiteral("PLASCAN_TEST_UNUSED_MODEL_DIR");
    const xjw::common::model::ModelFileResolver resolver(options);

    EXPECT_EQ(
        QDir::cleanPath(resolver.defaultModelDir()),
        QDir::cleanPath(QDir(source_root).filePath(QStringLiteral("resources/models"))));
}

TEST(ModelFileResolverTest, UsesUserDataModelsWhenSourceTreeIsUnavailable)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    const QString user_models = QDir(temp_dir.path()).filePath(QStringLiteral("user_models"));

    xjw::common::model::ModelFileSearchOptions options;
    options.sourceRoot = QDir(temp_dir.path()).filePath(QStringLiteral("missing_source"));
    options.applicationDir = QDir(temp_dir.path()).filePath(QStringLiteral("bin"));
    options.userModelDir = user_models;
    options.environmentVariable = QStringLiteral("PLASCAN_TEST_UNUSED_MODEL_DIR");
    const xjw::common::model::ModelFileResolver resolver(options);

    EXPECT_EQ(QDir::cleanPath(resolver.defaultModelDir()), QDir::cleanPath(user_models));
}

TEST(ModelFileResolverTest, UsesUserDataForInstalledBinaryEvenWhenSourceCheckoutExists)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    const QString source_root = QDir(temp_dir.path()).filePath(QStringLiteral("source"));
    const QString user_models = QDir(temp_dir.path()).filePath(QStringLiteral("user_models"));
    writeFile(QDir(source_root).filePath(QStringLiteral("CMakeLists.txt")));

    xjw::common::model::ModelFileSearchOptions options;
    options.sourceRoot = source_root;
    options.applicationDir = QDir(temp_dir.path()).filePath(QStringLiteral("installed/bin"));
    options.userModelDir = user_models;
    options.environmentVariable = QStringLiteral("PLASCAN_TEST_UNUSED_MODEL_DIR");
    const xjw::common::model::ModelFileResolver resolver(options);

    const auto location = resolver.installLocation();
    EXPECT_EQ(location.kind, xjw::common::model::ModelInstallLocationKind::UserData);
    EXPECT_EQ(QDir::cleanPath(location.directory), QDir::cleanPath(user_models));
}

TEST(ModelAssetCatalogTest, DefinesDirectlyRunnableReleasePackages)
{
    const auto u2net = xjw::common::model::u2NetOnnxPackage();
    ASSERT_TRUE(u2net.isValid());
    ASSERT_EQ(u2net.files.size(), 1);
    EXPECT_EQ(u2net.entryPointFile, QStringLiteral("U2Net_v1.onnx"));
    EXPECT_EQ(u2net.totalBytes(), 175997641);
    EXPECT_EQ(u2net.files.front().sha256,
              QStringLiteral("8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"));

    const auto light_glue = xjw::common::model::lightGlueTensorRtPackage();
    EXPECT_TRUE(light_glue.isValid());
    EXPECT_EQ(light_glue.files.size(), 2);
    EXPECT_EQ(light_glue.totalBytes(), 46713196);
    EXPECT_EQ(light_glue.releaseTag, QStringLiteral("models-v1.0.0"));
    EXPECT_TRUE(light_glue.entryPointFile.endsWith(QStringLiteral(".engine")));

    const auto loma_r = xjw::common::model::loMaRTensorRtPackage(2048);
    EXPECT_TRUE(loma_r.isValid());
    EXPECT_EQ(loma_r.files.size(), 3);
    EXPECT_EQ(loma_r.totalBytes(), 736243465);
    EXPECT_EQ(loma_r.entryPointFile, QStringLiteral("loma_r_k2048_fp16.json"));
    for (const auto &file : loma_r.files)
    {
        EXPECT_TRUE(file.downloadUrl.contains(QStringLiteral("models-v1.0.0")));
        EXPECT_EQ(file.sha256.size(), 64);
        EXPECT_GT(file.bytes, 0);
    }
}
