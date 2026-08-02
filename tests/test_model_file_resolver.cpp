#include "model/ModelFileResolver.h"
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
    const QString app_dir = QDir(temp_dir.path()).filePath(QStringLiteral("bin"));
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
    const QString app_dir = QDir(temp_dir.path()).filePath(QStringLiteral("bin"));
    const QString picked_model = QStringLiteral("matcher_engine.plan");
    const QString picked_path = QDir(source_root).filePath(
        QStringLiteral("resources/models/%1").arg(picked_model));
    writeFile(picked_path);
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
    qunsetenv("PLASCAN_MODEL_DIR");

    xjw::common::model::ModelFileSearchOptions options;
    options.sourceRoot = source_root;
    const xjw::common::model::ModelFileResolver resolver(options);
    const auto status = xjw::common::model::u2netModelStatus(resolver);
    EXPECT_TRUE(status.isInstalled);
    EXPECT_EQ(QDir::cleanPath(status.modelPath), QDir::cleanPath(model_path));
}
