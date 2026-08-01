#include "model/TorchScriptModelResolver.h"
#include "model/Sam21ModelCatalog.h"
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

TEST(TorchScriptModelResolverTest, SearchesEnvironmentBeforeSourceAndApplicationDirs)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString envDir = QDir(tempDir.path()).filePath(QStringLiteral("env_models"));
    const QString sourceRoot = QDir(tempDir.path()).filePath(QStringLiteral("source"));
    const QString appDir = QDir(tempDir.path()).filePath(QStringLiteral("bin"));
    const QString modelName = QStringLiteral("sam21_hiera_tiny_encoder_cpu.pt");

    const QString envPath = QDir(envDir).filePath(modelName);
    const QString sourcePath = QDir(sourceRoot).filePath(QStringLiteral("resources/models/%1").arg(modelName));
    const QString appPath = QDir(appDir).filePath(QStringLiteral("resources/models/%1").arg(modelName));
    writeFile(sourcePath);
    writeFile(appPath);
    writeFile(envPath);

    ScopedEnvVar env("PLASCAN_MODEL_DIR", envDir);

    xjw::common::model::TorchScriptModelSearchOptions options;
    options.sourceRoot = sourceRoot;
    options.applicationDir = appDir;
    xjw::common::model::TorchScriptModelResolver resolver(options);

    EXPECT_EQ(QDir::cleanPath(resolver.findModel(modelName)), QDir::cleanPath(envPath));
}

TEST(TorchScriptModelResolverTest, FindsFirstExistingModelAndReportsPickedName)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString sourceRoot = QDir(tempDir.path()).filePath(QStringLiteral("source"));
    const QString appDir = QDir(tempDir.path()).filePath(QStringLiteral("bin"));
    const QString pickedModel = QStringLiteral("sam21_hiera_tiny_encoder_cuda.pt");
    const QString pickedPath = QDir(sourceRoot).filePath(QStringLiteral("resources/models/%1").arg(pickedModel));
    writeFile(pickedPath);
    qunsetenv("PLASCAN_MODEL_DIR");

    xjw::common::model::TorchScriptModelSearchOptions options;
    options.sourceRoot = sourceRoot;
    options.applicationDir = appDir;
    xjw::common::model::TorchScriptModelResolver resolver(options);

    QString pickedName;
    const QString path = resolver.findFirstModel(
        QStringList{QStringLiteral("missing.torchscript"), pickedModel},
        &pickedName);

    EXPECT_EQ(QDir::cleanPath(path), QDir::cleanPath(pickedPath));
    EXPECT_EQ(pickedName, pickedModel);
    EXPECT_EQ(QDir::cleanPath(resolver.defaultModelDir()),
              QDir::cleanPath(QDir(sourceRoot).filePath(QStringLiteral("resources/models"))));
}

TEST(Sam21ModelCatalogTest, DefinesOfficialCheckpointUrlsAndTorchScriptNames)
{
    const auto specs = xjw::common::model::sam21ModelSpecs();
    ASSERT_EQ(specs.size(), 4);

    const auto tiny = xjw::common::model::sam21ModelSpecForToken(QStringLiteral("tiny"));
    ASSERT_TRUE(tiny.has_value());
    EXPECT_EQ(tiny->checkpointFileName, QStringLiteral("sam2.1_hiera_tiny.pt"));
    EXPECT_EQ(tiny->checkpointUrl,
              QStringLiteral("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"));

    const auto large = xjw::common::model::sam21ModelSpecForToken(QStringLiteral("large"));
    ASSERT_TRUE(large.has_value());
    EXPECT_EQ(large->checkpointFileName, QStringLiteral("sam2.1_hiera_large.pt"));

    const auto cudaFiles = xjw::common::model::sam21TorchScriptFiles(QStringLiteral("base_plus"), true);
    EXPECT_EQ(cudaFiles.encoder, QStringLiteral("sam21_hiera_base_plus_encoder_cuda.pt"));
    EXPECT_EQ(cudaFiles.decoder, QStringLiteral("sam21_hiera_base_plus_decoder_cuda.pt"));
}

TEST(Sam21ModelCatalogTest, ReportsInstalledAndMissingModelPieces)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString sourceRoot = QDir(tempDir.path()).filePath(QStringLiteral("source"));
    const QString modelsDir = QDir(sourceRoot).filePath(QStringLiteral("resources/models"));

    writeFile(QDir(modelsDir).filePath(QStringLiteral("sam2.1_hiera_tiny.pt")));
    writeFile(QDir(modelsDir).filePath(QStringLiteral("sam21_hiera_tiny_encoder_cpu.pt")));
    writeFile(QDir(modelsDir).filePath(QStringLiteral("sam21_hiera_tiny_decoder_cpu.pt")));
    qunsetenv("PLASCAN_MODEL_DIR");

    xjw::common::model::TorchScriptModelSearchOptions options;
    options.sourceRoot = sourceRoot;
    xjw::common::model::TorchScriptModelResolver resolver(options);

    const auto tinyStatus = xjw::common::model::sam21ModelStatus(resolver, QStringLiteral("tiny"));
    EXPECT_TRUE(tinyStatus.hasCheckpoint);
    EXPECT_TRUE(tinyStatus.hasCpuTorchScript);
    EXPECT_FALSE(tinyStatus.hasCudaTorchScript);
    EXPECT_EQ(tinyStatus.label, QStringLiteral("缺 CUDA"));
    EXPECT_TRUE(tinyStatus.missingFiles.contains(QStringLiteral("sam21_hiera_tiny_encoder_cuda.pt")));

    const auto smallStatus = xjw::common::model::sam21ModelStatus(resolver, QStringLiteral("small"));
    EXPECT_FALSE(smallStatus.hasCheckpoint);
    EXPECT_FALSE(smallStatus.hasCpuTorchScript);
    EXPECT_FALSE(smallStatus.hasCudaTorchScript);
    EXPECT_EQ(smallStatus.label, QStringLiteral("未安装"));
}

TEST(U2NetModelCatalogTest, ResolvesBundledOnnxModelFromResources)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString sourceRoot = QDir(tempDir.path()).filePath(QStringLiteral("source"));
    const QString modelsDir = QDir(sourceRoot).filePath(QStringLiteral("resources/models"));
    const QString modelPath = QDir(modelsDir).filePath(QStringLiteral("U2Net_v1.onnx"));
    writeFile(modelPath);
    qunsetenv("PLASCAN_MODEL_DIR");

    xjw::common::model::TorchScriptModelSearchOptions options;
    options.sourceRoot = sourceRoot;
    xjw::common::model::TorchScriptModelResolver resolver(options);

    const auto spec = xjw::common::model::u2netModelSpec();
    EXPECT_EQ(spec.token, QStringLiteral("u2net_v1"));
    EXPECT_EQ(spec.modelFileNames.front(), QStringLiteral("U2Net_v1.onnx"));

    const auto status = xjw::common::model::u2netModelStatus(resolver);
    EXPECT_TRUE(status.isInstalled);
    EXPECT_EQ(QDir::cleanPath(status.modelPath), QDir::cleanPath(modelPath));
    EXPECT_EQ(status.label, QStringLiteral("已安装"));
}

TEST(U2NetModelCatalogTest, ReportsMissingBundledOnnxModel)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    qunsetenv("PLASCAN_MODEL_DIR");

    xjw::common::model::TorchScriptModelSearchOptions options;
    options.sourceRoot = QDir(tempDir.path()).filePath(QStringLiteral("source"));
    xjw::common::model::TorchScriptModelResolver resolver(options);

    const auto status = xjw::common::model::u2netModelStatus(resolver);
    EXPECT_FALSE(status.isInstalled);
    EXPECT_TRUE(status.missingFiles.contains(QStringLiteral("U2Net_v1.onnx")));
    EXPECT_EQ(status.label, QStringLiteral("未安装"));
    EXPECT_TRUE(status.detail.contains(QStringLiteral("resources/models")));
}
