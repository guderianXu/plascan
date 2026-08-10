#include "model/ModelAssetCatalog.h"

#include <QDir>

#include <algorithm>

namespace xjw::common::model
{
namespace
{

constexpr auto kModelReleaseTag = "models-v1.1.0";
constexpr auto kReleaseBaseUrl =
    "https://github.com/guderianXu/plascan/releases/download/models-v1.1.0/";
constexpr auto kCompatibility =
    "发布包仅包含可移植 ONNX，不包含任何开发机生成的 TensorRT engine。"
    "PlaScan 首次使用时会依据本机 TensorRT 完整版本和 GPU Compute Capability 构建并缓存 engine。";

ModelAssetFile asset(const char *name, qint64 bytes, const char *sha256)
{
    ModelAssetFile file;
    file.fileName = QString::fromLatin1(name);
    file.downloadUrl = QString::fromLatin1(kReleaseBaseUrl) + file.fileName;
    file.sha256 = QString::fromLatin1(sha256);
    file.bytes = bytes;
    return file;
}

ModelAssetFile asset(const QString &name, qint64 bytes, const char *sha256)
{
    ModelAssetFile file;
    file.fileName = name;
    file.downloadUrl = QString::fromLatin1(kReleaseBaseUrl) + file.fileName;
    file.sha256 = QString::fromLatin1(sha256);
    file.bytes = bytes;
    return file;
}

int normalizedLoMaRBudget(int keypointBudget)
{
    if (keypointBudget >= 3840)
    {
        return 3840;
    }
    if (keypointBudget >= 2048)
    {
        return 2048;
    }
    return 1024;
}

} // namespace

bool ModelAssetPackage::isValid() const
{
    return !id.isEmpty() && !packageDirectory.isEmpty() && !entryPointFile.isEmpty() &&
        !releaseTag.isEmpty() && !files.isEmpty();
}

qint64 ModelAssetPackage::totalBytes() const
{
    qint64 total = 0;
    for (const ModelAssetFile &file : files)
    {
        total += std::max<qint64>(0, file.bytes);
    }
    return total;
}

ModelAssetPackage lightGlueTensorRtPackage()
{
    ModelAssetPackage package;
    package.id = QStringLiteral("sift_lightglue_onnx_k4096_v1");
    package.displayName = QStringLiteral("CUDA SIFT + LightGlue ONNX（K4096）");
    package.packageDirectory = QStringLiteral("lightglue_tensorrt");
    package.entryPointFile = QStringLiteral("lightglue_sift_bucket4096.onnx");
    package.releaseTag = QString::fromLatin1(kModelReleaseTag);
    package.compatibilitySummary = QString::fromUtf8(kCompatibility);
    package.files = {
        asset("lightglue_sift_bucket4096.onnx",
              51072656,
              "773d3de316c37e8d408312d39139352b45e2a93ba055e59cfa2806c5d54ede69"),
    };
    return package;
}

ModelAssetPackage u2NetOnnxPackage()
{
    ModelAssetPackage package;
    package.id = QStringLiteral("u2net_v1_onnx");
    package.displayName = QStringLiteral("U2Net v1 ONNX 蒙版模型");
    package.packageDirectory = QStringLiteral(".");
    package.entryPointFile = QStringLiteral("U2Net_v1.onnx");
    package.releaseTag = QString::fromLatin1(kModelReleaseTag);
    package.compatibilitySummary = QStringLiteral(
        "OpenCV DNN CPU 可直接使用；NVIDIA GPU 使用本机首次构建的 TensorRT engine。"
        "模型来源于 U-2-Net（Apache-2.0）。");
    package.files = {
        asset("U2Net_v1.onnx",
              175997641,
              "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"),
    };
    return package;
}

ModelAssetPackage loMaRTensorRtPackage(int keypointBudget)
{
    const int budget = normalizedLoMaRBudget(keypointBudget);
    ModelAssetPackage package;
    package.id = QStringLiteral("loma_r_onnx_k%1_v1").arg(budget);
    package.displayName = QStringLiteral("LoMa-R ONNX（K%1）").arg(budget);
    package.packageDirectory = QStringLiteral("loma_r_tensorrt");
    package.entryPointFile = QStringLiteral("loma_r_k%1_fp16.json").arg(budget);
    package.releaseTag = QString::fromLatin1(kModelReleaseTag);
    package.compatibilitySummary = QString::fromUtf8(kCompatibility);

    const QString manifestName = QStringLiteral("loma_r_k%1_fp16.json").arg(budget);
    const char *manifestHash = budget == 3840
        ? "5d55026fe3e0bc59bb93bc997d928ec46940905e22c7836b831a859e1dae2715"
        : (budget == 2048
               ? "68ae6a68bb184375285d486384344b7a6500195d5f372e96c8b429bd8787c91e"
               : "db3b242ed7cda10e16fd7c304844c1f809a3b37bc40af10a81c7248ca9e51aea");
    package.files = {
        asset("loma_r_features_k3840_fp16.onnx",
              1318960639,
              "2b2671850f6a79f071a171eb9b523a8807474bcde19b5ded0191b9593ed97e19"),
        asset("loma_r_matcher_dynamic_fp16.onnx",
              45501499,
              "5c91444393c2245e66553e8f493e5b35dc39e8a099b9988a684391fdcdf90195"),
        asset(manifestName, 644, manifestHash),
    };
    return package;
}

QString modelPackageInstallDirectory(const ModelAssetPackage &package,
                                     const ModelFileResolver &resolver)
{
    return QDir::cleanPath(QDir(resolver.defaultModelDir()).filePath(package.packageDirectory));
}

QString modelPackageEntryPoint(const ModelAssetPackage &package,
                               const ModelFileResolver &resolver)
{
    return QDir::cleanPath(
        QDir(modelPackageInstallDirectory(package, resolver)).filePath(package.entryPointFile));
}

QString modelPackageEngineCacheDirectory(const ModelAssetPackage &package,
                                         const ModelFileResolver &resolver)
{
    return QDir::cleanPath(
        QDir(modelPackageInstallDirectory(package, resolver))
            .filePath(QStringLiteral("engines")));
}

} // namespace xjw::common::model
