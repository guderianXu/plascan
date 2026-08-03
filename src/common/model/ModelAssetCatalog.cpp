#include "model/ModelAssetCatalog.h"

#include <QDir>

#include <algorithm>

namespace xjw::common::model
{
namespace
{

constexpr auto kModelReleaseTag = "models-v1.0.0";
constexpr auto kReleaseBaseUrl =
    "https://github.com/guderianXu/plascan/releases/download/models-v1.0.0/";
constexpr auto kCompatibility =
    "预构建环境：NVIDIA GeForce RTX 5080（SM 12.0），TensorRT 10.16.1.11。"
    "TensorRT engine 与 GPU 架构和 TensorRT 版本绑定；不兼容设备需要使用对应环境的模型包。";

ModelAssetFile asset(const char *name, qint64 bytes, const char *sha256)
{
    ModelAssetFile file;
    file.fileName = QString::fromLatin1(name);
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
    package.id = QStringLiteral("sift_lightglue_sm120_trt10_16");
    package.displayName = QStringLiteral("CUDA SIFT + TensorRT LightGlue（K4096）");
    package.packageDirectory = QStringLiteral("lightglue_tensorrt");
    package.entryPointFile = QStringLiteral("lightglue_sift_bucket4096_fp32.engine");
    package.releaseTag = QString::fromLatin1(kModelReleaseTag);
    package.compatibilitySummary = QString::fromUtf8(kCompatibility);
    package.files = {
        asset("lightglue_sift_bucket4096_fp32.engine",
              46712548,
              "4a7cbf13e1161702aa628e8c2dae1a9c976720d169a94bf6178f2add617ea59c"),
        asset("lightglue_sift_bucket4096_fp32.engine.json",
              648,
              "4dbe8302a5d3b2002e11e32f6b0eefd53009f90b9de61009313bdbb038198b04"),
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
        "OpenCV DNN CPU 可直接使用；CUDA 取决于本机 OpenCV DNN CUDA 后端。"
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
    package.id = QStringLiteral("loma_r_k%1_sm120_trt10_16").arg(budget);
    package.displayName = QStringLiteral("LoMa-R TensorRT（K%1）").arg(budget);
    package.packageDirectory = QStringLiteral("loma_r_tensorrt");
    package.entryPointFile = QStringLiteral("loma_r_k%1_fp16.json").arg(budget);
    package.releaseTag = QString::fromLatin1(kModelReleaseTag);
    package.compatibilitySummary = QString::fromUtf8(kCompatibility);

    if (budget == 3840)
    {
        package.files = {
            asset("loma_r_features_k3840_fp16.engine",
                  669429460,
                  "e029c088d5c04a4c106c4bfc891213469dd1aa1ad157da61d56849413ff4b103"),
            asset("loma_r_matcher_k3840_fp16.engine",
                  172246348,
                  "5b7366dbe21b7b5b97d6f8dce0e2be1704e0bcc41964617a81e334dbe65ee384"),
            asset("loma_r_k3840_fp16.json",
                  1041,
                  "d596bdb4eff68484957f4aabd45f5457aa1849747fa1dd9ed06bf05263d67f3b"),
        };
    }
    else if (budget == 2048)
    {
        package.files = {
            asset("loma_r_features_k2048_fp16.engine",
                  669356100,
                  "f83a5727da5b484e02115a3f55edbdd0cc2732efc74cb036b412173651a32dfb"),
            asset("loma_r_matcher_k2048_fp16.engine",
                  66886324,
                  "9df03784c2aa148de546f22948df7dd387ebad121113a4761a6c6a18aaa21548"),
            asset("loma_r_k2048_fp16.json",
                  1041,
                  "a1f840e6bae02e98d4be4ebdd498608a09cedf8bb310cb5aeb8de696a60dbe7e"),
        };
    }
    else
    {
        package.files = {
            asset("loma_r_features_k1024_fp16.engine",
                  669429100,
                  "dd19be8bfcbb652e62b0b4a6221013d317b9512da3bdf99c586e54d3fa4137d2"),
            asset("loma_r_matcher_k1024_fp16.engine",
                  35432372,
                  "5aa4e60e8a5bfe844f06132f354aafac42393c3062ad24ff3fa8146861e799c9"),
            asset("loma_r_k1024_fp16.json",
                  1041,
                  "c17f4da904918de3ff9269ced3f4768f2f7eb582af771911b1b6fa6303cd70fb"),
        };
    }
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

} // namespace xjw::common::model
