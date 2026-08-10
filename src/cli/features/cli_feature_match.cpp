/**
 * @file cli_feature_match.cpp
 * @brief 两幅原始影像的统一特征匹配入口。
 *
 * 本 CLI 不再接受或生成中间特征文件。两幅影像的 SIFT 特征只存在于本次
 * MatchPhotosTask 的内存缓存中，最终匹配按“一幅影像一个 `.pimatch` 分片”
 * 对称写入输出目录。这样 CLI、GUI 和空三共用完全相同的算法、几何验证、
 * 残差计算及二进制版本契约。
 */

#include "cli_common.h"

#include "MatchPhotosTask.h"
#include "PairTypes.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <string>

int main(int argc, char *argv[])
{
    QCoreApplication qtApplication(argc, argv);
    CLI::App app{"PlaScan 双影像匹配"};

    std::string leftImageArg;
    std::string rightImageArg;
    std::string outputDirectoryArg;
    std::string enginePathArg;
    std::string algorithmIdArg = "sift_lightglue";
    std::string deviceArg = "auto";
    int maxKeypoints = 40000;
    int maxImageDim = 0;
    int cudaDevice = 0;
    float matchThreshold = 0.15f;
    double geometryThreshold = 1.5;
    int geometryMinInliers = 20;

    app.add_option("-L,--left", leftImageArg, "左影像路径")->required();
    app.add_option("-R,--right", rightImageArg, "右影像路径")->required();
    app.add_option("-o,--output-dir", outputDirectoryArg,
                   "逐影像 .pimatch 输出目录")->required();
    app.add_option("-m,--model", enginePathArg,
                   "算法模型资源：LightGlue .onnx/本机 .engine 或 LoMa-R JSON 清单；"
                   "CUDA SIFT 无需模型");
    app.add_option("-a,--algorithm-id", algorithmIdArg,
                   "统一影像匹配算法 ID: sift_lightglue, cuda_sift, loma_r")
        ->check(CLI::IsMember({"sift_lightglue", "cuda_sift", "loma_r"}));
    app.add_option("-n,--max-keypoints", maxKeypoints, "每幅影像最大关键点数");
    app.add_option("--max-image-dim", maxImageDim,
                   "提取输入最长边，0 表示保持原始分辨率");
    app.add_option("--cuda-device", cudaDevice, "CUDA 设备 ID");
    app.add_option("--device", deviceArg, "计算设备: auto, cpu, cuda")
        ->check(CLI::IsMember({"auto", "cpu", "cuda"}));
    app.add_option("-t,--match-threshold", matchThreshold, "匹配置信度阈值");
    app.add_option("--geometry-threshold", geometryThreshold,
                   "几何验证像素残差阈值");
    app.add_option("--geometry-min-inliers", geometryMinInliers,
                   "几何验证最少内点数");

    CLI11_PARSE(app, argc, argv);

    const QString leftImage = QDir::cleanPath(
        QFileInfo(QString::fromStdString(leftImageArg)).absoluteFilePath());
    const QString rightImage = QDir::cleanPath(
        QFileInfo(QString::fromStdString(rightImageArg)).absoluteFilePath());
    const QString outputDirectory = QDir::cleanPath(
        QFileInfo(QString::fromStdString(outputDirectoryArg)).absoluteFilePath());
    if (!QFileInfo(leftImage).isFile() || !QFileInfo(rightImage).isFile())
    {
        std::fprintf(stderr, "错误: 左右影像必须是可读取文件。\n");
        return cli::EXIT_IO_ERR;
    }
    if (leftImage.compare(rightImage, Qt::CaseInsensitive) == 0)
    {
        std::fprintf(stderr, "错误: 左右影像不能是同一个文件。\n");
        return cli::EXIT_ARG_ERR;
    }
    if (!QDir().mkpath(outputDirectory))
    {
        std::fprintf(stderr, "错误: 无法创建输出目录: %s\n",
                     qUtf8Printable(outputDirectory));
        return cli::EXIT_IO_ERR;
    }

    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QString::fromStdString(algorithmIdArg).trimmed().toLower();
    options.profile = xjw::matchphotos::MatchPhotosProfile::HighAccuracy;
    if (deviceArg == "cpu")
    {
        options.device = xjw::matchphotos::ComputeDevice::Cpu;
    }
    else if (deviceArg == "cuda")
    {
        options.device = xjw::matchphotos::ComputeDevice::Cuda;
    }
    else
    {
        options.device = xjw::matchphotos::ComputeDevice::Auto;
    }
    if (options.algorithmId == QLatin1String("loma_r"))
    {
        options.lomaRTensorRtPackagePath = QString::fromStdString(enginePathArg);
    }
    else if (options.algorithmId == QLatin1String("sift_lightglue"))
    {
        options.lightGlueTensorRtEnginePath = QString::fromStdString(enginePathArg);
    }
    options.pairPolicy.mode = xjw::matchphotos::PairSelectionMode::ManualOnly;
    options.maxKeypoints = std::max(0, maxKeypoints);
    options.useExplicitKeypointLimit = true;
    options.maxImageDim = maxImageDim;
    options.cudaDevice = std::max(0, cudaDevice);
    options.matchThreshold = std::clamp(matchThreshold, 0.0f, 1.0f);
    options.geometryReprojThreshold = std::max(0.1, geometryThreshold);
    options.geometryMinInliers = std::max(4, geometryMinInliers);
    options.enableGeometryVerification = true;
    options.enableTrackBuild = true;
    options.useGenericPreselection = false;
    options.useReferencePreselection = false;
    options.reuseExistingMatches = false;
    options.planOnly = false;

    xjw::matchphotos::MatchPhotosContext context;
    context.workingDirectory = outputDirectory;
    context.matchDirectory = outputDirectory;
    context.pairInput.images = QStringList{leftImage, rightImage};
    context.pairInput.manualPairKeys.append(
        xjw::matchphotos::makePairKey(leftImage, rightImage));

    std::atomic_bool cancelFlag(false);
    context.cancelFlag = &cancelFlag;
    context.progressCallback = [](const QString &stageId,
                                  const QString &message,
                                  int current,
                                  int maximum)
    {
        std::fprintf(stdout, "[%s %d/%d] %s\n",
                     qUtf8Printable(stageId), current, maximum,
                     qUtf8Printable(message));
        std::fflush(stdout);
    };

    const xjw::matchphotos::MatchPhotosResult result =
        xjw::matchphotos::MatchPhotosTask(options).run(context);
    if (!result.success)
    {
        std::fprintf(stderr, "匹配失败: %s\n", qUtf8Printable(result.errorMessage));
        return cli::EXIT_ALGO_ERR;
    }

    std::fprintf(stdout, "status=ok\n");
    std::fprintf(stdout, "pair_matches=%d\n", static_cast<int>(result.matches.size()));
    std::fprintf(stdout, "tracks=%d\n", result.trackCount);
    for (const xjw::matchphotos::MatchPhotosImageMatchRecord &record : result.imageMatchFiles)
    {
        std::fprintf(stdout, "image_match_file=%s\n",
                     qUtf8Printable(record.matchFilePath));
    }
    return cli::EXIT_OK;
}
