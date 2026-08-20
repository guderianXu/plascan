#include "cli_common.h"
#include "cli_photogrammetry_common.h"
#include "CliTokenUtils.h"

#include "MatchPhotosTask.h"
#include "project/ProjectSession.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

using xjw::matchphotos::ComputeDevice;
using xjw::matchphotos::MatchPhotosOptions;
using xjw::matchphotos::MatchPhotosProfile;
using xjw::matchphotos::MatchPhotosResult;
using xjw::matchphotos::MatchPhotosStageStatus;
using xjw::matchphotos::PairSelectionMode;
using xjw::matchphotos::PairSelectionPolicy;
using xjw::matchphotos::PairSelectionPreset;

MatchPhotosProfile profileFromToken(const QString &token)
{
    const QString normalized = xjw::cli::normalizedToken(token, QStringLiteral("auto"));
    if (normalized == QStringLiteral("fast") || normalized == QStringLiteral("low"))
    {
        return MatchPhotosProfile::Fast;
    }
    if (normalized == QStringLiteral("high") || normalized == QStringLiteral("highest"))
    {
        return MatchPhotosProfile::HighAccuracy;
    }
    if (normalized == QStringLiteral("difficult") || normalized == QStringLiteral("difficult_texture"))
    {
        return MatchPhotosProfile::DifficultTexture;
    }
    if (normalized == QStringLiteral("cpu"))
    {
        return MatchPhotosProfile::CpuCompatible;
    }
    if (normalized == QStringLiteral("cuda"))
    {
        return MatchPhotosProfile::CudaAccelerated;
    }
    return MatchPhotosProfile::Auto;
}

PairSelectionPreset pairPresetFromProfile(MatchPhotosProfile profile)
{
    switch (profile)
    {
    case MatchPhotosProfile::Fast:
        return PairSelectionPreset::Fast;
    case MatchPhotosProfile::HighAccuracy:
        return PairSelectionPreset::HighAccuracy;
    case MatchPhotosProfile::CpuCompatible:
        return PairSelectionPreset::CpuCompatible;
    case MatchPhotosProfile::DifficultTexture:
        return PairSelectionPreset::DifficultTexture;
    case MatchPhotosProfile::CudaAccelerated:
    case MatchPhotosProfile::Auto:
    default:
        return PairSelectionPreset::Auto;
    }
}

ComputeDevice deviceFromToken(const QString &token)
{
    const QString normalized = xjw::cli::normalizedToken(token, QStringLiteral("auto"));
    if (normalized == QStringLiteral("cpu"))
    {
        return ComputeDevice::Cpu;
    }
    if (normalized == QStringLiteral("cuda") || normalized == QStringLiteral("gpu"))
    {
        return ComputeDevice::Cuda;
    }
    if (normalized == QStringLiteral("opencl"))
    {
        return ComputeDevice::OpenCl;
    }
    if (normalized == QStringLiteral("metal"))
    {
        return ComputeDevice::Metal;
    }
    return ComputeDevice::Auto;
}

PairSelectionMode pairModeFromToken(const QString &token)
{
    const QString normalized = xjw::cli::normalizedToken(token, QStringLiteral("auto"));
    if (normalized == QStringLiteral("exhaustive") || normalized == QStringLiteral("all"))
    {
        return PairSelectionMode::Exhaustive;
    }
    if (normalized == QStringLiteral("sequence") || normalized == QStringLiteral("sequential"))
    {
        return PairSelectionMode::Sequence;
    }
    if (normalized == QStringLiteral("manual-only") || normalized == QStringLiteral("manual_only"))
    {
        return PairSelectionMode::ManualOnly;
    }
    return PairSelectionMode::Auto;
}

QString stageStatusToString(MatchPhotosStageStatus status)
{
    switch (status)
    {
    case MatchPhotosStageStatus::Completed:
        return QStringLiteral("completed");
    case MatchPhotosStageStatus::Skipped:
        return QStringLiteral("skipped");
    case MatchPhotosStageStatus::Failed:
        return QStringLiteral("failed");
    case MatchPhotosStageStatus::Pending:
    default:
        return QStringLiteral("pending");
    }
}

QJsonObject algorithmPlanToJson(const xjw::matchphotos::MatchPhotosAlgorithmPlan &plan)
{
    QJsonObject object;
    object[QStringLiteral("strategy_id")] = plan.strategyId;
    object[QStringLiteral("display_name")] = plan.displayName;
    object[QStringLiteral("algorithm_id")] = plan.algorithmId;
    object[QStringLiteral("algorithm_version")] = static_cast<qint64>(plan.algorithmVersion);
    object[QStringLiteral("valid")] = plan.valid;
    object[QStringLiteral("extracts_features_in_memory")] = plan.extractsFeaturesInMemory;
    object[QStringLiteral("requires_cuda")] = plan.requiresCuda;
    object[QStringLiteral("prefer_cuda")] = plan.preferCuda;
    object[QStringLiteral("rotation_robust")] = plan.rotationRobust;
    object[QStringLiteral("max_image_dim")] = plan.maxImageDim;
    object[QStringLiteral("max_keypoints")] = plan.maxKeypoints;
    object[QStringLiteral("guided_matching")] = plan.enableGuidedMatching;
    object[QStringLiteral("reason")] = plan.reason;
    object[QStringLiteral("validation_error")] = plan.validationError;
    return object;
}

QJsonArray stagesToJson(const MatchPhotosResult &result)
{
    QJsonArray array;
    for (const xjw::matchphotos::MatchPhotosStageReport &stage : result.stages)
    {
        QJsonObject object;
        object[QStringLiteral("stage_id")] = stage.stageId;
        object[QStringLiteral("display_name")] = stage.displayName;
        object[QStringLiteral("status")] = stageStatusToString(stage.status);
        object[QStringLiteral("message")] = stage.message;
        object[QStringLiteral("item_count")] = stage.itemCount;
        array.append(object);
    }
    return array;
}

QJsonArray sourcesToJson(const std::vector<xjw::matchphotos::PairSource> &sources)
{
    QJsonArray array;
    for (const xjw::matchphotos::PairSource source : sources)
    {
        array.append(xjw::matchphotos::pairSourceId(source));
    }
    return array;
}

QJsonObject pairSelectionToJson(const MatchPhotosResult &result)
{
    QJsonObject object;
    object[QStringLiteral("restrict_pairs")] = result.pairSelection.restrictPairs;
    object[QStringLiteral("image_count")] = result.pairSelection.imageCount;
    object[QStringLiteral("all_pair_count")] = result.pairSelection.allPairCount;
    object[QStringLiteral("candidate_count")] = static_cast<int>(result.pairSelection.candidates.size());
    object[QStringLiteral("allowed_pair_count")] = result.pairSelection.allowedPairKeys.size();
    object[QStringLiteral("detail")] = result.pairSelection.detail;

    QJsonArray samples;
    const int sampleCount = std::min<int>(20, static_cast<int>(result.pairSelection.candidates.size()));
    for (int i = 0; i < sampleCount; ++i)
    {
        const xjw::matchphotos::PairCandidate &candidate = result.pairSelection.candidates.at(i);
        QJsonObject sample;
        sample[QStringLiteral("index_a")] = candidate.pair.indexA;
        sample[QStringLiteral("index_b")] = candidate.pair.indexB;
        sample[QStringLiteral("pair_key")] = candidate.pairKey;
        sample[QStringLiteral("priority_score")] = candidate.priorityScore;
        sample[QStringLiteral("overlap_score")] = candidate.overlapScore;
        sample[QStringLiteral("vocabulary_score")] = candidate.vocabularyScore;
        sample[QStringLiteral("sequence_score")] = candidate.sequenceScore;
        sample[QStringLiteral("sources")] = sourcesToJson(candidate.sources);
        sample[QStringLiteral("detail")] = candidate.detail;
        samples.append(sample);
    }
    object[QStringLiteral("candidate_samples")] = samples;
    return object;
}

QJsonArray matchesToJson(const MatchPhotosResult &result)
{
    QJsonArray array;
    for (const xjw::matchphotos::MatchPhotosMatchRecord &match : result.matches)
    {
        QJsonObject object;
        object[QStringLiteral("image0_path")] = match.image0Path;
        object[QStringLiteral("image1_path")] = match.image1Path;
        object[QStringLiteral("image0_match_file")] = match.image0MatchFilePath;
        object[QStringLiteral("image1_match_file")] = match.image1MatchFilePath;
        object[QStringLiteral("algorithm_id")] = match.algorithmId;
        object[QStringLiteral("algorithm_version")] = static_cast<qint64>(match.algorithmVersion);
        object[QStringLiteral("match_count")] = match.matchCount;
        object[QStringLiteral("geometric_inlier_count")] = match.geometricInlierCount;
        object[QStringLiteral("passed_geometry")] = match.passedGeometry;
        object[QStringLiteral("settings")] = match.settings;
        array.append(object);
    }
    return array;
}

QJsonArray imageMatchFilesToJson(const MatchPhotosResult &result)
{
    QJsonArray array;
    for (const xjw::matchphotos::MatchPhotosImageMatchRecord &image : result.imageMatchFiles)
    {
        QJsonObject object;
        object[QStringLiteral("image")] = image.imagePath;
        object[QStringLiteral("output")] = image.matchFilePath;
        object[QStringLiteral("neighbors")] = QJsonArray::fromStringList(image.neighborImagePaths);
        object[QStringLiteral("neighbor_variant_count")] = image.neighborVariantCount;
        object[QStringLiteral("settings")] = image.settings;
        array.append(object);
    }
    return array;
}

QJsonObject optionsToJson(const MatchPhotosOptions &options)
{
    QJsonObject object;
    object[QStringLiteral("algorithm_id")] = options.algorithmId;
    object[QStringLiteral("mask_apply_mode")] = options.maskApplyMode;
    object[QStringLiteral("max_image_dim")] = options.maxImageDim;
    object[QStringLiteral("keypoint_limit")] = options.maxKeypoints;
    object[QStringLiteral("keypoint_limit_per_mpx")] = options.keypointLimitPerMegapixel;
    object[QStringLiteral("tiepoint_limit")] = options.maxTiePointsPerImage;
    object[QStringLiteral("tiepoint_grid_columns")] = options.tiePointGridColumns;
    object[QStringLiteral("tiepoint_grid_rows")] = options.tiePointGridRows;
    object[QStringLiteral("tiepoint_grid_cell_limit")] = options.maxTiePointsPerGridCell;
    object[QStringLiteral("generic_preselection")] = options.useGenericPreselection;
    object[QStringLiteral("reference_preselection")] = options.useReferencePreselection;
    object[QStringLiteral("guided_image_matching")] = options.enableGuidedMatching;
    object[QStringLiteral("exclude_fixed_tie_points")] = options.excludeStationaryTiePoints;
    object[QStringLiteral("reuse_existing_matches")] = options.reuseExistingMatches;
    object[QStringLiteral("plan_only")] = options.planOnly;
    return object;
}

QJsonObject makeReport(const MatchPhotosResult &result,
                       const MatchPhotosOptions &options,
                       const QString &outputDir,
                       const QString &matchDir,
                       int imageCount,
                       double elapsedMs)
{
    QJsonObject report;
    report[QStringLiteral("success")] = result.success;
    report[QStringLiteral("error_message")] = result.errorMessage;
    report[QStringLiteral("image_count")] = imageCount;
    report[QStringLiteral("pair_match_count")] = static_cast<int>(result.matches.size());
    report[QStringLiteral("image_match_file_count")] =
        static_cast<int>(result.imageMatchFiles.size());
    report[QStringLiteral("track_count")] = result.trackCount;
    report[QStringLiteral("accepted_track_components")] = result.acceptedTrackComponents;
    report[QStringLiteral("rejected_track_conflict_components")] = result.rejectedTrackConflictComponents;
    report[QStringLiteral("tie_point_path")] = result.tiePointPath;
    report[QStringLiteral("output_dir")] = outputDir;
    report[QStringLiteral("match_dir")] = matchDir;
    report[QStringLiteral("elapsed_ms")] = elapsedMs;
    report[QStringLiteral("generated_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    report[QStringLiteral("options")] = optionsToJson(options);
    report[QStringLiteral("algorithm_plan")] = algorithmPlanToJson(result.algorithmPlan);
    report[QStringLiteral("pair_selection")] = pairSelectionToJson(result);
    report[QStringLiteral("stages")] = stagesToJson(result);
    report[QStringLiteral("pair_diagnostics")] = matchesToJson(result);
    report[QStringLiteral("image_match_files")] = imageMatchFilesToJson(result);
    report[QStringLiteral("track_summary")] = result.trackSummary;
    return report;
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication qtApplication(argc, argv);
    CLI::App app{"PlaScan 创建连接点 CLI — 注册算法匹配与连接点轨迹构建"};

    std::string inputPath;
    std::string outputDirArg;
    std::string projectPathArg;
    std::string chunkIdArg;
    std::string chunkNameArg;
    std::string qualityArg = "high";
    std::string deviceArg = "auto";
    std::string algorithmIdArg = "auto_sift";
    std::string lightGlueEngineArg;
    std::string lomaRPackageArg;
    int lomaRKeypointBudget = 0;
    std::string maskApplyModeArg = "none";
    std::string maskDirArg;
    std::string pairModeArg = "auto";
    int keypointLimit = 40000;
    int keypointLimitPerMpx = 0;
    int tiepointLimit = 4000;
    int maxImageDim = 0;
    int sequenceWindow = 4;
    int maxPairs = 0;
    int cudaDevice = 0;
    bool genericPreselection = true;
    bool noGenericPreselection = false;
    bool referencePreselection = false;
    bool guidedImageMatching = false;
    bool excludeFixedTiePoints = true;
    bool includeFixedTiePoints = false;
    bool noReuseMatches = false;
    bool planOnly = false;
    bool force = false;

    app.add_option("-i,--input", inputPath, "影像列表；每行支持 '<image>' 或 '<image> <camera.tsai>'")->required();
    app.add_option("-o,--output-dir", outputDirArg, "输出目录")->required();
    app.add_option("--project", projectPathArg, "无头项目路径，默认写到输出目录 headless.plascan");
    app.add_option("--chunk-id", chunkIdArg, "使用指定 UUID 的 Chunk");
    app.add_option("--chunk-name", chunkNameArg, "使用指定名称的 Chunk");
    app.add_option("--quality", qualityArg, "精度预设: auto, fast, high, highest, difficult, cpu, cuda");
    app.add_option("--device", deviceArg, "计算设备: auto, cpu, cuda, opencl, metal");
    app.add_option("--algorithm-id", algorithmIdArg,
                   "统一影像匹配算法 ID: auto_sift, sift_lightglue, loma_r")
        ->check(CLI::IsMember({"auto_sift", "sift_lightglue", "loma_r"}));
    app.add_option("--lightglue-engine", lightGlueEngineArg,
                   "LightGlue .onnx（推荐）或兼容的本机 .engine；留空时按模型目录自动查找");
    app.add_option("--loma-r-package", lomaRPackageArg,
                   "LoMa-R TensorRT JSON 清单；留空时按模型目录自动查找");
    app.add_option("--loma-r-keypoint-budget", lomaRKeypointBudget,
                   "LoMa-R 静态特征档位: 0(自动), 1024, 2048, 3840");
    app.add_option("--keypoint-limit", keypointLimit, "每张影像关键点限制");
    app.add_option("--keypoint-limit-per-mpx", keypointLimitPerMpx, "每百万像素关键点限制，0 表示不额外限制");
    app.add_option("--tiepoint-limit", tiepointLimit, "每张影像连接点限制");
    app.add_option("--max-image-dim", maxImageDim, "特征提取最长边限制，0 表示按预设");
    app.add_option("--mask-apply-mode", maskApplyModeArg, "蒙版应用阶段: none, keypoints, tiepoints");
    app.add_option("--mask-dir", maskDirArg, "蒙版目录，按影像文件名或 *_mask 文件名匹配");
    app.add_flag("--generic-preselection", genericPreselection, "启用通用预选");
    app.add_flag("--no-generic-preselection", noGenericPreselection, "禁用通用预选");
    app.add_flag("--reference-preselection", referencePreselection, "启用参考预选；需要完整相机文件");
    app.add_flag("--guided-image-matching", guidedImageMatching, "启用指导图像匹配");
    app.add_flag("--include-fixed-tie-points", includeFixedTiePoints, "连接点输出中包含固定/近静止连接点");
    app.add_flag("--exclude-fixed-tie-points", excludeFixedTiePoints, "排除固定/近静止连接点");
    app.add_option("--pair-mode", pairModeArg, "影像对模式: auto, exhaustive, sequence, manual-only");
    app.add_option("--sequence-window", sequenceWindow, "序列预选邻域窗口");
    app.add_option("--max-pairs", maxPairs, "候选影像对上限，0 表示不限制");
    app.add_option("--cuda-device", cudaDevice, "CUDA 设备 ID");
    app.add_flag("--no-reuse-matches", noReuseMatches,
                 "不复用兼容的逐影像匹配分片，强制重新提取并匹配");
    app.add_flag("--plan-only", planOnly, "只解析参数和规划候选对，不提取/匹配/构建连接点");
    app.add_flag("--force", force, "允许输出目录非空");

    CLI11_PARSE(app, argc, argv);

    if (lomaRKeypointBudget != 0 && lomaRKeypointBudget != 1024 &&
        lomaRKeypointBudget != 2048 && lomaRKeypointBudget != 3840)
    {
        std::fprintf(stderr,
                     "错误: --loma-r-keypoint-budget 只接受 0、1024、2048 或 3840\n");
        return cli::EXIT_ARG_ERR;
    }

    const QString inputList = xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(inputPath));
    const QString outputDir = xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(outputDirArg));
    const QString projectPath = projectPathArg.empty()
        ? QDir(outputDir).filePath(QStringLiteral("headless.plascan"))
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(projectPathArg));

    QString errorMessage;
    if (!xjw::cli::validateOutputDirectory(outputDir, force, &errorMessage) ||
        !xjw::cli::ensureDirectory(outputDir, &errorMessage))
    {
        std::fprintf(stderr, "错误: %s\n", qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    xjw::cli::PhotogrammetryListOptions listOptions;
    listOptions.allowImageOnlyRows = true;
    listOptions.loadCameras = referencePreselection;
    listOptions.requireExistingCameras = referencePreselection;
    std::vector<xjw::cli::PhotogrammetryInputItem> items;
    if (!xjw::cli::readPhotogrammetryImageList(inputList, listOptions, &items, &errorMessage))
    {
        std::fprintf(stderr, "列表读取失败: %s\n", qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    if (noGenericPreselection)
    {
        genericPreselection = false;
    }
    if (includeFixedTiePoints)
    {
        excludeFixedTiePoints = false;
    }
    if (referencePreselection &&
        xjw::cli::referenceCameraMap(items).size() < static_cast<int>(items.size()))
    {
        std::fprintf(stderr, "错误: 参考预选需要列表中每张影像都有可读取的相机文件。\n");
        return cli::EXIT_ARG_ERR;
    }

    xjw::common::project::ProjectSession projectSession;
    if (!projectSession.openOrCreate(
            projectPath,
            QFileInfo(projectPath).completeBaseName(),
            &errorMessage))
    {
        std::fprintf(stderr,
                     "工程打开/创建失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }
    if (!chunkIdArg.empty() && !chunkNameArg.empty())
    {
        std::fprintf(stderr, "错误: --chunk-id 与 --chunk-name 不能同时使用\n");
        return cli::EXIT_ARG_ERR;
    }
    if (!projectSession.selectChunk(
            xjw::cli::fromStdString(chunkIdArg),
            xjw::cli::fromStdString(chunkNameArg),
            &errorMessage))
    {
        std::fprintf(stderr,
                     "Chunk 选择失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }
    if (!projectSession.mergeImages(
            xjw::cli::inputItemsToJson(items), &errorMessage)
        || !projectSession.save(&errorMessage))
    {
        std::fprintf(stderr,
                     "工程影像初始化失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    MatchPhotosOptions options;
    options.profile = profileFromToken(xjw::cli::fromStdString(qualityArg));
    options.device = deviceFromToken(xjw::cli::fromStdString(deviceArg));
    options.pairPolicy = xjw::matchphotos::makePairSelectionPolicy(pairPresetFromProfile(options.profile));
    options.pairPolicy.mode = pairModeFromToken(xjw::cli::fromStdString(pairModeArg));
    options.pairPolicy.sequenceWindow = std::max(1, sequenceWindow);
    options.pairPolicy.maxPairs = std::max(0, maxPairs);
    options.algorithmId = xjw::cli::normalizedToken(
        algorithmIdArg, QStringLiteral("auto_sift"));
    options.lightGlueTensorRtEnginePath = lightGlueEngineArg.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(lightGlueEngineArg));
    options.lomaRTensorRtPackagePath = lomaRPackageArg.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(lomaRPackageArg));
    options.lomaRKeypointBudget = lomaRKeypointBudget;
    options.maskApplyMode = xjw::cli::normalizedToken(maskApplyModeArg, QStringLiteral("none"));
    options.maxImageDim = maxImageDim;
    options.maxKeypoints = std::max(0, keypointLimit);
    options.keypointLimitPerMegapixel = std::max(0, keypointLimitPerMpx);
    options.cudaDevice = std::max(0, cudaDevice);
    options.maxTiePointsPerImage = std::max(0, tiepointLimit);
    options.tiePointGridColumns = 4;
    options.tiePointGridRows = 4;
    options.maxTiePointsPerGridCell = options.maxTiePointsPerImage > 0
        ? std::max(1, options.maxTiePointsPerImage / options.tiePointGridColumns)
        : 0;
    options.enableGuidedMatching = guidedImageMatching;
    options.useExplicitKeypointLimit = keypointLimit > 0 || keypointLimitPerMpx > 0;
    options.useGenericPreselection = genericPreselection;
    options.useReferencePreselection = referencePreselection;
    options.excludeStationaryTiePoints = excludeFixedTiePoints;
    options.reuseExistingMatches = !noReuseMatches;
    options.planOnly = planOnly;

    const QString assetDir = QDir(projectSession.activeChunkRoot())
        .filePath(QStringLiteral("assets"));
    xjw::matchphotos::MatchPhotosContext context;
    context.projectPath = projectPath;
    context.workingDirectory = assetDir;
    context.matchDirectory = QDir(assetDir).filePath(QStringLiteral("image_matches"));
    context.pairInput.images = xjw::cli::imagePaths(items);
    context.referenceCameras = xjw::cli::referenceCameraMap(items);
    context.maskPaths = xjw::cli::maskPathsFromDirectory(
        xjw::cli::fromStdString(maskDirArg), context.pairInput.images);

    std::atomic_bool cancelFlag(false);
    std::atomic_int progressCount(0);
    context.cancelFlag = &cancelFlag;
    context.progressCount = &progressCount;
    context.progressCallback = [](const QString &stageId, const QString &message, int current, int maximum)
    {
        std::fprintf(stdout,
                     "[%s %d/%d] %s\n",
                     qUtf8Printable(stageId),
                     current,
                     maximum,
                     qUtf8Printable(message));
        std::fflush(stdout);
    };

    QElapsedTimer timer;
    timer.start();
    xjw::matchphotos::MatchPhotosTask task(options);
    const MatchPhotosResult result = task.run(context);

    const QString reportPath =
        QDir(projectSession.activeChunkRoot()).filePath(
            QStringLiteral("reports/match_photos_report.json"));
    const QJsonObject report = makeReport(result,
                                          options,
                                          outputDir,
                                          context.matchDirectory,
                                          context.pairInput.images.size(),
                                          static_cast<double>(timer.elapsed()));
    if (!xjw::cli::writeJsonFile(reportPath, report, &errorMessage))
    {
        std::fprintf(stderr, "报告写入失败: %s\n", qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    // 工程索引只登记“一幅影像一个分片”的最终结果。SIFT 描述子仅存在于任务
    // 内存，像对诊断也已封装在对应 `.pimatch` 中，不能再写旧 ipfind/ipmatch 数组。
    for (const xjw::matchphotos::MatchPhotosImageMatchRecord &image :
         result.imageMatchFiles)
    {
        QJsonObject record{
            {QStringLiteral("image"), image.imagePath},
            {QStringLiteral("output"), image.matchFilePath},
            {QStringLiteral("neighbors"),
             QJsonArray::fromStringList(image.neighborImagePaths)},
            {QStringLiteral("neighbor_variant_count"), image.neighborVariantCount},
            {QStringLiteral("settings"), image.settings},
            {QStringLiteral("created_at"),
             QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)}
        };
        projectSession.upsertResultByPath(
            QStringLiteral("image_match_results"),
            QStringLiteral("output"),
            record);
    }
    QJsonObject reportRecord = report;
    reportRecord[QStringLiteral("kind")] =
        QStringLiteral("match_photos_cli");
    reportRecord[QStringLiteral("path")] = reportPath;
    projectSession.upsertResultByPath(
        QStringLiteral("report_results"),
        QStringLiteral("path"),
        reportRecord);
    if (!projectSession.save(&errorMessage))
    {
        std::fprintf(stderr,
                     "连接点结果已生成，但 Chunk 写回失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    std::fprintf(stdout, "status=%s\n", result.success ? "ok" : "failed");
    std::fprintf(stdout, "images=%d\n", static_cast<int>(context.pairInput.images.size()));
    std::fprintf(stdout, "pair_matches=%d\n", static_cast<int>(result.matches.size()));
    std::fprintf(stdout, "image_match_files=%d\n",
                 static_cast<int>(result.imageMatchFiles.size()));
    std::fprintf(stdout, "tracks=%d\n", result.trackCount);
    std::fprintf(stdout, "match_photos_report.json=%s\n", qUtf8Printable(reportPath));
    std::fprintf(stdout,
                 "project=%s\nchunk=%d\n",
                 qUtf8Printable(projectSession.projectPath()),
                 projectSession.activeChunk().directory);
    if (!result.tiePointPath.isEmpty())
    {
        std::fprintf(stdout, "tie_points=%s\n", qUtf8Printable(result.tiePointPath));
    }
    if (!result.success)
    {
        std::fprintf(stderr, "连接点匹配失败: %s\n", qUtf8Printable(result.errorMessage));
        return cli::EXIT_ALGO_ERR;
    }
    return cli::EXIT_OK;
}
