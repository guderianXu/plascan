#include "cli_common.h"

#include "DepthMapGenerator.h"
#include "MvsSourcePlanner.h"
#include "MvsWorkspaceManifest.h"
#include "MvsWorkspaceReplay.h"
#include "ProjectDenseWorkflowConfig.h"
#include "SparseCloudPreprocessor.h"
#include "io/PathIO.h"

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QTimer>

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace
{

bool ensureFreshOutputDirectory(const QString &path, QString *errorMessage)
{
    QDir directory(path);
    if (directory.exists())
    {
        const QStringList entries = directory.entryList(
            QDir::AllEntries | QDir::NoDotAndDotDot);
        if (!entries.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "输出目录非空，MVS replay 拒绝覆盖已有结果：%1")
                                    .arg(QDir::toNativeSeparators(path));
            }
            return false;
        }
        return true;
    }
    if (!QDir().mkpath(path))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建输出目录：%1")
                                .arg(QDir::toNativeSeparators(path));
        }
        return false;
    }
    return true;
}

bool writeReplayReport(const QString &path,
                       const QJsonObject &report,
                       QString *errorMessage)
{
    QSaveFile output(path);
    const QByteArray json = QJsonDocument(report).toJson(QJsonDocument::Indented);
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(json) != json.size() ||
        !output.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入 MVS replay 报告：%1")
                                .arg(QDir::toNativeSeparators(path));
        }
        return false;
    }
    return true;
}

bool validateFreshOutputDirectory(const QString &path, QString *errorMessage)
{
    const QFileInfo outputInfo(path);
    if (!outputInfo.exists())
    {
        return true;
    }
    if (!outputInfo.isDir())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS replay 输出路径不是目录：%1")
                                .arg(QDir::toNativeSeparators(path));
        }
        return false;
    }
    if (!QDir(path).entryList(QDir::AllEntries | QDir::NoDotAndDotDot).isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "输出目录非空，MVS replay 拒绝覆盖已有结果：%1")
                                .arg(QDir::toNativeSeparators(path));
        }
        return false;
    }
    return true;
}

bool isVerifiedSourcePlanEntry(const QJsonObject &entry)
{
    const QString status = entry.value(QStringLiteral("verification_status"))
                               .toString()
                               .trimmed()
                               .toLower();
    if (!status.isEmpty())
    {
        return status == QStringLiteral("verified");
    }
    return entry.value(QStringLiteral("verified_pair_geometry")).toBool(false);
}

QString resolveManifestPath(const QString &manifestPath, const QString &storedPath)
{
    const QString trimmedPath = storedPath.trimmed();
    if (trimmedPath.isEmpty())
    {
        return QString();
    }

    QFileInfo pathInfo(trimmedPath);
    if (pathInfo.isRelative())
    {
        pathInfo.setFile(
            QDir(QFileInfo(manifestPath).absolutePath()).filePath(trimmedPath));
    }

    const QString canonicalPath = pathInfo.canonicalFilePath();
    return QDir::cleanPath(
        canonicalPath.isEmpty() ? pathInfo.absoluteFilePath() : canonicalPath);
}

bool loadManifestSourcePlanPairQualities(
    const QString &manifestPath,
    std::vector<xjw::mvs::MvsSourcePairQuality> *qualities,
    QString *errorMessage)
{
    if (!qualities)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS source_plan 像对输出参数为空");
        }
        return false;
    }
    qualities->clear();

    xjw::mvs::MvsWorkspaceManifest manifest;
    QString manifestError;
    if (!manifest.load(manifestPath, &manifestError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法加载 MVS manifest source_plan：%1（%2）")
                                .arg(QDir::toNativeSeparators(manifestPath), manifestError);
        }
        return false;
    }

    std::map<int, QString> imageByIndex;
    for (const xjw::mvs::MvsDepthFrameRecord &record : manifest.frames())
    {
        if (record.refIndex >= 0 && !record.refImage.trimmed().isEmpty())
        {
            imageByIndex[record.refIndex] = resolveManifestPath(
                manifestPath, record.refImage);
        }
    }

    for (const xjw::mvs::MvsDepthFrameRecord &record : manifest.frames())
    {
        if (record.refImage.trimmed().isEmpty())
        {
            continue;
        }
        const QString referenceImage = resolveManifestPath(
            manifestPath, record.refImage);

        for (const QJsonValue &value : record.sourcePlan)
        {
            const QJsonObject entry = value.toObject();
            if (entry.isEmpty() || !isVerifiedSourcePlanEntry(entry))
            {
                continue;
            }

            QString sourceImage =
                entry.value(QStringLiteral("source_image")).toString().trimmed();
            if (sourceImage.isEmpty())
            {
                const int sourceIndex =
                    entry.value(QStringLiteral("view_index")).toInt(-1);
                const auto imageIt = imageByIndex.find(sourceIndex);
                if (imageIt != imageByIndex.end())
                {
                    sourceImage = imageIt->second;
                }
            }
            if (sourceImage.isEmpty())
            {
                continue;
            }
            sourceImage = resolveManifestPath(manifestPath, sourceImage);

            xjw::mvs::MvsSourcePairQuality quality;
            quality.imageA = xjw::common::io::toUtf8Path(referenceImage);
            quality.imageB = xjw::common::io::toUtf8Path(sourceImage);
            quality.totalMatches = std::max(
                0, entry.value(QStringLiteral("pair_total_matches")).toInt());
            quality.geometricInliers = std::max(
                0, entry.value(QStringLiteral("geometric_inliers")).toInt());
            const QJsonValue pairCoverage =
                entry.value(QStringLiteral("pair_coverage_score"));
            const double coverage = pairCoverage.isDouble()
                ? pairCoverage.toDouble()
                : entry.value(QStringLiteral("coverage_score")).toDouble(0.0);
            quality.geometricCoverage = static_cast<float>(
                std::clamp(coverage, 0.0, 1.0));
            quality.verified = true;
            quality.hasVerificationStatistics = true;
            QString reason = entry.value(QStringLiteral("verification_reason"))
                                 .toString()
                                 .trimmed();
            if (reason.isEmpty())
            {
                reason = QStringLiteral("verified_from_manifest_source_plan");
            }
            quality.verificationReason = xjw::common::io::toUtf8Path(reason);
            qualities->push_back(std::move(quality));
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication qtApp(argc, argv);
    CLI::App app{"PlaScan 现有 MVS 工作区深度重放工具"};

    std::string inputManifest;
    std::string pairAuditReport;
    std::string sparseCloudPath;
    std::string maskDirectory;
    std::string outputDirectory;
    std::string quality = "highest";
    std::string sceneProfile = "orbital_object";
    std::string depthFilter = "mild";
    std::string device = "cuda";
    int sourceViews = 4;
    int threads = 7;
    int gpuFrameWorkers = 1;
    int cpuFrameWorkers = 0;
    bool saveLevels = false;
    bool depthPoseCandidates = false;
    bool disableTargetedGapRecovery = false;

    app.add_option("--input-manifest", inputManifest,
                   "现有 mvs_manifest.json，用于读取影像顺序和相机")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--pair-audit", pairAuditReport,
                   "可选：mvs_pair_audit_cli 生成的 JSON 报告；缺失时复用 manifest source_plan")
        ->check(CLI::ExistingFile);
    app.add_option("--sparse-cloud", sparseCloudPath,
                   "用于深度范围与可见性预处理的 SFM 稀疏点云 PLY")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--mask-dir", maskDirectory,
                   "可选项目排除蒙版目录，要求每张影像均有 <stem>_mask.png");
    app.add_option("-o,--output-dir", outputDirectory,
                   "必须为空或不存在的独立 MVS 输出目录")
        ->required();
    app.add_option("--quality", quality, "深度质量：highest/high/medium/low/lowest")
        ->check(CLI::IsMember({"highest", "high", "medium", "low", "lowest"}));
    app.add_option("--scene-profile", sceneProfile,
                   "场景：auto/orbital_object/aerial_terrain")
        ->check(CLI::IsMember({"auto", "orbital_object", "aerial_terrain"}));
    app.add_option("--depth-filter", depthFilter,
                   "过滤：auto/mild/moderate/aggressive")
        ->check(CLI::IsMember({"auto", "mild", "moderate", "aggressive"}));
    app.add_option("--device", device, "设备：cuda/cpu")
        ->check(CLI::IsMember({"cuda", "cpu"}));
    app.add_option("--source-views", sourceViews, "请求源视图数")
        ->check(CLI::Range(1, 16));
    app.add_option("--threads", threads, "CPU 线程预算")
        ->check(CLI::Range(1, 64));
    app.add_option("--gpu-frame-workers", gpuFrameWorkers, "GPU 帧并发数")
        ->check(CLI::Range(0, 2));
    app.add_option("--cpu-frame-workers", cpuFrameWorkers, "CPU 帧并发数")
        ->check(CLI::Range(0, 4));
    app.add_flag("--save-levels", saveLevels, "保存中间深度金字塔层");
    app.add_flag(
        "--depth-pose-candidates",
        depthPoseCandidates,
        "实验：输出深度约束派生相机候选与安全门诊断；不覆盖项目相机或重算深度");
    app.add_flag(
        "--disable-targeted-gap-recovery",
        disableTargetedGapRecovery,
        "诊断：关闭缺口定向 PatchMatch 恢复，用于同输入 A/B 对比");
    CLI11_PARSE(app, argc, argv);

    const QString manifestPath = QFileInfo(
        QString::fromUtf8(inputManifest.c_str())).absoluteFilePath();
    const QString auditPath = pairAuditReport.empty()
        ? QString()
        : QFileInfo(QString::fromUtf8(pairAuditReport.c_str())).absoluteFilePath();
    const QString sparsePath = QFileInfo(
        QString::fromUtf8(sparseCloudPath.c_str())).absoluteFilePath();
    const QString maskDir = maskDirectory.empty()
        ? QString()
        : QFileInfo(QString::fromUtf8(maskDirectory.c_str())).absoluteFilePath();
    const QString outputDir = QFileInfo(
        QString::fromUtf8(outputDirectory.c_str())).absoluteFilePath();

    QString error;
    if (!validateFreshOutputDirectory(outputDir, &error))
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    std::vector<xjw::mvs::CameraView> views;
    if (!xjw::mvs::loadMvsReplayViews(
            manifestPath, maskDir, &views, &error))
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    std::vector<xjw::mvs::MvsSourcePairQuality> pairQualities;
    xjw::mvs::MvsPairAuditSummary pairSummary;
    QString pairEvidenceProvenance;
    if (!auditPath.isEmpty())
    {
        if (!xjw::mvs::loadMvsPairAuditReport(
                auditPath, &pairQualities, &pairSummary, &error))
        {
            std::fprintf(stderr, "%s\n", error.toUtf8().constData());
            return cli::EXIT_IO_ERR;
        }
        pairEvidenceProvenance = QStringLiteral("pair_audit");
    }
    else
    {
        if (!loadManifestSourcePlanPairQualities(
                manifestPath, &pairQualities, &error))
        {
            std::fprintf(stderr, "%s\n", error.toUtf8().constData());
            return cli::EXIT_IO_ERR;
        }
        pairEvidenceProvenance = QStringLiteral("manifest_source_plan");
    }

    std::vector<std::string> currentImages;
    currentImages.reserve(views.size());
    for (const xjw::mvs::CameraView &view : views)
    {
        currentImages.push_back(view.imagePath);
    }
    pairQualities = xjw::mvs::filterMvsSourcePairQualitiesForImages(
        pairQualities, currentImages);
    const int verifiedCurrentPairs = static_cast<int>(std::count_if(
        pairQualities.cbegin(),
        pairQualities.cend(),
        [](const xjw::mvs::MvsSourcePairQuality &quality)
        {
            return quality.verified && quality.hasVerificationStatistics;
        }));
    if (verifiedCurrentPairs <= 0)
    {
        if (auditPath.isEmpty())
        {
            std::fprintf(
                stderr,
                "未提供 --pair-audit，且 input manifest 的 source_plan "
                "不含当前影像集合的已验证 MVS 源像对，拒绝重放。\n");
        }
        else
        {
            std::fprintf(stderr,
                         "当前影像集合没有通过几何验证的 MVS 源像对，拒绝重放。\n");
        }
        return cli::EXIT_ALGO_ERR;
    }
    if (auditPath.isEmpty())
    {
        pairSummary.auditedPairCount = static_cast<int>(pairQualities.size());
        pairSummary.verifiedPairCount = verifiedCurrentPairs;
        pairSummary.failedPairCount = 0;
        pairSummary.missingStatisticsPairCount = 0;
    }

    std::fprintf(stdout,
                 "pair_evidence_provenance=%s verified_pairs=%d\n",
                 pairEvidenceProvenance.toUtf8().constData(),
                 verifiedCurrentPairs);
    std::fflush(stdout);

    xjw::mvs::SparseCloudPreprocessor preprocessor(
        plapoint::ProcessingDevice::CPU);
    xjw::mvs::PreprocessResult preprocessResult;
    std::string preprocessError;
    if (!preprocessor.run(xjw::common::io::toUtf8Path(sparsePath),
                          views,
                          preprocessResult,
                          &preprocessError))
    {
        std::fprintf(stderr,
                     "稀疏点云预处理失败：%s\n",
                     preprocessError.c_str());
        return cli::EXIT_ALGO_ERR;
    }

    // Keep input/evidence validation side-effect free.  Creating the replay
    // directory only after all preflight checks avoids leaving an empty
    // output that looks like a started or reusable workspace.
    if (!ensureFreshOutputDirectory(outputDir, &error))
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    const QJsonObject settingsJson{
        {QStringLiteral("qualityProfile"), QString::fromStdString(quality)},
        {QStringLiteral("sceneProfile"), QString::fromStdString(sceneProfile)},
        {QStringLiteral("depthFilterMode"), QString::fromStdString(depthFilter)},
        {QStringLiteral("minViews"), sourceViews},
        {QStringLiteral("threads"), threads},
        {QStringLiteral("gpu_frame_workers"), gpuFrameWorkers},
        {QStringLiteral("cpu_frame_workers"), cpuFrameWorkers},
        {QStringLiteral("cuda"), device == "cuda"},
        {QStringLiteral("saveIntermediatePyramidLevels"), saveLevels},
        {QStringLiteral("pipeline_mode"), true}
    };
    const xjw::gui::project::DenseGenerationSettings settings =
        xjw::gui::project::denseGenerationSettingsFromJson(settingsJson);
    xjw::mvs::DepthGenConfig config =
        xjw::gui::project::buildDepthGenConfig(
            settings, static_cast<int>(views.size()));
    config.runFusion = false;
    config.saveIntermediateDepthMaps = true;
    config.intermediateDir = xjw::common::io::toUtf8Path(outputDir);
    config.inputSignature =
        QStringLiteral("mvs-replay:%1:%2:%3")
            .arg(QFileInfo(manifestPath).lastModified().toMSecsSinceEpoch())
            .arg(pairEvidenceProvenance)
            .arg(auditPath.isEmpty()
                     ? 0
                     : QFileInfo(auditPath).lastModified().toMSecsSinceEpoch())
            .toStdString();
    config.sourcePairQualities = std::move(pairQualities);
    config.requireVerifiedSourcePairs = true;
    config.depthPoseRefinement.enabled = depthPoseCandidates;
    config.enableTargetedGapRecovery = !disableTargetedGapRecovery;

    std::fprintf(stdout,
                 "views=%zu verified_pairs=%d failed_pairs=%d "
                 "missing_stats_pairs=%d requested_sources=%d "
                 "pair_evidence_provenance=%s\n",
                 views.size(),
                 verifiedCurrentPairs,
                 pairSummary.failedPairCount,
                 pairSummary.missingStatisticsPairCount,
                 sourceViews,
                 pairEvidenceProvenance.toUtf8().constData());
    std::fflush(stdout);

    xjw::mvs::DepthMapGenerator generator;
    generator.setViews(views);
    generator.setSparseCloud(preprocessResult.cloud);
    generator.setConfig(config);
    generator.setOutputDir(xjw::common::io::toUtf8Path(outputDir));

    QEventLoop loop;
    bool success = false;
    QString generatorError;
    QJsonArray artifacts;
    int lastProgress = -1;
    QObject::connect(
        &generator,
        &xjw::mvs::DepthMapGenerator::progressChanged,
        &loop,
        [&lastProgress](const QString &stage, float ratio)
        {
            const int progress = std::clamp(
                static_cast<int>(ratio * 100.0f), 0, 100);
            if (progress >= lastProgress + 5 || progress == 100)
            {
                lastProgress = progress;
                std::fprintf(stdout,
                             "progress=%d stage=%s\n",
                             progress,
                             stage.toUtf8().constData());
                std::fflush(stdout);
            }
        });
    QObject::connect(
        &generator,
        &xjw::mvs::DepthMapGenerator::errorOccurred,
        &loop,
        [&generatorError](const QString &message)
        {
            generatorError = message;
            std::fprintf(stderr, "mvs_error=%s\n", message.toUtf8().constData());
        });
    QObject::connect(
        &generator,
        &xjw::mvs::DepthMapGenerator::depthMapArtifactSaved,
        &loop,
        [&artifacts](const QJsonObject &artifact)
        {
            artifacts.append(artifact);
        });
    QObject::connect(
        &generator,
        &xjw::mvs::DepthMapGenerator::finished,
        &loop,
        [&loop, &success](bool ok)
        {
            success = ok;
            loop.quit();
        });
    QTimer::singleShot(0, &generator, &xjw::mvs::DepthMapGenerator::start);
    loop.exec();

    const QJsonObject report{
        {QStringLiteral("schema"), QStringLiteral("plascan_mvs_depth_replay_v1")},
        {QStringLiteral("status"),
         success ? QStringLiteral("ok") : QStringLiteral("failed")},
        {QStringLiteral("input_manifest"), manifestPath},
        {QStringLiteral("pair_audit"), auditPath},
        {QStringLiteral("pair_evidence_provenance"), pairEvidenceProvenance},
        {QStringLiteral("sparse_cloud"), sparsePath},
        {QStringLiteral("mask_directory"), maskDir},
        {QStringLiteral("output_directory"), outputDir},
        {QStringLiteral("view_count"), static_cast<int>(views.size())},
        {QStringLiteral("audited_pair_count"), pairSummary.auditedPairCount},
        {QStringLiteral("verified_pair_count"), verifiedCurrentPairs},
        {QStringLiteral("failed_pair_count"), pairSummary.failedPairCount},
        {QStringLiteral("missing_statistics_pair_count"),
         pairSummary.missingStatisticsPairCount},
        {QStringLiteral("requested_source_view_count"), sourceViews},
        {QStringLiteral("quality"), QString::fromStdString(quality)},
        {QStringLiteral("scene_profile"), QString::fromStdString(sceneProfile)},
        {QStringLiteral("depth_filter"), QString::fromStdString(depthFilter)},
        {QStringLiteral("depth_pose_candidates"), depthPoseCandidates},
        {QStringLiteral("targeted_gap_recovery"),
         !disableTargetedGapRecovery},
        {QStringLiteral("targeted_gap_recovery_source_count"),
         config.targetedGapRecoverySourceCount},
        {QStringLiteral("targeted_gap_recovery_hypothesis_count"),
         config.targetedGapRecoveryHypothesisCount},
        {QStringLiteral("depth_artifact_count"), artifacts.size()},
        {QStringLiteral("depth_artifacts"), artifacts},
        {QStringLiteral("error"), generatorError}
    };
    const QString reportPath =
        QDir(outputDir).filePath(QStringLiteral("mvs_replay_report.json"));
    if (!writeReplayReport(reportPath, report, &error))
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    std::fprintf(stdout,
                 "status=%s artifacts=%d report=%s\n",
                 success ? "ok" : "failed",
                 static_cast<int>(artifacts.size()),
                 QDir::toNativeSeparators(reportPath).toUtf8().constData());
    return success ? cli::EXIT_OK : cli::EXIT_ALGO_ERR;
}
