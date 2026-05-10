#include "compat/QtTorchMacroGuard.h"

#include "SuperGlueMatcher.h"
#include "lightglue/LightGlueMatcher.h"
#include "SuperGlueMatchIO.h"
#include "MatchOutlierRejector.h"
#include "FeatureFileIO.h"
#include "SuperPoint.h"
#include "TraditionalFeatureMatcher.h"
#include "FeatureData.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "AlgorithmCompat.h"
#include "FeatureMatchRunner.h"
#include "Logger.h"
#include "ProjectIO.h"
#include "ProjectManager.h"
#include "ProjectSupportUtils.h"
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QCoreApplication>
#include <QDateTime>
#include <QProcess>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThread>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>

namespace
{

QString findModelFile(const QString &modelName)
{
    QStringList candidates;

    const QString envModelDir = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!envModelDir.isEmpty())
    {
        candidates.append(QDir(envModelDir).filePath(modelName));
    }

#ifdef PLASCAN_SOURCE_DIR
    candidates.append(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("resources/models/%1").arg(modelName)));
#endif

    const QString exePath = QCoreApplication::applicationDirPath();
    candidates.append(QDir(exePath).filePath(QStringLiteral("../models/%1").arg(modelName)));
    candidates.append(QStringLiteral("models/%1").arg(modelName));

    for (const QString &candidate : candidates)
    {
        if (QFile::exists(candidate))
        {
            return candidate;
        }
    }
    return QString();
}

} // namespace

void FeatureMatchRunner::run(const QJsonObject &config, const QStringList &imagePairs, ProjectManager *projectManager)
{
    std::atomic<bool> neverCancel{false};
    std::atomic<int> dummy{0};
    run(config, imagePairs, projectManager, neverCancel, dummy);
}

void FeatureMatchRunner::run(const QJsonObject &config, const QStringList &imagePairs,
                           ProjectManager *projectManager, std::atomic<bool> &cancelFlag)
{
    std::atomic<int> dummy{0};
    run(config, imagePairs, projectManager, cancelFlag, dummy);
}

void FeatureMatchRunner::run(const QJsonObject &config, const QStringList &imagePairs,
                           ProjectManager *projectManager, std::atomic<bool> &cancelFlag,
                           std::atomic<int> &progressCount)
{
    const QString matchAlgorithm = config.value("match_algorithm").toString("superglue").trimmed().toLower();
    const QString rawSuffix      = config.value("feature_suffix").toString();

    // "__all__" mode: try all compatible feature suffixes
    QStringList suffixesToTry;
    if (rawSuffix == "__all__" && xjw::feature_match::compatibleFeatureSuffixes(matchAlgorithm).size() > 1) {
        suffixesToTry = xjw::feature_match::compatibleFeatureSuffixes(matchAlgorithm);
        LOG_INFO("All-features mode: will try %d suffixes (%s)",
                 static_cast<int>(suffixesToTry.size()),
                 qUtf8Printable(suffixesToTry.join(", ")));
    } else {
        suffixesToTry << rawSuffix;
    }

    // "__all__" mode: recursively call run() for each suffix
    if (suffixesToTry.size() > 1)
    {
        for (const auto &suf : suffixesToTry)
        {
            if (cancelFlag.load()) break;
            QJsonObject cfg = config;
            cfg["feature_suffix"] = suf;
            LOG_INFO("All-features: trying suffix %s", qUtf8Printable(suf));
            run(cfg, imagePairs, projectManager, cancelFlag, progressCount);
        }
        return;
    }

    const QString featureSuffix = rawSuffix;
    const bool isSuperGlueMatch = (matchAlgorithm == QStringLiteral("superglue"));
    const bool isLightGlueMatch = (matchAlgorithm == QStringLiteral("lightglue"));
    const bool isLoftrMatch     = (matchAlgorithm == QStringLiteral("loftr"));
    const bool isRomaMatch      = (matchAlgorithm == QStringLiteral("roma"));
    const bool isPythonE2E      = isLoftrMatch || isRomaMatch;
    const bool isBfFlann        = (matchAlgorithm == QStringLiteral("orb_bf_hamming") ||
                                   matchAlgorithm == QStringLiteral("sift_bf_l2") ||
                                   matchAlgorithm == QStringLiteral("sift_flann"));
    LOG_INFO("%s", qUtf8Printable(QString("开始特征匹配(%1): %2 对影像")
        .arg(matchAlgorithm)
        .arg(imagePairs.size())));
    
    if (imagePairs.isEmpty()) 
    {
        LOG_WARN("匹配对列表为空，无需执行");
        return;
    }

    const QString plascanPath = projectManager ? projectManager->currentProjectPath() : QString();
    const QJsonObject currentMeta = projectManager ? projectManager->currentMeta() : QJsonObject();
    
    // 创建输出目录
    QString outputDir = config["output_dir"].toString();
    if (outputDir.isEmpty()) 
    {
        const QString assetsDir = ProjectIO::projectAssetsDir(projectManager->currentProjectPath());
        outputDir = QDir(assetsDir).filePath(QStringLiteral("matches"));
    }
    
    QDir dir;
    if (!dir.exists(outputDir)) 
    {
        if (!dir.mkpath(outputDir)) 
        { 
            LOG_ERROR("%s", qUtf8Printable(QString("无法创建输出目录: %1").arg(outputDir)));
            return;
        }
    }
    
    // ---- Python E2E branch (LoFTR / RoMa) ----â
    if (isPythonE2E)
    {
        const QString pyModelType = config["model_type"].toString("outdoor");

        const QString scriptName = isLoftrMatch ? QStringLiteral("run_loftr.py") : QStringLiteral("match_roma.py");
        QStringList scriptCandidates;
        scriptCandidates << QCoreApplication::applicationDirPath() + "/../scripts/" + scriptName;
        scriptCandidates << QCoreApplication::applicationDirPath() + "/../../scripts/" + scriptName;
        QString scriptPath;
        for (const auto &c : scriptCandidates)
        {
            if (QFile::exists(c)) { scriptPath = c; break; }
        }
        if (scriptPath.isEmpty())
        {
            LOG_ERROR("E2E script not found: %s", qUtf8Printable(scriptName));
            return;
        }

        for (const QString &pairStr : imagePairs)
        {
            if (cancelFlag.load()) break;
            struct PairDone { std::atomic<int> &cnt; ~PairDone() { cnt.fetch_add(1); } } _done{progressCount};

            QStringList parts = pairStr.split("__");
            if (parts.size() != 2)
            {
                LOG_WARN("%s", qUtf8Printable(QString("Invalid pair format: %1").arg(pairStr)));
                continue;
            }

            const QString imageToken0 = QFileInfo(parts[0]).fileName();
            const QString imageToken1 = QFileInfo(parts[1]).fileName();
            const QString imagePath0 = xjw::gui::project::resolveProjectImagePathFromToken(imageToken0, currentMeta);
            const QString imagePath1 = xjw::gui::project::resolveProjectImagePathFromToken(imageToken1, currentMeta);

            if (imagePath0.isEmpty() || imagePath1.isEmpty())
            {
                LOG_WARN("%s", qUtf8Printable(QString("Cannot resolve image path: %1").arg(pairStr)));
                continue;
            }

            const QString baseName0 = QFileInfo(imagePath0).completeBaseName();
            const QString baseName1 = QFileInfo(imagePath1).completeBaseName();
            const QString outPath = QDir(outputDir).filePath(baseName0 + "__" + baseName1 + ".match");

            QStringList args;
            args << scriptPath << "-L" << imagePath0 << "-R" << imagePath1 << "-o" << outPath;
            if (isRomaMatch)
                args << "--scene" << pyModelType;

            QProcess process;
            process.start("python3", args);
            if (!process.waitForFinished(300000))
            {
                LOG_ERROR("%s", qUtf8Printable(QString("Python E2E timeout: %1").arg(pairStr)));
                process.kill();
                continue;
            }

            if (process.exitCode() != 0)
            {
                LOG_ERROR("%s", qUtf8Printable(QString("Python E2E failed %1: %2")
                    .arg(pairStr).arg(QString::fromUtf8(process.readAllStandardError()))));
                continue;
            }

            LOG_INFO("%s", qUtf8Printable(QString("E2E match done: %1").arg(baseName0 + "__" + baseName1)));
        }

        LOG_INFO("%s", qUtf8Printable(QString("Python E2E (%1) done").arg(matchAlgorithm)));
        return;
    }

    // 构建 SuperGlueConfig
    superglue::SuperGlueConfig sgConfig;
    sgConfig.match_threshold = static_cast<float>(config["match_threshold"].toDouble(0.15));
    sgConfig.max_keypoints = config["max_keypoints"].toInt(-1);
    sgConfig.batch_size = config["batch_size"].toInt(1);
    sgConfig.sinkhorn_iterations = config["sinkhorn_iterations"].toInt(150);
    sgConfig.num_threads = config["num_threads"].toInt(-1);
    sgConfig.verbose = config["verbose"].toBool(false);
    sgConfig.enable_csv_output = config["save_csv"].toBool(false);
    sgConfig.enable_visualization = config["save_visualization"].toBool(false);

    superglue::OutlierFilterConfig outlierConfig;
    const QString outlierMethod = config["outlier_method"].toString("fundamental_usac_magsac");
    if (outlierMethod == "fundamental") 
    {
        outlierConfig.method = superglue::OutlierMethod::FundamentalRansac;
    } 
    else if (outlierMethod == "fundamental_usac_magsac") 
    {
        outlierConfig.method = superglue::OutlierMethod::FundamentalUsacMagsac;
    } 
    else if (outlierMethod == "homography") 
    {
        outlierConfig.method = superglue::OutlierMethod::HomographyRansac;
    } 
    else if (outlierMethod == "affine") 
    {
        outlierConfig.method = superglue::OutlierMethod::AffineRansac;
    } 
    else 
    {
        outlierConfig.method = superglue::OutlierMethod::None;
    }
    outlierConfig.reprojThreshold = config["outlier_reproj_threshold"].toDouble(1.5);
    outlierConfig.confidence = config["outlier_confidence"].toDouble(0.9999);
    outlierConfig.maxIters = config["outlier_max_iters"].toInt(10000);
    const int defaultMinInliers = isBfFlann ? 8 : 25;
    outlierConfig.minInliers = config["outlier_min_inliers"].toInt(defaultMinInliers);

    const int overrideInputWidth = config["input_width"].toInt(-1);
    const int overrideInputHeight = config["input_height"].toInt(-1);
    
    // 设备选择
    bool useCuda = config["use_cuda"].toBool(true);
    if (useCuda) 
    {
        if (torch::cuda::is_available()) 
        {
            sgConfig.use_cuda = true;
            LOG_INFO("%s", qUtf8Printable(QStringLiteral("使用 CUDA 设备")));
        } 
        else 
        {
            sgConfig.use_cuda = false;
            LOG_WARN("%s", qUtf8Printable(QStringLiteral("CUDA 不可用，使用 CPU")));
        }
    } 
    else 
    {
        sgConfig.use_cuda = false;
        LOG_INFO("%s", qUtf8Printable(QStringLiteral("使用 CPU 设备")));
    }
    
    // 确定模型路径（深度匹配算法需要）
    QString modelType = config["model_type"].toString("outdoor");
    QString deviceSuffix = sgConfig.use_cuda ? "cuda" : "cpu";
    QString modelName;
    if (isSuperGlueMatch)
    {
        modelName = QString("superglue_%1_%2.pt").arg(modelType).arg(deviceSuffix);
    }
    else if (isLightGlueMatch)
    {
        // 根据 feature_suffix 选择对应的 LightGlue 模型
        QString featureAlgo;
        if (featureSuffix == QStringLiteral(".dsk"))
            featureAlgo = QStringLiteral("disk");
        else if (featureSuffix == QStringLiteral(".alk"))
            featureAlgo = QStringLiteral("aliked");
        else
            featureAlgo = QStringLiteral("superpoint");

        if (featureAlgo == QStringLiteral("disk"))
            modelName = QString("lightglue_disk_%1.pt").arg(deviceSuffix);
        else if (featureAlgo == QStringLiteral("aliked"))
            modelName = QString("lightglue_aliked_%1.pt").arg(deviceSuffix);
        else
            modelName = QString("lightglue_matcher_%1.pt").arg(deviceSuffix);

        LOG_INFO("%s", qUtf8Printable(QString("LightGlue suffix=%1 algo=%2 model=%3")
                                          .arg(featureSuffix, featureAlgo, modelName)));
    }
    QString modelPath = findModelFile(modelName);

    // 如果专用模型不存在, 尝试通用 LightGlue 模型作为 fallback
    if (isLightGlueMatch && modelPath.isEmpty() && featureAlgo != QStringLiteral("superpoint"))
    {
        const QString fallbackName = QString("lightglue_matcher_%1.pt").arg(deviceSuffix);
        const QString fallbackPath = findModelFile(fallbackName);
        if (!fallbackPath.isEmpty())
        {
            LOG_WARN("%s", qUtf8Printable(QString("模型 %1 不存在, fallback 到 %2 (可能存在维度不匹配)")
                                              .arg(modelName, fallbackName)));
            modelName = fallbackName;
            modelPath = fallbackPath;
        }
    }

    if ((isSuperGlueMatch || isLightGlueMatch) && modelPath.isEmpty())
    {
        LOG_ERROR("%s", qUtf8Printable(QString("模型文件不存在: %1, 跳过此特征类型").arg(modelName)));
        return;
    }
    
    sgConfig.model_path = modelPath.toStdString();
    
    // 并行对数：CUDA 使用 cuda_parallel_pairs，CPU 使用 num_threads
    const bool useCudaFinal = sgConfig.use_cuda;
    const int  numThreads = useCudaFinal
        ? std::max(1, config["cuda_parallel_pairs"].toInt(1))
        : std::max(1, sgConfig.num_threads <= 0
            ? QThread::idealThreadCount() : sgConfig.num_threads);

    // 无匹配记录文件
    const QString assetsDir2 = ProjectIO::projectAssetsDir(projectManager->currentProjectPath());
    const QString noMatchFilePath = QDir(QDir(assetsDir2).filePath(QStringLiteral("matches")))
                                        .filePath(QStringLiteral("no_match_pairs.json"));

    // 构建 baseName -> 原始影像完整路径 的索引，便于写入更完整的元数据
    QMap<QString, QString> baseToImagePath;
    const QStringList allImages = projectManager->getAllImages();
    for (const QString &imgPath : allImages) 
    {
        QFileInfo fi(imgPath);
        const QString base = fi.completeBaseName();
        if (!baseToImagePath.contains(base)) 
        {
            baseToImagePath.insert(base, imgPath);
        }
    }

    try 
    {
        LOG_INFO("%s", qUtf8Printable(QString("匹配配置: algorithm=%1, 并行=%2")
            .arg(matchAlgorithm)
            .arg(numThreads)));

        if (isSuperGlueMatch)
        {
            LOG_INFO("%s", qUtf8Printable(QString("加载 SuperGlue 模型: %1").arg(modelPath)));
            superglue::SuperGlueMatcher testMatcher(sgConfig);
            if (!testMatcher.isLoaded())
            {
                LOG_ERROR("SuperGlue 模型加载失败");
                return;
            }
        }
        else if (isLightGlueMatch)
        {
            LOG_INFO("%s", qUtf8Printable(QString("加载 LightGlue 模型: %1").arg(modelPath)));
            xjw::feature_match::LightGlueConfig lgConfig;
            lgConfig.matcherModelPath = modelPath.toStdString();
            lgConfig.useCuda = sgConfig.use_cuda;
            lgConfig.scoreThreshold = sgConfig.match_threshold;
            xjw::feature_match::LightGlueMatcher testMatcher(lgConfig);
            if (!testMatcher.isLoaded())
            {
                LOG_ERROR("LightGlue 模型加载失败");
                return;
            }
        }
        
        LOG_INFO("%s", qUtf8Printable(QString("%1 配置: match_threshold=%2, max_keypoints=%3, use_cuda=%4")
            .arg(matchAlgorithm)
            .arg(sgConfig.match_threshold)
            .arg(sgConfig.max_keypoints)
            .arg(sgConfig.use_cuda ? "true" : "false")));
        
        std::atomic<int>  successCount{0};
        std::atomic<int>  failCount{0};
        std::atomic<int>  pairCursor{0};
        std::mutex        writeMutex;
        std::atomic<bool> errorFlag{false};
        std::string       errorMsg;
        // 收集无匹配对，线程安全
        QVector<QPair<QString,QString>> failedPairs;

        const int totalPairs = imagePairs.size();

        auto pairWorker = [&]() 
        {
            try 
            {
                std::unique_ptr<xjw::feature_match::IFeatureMatcher> localMatcher;
                if (isSuperGlueMatch)
                {
                    auto sgMatcher = std::make_unique<superglue::SuperGlueMatcher>(sgConfig);
                    if (!sgMatcher->isLoaded())
                    {
                        return;
                    }
                    localMatcher = std::move(sgMatcher);
                }
                else if (isLightGlueMatch)
                {
                    xjw::feature_match::LightGlueConfig lgConfig;
                    lgConfig.matcherModelPath = modelPath.toStdString();
                    lgConfig.useCuda = sgConfig.use_cuda;
                    lgConfig.scoreThreshold = sgConfig.match_threshold;
                    auto lgMatcher = std::make_unique<xjw::feature_match::LightGlueMatcher>(lgConfig);
                    if (!lgMatcher->isLoaded())
                    {
                        return;
                    }
                    localMatcher = std::move(lgMatcher);
                }

                while (!errorFlag.load() && !cancelFlag.load()) 
                {
                    const int idx = pairCursor.fetch_add(1);
                    if (idx >= totalPairs) break;

                    // 每处理一对（无论成败）通知调用方进度
                    struct PairDone { std::atomic<int> &cnt; ~PairDone() { cnt.fetch_add(1); } } _done{progressCount};

                    const QString &pairName = imagePairs[idx];

                    // 解析匹配对名称（格式: "image1__image2"）
                    QStringList parts = pairName.split("__");
                    if (parts.size() != 2) 
                    {
                        LOG_WARN("%s", qUtf8Printable(QString("无效的匹配对格式: %1 (应为 image1__image2)").arg(pairName)));
                        failCount.fetch_add(1);
                        continue;
                    }

                    const QString imageToken0 = QFileInfo(parts[0]).fileName();
                    const QString imageToken1 = QFileInfo(parts[1]).fileName();
                    const QString imagePath0 = xjw::gui::project::resolveProjectImagePathFromToken(imageToken0, currentMeta);
                    const QString imagePath1 = xjw::gui::project::resolveProjectImagePathFromToken(imageToken1, currentMeta);
                    const QString baseName0 = QFileInfo(imagePath0.isEmpty() ? imageToken0 : imagePath0).completeBaseName();
                    const QString baseName1 = QFileInfo(imagePath1.isEmpty() ? imageToken1 : imagePath1).completeBaseName();
                    const QString canonicalPairName = QString("%1__%2").arg(baseName0, baseName1);

                    // 根据项目和特征后缀解析特征文件路径
                    const QString sp0Path = xjw::gui::project::resolveFeaturePathBySuffix(plascanPath, currentMeta, imageToken0, featureSuffix);
                    const QString sp1Path = xjw::gui::project::resolveFeaturePathBySuffix(plascanPath, currentMeta, imageToken1, featureSuffix);

                    if (!QFile::exists(sp0Path) || !QFile::exists(sp1Path)) 
                    {
                        LOG_ERROR("%s", qUtf8Printable(QString("特征文件不存在: %1 或 %2").arg(sp0Path, sp1Path)));
                        failCount.fetch_add(1);
                        continue;
                    }

                    LOG_INFO("%s", qUtf8Printable(QString("匹配: %1 <-> %2").arg(baseName0, baseName1)));

                    try 
                    {
                        // 读取特征数据
                        QString imgName0, imgName1;
                        FeatureOutput sp0, sp1;

                        if (!FeatureFileIO::read(sp0Path, imgName0, sp0)) 
                        {
                            LOG_ERROR("%s", qUtf8Printable(QString("读取特征文件失败: %1").arg(sp0Path)));
                            failCount.fetch_add(1);
                            continue;
                        }
                        if (!FeatureFileIO::read(sp1Path, imgName1, sp1)) 
                        {
                            LOG_ERROR("%s", qUtf8Printable(QString("读取特征文件失败: %1").arg(sp1Path)));
                            failCount.fetch_add(1);
                            continue;
                        }

                        xjw::feature_match::MatchResult matchResult;

                        const QString imagePath0Resolved = imagePath0.isEmpty() ? baseToImagePath.value(baseName0) : imagePath0;
                        const QString imagePath1Resolved = imagePath1.isEmpty() ? baseToImagePath.value(baseName1) : imagePath1;

                        int h0 = 1000, w0 = 1000, h1 = 1000, w1 = 1000;
                        if (!imagePath0Resolved.isEmpty()) 
                        {
                            cv::Mat img0 = cv::imread(imagePath0Resolved.toStdString(), cv::IMREAD_GRAYSCALE);
                            if (!img0.empty()) { h0 = img0.rows; w0 = img0.cols; }
                        }
                        if (!imagePath1Resolved.isEmpty()) 
                        {
                            cv::Mat img1 = cv::imread(imagePath1Resolved.toStdString(), cv::IMREAD_GRAYSCALE);
                            if (!img1.empty()) { h1 = img1.rows; w1 = img1.cols; }
                        }
                        if (overrideInputWidth > 0 && overrideInputHeight > 0) 
                        {
                            w0 = overrideInputWidth; h0 = overrideInputHeight;
                            w1 = overrideInputWidth; h1 = overrideInputHeight;
                        }

                        if (isSuperGlueMatch || isLightGlueMatch)
                        {
                            // 从 ipfind_results 读取实际特征提取算法
                            QString featureAlgo = QStringLiteral("superpoint");
                            const QJsonArray ipResults = currentMeta.value(QLatin1String("ipfind_results")).toArray();
                            if (!ipResults.isEmpty())
                            {
                                const QString a = ipResults.first().toObject()
                                                      .value(QLatin1String("settings")).toObject()
                                                      .value(QLatin1String("feature_algorithm")).toString()
                                                      .toLower();
                                if (!a.isEmpty()) featureAlgo = a;
                            }
                            auto fd0 = xjw::feature_extractors::FeatureData::fromFeatureOutput(sp0, featureAlgo.toStdString());
                            auto fd1 = xjw::feature_extractors::FeatureData::fromFeatureOutput(sp1, featureAlgo.toStdString());
                            fd0.imageWidth = w0;
                            fd0.imageHeight = h0;
                            fd1.imageWidth = w1;
                            fd1.imageHeight = h1;
                            matchResult = localMatcher->match(fd0, fd1);
                        }
                        else if (isBfFlann)
                        {
                            // 根据匹配算法确定特征提取器名称
                            std::string featureAlgoName;
                            if (matchAlgorithm == QStringLiteral("orb_bf_hamming"))
                                featureAlgoName = "orb";
                            else
                                featureAlgoName = "sift";

                            auto fd0 = xjw::feature_extractors::FeatureData::fromFeatureOutput(sp0, featureAlgoName);
                            auto fd1 = xjw::feature_extractors::FeatureData::fromFeatureOutput(sp1, featureAlgoName);
                            fd0.imageWidth = w0;
                            fd0.imageHeight = h0;
                            fd1.imageWidth = w1;
                            fd1.imageHeight = h1;

                            cv::Mat desc0 = fd0.toCvDescriptors(matchAlgorithm.toStdString());
                            cv::Mat desc1 = fd1.toCvDescriptors(matchAlgorithm.toStdString());

                            std::vector<cv::DMatch> cvMatches;
                            if (matchAlgorithm == QStringLiteral("sift_flann"))
                            {
                                cv::FlannBasedMatcher matcher;
                                matcher.match(desc0, desc1, cvMatches);
                            }
                            else
                            {
                                int norm = (matchAlgorithm == QStringLiteral("orb_bf_hamming")) ? cv::NORM_HAMMING : cv::NORM_L2;
                                cv::BFMatcher matcher(norm, true);
                                matcher.match(desc0, desc1, cvMatches);
                            }

                            matchResult = xjw::feature_match::MatchResult::fromCvMatches(
                                cvMatches,
                                static_cast<int>(sp0.keypoints.size()),
                                static_cast<int>(sp1.keypoints.size()),
                                matchAlgorithm.toStdString());
                        }

                        const int rawMatchCount = matchResult.numMatches;
                        const std::vector<cv::DMatch> rawMatches = matchResult.cvMatches;

                        // 粗差剔除
                        int inlierCount = matchResult.numMatches;
                        matchResult = superglue::MatchOutlierRejector::filter(
                            matchResult, sp0.keypoints, sp1.keypoints,
                            outlierConfig, &inlierCount);

                        // BF/FLANN 分支容错：几何过滤过严导致全丢时，回退到原始匹配结果。
                        if (isBfFlann && rawMatchCount > 0 && matchResult.numMatches <= 0)
                        {
                            matchResult.cvMatches = rawMatches;
                            matchResult.numMatches = static_cast<int>(rawMatches.size());
                            matchResult.buildIndicesFromCvMatches(static_cast<int>(sp0.keypoints.size()),
                                                                  static_cast<int>(sp1.keypoints.size()));
                            LOG_WARN("%s", qUtf8Printable(QString("  %1 几何过滤后无内点，回退到原始匹配: %2")
                                .arg(canonicalPairName)
                                .arg(matchResult.numMatches)));
                        }

                        LOG_INFO("%s", qUtf8Printable(QString("  %1 内点数: %2")
                            .arg(canonicalPairName).arg(matchResult.numMatches)));

                        if (matchResult.matches0.empty() ||
                            matchResult.matches1.empty() ||
                            matchResult.matchingScores0.empty() ||
                            matchResult.matchingScores1.empty())
                        {
                            const int kpCount0 = static_cast<int>(sp0.keypoints.size());
                            const int kpCount1 = static_cast<int>(sp1.keypoints.size());
                            matchResult.buildIndicesFromCvMatches(kpCount0, kpCount1);
                        }

                        // ── 内点不足：记录到无匹配列表，不生成文件 ───────────────
                        if (matchResult.numMatches < outlierConfig.minInliers) 
                        {
                            if (isBfFlann && matchResult.numMatches > 0)
                            {
                                LOG_WARN("%s", qUtf8Printable(QString("  %1 内点(%2)低于阈值(%3)，仍保留 BF/FLANN 匹配结果")
                                    .arg(canonicalPairName)
                                    .arg(matchResult.numMatches)
                                    .arg(outlierConfig.minInliers)));
                            }
                            else
                            {
                            LOG_INFO("%s", qUtf8Printable(QString("  跳过 %1: 内点不足（已记录为无匹配对）").arg(canonicalPairName)));
                            {
                                std::lock_guard<std::mutex> lk(writeMutex);
                                failedPairs.append({imagePath0Resolved, imagePath1Resolved});
                            }
                            failCount.fetch_add(1);
                            continue;
                            }
                        }

                        // ── 保存 .match 文件 ────────────────────────────────────
                        const QString outputPath = QDir(outputDir).filePath(canonicalPairName + ".match");
                        const QString imagePath0ForMeta = imagePath0Resolved.isEmpty() ? imageToken0 : imagePath0Resolved;
                        const QString imagePath1ForMeta = imagePath1Resolved.isEmpty() ? imageToken1 : imagePath1Resolved;

                        if (!SuperGlueMatchIO::write(outputPath, baseName0, baseName1, matchResult)) 
                        {
                            LOG_ERROR("%s", qUtf8Printable(QString("保存匹配结果失败: %1").arg(outputPath)));
                            failCount.fetch_add(1);
                            continue;
                        }

                        // ── sidecar JSON ────────────────────────────────────────
                        QJsonObject sidecar;
                        sidecar["match_file"]   = outputPath;
                        sidecar["image0_name"]  = baseName0;
                        sidecar["image1_name"]  = baseName1;
                        sidecar["image0_path"]  = imagePath0ForMeta;
                        sidecar["image1_path"]  = imagePath1ForMeta;
                        sidecar["sp0_path"]     = sp0Path;
                        sidecar["sp1_path"]     = sp1Path;
                        sidecar["num_matches"]  = matchResult.numMatches;
                        sidecar["match_algorithm"] = matchAlgorithm;

                        QJsonArray pts0, pts1;
                        for (const auto &dm : matchResult.cvMatches) 
                        {
                            const int i0 = dm.queryIdx, i1 = dm.trainIdx;
                            if (i0 >= 0 && i0 < (int)sp0.keypoints.size() &&
                                i1 >= 0 && i1 < (int)sp1.keypoints.size()) 
                            {
                                QJsonArray p0; p0.append(sp0.keypoints[i0].pt.x); p0.append(sp0.keypoints[i0].pt.y);
                                QJsonArray p1; p1.append(sp1.keypoints[i1].pt.x); p1.append(sp1.keypoints[i1].pt.y);
                                pts0.append(p0); pts1.append(p1);
                            }
                        }
                        sidecar["matched_points0"] = pts0;
                        sidecar["matched_points1"] = pts1;

                        const QString sidecarPath = outputPath + ".json";
                        {
                            QFile sf(sidecarPath);
                            if (sf.open(QIODevice::WriteOnly | QIODevice::Truncate))
                                sf.write(QJsonDocument(sidecar).toJson(QJsonDocument::Compact));
                        }

                        // ── 更新项目元数据（每对立即通知，触发匹配查看器刷新）───
                        QJsonObject pairSettings = config;
                        QJsonArray imgFilesForMeta;
                        imgFilesForMeta.append(imagePath0ForMeta);
                        imgFilesForMeta.append(imagePath1ForMeta);
                        pairSettings["image_files"]  = imgFilesForMeta;
                        pairSettings["image0"]       = imagePath0ForMeta;
                        pairSettings["image1"]       = imagePath1ForMeta;
                        pairSettings["sp0_path"]     = sp0Path;
                        pairSettings["sp1_path"]     = sp1Path;
                        pairSettings["pair_name"]    = canonicalPairName;
                        pairSettings["sidecar_json"] = sidecarPath;
                        pairSettings["match_algorithm"] = matchAlgorithm;

                        QStringList singleOutput{outputPath};
                        QMetaObject::invokeMethod(projectManager, "appendIpmatchResult",
                                                  Qt::QueuedConnection,
                                                  Q_ARG(QStringList, singleOutput),
                                                  Q_ARG(QJsonObject, pairSettings));

                        successCount.fetch_add(1);
                        LOG_INFO("%s", qUtf8Printable(QString("  %1 匹配完成: %2 对")
                            .arg(canonicalPairName).arg(matchResult.numMatches)));

                        if (sgConfig.enable_csv_output) 
                        {
                            QString csvPath = QDir(outputDir).filePath(canonicalPairName + ".csv");
                            SuperGlueMatchIO::exportToCSV(csvPath, matchResult);
                        }

                    } 
                    catch (const std::exception &e) 
                    {
                        LOG_ERROR("%s", qUtf8Printable(QString("匹配失败 %1: %2")
                            .arg(canonicalPairName).arg(QString::fromStdString(e.what()))));
                        failCount.fetch_add(1);
                    }
                }
            } 
            catch (const std::exception &ex) 
            {
                std::lock_guard<std::mutex> lk(writeMutex);
                if (!errorFlag.exchange(true)) errorMsg = ex.what();
            }
        };

        // ── 启动工作线程 ─────────────────────────────────────────────
        std::vector<std::thread> workerThreads;
        workerThreads.reserve(numThreads);
        for (int t = 0; t < numThreads; ++t)
            workerThreads.emplace_back(pairWorker);
        for (auto &th : workerThreads) th.join();

        if (cancelFlag.load())
        {
            LOG_INFO("%s", qUtf8Printable(QString("%1 匹配已被用户取消").arg(matchAlgorithm)));
        }
        else
        {
            LOG_INFO("%s", qUtf8Printable(QString("%1 匹配完成: 成功 %2, 跳过/失败 %3")
                .arg(matchAlgorithm).arg(successCount.load()).arg(failCount.load())));
        }

        // ── 将无匹配对追加写入 no_match_pairs.json ──────────────────
        if (!failedPairs.isEmpty()) 
        {
            QJsonArray existing;
            {
                QFile nmf(noMatchFilePath);
                if (nmf.open(QIODevice::ReadOnly))
                    existing = QJsonDocument::fromJson(nmf.readAll()).array();
            }
            QSet<QString> existingKeys;
            for (const QJsonValue &v : existing) 
            {
                const QJsonObject o = v.toObject();
                existingKeys.insert(o["image0"].toString() + "|" + o["image1"].toString());
            }
            const QString ts = QDateTime::currentDateTime().toString(Qt::ISODate);
            for (const auto &fp : failedPairs) 
            {
                const QString key  = fp.first + "|" + fp.second;
                const QString keyR = fp.second + "|" + fp.first;
                if (!existingKeys.contains(key) && !existingKeys.contains(keyR)) 
                {
                    QJsonObject o;
                    o["image0"]     = fp.first;
                    o["image1"]     = fp.second;
                    o["checked_at"] = ts;
                    existing.append(o);
                    existingKeys.insert(key);
                }
            }
            QFile nmfOut(noMatchFilePath);
            if (nmfOut.open(QIODevice::WriteOnly | QIODevice::Truncate))
                nmfOut.write(QJsonDocument(existing).toJson(QJsonDocument::Compact));
        }
        
    } 
    catch (const std::exception &e) 
    {
        LOG_ERROR("%s", qUtf8Printable(QString("%1 执行异常: %2")
            .arg(matchAlgorithm).arg(QString::fromStdString(e.what()))));
    }
}
