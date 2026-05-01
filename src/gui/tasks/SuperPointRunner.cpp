#include "compat/QtTorchMacroGuard.h"

#include "SuperPoint.h"
#include "FeatureFileIO.h"
#include "tradition/TraditionalFeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "SuperPointRunner.h"
#include "Logger.h"
#include "ProjectIO.h"
#include "ProjectManager.h"
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QCoreApplication>
#include <QProcess>

#include <memory>

namespace
{

QString normalizedFeatureAlgorithm(const QJsonObject &config)
{
    const std::string normalized = xjw::feature_extractors::TraditionalFeatureExtractor::normalizeAlgorithmName(
        config.value("feature_algorithm").toString("superpoint").toStdString());
    return QString::fromStdString(normalized);
}

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

// ---------------------------------------------------------------------------
// DISK / ALIKED: invoke Python script, return false on failure
// ---------------------------------------------------------------------------
static bool runPythonExtractor(const QString &algo, const QStringList &inputs,
                                const QString &outputDir, int maxKeypoints,
                                const QString &device,
                                ProjectManager *projectManager,
                                const QJsonObject &config,
                                std::atomic<bool> &cancelFlag,
                                std::atomic<int> &progressCount)
{
    // Locate extract_features.py next to the source tree
    QString scriptPath;
#ifdef PLASCAN_SOURCE_DIR
    scriptPath = QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("scripts/extract_features.py"));
#endif
    if (scriptPath.isEmpty() || !QFile::exists(scriptPath))
    {
        // Fallback: relative to binary
        scriptPath = QDir(QCoreApplication::applicationDirPath())
                         .filePath(QStringLiteral("../scripts/extract_features.py"));
    }
    if (!QFile::exists(scriptPath))
    {
        LOG_ERROR("%s", qUtf8Printable(QString("找不到 extract_features.py: %1").arg(scriptPath)));
        return false;
    }

    // Find python executable (prefer conda env)
    QString pythonExe = QStringLiteral("python");
    const QString condaPrefix = qEnvironmentVariable("CONDA_PREFIX").trimmed();
    if (!condaPrefix.isEmpty())
    {
        const QString condaPython = QDir(condaPrefix).filePath(QStringLiteral("bin/python"));
        if (QFile::exists(condaPython))
            pythonExe = condaPython;
    }

    QStringList args;
    args << scriptPath
         << QStringLiteral("--algo") << algo
         << QStringLiteral("--output") << outputDir
         << QStringLiteral("--max-keypoints") << QString::number(maxKeypoints)
         << QStringLiteral("--device") << device.toLower()
         << QStringLiteral("--images");
    args << inputs;

    LOG_INFO("%s", qUtf8Printable(QString("[%1] 调用 Python 脚本提取特征: %2 张影像")
                                      .arg(algo.toUpper()).arg(inputs.size())));

    QProcess proc;
    proc.setProcessChannelMode(QProcess::MergedChannels);
    proc.start(pythonExe, args);

    if (!proc.waitForStarted(10000))
    {
        LOG_ERROR("%s", qUtf8Printable(QString("[%1] 无法启动 Python 进程").arg(algo.toUpper())));
        return false;
    }

    // Stream output line by line while waiting
    while (!proc.waitForFinished(500))
    {
        if (cancelFlag.load())
        {
            proc.kill();
            LOG_WARN("%s", qUtf8Printable(QString("[%1] 用户取消").arg(algo.toUpper())));
            return false;
        }
        const QByteArray out = proc.readAllStandardOutput();
        if (!out.isEmpty())
            LOG_INFO("%s", out.trimmed().constData());
    }
    const QByteArray remaining = proc.readAllStandardOutput();
    if (!remaining.isEmpty())
        LOG_INFO("%s", remaining.trimmed().constData());

    if (proc.exitCode() != 0)
    {
        LOG_ERROR("%s", qUtf8Printable(QString("[%1] Python 脚本退出码: %2")
                                           .arg(algo.toUpper()).arg(proc.exitCode())));
        return false;
    }

    // Register output .sp files with the project
    int successCount = 0;
    for (const QString &imagePath : inputs)
    {
        QFileInfo fi(imagePath);
        const QString spPath = QDir(outputDir).filePath(fi.completeBaseName() + QStringLiteral(".sp"));
        if (QFile::exists(spPath))
        {
            if (projectManager)
            {
                QMetaObject::invokeMethod(projectManager, "appendIpfindResult",
                                          Qt::QueuedConnection,
                                          Q_ARG(QString, imagePath),
                                          Q_ARG(QString, spPath),
                                          Q_ARG(QJsonObject, config));
            }
            successCount++;
        }
        else
        {
            LOG_WARN("%s", qUtf8Printable(QString("[%1] 未找到输出文件: %2").arg(algo.toUpper()).arg(spPath)));
        }
        progressCount.fetch_add(1);
    }

    LOG_INFO("%s", qUtf8Printable(QString("[%1] 特征提取完成: 成功 %2/%3 张")
                                      .arg(algo.toUpper()).arg(successCount).arg(inputs.size())));
    return successCount > 0;
}

bool SuperPointRunner::run(const QJsonObject &config, const QStringList &inputs, ProjectManager *projectManager)
{
    std::atomic<bool> neverCancel{false};
    std::atomic<int> dummy{0};
    return run(config, inputs, projectManager, neverCancel, dummy);
}

bool SuperPointRunner::run(const QJsonObject &config, const QStringList &inputs,
                            ProjectManager *projectManager, std::atomic<bool> &cancelFlag,
                            std::atomic<int> &progressCount)
{
    const QString featureAlgorithm = normalizedFeatureAlgorithm(config);
    LOG_INFO("%s", qUtf8Printable(QString("开始特征提取(%1): %2 张影像")
        .arg(featureAlgorithm.toUpper())
        .arg(inputs.size())));
    
    // 创建输出目录
    QString outputDir = config["output_dir"].toString();
    if (outputDir.isEmpty()) {
        const QString assetsDir = ProjectIO::projectAssetsDir(projectManager->currentProjectPath());
        outputDir = QDir(assetsDir).filePath(QStringLiteral("ip"));
    }
    
    QDir dir;
    if (!dir.exists(outputDir)) {
        if (!dir.mkpath(outputDir)) {
            LOG_ERROR("%s", qUtf8Printable(QString("无法创建输出目录: %1").arg(outputDir)));
            return false;
        }
    }

    // DISK / ALIKED: delegate entirely to Python script
    if (featureAlgorithm == QStringLiteral("disk") || featureAlgorithm == QStringLiteral("aliked"))
    {
        const int maxKpts = config["max_num_keypoints"].toInt(2048);
        const QString device = config["device"].toString("CPU");
        return runPythonExtractor(featureAlgorithm, inputs, outputDir, maxKpts, device,
                                  projectManager, config, cancelFlag, progressCount);
    }
    
    // 构建 SuperPointConfig
    SuperPointConfig spConfig;
    spConfig.nms_radius = config["nms_radius"].toInt(4);
    spConfig.detection_threshold = static_cast<float>(config["detection_threshold"].toDouble(0.005));
    spConfig.max_num_keypoints = config["max_num_keypoints"].toInt(-1);
    spConfig.remove_borders = config["remove_borders"].toInt(4);
    spConfig.grayscale_min = static_cast<float>(config["grayscale_min"].toDouble(0.0));
    spConfig.grayscale_max = static_cast<float>(config["grayscale_max"].toDouble(1.0));
    spConfig.normalize_input = config["normalize_input"].toBool(true);
    spConfig.descriptor_dim = config["descriptor_dim"].toInt(256);
    spConfig.grid_size = config["grid_size"].toInt(8);
    spConfig.batch_size = config["batch_size"].toInt(8);
    spConfig.neighborhood_check_radius = config["neighborhood_check_radius"].toInt(3);
    spConfig.neighborhood_threshold = static_cast<float>(config["neighborhood_threshold"].toDouble(0.05));
    
    // 设备选择
    QString deviceStr = config["device"].toString("CPU");
    if (deviceStr == "CUDA") 
    {
        if (torch::cuda::is_available()) 
        {
            spConfig.device = torch::kCUDA;
            LOG_INFO("%s", qUtf8Printable(QStringLiteral("使用 CUDA 设备")));
        } 
        else 
        {
            if (config["allow_device_fallback"].toBool(true)) 
            {
                spConfig.device = torch::kCPU;
                LOG_WARN("%s", qUtf8Printable(QStringLiteral("CUDA 不可用，回退到 CPU")));
            } 
            else
            {
                LOG_ERROR("%s", qUtf8Printable(QStringLiteral("CUDA 不可用且不允许回退")));
                return false;
            }
        }
    } 
    else 
    {
        spConfig.device = torch::kCPU;
        LOG_INFO("%s", qUtf8Printable(QStringLiteral("使用 CPU 设备")));
    }
    
    spConfig.allow_device_fallback = config["allow_device_fallback"].toBool(true);
    spConfig.save_keypoints_csv = config["save_keypoints_csv"].toBool(false);
    spConfig.save_overlay_image = config["save_overlay_image"].toBool(false);
    spConfig.max_image_size = config["max_image_size"].toInt(2048);  // 默认 2048，遥感大图自动缩放
    
    std::unique_ptr<SuperPoint> superPointExtractor;

    try 
    {
        if (featureAlgorithm == QStringLiteral("superpoint"))
        {
            // 确定模型路径（自动检测源码、安装和运行目录）
            const QString modelName = spConfig.device.is_cuda() ? "superpoint_v6_cuda.pt" : "superpoint_v6_cpu.pt";
            const QString modelPath = findModelFile(modelName);

            // 检查模型文件是否存在
            if (modelPath.isEmpty())
            {
                LOG_ERROR("%s", qUtf8Printable(QString("模型文件不存在: %1 (已尝试多个路径)").arg(modelName)));
                return false;
            }

            // 加载模型
            LOG_INFO("%s", qUtf8Printable(QString("加载 SuperPoint 模型: %1").arg(modelPath)));
            superPointExtractor = std::make_unique<SuperPoint>(modelPath.toStdString(), spConfig);
        }

        // 打印实际使用的配置，便于调试参数未生效的问题
        LOG_INFO("%s", qUtf8Printable(QString("特征提取配置: algorithm=%1, nms_radius=%2, detection_threshold=%3, max_num_keypoints=%4, remove_borders=%5, grayscale_min=%6, grayscale_max=%7, normalize_input=%8, descriptor_dim=%9, grid_size=%10, batch_size=%11, neighborhood_radius=%12, neighborhood_threshold=%13, device=%14")
            .arg(featureAlgorithm.toUpper())
            .arg(spConfig.nms_radius)
            .arg(spConfig.detection_threshold)
            .arg(spConfig.max_num_keypoints)
            .arg(spConfig.remove_borders)
            .arg(spConfig.grayscale_min)
            .arg(spConfig.grayscale_max)
            .arg(spConfig.normalize_input ? "true" : "false")
            .arg(spConfig.descriptor_dim)
            .arg(spConfig.grid_size)
            .arg(spConfig.batch_size)
            .arg(spConfig.neighborhood_check_radius)
            .arg(spConfig.neighborhood_threshold)
            .arg(spConfig.device.is_cuda() ? "CUDA" : "CPU")));
        
        // 处理每张图像
        int successCount = 0;
        int failCount = 0;
        
        for (const QString &imagePath : inputs) 
        {
            if (cancelFlag.load()) break;
            QFileInfo fileInfo(imagePath);
            LOG_INFO("%s", qUtf8Printable(QString("处理: %1").arg(fileInfo.fileName())));
            
            try
            {
                // 读取图像
                cv::Mat image = cv::imread(imagePath.toStdString(), cv::IMREAD_GRAYSCALE);
                if (image.empty())
                {
                    LOG_ERROR("%s", qUtf8Printable(QString("无法读取图像: %1").arg(imagePath)));
                    failCount++;
                    progressCount.fetch_add(1);
                    continue;
                }

                // 大图缩放：避免 CUDA OOM
                cv::Mat processImage = image;
                float scale = 1.0f;
                if (spConfig.max_image_size > 0)
                {
                    const int maxDim = std::max(image.rows, image.cols);
                    if (maxDim > spConfig.max_image_size)
                    {
                        scale = static_cast<float>(spConfig.max_image_size) / maxDim;
                        const int newW = static_cast<int>(image.cols * scale);
                        const int newH = static_cast<int>(image.rows * scale);
                        cv::resize(image, processImage, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
                        LOG_INFO("%s", qUtf8Printable(QString("  图像缩放: %1x%2 → %3x%4 (scale=%.3f)")
                            .arg(image.cols).arg(image.rows).arg(newW).arg(newH).arg(scale)));
                    }
                }

                // 执行特征提取
                SuperPointOutput output;
                if (superPointExtractor)
                {
                    output = superPointExtractor->detect(processImage);
                }
                else
                {
                    output = xjw::feature_extractors::TraditionalFeatureExtractor::detect(
                        processImage,
                        spConfig,
                        featureAlgorithm.toStdString());
                }

                // 还原关键点坐标到原图尺寸
                if (scale != 1.0f)
                {
                    const float invScale = 1.0f / scale;
                    for (auto &kp : output.keypoints)
                    {
                        kp.pt.x *= invScale;
                        kp.pt.y *= invScale;
                    }
                }
                
                // 保存结果为二进制格式(.sp)
                QString baseName = fileInfo.completeBaseName();
                QString outputPath = QDir(outputDir).filePath(baseName + ".sp");
                
                if (FeatureFileIO::write(outputPath, fileInfo.fileName(), output)) 
                {
                    LOG_INFO("%s", qUtf8Printable(QString("  检测到 %1 个特征点，已保存到: %2")
                        .arg(output.keypoints.size()).arg(outputPath)));
                    if (output.keypoints.empty()) 
                    {
                        LOG_WARN("%s", qUtf8Printable(QString("%1 对 %2 未检测到关键点（可能与灰度/阈值/边界过滤有关）")
                            .arg(featureAlgorithm.toUpper()).arg(imagePath)));
                    }
                    successCount++;
                    
                    // 可选：保存调试文件
                    if (spConfig.save_keypoints_csv) 
                    {
                        QString csvPath = QDir(outputDir).filePath(baseName + ".csv");
                        SuperPoint::saveKeypointsCSV(output, csvPath.toStdString());
                    }
                    
                    if (spConfig.save_overlay_image) 
                    {
                        QString overlayPath = QDir(outputDir).filePath(baseName + "_overlay.png");
                        SuperPoint::saveOverlayImage(image, output, overlayPath.toStdString());
                    }
                    // 将输出文件记录到项目元数据（通过 ProjectManager 转发到 ProjectData）
                    if (projectManager) 
                    {
                        // 使用主线程进行元数据更新，避免并发写入问题
                        QMetaObject::invokeMethod(projectManager, "appendIpfindResult",
                                                  Qt::QueuedConnection,
                                                  Q_ARG(QString, imagePath), Q_ARG(QString, outputPath), Q_ARG(QJsonObject, config));
                    }
                } 
                else 
                {
                    LOG_ERROR("%s", qUtf8Printable(QString("保存结果失败: %1").arg(outputPath)));
                    failCount++;
                }
                
            } 
            catch (const std::exception &e) 
            {
                LOG_ERROR("%s", qUtf8Printable(QString("处理图像失败 %1: %2")
                    .arg(fileInfo.fileName()).arg(QString::fromStdString(e.what()))));
                failCount++;
            }
            progressCount.fetch_add(1);
        }
        
        // 输出总结
        LOG_INFO("%s", qUtf8Printable(QString("特征提取完成(%1): 成功 %2 张，失败 %3 张")
            .arg(featureAlgorithm.toUpper())
            .arg(successCount).arg(failCount)));

        return successCount > 0;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("%s", qUtf8Printable(QString("SuperPoint 初始化失败: %1")
            .arg(QString::fromStdString(e.what()))));
        return false;
    }
}
