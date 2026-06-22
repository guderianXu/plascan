#include "compat/QtTorchMacroGuard.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

#include "SuperPoint.h"
#include "FeatureFileIO.h"
#include "ExtractorFactory.h"
#include "tradition/TraditionalFeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "FeatureExtractionRunner.h"
#include "Logger.h"
#include "ProjectIO.h"
#include "ProjectManager.h"
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QCoreApplication>
#include <QMetaObject>

#include <memory>

namespace
{

QString normalizedFeatureAlgorithm(const QJsonObject &config)
{
    const std::string normalized = xjw::feature_extractors::TraditionalFeatureExtractor::normalizeAlgorithmName(
        config.value("feature_algorithm").toString("superpoint").toStdString());
    return QString::fromStdString(normalized);
}

int maxKeypointsFromConfig(const QJsonObject &config)
{
    if (config.contains(QStringLiteral("max_num_keypoints")))
    {
        return config.value(QStringLiteral("max_num_keypoints")).toInt(-1);
    }
    return config.value(QStringLiteral("max_keypoints")).toInt(-1);
}

bool useCudaFromConfig(const QJsonObject &config)
{
    if (config.contains(QStringLiteral("use_cuda")))
    {
        return config.value(QStringLiteral("use_cuda")).toBool(false);
    }

    const QString deviceString = config.value(QStringLiteral("device")).toString(QStringLiteral("CPU"));
    const QString lowerDevice = deviceString.toLower();
    return lowerDevice == QStringLiteral("cuda") || lowerDevice == QStringLiteral("gpu");
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
    candidates.append(QDir(exePath).filePath(QStringLiteral("../resources/models/%1").arg(modelName)));
    candidates.append(QDir(exePath).filePath(QStringLiteral("../../resources/models/%1").arg(modelName)));
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

QStringList extractorModelCandidates(const QString &algorithm, bool useCuda)
{
    const QString suffix = useCuda ? QStringLiteral("cuda") : QStringLiteral("cpu");
    if (algorithm == QStringLiteral("disk"))
    {
        return {
            QStringLiteral("disk_extractor_%1_8192.torchscript").arg(suffix),
            QStringLiteral("disk_extractor_%1_8192.pt").arg(suffix),
            QStringLiteral("disk_extractor_%1_1200.torchscript").arg(suffix),
            QStringLiteral("disk_extractor_%1_1200.pt").arg(suffix),
            QStringLiteral("disk_extractor.torchscript"),
            QStringLiteral("disk_extractor.pt"),
        };
    }
    if (algorithm == QStringLiteral("aliked"))
    {
        return {
            QStringLiteral("aliked_extractor_%1_480.torchscript").arg(suffix),
            QStringLiteral("aliked_extractor_%1_480.pt").arg(suffix),
            QStringLiteral("aliked_extractor.torchscript"),
            QStringLiteral("aliked_extractor.pt"),
        };
    }

    return {};
}

bool isManagedExtractorModelPath(const QString &path)
{
    const QString fileName = QFileInfo(path).fileName().toLower();
    return fileName.startsWith(QStringLiteral("superpoint_extractor"))
        || fileName.startsWith(QStringLiteral("disk_extractor"))
        || fileName.startsWith(QStringLiteral("aliked_extractor"));
}

QString resolveExtractorModelPath(const QString &algorithm, bool useCuda, const QString &configuredPath)
{
    QString modelPath = configuredPath.trimmed();
    if (!modelPath.isEmpty())
    {
        const QFileInfo info(modelPath);
        const QString fileName = info.fileName().toLower();
        const QString expectedDeviceToken = useCuda ? QStringLiteral("_cuda") : QStringLiteral("_cpu");
        if (info.exists() && (!isManagedExtractorModelPath(modelPath) || fileName.contains(expectedDeviceToken)))
        {
            return QDir::cleanPath(info.absoluteFilePath());
        }
    }

    for (const QString &candidate : extractorModelCandidates(algorithm, useCuda))
    {
        modelPath = findModelFile(candidate);
        if (!modelPath.isEmpty())
        {
            return QDir::cleanPath(QFileInfo(modelPath).absoluteFilePath());
        }
    }

    return QString();
}

} // namespace

bool FeatureExtractionRunner::run(const QJsonObject &config, const QStringList &inputs, ProjectManager *projectManager)
{
    return run(config, inputs, QPointer<ProjectManager>(projectManager));
}

bool FeatureExtractionRunner::run(const QJsonObject &config, const QStringList &inputs,
                                  QPointer<ProjectManager> projectManager)
{
    std::atomic<bool> neverCancel{false};
    std::atomic<int> dummy{0};
    return run(config, inputs, projectManager, neverCancel, dummy);
}

bool FeatureExtractionRunner::run(const QJsonObject &config, const QStringList &inputs,
                            ProjectManager *projectManager, std::atomic<bool> &cancelFlag,
                            std::atomic<int> &progressCount)
{
    return run(config, inputs, QPointer<ProjectManager>(projectManager), cancelFlag, progressCount);
}

bool FeatureExtractionRunner::run(const QJsonObject &config, const QStringList &inputs,
                            QPointer<ProjectManager> projectManager, std::atomic<bool> &cancelFlag,
                            std::atomic<int> &progressCount)
{
    const QString featureAlgorithm = normalizedFeatureAlgorithm(config);
    const QString fileSuffix = QString::fromStdString(
        ExtractorSuffix::forAlgorithm(featureAlgorithm.toStdString()));
    LOG_INFO("%s", qUtf8Printable(QString("开始特征提取(%1): %2 张影像, 后缀=%3")
        .arg(featureAlgorithm.toUpper())
        .arg(inputs.size())
        .arg(fileSuffix)));
    
    // 创建输出目录
    QString outputDir = config["output_dir"].toString();
    if (outputDir.isEmpty())
    {
        if (!projectManager)
        {
            LOG_ERROR("%s", qUtf8Printable(QString("特征提取缺少输出目录，且项目已关闭或 ProjectManager 不可用")));
            return false;
        }
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

    // 构建 SuperPointConfig
    SuperPointConfig spConfig;
    spConfig.nms_radius = config["nms_radius"].toInt(4);
    spConfig.detection_threshold = static_cast<float>(config["detection_threshold"].toDouble(0.005));
    spConfig.max_num_keypoints = maxKeypointsFromConfig(config);
    spConfig.remove_borders = config["remove_borders"].toInt(4);
    spConfig.grayscale_min = static_cast<float>(config["grayscale_min"].toDouble(5.0 / 255.0));
    spConfig.grayscale_max = static_cast<float>(config["grayscale_max"].toDouble(1.0));
    spConfig.normalize_input = config["normalize_input"].toBool(true);
    spConfig.descriptor_dim = config["descriptor_dim"].toInt(256);
    spConfig.grid_size = config["grid_size"].toInt(8);
    spConfig.batch_size = config["batch_size"].toInt(8);
    spConfig.neighborhood_check_radius = config["neighborhood_check_radius"].toInt(3);
    spConfig.neighborhood_threshold = static_cast<float>(config["neighborhood_threshold"].toDouble(0.05));
    
    // 设备选择
    if (useCudaFromConfig(config))
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
    std::unique_ptr<IExtractor> nativeExtractor;

    try 
    {
        if (featureAlgorithm == QStringLiteral("superpoint"))
        {
            // 确定模型路径: 优先 CPU 模型 (始终存在), CUDA 模型可选
            QStringList modelCandidates;
            if (spConfig.device.is_cuda())
                modelCandidates << "superpoint_extractor_cuda.torchscript"
                                << "superpoint_extractor_cuda.pt";
            modelCandidates << "superpoint_extractor_cpu.torchscript"
                            << "superpoint_extractor_cpu.pt"
                            << "superpoint_extractor.torchscript"
                            << "superpoint_extractor.pt";

            QString modelPath = config["model_path"].toString().trimmed();
            if (!modelPath.isEmpty() && !QFile::exists(modelPath))
            {
                LOG_ERROR("%s", qUtf8Printable(QString("SuperPoint 模型文件不存在: %1").arg(modelPath)));
                return false;
            }

            if (modelPath.isEmpty())
            {
                for (const QString &name : modelCandidates)
                {
                    modelPath = findModelFile(name);
                    if (!modelPath.isEmpty()) break;
                }
            }

            if (modelPath.isEmpty())
            {
                LOG_ERROR("%s", "SuperPoint 模型文件不存在 (已尝试: superpoint_extractor_*.torchscript/.pt)");
                return false;
            }

            LOG_INFO("%s", qUtf8Printable(QString("加载 SuperPoint 模型: %1").arg(modelPath)));
            superPointExtractor = std::make_unique<SuperPoint>(modelPath.toStdString(), spConfig);
        }
        else if (featureAlgorithm == QStringLiteral("disk") || featureAlgorithm == QStringLiteral("aliked"))
        {
            const QString modelPath = resolveExtractorModelPath(
                featureAlgorithm,
                spConfig.device.is_cuda(),
                config["model_path"].toString());
            if (modelPath.isEmpty())
            {
                LOG_ERROR("%s", qUtf8Printable(QString("%1 C++ TorchScript 模型文件不存在")
                    .arg(featureAlgorithm.toUpper())));
                return false;
            }

            ExtractorConfig extractorCfg;
            extractorCfg.modelPath = modelPath.toStdString();
            extractorCfg.maxKeypoints = maxKeypointsFromConfig(config);
            extractorCfg.detThreshold = static_cast<float>(config["detection_threshold"].toDouble(0.0));
            extractorCfg.nmsRadius = spConfig.nms_radius;
            extractorCfg.removeBorder = spConfig.remove_borders;
            extractorCfg.maxImageDim = 0;
            extractorCfg.grayscaleMin = spConfig.grayscale_min;
            extractorCfg.grayscaleMax = spConfig.grayscale_max;
            extractorCfg.useCuda = spConfig.device.is_cuda();
            extractorCfg.cudaDevice = config["cuda_device"].toInt(0);

            LOG_INFO("%s", qUtf8Printable(QString("加载 %1 C++ TorchScript 模型: %2")
                .arg(featureAlgorithm.toUpper(), modelPath)));
            nativeExtractor = xjw::feature_extractors::createExtractor(
                featureAlgorithm.toStdString(), extractorCfg);
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
                FeatureOutput output;
                if (superPointExtractor)
                {
                    output = superPointExtractor->detect(processImage);
                }
                else if (nativeExtractor)
                {
                    output = nativeExtractor->extract(processImage);
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
                
                // 保存结果为二进制格式 (使用算法对应的后缀)
                QString baseName = fileInfo.completeBaseName();
                QString outputPath = QDir(outputDir).filePath(baseName + fileSuffix);
                
                if (FeatureFileIO::write(outputPath, fileInfo.fileName(), output,
                                         featureAlgorithm.toStdString()))
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
                        QMetaObject::invokeMethod(projectManager.data(),
                                                  [projectManager, imagePath, outputPath, config]()
                        {
                            if (!projectManager)
                            {
                                return;
                            }
                            projectManager->appendIpfindResult(imagePath, outputPath, config);
                        },
                        Qt::QueuedConnection);
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
        LOG_ERROR("%s", qUtf8Printable(QString("特征提取初始化失败: %1")
            .arg(QString::fromStdString(e.what()))));
        return false;
    }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
