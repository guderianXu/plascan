#include "DenseMatchRunner.h"

#include "DenseMatchConfig.h"
#include "DenseMatchService.h"
#include "Logger.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>

void DenseMatchRunner::run(const QJsonObject &settings,
                           std::shared_ptr<std::atomic<int>> progress,
                           std::shared_ptr<std::atomic<bool>> cancelFlag)
{
    const QJsonArray pairs = settings.value(QStringLiteral("match_pairs")).toArray();
    const QString outputDir = settings.value(QStringLiteral("output_dir")).toString();
    const int algo = settings.value(QStringLiteral("algorithm")).toInt(2);
    const int costFunc = settings.value(QStringLiteral("cost_func")).toInt(3);
    const int minDisp = settings.value(QStringLiteral("min_disparity")).toInt(0);
    const int maxDisp = settings.value(QStringLiteral("max_disparity")).toInt(256);
    const int kernelW = settings.value(QStringLiteral("kernel_w")).toInt(15);
    const int kernelH = settings.value(QStringLiteral("kernel_h")).toInt(15);
    const bool useCuda = settings.value(QStringLiteral("use_cuda")).toBool(true);
    const int cudaDevice = settings.value(QStringLiteral("cuda_device")).toInt(0);
    const int subpixel = settings.value(QStringLiteral("subpixel_mode")).toInt(1);
    const int p1 = settings.value(QStringLiteral("p1")).toInt(8);
    const int p2 = settings.value(QStringLiteral("p2")).toInt(32);
    const int directions = settings.value(QStringLiteral("directions")).toInt(8);
    const int pyramid = settings.value(QStringLiteral("pyramid")).toInt(2);
    const double lrThreshold = settings.value(QStringLiteral("lr_threshold")).toDouble(1.0);
    const int medianFilter = settings.value(QStringLiteral("median_filter")).toInt(3);
    const int numThreads = settings.value(QStringLiteral("threads")).toInt(4);

    LOG_INFO(QStringLiteral("密集匹配: %1 个匹配对, 算法=%2 代价=%3 CUDA=%4 视差=[%5,%6] 线程=%7")
        .arg(pairs.size()).arg(algo).arg(costFunc).arg(useCuda).arg(minDisp).arg(maxDisp)
        .arg(numThreads));

    QDir().mkpath(outputDir);

    int completed = 0;
    for (const QJsonValue &val : pairs)
    {
        if (cancelFlag && cancelFlag->load())
        {
            LOG_INFO(QStringLiteral("密集匹配已请求取消，停止处理剩余匹配对"));
            break;
        }

        const QJsonObject pair = val.toObject();
        const QString imgA = pair.value(QStringLiteral("imgA")).toString();
        const QString imgB = pair.value(QStringLiteral("imgB")).toString();

        QFileInfo fiA(imgA);
        QFileInfo fiB(imgB);
        LOG_INFO(QStringLiteral("[密集匹配 %1/%2] %3 <-> %4")
            .arg(completed + 1).arg(pairs.size())
            .arg(fiA.fileName()).arg(fiB.fileName()));

        xjw::dense_match::DenseMatchConfig cfg;
        cfg.algorithm = static_cast<xjw::dense_match::StereoAlgorithm>(algo);
        cfg.costFunc = static_cast<xjw::dense_match::CostFunction>(costFunc);
        cfg.subpixel = static_cast<xjw::dense_match::SubpixelMode>(subpixel);
        cfg.minDisparity = minDisp;
        cfg.maxDisparity = maxDisp;
        cfg.corrKernelW = kernelW;
        cfg.corrKernelH = kernelH;
        cfg.useCuda = useCuda;
        cfg.cudaDevice = cudaDevice;
        cfg.p1 = p1;
        cfg.p2 = p2;
        cfg.sgmDirections = directions;
        cfg.pyramidLevels = pyramid;
        cfg.lrCheckThreshold = static_cast<float>(lrThreshold);
        cfg.medianFilterSize = medianFilter;
        cfg.enableLRCheck = lrThreshold > 0.0;
        cfg.numThreads = numThreads;
        cfg.leftImagePath = imgA.toStdString();
        cfg.rightImagePath = imgB.toStdString();

        cfg.outputDisparityPath = QStringLiteral("%1/%2__%3_disp.tif")
            .arg(outputDir)
            .arg(fiA.completeBaseName())
            .arg(fiB.completeBaseName()).toStdString();

        if (progress)
        {
            progress->store(completed * 5 + 1);
        }

        xjw::dense_match::DenseMatchService service(cfg);
        auto result = service.process();
        if (cancelFlag && cancelFlag->load())
        {
            LOG_INFO(QStringLiteral("密集匹配已请求取消，跳过当前结果保存"));
            break;
        }

        if (!result.disparity.empty())
        {
            if (progress)
            {
                progress->store(completed * 5 + 5);
            }

            xjw::dense_match::DenseMatchService::saveDisparity(result, cfg.outputDisparityPath);
            LOG_INFO(QStringLiteral("[密集匹配 %1/%2] 完成 -> %3")
                .arg(completed + 1).arg(pairs.size())
                .arg(QString::fromStdString(cfg.outputDisparityPath)));
        }
        else
        {
            LOG_INFO(QStringLiteral("[密集匹配 %1/%2] 失败: 视差图为空")
                .arg(completed + 1).arg(pairs.size()));
        }

        ++completed;
        if (progress)
        {
            progress->store(completed * 5);
        }
    }
}
