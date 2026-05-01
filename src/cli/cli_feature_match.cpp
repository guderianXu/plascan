// =============================================================================
// 文件: cli_feature_match.cpp
// 功能: 统一特征匹配 CLI — SuperGlue | LightGlue | LoFTR | BF | FLANN
// 用法:
//   # 稀疏匹配 (需要 .sp 文件)
//   feature_match_cli -a superglue  -m sg.pt --sp1 a.sp --sp2 b.sp -o out.match
//   feature_match_cli -a lightglue  -m lg.pt --sp1 a.sp --sp2 b.sp -o out.match
//   feature_match_cli -a bf         --sp1 a.sp --sp2 b.sp -o out.match
//   # LoFTR 端到端 (直接影像, 无需特征提取)
//   feature_match_cli -a loftr -m loftr_outdoor_cuda.pt -L A.tif -R B.tif -o out.match --cuda
// =============================================================================
#include "cli_common.h"
#include "SuperGlueMatcher.h"
#include "LightGlueMatcher.h"
#include "TraditionalFeatureMatcher.h"
#include "FeatureData.h"
#include "SuperPoint.h"
#include "FeatureFileIO.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <QString>
#include <QFile>
#include <QDataStream>
#include <QProcess>
#include <QCoreApplication>

// 保存 .match 文件 (matches0 索引格式)
static int saveMatchFile(const std::string &path,
                         const xjw::feature_match::MatchResult &mr,
                         const std::vector<cv::KeyPoint> &kp0,
                         const std::vector<cv::KeyPoint> &kp1)
{
    QFile f(QString::fromStdString(path));
    if (!f.open(QIODevice::WriteOnly)) return cli::EXIT_IO_ERR;
    QDataStream ds(&f);
    ds.setByteOrder(QDataStream::BigEndian);

    qint32 count = 0;
    for (size_t i = 0; i < mr.matches0.size(); ++i)
        if (mr.matches0[i] >= 0) ++count;

    ds << count;
    for (size_t i = 0; i < mr.matches0.size(); ++i)
    {
        int m1 = mr.matches0[i];
        if (m1 < 0) continue;
        ds << kp0[i].pt.x << kp0[i].pt.y << kp1[m1].pt.x << kp1[m1].pt.y;
    }
    f.close();
    return count;
}

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 统一特征匹配 — SuperGlue | LightGlue | LightGlue-E2E | BF | FLANN"};

    std::string algo = "superglue";
    app.add_option("-a,--algorithm", algo,
        "算法: superglue, lightglue, lightglue-e2e, bf, flann");

    std::string modelPath, spModelPath, sp1, sp2, imgL, imgR, outPath;
    app.add_option("-m,--model",    modelPath,   "匹配模型路径 (.pt)");
    app.add_option("--sp-model",    spModelPath, "SuperPoint 模型 (lightglue-e2e 需要)");
    app.add_option("--sp1",         sp1,         "左特征文件 (.sp)");
    app.add_option("--sp2",         sp2,         "右特征文件 (.sp)");
    app.add_option("-L,--left",     imgL,        "左影像 (lightglue-e2e)");
    app.add_option("-R,--right",    imgR,        "右影像 (lightglue-e2e)");
    app.add_option("-o,--output",   outPath,     "输出 .match 文件")->required();

    float matchThresh = 0.2f;
    int   maxKp = 2048, gpu = 0;
    bool  cuda = true;
    app.add_option("-t,--match-threshold", matchThresh, "匹配置信度阈值");
    app.add_option("-n,--max-keypoints",   maxKp,       "最大关键点数");
    app.add_flag("--cuda{true}", cuda, "使用 CUDA");
    app.add_flag("--no-cuda{false}", cuda);
    app.add_option("--gpu", gpu, "CUDA 设备 ID");
    bool verbose = false;
    app.add_flag("-V,--verbose", verbose);

    CLI11_PARSE(app, argc, argv);

    if (algo == "lightglue-e2e")
    {
        // LightGlue 端到端: 直接匹配影像
        if (modelPath.empty()) cli::fatal("lightglue-e2e 需要 -m 匹配器模型");
        if (spModelPath.empty()) cli::fatal("lightglue-e2e 需要 --sp-model SP模型");
        if (imgL.empty() || imgR.empty()) cli::fatal("lightglue-e2e 需要 -L/-R 影像");

        xjw::feature_match::LightGlueConfig lgCfg;
        lgCfg.matcherModelPath = modelPath;
        lgCfg.spModelPath      = spModelPath;
        lgCfg.useCuda          = cuda;
        lgCfg.scoreThreshold   = matchThresh;

        xjw::feature_match::LightGlueMatcher matcher(lgCfg);

        cv::Mat left  = cv::imread(imgL, cv::IMREAD_COLOR);
        cv::Mat right = cv::imread(imgR, cv::IMREAD_COLOR);
        if (left.empty() || right.empty())
            cli::fatal("影像加载失败", cli::EXIT_IO_ERR);

        fprintf(stdout, "LightGlue端到端: %s <-> %s\n", imgL.c_str(), imgR.c_str());

        auto mr = matcher.matchImages(left, right);
        if (mr.numMatches == 0) cli::fatal("未找到匹配点", cli::EXIT_ALGO_ERR);

        // 提取关键点 (需要重新提取以获得坐标)
        auto fd0 = matcher.extractNative(left);
        auto fd1 = matcher.extractNative(right);
        int n = saveMatchFile(outPath, mr, fd0.keypoints, fd1.keypoints);
        fprintf(stdout, "匹配完成: %d 点 -> %s\n", n, outPath.c_str());
    }
    else if (algo == "lightglue")
    {
        // LightGlue 稀疏模式: 需要 .sp 文件
        if (modelPath.empty()) cli::fatal("lightglue 需要 -m 匹配器模型");
        if (sp1.empty() || sp2.empty()) cli::fatal("lightglue 需要 --sp1/--sp2");

        xjw::feature_match::LightGlueConfig lgCfg;
        lgCfg.matcherModelPath = modelPath;
        lgCfg.useCuda          = cuda;
        lgCfg.scoreThreshold   = matchThresh;

        xjw::feature_match::LightGlueMatcher matcher(lgCfg);

        FeatureOutput spo1, spo2; QString n1, n2;
        if (!FeatureFileIO::read(QString::fromStdString(sp1), n1, spo1))
            cli::fatal("加载失败: " + sp1, cli::EXIT_IO_ERR);
        if (!FeatureFileIO::read(QString::fromStdString(sp2), n2, spo2))
            cli::fatal("加载失败: " + sp2, cli::EXIT_IO_ERR);

        auto fd0 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo1);
        auto fd1 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo2);

        fprintf(stdout, "LightGlue: %s <-> %s (%zu/%zu kp)\n",
                sp1.c_str(), sp2.c_str(), fd0.keypoints.size(), fd1.keypoints.size());

        auto mr = matcher.match(fd0, fd1);
        if (mr.numMatches == 0) cli::fatal("未找到匹配点", cli::EXIT_ALGO_ERR);
        int n = saveMatchFile(outPath, mr, fd0.keypoints, fd1.keypoints);
        fprintf(stdout, "匹配完成: %d 点 -> %s\n", n, outPath.c_str());
    }
    else if (algo == "superglue")
    {
        if (modelPath.empty()) cli::fatal("superglue 需要 -m 模型");
        if (sp1.empty() || sp2.empty()) cli::fatal("superglue 需要 --sp1/--sp2");

        superglue::SuperGlueConfig sgCfg;
        sgCfg.model_path      = modelPath;
        sgCfg.match_threshold = matchThresh;
        sgCfg.max_keypoints   = maxKp;
        sgCfg.use_cuda        = cuda;
        sgCfg.cuda_device_id  = gpu;

        superglue::SuperGlueMatcher matcher(sgCfg);

        FeatureOutput spo1, spo2; QString n1, n2;
        if (!FeatureFileIO::read(QString::fromStdString(sp1), n1, spo1))
            cli::fatal("加载失败: " + sp1, cli::EXIT_IO_ERR);
        if (!FeatureFileIO::read(QString::fromStdString(sp2), n2, spo2))
            cli::fatal("加载失败: " + sp2, cli::EXIT_IO_ERR);

        auto fd0 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo1);
        auto fd1 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo2);

        fprintf(stdout, "SuperGlue: %s <-> %s (%zu/%zu kp)\n",
                sp1.c_str(), sp2.c_str(), fd0.keypoints.size(), fd1.keypoints.size());

        auto mr = matcher.match(fd0, fd1);
        if (mr.numMatches == 0) cli::fatal("未找到匹配点", cli::EXIT_ALGO_ERR);
        int n = saveMatchFile(outPath, mr, fd0.keypoints, fd1.keypoints);
        fprintf(stdout, "匹配完成: %d 点 -> %s\n", n, outPath.c_str());
    }
    else if (algo == "bf" || algo == "flann")
    {
        if (sp1.empty() || sp2.empty()) cli::fatal("bf/flann 需要 --sp1/--sp2");

        FeatureOutput spo1, spo2; QString n1, n2;
        if (!FeatureFileIO::read(QString::fromStdString(sp1), n1, spo1))
            cli::fatal("加载失败: " + sp1, cli::EXIT_IO_ERR);
        if (!FeatureFileIO::read(QString::fromStdString(sp2), n2, spo2))
            cli::fatal("加载失败: " + sp2, cli::EXIT_IO_ERR);

        auto fd0 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo1);
        auto fd1 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo2);

        xjw::feature_match::tradition::TraditionalMatchConfig tc;
        // SP float描述子需要 L2距离, 归一化名必须传 sift_bf_l2/sift_flann
        tc.algorithmName = (algo == "bf") ? "sift_bf_l2" : "sift_flann";
        tc.requireMutualConsistency = true;
        tc.ratioTestThreshold = 0.85f;

        fprintf(stdout, "%s(L2): %s <-> %s (%zu/%zu kp)\n",
                algo.c_str(), sp1.c_str(), sp2.c_str(),
                fd0.keypoints.size(), fd1.keypoints.size());

        auto mr = xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
            fd0.descriptors, fd1.descriptors,
            fd0.keypoints.size(), fd1.keypoints.size(), tc);

        if (mr.numMatches == 0) cli::fatal("未找到匹配点", cli::EXIT_ALGO_ERR);
        int n = saveMatchFile(outPath, mr, fd0.keypoints, fd1.keypoints);
        fprintf(stdout, "匹配完成: %d 点 -> %s\n", n, outPath.c_str());
    }
    else if (algo == "loftr")
    {
        // LoFTR 通过 Python subprocess 运行 (TorchScript trace 有 bug)
        if (imgL.empty() || imgR.empty()) cli::fatal("loftr 需要 -L/-R 影像");

        fprintf(stdout, "LoFTR (via Python): %s <-> %s\n",
                imgL.c_str(), imgR.c_str());

        // 查找 Python 脚本路径 (相对于可执行文件)
        QString scriptPath = QCoreApplication::applicationDirPath()
            + "/../../scripts/run_loftr.py";

        QStringList args;
        args << scriptPath
             << "-L" << QString::fromStdString(imgL)
             << "-R" << QString::fromStdString(imgR)
             << "-o" << QString::fromStdString(outPath)
             << "--max-dim" << QString::number(maxKp > 0 ? maxKp : 1200);
        if (cuda) args << "--cuda";

        QProcess proc;
        proc.start("python3", args);
        if (!proc.waitForFinished(600000))  // 10 min timeout
        {
            proc.kill();
            cli::fatal("LoFTR 超时");
        }

        QString output = proc.readAllStandardOutput();
        fprintf(stdout, "%s", output.toStdString().c_str());

        if (proc.exitCode() != 0)
        {
            QString err = proc.readAllStandardError();
            fprintf(stderr, "%s", err.toStdString().c_str());
            cli::fatal("LoFTR 失败", cli::EXIT_ALGO_ERR);
        }
    }
    else if (algo == "disk" || algo == "aliked")
    {
        // DISK/ALIKED 通过 Python subprocess (TorchScript trace 有 bug)
        if (imgL.empty() || imgR.empty())
            cli::fatal(algo + " 需要 -L/-R 影像");

        fprintf(stdout, "%s (via Python): %s <-> %s\n",
                algo.c_str(), imgL.c_str(), imgR.c_str());

        QString script = QCoreApplication::applicationDirPath()
            + "/../../scripts/run_disk_aliked.py";

        QStringList args;
        args << script << "-a" << QString::fromStdString(algo)
             << "-L" << QString::fromStdString(imgL)
             << "-R" << QString::fromStdString(imgR)
             << "-o" << QString::fromStdString(outPath);
        if (cuda) args << "--cuda";

        QProcess proc;
        proc.start("python3", args);
        if (!proc.waitForFinished(300000))
        {
            proc.kill();
            cli::fatal(algo + " 超时");
        }

        QString output = proc.readAllStandardOutput();
        fprintf(stdout, "%s", output.toStdString().c_str());

        if (proc.exitCode() != 0)
        {
            fprintf(stderr, "%s", proc.readAllStandardError().constData());
            cli::fatal(algo + " 失败", cli::EXIT_ALGO_ERR);
        }
    }
    else
    {
        cli::fatal("未知算法: " + algo + ". 支持: superglue, lightglue, loftr, disk, aliked, bf, flann");
    }
    return cli::EXIT_OK;
}
