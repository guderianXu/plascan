// =============================================================================
// 文件: cli_feature_match.cpp
// 功能: 统一特征匹配 CLI — 工厂模式, 自动检测算法
// =============================================================================
#include "cli_common.h"
#include "MatcherFactory.h"
#include "TraditionalFeatureMatcher.h"
#include "FeatureData.h"
#include "FeatureOutput.h"
#include "FeatureFileIO.h"

#include <QString>
#include <QFile>
#include <QDataStream>

// 根据文件后缀自动选择匹配器
static std::string autoMatcher(const std::string &spPath)
{
    auto pos = spPath.rfind('.');
    if (pos == std::string::npos) return "superglue";
    std::string ext = spPath.substr(pos);
    if (ext == ".sp" || ext == ".dedode") return "superglue";
    if (ext == ".dsk" || ext == ".alk" || ext == ".sift") return "bf";
    return "superglue";
}

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 统一特征匹配 — 自动检测, 无需手动指定算法"};

    std::string algo;
    app.add_option("-a,--algorithm", algo, "留空自动检测");

    std::string modelPath, sp1, sp2, imgL, imgR, outPath;
    app.add_option("-m,--model",  modelPath, "模型路径 (.pt)");
    app.add_option("--sp1", sp1, "左特征文件");
    app.add_option("--sp2", sp2, "右特征文件");
    app.add_option("-L,--left",  imgL,    "左影像 (端到端模式)");
    app.add_option("-R,--right", imgR,    "右影像 (端到端模式)");
    app.add_option("-o,--output", outPath, "输出 .match 文件")->required();

    float matchThresh = 0.2f;
    int   maxKp = 2048, gpu = 0;
    bool  cuda = true;
    app.add_option("-t,--match-threshold", matchThresh, "匹配置信度阈值");
    app.add_option("-n,--max-keypoints",   maxKp,       "最大关键点数");
    app.add_flag("--cuda{true}", cuda, "CUDA");
    app.add_flag("--no-cuda{false}", cuda);
    app.add_option("--gpu", gpu, "CUDA 设备 ID");

    CLI11_PARSE(app, argc, argv);

    // 自动检测
    if (algo.empty() && !sp1.empty())
    {
        algo = autoMatcher(sp1);
        fprintf(stdout, "自动检测: %s → %s\n", sp1.c_str(), algo.c_str());
    }
    if (algo.empty())
    {
        algo = "superglue";
    }

    // ── 工厂统一处理 ──
    if (algo == "superglue" || algo == "lightglue"
        || algo == "loftr" || algo == "disk" || algo == "aliked")
    {
        MatcherConfig mCfg;
        mCfg.modelPath      = modelPath;
        mCfg.matchThreshold = matchThresh;
        mCfg.maxKeypoints   = maxKp;
        mCfg.useCuda        = cuda;
        mCfg.cudaDevice     = gpu;

        auto matcher = createMatcher(algo, mCfg);
        fprintf(stdout, "%s: %s <-> %s\n",
                matcher->algorithmName().c_str(), sp1.c_str(), sp2.c_str());

        int n = matcher->match(sp1, sp2, imgL, imgR, outPath);
        if (n < 0)
        {
            cli::fatal("匹配失败", cli::EXIT_ALGO_ERR);
        }
        if (n == 0)
        {
            cli::fatal("未找到匹配点", cli::EXIT_ALGO_ERR);
        }
        fprintf(stdout, "匹配完成: %d 点 -> %s\n", n, outPath.c_str());
    }
    else if (algo == "bf" || algo == "flann")
    {
        if (sp1.empty() || sp2.empty())
        {
            cli::fatal("bf/flann 需要 --sp1/--sp2");
        }

        FeatureOutput spo1, spo2;
        QString n1, n2;
        if (!FeatureFileIO::read(QString::fromStdString(sp1), n1, spo1))
        {
            cli::fatal("加载失败: " + sp1, cli::EXIT_IO_ERR);
        }
        if (!FeatureFileIO::read(QString::fromStdString(sp2), n2, spo2))
        {
            cli::fatal("加载失败: " + sp2, cli::EXIT_IO_ERR);
        }

        auto fd0 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo1);
        auto fd1 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo2);

        xjw::feature_match::tradition::TraditionalMatchConfig tc;
        tc.algorithmName = (algo == "bf") ? "sift_bf_l2" : "sift_flann";
        tc.requireMutualConsistency = true;
        tc.ratioTestThreshold = 0.85f;

        fprintf(stdout, "%s(L2): %s <-> %s (%zu/%zu kp)\n",
                algo.c_str(), sp1.c_str(), sp2.c_str(),
                fd0.keypoints.size(), fd1.keypoints.size());

        auto mr = xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
            fd0.descriptors, fd1.descriptors,
            fd0.keypoints.size(), fd1.keypoints.size(), tc);

        if (mr.numMatches == 0)
        {
            cli::fatal("未找到匹配点", cli::EXIT_ALGO_ERR);
        }

        QFile f(QString::fromStdString(outPath));
        if (!f.open(QIODevice::WriteOnly))
        {
            cli::fatal("无法写入: " + outPath, cli::EXIT_IO_ERR);
        }
        QDataStream ds(&f);
        ds.setByteOrder(QDataStream::BigEndian);
        qint32 count = 0;
        for (size_t i = 0; i < mr.matches0.size(); ++i)
        {
            if (mr.matches0[i] >= 0) ++count;
        }
        ds << count;
        for (size_t i = 0; i < mr.matches0.size(); ++i)
        {
            int m1 = mr.matches0[i];
            if (m1 < 0) continue;
            ds << fd0.keypoints[i].pt.x << fd0.keypoints[i].pt.y
               << fd1.keypoints[m1].pt.x << fd1.keypoints[m1].pt.y;
        }
        f.close();
        fprintf(stdout, "匹配完成: %d 点 -> %s\n", count, outPath.c_str());
    }
    else
    {
        cli::fatal("未知算法: " + algo
            + ". 支持: superglue, lightglue, loftr, disk, aliked, bf, flann");
    }
    return cli::EXIT_OK;
}
