// =============================================================================
// 文件: cli_feature_match.cpp
// 功能: 统一特征匹配 CLI — 工厂模式, 自动检测算法
// =============================================================================
#include "cli_common.h"
#include "MatcherFactory.h"
#include "TraditionalFeatureMatcher.h"
#include "FeatureData.h"
#include "FeatureOutput.h"
#include "AlgorithmCompat.h"
#include "FeatureFileIO.h"
#include "MatchFileIO.h"

#include <QString>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace
{

QJsonArray makePointArray(double x, double y)
{
    QJsonArray point;
    point.append(x);
    point.append(y);
    return point;
}

void writeIndexedSidecar(const std::string &outPath,
                         const std::string &sp1,
                         const std::string &sp2,
                         const QString &imageName0,
                         const QString &imageName1,
                         const xjw::feature_extractors::FeatureData &fd0,
                         const xjw::feature_extractors::FeatureData &fd1,
                         const xjw::feature_match::MatchResult &matchResult,
                         const QString &featureAlgorithm,
                         const QString &matchAlgorithm,
                         float matchThreshold)
{
    QJsonArray points0;
    QJsonArray points1;
    QJsonArray indices0;
    QJsonArray indices1;
    QJsonArray scores;

    for (size_t i = 0; i < matchResult.matches0.size(); ++i)
    {
        const int idx1 = matchResult.matches0[i];
        if (idx1 < 0 ||
            i >= fd0.keypoints.size() ||
            idx1 >= static_cast<int>(fd1.keypoints.size()))
        {
            continue;
        }

        indices0.append(static_cast<int>(i));
        indices1.append(idx1);
        points0.append(makePointArray(fd0.keypoints[i].pt.x, fd0.keypoints[i].pt.y));
        points1.append(makePointArray(fd1.keypoints[static_cast<size_t>(idx1)].pt.x,
                                      fd1.keypoints[static_cast<size_t>(idx1)].pt.y));

        const float score = i < matchResult.matchingScores0.size()
            ? matchResult.matchingScores0[i]
            : 1.0f;
        scores.append(static_cast<double>(score));
    }

    QJsonObject sidecar;
    sidecar[QStringLiteral("match_file")] = QString::fromStdString(outPath);
    sidecar[QStringLiteral("image0_name")] = imageName0;
    sidecar[QStringLiteral("image1_name")] = imageName1;
    sidecar[QStringLiteral("image0_path")] = imageName0;
    sidecar[QStringLiteral("image1_path")] = imageName1;
    sidecar[QStringLiteral("feature0_path")] = QString::fromStdString(sp1);
    sidecar[QStringLiteral("feature1_path")] = QString::fromStdString(sp2);
    sidecar[QStringLiteral("sp0_path")] = QString::fromStdString(sp1);
    sidecar[QStringLiteral("sp1_path")] = QString::fromStdString(sp2);
    sidecar[QStringLiteral("feature_algorithm")] = featureAlgorithm;
    sidecar[QStringLiteral("match_algorithm")] = matchAlgorithm;
    sidecar[QStringLiteral("feature_format_version")] = 2;
    sidecar[QStringLiteral("num_matches")] = indices0.size();
    sidecar[QStringLiteral("match_threshold")] = static_cast<double>(matchThreshold);
    sidecar[QStringLiteral("matched_points0")] = points0;
    sidecar[QStringLiteral("matched_points1")] = points1;
    sidecar[QStringLiteral("matched_indices0")] = indices0;
    sidecar[QStringLiteral("matched_indices1")] = indices1;
    sidecar[QStringLiteral("matched_scores")] = scores;

    QFile sidecarFile(QString::fromStdString(outPath) + QStringLiteral(".json"));
    if (!sidecarFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        cli::fatal("无法写入 sidecar: " + outPath + ".json", cli::EXIT_IO_ERR);
    }
    sidecarFile.write(QJsonDocument(sidecar).toJson(QJsonDocument::Compact));
    sidecarFile.close();
}

} // namespace

// 根据文件后缀自动选择匹配器
static std::string autoMatcher(const std::string &spPath)
{
    return xjw::feature_match::defaultMatcherForFeatureSuffix(
        QString::fromStdString(spPath)).toStdString();
}

static bool isTraditionalMatcher(const std::string &algo)
{
    return algo == "bf" ||
        algo == "flann" ||
        algo == "sift_bf_l2" ||
        algo == "sift_flann" ||
        algo == "orb_bf_hamming";
}

static std::string normalizedTraditionalMatcher(const std::string &algo)
{
    if (algo == "bf") return "sift_bf_l2";
    if (algo == "flann") return "sift_flann";
    return xjw::feature_match::tradition::TraditionalFeatureMatcher::normalizeAlgorithmName(algo);
}

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 统一特征匹配 — 自动检测, 无需手动指定算法"};

    std::string algo;
    app.add_option("-a,--algorithm", algo, "留空自动检测");

    std::string modelPath, sp1, sp2, imgL, imgR, outPath;
    app.add_option("-m,--model",  modelPath, "模型路径 (.torchscript/.pt)");
    app.add_option("--sp1", sp1, "左特征文件");
    app.add_option("--sp2", sp2, "右特征文件");
    app.add_option("-L,--left",  imgL,    "左影像 (端到端模式)");
    app.add_option("-R,--right", imgR,    "右影像 (端到端模式)");
    app.add_option("-o,--output", outPath, "输出 .match 文件")->required();

    float matchThresh = 0.2f;
    int   maxKp = 2048, maxDim = 2048, gpu = 0;
    bool  cuda = true;
    app.add_option("-t,--match-threshold", matchThresh, "匹配置信度阈值");
    app.add_option("-n,--max-keypoints",   maxKp,       "最大关键点数");
    app.add_option("--max-dim", maxDim, "端到端匹配最大图像边长");
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
    if (algo == "superglue" || algo == "lightglue" ||
        algo == "loftr" || algo == "roma" || algo == "dedode")
    {
        MatcherConfig mCfg;
        mCfg.modelPath      = modelPath;
        mCfg.matchThreshold = matchThresh;
        mCfg.maxKeypoints   = maxKp;
        mCfg.maxImageDim    = maxDim;
        mCfg.useCuda        = cuda;
        mCfg.cudaDevice     = gpu;

        auto matcher = createMatcher(algo, mCfg);
        if ((algo == "superglue" || algo == "lightglue") && (sp1.empty() || sp2.empty()))
        {
            cli::fatal(algo + " 需要 --sp1/--sp2 特征文件", cli::EXIT_ARG_ERR);
        }
        if ((algo == "loftr" || algo == "roma") && (imgL.empty() || imgR.empty()))
        {
            cli::fatal(algo + " 需要 -L/--left 和 -R/--right 影像", cli::EXIT_ARG_ERR);
        }
        if (algo == "dedode" && (sp1.empty() || sp2.empty()) && (imgL.empty() || imgR.empty()))
        {
            cli::fatal("dedode 需要 --sp1/--sp2 特征文件，或 -L/-R 影像执行端到端提取+匹配", cli::EXIT_ARG_ERR);
        }

        fprintf(stdout, "%s: %s <-> %s\n",
                matcher->algorithmName().c_str(),
                sp1.empty() ? imgL.c_str() : sp1.c_str(),
                sp2.empty() ? imgR.c_str() : sp2.c_str());

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
    else if (isTraditionalMatcher(algo))
    {
        if (sp1.empty() || sp2.empty())
        {
            cli::fatal("传统匹配需要 --sp1/--sp2");
        }

        QString n1, n2;
        xjw::feature_extractors::FeatureData fd0;
        xjw::feature_extractors::FeatureData fd1;
        if (!FeatureFileIO::readData(QString::fromStdString(sp1), n1, fd0))
        {
            cli::fatal("加载失败: " + sp1, cli::EXIT_IO_ERR);
        }
        if (!FeatureFileIO::readData(QString::fromStdString(sp2), n2, fd1))
        {
            cli::fatal("加载失败: " + sp2, cli::EXIT_IO_ERR);
        }

        xjw::feature_match::tradition::TraditionalMatchConfig tc;
        tc.algorithmName = normalizedTraditionalMatcher(algo);
        tc.requireMutualConsistency = true;
        tc.ratioTestThreshold = 0.85f;
        tc.useCuda = cuda;
        tc.cudaDevice = gpu;

        fprintf(stdout, "%s: %s <-> %s (%zu/%zu kp)\n",
                tc.algorithmName.c_str(), sp1.c_str(), sp2.c_str(),
                fd0.keypoints.size(), fd1.keypoints.size());

        auto mr = xjw::feature_match::tradition::TraditionalFeatureMatcher::match(
            fd0.toCvDescriptors(tc.algorithmName),
            fd1.toCvDescriptors(tc.algorithmName),
            fd0.keypoints.size(), fd1.keypoints.size(), tc);

        if (mr.numMatches == 0)
        {
            cli::fatal("未找到匹配点", cli::EXIT_ALGO_ERR);
        }

        QByteArray writeError;
        const int count = xjw::feature_match::writeMatchFile(
            QString::fromStdString(outPath), mr, fd0.keypoints, fd1.keypoints, &writeError);
        if (count < 0)
        {
            cli::fatal("无法写入: " + outPath + " " + writeError.toStdString(), cli::EXIT_IO_ERR);
        }
        writeIndexedSidecar(outPath,
                            sp1,
                            sp2,
                            n1,
                            n2,
                            fd0,
                            fd1,
                            mr,
                            QString::fromStdString(fd0.sourceAlgorithm),
                            QString::fromStdString(tc.algorithmName),
                            matchThresh);
        fprintf(stdout, "匹配完成: %d 点 -> %s\n", count, outPath.c_str());
    }
    else
    {
        cli::fatal("未知算法: " + algo
            + ". 支持: superglue, lightglue, loftr, roma, dedode, bf, flann, "
              "sift_bf_l2, sift_flann, orb_bf_hamming");
    }
    return cli::EXIT_OK;
}
