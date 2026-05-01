// =============================================================================
// 文件: cli_feature_extract.cpp
// 功能: 统一特征提取 CLI — SuperPoint | SIFT | ORB | AKAZE
// 用法:
//   feature_extract_cli -a superpoint -m model.pt -i img.tif -o out.sp [--cuda --max-dim 2048]
//   feature_extract_cli -a sift        -i img.tif -o out.sp [-n 4096]
//   feature_extract_cli -a orb         -i img.tif -o out.sp [-n 5000]
// =============================================================================
#include "cli_common.h"
#include "SuperPoint.h"
#include "TraditionalFeatureExtractor.h"
#include "DiskExtractor.h"
#include "AlikedExtractor.h"
#include "FeatureFileIO.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <QFileInfo>
#include <QDir>
#include <QString>
#include <string>

static int processSP(const std::string &modelPath, SuperPointConfig spCfg,
                     const std::string &imgPath, const std::string &outPath, int maxDim)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) { fprintf(stderr, "加载失败: %s\n", imgPath.c_str()); return cli::EXIT_IO_ERR; }

    int origW = img.cols, origH = img.rows;
    double scale = 1.0;
    int maxSide = std::max(origW, origH);
    if (maxDim > 0 && maxSide > maxDim) {
        scale = static_cast<double>(maxDim) / maxSide;
        cv::resize(img, img, cv::Size(int(origW*scale), int(origH*scale)), 0, 0, cv::INTER_AREA);
        fprintf(stdout, "降采样: %dx%d -> %dx%d\n", origW, origH, img.cols, img.rows);
    }
    fprintf(stdout, "SuperPoint: %s (%dx%d)\n", imgPath.c_str(), img.cols, img.rows);

    SuperPoint sp(modelPath, spCfg);
    auto output = sp.detect(img);
    if (scale < 1.0) for (auto &kp : output.keypoints) { kp.pt.x /= scale; kp.pt.y /= scale; }

    if (output.keypoints.empty()) { fprintf(stderr, "未检测到关键点\n"); return cli::EXIT_ALGO_ERR; }

    QFileInfo fi(QString::fromStdString(imgPath));
    if (!FeatureFileIO::write(QString::fromStdString(outPath), fi.fileName(), output, "superpoint"))
    { fprintf(stderr, "写入失败: %s\n", outPath.c_str()); return cli::EXIT_IO_ERR; }

    fprintf(stdout, "已保存: %s (%zu kp, orig %dx%d)\n", outPath.c_str(), output.keypoints.size(), origW, origH);
    return cli::EXIT_OK;
}

static int processTraditional(const std::string &algo, SuperPointConfig spCfg,
                              const std::string &imgPath, const std::string &outPath)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) { fprintf(stderr, "加载失败: %s\n", imgPath.c_str()); return cli::EXIT_IO_ERR; }

    std::string norm = xjw::feature_extractors::TraditionalFeatureExtractor::normalizeAlgorithmName(algo);
    fprintf(stdout, "%s: %s (%dx%d)\n", norm.c_str(), imgPath.c_str(), img.cols, img.rows);

    auto output = xjw::feature_extractors::TraditionalFeatureExtractor::detect(img, spCfg, norm);
    if (output.keypoints.empty()) { fprintf(stderr, "未检测到关键点\n"); return cli::EXIT_ALGO_ERR; }

    QFileInfo fi(QString::fromStdString(imgPath));
    if (!FeatureFileIO::write(QString::fromStdString(outPath), fi.fileName(), output, algo))
    { fprintf(stderr, "写入失败: %s\n", outPath.c_str()); return cli::EXIT_IO_ERR; }

    fprintf(stdout, "已保存: %s (%zu kp)\n", outPath.c_str(), output.keypoints.size());
    return cli::EXIT_OK;
}

static int processDL(const std::string &algo, const std::string &modelPath,
                     const std::string &imgPath, const std::string &outPath,
                     int maxDim, bool cuda, int gpu, int maxKp)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) { fprintf(stderr, "加载失败: %s\n", imgPath.c_str()); return cli::EXIT_IO_ERR; }

    int origW = img.cols, origH = img.rows;
    double scale = 1.0;
    int maxSide = std::max(origW, origH);
    if (maxDim > 0 && maxSide > maxDim)
    {
        scale = static_cast<double>(maxDim) / maxSide;
        cv::resize(img, img, cv::Size(int(origW*scale), int(origH*scale)), 0, 0, cv::INTER_AREA);
        fprintf(stdout, "降采样: %dx%d -> %dx%d\n", origW, origH, img.cols, img.rows);
    }

    FeatureOutput output;

    if (algo == "disk")
    {
        xjw::feature_extractors::DiskConfig cfg;
        cfg.modelPath = modelPath;
        cfg.useCuda = cuda;
        cfg.cudaDevice = gpu;
        cfg.maxImageDim = 0;
        cfg.maxKeypoints = maxKp;
        xjw::feature_extractors::DiskExtractor ext(cfg);
        output = ext.extract(img);
    }
    else if (algo == "aliked")
    {
        xjw::feature_extractors::AlikedConfig cfg;
        cfg.modelPath = modelPath;
        cfg.useCuda = cuda;
        cfg.cudaDevice = gpu;
        cfg.maxImageDim = 0;
        cfg.maxKeypoints = maxKp;
        xjw::feature_extractors::AlikedExtractor ext(cfg);
        output = ext.extract(img);
    }

    if (scale < 1.0)
        for (auto &kp : output.keypoints) { kp.pt.x /= scale; kp.pt.y /= scale; }

    if (output.keypoints.empty()) { fprintf(stderr, "未检测到关键点\n"); return cli::EXIT_ALGO_ERR; }

    QFileInfo fi(QString::fromStdString(imgPath));
    if (!FeatureFileIO::write(QString::fromStdString(outPath), fi.fileName(), output, algo))
    { fprintf(stderr, "写入失败: %s\n", outPath.c_str()); return cli::EXIT_IO_ERR; }

    fprintf(stdout, "已保存: %s (%zu kp, %dd, orig %dx%d)\n",
            outPath.c_str(), output.keypoints.size(),
            output.descriptors.defined() ? (int)output.descriptors.size(1) : 0,
            origW, origH);
    return cli::EXIT_OK;
}

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 统一特征提取 — SuperPoint | SIFT | ORB | AKAZE | DISK | ALIKED"};

    std::string algo = "superpoint";
    app.add_option("-a,--algorithm", algo, "算法: superpoint, sift, orb, akaze, disk, aliked");

    std::string modelPath, imgPath, outPath;
    app.add_option("-m,--model", modelPath, "模型路径 (.pt, SuperPoint 必填)");
    app.add_option("-i,--input",  imgPath,  "输入影像或目录")->required();
    app.add_option("-o,--output", outPath,  "输出 .sp 文件或目录")->required();

    int  maxKp = 4096, nmsRadius = 3, removeBorder = 4, maxDim = 2048, gpu = 0;
    float detThresh = 0.003f;
    bool  cuda = true;

    app.add_option("-n,--max-keypoints", maxKp, "最大关键点数");
    app.add_option("-t,--det-threshold", detThresh, "检测阈值");
    app.add_option("--nms-radius", nmsRadius, "NMS 半径 (SuperPoint)");
    app.add_option("--remove-border", removeBorder, "边界移除像素");
    app.add_option("--max-dim", maxDim, "GPU 最大边长 (超则降采样)");
    app.add_flag("--cuda{true}", cuda, "使用 CUDA");
    app.add_flag("--no-cuda{false}", cuda);
    app.add_option("--gpu", gpu, "CUDA 设备 ID");
    bool verbose = false;
    app.add_flag("-V,--verbose", verbose);

    CLI11_PARSE(app, argc, argv);

    // 自动追加算法后缀
    std::string norm = TraditionalFeatureExtractor::normalizeAlgorithmName(algo);
    std::string suffix = ExtractorSuffix::forAlgorithm(norm);
    if (outPath.find('.') == std::string::npos)
        outPath += suffix;

    SuperPointConfig spCfg;
    spCfg.max_num_keypoints = maxKp;
    spCfg.detection_threshold = detThresh;
    spCfg.nms_radius = nmsRadius;
    spCfg.remove_borders = removeBorder;
    spCfg.allow_device_fallback = true;

    bool isSP = (norm == "superpoint");

    if (isSP && modelPath.empty())
        cli::fatal("SuperPoint 需要 -m/--model 指定模型路径");

    QFileInfo fiIn(QString::fromStdString(imgPath)), fiOut(QString::fromStdString(outPath));
    if (fiIn.isDir())
    {
        QDir inDir(fiIn.absoluteFilePath()), outDir(fiOut.absoluteFilePath());
        outDir.mkpath(".");
        QStringList filters = {"*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"};
        int ok = 0, fail = 0;
        for (const QString &fname : inDir.entryList(filters, QDir::Files, QDir::Name))
        {
            std::string in = inDir.absoluteFilePath(fname).toStdString();
            std::string out = outDir.absoluteFilePath(QFileInfo(fname).completeBaseName()+".sp").toStdString();
            int rc;
            if (isSP)           rc = processSP(modelPath, spCfg, in, out, maxDim);
            else if (norm == "disk" || norm == "aliked") rc = processDL(norm, modelPath, in, out, maxDim, cuda, gpu, maxKp);
            else                rc = processTraditional(algo, spCfg, in, out);
            (rc == cli::EXIT_OK) ? ++ok : ++fail;
        }
        fprintf(stdout, "批量: %d 成功 %d 失败\n", ok, fail);
        return fail > 0 ? cli::EXIT_ALGO_ERR : cli::EXIT_OK;
    }
    if (isSP)                           return processSP(modelPath, spCfg, imgPath, outPath, maxDim);
    if (norm == "disk" || norm == "aliked") return processDL(norm, modelPath, imgPath, outPath, maxDim, cuda, gpu, maxKp);
    return processTraditional(algo, spCfg, imgPath, outPath);
}
