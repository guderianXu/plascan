// =============================================================================
// 文件: cli_feature_extract.cpp
// 功能: 统一特征提取 CLI — SuperPoint | SIFT | ORB | AKAZE
// 用法:
//   feature_extract_cli -a superpoint -m model.pt -i img.tif -o out.sp [--cuda --max-dim 2048]
//   feature_extract_cli -a sift        -i img.tif -o out.sp [-n 4096]
//   feature_extract_cli -a orb         -i img.tif -o out.sp [-n 5000]
// =============================================================================
#include "cli_common.h"
#include "IExtractor.h"
#include "ExtractorFactory.h"
#include "FeatureFileIO.h"

#include <opencv2/imgcodecs.hpp>
#include <QFileInfo>
#include <QDir>
#include <QString>
#include <memory>

static int processOne(const std::string &algo,
                      std::unique_ptr<IExtractor> &extractor,
                      const std::string &imgPath,
                      const std::string &outPath)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) { fprintf(stderr, "加载失败: %s\n", imgPath.c_str()); return cli::EXIT_IO_ERR; }

    fprintf(stdout, "%s: %s (%dx%d)\n",
            extractor->algorithmName().c_str(), imgPath.c_str(), img.cols, img.rows);

    auto output = extractor->extract(img);
    if (output.empty()) { fprintf(stderr, "未检测到关键点\n"); return cli::EXIT_ALGO_ERR; }

    QFileInfo fi(QString::fromStdString(imgPath));
    if (!FeatureFileIO::write(QString::fromStdString(outPath), fi.fileName(),
                              output, extractor->algorithmName()))
    { fprintf(stderr, "写入失败: %s\n", outPath.c_str()); return cli::EXIT_IO_ERR; }

    fprintf(stdout, "已保存: %s (%zu kp)\n", outPath.c_str(), output.keypoints.size());
    return cli::EXIT_OK;
}

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 统一特征提取 — SuperPoint | SIFT | ORB | AKAZE | DISK | ALIKED"};

    std::string algo = "superpoint";
    app.add_option("-a,--algorithm", algo, "算法: superpoint, sift, orb, akaze, surf, disk, aliked");

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

    // 创建提取器 (工厂 + 多态, 无需 if/else 链)
    ExtractorConfig eCfg;
    eCfg.modelPath    = modelPath;
    eCfg.maxKeypoints = maxKp;
    eCfg.detThreshold = detThresh;
    eCfg.nmsRadius    = nmsRadius;
    eCfg.removeBorder = removeBorder;
    eCfg.maxImageDim  = maxDim;
    eCfg.useCuda      = cuda;
    eCfg.cudaDevice   = gpu;

    auto extractor = createExtractor(algo, eCfg);
    fprintf(stdout, "算法: %s\n", extractor->algorithmName().c_str());

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
            std::string out = outDir.absoluteFilePath(
                QFileInfo(fname).completeBaseName().toStdString() + suffix);
            int rc = processOne(algo, extractor, in, out);
            (rc == cli::EXIT_OK) ? ++ok : ++fail;
        }
        fprintf(stdout, "批量: %d 成功 %d 失败\n", ok, fail);
        return fail > 0 ? cli::EXIT_ALGO_ERR : cli::EXIT_OK;
    }
    return processOne(algo, extractor, imgPath, outPath);
}
