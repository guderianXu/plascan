// =============================================================================
// 文件: cli_feature_extract.cpp
// 功能: 统一特征提取 CLI — SuperPoint | DISK | ALIKED | SIFT | ORB | AKAZE | SURF | DeDoDe
// 用法:
//   feature_extract_cli -a superpoint -m model.torchscript -i img.tif -o out.sp [--cuda --max-dim 2048]
//   feature_extract_cli -a sift        -i img.tif -o out.sp [-n 4096]
//   feature_extract_cli -a orb         -i img.tif -o out.sp [-n 5000]
// =============================================================================
#include "cli_common.h"
#include "IExtractor.h"
#include "TraditionalFeatureExtractor.h"
#include "ExtractorFactory.h"
#include "FeatureFileIO.h"
#include "io/PathIO.h"
#include "string_utils/StringTransform.h"

#include <opencv2/imgcodecs.hpp>
#include <QFileInfo>
#include <QDir>
#include <QString>
#include <QCoreApplication>
#include <QProcess>
#include <QStandardPaths>
#include <memory>

namespace
{

QString resolvedExecutablePath(const QString &candidate)
{
    const QString trimmed = candidate.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    if (trimmed.contains(QLatin1Char('/')) || trimmed.contains(QLatin1Char('\\')))
    {
        const QFileInfo info(trimmed);
        if (info.exists() && info.isFile() && info.isExecutable())
        {
            return QDir::cleanPath(info.absoluteFilePath());
        }
        return QString();
    }

    const QString resolved = QStandardPaths::findExecutable(trimmed);
    return resolved.isEmpty() ? QString() : QDir::cleanPath(QFileInfo(resolved).absoluteFilePath());
}

QString repoLocalPythonExecutable()
{
#ifdef PLASCAN_SOURCE_DIR
#ifdef Q_OS_WIN
    return QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral(".venv/Scripts/python.exe"));
#else
    return QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral(".venv/bin/python"));
#endif
#else
    return QString();
#endif
}

QString pythonExecutable()
{
    static const QString cached = []()
    {
        QStringList candidates;
        candidates << qEnvironmentVariable("PLASCAN_PYTHON_EXECUTABLE").trimmed()
                   << qEnvironmentVariable("PLASCAN_PYTHON").trimmed()
                   << repoLocalPythonExecutable()
                   << qEnvironmentVariable("PYTHON").trimmed()
                   << QStringLiteral("python3")
                   << QStringLiteral("python");
        candidates.removeAll(QString());
        candidates.removeDuplicates();

        for (const QString &candidate : candidates)
        {
            const QString resolved = resolvedExecutablePath(candidate);
            if (!resolved.isEmpty())
            {
                return resolved;
            }
        }
        return QStringLiteral("python");
    }();
    return cached;
}

QString findScriptFile(const QString &scriptName)
{
    QStringList candidates;

    const QString envScriptDir = qEnvironmentVariable("PLASCAN_SCRIPT_DIR").trimmed();
    if (!envScriptDir.isEmpty())
    {
        candidates.append(QDir(envScriptDir).filePath(scriptName));
    }

#ifdef PLASCAN_SOURCE_DIR
    candidates.append(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("scripts/%1").arg(scriptName)));
#endif

    const QString exeDir = QCoreApplication::applicationDirPath();
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../scripts/%1").arg(scriptName)));
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../../scripts/%1").arg(scriptName)));
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../../../scripts/%1").arg(scriptName)));
    candidates.append(QDir(QDir::currentPath()).filePath(QStringLiteral("scripts/%1").arg(scriptName)));

    for (const QString &candidate : candidates)
    {
        if (QFileInfo::exists(candidate))
        {
            return QDir::cleanPath(QFileInfo(candidate).absoluteFilePath());
        }
    }
    return QString();
}

int runPythonDedodeExtract(const std::string &imgPath,
                           const std::string &outPath,
                           bool cuda,
                           int maxDim,
                           int maxKp)
{
    const QString script = findScriptFile(QStringLiteral("workflows/run_dedode.py"));
    if (script.isEmpty())
    {
        fprintf(stderr, "未找到 DeDoDe 脚本 scripts/workflows/run_dedode.py\n");
        return cli::EXIT_IO_ERR;
    }

    QStringList args;
    args << script
         << QStringLiteral("-i") << xjw::common::io::fromUtf8Path(imgPath)
         << QStringLiteral("-o") << xjw::common::io::fromUtf8Path(outPath)
         << QStringLiteral("--max-dim") << QString::number(maxDim)
         << QStringLiteral("--max-kp") << QString::number(maxKp);
    if (cuda)
    {
        args << QStringLiteral("--cuda");
    }

    QProcess process;
    process.start(pythonExecutable(), args);
    if (!process.waitForFinished(600000))
    {
        process.kill();
        fprintf(stderr, "DeDoDe 提取超时: %s\n", imgPath.c_str());
        return cli::EXIT_ALGO_ERR;
    }
    fprintf(stdout, "%s", process.readAllStandardOutput().constData());
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
    {
        fprintf(stderr, "%s", process.readAllStandardError().constData());
        return cli::EXIT_ALGO_ERR;
    }
    return cli::EXIT_OK;
}

} // namespace

static int processOne(const std::string &algo,
                      std::unique_ptr<IExtractor> &extractor,
                      const std::string &imgPath,
                      const std::string &outPath)
{
    cv::Mat img = xjw::common::io::readImage(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) { fprintf(stderr, "加载失败: %s\n", imgPath.c_str()); return cli::EXIT_IO_ERR; }

    fprintf(stdout, "%s: %s (%dx%d)\n",
            extractor->algorithmName().c_str(), imgPath.c_str(), img.cols, img.rows);

    auto output = extractor->extract(img);
    if (output.empty()) { fprintf(stderr, "未检测到关键点\n"); return cli::EXIT_ALGO_ERR; }

    QFileInfo fi(xjw::common::io::fromUtf8Path(imgPath));
    if (!FeatureFileIO::write(xjw::common::io::fromUtf8Path(outPath), fi.fileName(),
                              output, extractor->algorithmName()))
    { fprintf(stderr, "写入失败: %s\n", outPath.c_str()); return cli::EXIT_IO_ERR; }

    fprintf(stdout, "已保存: %s (%zu kp)\n", outPath.c_str(), output.keypoints.size());
    return cli::EXIT_OK;
}

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 统一特征提取 — SuperPoint | DISK | ALIKED | SIFT | ORB | AKAZE | SURF | DeDoDe"};

    std::string algo = "superpoint";
    app.add_option("-a,--algorithm", algo, "算法: superpoint, disk, aliked, sift, orb, akaze, surf, dedode");

    std::string modelPath, imgPath, outPath;
    app.add_option("-m,--model", modelPath, "模型路径 (.torchscript/.pt, 深度学习 C++ 提取器必填)");
    app.add_option("-i,--input",  imgPath,  "输入影像或目录")->required();
    app.add_option("-o,--output", outPath,  "输出特征文件或目录")->required();

    int  maxKp = 4096, nmsRadius = 3, removeBorder = 4, maxDim = 2048, gpu = 0;
    float detThresh = 0.003f, grayscaleMin = 0.0f, grayscaleMax = 1.0f;
    bool  cuda = true;

    app.add_option("-n,--max-keypoints", maxKp, "最大关键点数");
    app.add_option("-t,--det-threshold", detThresh, "检测阈值");
    app.add_option("--grayscale-min", grayscaleMin, "灰度阈值下限 [0,1]");
    app.add_option("--grayscale-max", grayscaleMax, "灰度阈值上限 [0,1]");
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
    std::string norm = xjw::feature_extractors::TraditionalFeatureExtractor::normalizeAlgorithmName(algo);
    const std::string requestedAlgo = xjw::common::string_utils::asciiLowerCopy(algo);
    if (requestedAlgo == "dedode")
    {
        norm = "dedode";
    }
    std::string suffix = ExtractorSuffix::forAlgorithm(norm);
    QFileInfo fiIn(xjw::common::io::fromUtf8Path(imgPath));
    if (!fiIn.isDir() && QFileInfo(xjw::common::io::fromUtf8Path(outPath)).suffix().isEmpty())
    {
        outPath += suffix;
    }

    if (norm == "dedode")
    {
        if (fiIn.isDir())
        {
            QDir inDir(fiIn.absoluteFilePath());
            QDir outDir(xjw::common::io::fromUtf8Path(outPath));
            outDir.mkpath(QStringLiteral("."));
            QStringList filters = {"*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"};
            int ok = 0, fail = 0;
            for (const QString &fname : inDir.entryList(filters, QDir::Files, QDir::Name))
            {
                const std::string in = xjw::common::io::toUtf8Path(inDir.absoluteFilePath(fname));
                const QString outName = QFileInfo(fname).completeBaseName() + QString::fromStdString(suffix);
                const std::string out = xjw::common::io::toUtf8Path(outDir.absoluteFilePath(outName));
                const int rc = runPythonDedodeExtract(in, out, cuda, maxDim, maxKp);
                (rc == cli::EXIT_OK) ? ++ok : ++fail;
            }
            fprintf(stdout, "DeDoDe 批量: %d 成功 %d 失败\n", ok, fail);
            return fail > 0 ? cli::EXIT_ALGO_ERR : cli::EXIT_OK;
        }
        return runPythonDedodeExtract(imgPath, outPath, cuda, maxDim, maxKp);
    }

    // 创建提取器 (工厂 + 多态, 无需 if/else 链)
    ExtractorConfig eCfg;
    eCfg.modelPath    = modelPath;
    eCfg.maxKeypoints = maxKp;
    eCfg.detThreshold = detThresh;
    eCfg.nmsRadius    = nmsRadius;
    eCfg.removeBorder = removeBorder;
    eCfg.maxImageDim  = maxDim;
    eCfg.grayscaleMin = grayscaleMin;
    eCfg.grayscaleMax = grayscaleMax;
    eCfg.useCuda      = cuda;
    eCfg.cudaDevice   = gpu;

    auto extractor = xjw::feature_extractors::createExtractor(algo, eCfg);
    fprintf(stdout, "算法: %s\n", extractor->algorithmName().c_str());

    QFileInfo fiOut(xjw::common::io::fromUtf8Path(outPath));
    if (fiIn.isDir())
    {
        QDir inDir(fiIn.absoluteFilePath()), outDir(fiOut.absoluteFilePath());
        outDir.mkpath(".");
        QStringList filters = {"*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"};
        int ok = 0, fail = 0;
        for (const QString &fname : inDir.entryList(filters, QDir::Files, QDir::Name))
        {
            std::string in = xjw::common::io::toUtf8Path(inDir.absoluteFilePath(fname));
            QString outName = QFileInfo(fname).completeBaseName() + QString::fromStdString(suffix);
            std::string out = xjw::common::io::toUtf8Path(outDir.absoluteFilePath(outName));
            int rc = processOne(algo, extractor, in, out);
            (rc == cli::EXIT_OK) ? ++ok : ++fail;
        }
        fprintf(stdout, "批量: %d 成功 %d 失败\n", ok, fail);
        return fail > 0 ? cli::EXIT_ALGO_ERR : cli::EXIT_OK;
    }
    return processOne(algo, extractor, imgPath, outPath);
}
