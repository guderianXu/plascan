// =============================================================================
// 文件: cli_epipolar_rectify.cpp
// 功能: 极线校正 CLI (基于 CLI11)
// 用法:
//   rectify_cli -L imgL.tif -R imgR.tif --camL camL.txt --camR camR.txt -o prefix
// 输出: prefix_L.tif, prefix_R.tif, prefix.xml
// =============================================================================
#include "cli_common.h"
#include "EpipolarRectifier.h"
#include "Camera.h"
#include "PositiveDepthCameraModel.h"
#include "io/PathIO.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <QIODevice>
#include <QSaveFile>
#include <string>

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 极线校正工具 — 将立体影像对校正为行对齐"};

    std::string imgL, imgR, camL, camR, outPref;
    app.add_option("-L,--left",   imgL,    "左影像路径")->required();
    app.add_option("-R,--right",  imgR,    "右影像路径")->required();
    app.add_option("--camL",      camL,    "左相机文件路径")->required();
    app.add_option("--camR",      camR,    "右相机文件路径")->required();
    app.add_option("-o,--output", outPref, "输出前缀 (生成 _L.tif, _R.tif, .xml)")->required();

    bool verbose = false;
    app.add_flag("-V,--verbose", verbose, "详细诊断日志");

    CLI11_PARSE(app, argc, argv);

    // 加载相机
    xjw::Camera camLObj, camRObj;
    if (!camLObj.loadFromFile(camL))
        cli::fatal("无法加载左相机: " + camL, cli::EXIT_IO_ERR);
    if (!camRObj.loadFromFile(camR))
        cli::fatal("无法加载右相机: " + camR, cli::EXIT_IO_ERR);

    // 加载影像
    cv::Mat left  = xjw::common::io::readImage(imgL, cv::IMREAD_GRAYSCALE);
    cv::Mat right = xjw::common::io::readImage(imgR, cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty())
        cli::fatal("无法加载影像", cli::EXIT_IO_ERR);

    fprintf(stdout, "极线校正: %s <-> %s\n", imgL.c_str(), imgR.c_str());

    if (verbose)
        fprintf(stdout, "  尺寸: %dx%d\n", left.cols, left.rows);

    // 校正
    xjw::mvs::EpipolarRectifier::RectifiedPair result;
    std::string errMsg;
    bool ok = xjw::mvs::EpipolarRectifier::rectify(
        left, right,
        camLObj.toPositiveDepthModel(),
        camRObj.toPositiveDepthModel(),
        result, &errMsg);

    if (!ok)
        cli::fatal("极线校正失败: " + errMsg, cli::EXIT_ALGO_ERR);

    // 保存校正影像
    std::string rectL = outPref + "_L.tif";
    std::string rectR = outPref + "_R.tif";
    if (!xjw::common::io::writeImage(rectL, result.rectLeft) ||
        !xjw::common::io::writeImage(rectR, result.rectRight))
    {
        cli::fatal("无法写出校正影像: " + outPref, cli::EXIT_IO_ERR);
    }

    // 保存单应矩阵
    cv::FileStorage fs("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
    fs << "H1inv" << result.H1inv;
    fs << "H2inv" << result.H2inv;
    fs << "origW" << result.origW;
    fs << "origH" << result.origH;
    fs << "refIsRight" << static_cast<int>(result.refIsRight);
    fs << "transposed" << static_cast<int>(result.transposed);
    const std::string rectXml = fs.releaseAndGetString();
    QSaveFile xmlFile(xjw::common::io::fromUtf8Path(outPref + ".xml"));
    if (!xmlFile.open(QIODevice::WriteOnly) ||
        xmlFile.write(rectXml.data(), static_cast<qint64>(rectXml.size())) != static_cast<qint64>(rectXml.size()) ||
        !xmlFile.commit())
    {
        cli::fatal("无法写出校正参数: " + outPref + ".xml", cli::EXIT_IO_ERR);
    }

    fprintf(stdout, "校正完成:\n  %s\n  %s\n  %s.xml\n",
            rectL.c_str(), rectR.c_str(), outPref.c_str());
    return cli::EXIT_OK;
}
