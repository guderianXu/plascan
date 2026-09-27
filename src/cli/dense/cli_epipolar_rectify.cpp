// =============================================================================
// 文件: cli_epipolar_rectify.cpp
// 功能: 极线校正 CLI (基于 CLI11)
// 用法:
//   rectify_cli -L imgL.tif -R imgR.tif --camL camL.txt --camR camR.txt -o prefix
// 输出: prefix_L.tif, prefix_R.tif, prefix.xml
// =============================================================================
#include "cli_common.h"
#include "EpipolarRectifier.h"
#include <placamera/tsai.h>
#include "io/PathIO.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <QIODevice>
#include <QSaveFile>
#include <optional>
#include <string>

namespace
{

    std::optional<placamera::FramePinholeModel> prepareRectificationImage(const cv::Mat& source,
                                                                          const placamera::FramePinholeModel& camera,
                                                                          cv::Mat* prepared,
                                                                          std::string* error)
    {
        try
        {
            const auto normalized = camera.normalizedForPositiveDepth(
                placamera::CameraDefinitionId(camera.definitionId().value() + "-positive-depth"),
                placamera::CameraInstanceId(camera.instanceId().value() + "-positive-depth"));
            const auto& intrinsics = normalized.pinholeDefinition().intrinsics();
            const auto& distortion = normalized.pinholeDefinition().distortion();
            if (intrinsics.uAxisSign != 1 || intrinsics.vAxisSign != 1)
            {
                if (error)
                {
                    *error = "极线校正不支持该相机规范化后的像素轴方向";
                }
                return std::nullopt;
            }
            cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            camera_matrix.at<double>(0, 0) = intrinsics.focalX;
            camera_matrix.at<double>(1, 1) = intrinsics.focalY;
            camera_matrix.at<double>(0, 2) = intrinsics.principalX;
            camera_matrix.at<double>(1, 2) = intrinsics.principalY;
            cv::Mat distortion_coefficients(1, 5, CV_64F);
            distortion_coefficients.at<double>(0, 0) = distortion.radialK1;
            distortion_coefficients.at<double>(0, 1) = distortion.radialK2;
            distortion_coefficients.at<double>(0, 2) = distortion.tangentialP1;
            distortion_coefficients.at<double>(0, 3) = distortion.tangentialP2;
            distortion_coefficients.at<double>(0, 4) = distortion.radialK3;
            cv::Mat map_x;
            cv::Mat map_y;
            cv::initUndistortRectifyMap(camera_matrix,
                                        distortion_coefficients,
                                        cv::Mat(),
                                        camera_matrix,
                                        source.size(),
                                        CV_32FC1,
                                        map_x,
                                        map_y);
            cv::remap(source, *prepared, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

            const auto definition = placamera::FramePinholeDefinition::create(
                placamera::CameraDefinitionId(normalized.definitionId().value() + "-undistorted"),
                intrinsics,
                placamera::BrownConradyDistortion{},
                normalized.pinholeDefinition().pixelConvention(),
                normalized.groundFrame());
            return placamera::FramePinholeModel::create(
                placamera::CameraInstanceId(normalized.instanceId().value() + "-undistorted"),
                placamera::ImageId(normalized.imageId().value() + "-undistorted"),
                definition,
                normalized.imageSize(),
                normalized.pose(),
                normalized.captureTime());
        }
        catch (const std::exception& exception)
        {
            if (error)
            {
                *error = exception.what();
            }
            return std::nullopt;
        }
    }

} // namespace

int main(int argc, char* argv[])
{
    CLI::App app{"PlaScan 极线校正工具 — 将立体影像对校正为行对齐"};
    cli::configureApp(app);

    std::string imgL, imgR, camL, camR, outPref;
    app.add_option("-L,--left", imgL, "左影像路径")->required();
    app.add_option("-R,--right", imgR, "右影像路径")->required();
    app.add_option("--camL", camL, "左相机文件路径")->required();
    app.add_option("--camR", camR, "右相机文件路径")->required();
    std::string groundFrame = "scene-local";
    app.add_option("--ground-frame", groundFrame, "两台相机共享的局部地面坐标系 ID");
    app.add_option("-o,--output", outPref, "输出前缀 (生成 _L.tif, _R.tif, .xml)")->required();

    bool verbose = false;
    app.add_flag("-V,--verbose", verbose, "详细诊断日志");

    CLI11_PARSE(app, argc, argv);
    if (groundFrame.find_first_not_of(" \t\r\n") == std::string::npos)
    {
        cli::fatal("地面坐标系 ID 不能为空", cli::EXIT_ARG_ERR);
    }

    // 加载影像
    cv::Mat left = xjw::common::io::readImage(imgL, cv::IMREAD_GRAYSCALE);
    cv::Mat right = xjw::common::io::readImage(imgR, cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty())
    {
        cli::fatal("无法加载影像", cli::EXIT_IO_ERR);
    }
    if (left.size() != right.size())
    {
        cli::fatal("左右影像尺寸不一致", cli::EXIT_ARG_ERR);
    }

    const placamera::FrameId ground_frame(groundFrame);
    const auto left_camera =
        placamera::loadTsaiFramePinhole(camL, placamera::CameraDefinitionId("rectify-left"), ground_frame);
    if (!left_camera)
    {
        cli::fatal("无法加载左相机: " + camL + " (" + left_camera.message() + ")", cli::EXIT_IO_ERR);
    }
    const auto right_camera =
        placamera::loadTsaiFramePinhole(camR, placamera::CameraDefinitionId("rectify-right"), ground_frame);
    if (!right_camera)
    {
        cli::fatal("无法加载右相机: " + camR + " (" + right_camera.message() + ")", cli::EXIT_IO_ERR);
    }
    const placamera::ImageSize image_size{left.cols, left.rows};
    const auto left_model = placamera::FramePinholeModel::create(placamera::CameraInstanceId("rectify-left"),
                                                                 placamera::ImageId("rectify-left-image"),
                                                                 left_camera.value().definition,
                                                                 image_size,
                                                                 left_camera.value().pose);
    const auto right_model = placamera::FramePinholeModel::create(placamera::CameraInstanceId("rectify-right"),
                                                                  placamera::ImageId("rectify-right-image"),
                                                                  right_camera.value().definition,
                                                                  image_size,
                                                                  right_camera.value().pose);

    fprintf(stdout, "极线校正: %s <-> %s\n", imgL.c_str(), imgR.c_str());

    if (verbose)
        fprintf(stdout, "  尺寸: %dx%d\n", left.cols, left.rows);

    cv::Mat preparedLeft;
    cv::Mat preparedRight;
    std::string preprocessError;
    const auto preparedLeftCamera = prepareRectificationImage(left, left_model, &preparedLeft, &preprocessError);
    if (!preparedLeftCamera)
    {
        cli::fatal("左影像去畸变失败: " + preprocessError, cli::EXIT_ALGO_ERR);
    }
    const auto preparedRightCamera = prepareRectificationImage(right, right_model, &preparedRight, &preprocessError);
    if (!preparedRightCamera)
    {
        cli::fatal("右影像去畸变失败: " + preprocessError, cli::EXIT_ALGO_ERR);
    }

    // 极线校正只接收已经去畸变、正深度归一化的影像与相机。
    xjw::mvs::EpipolarRectifier::RectifiedPair result;
    std::string errMsg;
    bool ok = xjw::mvs::EpipolarRectifier::rectify(
        preparedLeft, preparedRight, *preparedLeftCamera, *preparedRightCamera, result, &errMsg);

    if (!ok)
        cli::fatal("极线校正失败: " + errMsg, cli::EXIT_ALGO_ERR);

    // 保存校正影像
    std::string rectL = outPref + "_L.tif";
    std::string rectR = outPref + "_R.tif";
    if (!xjw::common::io::writeImage(rectL, result.rectLeft) || !xjw::common::io::writeImage(rectR, result.rectRight))
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

    fprintf(stdout, "校正完成:\n  %s\n  %s\n  %s.xml\n", rectL.c_str(), rectR.c_str(), outPref.c_str());
    return cli::EXIT_OK;
}
