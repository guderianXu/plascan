// =============================================================================
// 文件: cli_camera_convert.cpp
// 功能: 通用相机格式转换 CLI -> PlaScan/ASP .tsai + image_camera.lis
// =============================================================================
#include "cli_common.h"

#include "CameraFormatConverter.h"

#include <cstdio>
#include <filesystem>
#include <string>

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 相机格式转换工具 — 输出 tsai 和 image_camera.lis"};

    std::string formatName = "auto";
    std::string inputPath;
    std::string outputDir;
    std::string datasetId;
    bool overwrite = false;
    bool listFormats = false;

    app.add_flag("--list-formats", listFormats, "列出支持的输入相机格式");
    app.add_option("-f,--format", formatName, "输入格式: auto, middlebury-par, epfl-camera");
    app.add_option("-i,--input", inputPath, "输入相机文件或目录");
    app.add_option("-o,--output-dir", outputDir, "输出目录，将写入 image_camera.lis 和 cameras/*.tsai");
    app.add_option("--dataset-id", datasetId, "写入 summary.json 的数据集标识");
    app.add_flag("--overwrite", overwrite, "覆盖非空输出目录");

    CLI11_PARSE(app, argc, argv);

    if (listFormats)
    {
        std::fprintf(stdout, "支持的相机格式:\n");
        for (const std::string &name : xjw::camera::supportedFormatNames())
        {
            const auto parsed = xjw::camera::parseCameraFormat(name);
            const std::string description = parsed
                ? xjw::camera::cameraFormatDescription(*parsed)
                : std::string();
            std::fprintf(stdout, "  %-16s %s\n", name.c_str(), description.c_str());
        }
        return cli::EXIT_OK;
    }

    if (inputPath.empty())
    {
        cli::fatal("缺少 --input", cli::EXIT_ARG_ERR);
    }
    if (outputDir.empty())
    {
        cli::fatal("缺少 --output-dir", cli::EXIT_ARG_ERR);
    }

    const auto parsedFormat = xjw::camera::parseCameraFormat(formatName);
    if (!parsedFormat)
    {
        cli::fatal("不支持的相机格式: " + formatName, cli::EXIT_ARG_ERR);
    }

    xjw::camera::CameraConversionOptions options;
    options.format = *parsedFormat;
    options.inputPath = std::filesystem::path(inputPath);
    options.outputDir = std::filesystem::path(outputDir);
    options.datasetId = datasetId;
    options.overwrite = overwrite;

    const auto result = xjw::camera::convertCameraDataset(options);
    if (!result.success)
    {
        cli::fatal("相机转换失败: " + result.errorMessage, cli::EXIT_ALGO_ERR);
    }

    std::fprintf(stdout, "相机转换完成: %d 个相机\n", result.cameraCount);
    std::fprintf(stdout, "输入格式: %s\n", xjw::camera::cameraFormatName(result.inputFormat).c_str());
    std::fprintf(stdout, "image_camera.lis: %s\n", result.imageCameraList.string().c_str());
    std::fprintf(stdout, "summary.json: %s\n", result.summaryPath.string().c_str());
    for (const std::string &warning : result.warnings)
    {
        std::fprintf(stderr, "警告: %s\n", warning.c_str());
    }
    return cli::EXIT_OK;
}
