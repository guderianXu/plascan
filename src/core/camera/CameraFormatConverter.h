#pragma once

// ============================================================
// 文件：CameraFormatConverter.h
// 功能：声明外部摄影测量相机格式到 PlaScan Tsai 数据集的转换接口。
//
// 转换结果由三部分组成：`cameras/*.tsai`、影像与相机配对文件
// `image_camera.lis`，以及记录来源、数量和有损转换警告的 `summary.json`。
// 接口不抛出预期转换错误，而是通过 CameraConversionResult 返回原因。
// ============================================================

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xjw::camera
{

/// 支持的输入格式；Auto 按输入文件/目录结构依次探测具体格式。
enum class CameraFormat
{
    Auto,
    MiddleburyPar,
    EpflCamera,
    ColmapText,
    MetashapeXml
};

/**
 * @brief 一次相机数据集转换的输入参数。
 *
 * `inputPath` 可以是格式主文件或包含该文件的目录。`outputDir` 是完整
 * 转换产物目录；当其非空时，只有 `overwrite=true` 才允许删除后重建。
 */
struct CameraConversionOptions
{
    /// 指定输入格式；Auto 表示根据文件名和目录结构自动识别。
    CameraFormat format = CameraFormat::Auto;
    /// 输入文件或数据集目录。
    std::filesystem::path inputPath;
    /// 输出根目录，不能与解析得到的输入相机目录等价。
    std::filesystem::path outputDir;
    /// 写入 summary 的数据集标识；为空时使用来源目录名。
    std::string datasetId;
    /// 是否允许删除并重建已存在的非空输出目录。
    bool overwrite = false;
};

/**
 * @brief 相机转换的完整结果和产物清单。
 *
 * `success=false` 时 `errorMessage` 给出失败原因；部分字段可能已经填充，
 * 也可能已有文件写入，因此调用方不应仅凭目录存在判断转换成功。
 */
struct CameraConversionResult
{
    /// 全部记录和 summary 写入成功后才置为 true。
    bool success = false;
    /// 捕获到的解析、路径或写入错误；成功时为空。
    std::string errorMessage;
    /// 自动识别或显式指定后真正使用的输入格式。
    CameraFormat inputFormat = CameraFormat::Auto;
    /// 调用方提供或由来源目录名推导出的最终数据集标识。
    std::string datasetId;
    /// 影像相对路径的基准目录。
    std::filesystem::path sourceDir;
    /// 调用方指定的输出根目录。
    std::filesystem::path outputDir;
    /// 写出的 `image_camera.lis` 路径。
    std::filesystem::path imageCameraList;
    /// 写出的 `summary.json` 路径。
    std::filesystem::path summaryPath;
    /// 成功解析并计划写出的相机记录数。
    int cameraCount = 0;
    /// 不能无损写入 PlaScan Tsai 的 skew、畸变等信息，按影像名聚合。
    std::vector<std::string> warnings;
    /// 本次成功写出的 Tsai 文件路径，便于 GUI/CLI 登记产物。
    std::vector<std::filesystem::path> writtenCameraFiles;
};

/// 返回 CLI 接受的规范格式名，顺序与帮助文本保持稳定。
std::vector<std::string> supportedFormatNames();
/// 返回枚举对应的规范 CLI 名称；未知枚举值返回 `unknown`。
std::string cameraFormatName(CameraFormat format);
/// 返回供 GUI/CLI 展示的中文格式说明。
std::string cameraFormatDescription(CameraFormat format);
/// 解析大小写不敏感且允许下划线别名的格式名；无法识别时返回 nullopt。
std::optional<CameraFormat> parseCameraFormat(const std::string &name);

/**
 * @brief 将一个外部相机数据集转换为 PlaScan Tsai 数据集。
 * @return 结构化结果；输入、解析和写入异常会被捕获到 `errorMessage`。
 * @warning 当 `overwrite=true` 且输出目录非空时，目录会在写入前被整体删除。
 */
CameraConversionResult convertCameraDataset(const CameraConversionOptions &options);

} // namespace xjw::camera
