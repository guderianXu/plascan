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
#include <functional>
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
 * 转换产物目录；当其非空时，只有 `overwrite=true` 才允许事务式替换。
 */
struct CameraConversionOptions
{
    /// 指定输入格式；Auto 表示根据文件名和目录结构自动识别。
    CameraFormat format = CameraFormat::Auto;
    /// 输入文件或数据集目录。
    std::filesystem::path inputPath;
    /// 输出根目录，不能是文件系统根，也不能等于或包含任何输入依赖。
    std::filesystem::path outputDir;
    /// 写入 summary 的数据集标识；为空时使用来源目录名。
    std::string datasetId;
    /// 是否允许通过同级暂存和备份替换已存在的非空输出目录。
    bool overwrite = false;
    /// COLMAP 输入是否在导入边界生成无畸变、全有效的 PNG 针孔栅格。
    bool preUndistortColmapImages = false;
    /// 可选的提交前同步回调；在暂存完成、源影像身份复检前调用。
    /// 主要用于调用方协调取消/测试，回调抛出的异常会使转换安全回滚。
    std::function<void()> beforeCommitHook;
};

/**
 * @brief 相机转换的完整结果和产物清单。
 *
 * `success=false` 时 `errorMessage` 给出失败原因；部分字段可能已经填充。
 * stage 安装前失败会保留原输出；安装成功后的旧备份清理问题作为 warning
 * 返回，不能用可能已部分清理的备份覆盖完整新结果。
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
    /// 有损参数转换及提交后备份清理等非致命问题；清理告警包含残留路径。
    std::vector<std::string> warnings;
    /// 本次成功写出的 Tsai 文件路径，便于 GUI/CLI 登记产物。
    std::vector<std::filesystem::path> writtenCameraFiles;
    /// 新输出成功安装但旧备份清理不完整时，给出可人工检查的残留路径。
    std::filesystem::path retainedBackupPath;
    /// 实际生成的预去畸变影像数量；未启用时为零。
    int preUndistortedImageCount = 0;
    /// 启用预去畸变时写出的原图/针孔图/valid mask 映射清单。
    std::filesystem::path preUndistortManifestPath;
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
 * @warning 输出切换使用同一父目录下的暂存和备份；无法安全比较路径时拒绝转换。
 */
CameraConversionResult convertCameraDataset(const CameraConversionOptions &options);

} // namespace xjw::camera
