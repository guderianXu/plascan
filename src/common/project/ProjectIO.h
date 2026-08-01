// =============================================================================
// 文件名: ProjectIO.h
// 描述:   项目路径约定与产物寻址工具类（纯静态工具函数集合）。
//
//         所有与 .plascan 相关的目录结构都在此处集中定义，
//         保证路径规则只有一处，避免各模块硬编码路径字符串。
//
// 工程约定（以 /path/to/proj.plascan 为例）：
//   /path/to/proj.plascan             <- 轻量 XML 项目描述
//   /path/to/proj.files/project.zip   <- 项目与 Chunk 索引
//   /path/to/proj.files/1/            <- 当前 Chunk 数字目录
//   ├── chunk.zip                     <- Chunk 元数据归档
//   ├── assets/                       <- 按需创建：特征、匹配和导入资源
//   ├── bundle_adjust/                <- 按需创建：BA 运行产物
//   ├── reconstruction/               <- 按需创建：重建成果
//   ├── reports/                      <- 按需创建：综合报告
//   └── .plascan_tmp/                 <- 崩溃恢复数据
//       ├── project_files.json     <- 崩溃恢复缓存，不是归档条目
//       ├── project_results.json   <- qCompress 后的崩溃恢复缓存
//       └── project_config.json    <- 崩溃恢复缓存
// =============================================================================
#pragma once

#include <QMap>
#include <QString>
#include <QStringList>

namespace xjw::common::project
{

// ProjectIO: 项目路径与产物寻址工具（所有方法为静态方法）
class ProjectIO
{
public:
    // 注册当前进程中的项目运行根目录。新版工程使用当前 Chunk
    // 的 <project>.files/<number> 数字目录。
    static void registerRuntimeRoot(const QString &plascanPath,
                                    const QString &runtimeRoot);
    static void unregisterRuntimeRoot(const QString &plascanPath);
    static QString registeredRuntimeRoot(const QString &plascanPath);

    // 已注册或有效工程返回当前 Chunk 数字目录；尚未创建的路径返回物理父目录。
    static QString projectRootFromPlascan(const QString &plascanPath);

    // 始终返回 .plascan 文件所在的物理目录，不受运行根目录注册影响。
    static QString physicalProjectRoot(const QString &plascanPath);

    // 将项目元数据中的相对资源路径解析为以 .plascan 所在目录为基准的绝对路径。
    static QString resolveProjectResourcePath(const QString &plascanPath,
                                              const QString &resourcePath);

    // 返回 assets/ 目录的绝对路径
    static QString projectAssetsDir(const QString &plascanPath);

    // 返回当前 Chunk 的 bundle_adjust/ 目录。
    static QString projectBundleAdjustDir(const QString &plascanPath);

    // 返回 .files/shared/images/ 工程级内容寻址影像库。
    static QString projectImagesDir(const QString &plascanPath);

    // 返回 assets/control_points/ 目录及标记 sidecar 的标准路径
    static QString projectControlPointsDir(const QString &plascanPath);
    static QString markerSetPath(const QString &plascanPath);
    static QString markerDetectionReviewPath(const QString &plascanPath);

    // 返回 .plascan_tmp/ 临时目录的绝对路径
    static QString tmpDir(const QString &plascanPath);

    // 返回崩溃恢复目录中 project_files.json 的完整路径
    static QString tempFilesPath(const QString &plascanPath);

    // 返回崩溃恢复目录中 project_results.json 的完整路径
    static QString tempResultsPath(const QString &plascanPath);

    // 返回崩溃恢复目录中 project_config.json 的完整路径
    static QString tempConfigPath(const QString &plascanPath);

    // 返回崩溃恢复目录中 project_ui_state.json 的完整路径
    static QString tempUiStatePath(const QString &plascanPath);

    // 返回 assets/image_matches/ 目录。目录内每幅影像只有一个 `.pimatch` 分片，
    // 文件名与格式由 core/image_matching/ImageMatchFile 统一管理。
    static QString imageMatchOutputDir(const QString &plascanPath);

    // 返回 assets/masks/ 目录（照片蒙版标准输出目录）
    static QString maskOutputDir(const QString &plascanPath);

    // 根据规范化影像路径生成稳定产物键，避免不同目录下的同名影像互相覆盖。
    static QString imageArtifactKey(const QString &imagePath);

    // 根据影像路径推算对应蒙版文件输出路径
    // 输出格式：assets/masks/<basename>-<sha256>_mask.png
    static QString maskOutputPathForImage(const QString &plascanPath, const QString &imagePath);

    // 查找指定影像对应的项目蒙版文件。
    static QString findMaskForImage(const QString &plascanPath, const QString &imagePath);
    // 批量生成影像到蒙版的映射；仅返回真实存在的蒙版，供连接点任务按需过滤。
    static QMap<QString, QString> maskPathsForImages(const QString &plascanPath,
                                                     const QStringList &imagePaths);

};

} // namespace xjw::common::project
