// =============================================================================
// 文件名: ProjectIO.h
// 描述:   项目路径约定与产物寻址工具类（纯静态工具函数集合）。
//
//         所有与 .plascan 相关的目录结构都在此处集中定义，
//         保证路径规则只有一处，避免各模块硬编码路径字符串。
//
// 目录结构约定（以 /path/to/proj.plascan 为例）：
//   /path/to/
//   ├── proj.plascan         <- ZIP 归档，含 project_files.json / project_results.json / project_config.json
//   ├── assets/
//   │   ├── images/          <- 预留给打包/复制进项目的影像，默认添加影像仅保存外部引用
//   │   ├── ip/              <- ipfind 输出的特征点文件 (.vwip/.sp)
//   │   └── matches/         <- ipmatch 输出的匹配文件
//   └── .plascan_tmp/        <- 临时缓存目录（崩溃恢复用）
//       ├── project_files.json
//       ├── project_results.json  <- qCompress 后的 Compact JSON
//       └── project_config.json
// =============================================================================
#pragma once

#include <QString>
#include <QStringList>

// ProjectIO: 项目路径与产物寻址工具（所有方法为静态方法）
class ProjectIO
{
public:
    // 从 .plascan 文件路径推导出项目根目录（即 .plascan 所在目录）
    static QString projectRootFromPlascan(const QString &plascanPath);

    // 返回 assets/ 目录的绝对路径
    static QString projectAssetsDir(const QString &plascanPath);

    // 返回 assets/images/ 目录的绝对路径（预留给打包/复制进项目的影像）
    static QString projectImagesDir(const QString &plascanPath);

    // 返回 .plascan_tmp/ 临时目录的绝对路径
    static QString tmpDir(const QString &plascanPath);

    // 返回临时目录中 project_files.json 的完整路径
    static QString tempFilesPath(const QString &plascanPath);

    // 返回临时目录中 project_results.json 的完整路径
    static QString tempResultsPath(const QString &plascanPath);

    // 返回临时目录中 project_config.json 的完整路径
    static QString tempConfigPath(const QString &plascanPath);

    // 返回 assets/ip/ 目录（ipfind 标准输出目录）
    static QString ipfindOutputDir(const QString &plascanPath);

    // 返回 assets/matches/ 目录（ipmatch 标准输出目录）
    static QString ipmatchOutputDir(const QString &plascanPath);

    // 返回 .plascan_tmp/ip/ 目录（ipfind 临时输出目录，用于异步处理）
    static QString tmpIpfindDir(const QString &plascanPath);

    // 根据影像路径推算对应的 vwip 特征点文件输出路径
    // 输出格式：assets/ip/<basename>.sp
    static QString vwipOutputPathForImage(const QString &plascanPath, const QString &imagePath);

    // 根据算法后缀推算特征文件输出路径
    // 输出格式：assets/ip/<basename><suffix>
    static QString featureOutputPathForImage(const QString &plascanPath,
                                              const QString &imagePath,
                                              const QString &suffix);

    // 在多个候选目录中查找指定影像对应的 .sp 特征点文件
    // 返回第一个存在的文件路径；若均不存在则返回空字符串
    static QString findSpForImage(const QString &plascanPath, const QString &imagePath);
    // 查找任意提取器的特征文件 (尝试 .sp/.dsk/.alk/.sift/.orb/.akz)
    static QString findFeatureForImage(const QString &plascanPath, const QString &imagePath);
    // 按固定后缀查找特征文件 (不扫描所有后缀)
    static QString featureFileForSuffix(const QString &plascanPath, const QString &imagePath,
                                        const QString &suffix);
    // 列出影像所有存在的特征文件后缀 (用于 GUI 多提取器切换)
    static QStringList availableFeatureSuffixes(const QString &plascanPath,
                                                 const QString &imagePath);

private:
    // 生成 .sp 文件的候选路径列表（优先检查临时目录，其次标准目录，再次影像同目录）
    static QStringList spCandidates(const QString &plascanPath, const QString &imagePath);

    // 生成 vwip 文件的候选路径列表（保留供将来支持 .vwip 格式）
    static QStringList vwipCandidates(const QString &plascanPath, const QString &imagePath);
};
