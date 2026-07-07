#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

struct MatchValidityContext
{
    QStringList sparseSidecarPaths;     ///< 当前项目可用于判断最终轨迹有效性的 sparse sidecar。
};

struct MatchValidityResult
{
    bool hasTrackValidity = false;      ///< true 表示已从空三轨迹 sidecar 得到有效/无效分类
    int validCount = 0;                 ///< 进入最终 sparse track 的匹配数
    int invalidCount = 0;               ///< 未进入最终 sparse track 的匹配数
    QString sparseSidecarPath;          ///< 使用的 sfm_sparse_points.json 路径
    QVector<bool> inlierMask;           ///< 与当前显示匹配顺序一致的有效标记
};

// 为一批匹配文件预先发现空三 sparse sidecar，避免表格每行重复递归扫描 assets 目录。
MatchValidityContext buildMatchValidityContextForMatchDirectory(const QString &matchDirectory);

// 根据匹配文件和空三导出的轨迹观测，判断每条匹配是否进入最终连接点。
MatchValidityResult analyzeMatchTrackValidity(const QString &matchFile,
                                              const QString &displayImageA,
                                              const QString &displayImageB);

MatchValidityResult analyzeMatchTrackValidity(const QString &matchFile,
                                              const QString &displayImageA,
                                              const QString &displayImageB,
                                              const MatchValidityContext &context);
