#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

struct MatchValidityContext
{
    QStringList tiePointSidecarPaths;   ///< 连接点轨迹，特征索引与匹配缓存保持同一坐标系。
    QStringList sparseSidecarPaths;     ///< 兼容旧工程时用于判断最终稀疏点的 sparse sidecar。
};

struct MatchValidityResult
{
    bool hasTrackValidity = false;      ///< true 表示已从空三轨迹 sidecar 得到有效/无效分类
    int validCount = 0;                 ///< 进入有效连接点轨迹的匹配数
    int invalidCount = 0;               ///< 未进入有效连接点轨迹的匹配数
    QString sparseSidecarPath;          ///< 实际使用的轨迹 sidecar 路径，保留字段名以兼容既有调用。
    QVector<bool> inlierMask;           ///< 与当前显示匹配顺序一致的有效标记
};

// 为一批匹配文件预先发现连接点与空三 sidecar，避免表格每行重复递归扫描 assets 目录。
MatchValidityContext buildMatchValidityContextForMatchDirectory(const QString &matchDirectory);

// 根据匹配文件和连接点轨迹观测，判断每条匹配是否进入有效连接点。
MatchValidityResult analyzeMatchTrackValidity(const QString &matchFile,
                                              const QString &displayImageA,
                                              const QString &displayImageB);

MatchValidityResult analyzeMatchTrackValidity(const QString &matchFile,
                                              const QString &displayImageA,
                                              const QString &displayImageB,
                                              const MatchValidityContext &context);
