#pragma once

#include <QString>
#include <QVector>

struct MatchValidityResult
{
    bool hasTrackValidity = false;      ///< true 表示已从空三轨迹 sidecar 得到有效/无效分类
    int validCount = 0;                 ///< 进入最终 sparse track 的匹配数
    int invalidCount = 0;               ///< 未进入最终 sparse track 的匹配数
    QString sparseSidecarPath;          ///< 使用的 sfm_sparse_points.json 路径
    QVector<bool> inlierMask;           ///< 与当前显示匹配顺序一致的有效标记
};

// 根据匹配文件和空三导出的轨迹观测，判断每条匹配是否进入最终连接点。
MatchValidityResult analyzeMatchTrackValidity(const QString &matchFile,
                                              const QString &displayImageA,
                                              const QString &displayImageB);
