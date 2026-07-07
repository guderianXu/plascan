#pragma once

#include "Camera.h"
#include "PairSelector.h"

#include <QMap>
#include <QString>
#include <atomic>
#include <functional>

namespace xjw
{
namespace matchphotos
{

// 运行上下文与 options 分离，便于同一个配置好的任务复用于不同项目或测试数据。
struct MatchPhotosContext
{
    QString projectPath;
    QString workingDirectory;
    QString featureDirectory;
    QString matchDirectory;
    PairSelectionInput pairInput;
    QMap<QString, xjw::Camera> referenceCameras;
    // 影像路径到蒙版路径的映射。键可以是绝对路径、文件名或 baseName，运行时会做宽松匹配。
    QMap<QString, QString> maskPaths;
    std::atomic_bool *cancelFlag = nullptr;
    std::atomic_int *progressCount = nullptr;
    // 后台任务细粒度进度回调。参数依次为阶段 id、面向 UI 的阶段文本、当前计数、总计数。
    std::function<void(const QString &, const QString &, int, int)> progressCallback;
};

} // namespace matchphotos
} // namespace xjw
