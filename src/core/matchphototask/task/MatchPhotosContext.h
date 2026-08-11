#pragma once

#include "Camera.h"
#include "PairSelector.h"

#include <QMap>
#include <QString>
#include <atomic>
#include <functional>
#include <memory>

namespace xjw
{
namespace matchphotos
{

class MatchPhotosFeatureCache;

// 运行上下文与 options 分离，便于同一个配置好的任务复用于不同项目或测试数据。
struct MatchPhotosContext
{
    QString projectPath;
    QString workingDirectory;
    QString matchDirectory;
    PairSelectionInput pairInput;
    QMap<QString, xjw::Camera> referenceCameras;
    QString referenceSparsePointsPath;
    // 影像路径到蒙版路径的映射。键可以是绝对路径、文件名或 baseName，运行时会做宽松匹配。
    QMap<QString, QString> maskPaths;
    // 特征只在本次任务内存在。调用方通常无需设置，MatchPhotosTask 会创建并在
    // 所有阶段间共享；测试和批处理可以注入预填充缓存以实现确定性验证。
    std::shared_ptr<MatchPhotosFeatureCache> featureCache;
    std::atomic_bool *cancelFlag = nullptr;
    std::atomic_int *progressCount = nullptr;
    // 后台任务细粒度进度回调。参数依次为阶段 id、面向 UI 的阶段文本、当前计数、总计数；
    // 总计数为 0 时表示该阶段没有可信百分比，应显示忙碌状态。
    std::function<void(const QString &, const QString &, int, int)> progressCallback;
};

} // namespace matchphotos
} // namespace xjw
