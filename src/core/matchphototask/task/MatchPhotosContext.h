#pragma once

#include "PairSelector.h"

#include <QString>
#include <atomic>

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
    std::atomic_bool *cancelFlag = nullptr;
    std::atomic_int *progressCount = nullptr;
};

} // namespace matchphotos
} // namespace xjw
