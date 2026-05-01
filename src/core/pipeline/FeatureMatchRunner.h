#pragma once

#include <QJsonObject>
#include <QStringList>
#include <atomic>

class ProjectManager;

// FeatureMatchRunner: 执行特征匹配（SuperGlue / LightGlue / 传统算法，隔离 LibTorch 依赖）
// 目的：避免 Qt 和 LibTorch 的宏冲突
class FeatureMatchRunner
{
public:
    static void run(const QJsonObject &config, const QStringList &imagePairs, ProjectManager *projectManager);

    /// 带取消标志的版本：外部可通过设置 cancelFlag = true 来中止匹配
    static void run(const QJsonObject &config, const QStringList &imagePairs,
                    ProjectManager *projectManager, std::atomic<bool> &cancelFlag);

    /// 带取消标志和实时进度计数的版本
    /// progressCount: 每完成一对就 +1（成功或失败），调用方可轮询用于更新 UI
    static void run(const QJsonObject &config, const QStringList &imagePairs,
                    ProjectManager *projectManager, std::atomic<bool> &cancelFlag,
                    std::atomic<int> &progressCount);
};
