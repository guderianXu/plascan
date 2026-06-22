#pragma once

#include <QJsonObject>
#include <QPointer>
#include <QStringList>
#include <atomic>

class ProjectManager;

// FeatureExtractionRunner: 执行通用特征提取（隔离 LibTorch 依赖）
// 目的：避免 Qt 和 LibTorch 的宏冲突
class FeatureExtractionRunner
{
public:
    /// 返回 true 表示至少有一张图像处理成功
    static bool run(const QJsonObject &config, const QStringList &inputs, ProjectManager *projectManager);
    static bool run(const QJsonObject &config, const QStringList &inputs,
                    QPointer<ProjectManager> projectManager);

    /// 带取消标志和实时进度计数的版本；返回 true 表示至少一张成功
    static bool run(const QJsonObject &config, const QStringList &inputs,
                    ProjectManager *projectManager, std::atomic<bool> &cancelFlag,
                    std::atomic<int> &progressCount);
    static bool run(const QJsonObject &config, const QStringList &inputs,
                    QPointer<ProjectManager> projectManager, std::atomic<bool> &cancelFlag,
                    std::atomic<int> &progressCount);
};
