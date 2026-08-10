#pragma once

#include <QString>
#include <QStringList>

namespace xjw::common::model
{

// 描述模型资源的统一搜索位置。解析器只负责文件发现，不绑定具体推理框架。
struct ModelFileSearchOptions
{
    QString sourceRoot;
    QString applicationDir;
    QString userModelDir;
    QString environmentVariable = QStringLiteral("PLASCAN_MODEL_DIR");
    QStringList extraSearchDirs;
};

enum class ModelInstallLocationKind
{
    EnvironmentOverride,
    SourceTree,
    UserData
};

struct ModelInstallLocation
{
    QString directory;
    QString label;
    ModelInstallLocationKind kind = ModelInstallLocationKind::UserData;
};

class ModelFileResolver
{
public:
    explicit ModelFileResolver(ModelFileSearchOptions options = {});

    QString findModel(const QString &modelName) const;
    QString findFirstModel(const QStringList &modelNames, QString *pickedModelName = nullptr) const;
    /// 返回按优先级排列的模型根目录，供需要扫描 manifest 的模块复用。
    QStringList searchDirectories() const;
    QStringList candidatePaths(const QString &modelName) const;
    /// 返回当前运行形态下可写的模型根目录，不会指向受保护的安装目录。
    QString defaultModelDir() const;
    ModelInstallLocation installLocation() const;

private:
    bool isSourceTreeRuntime() const;

    ModelFileSearchOptions _options;
};

} // namespace xjw::common::model
