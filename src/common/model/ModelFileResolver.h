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
    QString environmentVariable = QStringLiteral("PLASCAN_MODEL_DIR");
    QStringList extraSearchDirs;
};

class ModelFileResolver
{
public:
    explicit ModelFileResolver(ModelFileSearchOptions options = {});

    QString findModel(const QString &modelName) const;
    QString findFirstModel(const QStringList &modelNames, QString *pickedModelName = nullptr) const;
    QStringList candidatePaths(const QString &modelName) const;
    QString defaultModelDir() const;

private:
    ModelFileSearchOptions _options;
};

} // namespace xjw::common::model
