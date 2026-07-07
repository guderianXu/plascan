#pragma once

#include <QString>
#include <QStringList>

namespace xjw::common::model
{

struct TorchScriptModelSearchOptions
{
    QString sourceRoot;
    QString applicationDir;
    QString environmentVariable = QStringLiteral("PLASCAN_MODEL_DIR");
    QStringList extraSearchDirs;
};

class TorchScriptModelResolver
{
public:
    explicit TorchScriptModelResolver(TorchScriptModelSearchOptions options = {});

    QString findModel(const QString &modelName) const;
    QString findFirstModel(const QStringList &modelNames, QString *pickedModelName = nullptr) const;
    QStringList candidatePaths(const QString &modelName) const;
    QString defaultModelDir() const;

private:
    TorchScriptModelSearchOptions _options;
};

} // namespace xjw::common::model
