#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::gui::project
{

struct RunDiagnosticsSpec
{
    QString label;
    QString diagnosticsType;
    QString runIdKey;
    QString runDirectoryKey;
    QString diagnosticsPathKey;
    QStringList containedArtifactKeys;
    QStringList matchingPathKeys;
};

bool isPhysicalDirectory(const QString &path);
bool sameExistingPath(const QString &first, const QString &second);
bool pathBelongsToDirectory(const QString &path, const QString &directory);
bool validateRunDiagnostics(const QJsonObject &record,
                            const RunDiagnosticsSpec &spec,
                            QString *errorMessage = nullptr);

} // namespace xjw::gui::project
