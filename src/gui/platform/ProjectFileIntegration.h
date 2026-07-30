#pragma once

#include <QString>
#include <QStringList>

namespace xjw::gui::platform
{

struct FileAssociationResult
{
    bool success{true};
    bool changed{false};
    QString errorMessage;
};

QString startupProjectPath(const QStringList &arguments);
QString projectOpenCommand(const QString &executablePath);
FileAssociationResult ensureProjectFileAssociation(const QString &executablePath);

} // namespace xjw::gui::platform
