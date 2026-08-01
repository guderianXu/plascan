#pragma once

#include <QDir>
#include <QFileInfo>
#include <QString>
#include <QtGlobal>

namespace xjw::gui::project
{

struct ProjectSessionContext
{
    QString projectPath;
    QString chunkId;
    quint64 generation = 0;

    bool matches(const ProjectSessionContext &other) const
    {
        return generation == other.generation
            && chunkId == other.chunkId
            && normalizedProjectPath(projectPath) == normalizedProjectPath(other.projectPath);
    }

private:
    static QString normalizedProjectPath(const QString &path)
    {
        if (path.trimmed().isEmpty())
        {
            return {};
        }

        const QString normalized = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
#ifdef Q_OS_WIN
        return normalized.toCaseFolded();
#else
        return normalized;
#endif
    }
};

} // namespace xjw::gui::project
