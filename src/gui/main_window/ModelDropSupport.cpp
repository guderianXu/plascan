#include "ModelDropSupport.h"

#include <QFileInfo>
#include <QSet>

namespace xjw
{
namespace gui
{
namespace main_window
{

bool isStandaloneModelFile(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    static const QSet<QString> kSupportedSuffixes = 
    {
        QStringLiteral("ply"),
        QStringLiteral("obj"),
        QStringLiteral("xyz"),
        QStringLiteral("txt"),
    };
    return kSupportedSuffixes.contains(suffix);
}

QString firstStandaloneModelFile(const QList<QUrl> &urls)
{
    for (const QUrl &url : urls)
    {
        if (!url.isLocalFile())
        {
            continue;
        }

        const QString path = url.toLocalFile();
        if (isStandaloneModelFile(path))
        {
            return path;
        }
    }

    return QString();
}

} // namespace main_window
} // namespace gui
} // namespace xjw
