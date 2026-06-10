#pragma once

#include <QList>
#include <QString>
#include <QUrl>

namespace xjw
{
namespace gui
{
namespace main_window
{

bool isStandaloneModelFile(const QString &path);
QString firstStandaloneModelFile(const QList<QUrl> &urls);

} // namespace main_window
} // namespace gui
} // namespace xjw
