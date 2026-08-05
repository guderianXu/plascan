#pragma once

#include <QImage>
#include <QSize>
#include <QString>

namespace xjw::gui::views
{

QImage loadImageForDisplay(const QString &path, const QString &plascanPath);
QImage loadImageForDisplay(const QString &path,
                           const QString &plascanPath,
                           const QSize &maximum_size,
                           QSize *source_size);

} // namespace xjw::gui::views
