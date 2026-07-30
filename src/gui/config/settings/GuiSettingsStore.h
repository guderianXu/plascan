#pragma once

#include <QSettings>

namespace xjw::gui::settings
{

inline QSettings createGuiSettings()
{
    return QSettings(QStringLiteral("PlaScan"), QStringLiteral("plascan_gui"));
}

} // namespace xjw::gui::settings
