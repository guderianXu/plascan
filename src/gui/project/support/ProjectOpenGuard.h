#pragma once

#include <QString>

class ProjectData;
class QWidget;

namespace xjw::gui::project
{

// Performs the common project-session precondition check while leaving the
// caller in control of the user-facing message and title.
bool requireOpenProject(const ProjectData *projectData,
                        QWidget *parentWidget,
                        const QString &message = QStringLiteral("请先打开项目"),
                        const QString &title = QStringLiteral("提示"));

} // namespace xjw::gui::project
