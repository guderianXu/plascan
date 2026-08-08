#include "ProjectOpenGuard.h"

#include "ProjectData.h"

#include <QMessageBox>

namespace xjw::gui::project
{

bool requireOpenProject(const ProjectData *projectData,
                        QWidget *parentWidget,
                        const QString &message,
                        const QString &title)
{
    if (projectData && projectData->hasProject())
    {
        return true;
    }

    QMessageBox::warning(parentWidget, title, message);
    return false;
}

} // namespace xjw::gui::project
