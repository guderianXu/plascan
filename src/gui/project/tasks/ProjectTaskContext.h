#pragma once

#include "project/support/ProjectSessionContext.h"

#include <QString>

#include <atomic>
#include <memory>

namespace xjw::gui::project
{

struct ProjectTaskContext
{
    QString taskId;
    ProjectSessionContext session;
    std::shared_ptr<std::atomic<bool>> cancelFlag;
};

} // namespace xjw::gui::project
