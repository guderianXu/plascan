#pragma once

class ProjectData;

namespace xjw::app::project
{
    /// Install pre-open recovery and idempotent project/chunk activation bindings.
    class ProjectResourceRecoveryBinding final
    {
    public:
        static void install(ProjectData* projectData);
    };
} // namespace xjw::app::project
