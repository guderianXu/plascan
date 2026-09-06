#pragma once

#include "TaskTypes.h"

#include <filesystem>
#include <string>
#include <vector>

namespace xjw::task_runtime
{

    struct TaskJournalLoadResult
    {
        bool succeeded = false;
        std::string error;
        std::vector<TaskRunSnapshot> snapshots;
    };

    class TaskJournal final
    {
    public:
        static constexpr std::uint32_t kSchemaVersion = 1;

        static bool save(const std::filesystem::path& path,
                         const std::vector<TaskRunSnapshot>& snapshots,
                         std::string* error = nullptr);
        static TaskJournalLoadResult load(const std::filesystem::path& path);
    };

} // namespace xjw::task_runtime
