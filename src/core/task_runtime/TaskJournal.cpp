#include "TaskJournal.h"

#include <array>
#include <charconv>
#include <chrono>
#include <fstream>
#include <sstream>
#include <system_error>

namespace xjw::task_runtime
{
    namespace
    {

        constexpr std::string_view kHeader = "PLASCAN_TASK_JOURNAL\t2";

        std::string escapeField(std::string_view value)
        {
            constexpr char digits[] = "0123456789ABCDEF";
            std::string escaped;
            escaped.reserve(value.size());
            for (const unsigned char character : value)
            {
                if (character == '%' || character == '\t' || character == '\n' || character == '\r' || character == ',')
                {
                    escaped.push_back('%');
                    escaped.push_back(digits[character >> 4]);
                    escaped.push_back(digits[character & 0x0F]);
                }
                else
                {
                    escaped.push_back(static_cast<char>(character));
                }
            }
            return escaped;
        }

        int hexValue(char character)
        {
            if (character >= '0' && character <= '9')
            {
                return character - '0';
            }
            if (character >= 'A' && character <= 'F')
            {
                return character - 'A' + 10;
            }
            if (character >= 'a' && character <= 'f')
            {
                return character - 'a' + 10;
            }
            return -1;
        }

        bool unescapeField(std::string_view value, std::string* decoded)
        {
            decoded->clear();
            decoded->reserve(value.size());
            for (std::size_t index = 0; index < value.size(); ++index)
            {
                if (value[index] != '%')
                {
                    decoded->push_back(value[index]);
                    continue;
                }
                if (index + 2 >= value.size())
                {
                    return false;
                }
                const int high = hexValue(value[index + 1]);
                const int low = hexValue(value[index + 2]);
                if (high < 0 || low < 0)
                {
                    return false;
                }
                decoded->push_back(static_cast<char>((high << 4) | low));
                index += 2;
            }
            return true;
        }

        std::vector<std::string_view> split(std::string_view value, char delimiter)
        {
            std::vector<std::string_view> result;
            std::size_t start = 0;
            while (start <= value.size())
            {
                const std::size_t end = value.find(delimiter, start);
                result.push_back(
                    value.substr(start, end == std::string_view::npos ? value.size() - start : end - start));
                if (end == std::string_view::npos)
                {
                    break;
                }
                start = end + 1;
            }
            return result;
        }

        template <typename Integer> bool parseInteger(std::string_view value, Integer* parsed)
        {
            const char* begin = value.data();
            const char* end = value.data() + value.size();
            const auto result = std::from_chars(begin, end, *parsed);
            return result.ec == std::errc() && result.ptr == end;
        }

        std::string joinDependencies(const std::vector<TaskId>& dependencies)
        {
            std::string joined;
            for (std::size_t index = 0; index < dependencies.size(); ++index)
            {
                if (index > 0)
                {
                    joined.push_back(',');
                }
                joined += escapeField(dependencies[index]);
            }
            return joined;
        }

        bool parseDependencies(std::string_view value, std::vector<TaskId>* dependencies)
        {
            dependencies->clear();
            if (value.empty())
            {
                return true;
            }
            for (const std::string_view encoded : split(value, ','))
            {
                std::string dependency;
                if (!unescapeField(encoded, &dependency))
                {
                    return false;
                }
                dependencies->push_back(std::move(dependency));
            }
            return true;
        }

        std::int64_t toMilliseconds(const std::optional<std::chrono::system_clock::time_point>& value)
        {
            if (!value)
            {
                return -1;
            }
            return std::chrono::duration_cast<std::chrono::milliseconds>(value->time_since_epoch()).count();
        }

        std::optional<std::chrono::system_clock::time_point> fromMilliseconds(std::int64_t value)
        {
            if (value < 0)
            {
                return std::nullopt;
            }
            return std::chrono::system_clock::time_point(std::chrono::milliseconds(value));
        }

        void appendField(std::ostream& stream, std::string_view value)
        {
            stream << '\t' << escapeField(value);
        }

        bool decodeString(const std::vector<std::string_view>& fields, std::size_t index, std::string* value)
        {
            return index < fields.size() && unescapeField(fields[index], value);
        }

    } // namespace

    bool TaskJournal::save(const std::filesystem::path& path,
                           const std::vector<TaskRunSnapshot>& snapshots,
                           std::string* error)
    {
        std::error_code filesystem_error;
        if (!path.parent_path().empty())
        {
            std::filesystem::create_directories(path.parent_path(), filesystem_error);
            if (filesystem_error)
            {
                if (error)
                {
                    *error = "cannot_create_journal_directory:" + filesystem_error.message();
                }
                return false;
            }
        }

        const std::filesystem::path temporary_path = path.string() + ".tmp";
        std::ofstream stream(temporary_path, std::ios::binary | std::ios::trunc);
        if (!stream)
        {
            if (error)
            {
                *error = "cannot_open_temporary_journal";
            }
            return false;
        }
        stream << kHeader << '\n';
        for (const TaskRunSnapshot& snapshot : snapshots)
        {
            stream << "RUN";
            appendField(stream, snapshot.runId);
            stream << '\t' << snapshot.attemptId;
            appendField(stream, snapshot.definition.taskId);
            appendField(stream, snapshot.definition.kind);
            appendField(stream, snapshot.definition.displayName);
            appendField(stream, snapshot.definition.projectKey);
            appendField(stream, snapshot.definition.chunkId);
            stream << '\t' << snapshot.definition.projectGeneration;
            stream << '\t' << snapshot.definition.priority;
            stream << '\t' << snapshot.queueSequence;
            stream << '\t' << snapshot.revision;
            appendField(stream, taskStateName(snapshot.state));
            appendField(stream, snapshot.blockedReason);
            stream << '\t' << joinDependencies(snapshot.definition.dependencies);
            stream << '\t' << snapshot.definition.capabilities.canPause;
            stream << '\t' << snapshot.definition.capabilities.canCheckpoint;
            stream << '\t' << snapshot.definition.capabilities.canReorder;
            stream << '\t' << snapshot.definition.capabilities.canCancel;
            stream << '\t' << snapshot.definition.resources.cpuSlots;
            appendField(stream, snapshot.definition.resources.accelerator);
            stream << '\t' << snapshot.definition.resources.acceleratorSlots;
            stream << '\t' << static_cast<int>(snapshot.definition.resources.projectAccess);
            appendField(stream, snapshot.definition.payload);
            appendField(stream, snapshot.progress.stage);
            stream << '\t' << snapshot.progress.completedUnits;
            stream << '\t' << snapshot.progress.totalUnits;
            stream << '\t' << snapshot.checkpoint.has_value();
            stream << '\t' << (snapshot.checkpoint ? snapshot.checkpoint->schemaVersion : 0);
            appendField(stream, snapshot.checkpoint ? snapshot.checkpoint->location : std::string());
            appendField(stream, snapshot.checkpoint ? snapshot.checkpoint->inputSignature : std::string());
            stream << '\t' << (snapshot.checkpoint ? snapshot.checkpoint->completedUnits : 0);
            stream << '\t' << (snapshot.checkpoint ? snapshot.checkpoint->totalUnits : 0);
            stream << '\t' << snapshot.result.has_value();
            appendField(stream, snapshot.result ? snapshot.result->summary : std::string());
            appendField(stream, snapshot.result ? snapshot.result->outputLocation : std::string());
            stream << '\t' << snapshot.error.has_value();
            appendField(stream, snapshot.error ? snapshot.error->code : std::string());
            appendField(stream, snapshot.error ? snapshot.error->message : std::string());
            stream << '\t' << (snapshot.error && snapshot.error->retryable);
            stream << '\t' << toMilliseconds(snapshot.submittedAt);
            stream << '\t' << toMilliseconds(snapshot.startedAt);
            stream << '\t' << toMilliseconds(snapshot.finishedAt);
            stream << '\n';
        }
        stream.flush();
        if (!stream)
        {
            if (error)
            {
                *error = "cannot_write_journal";
            }
            stream.close();
            std::filesystem::remove(temporary_path, filesystem_error);
            return false;
        }
        stream.close();

        std::filesystem::rename(temporary_path, path, filesystem_error);
        if (filesystem_error)
        {
            std::filesystem::remove(path, filesystem_error);
            filesystem_error.clear();
            std::filesystem::rename(temporary_path, path, filesystem_error);
        }
        if (filesystem_error)
        {
            if (error)
            {
                *error = "cannot_replace_journal:" + filesystem_error.message();
            }
            std::filesystem::remove(temporary_path, filesystem_error);
            return false;
        }
        return true;
    }

    TaskJournalLoadResult TaskJournal::load(const std::filesystem::path& path)
    {
        TaskJournalLoadResult result;
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
        {
            result.error = "cannot_open_journal";
            return result;
        }

        std::string line;
        if (!std::getline(stream, line) || line != kHeader)
        {
            result.error = "unsupported_journal_schema";
            return result;
        }

        std::size_t line_number = 1;
        while (std::getline(stream, line))
        {
            ++line_number;
            if (line.empty())
            {
                continue;
            }
            const std::vector<std::string_view> fields = split(line, '\t');
            if (fields.size() != 43 || fields[0] != "RUN")
            {
                result.error = "invalid_journal_record:" + std::to_string(line_number);
                result.snapshots.clear();
                return result;
            }

            TaskRunSnapshot snapshot;
            std::string state_name;
            int project_access = 0;
            int can_pause = 0;
            int can_checkpoint = 0;
            int can_reorder = 0;
            int can_cancel = 0;
            int has_checkpoint = 0;
            int has_result = 0;
            int has_error = 0;
            int error_retryable = 0;
            std::int64_t submitted_ms = -1;
            std::int64_t started_ms = -1;
            std::int64_t finished_ms = -1;
            std::uint32_t checkpoint_schema = 0;
            TaskCheckpoint checkpoint;
            TaskResult task_result;
            TaskError task_error;
            if (!decodeString(fields, 1, &snapshot.runId) || !parseInteger(fields[2], &snapshot.attemptId) ||
                !decodeString(fields, 3, &snapshot.definition.taskId) ||
                !decodeString(fields, 4, &snapshot.definition.kind) ||
                !decodeString(fields, 5, &snapshot.definition.displayName) ||
                !decodeString(fields, 6, &snapshot.definition.projectKey) ||
                !decodeString(fields, 7, &snapshot.definition.chunkId) ||
                !parseInteger(fields[8], &snapshot.definition.projectGeneration) ||
                !parseInteger(fields[9], &snapshot.definition.priority) ||
                !parseInteger(fields[10], &snapshot.queueSequence) || !parseInteger(fields[11], &snapshot.revision) ||
                !decodeString(fields, 12, &state_name) || !decodeString(fields, 13, &snapshot.blockedReason) ||
                !parseDependencies(fields[14], &snapshot.definition.dependencies) ||
                !parseInteger(fields[15], &can_pause) || !parseInteger(fields[16], &can_checkpoint) ||
                !parseInteger(fields[17], &can_reorder) || !parseInteger(fields[18], &can_cancel) ||
                !parseInteger(fields[19], &snapshot.definition.resources.cpuSlots) ||
                !decodeString(fields, 20, &snapshot.definition.resources.accelerator) ||
                !parseInteger(fields[21], &snapshot.definition.resources.acceleratorSlots) ||
                !parseInteger(fields[22], &project_access) || !decodeString(fields, 23, &snapshot.definition.payload) ||
                !decodeString(fields, 24, &snapshot.progress.stage) ||
                !parseInteger(fields[25], &snapshot.progress.completedUnits) ||
                !parseInteger(fields[26], &snapshot.progress.totalUnits) ||
                !parseInteger(fields[27], &has_checkpoint) || !parseInteger(fields[28], &checkpoint_schema) ||
                !decodeString(fields, 29, &checkpoint.location) ||
                !decodeString(fields, 30, &checkpoint.inputSignature) ||
                !parseInteger(fields[31], &checkpoint.completedUnits) ||
                !parseInteger(fields[32], &checkpoint.totalUnits) || !parseInteger(fields[33], &has_result) ||
                !decodeString(fields, 34, &task_result.summary) ||
                !decodeString(fields, 35, &task_result.outputLocation) || !parseInteger(fields[36], &has_error) ||
                !decodeString(fields, 37, &task_error.code) || !decodeString(fields, 38, &task_error.message) ||
                !parseInteger(fields[39], &error_retryable) || !parseInteger(fields[40], &submitted_ms) ||
                !parseInteger(fields[41], &started_ms) || !parseInteger(fields[42], &finished_ms))
            {
                result.error = "invalid_journal_field:" + std::to_string(line_number);
                result.snapshots.clear();
                return result;
            }
            const std::optional<TaskState> state = taskStateFromName(state_name);
            if (!state || project_access < static_cast<int>(ProjectAccess::None) ||
                project_access > static_cast<int>(ProjectAccess::Write))
            {
                result.error = "invalid_journal_enum:" + std::to_string(line_number);
                result.snapshots.clear();
                return result;
            }
            snapshot.state = *state;
            snapshot.definition.capabilities = {can_pause != 0, can_checkpoint != 0, can_reorder != 0, can_cancel != 0};
            snapshot.definition.resources.projectAccess = static_cast<ProjectAccess>(project_access);
            if (snapshot.progress.totalUnits > 0)
            {
                snapshot.progress.fraction = static_cast<double>(snapshot.progress.completedUnits) /
                                             static_cast<double>(snapshot.progress.totalUnits);
            }
            if (has_checkpoint != 0)
            {
                checkpoint.schemaVersion = checkpoint_schema;
                snapshot.checkpoint = std::move(checkpoint);
            }
            if (has_result != 0)
            {
                snapshot.result = std::move(task_result);
            }
            if (has_error != 0)
            {
                task_error.retryable = error_retryable != 0;
                snapshot.error = std::move(task_error);
            }
            snapshot.submittedAt = fromMilliseconds(submitted_ms).value_or(std::chrono::system_clock::time_point{});
            snapshot.startedAt = fromMilliseconds(started_ms);
            snapshot.finishedAt = fromMilliseconds(finished_ms);
            result.snapshots.push_back(std::move(snapshot));
        }
        result.succeeded = true;
        return result;
    }

} // namespace xjw::task_runtime
