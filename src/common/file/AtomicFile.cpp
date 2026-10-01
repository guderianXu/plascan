#include "file/FileIO.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <system_error>
#include <limits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace xjw::common::file
{
    namespace
    {

        std::error_code reserveTemporary(const std::filesystem::path& path)
        {
#ifdef _WIN32
            const HANDLE handle =
                CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle == INVALID_HANDLE_VALUE)
            {
                return {static_cast<int>(GetLastError()), std::system_category()};
            }
            CloseHandle(handle);
#else
            const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
            if (descriptor < 0)
            {
                return {errno, std::generic_category()};
            }
            ::close(descriptor);
#endif
            return {};
        }

        std::error_code syncTemporary(const std::filesystem::path& path)
        {
#ifdef _WIN32
            const HANDLE handle = CreateFileW(
                path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle == INVALID_HANDLE_VALUE)
            {
                return {static_cast<int>(GetLastError()), std::system_category()};
            }
            const bool ok = FlushFileBuffers(handle);
            const auto failure =
                ok ? std::error_code{} : std::error_code(static_cast<int>(GetLastError()), std::system_category());
            CloseHandle(handle);
#else
            const int descriptor = ::open(path.c_str(), O_RDONLY);
            if (descriptor < 0)
            {
                return {errno, std::generic_category()};
            }
            int result;
            do
            {
                result = ::fsync(descriptor);
            } while (result < 0 && errno == EINTR);
            const auto failure = result == 0 ? std::error_code{} : std::error_code(errno, std::generic_category());
            ::close(descriptor);
#endif
            return failure;
        }

        std::error_code replaceFile(const std::filesystem::path& temporary, const std::filesystem::path& destination)
        {
#ifdef _WIN32
            // Another writer can briefly hold the destination while publishing its own temporary file.
            DWORD error_code = ERROR_SUCCESS;
            for (int attempt = 0; attempt < 64; ++attempt)
            {
                if (MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING))
                {
                    return {};
                }
                error_code = GetLastError();
                if ((error_code != ERROR_ACCESS_DENIED && error_code != ERROR_SHARING_VIOLATION &&
                     error_code != ERROR_LOCK_VIOLATION) ||
                    attempt == 63)
                {
                    break;
                }
                Sleep(1);
            }
            return {static_cast<int>(error_code), std::system_category()};
#else
            std::error_code error;
            std::filesystem::rename(temporary, destination, error);
            return error;
#endif
        }

    } // namespace

    AtomicFile::AtomicFile(std::filesystem::path destination) : _destination(std::move(destination))
    {
    }

    AtomicFile::~AtomicFile()
    {
        _stream.exceptions(std::ios::goodbit);
        _stream.close();
        if (!_committed && !_temporary.empty())
        {
            std::error_code ignored;
            std::filesystem::remove(_temporary, ignored);
        }
    }

    bool AtomicFile::open(std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        const auto fail = [&](const std::string& reason)
        {
            if (error)
            {
                *error = "无法打开原子写入文件 " + pathToUtf8(_destination) + ": " + reason;
            }
            return false;
        };
        if (_destination.empty() || _stream.is_open() || !_temporary.empty() || _committed)
        {
            return fail("目标路径为空或写入器已使用");
        }
        // Preserve existing file aliases: publishing through a symlink replaces its target.
        std::error_code link_error;
        const auto link_status = std::filesystem::symlink_status(_destination, link_error);
        if (std::filesystem::is_symlink(link_status))
        {
            const auto target = std::filesystem::canonical(_destination, link_error);
            if (link_error)
            {
                return fail("无法解析文件符号链接: " + link_error.message());
            }
            _destination = target;
        }
        else if (link_error && link_error != std::errc::no_such_file_or_directory)
        {
            return fail(link_error.message());
        }
        const auto parent = _destination.has_parent_path() ? _destination.parent_path() : std::filesystem::path(".");
        if (!ensureDirectory(parent, error))
        {
            return false;
        }
        static std::atomic<std::uint64_t> sequence{0};
        for (int attempt = 0; attempt < 64; ++attempt)
        {
            auto candidate = _destination;
            candidate +=
                pathFromUtf8(".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                             "-" + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
            const std::error_code failure = reserveTemporary(candidate);
            if (!failure)
            {
                _temporary = std::move(candidate);
                _stream.open(_temporary, std::ios::binary | std::ios::trunc);
                return _stream ? true : fail("无法打开临时文件流");
            }
            if (failure != std::errc::file_exists)
            {
                return fail(failure.message());
            }
        }
        return fail("无法分配唯一临时文件");
    }

    std::ofstream& AtomicFile::stream()
    {
        return _stream;
    }

    bool AtomicFile::commit(std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        const auto fail = [&](const std::string& reason)
        {
            if (error)
            {
                *error = "原子提交失败 " + pathToUtf8(_destination) + ": " + reason;
            }
            return false;
        };
        if (!_stream.is_open() || _temporary.empty() || _committed)
        {
            return fail("写入器尚未打开或已提交");
        }
        _stream.flush();
        if (!_stream)
        {
            return fail("临时文件写入失败");
        }
        _stream.close();
        if (_stream.fail())
        {
            return fail("关闭临时文件失败");
        }
        std::error_code failure = syncTemporary(_temporary);
        if (failure)
        {
            return fail("同步临时文件失败: " + failure.message());
        }
        const auto status = std::filesystem::status(_destination, failure);
        if (!failure && std::filesystem::is_regular_file(status))
        {
            std::filesystem::permissions(_temporary, status.permissions(), failure);
            if (failure)
            {
                return fail(failure.message());
            }
        }
        else if (failure && failure != std::errc::no_such_file_or_directory)
        {
            return fail(failure.message());
        }
        failure = replaceFile(_temporary, _destination);
        if (failure)
        {
            return fail(failure.message());
        }
        _committed = true;
        return true;
    }

    bool writeFileAtomic(const std::filesystem::path& path, std::string_view bytes, std::string* error)
    {
        if (bytes.size() > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        {
            if (error)
            {
                *error = "写入文件过大: " + pathToUtf8(path);
            }
            return false;
        }
        AtomicFile output(path);
        if (!output.open(error))
        {
            return false;
        }
        if (!bytes.empty())
        {
            output.stream().write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        return output.commit(error);
    }

} // namespace xjw::common::file
