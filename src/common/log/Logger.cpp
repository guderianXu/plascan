#include "Logger.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace
{
std::tm localTime(std::time_t timeValue)
{
    std::tm tmValue{};
#if defined(_WIN32)
    localtime_s(&tmValue, &timeValue);
#else
    localtime_r(&timeValue, &tmValue);
#endif
    return tmValue;
}
}

Logger *Logger::instance()
{
    static Logger logger;
    return &logger;
}

Logger::Logger()
    : _logDir(defaultLogDirectory())
{
    _logFilePath = (std::filesystem::path(_logDir) / "plascan.log").string();
}

int Logger::registerSink(SinkCallback sink)
{
    if (!sink)
    {
        return 0;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    const int sinkId = _nextSinkId++;
    _sinks.emplace(sinkId, std::move(sink));
    return sinkId;
}

void Logger::unregisterSink(int sinkId)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _sinks.erase(sinkId);
}

void Logger::setLogDirectory(const std::string &dir)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _logDir = dir.empty() ? defaultLogDirectory() : dir;
    _logFilePath = (std::filesystem::path(_logDir) / "plascan.log").string();
    if (_file.is_open())
    {
        _file.close();
    }
    openLogFileLocked(false);
}

std::string Logger::logDirectory() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _logDir;
}

std::string Logger::logFilePath() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _logFilePath;
}

void Logger::clearLogFile()
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (_file.is_open())
    {
        _file.close();
    }
    openLogFileLocked(true);
}

void Logger::setMaxFileSize(std::uintmax_t bytes)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _maxSize = bytes;
}

void Logger::setMaxBackupFiles(int n)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _maxFiles = (n < 1) ? 1 : n;
}

void Logger::log(Level level, std::string_view message)
{
    Entry entry;
    entry.level = level;
    entry.timestamp = currentTimestamp();
    entry.message.assign(message.begin(), message.end());
    entry.formatted = formatLine(level, entry.timestamp, entry.message);

    std::vector<SinkCallback> sinks;
    bool terminalFallback = false;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (openLogFileLocked(false))
        {
            _file << entry.formatted;
            _file.flush();
            rotateIfNeededLocked();
        }

        sinks.reserve(_sinks.size());
        for (const auto &item : _sinks)
        {
            sinks.push_back(item.second);
        }
        terminalFallback = sinks.empty();
    }

    if (terminalFallback)
    {
        writeToTerminalFallback(entry);
    }

    for (const auto &sink : sinks)
    {
        try
        {
            sink(entry);
        }
        catch (...)
        {
        }
    }
}

void Logger::logf(Level level, const char *fmt, ...)
{
    std::va_list args;
    va_start(args, fmt);
    vlogf(level, fmt, args);
    va_end(args);
}

void Logger::vlogf(Level level, const char *fmt, std::va_list args)
{
    log(level, formatString(fmt, args));
}

void Logger::debugf(const char *fmt, ...)
{
    std::va_list args;
    va_start(args, fmt);
    vlogf(Debug, fmt, args);
    va_end(args);
}

void Logger::infof(const char *fmt, ...)
{
    std::va_list args;
    va_start(args, fmt);
    vlogf(Info, fmt, args);
    va_end(args);
}

void Logger::warnf(const char *fmt, ...)
{
    std::va_list args;
    va_start(args, fmt);
    vlogf(Warn, fmt, args);
    va_end(args);
}

void Logger::errorf(const char *fmt, ...)
{
    std::va_list args;
    va_start(args, fmt);
    vlogf(Error, fmt, args);
    va_end(args);
}

std::string Logger::defaultLogDirectory()
{
    if (const char *xdgStateHome = std::getenv("XDG_STATE_HOME"))
    {
        return (std::filesystem::path(xdgStateHome) / "PlaScan" / "logs").string();
    }
    if (const char *home = std::getenv("HOME"))
    {
        return (std::filesystem::path(home) / ".local" / "state" / "PlaScan" / "logs").string();
    }
    return (std::filesystem::current_path() / "logs").string();
}

std::string Logger::currentTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t timeValue = std::chrono::system_clock::to_time_t(now);
    const std::tm tmValue = localTime(timeValue);

    std::ostringstream stream;
    stream << std::put_time(&tmValue, "%Y-%m-%dT%H:%M:%S");
    return stream.str();
}

const char *Logger::levelName(Level level)
{
    switch (level)
    {
    case Debug:
        return "DEBUG";
    case Info:
        return "INFO";
    case Warn:
        return "WARN";
    case Error:
        return "ERROR";
    default:
        return "INFO";
    }
}

std::string Logger::formatString(const char *fmt, std::va_list args)
{
    if (!fmt)
    {
        return std::string();
    }

    std::va_list argsCopy;
    va_copy(argsCopy, args);
    const int required = std::vsnprintf(nullptr, 0, fmt, argsCopy);
    va_end(argsCopy);
    if (required <= 0)
    {
        return std::string(fmt);
    }

    std::vector<char> buffer(static_cast<std::size_t>(required) + 1u, '\0');
    std::va_list argsCopy2;
    va_copy(argsCopy2, args);
    std::vsnprintf(buffer.data(), buffer.size(), fmt, argsCopy2);
    va_end(argsCopy2);
    return std::string(buffer.data(), static_cast<std::size_t>(required));
}

std::string Logger::formatLine(Level level,
                               const std::string &timestamp,
                               std::string_view message) const
{
    std::ostringstream stream;
    stream << '[' << timestamp << "] [" << levelName(level) << "] " << message << '\n';
    return stream.str();
}

void Logger::ensureLogDirectoryLocked()
{
    std::error_code errorCode;
    std::filesystem::create_directories(_logDir, errorCode);
}

bool Logger::openLogFileLocked(bool truncate)
{
    if (_logFilePath.empty())
    {
        _logFilePath = (std::filesystem::path(_logDir) / "plascan.log").string();
    }
    if (_file.is_open())
    {
        return true;
    }

    ensureLogDirectoryLocked();
    const auto mode = truncate ? (std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::app);
    _file.open(_logFilePath, mode);
    return _file.is_open();
}

void Logger::rotateIfNeededLocked()
{
    if (!_file.is_open())
    {
        return;
    }

    std::error_code errorCode;
    const auto size = std::filesystem::file_size(_logFilePath, errorCode);
    if (errorCode || size < _maxSize)
    {
        return;
    }

    _file.close();
    const std::filesystem::path basePath(_logFilePath);
    for (int index = _maxFiles - 1; index >= 1; --index)
    {
        const std::filesystem::path fromPath = basePath.string() + "." + std::to_string(index);
        const std::filesystem::path toPath = basePath.string() + "." + std::to_string(index + 1);
        std::filesystem::remove(toPath, errorCode);
        errorCode.clear();
        if (std::filesystem::exists(fromPath, errorCode))
        {
            errorCode.clear();
            std::filesystem::rename(fromPath, toPath, errorCode);
        }
        errorCode.clear();
    }

    const std::filesystem::path firstBackupPath = basePath.string() + ".1";
    std::filesystem::remove(firstBackupPath, errorCode);
    errorCode.clear();
    std::filesystem::rename(basePath, firstBackupPath, errorCode);
    errorCode.clear();
    openLogFileLocked(true);
}

void Logger::writeToTerminalFallback(const Entry &entry) const
{
    std::ostream &stream = (entry.level >= Warn) ? std::cerr : std::cout;
    stream << entry.formatted;
    stream.flush();
}