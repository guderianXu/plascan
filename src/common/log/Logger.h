#pragma once

#include <cstdarg>
#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#if __has_include(<QString>)
#include <QString>
#include <QByteArray>
#endif

class Logger
{
public:
    enum Level
    {
        Debug = 0,
        Info = 1,
        Warn = 2,
        Error = 3
    };

    struct Entry
    {
        Level level = Info;
        std::string timestamp;
        std::string message;
        std::string formatted;
    };

    using SinkCallback = std::function<void(const Entry &entry)>;

    /**
     * @brief Temporarily suppress low-priority log entries on the current thread.
     *
     * The filter is thread-local, so noisy parallel probe workers can avoid
     * contending on the global file sink without hiding warnings from other tasks.
     */
    class ScopedThreadMinimumLevel
    {
    public:
        explicit ScopedThreadMinimumLevel(Level level)
            : _previous(threadMinimumLevelStorage())
        {
            if (level > threadMinimumLevelStorage())
            {
                threadMinimumLevelStorage() = level;
            }
        }

        ~ScopedThreadMinimumLevel()
        {
            threadMinimumLevelStorage() = _previous;
        }

        ScopedThreadMinimumLevel(const ScopedThreadMinimumLevel &) = delete;
        ScopedThreadMinimumLevel &operator=(const ScopedThreadMinimumLevel &) = delete;

    private:
        Level _previous;
    };

    static Logger *instance();

    int registerSink(SinkCallback sink);
    void unregisterSink(int sinkId);

    void setLogDirectory(const std::string &dir);
#if __has_include(<QString>)
    void setLogDirectory(const QString &dir)
    {
        setLogDirectory(qStringToUtf8(dir));
    }
#endif
    std::string logDirectory() const;
    std::string logFilePath() const;
    void clearLogFile();

    void setMaxFileSize(std::uintmax_t bytes);
    void setMaxBackupFiles(int n);
    void setTerminalMinimumLevel(Level level);
    Level terminalMinimumLevel() const;

    void log(Level level, std::string_view message);
    void logf(Level level, const char *fmt, ...);
    void vlogf(Level level, const char *fmt, std::va_list args);

    void debug(std::string_view msg) { log(Debug, msg); }
    void info(std::string_view msg) { log(Info, msg); }
    void warn(std::string_view msg) { log(Warn, msg); }
    void error(std::string_view msg) { log(Error, msg); }
    void debug(const char *msg) { log(Debug, msg ? std::string_view(msg) : std::string_view()); }
    void info(const char *msg) { log(Info, msg ? std::string_view(msg) : std::string_view()); }
    void warn(const char *msg) { log(Warn, msg ? std::string_view(msg) : std::string_view()); }
    void error(const char *msg) { log(Error, msg ? std::string_view(msg) : std::string_view()); }

    void debugf(const char *fmt, ...);
    void infof(const char *fmt, ...);
    void warnf(const char *fmt, ...);
    void errorf(const char *fmt, ...);

#if __has_include(<QString>)
    void debug(const QString &msg) { debug(qStringToUtf8(msg)); }
    void info(const QString &msg) { info(qStringToUtf8(msg)); }
    void warn(const QString &msg) { warn(qStringToUtf8(msg)); }
    void error(const QString &msg) { error(qStringToUtf8(msg)); }
#endif

private:
    Logger();

    static Level &threadMinimumLevelStorage()
    {
        static thread_local Level level = Debug;
        return level;
    }

    static std::string defaultLogDirectory();
    static std::string currentTimestamp();
    static const char *levelName(Level level);
    static std::string formatString(const char *fmt, std::va_list args);
#if __has_include(<QString>)
    static std::string qStringToUtf8(const QString &msg)
    {
        const QByteArray bytes = msg.toUtf8();
        return std::string(bytes.constData(), static_cast<std::size_t>(bytes.size()));
    }
#endif

    std::string formatLine(Level level,
                           const std::string &timestamp,
                           std::string_view message) const;
    void ensureLogDirectoryLocked();
    bool openLogFileLocked(bool truncate);
    void rotateIfNeededLocked();
    void writeToTerminal(const Entry &entry) const;

    mutable std::mutex _mutex;
    std::string _logDir;
    std::string _logFilePath;
    std::ofstream _file;
    std::uintmax_t _maxSize{5u * 1024u * 1024u};
    int _maxFiles{3};
    Level _terminalMinimumLevel{Info};
    int _nextSinkId{1};
    std::unordered_map<int, SinkCallback> _sinks;
    mutable std::mutex _terminalMutex;
};

namespace logger_detail
{
inline void debug(const char *msg) { Logger::instance()->debug(msg); }
inline void info(const char *msg) { Logger::instance()->info(msg); }
inline void warn(const char *msg) { Logger::instance()->warn(msg); }
inline void error(const char *msg) { Logger::instance()->error(msg); }

inline void debug(std::string_view msg) { Logger::instance()->debug(msg); }
inline void info(std::string_view msg) { Logger::instance()->info(msg); }
inline void warn(std::string_view msg) { Logger::instance()->warn(msg); }
inline void error(std::string_view msg) { Logger::instance()->error(msg); }

inline void debug(const std::string &msg) { Logger::instance()->debug(msg); }
inline void info(const std::string &msg) { Logger::instance()->info(msg); }
inline void warn(const std::string &msg) { Logger::instance()->warn(msg); }
inline void error(const std::string &msg) { Logger::instance()->error(msg); }

#if __has_include(<QString>)
inline void debug(const QString &msg) { Logger::instance()->debug(msg); }
inline void info(const QString &msg) { Logger::instance()->info(msg); }
inline void warn(const QString &msg) { Logger::instance()->warn(msg); }
inline void error(const QString &msg) { Logger::instance()->error(msg); }
#endif

template <typename... Args>
inline void debug(const char *fmt, Args... args) { Logger::instance()->debugf(fmt, args...); }

template <typename... Args>
inline void info(const char *fmt, Args... args) { Logger::instance()->infof(fmt, args...); }

template <typename... Args>
inline void warn(const char *fmt, Args... args) { Logger::instance()->warnf(fmt, args...); }

template <typename... Args>
inline void error(const char *fmt, Args... args) { Logger::instance()->errorf(fmt, args...); }
}

#define LOG_DEBUG(...) logger_detail::debug(__VA_ARGS__)
#define LOG_INFO(...) logger_detail::info(__VA_ARGS__)
#define LOG_WARN(...) logger_detail::warn(__VA_ARGS__)
#define LOG_ERROR(...) logger_detail::error(__VA_ARGS__)
