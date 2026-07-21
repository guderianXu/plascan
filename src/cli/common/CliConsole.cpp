#include "CliConsole.h"

#include "Logger.h"

#include <QByteArray>

namespace xjw::cli
{

void printUtf8(FILE *stream, const QString &message, bool appendNewline)
{
    if (!stream)
    {
        return;
    }
    const QByteArray bytes = message.toUtf8();
    std::fwrite(bytes.constData(), 1, static_cast<size_t>(bytes.size()), stream);
    if (appendNewline && !bytes.endsWith('\n'))
    {
        std::fputc('\n', stream);
    }
    std::fflush(stream);
}

void printError(const QString &message)
{
    printUtf8(stderr, QStringLiteral("错误: %1").arg(message));
}

int registerConsoleLogger()
{
    static int sinkId = 0;
    if (sinkId == 0)
    {
        sinkId = Logger::instance()->registerSink([](const Logger::Entry &entry) {
            FILE *stream = entry.level >= Logger::Warn ? stderr : stdout;
            std::fwrite(entry.formatted.data(), 1, entry.formatted.size(), stream);
            std::fflush(stream);
        });
    }
    return sinkId;
}

} // namespace xjw::cli
