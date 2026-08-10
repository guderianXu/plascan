#include "io/PathIO.h"

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QSaveFile>

#include <system_error>

namespace xjw::common::io
{
namespace
{

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

void clearError(QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
}

bool ensureParentDirectory(const QString &path, QString *errorMessage)
{
    const QFileInfo info(path);
    const QString parentPath = info.absolutePath();
    if (parentPath.isEmpty() || QDir(parentPath).exists())
    {
        return true;
    }
    if (QDir().mkpath(parentPath))
    {
        return true;
    }

    setError(errorMessage, QStringLiteral("无法创建父目录: %1").arg(parentPath));
    return false;
}

bool normalizePathForSafety(const std::filesystem::path &path,
                            std::filesystem::path *normalized,
                            std::error_code *error)
{
    if (path.empty())
    {
        *error = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    std::error_code current_error;
    const std::filesystem::path absolute_path = std::filesystem::absolute(path, current_error);
    if (current_error)
    {
        *error = current_error;
        return false;
    }

    std::filesystem::path canonical_path = std::filesystem::weakly_canonical(absolute_path, current_error);
    if (current_error || canonical_path.empty() || !canonical_path.is_absolute())
    {
        *error = current_error ? current_error : std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    *normalized = canonical_path.lexically_normal();
    error->clear();
    return true;
}

bool pathComponentEquals(const std::filesystem::path &first,
                         const std::filesystem::path &second)
{
#ifdef _WIN32
    return QString::fromStdWString(first.native()).compare(QString::fromStdWString(second.native()),
                                                            Qt::CaseInsensitive) == 0;
#else
    return first == second;
#endif
}

bool pathComponentsEqual(const std::filesystem::path &first,
                         const std::filesystem::path &second)
{
    auto first_it = first.begin();
    auto second_it = second.begin();
    for (; first_it != first.end() && second_it != second.end(); ++first_it, ++second_it)
    {
        if (!pathComponentEquals(*first_it, *second_it))
        {
            return false;
        }
    }
    return first_it == first.end() && second_it == second.end();
}

bool isStrictPathAncestor(const std::filesystem::path &ancestor,
                          const std::filesystem::path &descendant)
{
    auto ancestor_it = ancestor.begin();
    auto descendant_it = descendant.begin();
    for (; ancestor_it != ancestor.end() && descendant_it != descendant.end();
         ++ancestor_it, ++descendant_it)
    {
        if (!pathComponentEquals(*ancestor_it, *descendant_it))
        {
            return false;
        }
    }
    return ancestor_it == ancestor.end() && descendant_it != descendant.end();
}

bool pathExistsForSafety(const std::filesystem::path &path,
                         bool *exists,
                         std::error_code *error)
{
    std::error_code current_error;
    *exists = std::filesystem::exists(path, current_error);
    if (current_error)
    {
        *error = current_error;
        return false;
    }
    return true;
}

} // namespace

std::filesystem::path toFilesystemPath(const QString &path)
{
#ifdef _WIN32
    return std::filesystem::path(path.toStdWString());
#else
    const QByteArray bytes = path.toUtf8();
    return std::filesystem::path(bytes.constData());
#endif
}

std::filesystem::path toFilesystemPath(const std::string &path)
{
    return toFilesystemPath(fromUtf8Path(path));
}

std::string toUtf8Path(const QString &path)
{
    const QByteArray bytes = path.toUtf8();
    return std::string(bytes.constData(), static_cast<size_t>(bytes.size()));
}

std::string toUtf8Path(const std::filesystem::path &path)
{
    return toUtf8Path(fromFilesystemPath(path));
}

std::string toNativeNarrowPath(const QString &path)
{
#ifdef _WIN32
    const QByteArray bytes = QDir::toNativeSeparators(path).toLocal8Bit();
#else
    const QByteArray bytes = path.toUtf8();
#endif
    return std::string(bytes.constData(), static_cast<size_t>(bytes.size()));
}

std::string toNativeNarrowPath(const std::string &path)
{
    return toNativeNarrowPath(fromUtf8Path(path));
}

std::string toNativeNarrowPath(const std::filesystem::path &path)
{
    return toNativeNarrowPath(fromFilesystemPath(path));
}

QString fromUtf8Path(const std::string &path)
{
    return QString::fromUtf8(path.data(), static_cast<int>(path.size()));
}

QString fromFilesystemPath(const std::filesystem::path &path)
{
#ifdef _WIN32
    return QString::fromStdWString(path.wstring());
#else
    const std::string nativePath = path.string();
    return QString::fromUtf8(nativePath.data(), static_cast<int>(nativePath.size()));
#endif
}

SafePathComparison comparePathsSafely(const std::filesystem::path &first,
                                      const std::filesystem::path &second)
{
    SafePathComparison comparison;
    if (!normalizePathForSafety(first, &comparison.normalizedFirst, &comparison.error)
        || !normalizePathForSafety(second, &comparison.normalizedSecond, &comparison.error))
    {
        return comparison;
    }

    bool first_exists = false;
    bool second_exists = false;
    if (!pathExistsForSafety(comparison.normalizedFirst, &first_exists, &comparison.error)
        || !pathExistsForSafety(comparison.normalizedSecond, &second_exists, &comparison.error))
    {
        return comparison;
    }

    comparison.equivalent = pathComponentsEqual(comparison.normalizedFirst,
                                                 comparison.normalizedSecond);
    if (!comparison.equivalent && first_exists && second_exists)
    {
        std::error_code equivalent_error;
        comparison.equivalent = std::filesystem::equivalent(comparison.normalizedFirst,
                                                             comparison.normalizedSecond,
                                                             equivalent_error);
        if (equivalent_error)
        {
            comparison.error = equivalent_error;
            return comparison;
        }
    }

    comparison.firstIsAncestorOfSecond = !comparison.equivalent
        && isStrictPathAncestor(comparison.normalizedFirst, comparison.normalizedSecond);
    comparison.secondIsAncestorOfFirst = !comparison.equivalent
        && isStrictPathAncestor(comparison.normalizedSecond, comparison.normalizedFirst);
    comparison.firstIsRoot = !comparison.normalizedFirst.root_path().empty()
        && pathComponentsEqual(comparison.normalizedFirst, comparison.normalizedFirst.root_path());
    comparison.secondIsRoot = !comparison.normalizedSecond.root_path().empty()
        && pathComponentsEqual(comparison.normalizedSecond, comparison.normalizedSecond.root_path());
    comparison.error.clear();
    comparison.valid = true;
    return comparison;
}

std::ifstream openInputFile(const QString &path, std::ios::openmode mode)
{
    return std::ifstream(toFilesystemPath(path), mode);
}

std::ifstream openInputFile(const std::string &path, std::ios::openmode mode)
{
    return openInputFile(fromUtf8Path(path), mode);
}

std::ifstream openInputFile(const std::filesystem::path &path, std::ios::openmode mode)
{
    return std::ifstream(path, mode);
}

std::ofstream openOutputFile(const QString &path, std::ios::openmode mode)
{
    return std::ofstream(toFilesystemPath(path), mode);
}

std::ofstream openOutputFile(const std::string &path, std::ios::openmode mode)
{
    return openOutputFile(fromUtf8Path(path), mode);
}

std::ofstream openOutputFile(const std::filesystem::path &path, std::ios::openmode mode)
{
    return std::ofstream(path, mode);
}

QByteArray readFileBytes(const QString &path, QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage, QStringLiteral("无法读取文件: %1 (%2)").arg(path, file.errorString()));
        return {};
    }

    clearError(errorMessage);
    return file.readAll();
}

QByteArray readFileBytes(const std::string &path, QString *errorMessage)
{
    return readFileBytes(fromUtf8Path(path), errorMessage);
}

QByteArray readFileBytes(const std::filesystem::path &path, QString *errorMessage)
{
    return readFileBytes(fromFilesystemPath(path), errorMessage);
}

bool writeFileBytesAtomic(const QString &path, const QByteArray &bytes, QString *errorMessage)
{
    if (!ensureParentDirectory(path, errorMessage))
    {
        return false;
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        setError(errorMessage, QStringLiteral("无法写入文件: %1 (%2)").arg(path, file.errorString()));
        return false;
    }

    const qint64 expectedSize = bytes.size();
    const qint64 written = file.write(bytes.constData(), expectedSize);
    if (written != expectedSize)
    {
        file.cancelWriting();
        setError(errorMessage, QStringLiteral("文件写入不完整: %1").arg(path));
        return false;
    }

    if (!file.commit())
    {
        setError(errorMessage, QStringLiteral("无法提交文件写入: %1 (%2)").arg(path, file.errorString()));
        return false;
    }

    clearError(errorMessage);
    return true;
}

bool writeFileBytesAtomic(const std::string &path, const QByteArray &bytes, QString *errorMessage)
{
    return writeFileBytesAtomic(fromUtf8Path(path), bytes, errorMessage);
}

bool writeFileBytesAtomic(const std::filesystem::path &path, const QByteArray &bytes, QString *errorMessage)
{
    return writeFileBytesAtomic(fromFilesystemPath(path), bytes, errorMessage);
}

} // namespace xjw::common::io
