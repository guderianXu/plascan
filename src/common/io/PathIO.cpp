#include "io/PathIO.h"

#include <opencv2/imgcodecs.hpp>

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QSaveFile>

#include <cstring>

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

cv::Mat readImage(const QString &path, int flags)
{
    const QByteArray bytes = readFileBytes(path);
    if (bytes.isEmpty())
    {
        return {};
    }

    std::vector<uchar> encoded(static_cast<size_t>(bytes.size()));
    std::memcpy(encoded.data(), bytes.constData(), static_cast<size_t>(bytes.size()));
    return cv::imdecode(encoded, flags);
}

cv::Mat readImage(const std::string &path, int flags)
{
    return readImage(fromUtf8Path(path), flags);
}

cv::Mat readImage(const std::filesystem::path &path, int flags)
{
    return readImage(fromFilesystemPath(path), flags);
}

bool writeImage(const QString &path, const cv::Mat &image, const std::vector<int> &params)
{
    if (image.empty())
    {
        return false;
    }

    const QString suffix = QFileInfo(path).suffix().trimmed();
    if (suffix.isEmpty())
    {
        return false;
    }

    const QByteArray extension = QStringLiteral(".%1").arg(suffix).toLatin1();
    std::vector<uchar> encoded;
    if (!cv::imencode(extension.constData(), image, encoded, params))
    {
        return false;
    }

    const QByteArray bytes(reinterpret_cast<const char *>(encoded.data()), static_cast<qsizetype>(encoded.size()));
    return writeFileBytesAtomic(path, bytes);
}

bool writeImage(const std::string &path, const cv::Mat &image, const std::vector<int> &params)
{
    return writeImage(fromUtf8Path(path), image, params);
}

bool writeImage(const std::filesystem::path &path, const cv::Mat &image, const std::vector<int> &params)
{
    return writeImage(fromFilesystemPath(path), image, params);
}

} // namespace xjw::common::io
