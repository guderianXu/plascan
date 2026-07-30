#include "project/PlascanArchive.h"

#include "project/ProjectPackageLayout.h"
#include "project/PortableProjectFormat.h"

#include <QByteArray>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSaveFile>
#include <QStringList>

#include <limits>
#include <string>

// libzip headers if available
#define HAVE_LIBZIP
#if defined(HAVE_LIBZIP)
#include <zip.h>
#endif

namespace
{

#if defined(HAVE_LIBZIP)

QByteArray zipEntryName(const QString &entryPath)
{
    return entryPath.toUtf8();
}

QString zipErrorToString(zip_error_t *zipError)
{
    if (!zipError)
    {
        return QStringLiteral("未知 libzip 错误");
    }

    const char *message = zip_error_strerror(zipError);
    if (!message)
    {
        return QStringLiteral("未知 libzip 错误");
    }
    return QString::fromUtf8(message);
}

QString zipOpenErrorCodeToString(int errorCode)
{
    zip_error_t zipError;
    zip_error_init_with_code(&zipError, errorCode);
    const QString message = zipErrorToString(&zipError);
    zip_error_fini(&zipError);
    return message;
}

zip_t *openArchiveFile(const QString &path, int flags, QString *errorMessage = nullptr)
{
#if defined(Q_OS_WIN)
    zip_error_t zipError;
    zip_error_init(&zipError);

    const std::wstring nativePath = QDir::toNativeSeparators(path).toStdWString();
    zip_source_t *source = zip_source_win32w_create(nativePath.c_str(), 0, -1, &zipError);
    if (!source)
    {
        if (errorMessage)
        {
            *errorMessage = zipErrorToString(&zipError);
        }
        zip_error_fini(&zipError);
        return nullptr;
    }

    zip_t *archive = zip_open_from_source(source, flags, &zipError);
    if (!archive)
    {
        if (errorMessage)
        {
            *errorMessage = zipErrorToString(&zipError);
        }
        zip_source_free(source);
        zip_error_fini(&zipError);
        return nullptr;
    }

    zip_error_fini(&zipError);
    return archive;
#else
    int errorCode = 0;
    const QByteArray nativePath = QFile::encodeName(path);
    zip_t *archive = zip_open(nativePath.constData(), flags, &errorCode);
    if (!archive && errorMessage)
    {
        *errorMessage = zipOpenErrorCodeToString(errorCode);
    }
    return archive;
#endif
}

zip_source_t *createFileSource(zip_t *archive,
                               const QString &sourcePath,
                               QString *errorMessage)
{
#if defined(Q_OS_WIN)
    const std::wstring nativePath =
        QDir::toNativeSeparators(sourcePath).toStdWString();
    zip_source_t *source = zip_source_win32w(
        archive, nativePath.c_str(), 0, -1);
#else
    const QByteArray nativePath = QFile::encodeName(sourcePath);
    zip_source_t *source = zip_source_file(
        archive, nativePath.constData(), 0, -1);
#endif
    if (!source && errorMessage)
    {
        *errorMessage = QString::fromUtf8(zip_strerror(archive));
    }
    return source;
}

#endif

} // namespace

PlascanArchive::PlascanArchive(const QString &path,
                               PlascanArchivePathType pathType)
    : _path(pathType == PlascanArchivePathType::ProjectDescriptor
                ? xjw::common::project::ProjectPackageLayout::
                      resolveMetadataArchive(path)
                : QDir::cleanPath(QFileInfo(path).absoluteFilePath()))
{
#if defined(HAVE_LIBZIP)
    zip_t *za = _path.isEmpty()
        ? nullptr
        : openArchiveFile(_path, ZIP_RDONLY);
    if (za)
    {
        _impl = za;
        _valid = true;
    }
    else
    {
        _impl = nullptr;
        _valid = false;
    }
#else
    Q_UNUSED(path);
    _valid = false;
#endif
}

PlascanArchive::~PlascanArchive()
{
#if defined(HAVE_LIBZIP)
    if (_impl)
    {
        zip_t *za = static_cast<zip_t*>(_impl);
        zip_close(za);
        _impl = nullptr;
    }
#endif
}

bool PlascanArchive::isValid() const
{
    return _valid;
}

bool PlascanArchive::containsEntry(const QString &entryPath) const
{
#if defined(HAVE_LIBZIP)
    if (!_impl
        || !xjw::common::project::PortableProjectFormat::isSafeEntryPath(
            entryPath))
    {
        return false;
    }
    zip_t *archive = static_cast<zip_t *>(_impl);
    const QByteArray entryName = zipEntryName(
        xjw::common::project::PortableProjectFormat::normalizeEntryPath(
            entryPath));
    return zip_name_locate(
               archive, entryName.constData(), ZIP_FL_ENC_UTF_8)
        >= 0;
#else
    Q_UNUSED(entryPath);
    return false;
#endif
}

QVector<QString> PlascanArchive::listEntries()
{
    QVector<QString> out;
#if defined(HAVE_LIBZIP)
    if (!_impl)
        return out;
    zip_t *za = static_cast<zip_t*>(_impl);
    zip_int64_t n = zip_get_num_entries(za, 0);
    for (zip_uint64_t i = 0; i < (zip_uint64_t)n; ++i)
    {
        const char *name = zip_get_name(za, (zip_uint64_t)i, 0);
        if (name)
            out.append(QString::fromUtf8(name));
    }
#endif
    return out;
}

QByteArray PlascanArchive::readEntry(const QString &entryPath, QString *err)
{
    QByteArray data;
#if defined(HAVE_LIBZIP)
    if (err)
    {
        err->clear();
    }
    if (!xjw::common::project::PortableProjectFormat::isSafeEntryPath(
            entryPath))
    {
        if (err)
        {
            *err = QStringLiteral("归档条目路径不安全: %1").arg(entryPath);
        }
        return data;
    }
    if (!_impl)
    {
        if (err) *err = QStringLiteral("libzip not available or archive not opened");
        return data;
    }
    zip_t *za = static_cast<zip_t*>(_impl);
    const QByteArray entryName = zipEntryName(
        xjw::common::project::PortableProjectFormat::normalizeEntryPath(
            entryPath));
    zip_file_t *zf = zip_fopen(za, entryName.constData(), ZIP_FL_ENC_UTF_8);
    if (!zf)
    {
        if (err) *err = QStringLiteral("entry not found");
        return data;
    }
    zip_stat_t st;
    if (zip_stat(za, entryName.constData(), ZIP_FL_ENC_UTF_8, &st) == 0)
    {
        if (st.size
            > static_cast<zip_uint64_t>(std::numeric_limits<int>::max()))
        {
            zip_fclose(zf);
            if (err)
            {
                *err = QStringLiteral(
                    "归档条目过大，必须使用 extractEntryToFile() 流式读取");
            }
            return {};
        }
        data.resize(static_cast<int>(st.size));
        const zip_int64_t r =
            zip_fread(zf, data.data(), static_cast<zip_uint64_t>(st.size));
        zip_fclose(zf);
        if (r < 0 || static_cast<zip_uint64_t>(r) != st.size)
        {
            if (err) *err = QStringLiteral("归档条目读取不完整");
            data.clear();
        }
    }
    else
    {
        zip_fclose(zf);
        if (err) *err = QStringLiteral("stat failed");
    }
#else
    Q_UNUSED(entryPath);
    if (err) *err = QStringLiteral("libzip not available at build time");
#endif
    return data;
}

bool PlascanArchive::createArchive(
    const QString &path,
    const QVector<QPair<QString, QByteArray>> &entries,
    QString *err)
{
#if defined(HAVE_LIBZIP)
    if (err)
    {
        err->clear();
    }
    if (entries.isEmpty())
    {
        if (err)
        {
            *err = QStringLiteral("不能创建没有条目的归档");
        }
        return false;
    }
    for (const auto &entry : entries)
    {
        if (!xjw::common::project::PortableProjectFormat::isSafeEntryPath(
                entry.first))
        {
            if (err)
            {
                *err = QStringLiteral("归档条目路径不安全: %1")
                           .arg(entry.first);
            }
            return false;
        }
    }
    if (!QDir().mkpath(QFileInfo(path).absolutePath()))
    {
        if (err)
        {
            *err = QStringLiteral("无法创建归档目录: %1")
                       .arg(QFileInfo(path).absolutePath());
        }
        return false;
    }

    QString openError;
    zip_t *za = openArchiveFile(path, ZIP_CREATE | ZIP_TRUNCATE, &openError);
    if (!za)
    {
        if (err) *err = QStringLiteral("无法创建归档: %1").arg(openError);
        return false;
    }

    for (const auto &entry : entries)
    {
        const QByteArray entryName = entry.first.toUtf8();
        zip_source_t *source = zip_source_buffer(
            za,
            entry.second.constData(),
            static_cast<zip_uint64_t>(entry.second.size()),
            0);
        if (!source
            || zip_file_add(
                   za,
                   entryName.constData(),
                   source,
                   ZIP_FL_ENC_UTF_8) < 0)
        {
            if (source)
            {
                zip_source_free(source);
            }
            if (err)
            {
                *err = QString::fromUtf8(zip_strerror(za));
            }
            zip_discard(za);
            return false;
        }
    }

    if (zip_close(za) < 0)
    {
        if (err) *err = QString::fromUtf8(zip_strerror(za));
        return false;
    }
    return true;
#else
    Q_UNUSED(path);
    Q_UNUSED(entries);
    if (err) *err = QStringLiteral("libzip 未在构建时启用，无法创建 .plascan 文件");
    return false;
#endif
}

bool PlascanArchive::writeEntry(const QString &entryPath,
                                const QByteArray &data,
                                QString *err)
{
    return writeEntries({qMakePair(entryPath, data)}, err);
}

bool PlascanArchive::writeEntries(const QVector<QPair<QString, QByteArray>> &entries,
                                  QString *err)
{
#if defined(HAVE_LIBZIP)
    if (err)
    {
        err->clear();
    }
    if (entries.isEmpty())
    {
        return true;
    }
    if (_path.isEmpty())
    {
        if (err)
            *err = QStringLiteral("归档路径为空");
        return false;
    }

    // Windows 不允许在同一归档仍被本对象的只读句柄占用时，用 libzip
    // 在 zip_close() 阶段重命名临时文件覆盖原归档。
    for (const auto &entry : entries)
    {
        if (!xjw::common::project::PortableProjectFormat::isSafeEntryPath(
                entry.first))
        {
            if (err)
            {
                *err = QStringLiteral("归档条目路径不安全: %1")
                           .arg(entry.first);
            }
            return false;
        }
    }
    closeReadHandle();

    // 以可写方式打开（如果不存在则创建）
    QString openError;
    zip_t *za = openArchiveFile(_path, ZIP_CREATE, &openError);
    if (!za)
    {
        if (err)
            *err = QStringLiteral("无法打开归档以写入: %1").arg(openError);
        return false;
    }
    for (const auto &entry : entries)
    {
        const QByteArray entryName = zipEntryName(
            xjw::common::project::PortableProjectFormat::normalizeEntryPath(
                entry.first));
        const zip_int64_t idx =
            zip_name_locate(za, entryName.constData(), ZIP_FL_ENC_UTF_8);
        if (idx >= 0 && zip_delete(za, static_cast<zip_uint64_t>(idx)) < 0)
        {
            if (err)
            {
                *err = QString::fromUtf8(zip_strerror(za));
            }
            zip_discard(za);
            return false;
        }

        zip_source_t *src = zip_source_buffer(
            za,
            entry.second.constData(),
            static_cast<zip_uint64_t>(entry.second.size()),
            0);
        if (!src)
        {
            if (err)
            {
                *err = QString::fromUtf8(zip_strerror(za));
            }
            zip_discard(za);
            return false;
        }

        if (zip_file_add(za, entryName.constData(), src, ZIP_FL_ENC_UTF_8) < 0)
        {
            if (err)
            {
                *err = QString::fromUtf8(zip_strerror(za));
            }
            zip_source_free(src);
            zip_discard(za);
            return false;
        }
    }

    if (zip_close(za) < 0)
    {
        if (err)
            *err = QString::fromUtf8(zip_strerror(za));
        zip_discard(za);
        return false;
    }

    reopenReadHandle();

    return true;
#else
    Q_UNUSED(entries);
    if (err)
        *err = QStringLiteral("libzip 未启用，无法写入归档");
    return false;
#endif
}

bool PlascanArchive::writeFileEntry(
    const QString &entryPath,
    const QString &sourcePath,
    PlascanArchiveCompression compression,
    QString *err)
{
#if defined(HAVE_LIBZIP)
    if (err)
    {
        err->clear();
    }
    if (!xjw::common::project::PortableProjectFormat::isSafeEntryPath(
            entryPath))
    {
        if (err)
        {
            *err = QStringLiteral("归档条目路径不安全: %1").arg(entryPath);
        }
        return false;
    }

    const QFileInfo sourceInfo(sourcePath);
    if (!sourceInfo.isFile())
    {
        if (err)
        {
            *err = QStringLiteral("待打包资源不存在或不是文件: %1")
                       .arg(sourcePath);
        }
        return false;
    }

    closeReadHandle();

    QString openError;
    zip_t *archive = openArchiveFile(_path, ZIP_CREATE, &openError);
    if (!archive)
    {
        if (err)
        {
            *err = QStringLiteral("无法打开归档以写入资源: %1")
                       .arg(openError);
        }
        reopenReadHandle();
        return false;
    }

    const QString normalized =
        xjw::common::project::PortableProjectFormat::normalizeEntryPath(
            entryPath);
    const QByteArray entryName = zipEntryName(normalized);
    const zip_int64_t existing =
        zip_name_locate(archive, entryName.constData(), ZIP_FL_ENC_UTF_8);
    if (existing >= 0
        && zip_delete(archive, static_cast<zip_uint64_t>(existing)) < 0)
    {
        if (err)
        {
            *err = QString::fromUtf8(zip_strerror(archive));
        }
        zip_discard(archive);
        reopenReadHandle();
        return false;
    }

    QString sourceError;
    zip_source_t *source =
        createFileSource(archive, sourceInfo.absoluteFilePath(), &sourceError);
    if (!source)
    {
        if (err)
        {
            *err = sourceError;
        }
        zip_discard(archive);
        reopenReadHandle();
        return false;
    }

    const zip_int64_t index =
        zip_file_add(archive, entryName.constData(), source, ZIP_FL_ENC_UTF_8);
    if (index < 0)
    {
        if (err)
        {
            *err = QString::fromUtf8(zip_strerror(archive));
        }
        zip_source_free(source);
        zip_discard(archive);
        reopenReadHandle();
        return false;
    }

    const zip_int32_t method =
        compression == PlascanArchiveCompression::Deflate
        ? ZIP_CM_DEFLATE
        : ZIP_CM_STORE;
    if (zip_set_file_compression(
            archive, static_cast<zip_uint64_t>(index), method, 0)
        < 0)
    {
        if (err)
        {
            *err = QString::fromUtf8(zip_strerror(archive));
        }
        zip_discard(archive);
        reopenReadHandle();
        return false;
    }

    if (zip_close(archive) < 0)
    {
        if (err)
        {
            *err = QString::fromUtf8(zip_strerror(archive));
        }
        zip_discard(archive);
        reopenReadHandle();
        return false;
    }

    reopenReadHandle();
    return true;
#else
    Q_UNUSED(entryPath);
    Q_UNUSED(sourcePath);
    Q_UNUSED(compression);
    if (err)
    {
        *err = QStringLiteral("libzip 未启用，无法写入归档");
    }
    return false;
#endif
}

bool PlascanArchive::updateFileEntries(
    const QVector<QPair<QString, QString>> &fileEntries,
    const QStringList &entriesToDelete,
    PlascanArchiveCompression compression,
    QString *err)
{
#if defined(HAVE_LIBZIP)
    if (err)
    {
        err->clear();
    }
    if (fileEntries.isEmpty() && entriesToDelete.isEmpty())
    {
        return true;
    }

    for (const auto &entry : fileEntries)
    {
        if (!xjw::common::project::PortableProjectFormat::isSafeEntryPath(
                entry.first)
            || !QFileInfo(entry.second).isFile())
        {
            if (err)
            {
                *err = QStringLiteral("待写入归档的文件或条目无效: %1 -> %2")
                           .arg(entry.first, entry.second);
            }
            return false;
        }
    }
    for (const QString &entry : entriesToDelete)
    {
        if (!xjw::common::project::PortableProjectFormat::isSafeEntryPath(
                entry))
        {
            if (err)
            {
                *err = QStringLiteral("待删除的归档条目路径不安全: %1")
                           .arg(entry);
            }
            return false;
        }
    }

    closeReadHandle();
    QString openError;
    zip_t *archive = openArchiveFile(_path, ZIP_CREATE, &openError);
    if (!archive)
    {
        if (err)
        {
            *err = openError.isEmpty()
                ? QStringLiteral("无法以写入模式打开归档")
                : openError;
        }
        reopenReadHandle();
        return false;
    }

    auto deleteEntry = [archive](const QString &entryPath) -> bool
    {
        const QByteArray entryName = zipEntryName(
            xjw::common::project::PortableProjectFormat::normalizeEntryPath(
                entryPath));
        const zip_int64_t index =
            zip_name_locate(archive, entryName.constData(), ZIP_FL_ENC_UTF_8);
        return index < 0
            || zip_delete(archive, static_cast<zip_uint64_t>(index)) >= 0;
    };

    for (const QString &entry : entriesToDelete)
    {
        if (!deleteEntry(entry))
        {
            if (err)
            {
                *err = QString::fromUtf8(zip_strerror(archive));
            }
            zip_discard(archive);
            reopenReadHandle();
            return false;
        }
    }

    const zip_int32_t method =
        compression == PlascanArchiveCompression::Deflate
        ? ZIP_CM_DEFLATE
        : ZIP_CM_STORE;
    for (const auto &entry : fileEntries)
    {
        if (!deleteEntry(entry.first))
        {
            if (err)
            {
                *err = QString::fromUtf8(zip_strerror(archive));
            }
            zip_discard(archive);
            reopenReadHandle();
            return false;
        }

        const QByteArray entryName = zipEntryName(
            xjw::common::project::PortableProjectFormat::normalizeEntryPath(
                entry.first));
        QString sourceError;
        zip_source_t *source =
            createFileSource(archive, entry.second, &sourceError);
        if (!source)
        {
            if (err)
            {
                *err = sourceError;
            }
            zip_discard(archive);
            reopenReadHandle();
            return false;
        }

        const zip_int64_t index = zip_file_add(
            archive, entryName.constData(), source, ZIP_FL_ENC_UTF_8);
        if (index < 0)
        {
            if (err)
            {
                *err = QString::fromUtf8(zip_strerror(archive));
            }
            zip_source_free(source);
            zip_discard(archive);
            reopenReadHandle();
            return false;
        }
        if (zip_set_file_compression(
                archive, static_cast<zip_uint64_t>(index), method, 0)
            < 0)
        {
            if (err)
            {
                *err = QString::fromUtf8(zip_strerror(archive));
            }
            zip_discard(archive);
            reopenReadHandle();
            return false;
        }
    }

    if (zip_close(archive) < 0)
    {
        if (err)
        {
            *err = QString::fromUtf8(zip_strerror(archive));
        }
        zip_discard(archive);
        reopenReadHandle();
        return false;
    }

    reopenReadHandle();
    return true;
#else
    Q_UNUSED(fileEntries);
    Q_UNUSED(entriesToDelete);
    Q_UNUSED(compression);
    if (err)
    {
        *err = QStringLiteral("libzip 未启用，无法更新归档");
    }
    return false;
#endif
}

bool PlascanArchive::extractEntryToFile(const QString &entryPath,
                                        const QString &destinationPath,
                                        QString *err)
{
#if defined(HAVE_LIBZIP)
    if (err)
    {
        err->clear();
    }
    if (!_impl)
    {
        if (err)
        {
            *err = QStringLiteral("归档未打开");
        }
        return false;
    }
    if (!xjw::common::project::PortableProjectFormat::isSafeEntryPath(
            entryPath))
    {
        if (err)
        {
            *err = QStringLiteral("归档条目路径不安全: %1").arg(entryPath);
        }
        return false;
    }
    if (destinationPath.trimmed().isEmpty())
    {
        if (err)
        {
            *err = QStringLiteral("提取目标路径为空");
        }
        return false;
    }

    const QFileInfo destinationInfo(destinationPath);
    if (!QDir().mkpath(destinationInfo.absolutePath()))
    {
        if (err)
        {
            *err = QStringLiteral("无法创建提取目录: %1")
                       .arg(destinationInfo.absolutePath());
        }
        return false;
    }

    zip_t *archive = static_cast<zip_t *>(_impl);
    const QByteArray entryName = zipEntryName(
        xjw::common::project::PortableProjectFormat::normalizeEntryPath(
            entryPath));
    zip_file_t *entry =
        zip_fopen(archive, entryName.constData(), ZIP_FL_ENC_UTF_8);
    if (!entry)
    {
        if (err)
        {
            *err = QStringLiteral("归档条目不存在: %1").arg(entryPath);
        }
        return false;
    }

    QSaveFile output(destinationPath);
    if (!output.open(QIODevice::WriteOnly))
    {
        zip_fclose(entry);
        if (err)
        {
            *err = QStringLiteral("无法创建提取文件 %1: %2")
                       .arg(destinationPath, output.errorString());
        }
        return false;
    }

    QByteArray buffer(1024 * 1024, Qt::Uninitialized);
    while (true)
    {
        const zip_int64_t readCount = zip_fread(
            entry, buffer.data(), static_cast<zip_uint64_t>(buffer.size()));
        if (readCount < 0)
        {
            zip_fclose(entry);
            output.cancelWriting();
            if (err)
            {
                *err = QStringLiteral("读取归档条目失败: %1").arg(entryPath);
            }
            return false;
        }
        if (readCount == 0)
        {
            break;
        }
        if (output.write(buffer.constData(), readCount) != readCount)
        {
            zip_fclose(entry);
            output.cancelWriting();
            if (err)
            {
                *err = QStringLiteral("写入提取文件失败 %1: %2")
                           .arg(destinationPath, output.errorString());
            }
            return false;
        }
    }
    zip_fclose(entry);

    if (!output.commit())
    {
        if (err)
        {
            *err = QStringLiteral("提交提取文件失败 %1: %2")
                       .arg(destinationPath, output.errorString());
        }
        return false;
    }
    return true;
#else
    Q_UNUSED(entryPath);
    Q_UNUSED(destinationPath);
    if (err)
    {
        *err = QStringLiteral("libzip 未启用，无法读取归档");
    }
    return false;
#endif
}

void PlascanArchive::closeReadHandle()
{
#if defined(HAVE_LIBZIP)
    if (_impl)
    {
        zip_t *readArchive = static_cast<zip_t *>(_impl);
        zip_close(readArchive);
        _impl = nullptr;
    }
#endif
    _valid = false;
}

void PlascanArchive::reopenReadHandle()
{
#if defined(HAVE_LIBZIP)
    zip_t *readArchive = openArchiveFile(_path, ZIP_RDONLY);
    if (readArchive)
    {
        _impl = readArchive;
        _valid = true;
    }
#endif
}
