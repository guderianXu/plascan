#include "PlascanArchive.h"

#include <QByteArray>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QStringList>

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

#endif

} // namespace

PlascanArchive::PlascanArchive(const QString &path)
    : _path(path)
{
#if defined(HAVE_LIBZIP)
    zip_t *za = openArchiveFile(path, ZIP_RDONLY);
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
    if (!_impl)
    {
        if (err) *err = QStringLiteral("libzip not available or archive not opened");
        return data;
    }
    zip_t *za = static_cast<zip_t*>(_impl);
    const QByteArray entryName = zipEntryName(entryPath);
    zip_file_t *zf = zip_fopen(za, entryName.constData(), ZIP_FL_ENC_UTF_8);
    if (!zf)
    {
        if (err) *err = QStringLiteral("entry not found");
        return data;
    }
    zip_stat_t st;
    if (zip_stat(za, entryName.constData(), ZIP_FL_ENC_UTF_8, &st) == 0)
    {
        data.resize(st.size);
        zip_int64_t r = zip_fread(zf, data.data(), st.size);
        zip_fclose(zf);
        if (r < 0)
        {
            if (err) *err = QStringLiteral("read error");
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

bool PlascanArchive::createArchive(const QString &path,
                                   const QByteArray &manifestJson,
                                   const QByteArray &projectJson,
                                   QString *err)
{
#if defined(HAVE_LIBZIP)
    // 使用 libzip 创建 ZIP 容器并加入两个初始条目
    QString openError;
    zip_t *za = openArchiveFile(path, ZIP_CREATE | ZIP_TRUNCATE, &openError);
    if (!za)
    {
        if (err) *err = QStringLiteral("无法创建归档: %1").arg(openError);
        return false;
    }

    // manifest.json
    zip_source_t *s1 = zip_source_buffer(za, manifestJson.constData(), (zip_uint64_t)manifestJson.size(), 0);
    if (!s1 || zip_file_add(za, "manifest.json", s1, ZIP_FL_ENC_UTF_8) < 0)
    {
        if (s1) zip_source_free(s1);
        if (err) *err = QString::fromUtf8(zip_strerror(za));
        zip_close(za);
        return false;
    }

    // project.json
    zip_source_t *s2 = zip_source_buffer(za, projectJson.constData(), (zip_uint64_t)projectJson.size(), 0);
    if (!s2 || zip_file_add(za, "project.json", s2, ZIP_FL_ENC_UTF_8) < 0)
    {
        if (s2) zip_source_free(s2);
        if (err) *err = QString::fromUtf8(zip_strerror(za));
        zip_close(za);
        return false;
    }

    if (zip_close(za) < 0)
    {
        if (err) *err = QString::fromUtf8(zip_strerror(za));
        return false;
    }
    return true;
#else
    Q_UNUSED(path);
    Q_UNUSED(manifestJson);
    Q_UNUSED(projectJson);
    if (err) *err = QStringLiteral("libzip 未在构建时启用，无法创建 .plascan 文件");
    return false;
#endif
}

bool PlascanArchive::writeEntry(const QString &entryPath,
                                const QByteArray &data,
                                QString *err)
{
#if defined(HAVE_LIBZIP)
    if (_path.isEmpty())
    {
        if (err)
            *err = QStringLiteral("归档路径为空");
        return false;
    }

    // Windows 不允许在同一归档仍被本对象的只读句柄占用时，用 libzip
    // 在 zip_close() 阶段重命名临时文件覆盖原归档。
    if (_impl)
    {
        zip_t *readArchive = static_cast<zip_t*>(_impl);
        zip_close(readArchive);
        _impl = nullptr;
        _valid = false;
    }

    // 以可写方式打开（如果不存在则创建）
    QString openError;
    zip_t *za = openArchiveFile(_path, ZIP_CREATE, &openError);
    if (!za)
    {
        if (err)
            *err = QStringLiteral("无法打开归档以写入: %1").arg(openError);
        return false;
    }

    // 如果条目已经存在，则先移除（确保替换效果）
    const QByteArray entryName = zipEntryName(entryPath);
    const zip_int64_t idx = zip_name_locate(za, entryName.constData(), ZIP_FL_ENC_UTF_8);
    if (idx >= 0)
    {
        if (zip_delete(za, static_cast<zip_uint64_t>(idx)) < 0)
        {
            if (err)
                *err = QString::fromUtf8(zip_strerror(za));
            zip_close(za);
            return false;
        }
    }

    // 创建源并添加
    zip_source_t *src = zip_source_buffer(za, data.constData(), (zip_uint64_t)data.size(), 0);
    if (!src)
    {
        if (err)
            *err = QString::fromUtf8(zip_strerror(za));
        zip_close(za);
        return false;
    }

    if (zip_file_add(za, entryName.constData(), src, ZIP_FL_ENC_UTF_8) < 0)
    {
        if (err)
            *err = QString::fromUtf8(zip_strerror(za));
        zip_source_free(src);
        zip_close(za);
        return false;
    }

    if (zip_close(za) < 0)
    {
        if (err)
            *err = QString::fromUtf8(zip_strerror(za));
        zip_discard(za);
        return false;
    }

    zip_t *readArchive = openArchiveFile(_path, ZIP_RDONLY);
    if (readArchive)
    {
        _impl = readArchive;
        _valid = true;
    }

    return true;
#else
    Q_UNUSED(entryPath);
    Q_UNUSED(data);
    if (err)
        *err = QStringLiteral("libzip 未启用，无法写入归档");
    return false;
#endif
}
