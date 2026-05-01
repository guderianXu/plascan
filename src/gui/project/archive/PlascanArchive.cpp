#include "PlascanArchive.h"

#include <QStringList>
#include <QFile>
#include <QByteArray>
#include <QDebug>

// libzip headers if available
#define HAVE_LIBZIP
#if defined(HAVE_LIBZIP)
#include <zip.h>
#endif

PlascanArchive::PlascanArchive(const QString &path)
    : m_path(path)
{
#if defined(HAVE_LIBZIP)
    int err = 0;
    zip_t *za = zip_open(path.toLocal8Bit().constData(), ZIP_RDONLY, &err);
    if (za) {
        m_impl = za;
        m_valid = true;
    } else {
        m_impl = nullptr;
        m_valid = false;
    }
#else
    Q_UNUSED(path);
    m_valid = false;
#endif
}

PlascanArchive::~PlascanArchive()
{
#if defined(HAVE_LIBZIP)
    if (m_impl) {
        zip_t *za = static_cast<zip_t*>(m_impl);
        zip_close(za);
        m_impl = nullptr;
    }
#endif
}

bool PlascanArchive::isValid() const
{
    return m_valid;
}

QVector<QString> PlascanArchive::listEntries()
{
    QVector<QString> out;
#if defined(HAVE_LIBZIP)
    if (!m_impl)
        return out;
    zip_t *za = static_cast<zip_t*>(m_impl);
    zip_int64_t n = zip_get_num_entries(za, 0);
    for (zip_uint64_t i = 0; i < (zip_uint64_t)n; ++i) {
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
    if (!m_impl) {
        if (err) *err = QStringLiteral("libzip not available or archive not opened");
        return data;
    }
    zip_t *za = static_cast<zip_t*>(m_impl);
    zip_file_t *zf = zip_fopen(za, entryPath.toLocal8Bit().constData(), 0);
    if (!zf) {
        if (err) *err = QStringLiteral("entry not found");
        return data;
    }
    zip_stat_t st;
    if (zip_stat(za, entryPath.toLocal8Bit().constData(), 0, &st) == 0) {
        data.resize(st.size);
        zip_int64_t r = zip_fread(zf, data.data(), st.size);
        zip_fclose(zf);
        if (r < 0) {
            if (err) *err = QStringLiteral("read error");
            data.clear();
        }
    } else {
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
    int errorp = 0;
    zip_t *za = zip_open(path.toLocal8Bit().constData(), ZIP_CREATE | ZIP_TRUNCATE, &errorp);
    if (!za) {
        if (err) *err = QStringLiteral("无法创建归档（zip_open 失败）");
        return false;
    }

    // manifest.json
    zip_source_t *s1 = zip_source_buffer(za, manifestJson.constData(), (zip_uint64_t)manifestJson.size(), 0);
    if (!s1 || zip_file_add(za, "manifest.json", s1, ZIP_FL_ENC_UTF_8) < 0) {
        if (s1) zip_source_free(s1);
        if (err) *err = QString::fromUtf8(zip_strerror(za));
        zip_close(za);
        return false;
    }

    // project.json
    zip_source_t *s2 = zip_source_buffer(za, projectJson.constData(), (zip_uint64_t)projectJson.size(), 0);
    if (!s2 || zip_file_add(za, "project.json", s2, ZIP_FL_ENC_UTF_8) < 0) {
        if (s2) zip_source_free(s2);
        if (err) *err = QString::fromUtf8(zip_strerror(za));
        zip_close(za);
        return false;
    }

    if (zip_close(za) < 0) {
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
    if (m_path.isEmpty())
    {
        if (err)
            *err = QStringLiteral("归档路径为空");
        return false;
    }

    int errorp = 0;
    // 以可写方式打开（如果不存在则创建）
    zip_t *za = zip_open(m_path.toLocal8Bit().constData(), ZIP_CREATE, &errorp);
    if (!za)
    {
        if (err)
            *err = QStringLiteral("无法打开归档以写入");
        return false;
    }

    // 如果条目已经存在，则先移除（确保替换效果）
    zip_uint64_t idx = zip_name_locate(za, entryPath.toLocal8Bit().constData(), 0);
    if (idx != ZIP_UINT64_MAX)
    {
        if (zip_delete(za, idx) < 0)
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

    if (zip_file_add(za, entryPath.toLocal8Bit().constData(), src, ZIP_FL_ENC_UTF_8) < 0)
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
        return false;
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
