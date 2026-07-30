#include "ProjectFileIntegration.h"

#include <QDir>
#include <QFileInfo>

#include <string>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#endif

namespace xjw::gui::platform
{
namespace
{
QString unquote(QString value)
{
    value = value.trimmed();
    if (value.size() >= 2 && value.front() == QLatin1Char('"') && value.back() == QLatin1Char('"'))
    {
        value = value.mid(1, value.size() - 2);
    }
    return value;
}

#ifdef Q_OS_WIN
bool setRegistryString(const wchar_t *subkey,
                       const wchar_t *valueName,
                       const QString &value,
                       bool *changed,
                       QString *errorMessage)
{
    HKEY key = nullptr;
    const LSTATUS create_status = RegCreateKeyExW(HKEY_CURRENT_USER,
                                                   subkey,
                                                   0,
                                                   nullptr,
                                                   REG_OPTION_NON_VOLATILE,
                                                   KEY_QUERY_VALUE | KEY_SET_VALUE,
                                                   nullptr,
                                                   &key,
                                                   nullptr);
    if (create_status != ERROR_SUCCESS)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建注册表项 %1，错误码 %2")
                                .arg(QString::fromWCharArray(subkey))
                                .arg(create_status);
        }
        return false;
    }

    const std::wstring expected = value.toStdWString();
    DWORD type = 0;
    DWORD byte_count = 0;
    LSTATUS query_status = RegQueryValueExW(key, valueName, nullptr, &type, nullptr, &byte_count);
    bool needs_write = query_status != ERROR_SUCCESS || type != REG_SZ;
    if (!needs_write)
    {
        std::wstring current(byte_count / sizeof(wchar_t), L'\0');
        query_status = RegQueryValueExW(key,
                                       valueName,
                                       nullptr,
                                       &type,
                                       reinterpret_cast<LPBYTE>(current.data()),
                                       &byte_count);
        if (query_status == ERROR_SUCCESS)
        {
            current.resize(byte_count / sizeof(wchar_t));
            while (!current.empty() && current.back() == L'\0')
            {
                current.pop_back();
            }
            needs_write = current != expected;
        }
        else
        {
            needs_write = true;
        }
    }

    if (needs_write)
    {
        const DWORD value_bytes = static_cast<DWORD>((expected.size() + 1) * sizeof(wchar_t));
        const LSTATUS write_status = RegSetValueExW(key,
                                                    valueName,
                                                    0,
                                                    REG_SZ,
                                                    reinterpret_cast<const BYTE *>(expected.c_str()),
                                                    value_bytes);
        if (write_status != ERROR_SUCCESS)
        {
            RegCloseKey(key);
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法写入注册表项 %1，错误码 %2")
                                    .arg(QString::fromWCharArray(subkey))
                                    .arg(write_status);
            }
            return false;
        }
        if (changed)
        {
            *changed = true;
        }
    }

    RegCloseKey(key);
    return true;
}
#endif
} // namespace

QString startupProjectPath(const QStringList &arguments)
{
    for (qsizetype index = 1; index < arguments.size(); ++index)
    {
        const QString argument = unquote(arguments.at(index));
        if (argument.isEmpty() || argument.startsWith(QLatin1Char('-')))
        {
            continue;
        }

        const QFileInfo file_info(argument);
        if (file_info.suffix().compare(QStringLiteral("plascan"), Qt::CaseInsensitive) != 0)
        {
            continue;
        }
        return QDir::cleanPath(file_info.absoluteFilePath());
    }
    return {};
}

QString projectOpenCommand(const QString &executablePath)
{
    const QString native_path = QDir::toNativeSeparators(QFileInfo(executablePath).absoluteFilePath());
    return QStringLiteral("\"%1\" \"%2\"").arg(native_path, QStringLiteral("%1"));
}

FileAssociationResult ensureProjectFileAssociation(const QString &executablePath)
{
    FileAssociationResult result;
#ifdef Q_OS_WIN
    const QFileInfo executable_info(executablePath);
    if (!executable_info.exists() || !executable_info.isFile())
    {
        result.success = false;
        result.errorMessage = QStringLiteral("PlaScan 可执行文件不存在: %1")
                                  .arg(executable_info.absoluteFilePath());
        return result;
    }

    constexpr wchar_t extension_key[] = L"Software\\Classes\\.plascan";
    constexpr wchar_t project_key[] = L"Software\\Classes\\PlaScan.Project";
    constexpr wchar_t icon_key[] = L"Software\\Classes\\PlaScan.Project\\DefaultIcon";
    constexpr wchar_t command_key[] = L"Software\\Classes\\PlaScan.Project\\shell\\open\\command";

    const QString native_executable = QDir::toNativeSeparators(executable_info.absoluteFilePath());
    const QString icon_value = QStringLiteral("\"%1\",0").arg(native_executable);
    const QString command_value = projectOpenCommand(native_executable);

    if (!setRegistryString(extension_key, nullptr, QStringLiteral("PlaScan.Project"),
                           &result.changed, &result.errorMessage)
        || !setRegistryString(project_key, nullptr, QStringLiteral("PlaScan 项目"),
                              &result.changed, &result.errorMessage)
        || !setRegistryString(icon_key, nullptr, icon_value,
                              &result.changed, &result.errorMessage)
        || !setRegistryString(command_key, nullptr, command_value,
                              &result.changed, &result.errorMessage))
    {
        result.success = false;
        return result;
    }

    if (result.changed)
    {
        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
    }
#else
    Q_UNUSED(executablePath);
#endif
    return result;
}

} // namespace xjw::gui::platform
