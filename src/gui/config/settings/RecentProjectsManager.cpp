/**
 * @file RecentProjectsManager.cpp
 * @brief RecentProjectsManager 的实现文件。
 *
 * 所有数据均通过 QSettings("PlaScan", "plascan_gui") 以键
 * "Project/recent" 持久化；每次操作都直接读写 QSettings，
 * 不在内存中预缓存，以保证多处调用时数据的一致性。
 */
#include "RecentProjectsManager.h"

#include "GuiSettingsStore.h"

#include <QDir>
#include <QFileInfo>
#include <QSet>

namespace
{

QString normalizedProjectPath(const QString &path)
{
    const QString trimmed_path = path.trimmed();
    if (trimmed_path.isEmpty())
    {
        return {};
    }

    const QFileInfo file_info(trimmed_path);
    if (!file_info.exists() || !file_info.isFile())
    {
        return {};
    }

    const QString canonical_path = file_info.canonicalFilePath();
    if (!canonical_path.isEmpty())
    {
        return QDir::cleanPath(canonical_path);
    }

    return QDir::cleanPath(file_info.absoluteFilePath());
}

QString projectPathKey(const QString &path)
{
#ifdef Q_OS_WIN
    return path.toCaseFolded();
#else
    return path;
#endif
}

} // namespace

/**
 * @brief 构造函数，无额外初始化逻辑。
 */
RecentProjectsManager::RecentProjectsManager(QObject *parent)
    : QObject(parent)
{
}

/**
 * @brief 获取最近打开的项目路径列表。
 *
 * 直接从 QSettings 读取，确保反映磁盘的最新状态。
 *
 * @return 路径字符串列表（最多 10 条），最新的排在最前；无记录时返回空列表。
 */
QStringList RecentProjectsManager::recentProjects() const
{
    QSettings settings = xjw::gui::settings::createGuiSettings();
    const QStringList stored_paths = settings.value(QStringLiteral("Project/recent"), QStringList()).toStringList();

    QSet<QString> seen_paths;
    QStringList valid_paths;
    for (const QString &stored_path : stored_paths)
    {
        const QString normalized_path = normalizedProjectPath(stored_path);
        if (normalized_path.isEmpty())
        {
            continue;
        }

        const QString path_key = projectPathKey(normalized_path);
        if (seen_paths.contains(path_key))
        {
            continue;
        }

        seen_paths.insert(path_key);
        valid_paths.append(normalized_path);
        if (valid_paths.size() == 10)
        {
            break;
        }
    }

    if (valid_paths != stored_paths)
    {
        settings.setValue(QStringLiteral("Project/recent"), valid_paths);
    }

    return valid_paths;
}

/**
 * @brief 将指定路径添加到最近打开列表的首位并持久化。
 *
 * @param plascanPath .plascan 项目文件的绝对路径；空字符串时直接忽略。
 */
void RecentProjectsManager::addRecentProject(const QString &plascanPath)
{
    const QString normalized_path = normalizedProjectPath(plascanPath);
    if (normalized_path.isEmpty())
    {
        return;
    }

    QSettings settings = xjw::gui::settings::createGuiSettings();
    QStringList list = recentProjects();

    const QString path_key = projectPathKey(normalized_path);
    for (int index = list.size() - 1; index >= 0; --index)
    {
        if (projectPathKey(list.at(index)) == path_key)
        {
            list.removeAt(index);
        }
    }

    list.prepend(normalized_path);

    // 保持列表条目数不超过 10，超出时从尾部删除最旧记录
    while (list.size() > 10)
    {
        list.removeLast();
    }

    settings.setValue(QStringLiteral("Project/recent"), list);
}

/**
 * @brief 清空全部最近打开记录。
 *
 * 将 "Project/recent" 键写为空列表，下次读取时返回空。
 */
void RecentProjectsManager::clearRecentProjects()
{
    QSettings settings = xjw::gui::settings::createGuiSettings();
    settings.setValue(QStringLiteral("Project/recent"), QStringList());
}
