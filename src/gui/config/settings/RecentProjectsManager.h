#pragma once

/**
 * @file RecentProjectsManager.h
 * @brief 最近打开项目列表管理器的声明文件。
 *
 * RecentProjectsManager 维护用户最近打开的 .plascan 项目路径列表，
 * 数据通过 QSettings("PlaScan", "plascan_gui") 以键 "Project/recent" 持久化。
 *
 * 功能特性：
 *   - 最多保留 10 条记录（FIFO 策略，溢出时自动删除最旧条目）；
 *   - 新添加的路径自动提升至列表首位（便于"文件 > 最近打开"菜单按最新排序）；
 *   - 支持一键清空全部记录。
 */

#include <QObject>
#include <QStringList>

/**
 * @class RecentProjectsManager
 * @brief 管理最近打开的 .plascan 项目路径列表。
 *
 * 所有读写操作均直接访问 QSettings，不在内存中缓存列表，
 * 因此每次调用均反映磁盘的最新状态，适合多窗口场景。
 */
class RecentProjectsManager : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数。
     * @param parent 父对象指针，用于 Qt 内存管理。
     */
    explicit RecentProjectsManager(QObject *parent = nullptr);

    /**
     * @brief 获取最近打开的项目路径列表（最多 10 条，最新的排在最前）。
     * @return 项目路径字符串列表；若无记录则返回空列表。
     */
    QStringList recentProjects() const;

    /**
     * @brief 将指定项目路径添加到最近打开列表的首位。
     *
     * 实现策略：
     * 1. 若 plascanPath 为空字符串则直接返回，忽略无效输入；
     * 2. 先从现有列表中移除该路径的重复项，防止列表中出现重复；
     * 3. 然后将路径插入到列表首位（prepend）；
     * 4. 若列表超过 10 条，循环删除末尾条目直至满足限制。
     *
     * @param plascanPath 需要记录的 .plascan 文件的绝对路径。
     */
    void addRecentProject(const QString &plascanPath);

    /**
     * @brief 清空全部最近打开记录，将列表重置为空。
     */
    void clearRecentProjects();
};
