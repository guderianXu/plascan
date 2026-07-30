#pragma once

/**
 * @file ProjectDialogJsonSettingBase.h
 * @brief 项目对话框 JSON 记忆化基类声明。
 *
 * ProjectDialogJsonSettingBase 为所有需要在项目级别持久化对话框参数的
 * Setting 类提供统一的读写基础设施。
 *
 * 设计思路：
 *   - 每个对话框对应一个派生 Setting 类，各自拥有一个唯一的顶层键名（dialogKey）。
 *   - 所有对话框参数集中写入同一个 JSON 文件（默认 project_dialog.json），
 *     放置在 .plascan 文件同级目录中，随项目一起管理。
 *   - 基类负责文件的打开/读取/写入/路径计算等底层操作；
 *     派生类只需实现 load()/save()/dialogKey() 即可完成记忆化。
 *
 * 数据文件结构示例：
 * @code{.json}
 * {
 *   "aerial_triangulation": { ... },
 *   "bundle_adjust":        { ... }
 * }
 * @endcode
 *
 * @see DialogSettingStore  — 通用 key 驱动的对话框记忆化实现
 */

#include <QJsonObject>
#include <QString>

/**
 * @class ProjectDialogJsonSettingBase
 * @brief 项目对话框参数 JSON 持久化基类。
 *
 * 本类 **不是** QObject 子类，因此无法独立使用信号槽；
 * 派生类可通过多继承同时继承 QObject 和本类来获取信号槽能力。
 */
class ProjectDialogJsonSettingBase
{
public:
    /** @brief 虚析构函数，确保派生类正确销毁。 */
    virtual ~ProjectDialogJsonSettingBase() = default;

    /**
     * @brief 设置当前项目的 .plascan 文件路径。
     *
     * 该路径决定了 project_dialog.json 的存放位置
     * （与 .plascan 同目录，即项目根目录）。
     *
     * @param plascanPath .plascan 文件的绝对路径。
     */
    void setProjectPath(const QString &plascanPath);

protected:
    /**
     * @brief 返回对话框参数文件的文件名（不含路径）。
     *
     * 默认实现返回 "project_dialog.json"。
     * 若需将参数写入不同文件可在派生类中覆盖此方法。
     *
     * @return 对话框参数文件名。
     */
    virtual QString dialogFileName() const;

    /**
     * @brief 按键名加载对应的 JSON 子对象。
     *
     * 从 project_dialog.json 中读取顶层键 @p key 对应的对象。
     * 若文件不存在、路径未设置或键不存在，均返回空 QJsonObject。
     *
     * @param key 顶层键名（如 "aerial_triangulation"）。
     * @return 对应的 JSON 对象，或空对象。
     */
    QJsonObject loadByKey(const QString &key, QString *errorMessage = nullptr) const;

    /**
     * @brief 按键名保存 JSON 子对象（只更新指定键，不影响其他键）。
     *
     * 读取现有 project_dialog.json → 插入/覆盖 @p key → 整体写回文件。
     *
     * @param key   顶层键名。
     * @param value 要保存的 JSON 对象。
     * @return true 表示写入成功；false 表示路径无效或写入失败。
     */
    bool saveByKey(const QString &key, const QJsonObject &value, QString *errorMessage = nullptr) const;

private:
    /**
     * @brief 计算 project_dialog.json 的完整绝对路径。
     * @return 文件绝对路径；若 _plascanPath 未设置则返回空字符串。
     */
    QString dialogFilePath() const;

    QString _plascanPath; ///< 当前项目 .plascan 文件的绝对路径
};
