/**
 * @file ProjectDialogJsonSettingBase.cpp
 * @brief ProjectDialogJsonSettingBase 的实现文件。
 *
 * 提供 project_dialog.json 的文件路径计算、JSON 读写，
 * 以及按顶层键名进行增量更新（load/save）的通用逻辑。
 */

#include "ProjectDialogJsonSettingBase.h"

#include "io/JsonObjectFile.h"
#include "project/ProjectIO.h"

#include <QDir>

// ---------------------------------------------------------------------------
// 公共 API
// ---------------------------------------------------------------------------

/**
 * @brief 设置当前项目的 .plascan 文件路径。
 *
 * 所有后续的 loadByKey()/saveByKey() 调用都将基于此路径
 * 推算出 project_dialog.json 的位置。
 *
 * @param plascanPath .plascan 文件的绝对路径。
 */
void ProjectDialogJsonSettingBase::setProjectPath(const QString &plascanPath)
{
    _plascanPath = plascanPath;
}

// ---------------------------------------------------------------------------
// 受保护方法（供派生类使用）
// ---------------------------------------------------------------------------

/**
 * @brief 返回对话框参数文件的文件名。
 *
 * 默认值为 "project_dialog.json"。
 * 所有派生 Setting 类共享同一文件，各自通过唯一顶层键名隔离数据。
 *
 * @return 文件名字符串（不含路径）。
 */
QString ProjectDialogJsonSettingBase::dialogFileName() const
{
    return QStringLiteral("project_dialog.json");
}

/**
 * @brief 计算 project_dialog.json 的完整绝对路径。
 *
 * 路径 = projectRootFromPlascan(_plascanPath) / dialogFileName()。
 * 若 .plascan 路径未设置或无法推导出项目根目录，则返回空字符串。
 *
 * @return 文件绝对路径；失败时返回空字符串。
 */
QString ProjectDialogJsonSettingBase::dialogFilePath() const
{
    if (_plascanPath.trimmed().isEmpty())
    {
        return QString();
    }
    const QString root = xjw::common::project::ProjectIO::projectRootFromPlascan(_plascanPath);
    if (root.isEmpty())
    {
        return QString();
    }
    return QDir(root).filePath(dialogFileName());
}

/**
 * @brief 按顶层键名从 project_dialog.json 加载对应的 JSON 子对象。
 *
 * 流程：打开文件 → 解析根对象 → 提取 key 对应的值。
 *
 * @param key 顶层键名（如 "aerial_triangulation"）。
 * @return 对应的 QJsonObject；键不存在或文件未找到时返回空对象。
 */
QJsonObject ProjectDialogJsonSettingBase::loadByKey(const QString &key, QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (key.isEmpty())
    {
        return QJsonObject();
    }
    const QString path = dialogFilePath();
    if (path.isEmpty())
    {
        return QJsonObject();
    }
    const xjw::common::io::JsonObjectReadResult result = xjw::common::io::readJsonObjectFile(path);
    if (!result.success)
    {
        if (errorMessage)
        {
            *errorMessage = result.errorMessage;
        }
        return QJsonObject();
    }

    return result.object.value(key).toObject();
}

/**
 * @brief 按顶层键名将 JSON 子对象写入 project_dialog.json。
 *
 * 采用 read-modify-write 策略：先读取现有根对象，再插入/覆盖
 * 指定键的值，最后将整个根对象写回文件。
 * 这样不会影响文件中其他对话框的设置数据。
 *
 * @param key   顶层键名。
 * @param value 要保存的 JSON 对象。
 * @return true 表示写入成功。
 */
bool ProjectDialogJsonSettingBase::saveByKey(const QString &key,
                                             const QJsonObject &value,
                                             QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (key.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("对话框设置键不能为空");
        }
        return false;
    }
    const QString path = dialogFilePath();
    if (path.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("未设置有效的项目路径");
        }
        return false;
    }

    const xjw::common::io::JsonObjectReadResult result = xjw::common::io::readJsonObjectFile(path);
    if (!result.success)
    {
        if (errorMessage)
        {
            *errorMessage = result.errorMessage;
        }
        return false;
    }

    QJsonObject root = result.object;
    root.insert(key, value);
    return xjw::common::io::writeJsonObjectFileAtomic(path, root, errorMessage);
}
