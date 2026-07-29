/**
 * @file ProjectDialogJsonSettingBase.cpp
 * @brief ProjectDialogJsonSettingBase 的实现文件。
 *
 * 提供 project_dialog.json 的文件路径计算、JSON 读写，
 * 以及按顶层键名进行增量更新（load/save）的通用逻辑。
 */

#include "ProjectDialogJsonSettingBase.h"

#include "project/ProjectIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSaveFile>

namespace {

/**
 * @brief 确保 QJsonDocument 可以安全转换为 QJsonObject。
 *
 * 若文档本身就是对象类型则直接返回，否则返回空对象。
 * 用于防止 JSON 文件根节点为数组或其他非对象类型时导致崩溃。
 *
 * @param doc 待检查的 QJsonDocument。
 * @return QJsonObject 或空对象。
 */
QJsonObject ensureObject(const QJsonDocument &doc)
{
    if (doc.isObject())
    {
        return doc.object();
    }
    return QJsonObject();
}

} // anonymous namespace

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

// ---------------------------------------------------------------------------
// 私有方法
// ---------------------------------------------------------------------------

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
 * @brief 读取指定路径的 JSON 文件并解析为 QJsonObject。
 *
 * 若文件不存在、无法打开或解析失败，均不产生错误日志，
 * 仅静默返回空对象（适用于首次使用时文件尚未创建的场景）。
 *
 * @param path JSON 文件的绝对路径。
 * @return 解析后的 QJsonObject；失败时返回空对象。
 */
QJsonObject ProjectDialogJsonSettingBase::readJsonFile(const QString &path)
{
    if (path.isEmpty() || !QFileInfo::exists(path))
    {
        return QJsonObject();
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return QJsonObject();
    }
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    return ensureObject(doc);
}

/**
 * @brief 将 JSON 对象以缩进格式写入指定路径的文件。
 *
 * 使用 Truncate 模式打开文件，确保旧内容被完全替换。
 *
 * @param path JSON 文件的绝对路径。
 * @param root 要写入的完整 JSON 根对象。
 * @return true 表示写入成功；false 表示路径为空或文件打开失败。
 */
bool ProjectDialogJsonSettingBase::writeJsonFile(const QString &path, const QJsonObject &root)
{
    if (path.isEmpty())
    {
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        return false;
    }
    const QJsonDocument doc(root);
    const QByteArray bytes = doc.toJson(QJsonDocument::Indented);
    return file.write(bytes) == bytes.size() && file.commit();
}

/**
 * @brief 按顶层键名从 project_dialog.json 加载对应的 JSON 子对象。
 *
 * 流程：打开文件 → 解析根对象 → 提取 key 对应的值。
 *
 * @param key 顶层键名（如 "aerial_triangulation"）。
 * @return 对应的 QJsonObject；键不存在或文件未找到时返回空对象。
 */
QJsonObject ProjectDialogJsonSettingBase::loadByKey(const QString &key) const
{
    if (key.isEmpty())
    {
        return QJsonObject();
    }
    const QString path = dialogFilePath();
    if (path.isEmpty())
    {
        return QJsonObject();
    }
    const QJsonObject root = readJsonFile(path);
    return root.value(key).toObject();
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
bool ProjectDialogJsonSettingBase::saveByKey(const QString &key, const QJsonObject &value) const
{
    if (key.isEmpty())
    {
        return false;
    }
    const QString path = dialogFilePath();
    if (path.isEmpty())
    {
        return false;
    }
    QJsonObject root = readJsonFile(path);
    root.insert(key, value);
    return writeJsonFile(path, root);
}
