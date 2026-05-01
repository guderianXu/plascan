#pragma once

/**
 * @file FileDialogStateManager.h
 * @brief 文件对话框最近目录状态管理器的声明文件。
 *
 * FileDialogStateManager 为应用中各处打开/保存文件对话框提供"记忆"功能：
 * 当用户在某个对话框最终选择目录后，调用 setLastDir 将其持久化；
 * 下次打开同一类型对话框时，调用 lastDir 获取上次目录并作为默认起始路径，
 * 从而避免用户每次都要从根目录重新导航。
 *
 * 每个对话框类型用一个字符串 key 加以区分（例如 "importPhoto"、"exportResult"）。
 * 数据通过 QSettings 以 "Dialogs/lastDir/<key>" 的格式持久化到系统注册表/INI 文件。
 */

#include <QObject>
#include <QString>

/**
 * @class FileDialogStateManager
 * @brief 管理各文件对话框所记忆的上次浏览目录。
 *
 * 使用方式：
 * @code
 *   // 获取上次目录（对话框打开前）
 *   QString dir = appConfig->fileDialogs()->lastDir("importPhoto");
 *   // 用户选择文件后，记住新目录
 *   appConfig->fileDialogs()->setLastDir("importPhoto", QFileInfo(selectedFile).absolutePath());
 * @endcode
 */
class FileDialogStateManager : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数。
     * @param parent 父对象指针，用于 Qt 内存管理。
     */
    explicit FileDialogStateManager(QObject *parent = nullptr);

    /**
     * @brief 获取指定 key 对应对话框的上次浏览目录。
     * @param key  对话框类型标识符；空字符串时使用 "default" 键。
     * @return     上次记录的目录路径；若无记录则返回 QDir::homePath()。
     */
    QString lastDir(const QString &key) const;

    /**
     * @brief 记录指定 key 对应对话框的最近浏览目录。
     * @param key  对话框类型标识符；空字符串时使用 "default" 键。
     * @param dir  需要持久化的目录路径。
     */
    void setLastDir(const QString &key, const QString &dir);
};
