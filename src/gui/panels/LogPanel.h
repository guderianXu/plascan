#pragma once

/**
 * @file LogPanel.h
 * @brief 日志显示面板的声明文件。
 *
 * LogPanel 是一个 Qt Widget，用于在 GUI 中实时展示来自 Logger 的日志信息。
 * 它向全局 Logger 注册回调，并根据用户在下拉框中设置的最低显示等级
 * 对日志进行 UI 层过滤（不影响磁盘日志文件的完整记录）。
 *
 * 主要布局：
 * - 顶部工具栏：日志等级下拉框、清空按钮、保存按钮；
 * - 主体区域  ：只读 QTextEdit，实时追加显示日志文本。
 *
 * 线程安全说明：
 * LogPanel 通过 QMetaObject::invokeMethod 将日志回调排队到主线程，
 * 即使 Logger 在工作线程中调用 log()，UI 更新也会在主线程处理，
 * 从而保证 UI 操作的线程安全。
 */

#include <QWidget>
#include <QTextEdit>
#include <QComboBox>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include "Logger.h"

/**
 * @class LogPanel
 * @brief 日志显示面板，包含等级过滤下拉框与可滚动日志文本区。
 *
 * 注册 Logger sink，按用户设定的最低显示等级过滤后追加到文本区。
 * 提供清空面板和将日志另存为文件的操作。
 */
class LogPanel : public QWidget
{
    Q_OBJECT
public:
    /**
    * @brief 构造函数：初始化 UI 控件并注册 Logger 回调。
     * @param parent 父 Widget 指针，用于 Qt 内存管理。
     */
    explicit LogPanel(QWidget *parent = nullptr);

    /** @brief 析构函数，无特殊资源需要释放（子控件由 Qt 对象树管理）。 */
    ~LogPanel() override;

    /**
     * @brief 设置面板当前显示的最低日志等级。
     *
     * 低于该等级的日志条目将不会在面板中显示（但仍会写入磁盘）。
     * 同时更新等级下拉框的选中项，保持 UI 与内部状态一致。
     *
     * @param level 最低显示等级，例如 Logger::Info 表示只显示 Info 及以上。
     */
    void setDisplayLevel(Logger::Level level);

    /**
     * @brief 便捷方法：将一行未格式化的文本追加到面板（视为 Info 级别）。
     *
     * 主要供不经过 Logger 但需要在面板中显示信息的场景使用。
     *
     * @param text 要追加的文本内容（无需包含时间戳或级别前缀）。
     */
    void append(const QString &text);

    /**
     * @brief 从磁盘日志文件加载历史日志并填充到面板中。
     *
     * 适用于打开已有项目时将历史日志回填到面板的场景。
     * 若日志文件超过 2 MB，仅读取最后 2 MB 的内容，避免加载过慢。
     * 加载过程中会根据当前显示等级过滤各行，不显示低于阈值的历史记录。
     */
    void loadFromLogFile();

public slots:
    /**
    * @brief 追加一行已格式化的日志到文本区（由 Logger sink 回调触发）。
     *
     * 根据 level 与 m_displayLevel 比较进行过滤；
     * 通过 moveCursor + insertPlainText 追加到末尾并自动滚动。
     *
     * @param formatted 格式化后的完整日志行（含时间戳、级别与末尾换行符）。
     * @param level     日志级别整数值（对应 Logger::Level）。
     */
    void appendLog(const QString &formatted, int level);

    /**
     * @brief 清空面板内的所有日志文本，同时截断磁盘日志文件为空。
     *
     * 截断磁盘文件是为了防止下次调用 loadFromLogFile 时将旧日志重新读回。
     * 若磁盘文件截断失败（权限不足等），UI 清空操作仍会成功，忽略文件错误。
     */
    void clearLogs();

    /**
     * @brief 将面板当前显示的全部日志内容另存为指定文件。
     *
     * @param filePath 目标文件的绝对路径。
     * @return         保存成功返回 true；文件创建失败返回 false。
     */
    bool saveLogsToFile(const QString &filePath);

private slots:
    /**
     * @brief 响应等级下拉框的选项变更，更新 m_displayLevel 并发出信号。
     * @param index 下拉框当前选中的索引（对应 Logger::Level 枚举值）。
     */
    void onLevelChanged(int index);

signals:
    /**
     * @brief 用户在 UI 中更改日志显示等级时发出，供外部（如项目设置）持久化。
     * @param level 新的显示等级整数值（对应 Logger::Level）。
     */
    void displayLevelChanged(int level);

private:
    /** @brief 只读文本区，用于显示所有满足过滤条件的日志条目。 */
    QTextEdit   *m_text{nullptr};

    /** @brief 日志等级过滤下拉框（Debug / Info / Warn / Error）。 */
    QComboBox   *m_levelCombo{nullptr};

    /** @brief 清空面板按钮。 */
    QPushButton *m_clearBtn{nullptr};

    /** @brief 将日志另存为文件的按钮。 */
    QPushButton *m_saveBtn{nullptr};

    /** @brief 当前 UI 显示的最低日志等级，低于此等级的日志不在面板中显示。 */
    Logger::Level m_displayLevel{Logger::Debug};

    /** @brief 注册到全局 Logger 的 sink 令牌，析构时用于注销。 */
    int m_sinkId{0};
};
