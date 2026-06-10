/**
 * @file LogPanel.cpp
 * @brief LogPanel 的实现文件。
 *
 * 包含 UI 布局初始化、Logger 回调注册、日志过滤显示、清空和保存等全部实现逻辑。
 * LogPanel 实现：包含级别选择控件与文本区，注册全局 Logger sink 并在 UI 中显示日志。
 */
#include "LogPanel.h"
#include "ui_LogPanel.h"

#include <QComboBox>
#include <QFile>
#include <QTextStream>
#include <QFileDialog>
#include <QDir>
#include <QIODevice>
#include <QMetaObject>
#include <QPointer>
#include <QStringList>
#include <QTextCursor>
#include <QTextEdit>
#include <QPushButton>

/**
 * @brief 构造函数：创建 UI 布局，初始化所有子控件，并注册 Logger sink。
 *
 * 布局结构：
 * ┌──────────────────────────────────────────┐
 * │ 显示等级: [下拉框]   [清空]  [保存]       │  ← topLayout
 * ├──────────────────────────────────────────┤
 * │                                          │
 * │           只读日志文本区（m_text）         │  ← QTextEdit
 * │                                          │
 * └──────────────────────────────────────────┘
 *
 * 关键连接：
 * - Logger sink → LogPanel::appendLog（invokeMethod 排队到 UI 线程）
 * - m_levelCombo::currentIndexChanged → LogPanel::onLevelChanged
 * - m_clearBtn::clicked → LogPanel::clearLogs
 * - m_saveBtn::clicked → 打开文件保存对话框 → saveLogsToFile
 *
 * @param parent 父 Widget 指针。
 */
LogPanel::LogPanel(QWidget *parent)
    : QWidget(parent)
{
    Ui::LogPanel ui;
    ui.setupUi(this);

    m_levelCombo = ui.m_levelCombo;
    m_clearBtn = ui.m_clearBtn;
    m_saveBtn = ui.m_saveBtn;
    m_text = ui.m_text;

    m_levelCombo->setCurrentIndex(static_cast<int>(m_displayLevel));
    connect(m_levelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &LogPanel::onLevelChanged);

    connect(m_clearBtn, &QPushButton::clicked, this, &LogPanel::clearLogs);

    connect(m_saveBtn, &QPushButton::clicked, this, [this]() {
        QString p = QFileDialog::getSaveFileName(
            this, tr("保存日志为"), QDir::homePath(), tr("文本文件 (*.txt);;所有文件 (*)"));
        if (!p.isEmpty()) saveLogsToFile(p);
    });

    QPointer<LogPanel> self(this);
    m_sinkId = Logger::instance()->registerSink(
        [self](const Logger::Entry &entry)
        {
            if (!self)
            {
                return;
            }

            const QString formatted = QString::fromUtf8(entry.formatted.c_str(), static_cast<int>(entry.formatted.size()));
            const int level = static_cast<int>(entry.level);
            QMetaObject::invokeMethod(
                self,
                [self, formatted, level]()
                {
                    if (self)
                    {
                        self->appendLog(formatted, level);
                    }
                },
                Qt::QueuedConnection);
        });
}

/** @brief 析构函数，无特殊资源需要释放。 */
LogPanel::~LogPanel()
{
    if (m_sinkId != 0)
    {
        Logger::instance()->unregisterSink(m_sinkId);
    }
}

/**
 * @brief 同步更新面板的最低显示等级和下拉框的选中状态。
 * @param level 新的最低显示等级。
 */
void LogPanel::setDisplayLevel(Logger::Level level)
{
    m_displayLevel = level;
    m_levelCombo->setCurrentIndex(static_cast<int>(level));
}

/**
 * @brief 响应等级下拉框选项变更：更新内部状态并通知外部监听者。
 *
 * 发出 displayLevelChanged 信号，使外部（如项目设置管理器）
 * 能将新的等级持久化到配置文件中。
 *
 * @param index 下拉框当前选中索引，直接对应 Logger::Level 枚举值。
 */
void LogPanel::onLevelChanged(int index)
{
    m_displayLevel = static_cast<Logger::Level>(index);
    emit displayLevelChanged(index); // 通知外部持久化新等级
}

/**
 * @brief 便捷方法：将一行未格式化文本以 Info 级别追加到面板。
 * @param text 要追加的文本内容。
 */
void LogPanel::append(const QString &text)
{
    appendLog(text + "\n", static_cast<int>(Logger::Info));
}

/**
 * @brief 追加一行已格式化日志到文本区（Logger 回调的 UI 线程处理槽）。
 *
 * UI 过滤策略：
 * - 仅在面板中过滤显示，磁盘上的完整日志不受影响；
 * - level < m_displayLevel 时直接返回，不追加到文本区。
 *
 * 追加方式：
 * - moveCursor(End) + insertPlainText 比 append() 更高效（不触发段落格式化）；
 * - ensureCursorVisible 保证新日志自动滚动到可视区域底部。
 *
 * @param formatted 格式化后的完整日志行。
 * @param level     日志级别整数值。
 */
void LogPanel::appendLog(const QString &formatted, int level)
{
    // UI 过滤：仅显示达到或超过最低等级的日志条目
    if (level < static_cast<int>(m_displayLevel))
        return;

    m_text->moveCursor(QTextCursor::End);    // 将光标移到末尾
    m_text->insertPlainText(formatted);       // 追加文本
    m_text->ensureCursorVisible();           // 自动滚动到最新一行
}

/**
 * @brief 清空面板文本区内容，并尝试截断磁盘日志文件。
 *
 * 截断磁盘文件的目的：防止调用 loadFromLogFile 时将旧日志重新读回，
 * 使清空操作在视觉上和文件层面保持一致。
 * 若截断失败（权限不足等），UI 清空仍会成功，磁盘文件保持不变（静默忽略）。
 */
void LogPanel::clearLogs()
{
    // 清空 UI 文本区
    if (m_text) m_text->clear();

    Logger::instance()->clearLogFile();
}

/**
 * @brief 将面板当前显示的全部日志内容另存为指定路径的文本文件。
 *
 * @param filePath 目标文件的绝对路径。
 * @return         文件打开并写入成功返回 true；否则返回 false。
 */
bool LogPanel::saveLogsToFile(const QString &filePath)
{
    QFile f(filePath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) return false;
    QTextStream ts(&f);
    ts << m_text->toPlainText();
    f.close();
    return true;
}

/**
 * @brief 从磁盘日志文件加载历史记录并填充到面板中。
 *
 * 读取策略：
 * 1. 如果日志文件大小 ≤ 2 MB，则完整读取；
 * 2. 如果超过 2 MB，则 seek 到末尾前 2 MB 处，跳过第一行（可能是不完整行），
 *    再读取剩余内容，避免首行数据残缺。
 *
 * 过滤策略：
 * - 按行解析，查找 "] [LEVEL]" 模式提取日志级别；
 * - 仅追加级别 ≥ m_displayLevel 的行；
 * - 最终批量 insertPlainText，减少 UI 重绘次数。
 *
 * 使用场景：打开已有项目时，将上次运行留下的历史日志回填到面板。
 */
void LogPanel::loadFromLogFile()
{
    const QString logPath = QString::fromStdString(Logger::instance()->logFilePath());

    QFile f(logPath);
    if (!f.exists()) return;
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;

    // 限制读取大小：超过 2 MB 时只读末尾部分，避免 UI 卡顿
    const qint64 limit = 2 * 1024 * 1024; // 2 MB
    qint64 size = f.size();
    QByteArray data;
    if (size <= limit) {
        // 文件较小，完整读取
        data = f.readAll();
    } else {
        // 文件过大，跳转到 (size - limit) 处，丢弃首个不完整行后读取剩余
        f.seek(size - limit);
        // 读取并丢弃当前位置到第一个换行符之间的内容（可能是半截行）
        QByteArray prefix = f.readLine(); Q_UNUSED(prefix);
        data = f.readAll();
    }
    f.close();

    if (!data.isEmpty()) {
        // 将字节数据解码为 UTF-8 字符串，按换行符拆分为行列表
        QString content = QString::fromUtf8(data);
        QStringList lines = content.split('\n', Qt::KeepEmptyParts);
        QString toInsert;

        for (const QString &ln : lines) {
            if (ln.trimmed().isEmpty()) continue; // 跳过空行

            // 解析日志级别：期望格式 "[ISO时间] [LEVEL] message"
            // 在 "] [" 处定位级别字段的起始位置
            int p = ln.indexOf("] [");
            int lvl = static_cast<int>(Logger::Info); // 默认视为 Info 级别
            if (p != -1) {
                int start = p + 3; // 跳过 "] ["
                int end = ln.indexOf(']', start);
                if (end != -1 && end > start) {
                    // 提取级别字符串并映射到枚举值
                    QString lstr = ln.mid(start, end - start).trimmed();
                    if      (lstr == QLatin1String("DEBUG")) lvl = static_cast<int>(Logger::Debug);
                    else if (lstr == QLatin1String("INFO"))  lvl = static_cast<int>(Logger::Info);
                    else if (lstr == QLatin1String("WARN"))  lvl = static_cast<int>(Logger::Warn);
                    else if (lstr == QLatin1String("ERROR")) lvl = static_cast<int>(Logger::Error);
                }
            }

            // 按当前显示等级过滤：只追加达到或超过阈值的行
            if (lvl >= static_cast<int>(m_displayLevel)) {
                toInsert.append(ln);
                toInsert.append('\n');
            }
        }

        // 批量插入，减少多次单行插入带来的 UI 重绘开销
        if (!toInsert.isEmpty()) {
            m_text->moveCursor(QTextCursor::End);
            m_text->insertPlainText(toInsert);
            m_text->ensureCursorVisible();
        }
    }
}
