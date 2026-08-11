/**
 * @file LogPanel.cpp
 * @brief LogPanel 的实现文件。
 *
 * 包含 UI 布局初始化、Logger 回调注册、完整输出显示、清空和保存等实现逻辑。
 */
#include "LogPanel.h"
#include "ui_LogPanel.h"

#include <QFile>
#include <QFontDatabase>
#include <QTextStream>
#include <QFileDialog>
#include <QDir>
#include <QEvent>
#include <QHBoxLayout>
#include <QIODevice>
#include <QMetaObject>
#include <QPointer>
#include <QTextCursor>
#include <QTextEdit>
#include <QPushButton>
#include <QStyle>
#include <QTextDocument>

#include <algorithm>

/**
 * @brief 构造函数：创建 UI 布局，初始化所有子控件，并注册 Logger sink。
 *
 * 布局结构：
 * ┌──────────────────────────────────────────┐
 * │                                          │
 * │           只读日志文本区（_text） [按钮]  │  ← 按钮悬浮，不占布局高度
 * │                                          │
 * └──────────────────────────────────────────┘
 *
 * 关键连接：
 * - Logger sink → LogPanel::appendLog（invokeMethod 排队到 UI 线程）
 * - _clearBtn::clicked → LogPanel::clearLogs
 * - _saveBtn::clicked → 打开文件保存对话框 → saveLogsToFile
 *
 * @param parent 父 Widget 指针。
 */
LogPanel::LogPanel(QWidget *parent)
    : QWidget(parent)
{
    Ui::LogPanel ui;
    ui.setupUi(this);

    _text = ui.m_text;

    _toolOverlay = new QWidget(_text->viewport());
    _toolOverlay->setObjectName(QStringLiteral("consoleToolOverlay"));
    _toolOverlay->setAttribute(Qt::WA_TranslucentBackground);
    auto *tool_layout = new QHBoxLayout(_toolOverlay);
    tool_layout->setContentsMargins(0, 0, 0, 0);
    tool_layout->setSpacing(2);

    _clearBtn = new QPushButton(_toolOverlay);
    _clearBtn->setObjectName(QStringLiteral("clearConsoleButton"));
    _saveBtn = new QPushButton(_toolOverlay);
    _saveBtn->setObjectName(QStringLiteral("saveConsoleButton"));
    tool_layout->addWidget(_clearBtn);
    tool_layout->addWidget(_saveBtn);

    _clearBtn->setText(QString());
    _clearBtn->setFixedSize(24, 24);
    _clearBtn->setFlat(true);
    _clearBtn->setIcon(style()->standardIcon(QStyle::SP_TrashIcon));
    _clearBtn->setToolTip(tr("清空控制台"));
    _saveBtn->setText(QString());
    _saveBtn->setFixedSize(24, 24);
    _saveBtn->setFlat(true);
    _saveBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    _saveBtn->setToolTip(tr("保存控制台输出"));
    _text->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    _text->setLineWrapMode(QTextEdit::NoWrap);
    _text->setFrameShape(QFrame::NoFrame);
    _text->setUndoRedoEnabled(false);
    _text->document()->setMaximumBlockCount(20000);
    _text->setPlaceholderText(tr("处理信息和诊断输出将显示在这里"));
    _text->viewport()->installEventFilter(this);
    _toolOverlay->setFixedSize(_toolOverlay->sizeHint());
    updateToolOverlayGeometry();

    connect(_clearBtn, &QPushButton::clicked, this, &LogPanel::clearLogs);

    connect(_saveBtn, &QPushButton::clicked, this, [this]() {
        QString p = QFileDialog::getSaveFileName(
            this, tr("保存控制台输出"), QDir::homePath(), tr("文本文件 (*.txt);;所有文件 (*)"));
        if (!p.isEmpty()) saveLogsToFile(p);
    });

    QPointer<LogPanel> self(this);
    _sinkId = Logger::instance()->registerSink(
        [self](const Logger::Entry &entry)
        {
            if (!self)
            {
                return;
            }

            const QString formatted =
                QString::fromUtf8(entry.formatted.c_str(), static_cast<int>(entry.formatted.size()));
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
    if (_sinkId != 0)
    {
        Logger::instance()->unregisterSink(_sinkId);
    }
}

QSize LogPanel::minimumSizeHint() const
{
    return QSize(220, 90);
}

QSize LogPanel::sizeHint() const
{
    return QSize(720, 180);
}

bool LogPanel::eventFilter(QObject *watched, QEvent *event)
{
    if (_text && watched == _text->viewport() && event->type() == QEvent::Resize)
    {
        updateToolOverlayGeometry();
    }
    return QWidget::eventFilter(watched, event);
}

void LogPanel::updateToolOverlayGeometry()
{
    if (!_text || !_toolOverlay)
    {
        return;
    }
    constexpr int margin = 2;
    _toolOverlay->move(
        std::max(0, _text->viewport()->width() - _toolOverlay->width() - margin),
        margin);
    _toolOverlay->raise();
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
 * 追加方式：
 * - moveCursor(End) + insertPlainText 比 append() 更高效（不触发段落格式化）；
 * - ensureCursorVisible 保证新日志自动滚动到可视区域底部。
 *
 * @param formatted 格式化后的完整日志行。
 * @param level     日志级别整数值。
 */
void LogPanel::appendLog(const QString &formatted, int level)
{
    Q_UNUSED(level);

    _text->moveCursor(QTextCursor::End);    // 将光标移到末尾
    _text->insertPlainText(formatted);       // 追加文本
    _text->ensureCursorVisible();           // 自动滚动到最新一行
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
    if (_text) _text->clear();

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
    ts << _text->toPlainText();
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
 * 读取结果会批量插入文本区，减少 UI 重绘次数。
 *
 * 使用场景：打开已有项目时，将上次运行留下的历史日志回填到面板。
 */
void LogPanel::loadFromLogFile()
{
    const QString logPath = QString::fromStdString(Logger::instance()->logFilePath());

    QFile f(logPath);
    if (!f.exists()) return;
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    if (_text)
    {
        _text->clear();
    }

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

    if (!data.isEmpty())
    {
        _text->moveCursor(QTextCursor::End);
        _text->insertPlainText(QString::fromUtf8(data));
        _text->ensureCursorVisible();
    }
}
