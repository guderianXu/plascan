#include "application/PythonRuntimeDialog.h"

#include "runtime/PythonRuntimeManager.h"

#include <QCheckBox>
#include <QCloseEvent>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QFormLayout>
#include <QLabel>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QTextCursor>
#include <QVBoxLayout>

PythonRuntimeDialog::PythonRuntimeDialog(Mode mode, QWidget *parent)
    : QDialog(parent)
    , _mode(mode)
    , _manager(new PythonRuntimeManager(this))
{
    setWindowTitle(mode == Mode::StartupPrompt
        ? tr("需要 Python 环境")
        : tr("更新 Python 环境"));
    setModal(true);
    setMinimumWidth(620);

    auto *rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(18, 18, 18, 14);
    rootLayout->setSpacing(12);

    auto *summaryLabel = new QLabel(this);
    summaryLabel->setObjectName(QStringLiteral("pythonRuntimeSummaryLabel"));
    summaryLabel->setWordWrap(true);
    summaryLabel->setText(mode == Mode::StartupPrompt
        ? tr("未检测到 PlaScan Python 环境。可以现在联网自动下载并安装；下载内容可能较大，"
             "暂不安装不会影响主界面启动。")
        : tr("检查并更新 PlaScan 管理的 Python 环境及其依赖。需要网络连接，更新过程在后台运行，"
             "可以随时取消。"));
    rootLayout->addWidget(summaryLabel);

    auto *formLayout = new QFormLayout;
    formLayout->setLabelAlignment(Qt::AlignRight | Qt::AlignTop);
    formLayout->setHorizontalSpacing(12);
    formLayout->setVerticalSpacing(8);

    _statusLabel = new QLabel(this);
    _statusLabel->setObjectName(QStringLiteral("pythonRuntimeStatusLabel"));
    _statusLabel->setWordWrap(true);
    const QString currentPython = PythonRuntimeManager::currentPythonExecutable();
    _statusLabel->setText(currentPython.isEmpty() ? tr("未安装") : tr("已检测到环境"));
    formLayout->addRow(tr("状态:"), _statusLabel);

    _pathLabel = new QLabel(this);
    _pathLabel->setObjectName(QStringLiteral("pythonRuntimePathLabel"));
    _pathLabel->setWordWrap(true);
    _pathLabel->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
    _pathLabel->setText(currentPython.isEmpty()
        ? PythonRuntimeManager::managedPythonExecutable()
        : QFileInfo(currentPython).absoluteFilePath());
    formLayout->addRow(tr("环境位置:"), _pathLabel);
    rootLayout->addLayout(formLayout);

    _suppressPromptCheck = new QCheckBox(tr("下次启动时不再提醒"), this);
    _suppressPromptCheck->setObjectName(QStringLiteral("suppressPythonRuntimePromptCheck"));
    _suppressPromptCheck->setChecked(PythonRuntimeManager::startupPromptSuppressed());
    _suppressPromptCheck->setVisible(mode == Mode::StartupPrompt);
    rootLayout->addWidget(_suppressPromptCheck);

    _progressBar = new QProgressBar(this);
    _progressBar->setObjectName(QStringLiteral("pythonRuntimeProgressBar"));
    _progressBar->setRange(0, 1);
    _progressBar->setValue(0);
    _progressBar->setTextVisible(false);
    _progressBar->hide();
    rootLayout->addWidget(_progressBar);

    _detailsEdit = new QPlainTextEdit(this);
    _detailsEdit->setObjectName(QStringLiteral("pythonRuntimeDetailsEdit"));
    _detailsEdit->setReadOnly(true);
    _detailsEdit->setMaximumBlockCount(2000);
    _detailsEdit->setMinimumHeight(180);
    _detailsEdit->hide();
    rootLayout->addWidget(_detailsEdit);

    auto *buttons = new QDialogButtonBox(this);
    _detailsButton = buttons->addButton(tr("详细信息"), QDialogButtonBox::ActionRole);
    _detailsButton->setObjectName(QStringLiteral("pythonRuntimeDetailsButton"));
    _installButton = buttons->addButton(
        mode == Mode::StartupPrompt ? tr("自动下载") : tr("更新或下载"),
        QDialogButtonBox::AcceptRole);
    _installButton->setObjectName(QStringLiteral("installPythonRuntimeButton"));
    _installButton->setDefault(true);
    _closeButton = buttons->addButton(
        mode == Mode::StartupPrompt ? tr("暂不处理") : tr("关闭"),
        QDialogButtonBox::RejectRole);
    _closeButton->setObjectName(QStringLiteral("closePythonRuntimeDialogButton"));
    rootLayout->addWidget(buttons);

    connect(_detailsButton, &QPushButton::clicked, this, [this]()
    {
        const bool showDetails = _detailsEdit->isHidden();
        _detailsEdit->setVisible(showDetails);
        _detailsButton->setText(showDetails ? tr("隐藏详细信息") : tr("详细信息"));
        adjustSize();
    });
    connect(_installButton, &QPushButton::clicked, this, &PythonRuntimeDialog::startInstall);
    connect(_closeButton, &QPushButton::clicked, this, &PythonRuntimeDialog::closeOrCancel);
    connect(_manager, &PythonRuntimeManager::outputReceived, this, [this](const QString &text)
    {
        _detailsEdit->moveCursor(QTextCursor::End);
        _detailsEdit->insertPlainText(text);
        _detailsEdit->moveCursor(QTextCursor::End);
    });
    connect(_manager, &PythonRuntimeManager::finished, this, [this](bool success, const QString &message)
    {
        setRunning(false);
        _statusLabel->setText(message);
        if (success)
        {
            _progressBar->setRange(0, 1);
            _progressBar->setValue(1);
            _pathLabel->setText(PythonRuntimeManager::managedPythonExecutable());
            _installButton->setEnabled(false);
            _closeButton->setText(tr("关闭"));
            PythonRuntimeManager::setStartupPromptSuppressed(false);
        }
        else
        {
            _detailsEdit->show();
            _detailsButton->setText(tr("隐藏详细信息"));
        }

        if (_closeAfterCancellation)
        {
            persistReminderPreference();
            reject();
        }
    });
}

void PythonRuntimeDialog::closeEvent(QCloseEvent *event)
{
    if (_manager->isRunning())
    {
        _closeAfterCancellation = true;
        _manager->cancel();
        event->ignore();
        return;
    }
    persistReminderPreference();
    QDialog::closeEvent(event);
}

void PythonRuntimeDialog::startInstall()
{
    persistReminderPreference();
    _detailsEdit->clear();
    _statusLabel->setText(tr("正在准备下载和安装，请稍候..."));
    setRunning(true);
    _manager->startInstall();
}

void PythonRuntimeDialog::closeOrCancel()
{
    if (_manager->isRunning())
    {
        _closeAfterCancellation = true;
        _statusLabel->setText(tr("正在取消 Python 环境更新..."));
        _closeButton->setEnabled(false);
        _manager->cancel();
        return;
    }

    persistReminderPreference();
    reject();
}

void PythonRuntimeDialog::setRunning(bool running)
{
    _progressBar->setVisible(running || _progressBar->value() > 0);
    if (running)
    {
        _progressBar->setRange(0, 0);
    }
    _installButton->setEnabled(!running);
    _suppressPromptCheck->setEnabled(!running);
    _closeButton->setText(running ? tr("取消") : tr("关闭"));
    _closeButton->setEnabled(true);
}

void PythonRuntimeDialog::persistReminderPreference()
{
    if (_mode == Mode::StartupPrompt)
    {
        PythonRuntimeManager::setStartupPromptSuppressed(_suppressPromptCheck->isChecked());
    }
}
