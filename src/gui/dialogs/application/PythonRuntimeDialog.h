#pragma once

#include <QDialog>

class QCheckBox;
class QCloseEvent;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class PythonRuntimeManager;

class PythonRuntimeDialog : public QDialog
{
    Q_OBJECT
public:
    enum class Mode
    {
        StartupPrompt,
        Update
    };

    explicit PythonRuntimeDialog(Mode mode, QWidget *parent = nullptr);

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    void startInstall();
    void closeOrCancel();
    void setRunning(bool running);
    void persistReminderPreference();

    Mode _mode;
    PythonRuntimeManager *_manager{};
    QLabel *_statusLabel{};
    QLabel *_pathLabel{};
    QCheckBox *_suppressPromptCheck{};
    QProgressBar *_progressBar{};
    QPlainTextEdit *_detailsEdit{};
    QPushButton *_installButton{};
    QPushButton *_detailsButton{};
    QPushButton *_closeButton{};
    bool _closeAfterCancellation{};
};
