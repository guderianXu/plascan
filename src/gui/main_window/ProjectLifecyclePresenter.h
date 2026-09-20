#pragma once

#include <QObject>

class QMainWindow;
class QProgressDialog;
class QStatusBar;

class ProjectLifecycleService;

namespace xjw::gui::project
{
class ProjectSession;
}

class ProjectLifecyclePresenter final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectLifecyclePresenter(ProjectLifecycleService *lifecycle,
                                       xjw::gui::project::ProjectSession *session,
                                       QMainWindow *window,
                                       QStatusBar *statusBar,
                                       QObject *parent = nullptr);

    bool isCloseSavePending() const;
    void requestCloseAfterSave();

signals:
    void closeAfterSaveRequested();

private slots:
    void showOpenProgress(const QString &projectPath);
    void updateOpenProgress(const QString &message, int percent);
    void finishOpenProgress(bool success, const QString &message);
    void showSaveProgress();
    void finishSaveProgress(bool success);
    void updateWindowTitle(bool dirty);

private:
    ProjectLifecycleService *_lifecycle = nullptr;
    xjw::gui::project::ProjectSession *_session = nullptr;
    QMainWindow *_window = nullptr;
    QStatusBar *_statusBar = nullptr;
    QProgressDialog *_openProgress = nullptr;
    QProgressDialog *_saveProgress = nullptr;
    bool _closeSavePending = false;
};
