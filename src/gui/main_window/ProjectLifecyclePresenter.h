#pragma once

#include <QObject>

class ProjectManager;
class QMainWindow;
class QProgressDialog;
class QStatusBar;

class ProjectLifecyclePresenter final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectLifecyclePresenter(ProjectManager *projectManager,
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
    ProjectManager *_projectManager = nullptr;
    QMainWindow *_window = nullptr;
    QStatusBar *_statusBar = nullptr;
    QProgressDialog *_openProgress = nullptr;
    QProgressDialog *_saveProgress = nullptr;
    bool _closeSavePending = false;
};
