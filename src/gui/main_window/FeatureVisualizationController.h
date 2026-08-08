#pragma once

#include "LayerRenderer.h"

#include <QJsonObject>
#include <QObject>
#include <QPointer>

class DialogSettingStore;
class ProjectManager;
class QMainWindow;

// Owns the feature-overlay dialog and its project-scoped persistence. Keeping
// this workflow separate prevents menu binding from also becoming a settings
// codec and canvas presentation controller.
class FeatureVisualizationController : public QObject
{
    Q_OBJECT

public:
    explicit FeatureVisualizationController(QMainWindow *mainWindow,
                                            QObject *parent = nullptr);

    void setProjectManager(ProjectManager *projectManager);

public slots:
    void openDialog();
    void applySavedOptions(const QJsonObject &uiSettings);

signals:
    void optionsChanged(const LayerRenderer::FeatureDisplayOptions &options);

private:
    DialogSettingStore *ensureSettingStore();
    QJsonObject loadSettings(const QJsonObject &fallback = QJsonObject());

    QPointer<QMainWindow> _mainWindow;
    ProjectManager *_projectManager = nullptr;
    DialogSettingStore *_settingStore = nullptr;
};
