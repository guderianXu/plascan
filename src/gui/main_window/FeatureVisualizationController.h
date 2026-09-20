#pragma once

#include "LayerRenderer.h"

#include <QJsonObject>
#include <QObject>
#include <QPointer>

class DialogSettingStore;
class QMainWindow;

namespace xjw::gui::project
{
class ProjectSession;
}

// Owns the feature-overlay dialog and its project-scoped persistence. Keeping
// this workflow separate prevents menu binding from also becoming a settings
// codec and canvas presentation controller.
class FeatureVisualizationController : public QObject
{
    Q_OBJECT

public:
    explicit FeatureVisualizationController(QMainWindow *mainWindow,
                                            QObject *parent = nullptr);

    void setProjectSession(xjw::gui::project::ProjectSession *session);

public slots:
    void openDialog();
    void applySavedOptions(const QJsonObject &uiSettings);

signals:
    void optionsChanged(const LayerRenderer::FeatureDisplayOptions &options);

private:
    DialogSettingStore *ensureSettingStore();
    QJsonObject loadSettings(const QJsonObject &fallback = QJsonObject());

    QPointer<QMainWindow> _mainWindow;
    xjw::gui::project::ProjectSession *_session = nullptr;
    DialogSettingStore *_settingStore = nullptr;
};
