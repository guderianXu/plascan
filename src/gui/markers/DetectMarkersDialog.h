#pragma once

#include "MarkerTaskRunner.h"

#include <QDialog>
#include <QPointer>

class ProjectData;
class QCloseEvent;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QProgressBar;
class QPushButton;
class QSpinBox;

namespace xjw::gui::markers
{

class MarkerWorkspaceController;

class DetectMarkersDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit DetectMarkersDialog(QWidget *parent = nullptr);

    bool setContext(MarkerWorkspaceController *controller, ProjectData *projectData);

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    void setupUi();
    void populateTargetFamilies();
    void updateFamilyAvailability();
    void startDetection();
    void cancelDetection();
    void handleProgress(const MarkerDetectionProgress &progress);
    void handleFinished(const MarkerDetectionTaskResult &result);
    void setRunning(bool running);

    QPointer<MarkerWorkspaceController> _controller;
    QPointer<ProjectData> _projectData;
    MarkerTaskRunner _runner;
    QStringList _preflightWarnings;
    bool _closeAfterCancel = false;

    QComboBox *_familyCombo = nullptr;
    QDoubleSpinBox *_decisionMarginSpin = nullptr;
    QSpinBox *_maxHammingSpin = nullptr;
    QSpinBox *_concurrentImagesSpin = nullptr;
    QProgressBar *_progressBar = nullptr;
    QLabel *_statusLabel = nullptr;
    QPushButton *_startButton = nullptr;
    QPushButton *_cancelButton = nullptr;
    QPushButton *_closeButton = nullptr;
};

} // namespace xjw::gui::markers
