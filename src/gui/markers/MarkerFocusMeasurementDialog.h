#pragma once

#include <QDialog>
#include <QPointer>

#include <optional>

class QLabel;
class QComboBox;
class QPushButton;
class ProjectData;
class DualImageViewer;

namespace xjw::gui::markers
{

class MarkerWorkspaceController;

class MarkerFocusMeasurementDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit MarkerFocusMeasurementDialog(QWidget *parent = nullptr);

    bool setContext(MarkerWorkspaceController *controller,
                    ProjectData *projectData,
                    const QString &markerId,
                    const QString &anchorImagePath,
                    const QString &candidateImagePath = {});
    void setCandidatePixel(const QPointF &pixel);

private:
    void setupUi();
    void rebuildCandidates(const QString &requestedCandidate);
    void generateGeometryPredictions();
    void refreshMeasurement();
    void confirmCandidate();
    void setCandidateState(bool block);
    QString currentCandidatePath() const;

    QPointer<MarkerWorkspaceController> _controller;
    ProjectData *_projectData = nullptr;
    QString _markerId;
    QString _anchorImagePath;
    std::optional<QPointF> _candidatePixel;
    QComboBox *_candidateCombo = nullptr;
    QLabel *_statusLabel = nullptr;
    DualImageViewer *_viewer = nullptr;
    QPushButton *_confirmButton = nullptr;
    QPushButton *_blockButton = nullptr;
    QPushButton *_disableButton = nullptr;
    QPushButton *_skipButton = nullptr;
};

} // namespace xjw::gui::markers
