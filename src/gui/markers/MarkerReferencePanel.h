#pragma once

#include "model/MarkerTypes.h"

#include <QPointer>
#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;
class QTabWidget;

namespace xjw::gui::markers
{

class MarkerProjectionPanel;
class MarkerWorkspaceController;

class MarkerReferencePanel final : public QWidget
{
    Q_OBJECT

public:
    explicit MarkerReferencePanel(QWidget *parent = nullptr);

    void setController(MarkerWorkspaceController *controller);
    void refresh();
    void selectMarker(const control_points::MarkerId &markerId);

signals:
    void focusMeasurementRequested(const QString &markerId);

private:
    void setupUi();
    void loadSelectedMarker();
    void applySelectedMarker();
    void updateReferenceStatus();
    const control_points::Marker *selectedMarker() const;
    void setReferenceEditorsEnabled(bool enabled);

    QPointer<MarkerWorkspaceController> _controller;
    QTableWidget *_markerTable = nullptr;
    QTabWidget *_detailTabs = nullptr;
    QLineEdit *_labelEdit = nullptr;
    QComboBox *_roleCombo = nullptr;
    QCheckBox *_enabledCheck = nullptr;
    QCheckBox *_hasReferenceCheck = nullptr;
    QDoubleSpinBox *_xSpin = nullptr;
    QDoubleSpinBox *_ySpin = nullptr;
    QDoubleSpinBox *_zSpin = nullptr;
    QDoubleSpinBox *_sigmaXySpin = nullptr;
    QDoubleSpinBox *_sigmaZSpin = nullptr;
    QLineEdit *_crsEdit = nullptr;
    QComboBox *_axisOrderCombo = nullptr;
    QLineEdit *_verticalDatumEdit = nullptr;
    QComboBox *_verticalUnitCombo = nullptr;
    QLabel *_referenceStatusLabel = nullptr;
    QPushButton *_applyButton = nullptr;
    MarkerProjectionPanel *_projectionPanel = nullptr;
    QTableWidget *_scaleBarTable = nullptr;
    control_points::MarkerId _selectedMarkerId;
};

} // namespace xjw::gui::markers
