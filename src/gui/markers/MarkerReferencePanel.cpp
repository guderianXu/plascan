#include "MarkerReferencePanel.h"

#include "MarkerProjectionPanel.h"
#include "MarkerWorkspaceController.h"
#include "reference/CoordinateReference.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTableWidget>
#include <QTabWidget>
#include <QVBoxLayout>

namespace xjw::gui::markers
{

namespace
{

QString roleText(control_points::MarkerRole role)
{
    switch (role)
    {
    case control_points::MarkerRole::TieMarker: return QStringLiteral("标记点");
    case control_points::MarkerRole::ControlPoint: return QStringLiteral("控制点");
    case control_points::MarkerRole::CheckPoint: return QStringLiteral("检查点");
    }
    return QStringLiteral("标记点");
}

QDoubleSpinBox *makeCoordinateSpin(const QString &name, QWidget *parent)
{
    auto *spin = new QDoubleSpinBox(parent);
    spin->setObjectName(name);
    spin->setRange(-1.0e12, 1.0e12);
    spin->setDecimals(8);
    return spin;
}

QDoubleSpinBox *makeSigmaSpin(const QString &name, QWidget *parent)
{
    auto *spin = new QDoubleSpinBox(parent);
    spin->setObjectName(name);
    spin->setRange(1.0e-8, 1.0e9);
    spin->setDecimals(8);
    spin->setValue(1.0);
    return spin;
}

} // namespace

MarkerReferencePanel::MarkerReferencePanel(QWidget *parent)
    : QWidget(parent)
{
    setupUi();
}

void MarkerReferencePanel::setupUi()
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(6);

    _markerTable = new QTableWidget(this);
    _markerTable->setObjectName(QStringLiteral("markerTable"));
    _markerTable->setColumnCount(4);
    _markerTable->setHorizontalHeaderLabels({QStringLiteral("启用"),
                                             QStringLiteral("名称"),
                                             QStringLiteral("角色"),
                                             QStringLiteral("投影")});
    _markerTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _markerTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _markerTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _markerTable->verticalHeader()->setVisible(false);
    _markerTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _markerTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    _markerTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    _markerTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    layout->addWidget(_markerTable, 2);

    _detailTabs = new QTabWidget(this);
    auto *reference_page = new QWidget(_detailTabs);
    auto *form = new QFormLayout(reference_page);
    form->setContentsMargins(6, 6, 6, 6);
    _enabledCheck = new QCheckBox(QStringLiteral("参与处理"), reference_page);
    _enabledCheck->setObjectName(QStringLiteral("markerEnabledCheck"));
    _labelEdit = new QLineEdit(reference_page);
    _labelEdit->setObjectName(QStringLiteral("markerLabelEdit"));
    _roleCombo = new QComboBox(reference_page);
    _roleCombo->setObjectName(QStringLiteral("markerRoleCombo"));
    _roleCombo->addItem(QStringLiteral("标记点"), static_cast<int>(control_points::MarkerRole::TieMarker));
    _roleCombo->addItem(QStringLiteral("控制点"), static_cast<int>(control_points::MarkerRole::ControlPoint));
    _roleCombo->addItem(QStringLiteral("检查点"), static_cast<int>(control_points::MarkerRole::CheckPoint));
    _hasReferenceCheck = new QCheckBox(QStringLiteral("使用参考坐标"), reference_page);
    _hasReferenceCheck->setObjectName(QStringLiteral("markerHasReferenceCheck"));
    _xSpin = makeCoordinateSpin(QStringLiteral("markerReferenceX"), reference_page);
    _ySpin = makeCoordinateSpin(QStringLiteral("markerReferenceY"), reference_page);
    _zSpin = makeCoordinateSpin(QStringLiteral("markerReferenceZ"), reference_page);
    _sigmaXySpin = makeSigmaSpin(QStringLiteral("markerReferenceSigmaXY"), reference_page);
    _sigmaZSpin = makeSigmaSpin(QStringLiteral("markerReferenceSigmaZ"), reference_page);
    _crsEdit = new QLineEdit(reference_page);
    _crsEdit->setObjectName(QStringLiteral("markerReferenceCrs"));
    _crsEdit->setPlaceholderText(QStringLiteral("例如 EPSG:4978"));
    _axisOrderCombo = new QComboBox(reference_page);
    _axisOrderCombo->setObjectName(QStringLiteral("markerReferenceAxisOrder"));
    _axisOrderCombo->addItem(QStringLiteral("传统 GIS（X/Y 或经度/纬度）"),
                             QStringLiteral("traditional_gis"));
    _axisOrderCombo->addItem(QStringLiteral("经度 / 纬度"),
                             QStringLiteral("longitude_latitude"));
    _axisOrderCombo->addItem(QStringLiteral("纬度 / 经度"),
                             QStringLiteral("latitude_longitude"));
    _axisOrderCombo->addItem(QStringLiteral("CRS 官方轴序"),
                             QStringLiteral("authority_compliant"));
    _verticalDatumEdit = new QLineEdit(reference_page);
    _verticalDatumEdit->setObjectName(QStringLiteral("markerReferenceVerticalDatum"));
    _verticalDatumEdit->setPlaceholderText(QStringLiteral("例如 ellipsoidal 或 EGM2008"));
    _verticalUnitCombo = new QComboBox(reference_page);
    _verticalUnitCombo->setObjectName(QStringLiteral("markerReferenceVerticalUnit"));
    _verticalUnitCombo->addItem(QStringLiteral("未指定"), QString());
    _verticalUnitCombo->addItem(QStringLiteral("米"), QStringLiteral("m"));
    _verticalUnitCombo->addItem(QStringLiteral("国际英尺"), QStringLiteral("ft"));
    _verticalUnitCombo->addItem(QStringLiteral("美国测量英尺"), QStringLiteral("us_survey_ft"));
    _referenceStatusLabel = new QLabel(reference_page);
    _referenceStatusLabel->setObjectName(QStringLiteral("markerReferenceStatus"));
    _referenceStatusLabel->setWordWrap(true);
    _applyButton = new QPushButton(QStringLiteral("应用"), reference_page);
    _applyButton->setObjectName(QStringLiteral("applyMarkerPropertiesButton"));

    form->addRow(_enabledCheck);
    form->addRow(QStringLiteral("名称"), _labelEdit);
    form->addRow(QStringLiteral("角色"), _roleCombo);
    form->addRow(_hasReferenceCheck);
    form->addRow(QStringLiteral("X"), _xSpin);
    form->addRow(QStringLiteral("Y"), _ySpin);
    form->addRow(QStringLiteral("Z"), _zSpin);
    form->addRow(QStringLiteral("XY 精度"), _sigmaXySpin);
    form->addRow(QStringLiteral("Z 精度"), _sigmaZSpin);
    form->addRow(QStringLiteral("源 CRS"), _crsEdit);
    form->addRow(QStringLiteral("坐标轴顺序"), _axisOrderCombo);
    form->addRow(QStringLiteral("垂直基准"), _verticalDatumEdit);
    form->addRow(QStringLiteral("高程单位"), _verticalUnitCombo);
    form->addRow(QStringLiteral("参考状态"), _referenceStatusLabel);
    form->addRow(QString(), _applyButton);

    _projectionPanel = new MarkerProjectionPanel(_detailTabs);
    _scaleBarTable = new QTableWidget(_detailTabs);
    _scaleBarTable->setObjectName(QStringLiteral("markerScaleBarTable"));
    _scaleBarTable->setColumnCount(5);
    _scaleBarTable->setHorizontalHeaderLabels({QStringLiteral("名称"),
                                               QStringLiteral("端点"),
                                               QStringLiteral("测量距离"),
                                               QStringLiteral("估计距离"),
                                               QStringLiteral("残差")});
    _scaleBarTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    _scaleBarTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _detailTabs->addTab(reference_page, QStringLiteral("参考"));
    _detailTabs->addTab(_projectionPanel, QStringLiteral("投影"));
    _detailTabs->addTab(_scaleBarTable, QStringLiteral("比例尺"));
    layout->addWidget(_detailTabs, 3);

    connect(_markerTable, &QTableWidget::itemSelectionChanged,
            this, &MarkerReferencePanel::loadSelectedMarker);
    connect(_markerTable, &QTableWidget::cellDoubleClicked, this, [this](int row, int)
    {
        if (auto *item = _markerTable->item(row, 1))
        {
            emit focusMeasurementRequested(item->data(Qt::UserRole).toString());
        }
    });
    connect(_hasReferenceCheck, &QCheckBox::toggled,
            this, &MarkerReferencePanel::setReferenceEditorsEnabled);
    connect(_crsEdit, &QLineEdit::textChanged, this, &MarkerReferencePanel::updateReferenceStatus);
    connect(_axisOrderCombo, &QComboBox::currentIndexChanged,
            this, &MarkerReferencePanel::updateReferenceStatus);
    connect(_verticalDatumEdit, &QLineEdit::textChanged,
            this, &MarkerReferencePanel::updateReferenceStatus);
    connect(_verticalUnitCombo, &QComboBox::currentIndexChanged,
            this, &MarkerReferencePanel::updateReferenceStatus);
    connect(_applyButton, &QPushButton::clicked,
            this, &MarkerReferencePanel::applySelectedMarker);
    setReferenceEditorsEnabled(false);
}

void MarkerReferencePanel::setController(MarkerWorkspaceController *controller)
{
    if (_controller) disconnect(_controller, nullptr, this, nullptr);
    _controller = controller;
    if (_controller)
    {
        connect(_controller, &MarkerWorkspaceController::markerSetChanged,
                this, &MarkerReferencePanel::refresh);
    }
    refresh();
}

void MarkerReferencePanel::refresh()
{
    const QString selected_id = _selectedMarkerId;
    _markerTable->setRowCount(_controller ? _controller->markerSet().markers().size() : 0);
    if (!_controller)
    {
        _projectionPanel->setMarker(nullptr);
        return;
    }

    int selected_row = -1;
    const auto &markers = _controller->markerSet().markers();
    for (int row = 0; row < markers.size(); ++row)
    {
        const auto &marker = markers.at(row);
        _markerTable->setItem(row, 0, new QTableWidgetItem(marker.enabled
                                                              ? QStringLiteral("是")
                                                              : QStringLiteral("否")));
        auto *label_item = new QTableWidgetItem(marker.label);
        label_item->setData(Qt::UserRole, marker.id);
        _markerTable->setItem(row, 1, label_item);
        _markerTable->setItem(row, 2, new QTableWidgetItem(roleText(marker.role)));
        _markerTable->setItem(row, 3, new QTableWidgetItem(QString::number(marker.projections.size())));
        if (marker.id == selected_id) selected_row = row;
    }
    if (selected_row < 0 && !markers.isEmpty()) selected_row = 0;
    if (selected_row >= 0) _markerTable->selectRow(selected_row);

    _scaleBarTable->setRowCount(_controller->markerSet().scaleBars().size());
    for (int row = 0; row < _controller->markerSet().scaleBars().size(); ++row)
    {
        const auto &scale = _controller->markerSet().scaleBars().at(row);
        _scaleBarTable->setItem(row, 0, new QTableWidgetItem(scale.label));
        _scaleBarTable->setItem(row, 1, new QTableWidgetItem(
            QStringLiteral("%1 - %2").arg(scale.firstMarkerId, scale.secondMarkerId)));
        _scaleBarTable->setItem(row, 2, new QTableWidgetItem(QString::number(scale.measuredDistance)));
        _scaleBarTable->setItem(row, 3, new QTableWidgetItem(QString::number(scale.estimatedDistance)));
        _scaleBarTable->setItem(row, 4, new QTableWidgetItem(QString::number(scale.residual)));
    }
}

void MarkerReferencePanel::selectMarker(const control_points::MarkerId &markerId)
{
    _selectedMarkerId = markerId;
    for (int row = 0; row < _markerTable->rowCount(); ++row)
    {
        if (_markerTable->item(row, 1)
            && _markerTable->item(row, 1)->data(Qt::UserRole).toString() == markerId)
        {
            _markerTable->selectRow(row);
            return;
        }
    }
}

const control_points::Marker *MarkerReferencePanel::selectedMarker() const
{
    if (!_controller || _selectedMarkerId.isEmpty()) return nullptr;
    try
    {
        return &_controller->markerSet().marker(_selectedMarkerId);
    }
    catch (const std::exception &)
    {
        return nullptr;
    }
}

void MarkerReferencePanel::loadSelectedMarker()
{
    const int row = _markerTable->currentRow();
    if (row < 0 || !_markerTable->item(row, 1))
    {
        _selectedMarkerId.clear();
        _projectionPanel->setMarker(nullptr);
        return;
    }
    _selectedMarkerId = _markerTable->item(row, 1)->data(Qt::UserRole).toString();
    const control_points::Marker *marker = selectedMarker();
    if (!marker) return;

    _labelEdit->setText(marker->label);
    _enabledCheck->setChecked(marker->enabled);
    _roleCombo->setCurrentIndex(_roleCombo->findData(static_cast<int>(marker->role)));
    const bool has_reference = marker->referenceCoordinate.has_value();
    _hasReferenceCheck->setChecked(has_reference);
    if (has_reference)
    {
        const auto &reference = marker->referenceCoordinate.value();
        _xSpin->setValue(reference.x);
        _ySpin->setValue(reference.y);
        _zSpin->setValue(reference.z);
        _sigmaXySpin->setValue((reference.sigmaX + reference.sigmaY) * 0.5);
        _sigmaZSpin->setValue(reference.sigmaZ);
        _crsEdit->setText(reference.sourceCrs);
        _axisOrderCombo->setCurrentIndex(
            qMax(0, _axisOrderCombo->findData(reference.axisOrder)));
        _verticalDatumEdit->setText(reference.verticalDatum);
        _verticalUnitCombo->setCurrentIndex(
            qMax(0, _verticalUnitCombo->findData(reference.verticalUnit)));
    }
    updateReferenceStatus();
    _projectionPanel->setMarker(marker);
}

void MarkerReferencePanel::applySelectedMarker()
{
    if (!_controller || _selectedMarkerId.isEmpty()) return;
    std::optional<control_points::ReferenceCoordinate> reference;
    if (_hasReferenceCheck->isChecked())
    {
        control_points::ReferenceCoordinate value;
        value.x = _xSpin->value();
        value.y = _ySpin->value();
        value.z = _zSpin->value();
        value.sigmaX = _sigmaXySpin->value();
        value.sigmaY = _sigmaXySpin->value();
        value.sigmaZ = _sigmaZSpin->value();
        value.sourceCrs = _crsEdit->text().trimmed();
        value.axisOrder = _axisOrderCombo->currentData().toString();
        value.verticalDatum = _verticalDatumEdit->text().trimmed();
        value.verticalUnit = _verticalUnitCombo->currentData().toString();
        reference = value;
    }
    QString error;
    const auto role = static_cast<control_points::MarkerRole>(_roleCombo->currentData().toInt());
    if (!_controller->updateMarkerProperties(_selectedMarkerId,
                                             _labelEdit->text(),
                                             role,
                                             _enabledCheck->isChecked(),
                                             reference,
                                             &error))
    {
        QMessageBox::warning(this, QStringLiteral("标记属性"), error);
    }
}

void MarkerReferencePanel::setReferenceEditorsEnabled(bool enabled)
{
    _xSpin->setEnabled(enabled);
    _ySpin->setEnabled(enabled);
    _zSpin->setEnabled(enabled);
    _sigmaXySpin->setEnabled(enabled);
    _sigmaZSpin->setEnabled(enabled);
    _crsEdit->setEnabled(enabled);
    _axisOrderCombo->setEnabled(enabled);
    _verticalDatumEdit->setEnabled(enabled);
    _verticalUnitCombo->setEnabled(enabled);
    updateReferenceStatus();
}

void MarkerReferencePanel::updateReferenceStatus()
{
    if (!_hasReferenceCheck->isChecked())
    {
        _referenceStatusLabel->setText(QStringLiteral("未使用参考坐标"));
        return;
    }

    control_points::ReferenceCoordinate coordinate;
    coordinate.x = _xSpin->value();
    coordinate.y = _ySpin->value();
    coordinate.z = _zSpin->value();
    coordinate.sigmaX = _sigmaXySpin->value();
    coordinate.sigmaY = _sigmaXySpin->value();
    coordinate.sigmaZ = _sigmaZSpin->value();
    coordinate.sourceCrs = _crsEdit->text().trimmed();
    coordinate.axisOrder = _axisOrderCombo->currentData().toString();
    coordinate.verticalDatum = _verticalDatumEdit->text().trimmed();
    coordinate.verticalUnit = _verticalUnitCombo->currentData().toString();
    const control_points::ReferenceCoordinateAssessment assessment =
        control_points::assessReferenceCoordinate(coordinate);
    _referenceStatusLabel->setText(assessment.usable
                                       ? QStringLiteral("可用于绝对定向和 BA")
                                       : assessment.error);
}

} // namespace xjw::gui::markers
