#include "reconstruction/MapProjectDialog.h"

#include "shared/WorkflowParameterDialogStyle.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollArea>
#include <QSizePolicy>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

namespace
{

QHBoxLayout *pathRow(QLineEdit *lineEdit, QPushButton *button)
{
    auto *layout = new QHBoxLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    layout->addWidget(lineEdit, 1);
    layout->addWidget(button);
    return layout;
}

void configureCoordinateSpin(QDoubleSpinBox *spinBox, const QString &objectName)
{
    spinBox->setObjectName(objectName);
    spinBox->setRange(-1.0e12, 1.0e12);
    spinBox->setDecimals(6);
    spinBox->setSingleStep(1.0);
    spinBox->setKeyboardTracking(false);
}

} // namespace

void MapProjectDialog::setupUi()
{
    setObjectName(QStringLiteral("mapProjectDialog"));
    setWindowTitle(tr("创建正射影像"));
    resize(680, 780);
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);

    auto *mainLayout = new QVBoxLayout(this);
    xjw::gui::dialogs::configureWorkflowDialogLayout(mainLayout);

    _contentScrollArea = new QScrollArea(this);
    _contentScrollArea->setObjectName(QStringLiteral("orthoParameterScrollArea"));
    xjw::gui::dialogs::configureWorkflowScrollArea(_contentScrollArea);

    auto *contentWidget = new QWidget(_contentScrollArea);
    contentWidget->setObjectName(QStringLiteral("orthoParameterContent"));
    auto *contentLayout = new QVBoxLayout(contentWidget);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(9);

    _projectionGroup = new QGroupBox(tr("投影"), contentWidget);
    _projectionGroup->setObjectName(QStringLiteral("orthoProjectionGroup"));
    auto *projectionForm = new QFormLayout(_projectionGroup);
    xjw::gui::dialogs::configureWorkflowForm(projectionForm);
    auto *projectionTypes = new QWidget(_projectionGroup);
    auto *projectionTypeLayout = new QHBoxLayout(projectionTypes);
    projectionTypeLayout->setContentsMargins(0, 0, 0, 0);
    projectionTypeLayout->setSpacing(18);
    _demGridProjectionRadio = new QRadioButton(tr("地理（跟随 DEM 网格）"), projectionTypes);
    _planarProjectionRadio = new QRadioButton(tr("平面"), projectionTypes);
    _cylindricalProjectionRadio = new QRadioButton(tr("全球等距圆柱"), projectionTypes);
    _demGridProjectionRadio->setObjectName(QStringLiteral("orthoProjectionDemGridRadio"));
    _planarProjectionRadio->setObjectName(QStringLiteral("orthoProjectionPlanarRadio"));
    _cylindricalProjectionRadio->setObjectName(QStringLiteral("orthoProjectionCylindricalRadio"));
    _demGridProjectionRadio->setProperty("settingValue", QStringLiteral("dem_grid"));
    _planarProjectionRadio->setProperty("settingValue", QStringLiteral("planar"));
    _cylindricalProjectionRadio->setProperty("settingValue", QStringLiteral("cylindrical"));
    _demGridProjectionRadio->setChecked(true);
    _planarProjectionRadio->setToolTip(tr("保留点云局部 XY 平面，以 Z 最大值选择可见点。"));
    _cylindricalProjectionRadio->setToolTip(
        tr("按体固连经纬度展开全球表面，并写出自定义小天体投影 WKT。"));
    projectionTypeLayout->addWidget(_demGridProjectionRadio);
    projectionTypeLayout->addWidget(_planarProjectionRadio);
    projectionTypeLayout->addWidget(_cylindricalProjectionRadio);
    projectionTypeLayout->addStretch(1);
    _coordinateSystemLabel = new QLabel(tr("等待读取 DEM 坐标系"), _projectionGroup);
    _coordinateSystemLabel->setObjectName(QStringLiteral("orthoCoordinateSystemLabel"));
    _coordinateSystemLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _coordinateSystemLabel->setWordWrap(true);
    projectionForm->addRow(tr("类型:"), projectionTypes);
    projectionForm->addRow(tr("坐标系统:"), _coordinateSystemLabel);

    _bodyReferenceWidget = new QWidget(_projectionGroup);
    auto *bodyGrid = new QGridLayout(_bodyReferenceWidget);
    bodyGrid->setContentsMargins(0, 0, 0, 0);
    bodyGrid->setHorizontalSpacing(6);
    bodyGrid->setVerticalSpacing(4);
    _bodyReferenceAutoCheck = new QCheckBox(tr("从点云自动估算中心和平均半径"), _bodyReferenceWidget);
    _bodyReferenceAutoCheck->setObjectName(QStringLiteral("orthoBodyReferenceAutoCheck"));
    _bodyReferenceAutoCheck->setChecked(true);
    _bodyCenterXSpin = new QDoubleSpinBox(_bodyReferenceWidget);
    _bodyCenterYSpin = new QDoubleSpinBox(_bodyReferenceWidget);
    _bodyCenterZSpin = new QDoubleSpinBox(_bodyReferenceWidget);
    _referenceRadiusSpin = new QDoubleSpinBox(_bodyReferenceWidget);
    _centralMeridianSpin = new QDoubleSpinBox(_bodyReferenceWidget);
    for (QDoubleSpinBox *spinBox : {_bodyCenterXSpin, _bodyCenterYSpin, _bodyCenterZSpin})
    {
        configureCoordinateSpin(spinBox, QString());
    }
    _bodyCenterXSpin->setObjectName(QStringLiteral("orthoBodyCenterXSpin"));
    _bodyCenterYSpin->setObjectName(QStringLiteral("orthoBodyCenterYSpin"));
    _bodyCenterZSpin->setObjectName(QStringLiteral("orthoBodyCenterZSpin"));
    _referenceRadiusSpin->setObjectName(QStringLiteral("orthoReferenceRadiusSpin"));
    _referenceRadiusSpin->setRange(0.0, 1.0e12);
    _referenceRadiusSpin->setDecimals(6);
    _referenceRadiusSpin->setSuffix(tr(" m"));
    _referenceRadiusSpin->setKeyboardTracking(false);
    _centralMeridianSpin->setObjectName(QStringLiteral("orthoCentralMeridianSpin"));
    _centralMeridianSpin->setRange(-180.0, 180.0);
    _centralMeridianSpin->setDecimals(6);
    _centralMeridianSpin->setSuffix(tr("°"));
    _centralMeridianSpin->setKeyboardTracking(false);
    bodyGrid->addWidget(_bodyReferenceAutoCheck, 0, 0, 1, 6);
    bodyGrid->addWidget(new QLabel(tr("中心 X/Y/Z:"), _bodyReferenceWidget), 1, 0);
    bodyGrid->addWidget(_bodyCenterXSpin, 1, 1);
    bodyGrid->addWidget(_bodyCenterYSpin, 1, 2);
    bodyGrid->addWidget(_bodyCenterZSpin, 1, 3);
    bodyGrid->addWidget(new QLabel(tr("半径:"), _bodyReferenceWidget), 2, 0);
    bodyGrid->addWidget(_referenceRadiusSpin, 2, 1, 1, 2);
    bodyGrid->addWidget(new QLabel(tr("中央经线:"), _bodyReferenceWidget), 2, 3);
    bodyGrid->addWidget(_centralMeridianSpin, 2, 4, 1, 2);
    projectionForm->addRow(tr("小天体参考:"), _bodyReferenceWidget);
    contentLayout->addWidget(_projectionGroup);

    _parametersGroup = new QGroupBox(tr("参数"), contentWidget);
    _parametersGroup->setObjectName(QStringLiteral("orthoParametersGroup"));
    auto *parametersForm = new QFormLayout(_parametersGroup);
    xjw::gui::dialogs::configureWorkflowForm(parametersForm);
    _surfaceCombo = new QComboBox(_parametersGroup);
    _surfaceCombo->setObjectName(QStringLiteral("orthoSurfaceCombo"));
    _surfaceCombo->addItem(tr("DEM"), QStringLiteral("dem"));
    _surfaceCombo->addItem(tr("彩色点云"), QStringLiteral("point_cloud"));
    _surfaceCombo->setToolTip(tr("DEM 使用相机影像着色；彩色点云直接使用点的 RGB。"));
    _demEdit = new QLineEdit(_parametersGroup);
    _demEdit->setObjectName(QStringLiteral("orthoDemPathEdit"));
    _demEdit->setPlaceholderText(tr("选择项目 DEM 或外部 DEM 栅格"));
    _demBrowseButton = new QPushButton(tr("浏览..."), _parametersGroup);
    _demBrowseButton->setObjectName(QStringLiteral("orthoDemBrowseButton"));
    _blendCombo = new QComboBox(_parametersGroup);
    _blendCombo->setObjectName(QStringLiteral("orthoBlendModeCombo"));
    _blendCombo->addItem(tr("马赛克（默认）"), QStringLiteral("mosaic"));
    _blendCombo->addItem(tr("加权平均"), QStringLiteral("weighted_average"));
    _blendCombo->addItem(tr("首个有效影像"), QStringLiteral("first_valid"));
    _colorSourceCombo = new QComboBox(_parametersGroup);
    _colorSourceCombo->setObjectName(QStringLiteral("orthoColorSourceCombo"));
    _colorSourceCombo->addItem(tr("影像"), QStringLiteral("images"));
    _colorSourceCombo->addItem(tr("点颜色 RGB"), QStringLiteral("point_colors"));
    _refineSeamsCheck = new QCheckBox(tr("完善接缝线"), _parametersGroup);
    _refineSeamsCheck->setObjectName(QStringLiteral("orthoRefineSeamsCheck"));
    _refineSeamsCheck->setEnabled(false);
    _refineSeamsCheck->setToolTip(
        tr("当前后端尚未实现全局接缝线优化，因此该选项不会作为运行参数提交。"));
    _fillHolesCheck = new QCheckBox(tr("启用孔洞填充"), _parametersGroup);
    _fillHolesCheck->setObjectName(QStringLiteral("orthoFillHolesCheck"));
    _fillHolesCheck->setChecked(true);
    _ghostFilterCheck = new QCheckBox(tr("启用重影过滤器"), _parametersGroup);
    _ghostFilterCheck->setObjectName(QStringLiteral("orthoGhostFilterCheck"));
    _colorCorrectionCheck = new QCheckBox(tr("自动色彩校正"), _parametersGroup);
    _colorCorrectionCheck->setObjectName(QStringLiteral("orthoColorCorrectionCheck"));
    _colorCorrectionCheck->setChecked(true);
    _sharpnessWeightingCheck = new QCheckBox(tr("启用离焦过滤"), _parametersGroup);
    _sharpnessWeightingCheck->setObjectName(QStringLiteral("orthoSharpnessWeightingCheck"));
    _sharpnessWeightingCheck->setToolTip(
        tr("降低模糊影像在多影像融合中的权重。"));
    _useProjectMasksCheck = new QCheckBox(tr("使用项目蒙版"), _parametersGroup);
    _useProjectMasksCheck->setObjectName(QStringLiteral("orthoUseProjectMasksCheck"));
    _useProjectMasksCheck->setEnabled(false);
    _useProjectMasksCheck->setToolTip(tr("当前项目没有可用蒙版。"));
    parametersForm->addRow(tr("表面:"), _surfaceCombo);
    parametersForm->addRow(tr("表面文件:"), pathRow(_demEdit, _demBrowseButton));
    parametersForm->addRow(tr("混合模式:"), _blendCombo);
    parametersForm->addRow(tr("颜色来源:"), _colorSourceCombo);
    parametersForm->addRow(QString(), _refineSeamsCheck);
    parametersForm->addRow(QString(), _fillHolesCheck);
    parametersForm->addRow(QString(), _ghostFilterCheck);
    parametersForm->addRow(QString(), _colorCorrectionCheck);
    parametersForm->addRow(QString(), _sharpnessWeightingCheck);
    parametersForm->addRow(QString(), _useProjectMasksCheck);

    _imageToggleButton = new QToolButton(_parametersGroup);
    _imageToggleButton->setObjectName(QStringLiteral("orthoImageListToggle"));
    _imageToggleButton->setCheckable(true);
    _imageToggleButton->setChecked(false);
    _imageToggleButton->setArrowType(Qt::RightArrow);
    _imageToggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    _imageToggleButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    parametersForm->addRow(QString(), _imageToggleButton);
    _imagePanel = new QWidget(_parametersGroup);
    _imagePanel->setObjectName(QStringLiteral("orthoImagePanel"));
    auto *imageLayout = new QVBoxLayout(_imagePanel);
    imageLayout->setContentsMargins(0, 0, 0, 0);
    imageLayout->setSpacing(4);
    _imageReadinessLabel = new QLabel(_imagePanel);
    _imageReadinessLabel->setObjectName(QStringLiteral("orthoImageReadinessLabel"));
    _imageReadinessLabel->setWordWrap(true);
    _imageList = new QListWidget(_imagePanel);
    _imageList->setObjectName(QStringLiteral("orthoImageList"));
    _imageList->setSelectionMode(QAbstractItemView::NoSelection);
    _imageList->setMinimumHeight(150);
    imageLayout->addWidget(_imageReadinessLabel);
    imageLayout->addWidget(_imageList);
    _imagePanel->setVisible(false);
    parametersForm->addRow(QString(), _imagePanel);
    contentLayout->addWidget(_parametersGroup);

    _regionGroup = new QGroupBox(tr("区域"), contentWidget);
    _regionGroup->setObjectName(QStringLiteral("orthoRegionGroup"));
    auto *regionForm = new QFormLayout(_regionGroup);
    xjw::gui::dialogs::configureWorkflowForm(regionForm);
    auto *resolutionModes = new QWidget(_regionGroup);
    auto *resolutionModeLayout = new QHBoxLayout(resolutionModes);
    resolutionModeLayout->setContentsMargins(0, 0, 0, 0);
    _pixelSizeRadio = new QRadioButton(tr("像元大小"), resolutionModes);
    _maximumDimensionRadio = new QRadioButton(tr("最大尺寸"), resolutionModes);
    _pixelSizeRadio->setObjectName(QStringLiteral("orthoPixelSizeModeRadio"));
    _maximumDimensionRadio->setObjectName(QStringLiteral("orthoMaximumDimensionModeRadio"));
    _pixelSizeRadio->setChecked(true);
    resolutionModeLayout->addWidget(_pixelSizeRadio);
    resolutionModeLayout->addWidget(_maximumDimensionRadio);
    resolutionModeLayout->addStretch(1);

    auto *pixelSizeWidget = new QWidget(_regionGroup);
    auto *pixelSizeLayout = new QGridLayout(pixelSizeWidget);
    pixelSizeLayout->setContentsMargins(0, 0, 0, 0);
    pixelSizeLayout->setHorizontalSpacing(6);
    _pixelSizeXSpin = new QDoubleSpinBox(pixelSizeWidget);
    _pixelSizeYSpin = new QDoubleSpinBox(pixelSizeWidget);
    for (QDoubleSpinBox *spinBox : {_pixelSizeXSpin, _pixelSizeYSpin})
    {
        spinBox->setRange(0.000001, 1.0e9);
        spinBox->setDecimals(9);
        spinBox->setValue(1.0);
        spinBox->setSuffix(tr(" m/px"));
        spinBox->setKeyboardTracking(false);
    }
    _pixelSizeXSpin->setObjectName(QStringLiteral("orthoPixelSizeXSpin"));
    _pixelSizeYSpin->setObjectName(QStringLiteral("orthoPixelSizeYSpin"));
    _restorePixelSizeButton = new QPushButton(tr("恢复 DEM 值"), pixelSizeWidget);
    _restorePixelSizeButton->setObjectName(QStringLiteral("orthoRestoreDemPixelSizeButton"));
    pixelSizeLayout->addWidget(new QLabel(tr("X:"), pixelSizeWidget), 0, 0);
    pixelSizeLayout->addWidget(_pixelSizeXSpin, 0, 1);
    pixelSizeLayout->addWidget(new QLabel(tr("Y:"), pixelSizeWidget), 1, 0);
    pixelSizeLayout->addWidget(_pixelSizeYSpin, 1, 1);
    pixelSizeLayout->addWidget(_restorePixelSizeButton, 0, 2, 2, 1);

    _maximumDimensionSpin = new QSpinBox(_regionGroup);
    _maximumDimensionSpin->setObjectName(QStringLiteral("orthoMaximumDimensionSpin"));
    _maximumDimensionSpin->setRange(1, 1000000);
    _maximumDimensionSpin->setValue(4096);
    _maximumDimensionSpin->setSuffix(tr(" px"));
    _maximumDimensionSpin->setKeyboardTracking(false);

    _boundsEnabledCheck = new QCheckBox(tr("设置边界"), _regionGroup);
    _boundsEnabledCheck->setObjectName(QStringLiteral("orthoBoundsEnabledCheck"));
    auto *boundsWidget = new QWidget(_regionGroup);
    auto *boundsLayout = new QGridLayout(boundsWidget);
    boundsLayout->setContentsMargins(0, 0, 0, 0);
    boundsLayout->setHorizontalSpacing(6);
    boundsLayout->setVerticalSpacing(4);
    _minXSpin = new QDoubleSpinBox(boundsWidget);
    _maxXSpin = new QDoubleSpinBox(boundsWidget);
    _minYSpin = new QDoubleSpinBox(boundsWidget);
    _maxYSpin = new QDoubleSpinBox(boundsWidget);
    configureCoordinateSpin(_minXSpin, QStringLiteral("orthoMinXSpin"));
    configureCoordinateSpin(_maxXSpin, QStringLiteral("orthoMaxXSpin"));
    configureCoordinateSpin(_minYSpin, QStringLiteral("orthoMinYSpin"));
    configureCoordinateSpin(_maxYSpin, QStringLiteral("orthoMaxYSpin"));
    _resetBoundsButton = new QPushButton(tr("重置为 DEM 范围"), boundsWidget);
    _resetBoundsButton->setObjectName(QStringLiteral("orthoResetBoundsButton"));
    boundsLayout->addWidget(new QLabel(tr("X:"), boundsWidget), 0, 0);
    boundsLayout->addWidget(_minXSpin, 0, 1);
    boundsLayout->addWidget(new QLabel(QStringLiteral("—"), boundsWidget), 0, 2);
    boundsLayout->addWidget(_maxXSpin, 0, 3);
    boundsLayout->addWidget(new QLabel(tr("Y:"), boundsWidget), 1, 0);
    boundsLayout->addWidget(_minYSpin, 1, 1);
    boundsLayout->addWidget(new QLabel(QStringLiteral("—"), boundsWidget), 1, 2);
    boundsLayout->addWidget(_maxYSpin, 1, 3);
    boundsLayout->addWidget(_resetBoundsButton, 2, 1, 1, 3);

    auto *estimateWidget = new QWidget(_regionGroup);
    auto *estimateLayout = new QHBoxLayout(estimateWidget);
    estimateLayout->setContentsMargins(0, 0, 0, 0);
    _estimateButton = new QPushButton(tr("预计"), estimateWidget);
    _estimateButton->setObjectName(QStringLiteral("orthoEstimateButton"));
    _totalSizeLabel = new QLabel(tr("总尺寸（像素）：等待 DEM"), estimateWidget);
    _totalSizeLabel->setObjectName(QStringLiteral("orthoTotalSizeLabel"));
    estimateLayout->addWidget(_estimateButton);
    estimateLayout->addWidget(_totalSizeLabel, 1);
    _memoryEstimateLabel = new QLabel(tr("预计处理内存：未知"), _regionGroup);
    _memoryEstimateLabel->setObjectName(QStringLiteral("orthoMemoryEstimateLabel"));
    regionForm->addRow(tr("分辨率模式:"), resolutionModes);
    regionForm->addRow(tr("像元大小:"), pixelSizeWidget);
    regionForm->addRow(tr("最大尺寸:"), _maximumDimensionSpin);
    regionForm->addRow(QString(), _boundsEnabledCheck);
    regionForm->addRow(tr("边界:"), boundsWidget);
    regionForm->addRow(QString(), estimateWidget);
    regionForm->addRow(QString(), _memoryEstimateLabel);
    contentLayout->addWidget(_regionGroup);

    _outputGroup = new QGroupBox(tr("输出"), contentWidget);
    _outputGroup->setObjectName(QStringLiteral("orthoOutputGroup"));
    auto *outputForm = new QFormLayout(_outputGroup);
    xjw::gui::dialogs::configureWorkflowForm(outputForm);
    _outputEdit = new QLineEdit(_outputGroup);
    _outputEdit->setObjectName(QStringLiteral("orthoOutputPathEdit"));
    _outputEdit->setPlaceholderText(tr("GeoTIFF 输出路径"));
    _outputBrowseButton = new QPushButton(tr("浏览..."), _outputGroup);
    _outputBrowseButton->setObjectName(QStringLiteral("orthoOutputBrowseButton"));
    auto *outputHint = new QLabel(
        tr("建议输出 GeoTIFF（.tif），以保留最终网格、Alpha 和坐标参考。"),
        _outputGroup);
    outputHint->setObjectName(QStringLiteral("orthoOutputHintLabel"));
    outputHint->setWordWrap(true);
    outputForm->addRow(tr("正射影像:"), pathRow(_outputEdit, _outputBrowseButton));
    outputForm->addRow(QString(), outputHint);
    contentLayout->addWidget(_outputGroup);

    _progressGroup = new QGroupBox(tr("进度"), contentWidget);
    _progressGroup->setObjectName(QStringLiteral("orthoProgressGroup"));
    auto *progressLayout = new QVBoxLayout(_progressGroup);
    _stageLabel = new QLabel(tr("等待开始"), _progressGroup);
    _stageLabel->setObjectName(QStringLiteral("orthoStageLabel"));
    _progressBar = new QProgressBar(_progressGroup);
    _progressBar->setObjectName(QStringLiteral("orthoProgressBar"));
    _progressBar->setRange(0, 100);
    _progressBar->setValue(0);
    _statusLabel = new QLabel(
        tr("检查参数后点击“生成”开始处理；任务运行期间可安全取消。"),
        _progressGroup);
    _statusLabel->setObjectName(QStringLiteral("orthoStatusLabel"));
    _statusLabel->setWordWrap(true);
    _statusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    progressLayout->addWidget(_stageLabel);
    progressLayout->addWidget(_progressBar);
    progressLayout->addWidget(_statusLabel);
    contentLayout->addWidget(_progressGroup);
    contentLayout->addStretch(1);

    _contentScrollArea->setWidget(contentWidget);
    mainLayout->addWidget(_contentScrollArea, 1);

    _buttonBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    _buttonBox->setObjectName(QStringLiteral("orthoButtonBox"));
    xjw::gui::dialogs::configureWorkflowButtonBox(_buttonBox);
    _runButton = _buttonBox->button(QDialogButtonBox::Ok);
    _cancelButton = _buttonBox->button(QDialogButtonBox::Cancel);
    _runButton->setObjectName(QStringLiteral("orthoRunButton"));
    _cancelButton->setObjectName(QStringLiteral("orthoCancelButton"));
    _runButton->setText(tr("生成"));
    _cancelButton->setText(tr("取消"));
    mainLayout->addWidget(_buttonBox);
}
