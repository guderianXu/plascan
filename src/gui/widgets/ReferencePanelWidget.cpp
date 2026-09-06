#include "ReferencePanelWidget.h"

#include "MarkerWorkspaceController.h"
#include "ReferenceMarkerModels.h"
#include "reference/ProjectCameraReferenceRepository.h"

#include <QAction>
#include <QActionGroup>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QMenu>
#include <QMessageBox>
#include <QTabBar>
#include <QTabWidget>
#include <QToolBar>
#include <QToolButton>
#include <QTreeView>
#include <QVBoxLayout>

#include <algorithm>

namespace
{

QTreeView *createReferenceTree(const QString &objectName, QWidget *parent)
{
    auto *tree = new QTreeView(parent);
    tree->setObjectName(objectName);
    tree->setAlternatingRowColors(true);
    tree->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tree->setSelectionBehavior(QAbstractItemView::SelectRows);
    tree->setSelectionMode(QAbstractItemView::SingleSelection);
    tree->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    tree->setTextElideMode(Qt::ElideMiddle);
    tree->setUniformRowHeights(true);
    tree->header()->setStretchLastSection(false);
    tree->header()->setMinimumSectionSize(56);
    tree->header()->setSectionResizeMode(QHeaderView::Interactive);
    tree->header()->setDefaultSectionSize(92);
    return tree;
}

void showOnlyColumns(QTreeView *tree, const QList<int> &visibleColumns, int columnCount)
{
    if (!tree)
    {
        return;
    }
    for (int column = 0; column < columnCount; ++column)
    {
        tree->setColumnHidden(column, !visibleColumns.contains(column));
    }
}

} // namespace

ReferencePanelWidget::ReferencePanelWidget(QWidget *parent)
    : QWidget(parent)
{
    buildInterface();
    refreshAll();
}

ReferencePanelWidget::~ReferencePanelWidget() = default;

void ReferencePanelWidget::setCameraReferenceRepository(
    xjw::gui::reference::ProjectCameraReferenceRepository *repository)
{
    if (_cameraRepository)
    {
        disconnect(_cameraRepository, nullptr, this, nullptr);
    }
    _cameraRepository = repository;
    if (_cameraRepository)
    {
        connect(_cameraRepository,
                &xjw::gui::reference::ProjectCameraReferenceRepository::referenceSetChanged,
                this,
                [this](quint64)
        {
            refreshCameraReferences();
            updateActionAvailability();
        });
    }
    refreshCameraReferences();
    updateActionAvailability();
}

void ReferencePanelWidget::setMarkerController(
    xjw::gui::markers::MarkerWorkspaceController *controller)
{
    if (_markerController)
    {
        disconnect(_markerController, nullptr, this, nullptr);
    }
    _markerController = controller;
    if (_markerController)
    {
        connect(_markerController,
                &xjw::gui::markers::MarkerWorkspaceController::markerSetChanged,
                this,
                [this]()
        {
            refreshMarkerReferences();
            updateActionAvailability();
        });
    }
    refreshMarkerReferences();
    updateActionAvailability();
}

void ReferencePanelWidget::loadFromJson(const QJsonObject &metadata)
{
    _projectMetadata = metadata;
    refreshCameraReferences();
    updateActionAvailability();
}

void ReferencePanelWidget::clearProject()
{
    _projectMetadata = {};
    refreshAll();
}

void ReferencePanelWidget::buildInterface()
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(3);

    auto* actionToolbar = new QToolBar(this);
    actionToolbar->setObjectName(QStringLiteral("referenceActionToolbar"));
    actionToolbar->setToolButtonStyle(Qt::ToolButtonTextOnly);
    actionToolbar->setMovable(false);

    _importCameraAction = new QAction(tr("相机参考"), this);
    _importCameraAction->setObjectName(QStringLiteral("importCameraReferencesAction"));
    _importMarkerAction = new QAction(tr("标记参考"), this);
    _importMarkerAction->setObjectName(QStringLiteral("importMarkerReferencesAction"));

    auto* importMenu = new QMenu(actionToolbar);
    importMenu->addAction(_importCameraAction);
    importMenu->addAction(_importMarkerAction);
    auto* importButton = new QToolButton(actionToolbar);
    importButton->setObjectName(QStringLiteral("referenceImportButton"));
    importButton->setText(tr("导入"));
    importButton->setMenu(importMenu);
    importButton->setPopupMode(QToolButton::InstantPopup);
    actionToolbar->addWidget(importButton);

    _exportCameraAction = actionToolbar->addAction(tr("导出"));
    _exportCameraAction->setObjectName(QStringLiteral("exportCameraReferencesAction"));
    _settingsAction = actionToolbar->addAction(tr("设置"));
    _settingsAction->setObjectName(QStringLiteral("cameraReferenceSettingsAction"));
    _toggleCameraAction = new QAction(tr("启用/禁用"), this);
    _toggleCameraAction->setObjectName(QStringLiteral("toggleCameraReferenceAction"));
    _editMarkerAction = new QAction(tr("标记属性"), this);
    _editMarkerAction->setObjectName(QStringLiteral("editMarkerReferenceAction"));

    auto* moreMenu = new QMenu(actionToolbar);
    moreMenu->addAction(_toggleCameraAction);
    moreMenu->addAction(_editMarkerAction);
    auto* moreButton = new QToolButton(actionToolbar);
    moreButton->setObjectName(QStringLiteral("referenceMoreButton"));
    moreButton->setText(tr("更多"));
    moreButton->setMenu(moreMenu);
    moreButton->setPopupMode(QToolButton::InstantPopup);
    actionToolbar->addWidget(moreButton);
    layout->addWidget(actionToolbar);

    auto *modeGroup = new QActionGroup(this);
    modeGroup->setExclusive(true);
    _sourceModeAction = new QAction(tr("源值"), this);
    _estimatedModeAction = new QAction(tr("估计值"), this);
    _errorModeAction = new QAction(tr("误差"), this);
    for (QAction *action : {_sourceModeAction, _estimatedModeAction, _errorModeAction})
    {
        action->setCheckable(true);
        modeGroup->addAction(action);
    }
    _sourceModeAction->setObjectName(QStringLiteral("referenceSourceModeAction"));
    _estimatedModeAction->setObjectName(QStringLiteral("referenceEstimatedModeAction"));
    _errorModeAction->setObjectName(QStringLiteral("referenceErrorModeAction"));
    _sourceModeAction->setChecked(true);

    auto* modeTabs = new QTabBar(this);
    modeTabs->setObjectName(QStringLiteral("referenceModeTabs"));
    modeTabs->setDocumentMode(true);
    modeTabs->setExpanding(true);
    modeTabs->setUsesScrollButtons(false);
    modeTabs->addTab(_sourceModeAction->text());
    modeTabs->addTab(_estimatedModeAction->text());
    modeTabs->addTab(_errorModeAction->text());
    layout->addWidget(modeTabs);

    auto* dataTabs = new QTabWidget(this);
    dataTabs->setObjectName(QStringLiteral("referenceDataTabs"));
    dataTabs->setDocumentMode(true);
    _cameraTree = createReferenceTree(QStringLiteral("cameraReferenceTree"), dataTabs);
    _markerTree = createReferenceTree(QStringLiteral("markerReferenceTree"), dataTabs);
    _scaleBarTree = createReferenceTree(QStringLiteral("scaleBarReferenceTree"), dataTabs);
    dataTabs->addTab(_cameraTree, tr("相机"));
    dataTabs->addTab(_markerTree, tr("标记"));
    dataTabs->addTab(_scaleBarTree, tr("标尺"));
    layout->addWidget(dataTabs, 1);

    _cameraModel = new xjw::gui::reference::CameraReferenceTreeModel(this);
    _markerModel = new xjw::gui::reference::MarkerReferenceTreeModel(this);
    _scaleBarModel = new xjw::gui::reference::ScaleBarReferenceTreeModel(this);
    _cameraTree->setModel(_cameraModel);
    _markerTree->setModel(_markerModel);
    _scaleBarTree->setModel(_scaleBarModel);
    _cameraTree->header()->resizeSection(0, 168);
    _markerTree->header()->resizeSection(0, 168);
    _scaleBarTree->header()->resizeSection(0, 168);

    connect(_importCameraAction, &QAction::triggered,
            this, &ReferencePanelWidget::importCameraReferencesRequested);
    connect(_importMarkerAction, &QAction::triggered,
            this, &ReferencePanelWidget::importMarkerReferencesRequested);
    connect(_exportCameraAction, &QAction::triggered,
            this, &ReferencePanelWidget::exportCameraReferencesRequested);
    connect(_settingsAction, &QAction::triggered,
            this, &ReferencePanelWidget::cameraReferenceSettingsRequested);
    connect(modeTabs,
            &QTabBar::currentChanged,
            this,
            [this](int index)
            {
                const auto mode = index == 1   ? xjw::gui::reference::ReferenceDisplayMode::Estimated
                                  : index == 2 ? xjw::gui::reference::ReferenceDisplayMode::Error
                                               : xjw::gui::reference::ReferenceDisplayMode::Source;
                QAction* action = index == 1 ? _estimatedModeAction : index == 2 ? _errorModeAction : _sourceModeAction;
                action->setChecked(true);
                applyMode(mode);
            });
    connect(_sourceModeAction, &QAction::triggered, modeTabs, [modeTabs]() { modeTabs->setCurrentIndex(0); });
    connect(_estimatedModeAction, &QAction::triggered, modeTabs, [modeTabs]() { modeTabs->setCurrentIndex(1); });
    connect(_errorModeAction, &QAction::triggered, modeTabs, [modeTabs]() { modeTabs->setCurrentIndex(2); });
    connect(_toggleCameraAction, &QAction::triggered, this, [this]()
    {
        const QString imageUuid = selectedCameraUuid();
        if (!_cameraRepository || imageUuid.isEmpty())
        {
            return;
        }
        const auto &records = _cameraRepository->referenceSet().records();
        const auto iterator = std::find_if(records.cbegin(), records.cend(),
                                           [&imageUuid](const auto &record)
        {
            return record.imageUuid == imageUuid;
        });
        if (iterator == records.cend())
        {
            return;
        }
        QString error;
        if (!_cameraRepository->setRecordEnabled(imageUuid, !iterator->enabled, &error))
        {
            QMessageBox::warning(this, tr("相机参考"), error);
        }
    });
    connect(_editMarkerAction, &QAction::triggered, this, [this]()
    {
        const QString markerId = selectedMarkerId();
        if (!markerId.isEmpty())
        {
            emit markerPropertiesRequested(markerId);
        }
    });
    connect(_cameraTree, &QTreeView::doubleClicked, this, [this](const QModelIndex &index)
    {
        const QString path = index.data(
            xjw::gui::reference::CameraReferenceTreeModel::ImagePathRole).toString();
        if (!path.isEmpty())
        {
            emit imageActivated(path);
        }
    });
    connect(_markerTree, &QTreeView::doubleClicked, this, [this](const QModelIndex &index)
    {
        const QString markerId = index.data(
            xjw::gui::reference::MarkerReferenceTreeModel::MarkerIdRole).toString();
        if (!markerId.isEmpty())
        {
            emit markerActivated(markerId);
        }
    });
    connect(_cameraTree->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, [this]() { updateActionAvailability(); });
    connect(_markerTree->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, [this]() { updateActionAvailability(); });
}

void ReferencePanelWidget::refreshAll()
{
    refreshCameraReferences();
    refreshMarkerReferences();
    updateActionAvailability();
}

void ReferencePanelWidget::refreshCameraReferences()
{
    const xjw::camera_reference::CameraReferenceSet empty;
    const auto &referenceSet = _cameraRepository
        ? _cameraRepository->referenceSet()
        : empty;
    _cameraModel->setReferenceData(referenceSet, _projectMetadata, _mode);
    _cameraTree->expandAll();
    applyMode(_mode);
}

void ReferencePanelWidget::refreshMarkerReferences()
{
    const xjw::control_points::MarkerSet empty;
    const auto &markerSet = _markerController
        ? _markerController->markerSet()
        : empty;
    _markerModel->setMarkerSet(markerSet);
    _scaleBarModel->setMarkerSet(markerSet);
    _markerTree->expandAll();
    _scaleBarTree->expandAll();
    applyMode(_mode);
}

void ReferencePanelWidget::applyMode(xjw::gui::reference::ReferenceDisplayMode mode)
{
    _mode = mode;
    if (_cameraRepository && _cameraModel)
    {
        _cameraModel->setReferenceData(
            _cameraRepository->referenceSet(), _projectMetadata, _mode);
        _cameraTree->expandAll();
    }

    using CameraTreeModel = xjw::gui::reference::CameraReferenceTreeModel;
    using Marker = xjw::gui::reference::MarkerReferenceTreeModel;
    using Scale = xjw::gui::reference::ScaleBarReferenceTreeModel;
    QList<int> cameraColumns{CameraTreeModel::LabelColumn, CameraTreeModel::XColumn, CameraTreeModel::YColumn,
                             CameraTreeModel::ZColumn, CameraTreeModel::YawColumn, CameraTreeModel::PitchColumn,
                             CameraTreeModel::RollColumn, CameraTreeModel::StatusColumn, CameraTreeModel::EnabledColumn};
    QList<int> markerColumns{Marker::LabelColumn};
    QList<int> scaleColumns{Scale::LabelColumn};
    if (mode == xjw::gui::reference::ReferenceDisplayMode::Source)
    {
        cameraColumns.insert(7, CameraTreeModel::HorizontalAccuracyColumn);
        cameraColumns.insert(8, CameraTreeModel::VerticalAccuracyColumn);
        markerColumns << Marker::SourceXColumn << Marker::SourceYColumn
                      << Marker::SourceZColumn << Marker::AccuracyXColumn
                      << Marker::AccuracyYColumn << Marker::AccuracyZColumn
                      << Marker::EnabledColumn;
        scaleColumns << Scale::FirstMarkerColumn << Scale::SecondMarkerColumn
                     << Scale::SourceValueColumn << Scale::AccuracyColumn
                     << Scale::EnabledColumn;
    }
    else if (mode == xjw::gui::reference::ReferenceDisplayMode::Estimated)
    {
        markerColumns << Marker::ProjectionCountColumn << Marker::ResidualColumn;
        scaleColumns << Scale::EstimatedValueColumn;
    }
    else
    {
        markerColumns << Marker::ResidualColumn;
        scaleColumns << Scale::ResidualColumn;
    }
    showOnlyColumns(_cameraTree, cameraColumns, CameraTreeModel::ColumnCount);
    showOnlyColumns(_markerTree, markerColumns, Marker::ColumnCount);
    showOnlyColumns(_scaleBarTree, scaleColumns, Scale::ColumnCount);
    updateActionAvailability();
}

void ReferencePanelWidget::updateActionAvailability()
{
    const bool hasProject = !_projectMetadata.isEmpty();
    const bool hasCameraReferences = _cameraRepository
        && (!_cameraRepository->referenceSet().records().isEmpty()
            || !_cameraRepository->referenceSet().unmatchedRecords().isEmpty());
    _importCameraAction->setEnabled(hasProject && _cameraRepository);
    _importMarkerAction->setEnabled(hasProject && _markerController);
    _exportCameraAction->setEnabled(hasProject && hasCameraReferences);
    const QModelIndex cameraIndex = _cameraTree ? _cameraTree->currentIndex() : QModelIndex();
    const bool editableCameraReference = cameraIndex.isValid()
        && cameraIndex.data(xjw::gui::reference::CameraReferenceTreeModel::NodeTypeRole).toInt()
            == static_cast<int>(xjw::gui::reference::CameraReferenceTreeModel::NodeType::Camera);
    _toggleCameraAction->setEnabled(editableCameraReference && !selectedCameraUuid().isEmpty());
    _editMarkerAction->setEnabled(!selectedMarkerId().isEmpty());
    _settingsAction->setEnabled(hasProject && _cameraRepository);
}

QString ReferencePanelWidget::selectedCameraUuid() const
{
    return _cameraTree && _cameraTree->currentIndex().isValid()
        ? _cameraTree->currentIndex()
              .data(xjw::gui::reference::CameraReferenceTreeModel::ImageUuidRole)
              .toString()
        : QString();
}

QString ReferencePanelWidget::selectedMarkerId() const
{
    return _markerTree && _markerTree->currentIndex().isValid()
        ? _markerTree->currentIndex()
              .data(xjw::gui::reference::MarkerReferenceTreeModel::MarkerIdRole)
              .toString()
        : QString();
}
