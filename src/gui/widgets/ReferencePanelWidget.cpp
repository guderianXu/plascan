#include "ReferencePanelWidget.h"

#include "MarkerWorkspaceController.h"
#include "ReferenceMarkerModels.h"
#include "reference/ProjectCameraReferenceRepository.h"

#include <QAction>
#include <QActionGroup>
#include <QGroupBox>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QMessageBox>
#include <QSplitter>
#include <QToolBar>
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
    tree->setUniformRowHeights(true);
    tree->header()->setStretchLastSection(false);
    tree->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
    return tree;
}

QGroupBox *createSection(const QString &title, QTreeView *tree, QWidget *parent)
{
    auto *section = new QGroupBox(title, parent);
    auto *layout = new QVBoxLayout(section);
    layout->setContentsMargins(0, 2, 0, 0);
    layout->addWidget(tree);
    return section;
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

    auto *toolbar = new QToolBar(this);
    toolbar->setObjectName(QStringLiteral("referenceToolbar"));
    toolbar->setToolButtonStyle(Qt::ToolButtonTextOnly);
    toolbar->setMovable(false);
    _importCameraAction = toolbar->addAction(tr("导入相机参考"));
    _importCameraAction->setObjectName(QStringLiteral("importCameraReferencesAction"));
    _importMarkerAction = toolbar->addAction(tr("导入标记参考"));
    _importMarkerAction->setObjectName(QStringLiteral("importMarkerReferencesAction"));
    _exportCameraAction = toolbar->addAction(tr("导出"));
    _exportCameraAction->setObjectName(QStringLiteral("exportCameraReferencesAction"));
    toolbar->addSeparator();
    _toggleCameraAction = toolbar->addAction(tr("启用/禁用"));
    _toggleCameraAction->setObjectName(QStringLiteral("toggleCameraReferenceAction"));
    _editMarkerAction = toolbar->addAction(tr("标记属性"));
    _editMarkerAction->setObjectName(QStringLiteral("editMarkerReferenceAction"));
    toolbar->addSeparator();

    auto *modeGroup = new QActionGroup(this);
    modeGroup->setExclusive(true);
    _sourceModeAction = toolbar->addAction(tr("源值"));
    _estimatedModeAction = toolbar->addAction(tr("估计值"));
    _errorModeAction = toolbar->addAction(tr("误差"));
    for (QAction *action : {_sourceModeAction, _estimatedModeAction, _errorModeAction})
    {
        action->setCheckable(true);
        modeGroup->addAction(action);
    }
    _sourceModeAction->setObjectName(QStringLiteral("referenceSourceModeAction"));
    _estimatedModeAction->setObjectName(QStringLiteral("referenceEstimatedModeAction"));
    _errorModeAction->setObjectName(QStringLiteral("referenceErrorModeAction"));
    _sourceModeAction->setChecked(true);
    toolbar->addSeparator();
    _settingsAction = toolbar->addAction(tr("设置"));
    _settingsAction->setObjectName(QStringLiteral("cameraReferenceSettingsAction"));
    layout->addWidget(toolbar);

    auto *splitter = new QSplitter(Qt::Vertical, this);
    splitter->setObjectName(QStringLiteral("referenceSectionSplitter"));
    splitter->setChildrenCollapsible(false);
    _cameraTree = createReferenceTree(QStringLiteral("cameraReferenceTree"), splitter);
    _markerTree = createReferenceTree(QStringLiteral("markerReferenceTree"), splitter);
    _scaleBarTree = createReferenceTree(QStringLiteral("scaleBarReferenceTree"), splitter);
    splitter->addWidget(createSection(tr("相机参考"), _cameraTree, splitter));
    splitter->addWidget(createSection(tr("标记"), _markerTree, splitter));
    splitter->addWidget(createSection(tr("标尺"), _scaleBarTree, splitter));
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);
    splitter->setStretchFactor(2, 1);
    layout->addWidget(splitter, 1);

    _cameraModel = new xjw::gui::reference::CameraReferenceTreeModel(this);
    _markerModel = new xjw::gui::reference::MarkerReferenceTreeModel(this);
    _scaleBarModel = new xjw::gui::reference::ScaleBarReferenceTreeModel(this);
    _cameraTree->setModel(_cameraModel);
    _markerTree->setModel(_markerModel);
    _scaleBarTree->setModel(_scaleBarModel);

    connect(_importCameraAction, &QAction::triggered,
            this, &ReferencePanelWidget::importCameraReferencesRequested);
    connect(_importMarkerAction, &QAction::triggered,
            this, &ReferencePanelWidget::importMarkerReferencesRequested);
    connect(_exportCameraAction, &QAction::triggered,
            this, &ReferencePanelWidget::exportCameraReferencesRequested);
    connect(_settingsAction, &QAction::triggered,
            this, &ReferencePanelWidget::cameraReferenceSettingsRequested);
    connect(_sourceModeAction, &QAction::triggered, this, [this]()
    {
        applyMode(xjw::gui::reference::ReferenceDisplayMode::Source);
    });
    connect(_estimatedModeAction, &QAction::triggered, this, [this]()
    {
        applyMode(xjw::gui::reference::ReferenceDisplayMode::Estimated);
    });
    connect(_errorModeAction, &QAction::triggered, this, [this]()
    {
        applyMode(xjw::gui::reference::ReferenceDisplayMode::Error);
    });
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

    using Camera = xjw::gui::reference::CameraReferenceTreeModel;
    using Marker = xjw::gui::reference::MarkerReferenceTreeModel;
    using Scale = xjw::gui::reference::ScaleBarReferenceTreeModel;
    QList<int> cameraColumns{Camera::LabelColumn, Camera::XColumn, Camera::YColumn,
                             Camera::ZColumn, Camera::YawColumn, Camera::PitchColumn,
                             Camera::RollColumn, Camera::StatusColumn, Camera::EnabledColumn};
    QList<int> markerColumns{Marker::LabelColumn};
    QList<int> scaleColumns{Scale::LabelColumn};
    if (mode == xjw::gui::reference::ReferenceDisplayMode::Source)
    {
        cameraColumns.insert(7, Camera::HorizontalAccuracyColumn);
        cameraColumns.insert(8, Camera::VerticalAccuracyColumn);
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
    showOnlyColumns(_cameraTree, cameraColumns, Camera::ColumnCount);
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
    _toggleCameraAction->setEnabled(!selectedCameraUuid().isEmpty());
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
