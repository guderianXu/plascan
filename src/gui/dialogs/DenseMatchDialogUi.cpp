// =============================================================================
// 文件: DenseMatchDialogUi.cpp
// 功能: DenseMatchDialog::setupUi() 实现 — 通过 Qt Designer .ui 构建静态布局
// =============================================================================
#include "DenseMatchDialog.h"
#include "ui_DenseMatchDialog.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>

void DenseMatchDialog::setupUi()
{
    Ui::DenseMatchDialog ui;
    ui.setupUi(this);

    _imageList = ui.m_imageList;
    _selectAllBtn = ui.m_selectAllBtn;
    _deselectAllBtn = ui.m_deselectAllBtn;
    _matchTable = ui.m_matchTable;
    _matchCountLabel = ui.m_matchCountLabel;
    _outputEdit = ui.m_outputEdit;

    _algorithmCombo = ui.m_algorithmCombo;
    _costFuncCombo = ui.m_costFuncCombo;
    _subpixelCombo = ui.m_subpixelCombo;
    _minDispSpin = ui.m_minDispSpin;
    _maxDispSpin = ui.m_maxDispSpin;
    _kernelWSpin = ui.m_kernelWSpin;
    _kernelHSpin = ui.m_kernelHSpin;
    _p1Spin = ui.m_p1Spin;
    _p2Spin = ui.m_p2Spin;
    _directionsSpin = ui.m_directionsSpin;
    _pyramidSpin = ui.m_pyramidSpin;
    _useCudaChk = ui.m_useCudaChk;
    _deviceSpin = ui.m_deviceSpin;
    _threadsSpin = ui.m_threadsSpin;
    _opencvCompareChk = ui.m_opencvCompareChk;
    _lrThresholdSpin = ui.m_lrThresholdSpin;
    _medianFilterSpin = ui.m_medianFilterSpin;

    _runBtn = ui.m_runBtn;
    _cancelBtn = ui.m_cancelBtn;
    _resetBtn = ui.m_resetBtn;

    if (ui.mainSplitter)
    {
        ui.mainSplitter->setStretchFactor(0, 2);
        ui.mainSplitter->setStretchFactor(1, 3);
    }

    _matchTable->horizontalHeader()->setStretchLastSection(true);
    _matchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _matchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _matchTable->setColumnWidth(0, 140);
    _matchTable->setColumnWidth(1, 140);
    _matchTable->setColumnWidth(2, 60);

    _algorithmCombo->clear();
    _algorithmCombo->addItem(tr("MGM (More Global Match)"), 2);
    _algorithmCombo->addItem(tr("SGM (Semi Global Match)"), 1);
    _algorithmCombo->addItem(tr("BM (Block Match)"), 0);
    _algorithmCombo->addItem(tr("OpenCV SGBM"), 3);

    _costFuncCombo->clear();
    _costFuncCombo->addItem(tr("Census Transform"), 3);
    _costFuncCombo->addItem(tr("NCC"), 2);
    _costFuncCombo->addItem(tr("Absolute Difference"), 0);
    _costFuncCombo->addItem(tr("Squared Difference"), 1);
    _costFuncCombo->addItem(tr("Ternary Census"), 4);

    _subpixelCombo->clear();
    _subpixelCombo->addItem(tr("Parabola Fitting"), 1);
    _subpixelCombo->addItem(tr("None"), 0);

    connect(_imageList, &QListWidget::itemChanged,
            this, &DenseMatchDialog::onImageSelectionChanged);
    connect(_selectAllBtn, &QPushButton::clicked, this, &DenseMatchDialog::onSelectAll);
    connect(_deselectAllBtn, &QPushButton::clicked, this, &DenseMatchDialog::onDeselectAll);
    connect(ui.m_browseOutputBtn, &QPushButton::clicked, this, &DenseMatchDialog::onBrowseOutput);

    connect(_runBtn, &QPushButton::clicked, this, &DenseMatchDialog::onRun);
    connect(_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

    connect(_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseMatchDialog::onAlgorithmChanged);
    connect(_costFuncCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseMatchDialog::emitSettingsNow);
    connect(_subpixelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseMatchDialog::emitSettingsNow);
    connect(_outputEdit, &QLineEdit::textChanged, this, &DenseMatchDialog::emitSettingsNow);

    const auto emitSpinChanged = [this](auto *spin)
    {
        connect(spin, QOverload<int>::of(&QSpinBox::valueChanged),
                this, &DenseMatchDialog::emitSettingsNow);
    };
    emitSpinChanged(_minDispSpin);
    emitSpinChanged(_maxDispSpin);
    emitSpinChanged(_kernelWSpin);
    emitSpinChanged(_kernelHSpin);
    emitSpinChanged(_p1Spin);
    emitSpinChanged(_p2Spin);
    emitSpinChanged(_directionsSpin);
    emitSpinChanged(_pyramidSpin);
    emitSpinChanged(_deviceSpin);
    emitSpinChanged(_threadsSpin);
    emitSpinChanged(_medianFilterSpin);

    connect(_useCudaChk, &QCheckBox::toggled, this, &DenseMatchDialog::emitSettingsNow);
    connect(_opencvCompareChk, &QCheckBox::toggled, this, &DenseMatchDialog::emitSettingsNow);
    connect(_lrThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &DenseMatchDialog::emitSettingsNow);

    onAlgorithmChanged(_algorithmCombo->currentIndex());
}
