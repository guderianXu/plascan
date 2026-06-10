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

    m_imageList = ui.m_imageList;
    m_selectAllBtn = ui.m_selectAllBtn;
    m_deselectAllBtn = ui.m_deselectAllBtn;
    m_matchTable = ui.m_matchTable;
    m_matchCountLabel = ui.m_matchCountLabel;
    m_outputEdit = ui.m_outputEdit;

    m_algorithmCombo = ui.m_algorithmCombo;
    m_costFuncCombo = ui.m_costFuncCombo;
    m_subpixelCombo = ui.m_subpixelCombo;
    m_minDispSpin = ui.m_minDispSpin;
    m_maxDispSpin = ui.m_maxDispSpin;
    m_kernelWSpin = ui.m_kernelWSpin;
    m_kernelHSpin = ui.m_kernelHSpin;
    m_p1Spin = ui.m_p1Spin;
    m_p2Spin = ui.m_p2Spin;
    m_directionsSpin = ui.m_directionsSpin;
    m_pyramidSpin = ui.m_pyramidSpin;
    m_useCudaChk = ui.m_useCudaChk;
    m_deviceSpin = ui.m_deviceSpin;
    m_threadsSpin = ui.m_threadsSpin;
    m_opencvCompareChk = ui.m_opencvCompareChk;
    m_lrThresholdSpin = ui.m_lrThresholdSpin;
    m_medianFilterSpin = ui.m_medianFilterSpin;

    m_runBtn = ui.m_runBtn;
    m_cancelBtn = ui.m_cancelBtn;
    m_resetBtn = ui.m_resetBtn;

    if (ui.mainSplitter)
    {
        ui.mainSplitter->setStretchFactor(0, 2);
        ui.mainSplitter->setStretchFactor(1, 3);
    }

    m_matchTable->horizontalHeader()->setStretchLastSection(true);
    m_matchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_matchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_matchTable->setColumnWidth(0, 140);
    m_matchTable->setColumnWidth(1, 140);
    m_matchTable->setColumnWidth(2, 60);

    m_algorithmCombo->clear();
    m_algorithmCombo->addItem(tr("MGM (More Global Match)"), 2);
    m_algorithmCombo->addItem(tr("SGM (Semi Global Match)"), 1);
    m_algorithmCombo->addItem(tr("BM (Block Match)"), 0);
    m_algorithmCombo->addItem(tr("OpenCV SGBM"), 3);

    m_costFuncCombo->clear();
    m_costFuncCombo->addItem(tr("Census Transform"), 3);
    m_costFuncCombo->addItem(tr("NCC"), 2);
    m_costFuncCombo->addItem(tr("Absolute Difference"), 0);
    m_costFuncCombo->addItem(tr("Squared Difference"), 1);
    m_costFuncCombo->addItem(tr("Ternary Census"), 4);

    m_subpixelCombo->clear();
    m_subpixelCombo->addItem(tr("Parabola Fitting"), 1);
    m_subpixelCombo->addItem(tr("None"), 0);

    connect(m_imageList, &QListWidget::itemChanged,
            this, &DenseMatchDialog::onImageSelectionChanged);
    connect(m_selectAllBtn, &QPushButton::clicked, this, &DenseMatchDialog::onSelectAll);
    connect(m_deselectAllBtn, &QPushButton::clicked, this, &DenseMatchDialog::onDeselectAll);
    connect(ui.m_browseOutputBtn, &QPushButton::clicked, this, &DenseMatchDialog::onBrowseOutput);

    connect(m_runBtn, &QPushButton::clicked, this, &DenseMatchDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

    connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseMatchDialog::onAlgorithmChanged);
    connect(m_costFuncCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseMatchDialog::emitSettingsNow);
    connect(m_subpixelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseMatchDialog::emitSettingsNow);
    connect(m_outputEdit, &QLineEdit::textChanged, this, &DenseMatchDialog::emitSettingsNow);

    const auto emitSpinChanged = [this](auto *spin)
    {
        connect(spin, QOverload<int>::of(&QSpinBox::valueChanged),
                this, &DenseMatchDialog::emitSettingsNow);
    };
    emitSpinChanged(m_minDispSpin);
    emitSpinChanged(m_maxDispSpin);
    emitSpinChanged(m_kernelWSpin);
    emitSpinChanged(m_kernelHSpin);
    emitSpinChanged(m_p1Spin);
    emitSpinChanged(m_p2Spin);
    emitSpinChanged(m_directionsSpin);
    emitSpinChanged(m_pyramidSpin);
    emitSpinChanged(m_deviceSpin);
    emitSpinChanged(m_threadsSpin);
    emitSpinChanged(m_medianFilterSpin);

    connect(m_useCudaChk, &QCheckBox::toggled, this, &DenseMatchDialog::emitSettingsNow);
    connect(m_opencvCompareChk, &QCheckBox::toggled, this, &DenseMatchDialog::emitSettingsNow);
    connect(m_lrThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &DenseMatchDialog::emitSettingsNow);

    onAlgorithmChanged(m_algorithmCombo->currentIndex());
}
