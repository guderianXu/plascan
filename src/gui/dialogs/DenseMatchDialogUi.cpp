// =============================================================================
// 文件: DenseMatchDialogUi.cpp
// 功能: DenseMatchDialog::setupUi() 实现 — 左右分栏 UI 构建
// =============================================================================
#include "DenseMatchDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QSplitter>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <QListWidget>
#include <QTableWidget>
#include <QHeaderView>
#include <QLabel>

void DenseMatchDialog::setupUi()
{
    auto *mainLayout = new QVBoxLayout(this);

    auto *splitter = new QSplitter(Qt::Horizontal, this);

    // ==================== 左栏：项目数据 ====================
    auto *leftWidget = new QWidget(this);
    auto *leftLayout = new QVBoxLayout(leftWidget);
    leftLayout->setContentsMargins(0, 0, 4, 0);

    auto *imageGroup = new QGroupBox(tr("选择影像"), leftWidget);
    auto *imageLayout = new QVBoxLayout(imageGroup);
    m_imageList = new QListWidget(leftWidget);
    m_imageList->setMinimumHeight(100);
    connect(m_imageList, &QListWidget::itemChanged,
            this, &DenseMatchDialog::onImageSelectionChanged);
    imageLayout->addWidget(m_imageList);

    auto *selBtnLayout = new QHBoxLayout();
    m_selectAllBtn   = new QPushButton(tr("全选"), leftWidget);
    m_deselectAllBtn = new QPushButton(tr("取消全选"), leftWidget);
    connect(m_selectAllBtn, &QPushButton::clicked, this, &DenseMatchDialog::onSelectAll);
    connect(m_deselectAllBtn, &QPushButton::clicked, this, &DenseMatchDialog::onDeselectAll);
    selBtnLayout->addWidget(m_selectAllBtn);
    selBtnLayout->addWidget(m_deselectAllBtn);
    selBtnLayout->addStretch();
    imageLayout->addLayout(selBtnLayout);
    leftLayout->addWidget(imageGroup);

    auto *matchGroup = new QGroupBox(
        tr("稀疏匹配对（将对这些影像对执行密集匹配）"), leftWidget);
    matchGroup->setCheckable(true);
    matchGroup->setChecked(true);
    auto *matchLayout = new QVBoxLayout(matchGroup);

    m_matchTable = new QTableWidget(0, 3, matchGroup);
    m_matchTable->setHorizontalHeaderLabels(
        {tr("左影像"), tr("右影像"), tr("匹配点")});
    m_matchTable->horizontalHeader()->setStretchLastSection(true);
    m_matchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_matchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_matchTable->setMinimumHeight(100);
    m_matchTable->setColumnWidth(0, 140);
    m_matchTable->setColumnWidth(1, 140);
    m_matchTable->setColumnWidth(2, 60);
    matchLayout->addWidget(m_matchTable);

    m_matchCountLabel = new QLabel(matchGroup);
    matchLayout->addWidget(m_matchCountLabel);
    leftLayout->addWidget(matchGroup);

    auto *outputGroup = new QGroupBox(tr("输出"), leftWidget);
    auto *outputLayout = new QHBoxLayout(outputGroup);
    outputLayout->addWidget(new QLabel(tr("输出目录:"), outputGroup));
    m_outputEdit = new QLineEdit(outputGroup);
    auto *browseBtn = new QPushButton(tr("浏览..."), outputGroup);
    browseBtn->setFixedWidth(64);
    connect(browseBtn, &QPushButton::clicked, this, &DenseMatchDialog::onBrowseOutput);
    outputLayout->addWidget(m_outputEdit);
    outputLayout->addWidget(browseBtn);
    leftLayout->addWidget(outputGroup);

    leftLayout->addStretch();
    splitter->addWidget(leftWidget);

    // ==================== 右栏：算法参数 ====================
    auto *rightWidget = new QWidget(this);
    auto *rightLayout = new QVBoxLayout(rightWidget);
    rightLayout->setContentsMargins(4, 0, 0, 0);

    auto *algoGroup = new QGroupBox(tr("算法参数"), rightWidget);
    auto *algoForm  = new QFormLayout(algoGroup);

    m_algorithmCombo = new QComboBox(rightWidget);
    m_algorithmCombo->addItem(tr("MGM (More Global Match)"), 2);
    m_algorithmCombo->addItem(tr("SGM (Semi Global Match)"), 1);
    m_algorithmCombo->addItem(tr("BM (Block Match)"), 0);
    m_algorithmCombo->addItem(tr("OpenCV SGBM"), 3);
    connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseMatchDialog::onAlgorithmChanged);
    algoForm->addRow(tr("匹配算法:"), m_algorithmCombo);

    m_costFuncCombo = new QComboBox(rightWidget);
    m_costFuncCombo->addItem(tr("Census Transform"), 3);
    m_costFuncCombo->addItem(tr("NCC"), 2);
    m_costFuncCombo->addItem(tr("Absolute Difference"), 0);
    m_costFuncCombo->addItem(tr("Squared Difference"), 1);
    m_costFuncCombo->addItem(tr("Ternary Census"), 4);
    algoForm->addRow(tr("代价函数:"), m_costFuncCombo);

    m_subpixelCombo = new QComboBox(rightWidget);
    m_subpixelCombo->addItem(tr("Parabola Fitting"), 1);
    m_subpixelCombo->addItem(tr("None"), 0);
    algoForm->addRow(tr("子像素精化:"), m_subpixelCombo);

    auto *dispRow = new QHBoxLayout();
    m_minDispSpin = new QSpinBox(rightWidget);
    m_minDispSpin->setRange(0, 1024);
    m_minDispSpin->setValue(0);
    m_minDispSpin->setPrefix(tr("最小:"));
    dispRow->addWidget(m_minDispSpin);
    m_maxDispSpin = new QSpinBox(rightWidget);
    m_maxDispSpin->setRange(16, 2048);
    m_maxDispSpin->setValue(256);
    m_maxDispSpin->setPrefix(tr("最大:"));
    dispRow->addWidget(m_maxDispSpin);
    algoForm->addRow(tr("视差范围:"), dispRow);

    auto *kernelRow = new QHBoxLayout();
    m_kernelWSpin = new QSpinBox(rightWidget);
    m_kernelWSpin->setRange(3, 31);
    m_kernelWSpin->setSingleStep(2);
    m_kernelWSpin->setValue(15);
    m_kernelWSpin->setPrefix(tr("W:"));
    kernelRow->addWidget(m_kernelWSpin);
    m_kernelHSpin = new QSpinBox(rightWidget);
    m_kernelHSpin->setRange(3, 31);
    m_kernelHSpin->setSingleStep(2);
    m_kernelHSpin->setValue(15);
    m_kernelHSpin->setPrefix(tr("H:"));
    kernelRow->addWidget(m_kernelHSpin);
    algoForm->addRow(tr("相关核:"), kernelRow);

    rightLayout->addWidget(algoGroup);

    auto *sgmGroup = new QGroupBox(tr("SGM/MGM 参数"), rightWidget);
    sgmGroup->setCheckable(true);
    sgmGroup->setChecked(true);
    auto *sgmForm = new QFormLayout(sgmGroup);

    auto *sgmRow1 = new QHBoxLayout();
    m_p1Spin = new QSpinBox(rightWidget);
    m_p1Spin->setRange(1, 255);
    m_p1Spin->setValue(8);
    m_p1Spin->setPrefix(tr("P1:"));
    sgmRow1->addWidget(m_p1Spin);
    m_p2Spin = new QSpinBox(rightWidget);
    m_p2Spin->setRange(1, 1024);
    m_p2Spin->setValue(32);
    m_p2Spin->setPrefix(tr("P2:"));
    sgmRow1->addWidget(m_p2Spin);
    sgmForm->addRow(tr("惩罚:"), sgmRow1);

    auto *sgmRow2 = new QHBoxLayout();
    m_directionsSpin = new QSpinBox(rightWidget);
    m_directionsSpin->setRange(4, 8);
    m_directionsSpin->setSingleStep(4);
    m_directionsSpin->setValue(8);
    m_directionsSpin->setPrefix(tr("方向:"));
    sgmRow2->addWidget(m_directionsSpin);
    m_pyramidSpin = new QSpinBox(rightWidget);
    m_pyramidSpin->setRange(0, 5);
    m_pyramidSpin->setValue(2);
    m_pyramidSpin->setPrefix(tr("金字塔:"));
    sgmRow2->addWidget(m_pyramidSpin);
    sgmForm->addRow(tr("优化:"), sgmRow2);

    rightLayout->addWidget(sgmGroup);

    auto *sysGroup = new QGroupBox(tr("系统参数"), rightWidget);
    sysGroup->setCheckable(true);
    sysGroup->setChecked(false);
    auto *sysForm  = new QFormLayout(sysGroup);

    m_useCudaChk = new QCheckBox(tr("使用 CUDA"), sysGroup);
    m_useCudaChk->setChecked(true);
    sysForm->addRow(m_useCudaChk);

    auto *sysRow = new QHBoxLayout();
    m_deviceSpin = new QSpinBox(sysGroup);
    m_deviceSpin->setRange(0, 7);
    m_deviceSpin->setValue(0);
    m_deviceSpin->setPrefix(tr("设备:"));
    sysRow->addWidget(m_deviceSpin);
    m_threadsSpin = new QSpinBox(sysGroup);
    m_threadsSpin->setRange(1, 64);
    m_threadsSpin->setValue(4);
    m_threadsSpin->setPrefix(tr("线程:"));
    sysRow->addWidget(m_threadsSpin);
    sysForm->addRow(sysRow);

    m_opencvCompareChk = new QCheckBox(tr("同时运行 OpenCV SGBM 对比"), sysGroup);
    sysForm->addRow(m_opencvCompareChk);

    rightLayout->addWidget(sysGroup);

    auto *postGroup = new QGroupBox(tr("后处理"), rightWidget);
    postGroup->setCheckable(true);
    postGroup->setChecked(false);
    auto *postForm  = new QFormLayout(postGroup);

    auto *postRow = new QHBoxLayout();
    m_lrThresholdSpin = new QDoubleSpinBox(postGroup);
    m_lrThresholdSpin->setRange(0.0, 10.0);
    m_lrThresholdSpin->setValue(1.0);
    m_lrThresholdSpin->setSingleStep(0.5);
    m_lrThresholdSpin->setPrefix(tr("L-R阈值:"));
    postRow->addWidget(m_lrThresholdSpin);
    m_medianFilterSpin = new QSpinBox(postGroup);
    m_medianFilterSpin->setRange(0, 15);
    m_medianFilterSpin->setValue(3);
    m_medianFilterSpin->setPrefix(tr("中值滤波:"));
    postRow->addWidget(m_medianFilterSpin);
    postForm->addRow(postRow);

    rightLayout->addWidget(postGroup);
    rightLayout->addStretch();
    splitter->addWidget(rightWidget);

    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);

    mainLayout->addWidget(splitter, 1);

    auto *btnLayout = new QHBoxLayout();
    m_runBtn    = new QPushButton(tr("运行"), this);
    m_cancelBtn = new QPushButton(tr("取消"), this);
    m_resetBtn  = new QPushButton(tr("恢复默认"), this);
    btnLayout->addStretch();
    btnLayout->addWidget(m_runBtn);
    btnLayout->addWidget(m_cancelBtn);
    btnLayout->addWidget(m_resetBtn);
    mainLayout->addLayout(btnLayout);

    connect(m_runBtn, &QPushButton::clicked, this, &DenseMatchDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

    connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this](int){ emitSettingsNow(); });
    connect(m_costFuncCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this](int){ emitSettingsNow(); });
    connect(m_minDispSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            [this](int){ emitSettingsNow(); });
    connect(m_maxDispSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            [this](int){ emitSettingsNow(); });
}
