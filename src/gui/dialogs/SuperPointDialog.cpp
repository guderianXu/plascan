// SuperPointDialog.cpp
// SuperPoint 特征点提取配置对话框实现

#include "SuperPointDialog.h"

#include <QListWidget>
#include <QPushButton>
#include <QJsonObject>
#include <QLineEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QFileDialog>
#include <QFile>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QLabel>
#include <QSplitter>
#include <QToolButton>
#include <QMessageBox>

SuperPointDialog::SuperPointDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    setupConnections();
}

SuperPointDialog::~SuperPointDialog() = default;

void SuperPointDialog::setupUi() 
{
    setWindowTitle(tr("特征点提取"));
    resize(900, 600);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // 上部：文件列表与输出设置 + 参数面板
    QSplitter* topSplit = new QSplitter(this);

    // ==================== 左侧文件面板 ====================
    QWidget* left = new QWidget(this);
    QVBoxLayout* leftLayout = new QVBoxLayout(left);
    
    m_fileList = new QListWidget(left);
    m_fileList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    leftLayout->addWidget(new QLabel(tr("输入影像:"), left));
    leftLayout->addWidget(m_fileList);

    // 文件操作按钮
    QHBoxLayout* btnRow = new QHBoxLayout();
    m_addFilesBtn = new QPushButton(tr("添加文件..."), left);
    m_addFolderBtn = new QPushButton(tr("添加文件夹..."), left);
    m_removeBtn = new QPushButton(tr("移除选中"), left);
    m_clearBtn = new QPushButton(tr("清空"), left);
    btnRow->addWidget(m_addFilesBtn);
    btnRow->addWidget(m_addFolderBtn);
    btnRow->addWidget(m_removeBtn);
    btnRow->addWidget(m_clearBtn);
    leftLayout->addLayout(btnRow);

    // 输出设置
    QHBoxLayout* outRow = new QHBoxLayout();
    m_outputLine = new QLineEdit(left);
    m_outputLine->setPlaceholderText(tr("保存到项目 assets/ip"));
    m_outputLine->setToolTip(tr("留空时：默认保存到项目 assets/ip 目录"));
    m_browseOutBtn = new QPushButton(tr("浏览..."), left);
    outRow->addWidget(new QLabel(tr("输出目录："), left));
    outRow->addWidget(m_outputLine);
    outRow->addWidget(m_browseOutBtn);
    leftLayout->addLayout(outRow);

    topSplit->addWidget(left);

    // ==================== 右侧：参数面板 ====================
    QWidget* right = new QWidget(this);
    QVBoxLayout* rightLayout = new QVBoxLayout(right);

    // --------------- 基础参数 ---------------
    QLabel* basicLabel = new QLabel(tr("<b>基础参数</b>"), right);
    rightLayout->addWidget(basicLabel);
    
    QFormLayout* basicForm = new QFormLayout();
    rightLayout->addLayout(basicForm);

    // 特征提取算法
    m_algorithmCombo = new QComboBox(right);
    m_algorithmCombo->addItem("SuperPoint", "superpoint");
    m_algorithmCombo->addItem("DISK", "disk");
    m_algorithmCombo->addItem("ALIKED", "aliked");
    m_algorithmCombo->addItem("ORB", "orb");
    m_algorithmCombo->addItem("SIFT", "sift");
    m_algorithmCombo->setCurrentIndex(0);
    m_algorithmCombo->setToolTip(tr("选择特征提取算法\n"
                                    "SuperPoint: 256维深度学习特征\n"
                                    "DISK: 128维旋转不变特征\n"
                                    "ALIKED: 128维轻量级特征\n"
                                    "输出文件格式始终保持为 .sp"));
    basicForm->addRow(tr("特征提取算法:"), m_algorithmCombo);

    // NMS 半径
    m_nmsRadiusSpin = new QSpinBox(right);
    m_nmsRadiusSpin->setRange(1, 16);
    m_nmsRadiusSpin->setValue(3);
    m_nmsRadiusSpin->setToolTip(tr("非极大值抑制半径（像素）\n"
                                   "值越大抑制越强，推荐范围 1-8\n"
                                   "图像尺度较大时可增大"));
    basicForm->addRow(tr("NMS 半径(像素):"), m_nmsRadiusSpin);

    // 检测阈值
    m_detectionThresholdSpin = new QDoubleSpinBox(right);
    m_detectionThresholdSpin->setRange(0.0001, 1.0);
    m_detectionThresholdSpin->setDecimals(4);
    m_detectionThresholdSpin->setSingleStep(0.001);
    m_detectionThresholdSpin->setValue(0.003);
    m_detectionThresholdSpin->setToolTip(tr("检测阈值（score map 阈值）\n"
                                            "阈值越低检测到的点越多\n"
                                            "推荐范围 0.001-0.02"));
    basicForm->addRow(tr("检测阈值:"), m_detectionThresholdSpin);

    // 最大关键点数量
    m_maxKeypointsSpin = new QSpinBox(right);
    m_maxKeypointsSpin->setRange(-1, 100000);
    m_maxKeypointsSpin->setValue(-1);
    m_maxKeypointsSpin->setSpecialValueText(tr("不限制"));
    m_maxKeypointsSpin->setToolTip(tr("最大关键点数量\n"
                                      "按 score 截断，-1 表示不限制\n"
                                      "实时 GUI 中建议设置 500-2000"));
    basicForm->addRow(tr("最大关键点数:"), m_maxKeypointsSpin);

    // 边界移除
    m_removeBordersSpin = new QSpinBox(right);
    m_removeBordersSpin->setRange(0, 50);
    m_removeBordersSpin->setValue(4);
    m_removeBordersSpin->setToolTip(tr("在图像边界内移除指定像素数\n"
                                       "防止靠近边界的点被误检"));
    basicForm->addRow(tr("边界移除(像素):"), m_removeBordersSpin);

    // 灰度过滤范围
    QHBoxLayout* grayLayout = new QHBoxLayout();
    m_grayscaleMinSpin = new QDoubleSpinBox(right);
    m_grayscaleMinSpin->setRange(0.0, 1.0);
    m_grayscaleMinSpin->setDecimals(2);
    m_grayscaleMinSpin->setSingleStep(0.05);
    m_grayscaleMinSpin->setValue(0.0);
    m_grayscaleMaxSpin = new QDoubleSpinBox(right);
    m_grayscaleMaxSpin->setRange(0.0, 1.0);
    m_grayscaleMaxSpin->setDecimals(2);
    m_grayscaleMaxSpin->setSingleStep(0.05);
    m_grayscaleMaxSpin->setValue(1.0);
    grayLayout->addWidget(new QLabel(tr("最小:"), right));
    grayLayout->addWidget(m_grayscaleMinSpin);
    grayLayout->addWidget(new QLabel(tr("最大:"), right));
    grayLayout->addWidget(m_grayscaleMaxSpin);
    QString grayTip = tr("灰度过滤阈值区间（归一化 0.0-1.0）\n"
                        "例如 [0.05,0.95] 可过滤极暗或极亮区域");
    m_grayscaleMinSpin->setToolTip(grayTip);
    m_grayscaleMaxSpin->setToolTip(grayTip);
    basicForm->addRow(tr("灰度过滤范围:"), grayLayout);

    // --------------- 高级参数（折叠） ---------------
    auto *advancedToggle = new QToolButton(right);
    advancedToggle->setText(tr("高级参数"));
    advancedToggle->setCheckable(true);
    advancedToggle->setChecked(false);
    advancedToggle->setArrowType(Qt::RightArrow);
    advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    rightLayout->addWidget(advancedToggle);

    m_advancedGroup = new QGroupBox(right);
    m_advancedGroup->setFlat(true);
    m_advancedGroup->setTitle(QString());
    m_advancedGroup->setVisible(false);
    QFormLayout* advancedForm = new QFormLayout(m_advancedGroup);
    rightLayout->addWidget(m_advancedGroup);

    connect(advancedToggle, &QToolButton::toggled, this, [this, advancedToggle](bool checked)
    {
        m_advancedGroup->setVisible(checked);
        advancedToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    // 归一化输入
    m_normalizeInputChk = new QCheckBox(tr("对输入图像归一化"), right);
    m_normalizeInputChk->setChecked(true);
    m_normalizeInputChk->setToolTip(tr("按位深度归一化到浮点数范围（0-1）\n"
                                       "建议启用并与训练时设置一致"));
    advancedForm->addRow(m_normalizeInputChk);

    // 描述子维度
    m_descriptorDimSpin = new QSpinBox(right);
    m_descriptorDimSpin->setRange(64, 512);
    m_descriptorDimSpin->setValue(256);
    m_descriptorDimSpin->setToolTip(tr("描述子维度（通常为 256）\n"
                                       "用于分配输出张量大小与匹配逻辑"));
    advancedForm->addRow(tr("描述子维度:"), m_descriptorDimSpin);

    // 网格大小
    m_gridSizeSpin = new QSpinBox(right);
    m_gridSizeSpin->setRange(1, 32);
    m_gridSizeSpin->setValue(8);
    m_gridSizeSpin->setToolTip(tr("网络输出到输入的下采样步长（通常为 8）\n"
                                  "用于从 dense descriptors 中采样局部描述子"));
    advancedForm->addRow(tr("网格大小:"), m_gridSizeSpin);

    // 批处理大小
    m_batchSizeSpin = new QSpinBox(right);
    m_batchSizeSpin->setRange(-1, 128);
    m_batchSizeSpin->setValue(8);
    m_batchSizeSpin->setSpecialValueText(tr("禁用批处理"));
    m_batchSizeSpin->setToolTip(tr("批处理大小（用于 detectBatch）\n"
                                   "同时处理多张图像以提高 GPU 利用率\n"
                                   "-1 或 0 表示禁用批处理，逐张调用"));
    advancedForm->addRow(tr("批处理大小:"), m_batchSizeSpin);

    // 邻域判断参数（边界点过滤）
    m_neighborhoodRadiusSpin = new QSpinBox(right);
    m_neighborhoodRadiusSpin->setRange(1, 20);
    m_neighborhoodRadiusSpin->setValue(3);
    m_neighborhoodRadiusSpin->setToolTip(tr("检查特征点周围邻域的半径（像素）\n检测邻域内是否有纯黑像素，如有则丢弃该点"));
    advancedForm->addRow(tr("邻域判断范围(px):"), m_neighborhoodRadiusSpin);

    m_neighborhoodThresholdSpin = new QDoubleSpinBox(right);
    m_neighborhoodThresholdSpin->setRange(0.0, 1.0);
    m_neighborhoodThresholdSpin->setDecimals(3);
    m_neighborhoodThresholdSpin->setSingleStep(0.01);
    m_neighborhoodThresholdSpin->setValue(0.05);
    m_neighborhoodThresholdSpin->setToolTip(tr("邻域灰度阈值 [0,1]\n邻域内有像素低于此值，则认为靠近边界并丢弃该点"));
    advancedForm->addRow(tr("邻域灰度阈值:"), m_neighborhoodThresholdSpin);

    // --------------- 系统参数（折叠） ---------------
    auto *systemToggle = new QToolButton(right);
    systemToggle->setText(tr("系统参数"));
    systemToggle->setCheckable(true);
    systemToggle->setChecked(false);
    systemToggle->setArrowType(Qt::RightArrow);
    systemToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    rightLayout->addWidget(systemToggle);

    m_systemGroup = new QGroupBox(right);
    m_systemGroup->setFlat(true);
    m_systemGroup->setTitle(QString());
    m_systemGroup->setVisible(false);
    QFormLayout* systemForm = new QFormLayout(m_systemGroup);
    rightLayout->addWidget(m_systemGroup);

    connect(systemToggle, &QToolButton::toggled, this, [this, systemToggle](bool checked)
    {
        m_systemGroup->setVisible(checked);
        systemToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    // 运行设备
    m_deviceCombo = new QComboBox(right);
    m_deviceCombo->addItems(QStringList() << "CPU" << "CUDA");
    m_deviceCombo->setCurrentText("CUDA");  // 默认使用CUDA
    m_deviceCombo->setToolTip(tr("运行设备（CPU 或 CUDA）\n"
                                 "如果选择 CUDA 且系统不支持，将根据回退设置处理"));
    systemForm->addRow(tr("运行设备:"), m_deviceCombo);

    // 允许设备回退
    m_allowFallbackChk = new QCheckBox(tr("允许设备回退到 CPU"), right);
    m_allowFallbackChk->setChecked(true);
    m_allowFallbackChk->setToolTip(tr("在 CUDA 不可用时自动回退到 CPU\n"
                                      "GUI 应用通常建议启用以提高容错性"));
    systemForm->addRow(m_allowFallbackChk);

    // --------------- 调试参数（折叠） ---------------
    auto *debugToggle = new QToolButton(right);
    debugToggle->setText(tr("调试参数"));
    debugToggle->setCheckable(true);
    debugToggle->setChecked(false);
    debugToggle->setArrowType(Qt::RightArrow);
    debugToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    rightLayout->addWidget(debugToggle);

    m_debugGroup = new QGroupBox(right);
    m_debugGroup->setFlat(true);
    m_debugGroup->setTitle(QString());
    m_debugGroup->setVisible(false);
    QFormLayout* debugForm = new QFormLayout(m_debugGroup);
    rightLayout->addWidget(m_debugGroup);

    connect(debugToggle, &QToolButton::toggled, this, [this, debugToggle](bool checked)
    {
        m_debugGroup->setVisible(checked);
        debugToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    // 保存 CSV
    m_saveCsvChk = new QCheckBox(tr("每次推理后保存关键点到 CSV"), right);
    m_saveCsvChk->setChecked(false);
    m_saveCsvChk->setToolTip(tr("调试用：将关键点坐标和分数写入 CSV 文件"));
    debugForm->addRow(m_saveCsvChk);

    // 保存叠加图像
    m_saveOverlayChk = new QCheckBox(tr("每次推理后保存叠加图像"), right);
    m_saveOverlayChk->setChecked(false);
    m_saveOverlayChk->setToolTip(tr("调试用：将关键点叠加到原图并保存为图像文件"));
    debugForm->addRow(m_saveOverlayChk);

    rightLayout->addStretch();
    topSplit->addWidget(right);
    mainLayout->addWidget(topSplit);

    // ==================== 底部按钮 ====================
    QHBoxLayout* bottomLayout = new QHBoxLayout();
    m_resetBtn = new QPushButton(tr("恢复默认参数"), this);
    m_resetBtn->setToolTip(tr("恢复 SuperPoint 的默认参数"));
    bottomLayout->addWidget(m_resetBtn);
    bottomLayout->addStretch();
    m_runBtn = new QPushButton(tr("运行"), this);
    m_cancelBtn = new QPushButton(tr("取消"), this);
    bottomLayout->addWidget(m_runBtn);
    bottomLayout->addWidget(m_cancelBtn);
    mainLayout->addLayout(bottomLayout);

    // 设置默认按钮
    m_runBtn->setDefault(true);
}

void SuperPointDialog::setupConnections() 
{
    // 文件操作
    connect(m_addFilesBtn, &QPushButton::clicked, this, &SuperPointDialog::onAddFiles);
    connect(m_addFolderBtn, &QPushButton::clicked, this, &SuperPointDialog::onAddFolder);
    connect(m_removeBtn, &QPushButton::clicked, this, &SuperPointDialog::onRemoveSelected);
    connect(m_clearBtn, &QPushButton::clicked, this, &SuperPointDialog::onClearFiles);
    connect(m_browseOutBtn, &QPushButton::clicked, this, &SuperPointDialog::onBrowseOutput);

    // 底部按钮
    connect(m_runBtn, &QPushButton::clicked, this, &SuperPointDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &SuperPointDialog::onCancel);
    connect(m_resetBtn, &QPushButton::clicked, this, &SuperPointDialog::onResetDefaults);

    // 参数变更信号（实时同步到项目配置）
    // 基础参数
        connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SuperPointDialog::onAlgorithmChanged);
        connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_nmsRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_detectionThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_maxKeypointsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_removeBordersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_grayscaleMinSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_grayscaleMaxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &SuperPointDialog::emitSettingsNow);

    // 高级参数
    connect(m_normalizeInputChk, &QCheckBox::toggled, this, &SuperPointDialog::emitSettingsNow);
    connect(m_descriptorDimSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_gridSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_batchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_neighborhoodRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &SuperPointDialog::emitSettingsNow);
    connect(m_neighborhoodThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SuperPointDialog::emitSettingsNow);

    // 系统参数
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SuperPointDialog::emitSettingsNow);
    connect(m_allowFallbackChk, &QCheckBox::toggled, this, &SuperPointDialog::emitSettingsNow);

    // 调试参数
    connect(m_saveCsvChk, &QCheckBox::toggled, this, &SuperPointDialog::emitSettingsNow);
    connect(m_saveOverlayChk, &QCheckBox::toggled, this, &SuperPointDialog::emitSettingsNow);

    // 初始状态
    onAlgorithmChanged(m_algorithmCombo->currentIndex());
}

void SuperPointDialog::onAddFiles() 
{
    QStringList files = QFileDialog::getOpenFileNames(
        this, 
        tr("选择输入影像"),
        QString(),
        tr("图像文件 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;所有文件 (*)")
    );
    
    for (const QString& file : files) {
        if (!file.isEmpty()) {
            QListWidgetItem *item = new QListWidgetItem(file, m_fileList);
            item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
            item->setCheckState(Qt::Checked);
        }
    }
}

void SuperPointDialog::onAddFolder() 
{
    QString dir = QFileDialog::getExistingDirectory(
        this,
        tr("选择输入文件夹"),
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (!dir.isEmpty()) {
        // 扫描文件夹中的图像文件
        QDir folder(dir);
        QStringList filters;
        filters << "*.jpg" << "*.jpeg" << "*.png" << "*.tif" << "*.tiff" << "*.bmp";
        QStringList files = folder.entryList(filters, QDir::Files);
        
        for (const QString &file : files) {
            QString fullPath = folder.filePath(file);
            // 创建带checkbox的item
            QListWidgetItem *item = new QListWidgetItem(fullPath, m_fileList);
            item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
            item->setCheckState(Qt::Checked); // 默认选中
        }
    }
}

void SuperPointDialog::onRemoveSelected() 
{
    QList<QListWidgetItem*> selected = m_fileList->selectedItems();
    for (QListWidgetItem* item : selected) {
        delete m_fileList->takeItem(m_fileList->row(item));
    }
}

void SuperPointDialog::onClearFiles() 
{
    m_fileList->clear();
}

void SuperPointDialog::onBrowseOutput() 
{
    QString dir = QFileDialog::getExistingDirectory(
        this,
        tr("选择输出目录"),
        m_outputLine->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (!dir.isEmpty()) {
        m_outputLine->setText(dir);
    }
}

void SuperPointDialog::onRun() 
{
    // 收集选中的输入文件列表（只处理checkbox选中的）
    QStringList inputs;
    for (int i = 0; i < m_fileList->count(); ++i) {
        QListWidgetItem *item = m_fileList->item(i);
        if (item->checkState() == Qt::Checked) {
            inputs.append(item->text());
        }
    }

    if (inputs.isEmpty()) {
        QMessageBox::warning(this, tr("警告"), tr("请至少选中一个输入影像！"));
        return;
    }

    // 收集配置参数
    QJsonObject config = collectSettings();

    // 发送运行请求
    emit runRequested(config, inputs);
    
    // 接受对话框
    accept();
}

void SuperPointDialog::onCancel() 
{
    reject();
}

void SuperPointDialog::onResetDefaults()
{
    // 恢复所有参数到默认值
    m_algorithmCombo->setCurrentIndex(0);
    m_nmsRadiusSpin->setValue(3);
    m_detectionThresholdSpin->setValue(0.003);
    m_maxKeypointsSpin->setValue(-1);
    m_removeBordersSpin->setValue(4);
    m_grayscaleMinSpin->setValue(0.0);
    m_grayscaleMaxSpin->setValue(1.0);

    m_normalizeInputChk->setChecked(true);
    m_descriptorDimSpin->setValue(256);
    m_gridSizeSpin->setValue(8);
    m_batchSizeSpin->setValue(8);
    m_neighborhoodRadiusSpin->setValue(3);
    m_neighborhoodThresholdSpin->setValue(0.05);

    m_deviceCombo->setCurrentText("CUDA");
    m_allowFallbackChk->setChecked(true);

    m_saveCsvChk->setChecked(false);
    m_saveOverlayChk->setChecked(false);
}

void SuperPointDialog::onAlgorithmChanged(int)
{
    const QString algo = m_algorithmCombo->currentData().toString();
    const bool isDL = (algo == "superpoint" || algo == "disk" || algo == "aliked");
    const bool isDeepLearning256 = (algo == "superpoint");

    // DISK/ALIKED 是 128 维，SuperPoint 是 256 维
    if (algo == "disk" || algo == "aliked") {
        m_descriptorDimSpin->setValue(128);
    } else if (algo == "superpoint") {
        m_descriptorDimSpin->setValue(256);
    }

    // 传统算法不支持 NMS、描述子维度等深度学习参数
    m_nmsRadiusSpin->setEnabled(isDL);
    m_detectionThresholdSpin->setEnabled(isDL);
    m_descriptorDimSpin->setEnabled(isDL);
    m_gridSizeSpin->setEnabled(isDL);
    m_normalizeInputChk->setEnabled(isDL);
}

void SuperPointDialog::applySettings(const QJsonObject &settings) 
{
    if (settings.contains("feature_algorithm"))
    {
        const QString featureAlgorithm = settings["feature_algorithm"].toString("superpoint").trimmed().toLower();
        const int index = m_algorithmCombo->findData(featureAlgorithm);
        m_algorithmCombo->setCurrentIndex(index >= 0 ? index : 0);
    }

    // 基础参数
    if (settings.contains("nms_radius"))
        m_nmsRadiusSpin->setValue(settings["nms_radius"].toInt(3));
    if (settings.contains("detection_threshold"))
        m_detectionThresholdSpin->setValue(settings["detection_threshold"].toDouble(0.003));
    if (settings.contains("max_num_keypoints"))
        m_maxKeypointsSpin->setValue(settings["max_num_keypoints"].toInt(-1));
    if (settings.contains("remove_borders"))
        m_removeBordersSpin->setValue(settings["remove_borders"].toInt(4));
    if (settings.contains("grayscale_min"))
        m_grayscaleMinSpin->setValue(settings["grayscale_min"].toDouble(0.0));
    if (settings.contains("grayscale_max"))
        m_grayscaleMaxSpin->setValue(settings["grayscale_max"].toDouble(1.0));

    // 高级参数
    if (settings.contains("normalize_input"))
        m_normalizeInputChk->setChecked(settings["normalize_input"].toBool(true));
    if (settings.contains("descriptor_dim"))
        m_descriptorDimSpin->setValue(settings["descriptor_dim"].toInt(256));
    if (settings.contains("grid_size"))
        m_gridSizeSpin->setValue(settings["grid_size"].toInt(8));
    if (settings.contains("batch_size"))
        m_batchSizeSpin->setValue(settings["batch_size"].toInt(8));
    if (settings.contains("neighborhood_check_radius"))
        m_neighborhoodRadiusSpin->setValue(settings["neighborhood_check_radius"].toInt(3));
    if (settings.contains("neighborhood_threshold"))
        m_neighborhoodThresholdSpin->setValue(settings["neighborhood_threshold"].toDouble(0.05));

    // 系统参数
    if (settings.contains("device"))
        m_deviceCombo->setCurrentText(settings["device"].toString("CUDA"));  // 默认CUDA
    if (settings.contains("allow_device_fallback"))
        m_allowFallbackChk->setChecked(settings["allow_device_fallback"].toBool(true));

    // 调试参数
    if (settings.contains("save_keypoints_csv"))
        m_saveCsvChk->setChecked(settings["save_keypoints_csv"].toBool(false));
    if (settings.contains("save_overlay_image"))
        m_saveOverlayChk->setChecked(settings["save_overlay_image"].toBool(false));

    // 输出目录
    if (settings.contains("output_dir")) {
        QString outputDir = settings["output_dir"].toString();
        if (!outputDir.isEmpty()) {
            m_outputLine->setText(outputDir);
        }
    }
}

void SuperPointDialog::setProjectImages(const QStringList &paths) 
{
    m_fileList->clear();
    for (const QString &path : paths) 
    {
        // 创建带checkbox的item
        QListWidgetItem *item = new QListWidgetItem(path, m_fileList);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked); // 默认全选
    }
}

void SuperPointDialog::updatePreview() 
{
    // TODO: 可选实现，提供参数预览或影响说明
}

void SuperPointDialog::emitSettingsNow() 
{
    emit settingsChanged(collectSettings());
}

QJsonObject SuperPointDialog::collectSettings() const 
{
    QJsonObject settings;

    settings["feature_algorithm"] = m_algorithmCombo->currentData().toString();

    // 基础参数
    settings["nms_radius"] = m_nmsRadiusSpin->value();
    settings["detection_threshold"] = m_detectionThresholdSpin->value();
    settings["max_num_keypoints"] = m_maxKeypointsSpin->value();
    settings["remove_borders"] = m_removeBordersSpin->value();
    settings["grayscale_min"] = m_grayscaleMinSpin->value();
    settings["grayscale_max"] = m_grayscaleMaxSpin->value();

    // 高级参数
    settings["normalize_input"] = m_normalizeInputChk->isChecked();
    settings["descriptor_dim"] = m_descriptorDimSpin->value();
    settings["grid_size"] = m_gridSizeSpin->value();
    settings["batch_size"] = m_batchSizeSpin->value();
    settings["neighborhood_check_radius"] = m_neighborhoodRadiusSpin->value();
    settings["neighborhood_threshold"] = m_neighborhoodThresholdSpin->value();

    // 系统参数
    settings["device"] = m_deviceCombo->currentText();
    settings["allow_device_fallback"] = m_allowFallbackChk->isChecked();

    // 调试参数
    settings["save_keypoints_csv"] = m_saveCsvChk->isChecked();
    settings["save_overlay_image"] = m_saveOverlayChk->isChecked();

    // 输出目录
    settings["output_dir"] = m_outputLine->text();

    return settings;
}
