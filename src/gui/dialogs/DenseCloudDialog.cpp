// =============================================================================
// 文件: DenseCloudDialog.cpp
// 说明: 稠密点云生成对话框实现
// =============================================================================
#include "DenseCloudDialog.h"

#include "ProjectManager.h"

#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QTextEdit>
#include <QTimer>

DenseCloudDialog::DenseCloudDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , m_projectManager(projectManager)
{
    setWindowTitle(tr("生成稠密点云 (MVS)"));
    setMinimumWidth(540);
    setMinimumHeight(640);
    setupUi();
}

// =============================================================================
// setupUi
// =============================================================================

void DenseCloudDialog::setupUi()
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(8);

    // ---- 顶部：输入选择 ----
    auto *inputGroup = new QGroupBox(tr("输入配置"), this);
    auto *inputForm = new QFormLayout(inputGroup);

    m_imagePairCombo = new QComboBox(this);
    m_imagePairCombo->setToolTip(tr("选择用于立体匹配的影像对（默认全部影像对）"));
    inputForm->addRow(tr("影像对:"), m_imagePairCombo);

    m_atResultCombo = new QComboBox(this);
    m_atResultCombo->setToolTip(tr("选择已完成空三/相对定向的 AT 结果（提供相机位姿）"));
    inputForm->addRow(tr("AT 结果:"), m_atResultCombo);

    // 填充 AT 结果列表
    if (m_projectManager) 
    {
        const QJsonArray atResults = m_projectManager->getAvailableAtResults();
        if (atResults.isEmpty()) 
        {
            m_atResultCombo->addItem(tr("（请先运行空三）"), -2);
        } 
        else 
        {
            m_atResultCombo->addItem(tr("最新 AT 结果（推荐）"), -1);
            for (int i = 0; i < atResults.size(); ++i) 
            {
                const QJsonObject item = atResults[i].toObject();
                const int idx = item.value(QStringLiteral("index")).toInt(i);
                const int imgCnt = item.value(QStringLiteral("image_count")).toInt(0);
                const int ptsCnt = item.value(QStringLiteral("sparse_point_count")).toInt(0);
                const QString createdAt = item.value(QStringLiteral("created_at")).toString().left(10);
                const QString outDir   = QFileInfo(item.value(QStringLiteral("output_dir")).toString()).fileName();
                const QString label = QStringLiteral("[%1] %2  %3 张影像  %4 点  (%5)")
                    .arg(idx).arg(outDir).arg(imgCnt).arg(ptsCnt).arg(createdAt);
                m_atResultCombo->addItem(label, idx);
            }
            m_atResultCombo->setCurrentIndex(0);
        }
    }

    // 预设选择
    m_presetCombo = new QComboBox(this);
    m_presetCombo->addItem(tr("快速 (低精度，适合预览)"), QStringLiteral("fast"));
    m_presetCombo->addItem(tr("标准 (推荐)"),              QStringLiteral("standard"));
    m_presetCombo->addItem(tr("精细 (高精度，较慢)"),      QStringLiteral("quality"));
    m_presetCombo->setCurrentIndex(1);
    inputForm->addRow(tr("参数预设:"), m_presetCombo);
    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseCloudDialog::onPresetChanged);
        connect(m_atResultCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseCloudDialog::onAnyChanged);

    // 输出目录
    auto *outputRow = new QHBoxLayout;
    m_outputDirEdit = new QLineEdit(this);
    auto *browseBtn = new QPushButton(tr("浏览..."), this);
    browseBtn->setFixedWidth(64);
    outputRow->addWidget(m_outputDirEdit);
    outputRow->addWidget(browseBtn);
    connect(browseBtn, &QPushButton::clicked, this, &DenseCloudDialog::onBrowseOutput);
    inputForm->addRow(tr("输出目录:"), outputRow);

    mainLayout->addWidget(inputGroup);

    // ---- 参数选项卡 ----
    auto *tabs = new QTabWidget(this);

    // Tab 1: SGBM 深度图参数
    auto *sgbmWidget = new QWidget;
    auto *sgbmGroup = new QGroupBox;
    auto *sgbmOuter = new QVBoxLayout(sgbmWidget);
    buildSgbmGroup(sgbmGroup);
    sgbmOuter->addWidget(sgbmGroup);
    sgbmOuter->addStretch();
    tabs->addTab(sgbmWidget, tr("深度图 (SGBM)"));

    // Tab 2: 点云参数
    auto *cloudWidget = new QWidget;
    auto *cloudGroup = new QGroupBox;
    auto *cloudOuter = new QVBoxLayout(cloudWidget);
    buildCloudGroup(cloudGroup);
    cloudOuter->addWidget(cloudGroup);
    cloudOuter->addStretch();
    tabs->addTab(cloudWidget, tr("点云"));

    // Tab 3: 网格重建（可选）
    auto *meshWidget = new QWidget;
    auto *meshGroup = new QGroupBox;
    auto *meshOuter = new QVBoxLayout(meshWidget);
    buildMeshGroup(meshGroup);
    meshOuter->addWidget(meshGroup);
    meshOuter->addStretch();
    tabs->addTab(meshWidget, tr("网格重建（可选）"));

    mainLayout->addWidget(tabs);

    // ---- 进度条 ----
    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    m_progressBar->setTextVisible(true);
    mainLayout->addWidget(m_progressBar);

    // ---- 日志 ----
    m_logEdit = new QTextEdit(this);
    m_logEdit->setReadOnly(true);
    m_logEdit->setFixedHeight(120);
    m_logEdit->setFontFamily(QStringLiteral("Monospace"));
    m_logEdit->setFontPointSize(9);
    mainLayout->addWidget(m_logEdit);

    // ---- 按钮 ----
    auto *btnLayout = new QHBoxLayout;
    m_runButton    = new QPushButton(tr("开始生成"), this);
    m_cancelButton = new QPushButton(tr("取消"),     this);
    m_runButton->setDefault(true);
    btnLayout->addStretch();
    btnLayout->addWidget(m_runButton);
    btnLayout->addWidget(m_cancelButton);
    mainLayout->addLayout(btnLayout);

    connect(m_runButton,    &QPushButton::clicked, this, &DenseCloudDialog::onRun);
    connect(m_cancelButton, &QPushButton::clicked, this, &DenseCloudDialog::onCancel);

    // 初始日志
    appendLog(tr("[MVS] 就绪。请确认已完成空中三角测量后再运行稠密重建。"));
    appendLog(tr("[提示] 匹配点数量少 ≠ 点云质量差；MVS 通过像素级密集匹配生成稠密点云。"));
}

// =============================================================================
// buildSgbmGroup
// =============================================================================

void DenseCloudDialog::buildSgbmGroup(QGroupBox *gb)
{
    gb->setTitle(tr("SGBM 视差估计参数"));
    auto *form = new QFormLayout(gb);

    m_numDispSpin = new QSpinBox(this);
    m_numDispSpin->setRange(16, 512);
    m_numDispSpin->setSingleStep(16);
    m_numDispSpin->setValue(128);
    m_numDispSpin->setToolTip(tr("视差搜索范围（像素，必须是16的倍数）。"
                                  "值越大可检测更大深度差，但运算更慢。\n"
                                  "建议：128（标准）~256（大场景）"));
    form->addRow(tr("视差范围:"), m_numDispSpin);

    m_blockSizeSpin = new QSpinBox(this);
    m_blockSizeSpin->setRange(3, 21);
    m_blockSizeSpin->setSingleStep(2);
    m_blockSizeSpin->setValue(9);
    m_blockSizeSpin->setToolTip(tr("立体匹配块大小（奇数，3~21）。"
                                    "越大越稳健但细节损失；纹理丰富时用小值。\n"
                                    "建议：7~11"));
    form->addRow(tr("匹配块大小:"), m_blockSizeSpin);

    m_uniquenessSpin = new QSpinBox(this);
    m_uniquenessSpin->setRange(0, 50);
    m_uniquenessSpin->setValue(10);
    m_uniquenessSpin->setSuffix(tr(" %"));
    m_uniquenessSpin->setToolTip(tr("唯一性比率：最优视差得分比次优好多少才接受（0=关闭）。\n"
                                     "值越高误匹配越少但覆盖率下降。建议：5~15"));
    form->addRow(tr("唯一性比率:"), m_uniquenessSpin);

    m_speckleSizeSpin = new QSpinBox(this);
    m_speckleSizeSpin->setRange(0, 1000);
    m_speckleSizeSpin->setValue(100);
    m_speckleSizeSpin->setToolTip(tr("噪斑消除窗口（像素数）。小于此尺寸的孤立视差区域将被过滤。\n"
                                      "建议：50~200"));
    form->addRow(tr("噪斑消除窗口:"), m_speckleSizeSpin);

    m_fullDpCheck = new QCheckBox(tr("使用 SGBM_3WAY 模式（更慢更精确）"), this);
    m_fullDpCheck->setChecked(true);
    form->addRow(QString(), m_fullDpCheck);

    m_wlsFilterCheck = new QCheckBox(tr("启用 WLS 视差后过滤（平滑边缘，需要 opencv_ximgproc）"), this);
    m_wlsFilterCheck->setChecked(true);
    form->addRow(QString(), m_wlsFilterCheck);

    // 深度范围
    auto *depthRow = new QHBoxLayout;
    m_minDepthSpin = new QDoubleSpinBox(this);
    m_minDepthSpin->setRange(0.001, 9999);
    m_minDepthSpin->setValue(0.01);
    m_minDepthSpin->setDecimals(3);

    m_maxDepthSpin = new QDoubleSpinBox(this);
    m_maxDepthSpin->setRange(0.1, 1e7);
    m_maxDepthSpin->setValue(1e5);
    m_maxDepthSpin->setDecimals(1);
    m_maxDepthSpin->setSuffix(tr(" (单位:与坐标系一致)"));

    depthRow->addWidget(new QLabel(tr("最小:")));
    depthRow->addWidget(m_minDepthSpin);
    depthRow->addWidget(new QLabel(tr("  最大:")));
    depthRow->addWidget(m_maxDepthSpin);
    form->addRow(tr("有效深度范围:"), depthRow);
}

// =============================================================================
// buildCloudGroup
// =============================================================================

void DenseCloudDialog::buildCloudGroup(QGroupBox *gb)
{
    gb->setTitle(tr("稠密点云参数"));
    auto *form = new QFormLayout(gb);

    m_minConfSpin = new QDoubleSpinBox(this);
    m_minConfSpin->setRange(0.0, 1.0);
    m_minConfSpin->setSingleStep(0.05);
    m_minConfSpin->setValue(0.1);
    m_minConfSpin->setDecimals(2);
    m_minConfSpin->setToolTip(tr("WLS 滤波后的最低置信度阈值（0~1）。"
                                  "越高误差越少但点云越稀疏。建议：0.1~0.3"));
    form->addRow(tr("最低置信度:"), m_minConfSpin);

    m_multiViewCheck = new QCheckBox(tr("多视图一致性融合（减少飞点，需要 ≥2 组深度图）"), this);
    m_multiViewCheck->setChecked(true);
    form->addRow(QString(), m_multiViewCheck);

    m_colorsCheck = new QCheckBox(tr("从影像采样颜色（RGB 点云）"), this);
    m_colorsCheck->setChecked(true);
    form->addRow(QString(), m_colorsCheck);

    m_normalsCheck = new QCheckBox(tr("估计点云法向量（Poisson 重建必须开启）"), this);
    m_normalsCheck->setChecked(true);
    form->addRow(QString(), m_normalsCheck);

    m_normalKnnSpin = new QSpinBox(this);
    m_normalKnnSpin->setRange(8, 64);
    m_normalKnnSpin->setValue(20);
    form->addRow(tr("法向量 KNN 邻域:"), m_normalKnnSpin);

    // 连接法向量复选框与 KNN 纺控件联动
    connect(m_normalsCheck, &QCheckBox::toggled, m_normalKnnSpin, &QSpinBox::setEnabled);
}

// =============================================================================
// buildMeshGroup
// =============================================================================

void DenseCloudDialog::buildMeshGroup(QGroupBox *gb)
{
    gb->setTitle(tr("（可选）网格重建"));
    auto *form = new QFormLayout(gb);

    m_buildMeshCheck = new QCheckBox(tr("同时重建三角网格模型"), this);
    m_buildMeshCheck->setChecked(false);
    form->addRow(QString(), m_buildMeshCheck);

    m_meshMethodCombo = new QComboBox(this);
    m_meshMethodCombo->addItem(tr("体素 TSDF + Marching Cubes（推荐）"), QStringLiteral("voxel_poisson"));
    m_meshMethodCombo->addItem(tr("Ball Pivoting（快速近似）"),            QStringLiteral("ball_pivoting"));
    form->addRow(tr("重建算法:"), m_meshMethodCombo);

    m_voxelResSpin = new QSpinBox(this);
    m_voxelResSpin->setRange(64, 512);
    m_voxelResSpin->setValue(256);
    m_voxelResSpin->setToolTip(tr("体素格沿最长轴的分辨率。256=精度与速度平衡，512=高精度但内存开销大。"));
    form->addRow(tr("体素分辨率:"), m_voxelResSpin);

    m_smoothIterSpin = new QSpinBox(this);
    m_smoothIterSpin->setRange(0, 10);
    m_smoothIterSpin->setValue(1);
    form->addRow(tr("Laplacian 平滑迭代:"), m_smoothIterSpin);

    // 联动：仅网格重建开启时相关控件可用
    auto onMeshToggle = [=](bool checked) {
        m_meshMethodCombo->setEnabled(checked);
        m_voxelResSpin->setEnabled(checked);
        m_smoothIterSpin->setEnabled(checked);
    };
    onMeshToggle(false);
    connect(m_buildMeshCheck, &QCheckBox::toggled, onMeshToggle);
}

// =============================================================================
// 事件槽
// =============================================================================

void DenseCloudDialog::onBrowseOutput()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this, tr("选择输出目录"), m_outputDirEdit->text());
    if (!dir.isEmpty()) m_outputDirEdit->setText(dir);
}

void DenseCloudDialog::onRun()
{
    const QJsonObject settings = collectSettings();
    emit settingsChanged(settings);
    emit runRequested(settings);
    m_runButton->setEnabled(false);
    m_progressBar->setValue(0);
    appendLog(tr("[MVS] 正在启动稠密点云重建..."));
}

void DenseCloudDialog::onCancel()
{
    reject();
}

void DenseCloudDialog::onAnyChanged()
{
    emit settingsChanged(collectSettings());
}

void DenseCloudDialog::onPresetChanged(int index)
{
    const QString preset = m_presetCombo->itemData(index).toString();
    if (preset == QLatin1String("fast")) 
    {
        m_numDispSpin->setValue(64);
        m_blockSizeSpin->setValue(11);
        m_uniquenessSpin->setValue(5);
        m_speckleSizeSpin->setValue(50);
        m_fullDpCheck->setChecked(false);
        m_wlsFilterCheck->setChecked(false);
        m_normalsCheck->setChecked(false);
        m_buildMeshCheck->setChecked(false);
    } else if (preset == QLatin1String("quality")) {
        m_numDispSpin->setValue(256);
        m_blockSizeSpin->setValue(7);
        m_uniquenessSpin->setValue(15);
        m_speckleSizeSpin->setValue(150);
        m_fullDpCheck->setChecked(true);
        m_wlsFilterCheck->setChecked(true);
        m_normalsCheck->setChecked(true);
    } else { // standard
        m_numDispSpin->setValue(128);
        m_blockSizeSpin->setValue(9);
        m_uniquenessSpin->setValue(10);
        m_speckleSizeSpin->setValue(100);
        m_fullDpCheck->setChecked(true);
        m_wlsFilterCheck->setChecked(true);
        m_normalsCheck->setChecked(true);
    }
}

// =============================================================================
// 设置持久化
// =============================================================================

QJsonObject DenseCloudDialog::collectSettings() const
{
    QJsonObject s;
    // at_index: AT结果索引（int），-1 表示始终使用最新结果
    const QVariant atData = m_atResultCombo->currentData();
    s["at_index"]          = atData.isValid() ? atData.toInt() : -1;
    s["at_selection_mode"] = (s["at_index"].toInt(-1) < 0)
        ? QStringLiteral("latest")
        : QStringLiteral("fixed");
    s["output_dir"]        = m_outputDirEdit->text();
    s["preset"]            = m_presetCombo->currentData().toString();

    // SGBM
    s["num_disparities"]   = m_numDispSpin->value();
    s["block_size"]        = m_blockSizeSpin->value();
    s["uniqueness_ratio"]  = m_uniquenessSpin->value();
    s["speckle_window_size"] = m_speckleSizeSpin->value();
    s["use_full_dp"]       = m_fullDpCheck->isChecked();
    s["use_wls_filter"]    = m_wlsFilterCheck->isChecked();
    s["min_depth"]         = m_minDepthSpin->value();
    s["max_depth"]         = m_maxDepthSpin->value();

    // Cloud
    s["min_confidence"]    = m_minConfSpin->value();
    s["multi_view_fusion"] = m_multiViewCheck->isChecked();
    s["output_colors"]     = m_colorsCheck->isChecked();
    s["output_normals"]    = m_normalsCheck->isChecked();
    s["normal_knn"]        = m_normalKnnSpin->value();

    // Mesh
    s["build_mesh"]        = m_buildMeshCheck->isChecked();
    s["mesh_method"]       = m_meshMethodCombo->currentData().toString();
    s["voxel_resolution"]  = m_voxelResSpin->value();
    s["smooth_iterations"] = m_smoothIterSpin->value();

    return s;
}

void DenseCloudDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("at_selection_mode") && s.value("at_selection_mode").toString() == QStringLiteral("latest")) {
        const int latestIdx = m_atResultCombo->findData(-1);
        if (latestIdx >= 0) {
            m_atResultCombo->setCurrentIndex(latestIdx);
        }
    } else if (s.contains("at_index")) {
        const int latestIdx = m_atResultCombo->findData(-1);
        if (!s.contains("at_selection_mode") && latestIdx >= 0) {
            m_atResultCombo->setCurrentIndex(latestIdx);
        } else {
            const int idx = m_atResultCombo->findData(s.value("at_index").toInt(-1));
            if (idx >= 0) {
                m_atResultCombo->setCurrentIndex(idx);
            }
        }
    }
    if (s.contains("output_dir"))       m_outputDirEdit->setText(s["output_dir"].toString());
    if (s.contains("num_disparities"))  m_numDispSpin->setValue(s["num_disparities"].toInt(128));
    if (s.contains("block_size"))       m_blockSizeSpin->setValue(s["block_size"].toInt(9));
    if (s.contains("uniqueness_ratio")) m_uniquenessSpin->setValue(s["uniqueness_ratio"].toInt(10));
    if (s.contains("speckle_window_size")) m_speckleSizeSpin->setValue(s["speckle_window_size"].toInt(100));
    if (s.contains("use_full_dp"))      m_fullDpCheck->setChecked(s["use_full_dp"].toBool(true));
    if (s.contains("use_wls_filter"))   m_wlsFilterCheck->setChecked(s["use_wls_filter"].toBool(true));
    if (s.contains("min_depth"))        m_minDepthSpin->setValue(s["min_depth"].toDouble(0.01));
    if (s.contains("max_depth"))        m_maxDepthSpin->setValue(s["max_depth"].toDouble(1e5));
    if (s.contains("min_confidence"))   m_minConfSpin->setValue(s["min_confidence"].toDouble(0.1));
    if (s.contains("multi_view_fusion")) m_multiViewCheck->setChecked(s["multi_view_fusion"].toBool(true));
    if (s.contains("output_colors"))    m_colorsCheck->setChecked(s["output_colors"].toBool(true));
    if (s.contains("output_normals"))   m_normalsCheck->setChecked(s["output_normals"].toBool(true));
    if (s.contains("normal_knn"))       m_normalKnnSpin->setValue(s["normal_knn"].toInt(20));
    if (s.contains("build_mesh"))       m_buildMeshCheck->setChecked(s["build_mesh"].toBool(false));
    if (s.contains("voxel_resolution")) m_voxelResSpin->setValue(s["voxel_resolution"].toInt(256));
    if (s.contains("smooth_iterations")) m_smoothIterSpin->setValue(s["smooth_iterations"].toInt(1));
}

void DenseCloudDialog::appendLog(const QString &line)
{
    if (m_logEdit) m_logEdit->append(line);
}
