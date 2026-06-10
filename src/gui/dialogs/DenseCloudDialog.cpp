// =============================================================================
// 文件: DenseCloudDialog.cpp
// 说明: 稠密点云生成对话框实现
// =============================================================================
#include "DenseCloudDialog.h"

#include "ProjectManager.h"
#include "ui_DenseCloudDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QJsonArray>
#include <QJsonObject>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTextEdit>

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
    Ui::DenseCloudDialog form;
    form.setupUi(this);

    m_imagePairCombo = form.m_imagePairCombo;
    m_atResultCombo = form.m_atResultCombo;
    m_presetCombo = form.m_presetCombo;
    m_outputDirEdit = form.m_outputDirEdit;
    m_numDispSpin = form.m_numDispSpin;
    m_blockSizeSpin = form.m_blockSizeSpin;
    m_uniquenessSpin = form.m_uniquenessSpin;
    m_speckleSizeSpin = form.m_speckleSizeSpin;
    m_fullDpCheck = form.m_fullDpCheck;
    m_wlsFilterCheck = form.m_wlsFilterCheck;
    m_minDepthSpin = form.m_minDepthSpin;
    m_maxDepthSpin = form.m_maxDepthSpin;
    m_minConfSpin = form.m_minConfSpin;
    m_multiViewCheck = form.m_multiViewCheck;
    m_colorsCheck = form.m_colorsCheck;
    m_normalsCheck = form.m_normalsCheck;
    m_normalKnnSpin = form.m_normalKnnSpin;
    m_buildMeshCheck = form.m_buildMeshCheck;
    m_meshMethodCombo = form.m_meshMethodCombo;
    m_voxelResSpin = form.m_voxelResSpin;
    m_smoothIterSpin = form.m_smoothIterSpin;
    m_progressBar = form.m_progressBar;
    m_logEdit = form.m_logEdit;
    m_runButton = form.m_runButton;
    m_cancelButton = form.m_cancelButton;

    m_presetCombo->setItemData(0, QStringLiteral("fast"));
    m_presetCombo->setItemData(1, QStringLiteral("standard"));
    m_presetCombo->setItemData(2, QStringLiteral("quality"));
    m_presetCombo->setCurrentIndex(1);

    m_meshMethodCombo->setItemData(0, QStringLiteral("voxel_poisson"));
    m_meshMethodCombo->setItemData(1, QStringLiteral("ball_pivoting"));

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

    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseCloudDialog::onPresetChanged);
    connect(m_atResultCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseCloudDialog::onAnyChanged);

    connect(form.browseOutputButton, &QPushButton::clicked, this, &DenseCloudDialog::onBrowseOutput);
    connect(m_normalsCheck, &QCheckBox::toggled, m_normalKnnSpin, &QSpinBox::setEnabled);

    auto onMeshToggle = [this](bool checked)
    {
        m_meshMethodCombo->setEnabled(checked);
        m_voxelResSpin->setEnabled(checked);
        m_smoothIterSpin->setEnabled(checked);
    };
    onMeshToggle(false);
    connect(m_buildMeshCheck, &QCheckBox::toggled, this, onMeshToggle);

    connect(m_runButton,    &QPushButton::clicked, this, &DenseCloudDialog::onRun);
    connect(m_cancelButton, &QPushButton::clicked, this, &DenseCloudDialog::onCancel);

    // 初始日志
    appendLog(tr("[MVS] 就绪。请确认已完成空中三角测量后再运行稠密重建。"));
    appendLog(tr("[提示] 匹配点数量少 ≠ 点云质量差；MVS 通过像素级密集匹配生成稠密点云。"));
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
