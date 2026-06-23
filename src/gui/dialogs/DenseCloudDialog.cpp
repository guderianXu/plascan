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
#include <QTabWidget>
#include <QTextEdit>

DenseCloudDialog::DenseCloudDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , _projectManager(projectManager)
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

    _imagePairCombo = form.m_imagePairCombo;
    _atResultCombo = form.m_atResultCombo;
    _presetCombo = form.m_presetCombo;
    _outputDirEdit = form.m_outputDirEdit;
    _numDispSpin = form.m_numDispSpin;
    _blockSizeSpin = form.m_blockSizeSpin;
    _uniquenessSpin = form.m_uniquenessSpin;
    _speckleSizeSpin = form.m_speckleSizeSpin;
    _fullDpCheck = form.m_fullDpCheck;
    _wlsFilterCheck = form.m_wlsFilterCheck;
    _minDepthSpin = form.m_minDepthSpin;
    _maxDepthSpin = form.m_maxDepthSpin;
    _minConfSpin = form.m_minConfSpin;
    _multiViewCheck = form.m_multiViewCheck;
    _minConsistentViewsSpin = form.m_minConsistentViewsSpin;
    _geomConsistencyCheck = form.m_geomConsistencyCheck;
    _maxReprojErrorSpin = form.m_maxReprojErrorSpin;
    _speckleMinAreaSpin = form.m_speckleMinAreaSpin;
    _fusionMaxImageDimSpin = form.m_fusionMaxImageDimSpin;
    _colorsCheck = form.m_colorsCheck;
    _normalsCheck = form.m_normalsCheck;
    _normalKnnSpin = form.m_normalKnnSpin;
    _buildMeshCheck = form.m_buildMeshCheck;
    _meshMethodCombo = form.m_meshMethodCombo;
    _voxelResSpin = form.m_voxelResSpin;
    _smoothIterSpin = form.m_smoothIterSpin;
    _progressBar = form.m_progressBar;
    _logEdit = form.m_logEdit;
    _runButton = form.m_runButton;
    _cancelButton = form.m_cancelButton;

    _presetCombo->setItemData(0, QStringLiteral("fast"));
    _presetCombo->setItemData(1, QStringLiteral("standard"));
    _presetCombo->setItemData(2, QStringLiteral("quality"));
    _presetCombo->setCurrentIndex(1);

    _meshMethodCombo->setItemData(0, QStringLiteral("voxel_poisson"));
    _meshMethodCombo->setItemData(1, QStringLiteral("ball_pivoting"));

    if (form.denseTabs && form.sgbmTab)
    {
        const int sgbmTabIndex = form.denseTabs->indexOf(form.sgbmTab);
        if (sgbmTabIndex >= 0)
        {
            form.denseTabs->removeTab(sgbmTabIndex);
        }
        form.sgbmTab->setVisible(false);
    }

    // 填充 AT 结果列表
    if (_projectManager)
    {
        const QJsonArray atResults = _projectManager->getAvailableAtResults();
        if (atResults.isEmpty()) 
        {
            _atResultCombo->addItem(tr("（请先运行空三）"), -2);
        } 
        else 
        {
            _atResultCombo->addItem(tr("最新 AT 结果（推荐）"), -1);
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
                _atResultCombo->addItem(label, idx);
            }
            _atResultCombo->setCurrentIndex(0);
        }
    }

    connect(_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseCloudDialog::onPresetChanged);
    connect(_atResultCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseCloudDialog::onAnyChanged);

    connect(form.browseOutputButton, &QPushButton::clicked, this, &DenseCloudDialog::onBrowseOutput);
    connect(_normalsCheck, &QCheckBox::toggled, _normalKnnSpin, &QSpinBox::setEnabled);

    auto onMeshToggle = [this](bool checked)
    {
        _meshMethodCombo->setEnabled(checked);
        _voxelResSpin->setEnabled(checked);
        _smoothIterSpin->setEnabled(checked);
    };
    onMeshToggle(false);
    connect(_buildMeshCheck, &QCheckBox::toggled, this, onMeshToggle);

    connect(_runButton,    &QPushButton::clicked, this, &DenseCloudDialog::onRun);
    connect(_cancelButton, &QPushButton::clicked, this, &DenseCloudDialog::onCancel);

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
        this, tr("选择输出目录"), _outputDirEdit->text());
    if (!dir.isEmpty()) _outputDirEdit->setText(dir);
}

void DenseCloudDialog::onRun()
{
    const QJsonObject settings = collectSettings();
    emit settingsChanged(settings);
    emit runRequested(settings);
    _runButton->setEnabled(false);
    _progressBar->setValue(0);
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
    const QString preset = _presetCombo->itemData(index).toString();
    if (preset == QLatin1String("fast")) 
    {
        _numDispSpin->setValue(64);
        _blockSizeSpin->setValue(11);
        _uniquenessSpin->setValue(5);
        _speckleSizeSpin->setValue(50);
        _minConsistentViewsSpin->setValue(2);
        _geomConsistencyCheck->setChecked(true);
        _maxReprojErrorSpin->setValue(2.0);
        _speckleMinAreaSpin->setValue(16);
        _fusionMaxImageDimSpin->setValue(2048);
        _fullDpCheck->setChecked(false);
        _wlsFilterCheck->setChecked(false);
        _normalsCheck->setChecked(false);
        _buildMeshCheck->setChecked(false);
    } else if (preset == QLatin1String("quality")) {
        _numDispSpin->setValue(256);
        _blockSizeSpin->setValue(7);
        _uniquenessSpin->setValue(15);
        _speckleSizeSpin->setValue(150);
        _minConsistentViewsSpin->setValue(3);
        _geomConsistencyCheck->setChecked(true);
        _maxReprojErrorSpin->setValue(1.5);
        _speckleMinAreaSpin->setValue(24);
        _fusionMaxImageDimSpin->setValue(2048);
        _fullDpCheck->setChecked(true);
        _wlsFilterCheck->setChecked(true);
        _normalsCheck->setChecked(true);
    } else { // standard
        _numDispSpin->setValue(128);
        _blockSizeSpin->setValue(9);
        _uniquenessSpin->setValue(10);
        _speckleSizeSpin->setValue(100);
        _minConsistentViewsSpin->setValue(2);
        _geomConsistencyCheck->setChecked(true);
        _maxReprojErrorSpin->setValue(2.0);
        _speckleMinAreaSpin->setValue(16);
        _fusionMaxImageDimSpin->setValue(2048);
        _fullDpCheck->setChecked(true);
        _wlsFilterCheck->setChecked(true);
        _normalsCheck->setChecked(true);
    }
}

// =============================================================================
// 设置持久化
// =============================================================================

QJsonObject DenseCloudDialog::collectSettings() const
{
    QJsonObject s;
    // at_index: AT结果索引（int），-1 表示始终使用最新结果
    const QVariant atData = _atResultCombo->currentData();
    s["at_index"]          = atData.isValid() ? atData.toInt() : -1;
    s["at_selection_mode"] = (s["at_index"].toInt(-1) < 0)
        ? QStringLiteral("latest")
        : QStringLiteral("fixed");
    s["output_dir"]        = _outputDirEdit->text();
    s["preset"]            = _presetCombo->currentData().toString();

    const QString preset = s["preset"].toString(QStringLiteral("standard"));
    if (preset == QStringLiteral("fast"))
    {
        s["resScale"] = 0.25;
        s["iterations"] = 4;
    }
    else if (preset == QStringLiteral("quality"))
    {
        s["resScale"] = 0.5;
        s["iterations"] = 10;
    }
    else
    {
        s["resScale"] = 0.5;
        s["iterations"] = 6;
    }
    s["patchSize"] = 11;
    s["minViews"] = _minConsistentViewsSpin->value();
    s["confidence"] = _minConfSpin->value();
    s["minConfidence"] = _minConfSpin->value();
    s["cuda"] = true;

    // Cloud
    s["min_confidence"]    = _minConfSpin->value();
    s["minConsistentViews"] = _minConsistentViewsSpin->value();
    s["geomConsistency"]   = _geomConsistencyCheck->isChecked();
    s["maxReprojError"]    = _maxReprojErrorSpin->value();
    s["speckleMinArea"]    = _speckleMinAreaSpin->value();
    s["fusionMaxImageDim"] = _fusionMaxImageDimSpin->value();
    s["multi_view_fusion"] = _multiViewCheck->isChecked();
    s["output_colors"]     = _colorsCheck->isChecked();
    s["output_normals"]    = _normalsCheck->isChecked();
    s["normal_knn"]        = _normalKnnSpin->value();

    // Mesh
    s["build_mesh"]        = _buildMeshCheck->isChecked();
    s["mesh_method"]       = _meshMethodCombo->currentData().toString();
    s["voxel_resolution"]  = _voxelResSpin->value();
    s["smooth_iterations"] = _smoothIterSpin->value();

    return s;
}

void DenseCloudDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("at_selection_mode") && s.value("at_selection_mode").toString() == QStringLiteral("latest")) {
        const int latestIdx = _atResultCombo->findData(-1);
        if (latestIdx >= 0) {
            _atResultCombo->setCurrentIndex(latestIdx);
        }
    } else if (s.contains("at_index")) {
        const int latestIdx = _atResultCombo->findData(-1);
        if (!s.contains("at_selection_mode") && latestIdx >= 0) {
            _atResultCombo->setCurrentIndex(latestIdx);
        } else {
            const int idx = _atResultCombo->findData(s.value("at_index").toInt(-1));
            if (idx >= 0) {
                _atResultCombo->setCurrentIndex(idx);
            }
        }
    }
    if (s.contains("output_dir"))       _outputDirEdit->setText(s["output_dir"].toString());
    if (s.contains("num_disparities"))  _numDispSpin->setValue(s["num_disparities"].toInt(128));
    if (s.contains("block_size"))       _blockSizeSpin->setValue(s["block_size"].toInt(9));
    if (s.contains("uniqueness_ratio")) _uniquenessSpin->setValue(s["uniqueness_ratio"].toInt(10));
    if (s.contains("speckle_window_size")) _speckleSizeSpin->setValue(s["speckle_window_size"].toInt(100));
    if (s.contains("use_full_dp"))      _fullDpCheck->setChecked(s["use_full_dp"].toBool(true));
    if (s.contains("use_wls_filter"))   _wlsFilterCheck->setChecked(s["use_wls_filter"].toBool(true));
    if (s.contains("min_depth"))        _minDepthSpin->setValue(s["min_depth"].toDouble(0.01));
    if (s.contains("max_depth"))        _maxDepthSpin->setValue(s["max_depth"].toDouble(1e5));
    if (s.contains("min_confidence"))   _minConfSpin->setValue(s["min_confidence"].toDouble(0.1));
    if (s.contains("minConsistentViews")) _minConsistentViewsSpin->setValue(s["minConsistentViews"].toInt(2));
    if (s.contains("geomConsistency"))  _geomConsistencyCheck->setChecked(s["geomConsistency"].toBool(true));
    if (s.contains("maxReprojError"))   _maxReprojErrorSpin->setValue(s["maxReprojError"].toDouble(2.0));
    if (s.contains("speckleMinArea"))   _speckleMinAreaSpin->setValue(s["speckleMinArea"].toInt(16));
    if (s.contains("fusionMaxImageDim")) _fusionMaxImageDimSpin->setValue(s["fusionMaxImageDim"].toInt(2048));
    if (s.contains("multi_view_fusion")) _multiViewCheck->setChecked(s["multi_view_fusion"].toBool(true));
    if (s.contains("output_colors"))    _colorsCheck->setChecked(s["output_colors"].toBool(true));
    if (s.contains("output_normals"))   _normalsCheck->setChecked(s["output_normals"].toBool(true));
    if (s.contains("normal_knn"))       _normalKnnSpin->setValue(s["normal_knn"].toInt(20));
    if (s.contains("build_mesh"))       _buildMeshCheck->setChecked(s["build_mesh"].toBool(false));
    if (s.contains("voxel_resolution")) _voxelResSpin->setValue(s["voxel_resolution"].toInt(256));
    if (s.contains("smooth_iterations")) _smoothIterSpin->setValue(s["smooth_iterations"].toInt(1));
}

void DenseCloudDialog::appendLog(const QString &line)
{
    if (_logEdit) _logEdit->append(line);
}
