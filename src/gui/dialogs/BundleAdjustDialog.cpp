#include "BundleAdjustDialog.h"
#include "ui_BundleAdjustDialog.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QHeaderView>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QToolButton>
#include <QSet>

namespace {

QString valueOrEmpty(const QJsonObject &obj, const QString &key, int precision = 6)
{
    if (!obj.contains(key)) return QString();
    return QString::number(obj.value(key).toDouble(), 'f', precision);
}

} // namespace

BundleAdjustDialog::BundleAdjustDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("光束法平差"));
    resize(980, 680);

    {
        Ui::BundleAdjustDialog ui;
        ui.setupUi(this);

        _imageList = ui.m_imageList;
        _outputDirEdit = ui.m_outputDirEdit;
        _resultSummaryLabel = ui.m_resultSummaryLabel;
        _resultCameraTable = ui.m_resultCameraTable;
        _applyResultBtn = ui.m_applyResultBtn;
        _discardResultBtn = ui.m_discardResultBtn;
        _threadsSpin = ui.m_threadsSpin;
        _chunkSizeSpin = ui.m_chunkSizeSpin;
        _maxIterationsSpin = ui.m_maxIterationsSpin;
        _maxPointItersSpin = ui.m_maxPointItersSpin;
        _maxCameraItersSpin = ui.m_maxCameraItersSpin;
        _minMatchesSpin = ui.m_minMatchesSpin;
        _huberDeltaSpin = ui.m_huberDeltaSpin;
        _dampingSpin = ui.m_dampingSpin;
        _finiteDiffSpin = ui.m_finiteDiffSpin;
        _stepTolSpin = ui.m_stepTolSpin;
        _refinePoseCheck = ui.m_refinePoseCheck;
        _dryRunCheck = ui.m_dryRunCheck;
        _enableLaserConstraintsCheck = ui.m_enableLaserConstraintsCheck;
        _laserConstraintCloudEdit = ui.m_laserConstraintCloudEdit;
        _chooseLaserConstraintCloudBtn = ui.chooseLaserConstraintCloudBtn;
        _laserAssociationMaxDistanceSpin = ui.m_laserAssociationMaxDistanceSpin;
        _laserVoxelSizeSpin = ui.m_laserVoxelSizeSpin;
        _laserMaxCurvatureSpin = ui.m_laserMaxCurvatureSpin;
        _laserMaxSamplesSpin = ui.m_laserMaxSamplesSpin;
        _laserMissingNormalsAsHeightPlanesCheck = ui.m_laserMissingNormalsAsHeightPlanesCheck;
        _laserWeightSpin = ui.m_laserWeightSpin;
        _laserHuberDeltaSpin = ui.m_laserHuberDeltaSpin;
        _exportTsaiCheck = ui.m_exportTsaiCheck;
        _exportSummaryTxtCheck = ui.m_exportSummaryTxtCheck;
        _exportPointsCsvCheck = ui.m_exportPointsCsvCheck;
        _exportCameraCsvCheck = ui.m_exportCameraCsvCheck;
        _exportRunJsonCheck = ui.m_exportRunJsonCheck;
        _exportEvalPlotCheck = ui.m_exportEvalPlotCheck;

        _resultCameraTable->setColumnCount(10);
        _resultCameraTable->setHorizontalHeaderLabels({
            QStringLiteral("影像"), QStringLiteral("ΔC(m)"),
            QStringLiteral("yaw前"), QStringLiteral("yaw后"),
            QStringLiteral("pitch前"), QStringLiteral("pitch后"),
            QStringLiteral("roll前"), QStringLiteral("roll后"),
            QStringLiteral("RMS前"), QStringLiteral("RMS后")
        });
        _resultCameraTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        _resultCameraTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        _resultCameraTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        _resultCameraTable->horizontalHeader()->setStretchLastSection(true);

        connect(ui.chooseOutputBtn, &QToolButton::clicked, this, &BundleAdjustDialog::onChooseOutputDir);
        connect(_chooseLaserConstraintCloudBtn,
                &QToolButton::clicked,
                this,
                &BundleAdjustDialog::onChooseLaserConstraintCloud);
        connect(ui.runBtn, &QPushButton::clicked, this, &BundleAdjustDialog::onRun);
        connect(ui.closeBtn, &QPushButton::clicked, this, &BundleAdjustDialog::reject);
        connect(ui.restoreBtn, &QPushButton::clicked, this, &BundleAdjustDialog::requestRestore);

        connect(_applyResultBtn, &QToolButton::clicked, this, &BundleAdjustDialog::onApplyResult);
        connect(_discardResultBtn, &QToolButton::clicked, this, &BundleAdjustDialog::onDiscardResult);
        connect(_imageList, &QListWidget::itemChanged, this, &BundleAdjustDialog::emitSettingsNow);

        connect(_outputDirEdit, &QLineEdit::textChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_threadsSpin,
                QOverload<int>::of(&QSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_chunkSizeSpin,
                QOverload<int>::of(&QSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_maxIterationsSpin,
                QOverload<int>::of(&QSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_maxPointItersSpin,
                QOverload<int>::of(&QSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_maxCameraItersSpin,
                QOverload<int>::of(&QSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_minMatchesSpin,
                QOverload<int>::of(&QSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_huberDeltaSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_dampingSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_finiteDiffSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_stepTolSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_refinePoseCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_dryRunCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_enableLaserConstraintsCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_enableLaserConstraintsCheck, &QCheckBox::toggled, this, &BundleAdjustDialog::updateLaserControls);
        connect(_laserConstraintCloudEdit, &QLineEdit::textChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_laserAssociationMaxDistanceSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_laserVoxelSizeSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_laserMaxCurvatureSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_laserMaxSamplesSpin,
                QOverload<int>::of(&QSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_laserMissingNormalsAsHeightPlanesCheck,
                &QCheckBox::stateChanged,
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_laserWeightSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_laserHuberDeltaSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                &BundleAdjustDialog::emitSettingsNow);
        connect(_exportTsaiCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_exportSummaryTxtCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_exportPointsCsvCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_exportCameraCsvCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_exportRunJsonCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(_exportEvalPlotCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);

        updateLaserControls();
        updateResultButtons();
    }
}

void BundleAdjustDialog::setAvailableImages(const QStringList &images)
{
    const bool previousSuppress = _suppressSettingsChanged;
    _suppressSettingsChanged = true;
    _imageList->clear();
    const QSet<QString> savedSet(_savedSelectedImages.begin(), _savedSelectedImages.end());
    for (const QString &p : images) {
        auto *it = new QListWidgetItem(p, _imageList);
        it->setFlags(it->flags() | Qt::ItemIsUserCheckable);
        it->setCheckState(savedSet.isEmpty() || savedSet.contains(p) ? Qt::Checked : Qt::Unchecked);
    }
    _suppressSettingsChanged = previousSuppress;
}

void BundleAdjustDialog::setDefaultOutputDir(const QString &dirPath)
{
    if (_outputDirEdit && _outputDirEdit->text().trimmed().isEmpty()) {
        _outputDirEdit->setText(dirPath);
    }
}

void BundleAdjustDialog::applySettings(const QJsonObject &settings)
{
    const bool previousSuppress = _suppressSettingsChanged;
    _suppressSettingsChanged = true;
    _savedSelectedImages.clear();
    const QJsonArray selectedImages = settings.value(QStringLiteral("selected_images")).toArray();
    for (const QJsonValue &value : selectedImages)
    {
        const QString imagePath = value.toString();
        if (!imagePath.isEmpty())
        {
            _savedSelectedImages.append(imagePath);
        }
    }

    if (settings.contains(QStringLiteral("output_dir")))
    {
        _outputDirEdit->setText(settings.value(QStringLiteral("output_dir")).toString());
    }
    if (_outputDirEdit->text().trimmed().isEmpty() && settings.contains(QStringLiteral("out_prefix")))
    {
        // 兼容旧版本字段：历史上使用 out_prefix。
        _outputDirEdit->setText(settings.value(QStringLiteral("out_prefix")).toString());
    }
    if (settings.contains(QStringLiteral("threads")))
    {
        _threadsSpin->setValue(settings.value(QStringLiteral("threads")).toInt());
    }
    if (settings.contains(QStringLiteral("chunk_size")))
    {
        _chunkSizeSpin->setValue(settings.value(QStringLiteral("chunk_size")).toInt());
    }
    if (settings.contains(QStringLiteral("max_iterations")))
    {
        _maxIterationsSpin->setValue(settings.value(QStringLiteral("max_iterations")).toInt());
    }
    if (settings.contains(QStringLiteral("max_point_iterations")))
    {
        _maxPointItersSpin->setValue(settings.value(QStringLiteral("max_point_iterations")).toInt());
    }
    if (settings.contains(QStringLiteral("max_camera_iterations")))
    {
        _maxCameraItersSpin->setValue(settings.value(QStringLiteral("max_camera_iterations")).toInt());
    }
    if (settings.contains(QStringLiteral("min_matches")))
    {
        _minMatchesSpin->setValue(settings.value(QStringLiteral("min_matches")).toInt());
    }
    if (settings.contains(QStringLiteral("huber_delta")))
    {
        _huberDeltaSpin->setValue(settings.value(QStringLiteral("huber_delta")).toDouble());
    }
    if (settings.contains(QStringLiteral("damping")))
    {
        _dampingSpin->setValue(settings.value(QStringLiteral("damping")).toDouble());
    }
    if (settings.contains(QStringLiteral("finite_diff_eps")))
    {
        _finiteDiffSpin->setValue(settings.value(QStringLiteral("finite_diff_eps")).toDouble());
    }
    if (settings.contains(QStringLiteral("step_tolerance")))
    {
        _stepTolSpin->setValue(settings.value(QStringLiteral("step_tolerance")).toDouble());
    }
    if (settings.contains(QStringLiteral("refine_camera_pose")))
    {
        _refinePoseCheck->setChecked(settings.value(QStringLiteral("refine_camera_pose")).toBool());
    }
    if (settings.contains(QStringLiteral("dry_run")))
    {
        _dryRunCheck->setChecked(settings.value(QStringLiteral("dry_run")).toBool());
    }
    if (settings.contains(QStringLiteral("enable_laser_constraints")))
    {
        _enableLaserConstraintsCheck->setChecked(
            settings.value(QStringLiteral("enable_laser_constraints")).toBool());
    }
    if (settings.contains(QStringLiteral("laser_constraint_cloud_path")))
    {
        _laserConstraintCloudEdit->setText(
            settings.value(QStringLiteral("laser_constraint_cloud_path")).toString());
    }
    if (settings.contains(QStringLiteral("laser_association_max_distance_m")))
    {
        _laserAssociationMaxDistanceSpin->setValue(
            settings.value(QStringLiteral("laser_association_max_distance_m")).toDouble());
    }
    if (settings.contains(QStringLiteral("laser_voxel_size_m")))
    {
        _laserVoxelSizeSpin->setValue(settings.value(QStringLiteral("laser_voxel_size_m")).toDouble());
    }
    if (settings.contains(QStringLiteral("laser_max_curvature")))
    {
        _laserMaxCurvatureSpin->setValue(settings.value(QStringLiteral("laser_max_curvature")).toDouble());
    }
    if (settings.contains(QStringLiteral("laser_max_samples")))
    {
        _laserMaxSamplesSpin->setValue(settings.value(QStringLiteral("laser_max_samples")).toInt());
    }
    if (settings.contains(QStringLiteral("laser_missing_normals_as_height_planes")))
    {
        _laserMissingNormalsAsHeightPlanesCheck->setChecked(
            settings.value(QStringLiteral("laser_missing_normals_as_height_planes")).toBool());
    }
    if (settings.contains(QStringLiteral("laser_weight")))
    {
        _laserWeightSpin->setValue(settings.value(QStringLiteral("laser_weight")).toDouble());
    }
    if (settings.contains(QStringLiteral("laser_huber_delta_m")))
    {
        _laserHuberDeltaSpin->setValue(settings.value(QStringLiteral("laser_huber_delta_m")).toDouble());
    }
    if (settings.contains(QStringLiteral("export_tsai")))
    {
        _exportTsaiCheck->setChecked(settings.value(QStringLiteral("export_tsai")).toBool());
    }
    if (settings.contains(QStringLiteral("export_summary_txt")))
    {
        _exportSummaryTxtCheck->setChecked(settings.value(QStringLiteral("export_summary_txt")).toBool());
    }
    if (settings.contains(QStringLiteral("export_points_csv")))
    {
        _exportPointsCsvCheck->setChecked(settings.value(QStringLiteral("export_points_csv")).toBool());
    }
    if (settings.contains(QStringLiteral("export_camera_csv")))
    {
        _exportCameraCsvCheck->setChecked(settings.value(QStringLiteral("export_camera_csv")).toBool());
    }
    if (settings.contains(QStringLiteral("export_run_json")))
    {
        _exportRunJsonCheck->setChecked(settings.value(QStringLiteral("export_run_json")).toBool());
    }
    if (settings.contains(QStringLiteral("export_eval_plot")))
    {
        _exportEvalPlotCheck->setChecked(settings.value(QStringLiteral("export_eval_plot")).toBool());
    }

    if (!_savedSelectedImages.isEmpty() && _imageList->count() > 0)
    {
        const QSet<QString> savedSet(_savedSelectedImages.begin(), _savedSelectedImages.end());
        for (int i = 0; i < _imageList->count(); ++i)
        {
            QListWidgetItem *item = _imageList->item(i);
            if (item)
            {
                item->setCheckState(savedSet.contains(item->text()) ? Qt::Checked : Qt::Unchecked);
            }
        }
    }

    updateLaserControls();
    _suppressSettingsChanged = previousSuppress;
}

void BundleAdjustDialog::onChooseOutputDir()
{
    const QString outDir = QFileDialog::getExistingDirectory(this,
                                                             QStringLiteral("选择光束法平差输出目录"),
                                                             _outputDirEdit->text().trimmed());
    if (!outDir.isEmpty()) _outputDirEdit->setText(outDir);
}

void BundleAdjustDialog::onChooseLaserConstraintCloud()
{
    const QString cloudPath = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("选择 LiDAR 约束点云"),
        _laserConstraintCloudEdit->text().trimmed(),
        QStringLiteral("PLY 点云 (*.ply);;所有文件 (*)"));
    if (!cloudPath.isEmpty())
    {
        _laserConstraintCloudEdit->setText(cloudPath);
    }
}

QStringList BundleAdjustDialog::selectedImages() const
{
    QStringList out;
    if (!_imageList) return out;
    for (int i = 0; i < _imageList->count(); ++i) {
        QListWidgetItem *it = _imageList->item(i);
        if (it && it->checkState() == Qt::Checked) out.append(it->text());
    }
    return out;
}

void BundleAdjustDialog::onRun()
{
    const QStringList images = selectedImages();
    if (images.size() < 2) {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请至少选择两张影像。"));
        return;
    }

    const QString outputDir = _outputDirEdit->text().trimmed();
    if (outputDir.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请指定输出目录。"));
        return;
    }

    const bool enableLaserConstraints = _enableLaserConstraintsCheck->isChecked();
    const QString laserCloudPath = _laserConstraintCloudEdit->text().trimmed();
    if (enableLaserConstraints && laserCloudPath.isEmpty())
    {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请指定 LiDAR 约束点云 PLY 文件。"));
        return;
    }

    // 清空上一轮结果状态，避免误操作保留旧结果。
    _hasPendingResult = false;
    updateResultButtons();

    QJsonObject options;
    options[QStringLiteral("max_iterations")] = _maxIterationsSpin->value();
    options[QStringLiteral("max_point_iterations")] = _maxPointItersSpin->value();
    options[QStringLiteral("max_camera_iterations")] = _maxCameraItersSpin->value();
    options[QStringLiteral("chunk_size")] = _chunkSizeSpin->value();
    options[QStringLiteral("min_matches")] = _minMatchesSpin->value();
    options[QStringLiteral("huber_delta")] = _huberDeltaSpin->value();
    options[QStringLiteral("damping")] = _dampingSpin->value();
    options[QStringLiteral("finite_diff_eps")] = _finiteDiffSpin->value();
    options[QStringLiteral("step_tolerance")] = _stepTolSpin->value();
    options[QStringLiteral("refine_camera_pose")] = _refinePoseCheck->isChecked();
    options[QStringLiteral("ba_backend")] = QStringLiteral("auto");
    options[QStringLiteral("ba_cuda_device")] = 0;
    options[QStringLiteral("ba_min_cuda_cameras")] = 50;
    options[QStringLiteral("ba_min_cuda_observations")] = 500000;
    options[QStringLiteral("ba_native_cuda_device")] = 0;
    options[QStringLiteral("ba_min_native_cuda_cameras")] = 50;
    options[QStringLiteral("ba_min_native_cuda_observations")] = 500000;
    options[QStringLiteral("ba_native_cuda_max_pcg_iterations")] = 100;
    options[QStringLiteral("ba_native_cuda_pcg_tolerance")] = 1e-4;
    options[QStringLiteral("ba_min_cpu_observations")] = 50000;
    options[QStringLiteral("ba_max_ceres_point_only_observations")] = 100000;
    options[QStringLiteral("ba_allow_backend_fallback")] = true;
    options[QStringLiteral("ba_enable_backend_quality_gate")] = true;
    options[QStringLiteral("ba_max_accepted_rms_growth")] = 1.25;
    options[QStringLiteral("ba_min_accepted_valid_track_ratio")] = 0.60;
    options[QStringLiteral("ba_compare_auto_backend_with_legacy")] = true;
    options[QStringLiteral("enable_laser_constraints")] = enableLaserConstraints;
    options[QStringLiteral("laser_constraint_cloud_path")] = laserCloudPath;
    options[QStringLiteral("laser_association_max_distance_m")] = _laserAssociationMaxDistanceSpin->value();
    options[QStringLiteral("laser_voxel_size_m")] = _laserVoxelSizeSpin->value();
    options[QStringLiteral("laser_max_curvature")] = _laserMaxCurvatureSpin->value();
    options[QStringLiteral("laser_max_samples")] = _laserMaxSamplesSpin->value();
    options[QStringLiteral("laser_missing_normals_as_height_planes")] =
        _laserMissingNormalsAsHeightPlanesCheck->isChecked();
    options[QStringLiteral("laser_weight")] = _laserWeightSpin->value();
    options[QStringLiteral("laser_huber_delta_m")] = _laserHuberDeltaSpin->value();
    options[QStringLiteral("export_tsai")] = _exportTsaiCheck->isChecked();
    options[QStringLiteral("export_summary_txt")] = _exportSummaryTxtCheck->isChecked();
    options[QStringLiteral("export_points_csv")] = _exportPointsCsvCheck->isChecked();
    options[QStringLiteral("export_camera_csv")] = _exportCameraCsvCheck->isChecked();
    options[QStringLiteral("export_run_json")] = _exportRunJsonCheck->isChecked();
    options[QStringLiteral("export_eval_plot")] = _exportEvalPlotCheck->isChecked();

    emit requestRunBundleAdjust(images,
                                outputDir,
                                _threadsSpin->value(),
                                _dryRunCheck->isChecked(),
                                options);
}

void BundleAdjustDialog::setRunResult(const QJsonObject &result)
{
    if (result.isEmpty()) {
        _resultSummaryLabel->setText(QStringLiteral("尚未运行平差。"));
        _resultCameraTable->setRowCount(0);
        _hasPendingResult = false;
        updateResultButtons();
        return;
    }

    const int trackCount = result.value(QStringLiteral("track_count")).toInt();
    const int optimizedCount = result.value(QStringLiteral("optimized_count")).toInt();
    const double rmsBefore = result.value(QStringLiteral("mean_rms_before")).toDouble();
    const double rmsAfter = result.value(QStringLiteral("mean_rms_after")).toDouble();
    const QString requestedBackend =
        result.value(QStringLiteral("ba_requested_backend")).toString(QStringLiteral("legacy_cpu"));
    const QString usedBackend =
        result.value(QStringLiteral("ba_used_backend")).toString(QStringLiteral("legacy_cpu"));
    const bool usedGpu = result.value(QStringLiteral("ba_used_gpu")).toBool(false);
    const bool backendFallback = result.value(QStringLiteral("ba_backend_fallback")).toBool(false);
    const QString linearSolver =
        result.value(QStringLiteral("ba_ceres_linear_solver")).toString(QStringLiteral("none"));
    const int observationCount = result.value(QStringLiteral("ba_observation_count")).toInt();
    const double totalSeconds = result.value(QStringLiteral("ba_total_seconds")).toDouble();
    const double validTrackRatio = result.value(QStringLiteral("ba_valid_track_ratio")).toDouble(0.0);
    const QString backendReason =
        result.value(QStringLiteral("ba_backend_selection_reason")).toString();
    const bool qualityGateRejected =
        result.value(QStringLiteral("ba_quality_gate_rejected")).toBool(false);
    const QString qualityGateMessage =
        result.value(QStringLiteral("ba_quality_gate_message")).toString();

    const QJsonObject filesObj = result.value(QStringLiteral("files")).toObject();
    const QString txtFile = filesObj.value(QStringLiteral("summary_txt")).toString();
    const QString pointCsv = filesObj.value(QStringLiteral("points_csv")).toString();
    const QString cameraCsv = filesObj.value(QStringLiteral("camera_csv")).toString();
    const QString runJson = filesObj.value(QStringLiteral("run_json")).toString();

    QString summary =
        QStringLiteral("平差完成：轨迹 %1，成功优化 %2，平均 RMS: %3 -> %4\n"
                       "BA 后端: %5 -> %6，GPU: %7，回退: %8\n"
                       "观测: %9，有效轨迹比例: %10，求解器: %11，总耗时: %12 s\n"
                       "输出：\n- %13\n- %14\n- %15\n- %16")
            .arg(trackCount)
            .arg(optimizedCount)
            .arg(rmsBefore, 0, 'f', 6)
            .arg(rmsAfter, 0, 'f', 6)
            .arg(requestedBackend,
                 usedBackend,
                  usedGpu ? QStringLiteral("是") : QStringLiteral("否"),
                  backendFallback ? QStringLiteral("是") : QStringLiteral("否"))
            .arg(observationCount)
            .arg(validTrackRatio, 0, 'f', 4)
            .arg(linearSolver)
            .arg(totalSeconds, 0, 'f', 3)
            .arg(txtFile)
            .arg(pointCsv)
            .arg(cameraCsv)
            .arg(runJson);
    if (!backendReason.isEmpty())
    {
        summary += QStringLiteral("\n后端说明: %1").arg(backendReason);
    }
    if (qualityGateRejected || !qualityGateMessage.isEmpty())
    {
        summary += QStringLiteral("\n质量门控: %1")
                       .arg(qualityGateRejected ? QStringLiteral("拒绝候选后端") : QStringLiteral("通过"));
        if (!qualityGateMessage.isEmpty())
        {
            summary += QStringLiteral("，%1").arg(qualityGateMessage);
        }
    }
    _resultSummaryLabel->setText(summary);

    const QJsonArray cams = result.value(QStringLiteral("camera_preview")).toArray();
    _resultCameraTable->setRowCount(cams.size());
    for (int i = 0; i < cams.size(); ++i) {
        const QJsonObject one = cams.at(i).toObject();
        _resultCameraTable->setItem(i, 0, new QTableWidgetItem(one.value(QStringLiteral("image_name")).toString()));
        _resultCameraTable->setItem(i, 1, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("delta_c_m"), 6)));
        _resultCameraTable->setItem(i, 2, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("yaw_before"), 6)));
        _resultCameraTable->setItem(i, 3, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("yaw_after"), 6)));
        _resultCameraTable->setItem(i, 4, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("pitch_before"), 6)));
        _resultCameraTable->setItem(i, 5, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("pitch_after"), 6)));
        _resultCameraTable->setItem(i, 6, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("roll_before"), 6)));
        _resultCameraTable->setItem(i, 7, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("roll_after"), 6)));
        _resultCameraTable->setItem(
            i, 8, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("mean_rms_before"), 6)));
        _resultCameraTable->setItem(
            i, 9, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("mean_rms_after"), 6)));
    }

    _hasPendingResult = true;
    updateResultButtons();
}

void BundleAdjustDialog::onApplyResult()
{
    if (!_hasPendingResult) return;
    emit requestApplyBundleAdjustResult();
    _hasPendingResult = false;
    updateResultButtons();
}

void BundleAdjustDialog::onDiscardResult()
{
    if (!_hasPendingResult) return;
    emit requestDiscardBundleAdjustResult();
    _hasPendingResult = false;
    updateResultButtons();
}

void BundleAdjustDialog::updateResultButtons()
{
    if (_applyResultBtn) _applyResultBtn->setEnabled(_hasPendingResult);
    if (_discardResultBtn) _discardResultBtn->setEnabled(_hasPendingResult);
}

void BundleAdjustDialog::updateLaserControls()
{
    const bool enabled = _enableLaserConstraintsCheck && _enableLaserConstraintsCheck->isChecked();
    const QList<QWidget *> widgets = {
        _laserConstraintCloudEdit,
        _chooseLaserConstraintCloudBtn,
        _laserAssociationMaxDistanceSpin,
        _laserVoxelSizeSpin,
        _laserMaxCurvatureSpin,
        _laserMaxSamplesSpin,
        _laserMissingNormalsAsHeightPlanesCheck,
        _laserWeightSpin,
        _laserHuberDeltaSpin
    };
    for (QWidget *widget : widgets)
    {
        if (widget)
        {
            widget->setEnabled(enabled);
        }
    }
}

void BundleAdjustDialog::emitSettingsNow()
{
    if (_suppressSettingsChanged)
    {
        return;
    }

    QJsonObject settings;
    settings[QStringLiteral("selected_images")] = QJsonArray::fromStringList(selectedImages());
    settings[QStringLiteral("output_dir")] = _outputDirEdit->text().trimmed();
    settings[QStringLiteral("threads")] = _threadsSpin->value();
    settings[QStringLiteral("chunk_size")] = _chunkSizeSpin->value();
    settings[QStringLiteral("max_iterations")] = _maxIterationsSpin->value();
    settings[QStringLiteral("max_point_iterations")] = _maxPointItersSpin->value();
    settings[QStringLiteral("max_camera_iterations")] = _maxCameraItersSpin->value();
    settings[QStringLiteral("min_matches")] = _minMatchesSpin->value();
    settings[QStringLiteral("huber_delta")] = _huberDeltaSpin->value();
    settings[QStringLiteral("damping")] = _dampingSpin->value();
    settings[QStringLiteral("finite_diff_eps")] = _finiteDiffSpin->value();
    settings[QStringLiteral("step_tolerance")] = _stepTolSpin->value();
    settings[QStringLiteral("refine_camera_pose")] = _refinePoseCheck->isChecked();
    settings[QStringLiteral("ba_backend")] = QStringLiteral("auto");
    settings[QStringLiteral("ba_cuda_device")] = 0;
    settings[QStringLiteral("ba_min_cuda_cameras")] = 50;
    settings[QStringLiteral("ba_min_cuda_observations")] = 500000;
    settings[QStringLiteral("ba_native_cuda_device")] = 0;
    settings[QStringLiteral("ba_min_native_cuda_cameras")] = 50;
    settings[QStringLiteral("ba_min_native_cuda_observations")] = 500000;
    settings[QStringLiteral("ba_native_cuda_max_pcg_iterations")] = 100;
    settings[QStringLiteral("ba_native_cuda_pcg_tolerance")] = 1e-4;
    settings[QStringLiteral("ba_min_cpu_observations")] = 50000;
    settings[QStringLiteral("ba_max_ceres_point_only_observations")] = 100000;
    settings[QStringLiteral("ba_allow_backend_fallback")] = true;
    settings[QStringLiteral("ba_enable_backend_quality_gate")] = true;
    settings[QStringLiteral("ba_max_accepted_rms_growth")] = 1.25;
    settings[QStringLiteral("ba_min_accepted_valid_track_ratio")] = 0.60;
    settings[QStringLiteral("ba_compare_auto_backend_with_legacy")] = true;
    settings[QStringLiteral("dry_run")] = _dryRunCheck->isChecked();
    settings[QStringLiteral("enable_laser_constraints")] = _enableLaserConstraintsCheck->isChecked();
    settings[QStringLiteral("laser_constraint_cloud_path")] = _laserConstraintCloudEdit->text().trimmed();
    settings[QStringLiteral("laser_association_max_distance_m")] = _laserAssociationMaxDistanceSpin->value();
    settings[QStringLiteral("laser_voxel_size_m")] = _laserVoxelSizeSpin->value();
    settings[QStringLiteral("laser_max_curvature")] = _laserMaxCurvatureSpin->value();
    settings[QStringLiteral("laser_max_samples")] = _laserMaxSamplesSpin->value();
    settings[QStringLiteral("laser_missing_normals_as_height_planes")] =
        _laserMissingNormalsAsHeightPlanesCheck->isChecked();
    settings[QStringLiteral("laser_weight")] = _laserWeightSpin->value();
    settings[QStringLiteral("laser_huber_delta_m")] = _laserHuberDeltaSpin->value();
    settings[QStringLiteral("export_tsai")] = _exportTsaiCheck->isChecked();
    settings[QStringLiteral("export_summary_txt")] = _exportSummaryTxtCheck->isChecked();
    settings[QStringLiteral("export_points_csv")] = _exportPointsCsvCheck->isChecked();
    settings[QStringLiteral("export_camera_csv")] = _exportCameraCsvCheck->isChecked();
    settings[QStringLiteral("export_run_json")] = _exportRunJsonCheck->isChecked();
    settings[QStringLiteral("export_eval_plot")] = _exportEvalPlotCheck->isChecked();
    emit settingsChanged(settings);
}
