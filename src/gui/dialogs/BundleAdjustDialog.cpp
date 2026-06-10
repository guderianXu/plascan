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
    resize(980, 760);

    {
        Ui::BundleAdjustDialog ui;
        ui.setupUi(this);

        m_imageList = ui.m_imageList;
        m_outputDirEdit = ui.m_outputDirEdit;
        m_resultSummaryLabel = ui.m_resultSummaryLabel;
        m_resultCameraTable = ui.m_resultCameraTable;
        m_applyResultBtn = ui.m_applyResultBtn;
        m_discardResultBtn = ui.m_discardResultBtn;
        m_threadsSpin = ui.m_threadsSpin;
        m_chunkSizeSpin = ui.m_chunkSizeSpin;
        m_maxIterationsSpin = ui.m_maxIterationsSpin;
        m_maxPointItersSpin = ui.m_maxPointItersSpin;
        m_maxCameraItersSpin = ui.m_maxCameraItersSpin;
        m_minMatchesSpin = ui.m_minMatchesSpin;
        m_huberDeltaSpin = ui.m_huberDeltaSpin;
        m_dampingSpin = ui.m_dampingSpin;
        m_finiteDiffSpin = ui.m_finiteDiffSpin;
        m_stepTolSpin = ui.m_stepTolSpin;
        m_refinePoseCheck = ui.m_refinePoseCheck;
        m_dryRunCheck = ui.m_dryRunCheck;
        m_exportTsaiCheck = ui.m_exportTsaiCheck;
        m_exportSummaryTxtCheck = ui.m_exportSummaryTxtCheck;
        m_exportPointsCsvCheck = ui.m_exportPointsCsvCheck;
        m_exportCameraCsvCheck = ui.m_exportCameraCsvCheck;
        m_exportRunJsonCheck = ui.m_exportRunJsonCheck;
        m_exportEvalPlotCheck = ui.m_exportEvalPlotCheck;

        m_resultCameraTable->setColumnCount(10);
        m_resultCameraTable->setHorizontalHeaderLabels({
            QStringLiteral("影像"), QStringLiteral("ΔC(m)"),
            QStringLiteral("yaw前"), QStringLiteral("yaw后"),
            QStringLiteral("pitch前"), QStringLiteral("pitch后"),
            QStringLiteral("roll前"), QStringLiteral("roll后"),
            QStringLiteral("RMS前"), QStringLiteral("RMS后")
        });
        m_resultCameraTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        m_resultCameraTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        m_resultCameraTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        m_resultCameraTable->horizontalHeader()->setStretchLastSection(true);

        connect(ui.chooseOutputBtn, &QToolButton::clicked, this, &BundleAdjustDialog::onChooseOutputDir);
        connect(ui.runBtn, &QPushButton::clicked, this, &BundleAdjustDialog::onRun);
        connect(ui.closeBtn, &QPushButton::clicked, this, &BundleAdjustDialog::reject);
        connect(ui.restoreBtn, &QPushButton::clicked, this, &BundleAdjustDialog::requestRestore);

        connect(m_applyResultBtn, &QToolButton::clicked, this, &BundleAdjustDialog::onApplyResult);
        connect(m_discardResultBtn, &QToolButton::clicked, this, &BundleAdjustDialog::onDiscardResult);
        connect(m_imageList, &QListWidget::itemChanged, this, &BundleAdjustDialog::emitSettingsNow);

        connect(m_outputDirEdit, &QLineEdit::textChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_chunkSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_maxIterationsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_maxPointItersSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_maxCameraItersSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_minMatchesSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_huberDeltaSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_dampingSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_finiteDiffSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_stepTolSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_refinePoseCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_dryRunCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_exportTsaiCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_exportSummaryTxtCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_exportPointsCsvCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_exportCameraCsvCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_exportRunJsonCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);
        connect(m_exportEvalPlotCheck, &QCheckBox::stateChanged, this, &BundleAdjustDialog::emitSettingsNow);

        updateResultButtons();
    }
}

void BundleAdjustDialog::setAvailableImages(const QStringList &images)
{
    m_imageList->clear();
    const QSet<QString> savedSet(m_savedSelectedImages.begin(), m_savedSelectedImages.end());
    for (const QString &p : images) {
        auto *it = new QListWidgetItem(p, m_imageList);
        it->setFlags(it->flags() | Qt::ItemIsUserCheckable);
        it->setCheckState(savedSet.isEmpty() || savedSet.contains(p) ? Qt::Checked : Qt::Unchecked);
    }
}

void BundleAdjustDialog::setDefaultOutputDir(const QString &dirPath)
{
    if (m_outputDirEdit && m_outputDirEdit->text().trimmed().isEmpty()) {
        m_outputDirEdit->setText(dirPath);
    }
}

void BundleAdjustDialog::applySettings(const QJsonObject &settings)
{
    m_savedSelectedImages.clear();
    const QJsonArray selectedImages = settings.value(QStringLiteral("selected_images")).toArray();
    for (const QJsonValue &value : selectedImages)
    {
        const QString imagePath = value.toString();
        if (!imagePath.isEmpty())
        {
            m_savedSelectedImages.append(imagePath);
        }
    }

    if (settings.contains(QStringLiteral("output_dir"))) m_outputDirEdit->setText(settings.value(QStringLiteral("output_dir")).toString());
    if (m_outputDirEdit->text().trimmed().isEmpty() && settings.contains(QStringLiteral("out_prefix"))) {
        // 兼容旧版本字段：历史上使用 out_prefix。
        m_outputDirEdit->setText(settings.value(QStringLiteral("out_prefix")).toString());
    }
    if (settings.contains(QStringLiteral("threads"))) m_threadsSpin->setValue(settings.value(QStringLiteral("threads")).toInt());
    if (settings.contains(QStringLiteral("chunk_size"))) m_chunkSizeSpin->setValue(settings.value(QStringLiteral("chunk_size")).toInt());
    if (settings.contains(QStringLiteral("max_iterations"))) m_maxIterationsSpin->setValue(settings.value(QStringLiteral("max_iterations")).toInt());
    if (settings.contains(QStringLiteral("max_point_iterations"))) m_maxPointItersSpin->setValue(settings.value(QStringLiteral("max_point_iterations")).toInt());
    if (settings.contains(QStringLiteral("max_camera_iterations"))) m_maxCameraItersSpin->setValue(settings.value(QStringLiteral("max_camera_iterations")).toInt());
    if (settings.contains(QStringLiteral("min_matches"))) m_minMatchesSpin->setValue(settings.value(QStringLiteral("min_matches")).toInt());
    if (settings.contains(QStringLiteral("huber_delta"))) m_huberDeltaSpin->setValue(settings.value(QStringLiteral("huber_delta")).toDouble());
    if (settings.contains(QStringLiteral("damping"))) m_dampingSpin->setValue(settings.value(QStringLiteral("damping")).toDouble());
    if (settings.contains(QStringLiteral("finite_diff_eps"))) m_finiteDiffSpin->setValue(settings.value(QStringLiteral("finite_diff_eps")).toDouble());
    if (settings.contains(QStringLiteral("step_tolerance"))) m_stepTolSpin->setValue(settings.value(QStringLiteral("step_tolerance")).toDouble());
    if (settings.contains(QStringLiteral("refine_camera_pose"))) m_refinePoseCheck->setChecked(settings.value(QStringLiteral("refine_camera_pose")).toBool());
    if (settings.contains(QStringLiteral("dry_run"))) m_dryRunCheck->setChecked(settings.value(QStringLiteral("dry_run")).toBool());
    if (settings.contains(QStringLiteral("export_tsai"))) m_exportTsaiCheck->setChecked(settings.value(QStringLiteral("export_tsai")).toBool());
    if (settings.contains(QStringLiteral("export_summary_txt"))) m_exportSummaryTxtCheck->setChecked(settings.value(QStringLiteral("export_summary_txt")).toBool());
    if (settings.contains(QStringLiteral("export_points_csv"))) m_exportPointsCsvCheck->setChecked(settings.value(QStringLiteral("export_points_csv")).toBool());
    if (settings.contains(QStringLiteral("export_camera_csv"))) m_exportCameraCsvCheck->setChecked(settings.value(QStringLiteral("export_camera_csv")).toBool());
    if (settings.contains(QStringLiteral("export_run_json"))) m_exportRunJsonCheck->setChecked(settings.value(QStringLiteral("export_run_json")).toBool());
    if (settings.contains(QStringLiteral("export_eval_plot"))) m_exportEvalPlotCheck->setChecked(settings.value(QStringLiteral("export_eval_plot")).toBool());

    if (!m_savedSelectedImages.isEmpty() && m_imageList->count() > 0)
    {
        const QSet<QString> savedSet(m_savedSelectedImages.begin(), m_savedSelectedImages.end());
        for (int i = 0; i < m_imageList->count(); ++i)
        {
            QListWidgetItem *item = m_imageList->item(i);
            if (item)
            {
                item->setCheckState(savedSet.contains(item->text()) ? Qt::Checked : Qt::Unchecked);
            }
        }
    }
}

void BundleAdjustDialog::onChooseOutputDir()
{
    const QString outDir = QFileDialog::getExistingDirectory(this,
                                                             QStringLiteral("选择光束法平差输出目录"),
                                                             m_outputDirEdit->text().trimmed());
    if (!outDir.isEmpty()) m_outputDirEdit->setText(outDir);
}

QStringList BundleAdjustDialog::selectedImages() const
{
    QStringList out;
    if (!m_imageList) return out;
    for (int i = 0; i < m_imageList->count(); ++i) {
        QListWidgetItem *it = m_imageList->item(i);
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

    const QString outputDir = m_outputDirEdit->text().trimmed();
    if (outputDir.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请指定输出目录。"));
        return;
    }

    // 清空上一轮结果状态，避免误操作保留旧结果。
    m_hasPendingResult = false;
    updateResultButtons();

    QJsonObject options;
    options[QStringLiteral("max_iterations")] = m_maxIterationsSpin->value();
    options[QStringLiteral("max_point_iterations")] = m_maxPointItersSpin->value();
    options[QStringLiteral("max_camera_iterations")] = m_maxCameraItersSpin->value();
    options[QStringLiteral("chunk_size")] = m_chunkSizeSpin->value();
    options[QStringLiteral("min_matches")] = m_minMatchesSpin->value();
    options[QStringLiteral("huber_delta")] = m_huberDeltaSpin->value();
    options[QStringLiteral("damping")] = m_dampingSpin->value();
    options[QStringLiteral("finite_diff_eps")] = m_finiteDiffSpin->value();
    options[QStringLiteral("step_tolerance")] = m_stepTolSpin->value();
    options[QStringLiteral("refine_camera_pose")] = m_refinePoseCheck->isChecked();
    options[QStringLiteral("export_tsai")] = m_exportTsaiCheck->isChecked();
    options[QStringLiteral("export_summary_txt")] = m_exportSummaryTxtCheck->isChecked();
    options[QStringLiteral("export_points_csv")] = m_exportPointsCsvCheck->isChecked();
    options[QStringLiteral("export_camera_csv")] = m_exportCameraCsvCheck->isChecked();
    options[QStringLiteral("export_run_json")] = m_exportRunJsonCheck->isChecked();
    options[QStringLiteral("export_eval_plot")] = m_exportEvalPlotCheck->isChecked();

    emit requestRunBundleAdjust(images,
                                outputDir,
                                m_threadsSpin->value(),
                                m_dryRunCheck->isChecked(),
                                options);
}

void BundleAdjustDialog::setRunResult(const QJsonObject &result)
{
    if (result.isEmpty()) {
        m_resultSummaryLabel->setText(QStringLiteral("尚未运行平差。"));
        m_resultCameraTable->setRowCount(0);
        m_hasPendingResult = false;
        updateResultButtons();
        return;
    }

    const int trackCount = result.value(QStringLiteral("track_count")).toInt();
    const int optimizedCount = result.value(QStringLiteral("optimized_count")).toInt();
    const double rmsBefore = result.value(QStringLiteral("mean_rms_before")).toDouble();
    const double rmsAfter = result.value(QStringLiteral("mean_rms_after")).toDouble();

    const QJsonObject filesObj = result.value(QStringLiteral("files")).toObject();
    const QString txtFile = filesObj.value(QStringLiteral("summary_txt")).toString();
    const QString pointCsv = filesObj.value(QStringLiteral("points_csv")).toString();
    const QString cameraCsv = filesObj.value(QStringLiteral("camera_csv")).toString();
    const QString runJson = filesObj.value(QStringLiteral("run_json")).toString();

    m_resultSummaryLabel->setText(
        QStringLiteral("平差完成：轨迹 %1，成功优化 %2，平均 RMS: %3 -> %4\n输出：\n- %5\n- %6\n- %7\n- %8")
            .arg(trackCount)
            .arg(optimizedCount)
            .arg(rmsBefore, 0, 'f', 6)
            .arg(rmsAfter, 0, 'f', 6)
            .arg(txtFile)
            .arg(pointCsv)
            .arg(cameraCsv)
            .arg(runJson));

    const QJsonArray cams = result.value(QStringLiteral("camera_preview")).toArray();
    m_resultCameraTable->setRowCount(cams.size());
    for (int i = 0; i < cams.size(); ++i) {
        const QJsonObject one = cams.at(i).toObject();
        m_resultCameraTable->setItem(i, 0, new QTableWidgetItem(one.value(QStringLiteral("image_name")).toString()));
        m_resultCameraTable->setItem(i, 1, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("delta_c_m"), 6)));
        m_resultCameraTable->setItem(i, 2, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("yaw_before"), 6)));
        m_resultCameraTable->setItem(i, 3, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("yaw_after"), 6)));
        m_resultCameraTable->setItem(i, 4, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("pitch_before"), 6)));
        m_resultCameraTable->setItem(i, 5, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("pitch_after"), 6)));
        m_resultCameraTable->setItem(i, 6, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("roll_before"), 6)));
        m_resultCameraTable->setItem(i, 7, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("roll_after"), 6)));
        m_resultCameraTable->setItem(i, 8, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("mean_rms_before"), 6)));
        m_resultCameraTable->setItem(i, 9, new QTableWidgetItem(valueOrEmpty(one, QStringLiteral("mean_rms_after"), 6)));
    }

    m_hasPendingResult = true;
    updateResultButtons();
}

void BundleAdjustDialog::onApplyResult()
{
    if (!m_hasPendingResult) return;
    emit requestApplyBundleAdjustResult();
    m_hasPendingResult = false;
    updateResultButtons();
}

void BundleAdjustDialog::onDiscardResult()
{
    if (!m_hasPendingResult) return;
    emit requestDiscardBundleAdjustResult();
    m_hasPendingResult = false;
    updateResultButtons();
}

void BundleAdjustDialog::updateResultButtons()
{
    if (m_applyResultBtn) m_applyResultBtn->setEnabled(m_hasPendingResult);
    if (m_discardResultBtn) m_discardResultBtn->setEnabled(m_hasPendingResult);
}

void BundleAdjustDialog::emitSettingsNow()
{
    QJsonObject settings;
    settings[QStringLiteral("selected_images")] = QJsonArray::fromStringList(selectedImages());
    settings[QStringLiteral("output_dir")] = m_outputDirEdit->text().trimmed();
    settings[QStringLiteral("threads")] = m_threadsSpin->value();
    settings[QStringLiteral("chunk_size")] = m_chunkSizeSpin->value();
    settings[QStringLiteral("max_iterations")] = m_maxIterationsSpin->value();
    settings[QStringLiteral("max_point_iterations")] = m_maxPointItersSpin->value();
    settings[QStringLiteral("max_camera_iterations")] = m_maxCameraItersSpin->value();
    settings[QStringLiteral("min_matches")] = m_minMatchesSpin->value();
    settings[QStringLiteral("huber_delta")] = m_huberDeltaSpin->value();
    settings[QStringLiteral("damping")] = m_dampingSpin->value();
    settings[QStringLiteral("finite_diff_eps")] = m_finiteDiffSpin->value();
    settings[QStringLiteral("step_tolerance")] = m_stepTolSpin->value();
    settings[QStringLiteral("refine_camera_pose")] = m_refinePoseCheck->isChecked();
    settings[QStringLiteral("dry_run")] = m_dryRunCheck->isChecked();
    settings[QStringLiteral("export_tsai")] = m_exportTsaiCheck->isChecked();
    settings[QStringLiteral("export_summary_txt")] = m_exportSummaryTxtCheck->isChecked();
    settings[QStringLiteral("export_points_csv")] = m_exportPointsCsvCheck->isChecked();
    settings[QStringLiteral("export_camera_csv")] = m_exportCameraCsvCheck->isChecked();
    settings[QStringLiteral("export_run_json")] = m_exportRunJsonCheck->isChecked();
    settings[QStringLiteral("export_eval_plot")] = m_exportEvalPlotCheck->isChecked();
    emit settingsChanged(settings);
}
