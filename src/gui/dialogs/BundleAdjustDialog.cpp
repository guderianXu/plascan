#include "BundleAdjustDialog.h"
#include "ui_BundleAdjustDialog.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
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
#include <QVBoxLayout>
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

        auto *basicForm = ui.basicForm;
        auto *advForm = ui.advForm;
        auto *sysForm = ui.sysForm;
        auto *dbgForm = ui.dbgForm;

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

        m_maxIterationsSpin = new QSpinBox(this);
        m_maxIterationsSpin->setRange(1, 100);
        m_maxIterationsSpin->setValue(12);
        m_maxIterationsSpin->setToolTip(QStringLiteral("外层交替优化迭代次数。过大可能增加耗时。"));

        m_minMatchesSpin = new QSpinBox(this);
        m_minMatchesSpin->setRange(0, 100000);
        m_minMatchesSpin->setValue(10);
        m_minMatchesSpin->setToolTip(QStringLiteral("少于该匹配点数的影像对将被跳过。"));

        m_refinePoseCheck = new QCheckBox(QStringLiteral("优化相机位姿（推荐开启）"), this);
        m_refinePoseCheck->setChecked(true);
        m_refinePoseCheck->setToolTip(QStringLiteral("开启后会同时优化相机外参，可提升整体一致性。"));

        basicForm->addRow(QStringLiteral("外层迭代"), m_maxIterationsSpin);
        basicForm->addRow(QStringLiteral("最小匹配点阈值"), m_minMatchesSpin);
        basicForm->addRow(m_refinePoseCheck);

        m_maxPointItersSpin = new QSpinBox(this);
        m_maxPointItersSpin->setRange(1, 80);
        m_maxPointItersSpin->setValue(8);
        m_maxPointItersSpin->setToolTip(QStringLiteral("每条空间点轨迹的局部优化迭代上限。"));

        m_maxCameraItersSpin = new QSpinBox(this);
        m_maxCameraItersSpin->setRange(1, 80);
        m_maxCameraItersSpin->setValue(5);
        m_maxCameraItersSpin->setToolTip(QStringLiteral("每台相机位姿优化迭代上限。"));

        m_huberDeltaSpin = new QDoubleSpinBox(this);
        m_huberDeltaSpin->setRange(0.0, 1000.0);
        m_huberDeltaSpin->setDecimals(4);
        m_huberDeltaSpin->setValue(3.0);
        m_huberDeltaSpin->setToolTip(QStringLiteral("Huber 鲁棒核阈值（像素）。越小越强抑制粗差。"));

        m_dampingSpin = new QDoubleSpinBox(this);
        m_dampingSpin->setRange(1e-12, 1.0);
        m_dampingSpin->setDecimals(10);
        m_dampingSpin->setSingleStep(1e-6);
        m_dampingSpin->setValue(1e-6);
        m_dampingSpin->setToolTip(QStringLiteral("LM 阻尼系数，过大可能收敛慢，过小可能不稳定。"));

        m_stepTolSpin = new QDoubleSpinBox(this);
        m_stepTolSpin->setRange(1e-10, 1.0);
        m_stepTolSpin->setDecimals(10);
        m_stepTolSpin->setSingleStep(1e-6);
        m_stepTolSpin->setValue(1e-6);
        m_stepTolSpin->setToolTip(QStringLiteral("参数步长收敛阈值，小于该值视为收敛。"));

        m_finiteDiffSpin = new QDoubleSpinBox(this);
        m_finiteDiffSpin->setRange(1e-8, 1.0);
        m_finiteDiffSpin->setDecimals(10);
        m_finiteDiffSpin->setSingleStep(1e-4);
        m_finiteDiffSpin->setValue(1e-4);
        m_finiteDiffSpin->setToolTip(QStringLiteral("数值微分步长，用于雅可比近似。"));

        advForm->addRow(QStringLiteral("点优化迭代"), m_maxPointItersSpin);
        advForm->addRow(QStringLiteral("相机优化迭代"), m_maxCameraItersSpin);
        advForm->addRow(QStringLiteral("Huber δ"), m_huberDeltaSpin);
        advForm->addRow(QStringLiteral("阻尼"), m_dampingSpin);
        advForm->addRow(QStringLiteral("收敛阈值"), m_stepTolSpin);
        advForm->addRow(QStringLiteral("数值微分步长"), m_finiteDiffSpin);

        m_threadsSpin = new QSpinBox(this);
        m_threadsSpin->setRange(0, 128);
        m_threadsSpin->setValue(0);
        m_threadsSpin->setToolTip(QStringLiteral("并行线程数。0 表示自动根据系统决定。"));

        m_chunkSizeSpin = new QSpinBox(this);
        m_chunkSizeSpin->setRange(1, 1000000);
        m_chunkSizeSpin->setValue(2000);
        m_chunkSizeSpin->setToolTip(QStringLiteral("每批处理轨迹数（用于控制内存峰值与响应性）。"));

        sysForm->addRow(QStringLiteral("线程数"), m_threadsSpin);
        sysForm->addRow(QStringLiteral("批处理块大小"), m_chunkSizeSpin);

        m_dryRunCheck = new QCheckBox(QStringLiteral("Dry Run（仅统计输入，不执行优化）"), this);
        m_dryRunCheck->setChecked(false);
        m_dryRunCheck->setToolTip(QStringLiteral("用于检查当前数据是否可运行，不会修改任何结果。"));

        m_exportTsaiCheck = new QCheckBox(QStringLiteral("输出平差后 tsai 文件"), this);
        m_exportTsaiCheck->setChecked(true);
        m_exportTsaiCheck->setToolTip(QStringLiteral("输出到 output_dir/refined_tsai/*.ba.tsai。"));

        m_exportSummaryTxtCheck = new QCheckBox(QStringLiteral("输出 summary 文本"), this);
        m_exportSummaryTxtCheck->setChecked(true);
        m_exportSummaryTxtCheck->setToolTip(QStringLiteral("输出 ba_summary.txt，便于快速查看整体指标。"));

        m_exportPointsCsvCheck = new QCheckBox(QStringLiteral("输出点级指标 CSV"), this);
        m_exportPointsCsvCheck->setChecked(true);
        m_exportPointsCsvCheck->setToolTip(QStringLiteral("输出 ba_points_metrics.csv。"));

        m_exportCameraCsvCheck = new QCheckBox(QStringLiteral("输出相机级指标 CSV"), this);
        m_exportCameraCsvCheck->setChecked(true);
        m_exportCameraCsvCheck->setToolTip(QStringLiteral("输出 ba_camera_metrics.csv。"));

        m_exportRunJsonCheck = new QCheckBox(QStringLiteral("输出运行摘要 JSON"), this);
        m_exportRunJsonCheck->setChecked(true);
        m_exportRunJsonCheck->setToolTip(QStringLiteral("输出 ba_run_summary.json（用于后续读取和复现）。"));

        m_exportEvalPlotCheck = new QCheckBox(QStringLiteral("输出精度评价图像"), this);
        m_exportEvalPlotCheck->setChecked(true);
        m_exportEvalPlotCheck->setToolTip(QStringLiteral("输出 RMS 对比图、相机位移图等 PNG。"));

        dbgForm->addRow(m_exportTsaiCheck);
        dbgForm->addRow(m_exportSummaryTxtCheck);
        dbgForm->addRow(m_exportPointsCsvCheck);
        dbgForm->addRow(m_exportCameraCsvCheck);
        dbgForm->addRow(m_exportRunJsonCheck);
        dbgForm->addRow(m_exportEvalPlotCheck);
        dbgForm->addRow(m_dryRunCheck);

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
        return;
    }

    auto *root = new QVBoxLayout(this);

    auto *imagesBox = new QGroupBox(QStringLiteral("输入影像（勾选参与平差的影像）"), this);
    auto *imagesLay = new QVBoxLayout(imagesBox);
    m_imageList = new QListWidget(imagesBox);
    m_imageList->setSelectionMode(QAbstractItemView::NoSelection);
    imagesLay->addWidget(m_imageList);
    root->addWidget(imagesBox, 1);

    auto *ioBox = new QGroupBox(QStringLiteral("输出目录"), this);
    auto *ioForm = new QFormLayout(ioBox);
    m_outputDirEdit = new QLineEdit(ioBox);
    m_outputDirEdit->setPlaceholderText(QStringLiteral("请选择输出目录（将写入 tsai、txt、csv、json）"));
    m_outputDirEdit->setToolTip(QStringLiteral("只指定目录，不再指定前缀。\n输出内容包括：\n1) 平差后 tsai\n2) 精度评估 txt/csv\n3) 运行摘要 json"));
    auto *chooseBtn = new QToolButton(ioBox);
    chooseBtn->setText(QStringLiteral("选择..."));
    auto *outLay = new QHBoxLayout();
    outLay->addWidget(m_outputDirEdit, 1);
    outLay->addWidget(chooseBtn);
    ioForm->addRow(QStringLiteral("输出目录"), outLay);
    root->addWidget(ioBox);

    // =============================
    // 参数区：基础/高级/系统/调试（除基础外默认折叠）
    // =============================
    QWidget *basicContent = nullptr;
    root->addWidget(createCollapsibleGroup(QStringLiteral("基础参数（常用）"), true, &basicContent));
    auto *basicForm = new QFormLayout(basicContent);

    m_maxIterationsSpin = new QSpinBox(this);
    m_maxIterationsSpin->setRange(1, 100);
    m_maxIterationsSpin->setValue(12);
    m_maxIterationsSpin->setToolTip(QStringLiteral("外层交替优化迭代次数。过大可能增加耗时。"));

    m_minMatchesSpin = new QSpinBox(this);
    m_minMatchesSpin->setRange(0, 100000);
    m_minMatchesSpin->setValue(10);
    m_minMatchesSpin->setToolTip(QStringLiteral("少于该匹配点数的影像对将被跳过。"));

    m_refinePoseCheck = new QCheckBox(QStringLiteral("优化相机位姿（推荐开启）"), this);
    m_refinePoseCheck->setChecked(true);
    m_refinePoseCheck->setToolTip(QStringLiteral("开启后会同时优化相机外参，可提升整体一致性。"));

    basicForm->addRow(QStringLiteral("外层迭代"), m_maxIterationsSpin);
    basicForm->addRow(QStringLiteral("最小匹配点阈值"), m_minMatchesSpin);
    basicForm->addRow(m_refinePoseCheck);

    QWidget *advContent = nullptr;
    root->addWidget(createCollapsibleGroup(QStringLiteral("高级参数（鲁棒/收敛）"), false, &advContent));
    auto *advForm = new QFormLayout(advContent);

    m_maxPointItersSpin = new QSpinBox(this);
    m_maxPointItersSpin->setRange(1, 80);
    m_maxPointItersSpin->setValue(8);
    m_maxPointItersSpin->setToolTip(QStringLiteral("每条空间点轨迹的局部优化迭代上限。"));

    m_maxCameraItersSpin = new QSpinBox(this);
    m_maxCameraItersSpin->setRange(1, 80);
    m_maxCameraItersSpin->setValue(5);
    m_maxCameraItersSpin->setToolTip(QStringLiteral("每台相机位姿优化迭代上限。"));

    m_huberDeltaSpin = new QDoubleSpinBox(this);
    m_huberDeltaSpin->setRange(0.0, 1000.0);
    m_huberDeltaSpin->setDecimals(4);
    m_huberDeltaSpin->setValue(3.0);
    m_huberDeltaSpin->setToolTip(QStringLiteral("Huber 鲁棒核阈值（像素）。越小越强抑制粗差。"));

    m_dampingSpin = new QDoubleSpinBox(this);
    m_dampingSpin->setRange(1e-12, 1.0);
    m_dampingSpin->setDecimals(10);
    m_dampingSpin->setSingleStep(1e-6);
    m_dampingSpin->setValue(1e-6);
    m_dampingSpin->setToolTip(QStringLiteral("LM 阻尼系数，过大可能收敛慢，过小可能不稳定。"));

    m_stepTolSpin = new QDoubleSpinBox(this);
    m_stepTolSpin->setRange(1e-10, 1.0);
    m_stepTolSpin->setDecimals(10);
    m_stepTolSpin->setSingleStep(1e-6);
    m_stepTolSpin->setValue(1e-6);
    m_stepTolSpin->setToolTip(QStringLiteral("参数步长收敛阈值，小于该值视为收敛。"));

    m_finiteDiffSpin = new QDoubleSpinBox(this);
    m_finiteDiffSpin->setRange(1e-8, 1.0);
    m_finiteDiffSpin->setDecimals(10);
    m_finiteDiffSpin->setSingleStep(1e-4);
    m_finiteDiffSpin->setValue(1e-4);
    m_finiteDiffSpin->setToolTip(QStringLiteral("数值微分步长，用于雅可比近似。"));

    advForm->addRow(QStringLiteral("点优化迭代"), m_maxPointItersSpin);
    advForm->addRow(QStringLiteral("相机优化迭代"), m_maxCameraItersSpin);
    advForm->addRow(QStringLiteral("Huber δ"), m_huberDeltaSpin);
    advForm->addRow(QStringLiteral("阻尼"), m_dampingSpin);
    advForm->addRow(QStringLiteral("收敛阈值"), m_stepTolSpin);
    advForm->addRow(QStringLiteral("数值微分步长"), m_finiteDiffSpin);

    QWidget *sysContent = nullptr;
    root->addWidget(createCollapsibleGroup(QStringLiteral("系统参数（并发/资源）"), false, &sysContent));
    auto *sysForm = new QFormLayout(sysContent);

    m_threadsSpin = new QSpinBox(this);
    m_threadsSpin->setRange(0, 128);
    m_threadsSpin->setValue(0);
    m_threadsSpin->setToolTip(QStringLiteral("并行线程数。0 表示自动根据系统决定。"));

    m_chunkSizeSpin = new QSpinBox(this);
    m_chunkSizeSpin->setRange(1, 1000000);
    m_chunkSizeSpin->setValue(2000);
    m_chunkSizeSpin->setToolTip(QStringLiteral("每批处理轨迹数（用于控制内存峰值与响应性）。"));

    sysForm->addRow(QStringLiteral("线程数"), m_threadsSpin);
    sysForm->addRow(QStringLiteral("批处理块大小"), m_chunkSizeSpin);

    QWidget *dbgContent = nullptr;
    root->addWidget(createCollapsibleGroup(QStringLiteral("调试参数（开发/诊断）"), false, &dbgContent));
    auto *dbgForm = new QFormLayout(dbgContent);

    m_dryRunCheck = new QCheckBox(QStringLiteral("Dry Run（仅统计输入，不执行优化）"), this);
    m_dryRunCheck->setChecked(false);
    m_dryRunCheck->setToolTip(QStringLiteral("用于检查当前数据是否可运行，不会修改任何结果。"));

    m_exportTsaiCheck = new QCheckBox(QStringLiteral("输出平差后 tsai 文件"), this);
    m_exportTsaiCheck->setChecked(true);
    m_exportTsaiCheck->setToolTip(QStringLiteral("输出到 output_dir/refined_tsai/*.ba.tsai。"));

    m_exportSummaryTxtCheck = new QCheckBox(QStringLiteral("输出 summary 文本"), this);
    m_exportSummaryTxtCheck->setChecked(true);
    m_exportSummaryTxtCheck->setToolTip(QStringLiteral("输出 ba_summary.txt，便于快速查看整体指标。"));

    m_exportPointsCsvCheck = new QCheckBox(QStringLiteral("输出点级指标 CSV"), this);
    m_exportPointsCsvCheck->setChecked(true);
    m_exportPointsCsvCheck->setToolTip(QStringLiteral("输出 ba_points_metrics.csv。"));

    m_exportCameraCsvCheck = new QCheckBox(QStringLiteral("输出相机级指标 CSV"), this);
    m_exportCameraCsvCheck->setChecked(true);
    m_exportCameraCsvCheck->setToolTip(QStringLiteral("输出 ba_camera_metrics.csv。"));

    m_exportRunJsonCheck = new QCheckBox(QStringLiteral("输出运行摘要 JSON"), this);
    m_exportRunJsonCheck->setChecked(true);
    m_exportRunJsonCheck->setToolTip(QStringLiteral("输出 ba_run_summary.json（用于后续读取和复现）。"));

    m_exportEvalPlotCheck = new QCheckBox(QStringLiteral("输出精度评价图像"), this);
    m_exportEvalPlotCheck->setChecked(true);
    m_exportEvalPlotCheck->setToolTip(QStringLiteral("输出 RMS 对比图、相机位移图等 PNG。"));

    dbgForm->addRow(m_exportTsaiCheck);
    dbgForm->addRow(m_exportSummaryTxtCheck);
    dbgForm->addRow(m_exportPointsCsvCheck);
    dbgForm->addRow(m_exportCameraCsvCheck);
    dbgForm->addRow(m_exportRunJsonCheck);
    dbgForm->addRow(m_exportEvalPlotCheck);
    dbgForm->addRow(m_dryRunCheck);

    // =============================
    // 结果区：在软件中立即查看，决定保留或重跑
    // =============================
    auto *resultBox = new QGroupBox(QStringLiteral("平差结果预览（可立即决定保留或重平差）"), this);
    auto *resultLay = new QVBoxLayout(resultBox);

    m_resultSummaryLabel = new QLabel(QStringLiteral("尚未运行平差。"), resultBox);
    m_resultSummaryLabel->setWordWrap(true);
    resultLay->addWidget(m_resultSummaryLabel);

    m_resultCameraTable = new QTableWidget(resultBox);
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
    resultLay->addWidget(m_resultCameraTable, 1);

    auto *resultBtnLay = new QHBoxLayout();
    m_applyResultBtn = new QToolButton(resultBox);
    m_applyResultBtn->setText(QStringLiteral("保留本次平差结果"));
    m_applyResultBtn->setToolTip(QStringLiteral("将本次平差后的相机参数写回项目并生效。"));

    m_discardResultBtn = new QToolButton(resultBox);
    m_discardResultBtn->setText(QStringLiteral("丢弃本次平差结果"));
    m_discardResultBtn->setToolTip(QStringLiteral("放弃当前预览结果，项目相机参数保持不变。"));

    resultBtnLay->addWidget(m_applyResultBtn);
    resultBtnLay->addWidget(m_discardResultBtn);
    resultBtnLay->addStretch(1);
    resultLay->addLayout(resultBtnLay);

    root->addWidget(resultBox, 1);

    auto *btnLay = new QHBoxLayout();
    auto *restoreBtn = new QPushButton(QStringLiteral("恢复项目默认参数"), this);
    auto *runBtn = new QPushButton(QStringLiteral("运行平差"), this);
    auto *closeBtn = new QPushButton(QStringLiteral("关闭"), this);
    btnLay->addWidget(restoreBtn);
    btnLay->addStretch(1);
    btnLay->addWidget(runBtn);
    btnLay->addWidget(closeBtn);
    root->addLayout(btnLay);

    connect(chooseBtn, &QToolButton::clicked, this, &BundleAdjustDialog::onChooseOutputDir);
    connect(runBtn, &QPushButton::clicked, this, &BundleAdjustDialog::onRun);
    connect(closeBtn, &QPushButton::clicked, this, &BundleAdjustDialog::reject);
    connect(restoreBtn, &QPushButton::clicked, this, &BundleAdjustDialog::requestRestore);

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

QWidget* BundleAdjustDialog::createCollapsibleGroup(const QString &title, bool expandedByDefault, QWidget **contentOut)
{
    auto *wrapper = new QWidget(this);
    auto *layout = new QVBoxLayout(wrapper);
    layout->setContentsMargins(0, 0, 0, 0);

    auto *toggle = new QToolButton(wrapper);
    toggle->setText(title);
    toggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    toggle->setCheckable(true);
    toggle->setChecked(expandedByDefault);
    toggle->setArrowType(expandedByDefault ? Qt::DownArrow : Qt::RightArrow);

    auto *content = new QWidget(wrapper);
    content->setVisible(expandedByDefault);

    connect(toggle, &QToolButton::toggled, wrapper, [toggle, content](bool on) {
        toggle->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
        content->setVisible(on);
    });

    auto *line = new QFrame(wrapper);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);

    layout->addWidget(toggle);
    layout->addWidget(content);
    layout->addWidget(line);

    if (contentOut) *contentOut = content;
    return wrapper;
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
