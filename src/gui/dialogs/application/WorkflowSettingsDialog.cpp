/**
 * @file WorkflowSettingsDialog.cpp
 * @brief 工作流程高级参数对话框实现。
 */

#include "application/WorkflowSettingsDialog.h"

#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

namespace
{

/// JSON 中缺失字段时，把默认对象补到调用方对象中。
QJsonObject withDefaults(QJsonObject settings)
{
    const QJsonObject defaults = WorkflowSettingsDialog::defaultSettings();
    for (auto it = defaults.constBegin(); it != defaults.constEnd(); ++it)
    {
        if (!settings.contains(it.key()))
        {
            settings.insert(it.key(), it.value());
        }
    }
    return settings;
}

QSpinBox *makeIntegerSpin(int minimum,
                          int maximum,
                          const QString &toolTip,
                          QWidget *parent)
{
    auto *spin = new QSpinBox(parent);
    spin->setRange(minimum, maximum);
    spin->setToolTip(toolTip);
    spin->setKeyboardTracking(false);
    return spin;
}

QDoubleSpinBox *makeRealSpin(double minimum,
                             double maximum,
                             int decimals,
                             double step,
                             const QString &toolTip,
                             QWidget *parent)
{
    auto *spin = new QDoubleSpinBox(parent);
    spin->setRange(minimum, maximum);
    spin->setDecimals(decimals);
    spin->setSingleStep(step);
    spin->setToolTip(toolTip);
    spin->setKeyboardTracking(false);
    return spin;
}

} // namespace

WorkflowSettingsDialog::WorkflowSettingsDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    applySettings(defaultSettings());
}

QJsonObject WorkflowSettingsDialog::defaultSettings()
{
    QJsonObject settings;
    settings[QStringLiteral("workflow_settings_version")] = 1;
    settings[QStringLiteral("algorithm_id")] = QStringLiteral("sift_lightglue");
    settings[QStringLiteral("device")] = QStringLiteral("cuda");
    settings[QStringLiteral("threads")] = std::max(1, QThread::idealThreadCount());
    settings[QStringLiteral("cuda_device")] = 0;
    settings[QStringLiteral("cuda_parallel_pairs")] = 0;
    settings[QStringLiteral("feature_prefetch_depth")] = 2;
    settings[QStringLiteral("feature_max_image_dim")] = 0;
    settings[QStringLiteral("match_threshold")] = 0.15;
    settings[QStringLiteral("geometry_reprojection_threshold_px")] = 1.5;
    settings[QStringLiteral("geometry_min_inliers")] = 20;
    settings[QStringLiteral("geometry_max_iterations")] = 10000;
    settings[QStringLiteral("tie_point_grid_columns")] = 8;
    settings[QStringLiteral("tie_point_grid_rows")] = 8;
    settings[QStringLiteral("tie_point_grid_cell_limit")] = 0;
    settings[QStringLiteral("stationary_tie_point_max_pixel_motion")] = 1.0;
    return settings;
}

void WorkflowSettingsDialog::setupUi()
{
    setWindowTitle(QStringLiteral("工作流程设置"));
    setModal(true);
    resize(620, 690);

    auto *rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(20, 18, 20, 18);
    rootLayout->setSpacing(12);

    auto *scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    auto *content = new QWidget(scrollArea);
    auto *contentLayout = new QVBoxLayout(content);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(12);

    auto *algorithmGroup = new QGroupBox(QStringLiteral("空中三角测量 - 算法与运行资源"), content);
    auto *algorithmForm = new QFormLayout(algorithmGroup);
    algorithmForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    auto *algorithmLabel = new QLabel(QStringLiteral("CUDA SIFT + TensorRT LightGlue"), algorithmGroup);
    algorithmLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    algorithmForm->addRow(QStringLiteral("匹配算法:"), algorithmLabel);

    _cpuThreadsSpin = makeIntegerSpin(
        1, 256, QStringLiteral("SfM、几何验证与 BA 共用的 CPU 线程预算。"), algorithmGroup);
    algorithmForm->addRow(QStringLiteral("CPU 线程预算:"), _cpuThreadsSpin);

    _cudaDeviceSpin = makeIntegerSpin(
        0, 31, QStringLiteral("CUDA SIFT 与 TensorRT LightGlue 使用的设备序号。"), algorithmGroup);
    algorithmForm->addRow(QStringLiteral("CUDA 设备:"), _cudaDeviceSpin);

    _cudaParallelPairsSpin = makeIntegerSpin(
        0, 16, QStringLiteral("同时执行的 LightGlue 像对数量；0 表示按可用显存自动决定。"), algorithmGroup);
    _cudaParallelPairsSpin->setSpecialValueText(QStringLiteral("自动"));
    algorithmForm->addRow(QStringLiteral("GPU 并行像对:"), _cudaParallelPairsSpin);

    _featurePrefetchDepthSpin = makeIntegerSpin(
        1, 4, QStringLiteral("CUDA SIFT 提取时预读到主机内存的影像数。"), algorithmGroup);
    algorithmForm->addRow(QStringLiteral("特征预读影像:"), _featurePrefetchDepthSpin);

    _featureMaxImageDimSpin = makeIntegerSpin(
        0, 32768, QStringLiteral("SIFT 输入影像最长边；0 表示由空三精度预设决定。"), algorithmGroup);
    _featureMaxImageDimSpin->setSpecialValueText(QStringLiteral("由精度预设决定"));
    _featureMaxImageDimSpin->setSuffix(QStringLiteral(" px"));
    algorithmForm->addRow(QStringLiteral("特征输入最长边:"), _featureMaxImageDimSpin);
    contentLayout->addWidget(algorithmGroup);

    auto *matchingGroup = new QGroupBox(QStringLiteral("匹配与几何验证"), content);
    auto *matchingForm = new QFormLayout(matchingGroup);
    matchingForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    _matchThresholdSpin = makeRealSpin(
        0.0, 1.0, 3, 0.01, QStringLiteral("LightGlue 输出匹配的最低置信度。"), matchingGroup);
    matchingForm->addRow(QStringLiteral("匹配置信度门限:"), _matchThresholdSpin);

    _geometryReprojectionSpin = makeRealSpin(
        0.1, 20.0, 2, 0.1, QStringLiteral("基础矩阵 USAC 内点的最大像素残差。"), matchingGroup);
    _geometryReprojectionSpin->setSuffix(QStringLiteral(" px"));
    matchingForm->addRow(QStringLiteral("几何残差门限:"), _geometryReprojectionSpin);

    _geometryMinInliersSpin = makeIntegerSpin(
        8, 10000, QStringLiteral("一个像对进入连接点网络所需的最少几何内点。"), matchingGroup);
    matchingForm->addRow(QStringLiteral("最少几何内点:"), _geometryMinInliersSpin);

    _geometryMaxIterationsSpin = makeIntegerSpin(
        100, 200000, QStringLiteral("USAC 几何模型估计的最大随机采样迭代次数。"), matchingGroup);
    _geometryMaxIterationsSpin->setSingleStep(500);
    matchingForm->addRow(QStringLiteral("USAC 最大迭代:"), _geometryMaxIterationsSpin);
    contentLayout->addWidget(matchingGroup);

    auto *tiePointGroup = new QGroupBox(QStringLiteral("连接点整理"), content);
    auto *tiePointForm = new QFormLayout(tiePointGroup);
    tiePointForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    _tiePointGridColumnsSpin = makeIntegerSpin(
        1, 64, QStringLiteral("连接点空间均匀化网格的列数。"), tiePointGroup);
    tiePointForm->addRow(QStringLiteral("网格列数:"), _tiePointGridColumnsSpin);

    _tiePointGridRowsSpin = makeIntegerSpin(
        1, 64, QStringLiteral("连接点空间均匀化网格的行数。"), tiePointGroup);
    tiePointForm->addRow(QStringLiteral("网格行数:"), _tiePointGridRowsSpin);

    _tiePointGridCellLimitSpin = makeIntegerSpin(
        0, 10000, QStringLiteral("单个网格最多保留的连接点数；0 按影像连接点总限额自动分配。"), tiePointGroup);
    _tiePointGridCellLimitSpin->setSpecialValueText(QStringLiteral("自动"));
    tiePointForm->addRow(QStringLiteral("每网格连接点上限:"), _tiePointGridCellLimitSpin);

    _stationaryMotionSpin = makeRealSpin(
        0.0, 100.0, 2, 0.1, QStringLiteral("跨影像位移不超过该值的轨迹可判为固定连接点。"), tiePointGroup);
    _stationaryMotionSpin->setSuffix(QStringLiteral(" px"));
    tiePointForm->addRow(QStringLiteral("固定点最大位移:"), _stationaryMotionSpin);
    contentLayout->addWidget(tiePointGroup);
    contentLayout->addStretch(1);

    scrollArea->setWidget(content);
    rootLayout->addWidget(scrollArea, 1);

    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::RestoreDefaults | QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
        this);
    buttons->button(QDialogButtonBox::RestoreDefaults)->setText(QStringLiteral("恢复默认值"));
    buttons->button(QDialogButtonBox::Ok)->setText(QStringLiteral("确定"));
    buttons->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("取消"));
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(buttons->button(QDialogButtonBox::RestoreDefaults), &QPushButton::clicked,
            this, [this]()
    {
        applySettings(defaultSettings());
    });
    rootLayout->addWidget(buttons);
}

void WorkflowSettingsDialog::applySettings(const QJsonObject &requestedSettings)
{
    const QJsonObject settings = withDefaults(requestedSettings);
    _cpuThreadsSpin->setValue(settings.value(QStringLiteral("threads")).toInt());
    _cudaDeviceSpin->setValue(settings.value(QStringLiteral("cuda_device")).toInt());
    _cudaParallelPairsSpin->setValue(
        settings.value(QStringLiteral("cuda_parallel_pairs")).toInt());
    _featurePrefetchDepthSpin->setValue(
        settings.value(QStringLiteral("feature_prefetch_depth")).toInt());
    _featureMaxImageDimSpin->setValue(
        settings.value(QStringLiteral("feature_max_image_dim")).toInt());
    _matchThresholdSpin->setValue(
        settings.value(QStringLiteral("match_threshold")).toDouble());
    _geometryReprojectionSpin->setValue(
        settings.value(QStringLiteral("geometry_reprojection_threshold_px")).toDouble());
    _geometryMinInliersSpin->setValue(
        settings.value(QStringLiteral("geometry_min_inliers")).toInt());
    _geometryMaxIterationsSpin->setValue(
        settings.value(QStringLiteral("geometry_max_iterations")).toInt());
    _tiePointGridColumnsSpin->setValue(
        settings.value(QStringLiteral("tie_point_grid_columns")).toInt());
    _tiePointGridRowsSpin->setValue(
        settings.value(QStringLiteral("tie_point_grid_rows")).toInt());
    _tiePointGridCellLimitSpin->setValue(
        settings.value(QStringLiteral("tie_point_grid_cell_limit")).toInt());
    _stationaryMotionSpin->setValue(
        settings.value(QStringLiteral("stationary_tie_point_max_pixel_motion")).toDouble());
}

QJsonObject WorkflowSettingsDialog::collectSettings() const
{
    QJsonObject settings;
    settings[QStringLiteral("workflow_settings_version")] = 1;
    // 当前注册表只提供这一条生产算法。固定 ID 和 CUDA 设备语义写入配置，
    // 以后新增算法时仍由注册表扩展，而不需要修改空三下游文件格式。
    settings[QStringLiteral("algorithm_id")] = QStringLiteral("sift_lightglue");
    settings[QStringLiteral("device")] = QStringLiteral("cuda");
    settings[QStringLiteral("threads")] = _cpuThreadsSpin->value();
    settings[QStringLiteral("cuda_device")] = _cudaDeviceSpin->value();
    settings[QStringLiteral("cuda_parallel_pairs")] = _cudaParallelPairsSpin->value();
    settings[QStringLiteral("feature_prefetch_depth")] = _featurePrefetchDepthSpin->value();
    settings[QStringLiteral("feature_max_image_dim")] = _featureMaxImageDimSpin->value();
    settings[QStringLiteral("match_threshold")] = _matchThresholdSpin->value();
    settings[QStringLiteral("geometry_reprojection_threshold_px")] =
        _geometryReprojectionSpin->value();
    settings[QStringLiteral("geometry_min_inliers")] = _geometryMinInliersSpin->value();
    settings[QStringLiteral("geometry_max_iterations")] = _geometryMaxIterationsSpin->value();
    settings[QStringLiteral("tie_point_grid_columns")] = _tiePointGridColumnsSpin->value();
    settings[QStringLiteral("tie_point_grid_rows")] = _tiePointGridRowsSpin->value();
    settings[QStringLiteral("tie_point_grid_cell_limit")] = _tiePointGridCellLimitSpin->value();
    settings[QStringLiteral("stationary_tie_point_max_pixel_motion")] =
        _stationaryMotionSpin->value();
    return settings;
}
