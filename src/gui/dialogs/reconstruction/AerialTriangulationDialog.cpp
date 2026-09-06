/**
 * @file AerialTriangulationDialog.cpp
 * @brief 空中三角测量参数对话框的界面初始化和 JSON 配置映射。
 *
 * 本文件不执行重建算法；真正的连接点准备、增量 SfM 和 BA 由上层工作流服务负责。
 */
#include "reconstruction/AerialTriangulationDialog.h"
#include "shared/WorkflowParameterDialogStyle.h"
#include "ui_AerialTriangulationDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QJsonObject>
#include <QLayout>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QToolButton>

namespace
{

    constexpr int kReferenceTiePointLimit = 4000;
    constexpr int kTiePointDefaultVersion = 2;

    // 组合框显示中文名称，itemData 保存供配置和 workflow 使用的稳定标识。
    void setComboByData(QComboBox* combo, const QString& data)
    {
        if (!combo)
        {
            return;
        }

        const int index = combo->findData(data);
        if (index >= 0)
        {
            combo->setCurrentIndex(index);
        }
    }

    QString comboDataOr(QComboBox* combo, const QString& fallback)
    {
        if (!combo)
        {
            return fallback;
        }

        const QString value = combo->currentData().toString();
        return value.isEmpty() ? fallback : value;
    }

} // namespace

AerialTriangulationDialog::AerialTriangulationDialog(QWidget* parent)
    : QDialog(parent), _ui(std::make_unique<Ui::AerialTriangulationDialog>())
{
    _ui->setupUi(this);
    setupUi();
}

AerialTriangulationDialog::~AerialTriangulationDialog() = default;

void AerialTriangulationDialog::setupUi()
{
    setWindowTitle(QStringLiteral("空中三角测量"));
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);
    _ui->m_statusLabel->setWordWrap(true);

    // 中文文本只用于显示；英文 token 会原样进入工作流配置。
    _ui->m_qualityCombo->clear();
    _ui->m_qualityCombo->addItem(QStringLiteral("最高"), QStringLiteral("highest"));
    _ui->m_qualityCombo->addItem(QStringLiteral("高"), QStringLiteral("high"));
    _ui->m_qualityCombo->addItem(QStringLiteral("中"), QStringLiteral("medium"));
    _ui->m_qualityCombo->addItem(QStringLiteral("低"), QStringLiteral("low"));
    _ui->m_qualityCombo->addItem(QStringLiteral("最低"), QStringLiteral("lowest"));
    setComboByData(_ui->m_qualityCombo, QStringLiteral("high"));

    _ui->m_referenceSourceCombo->clear();
    _ui->m_referenceSourceCombo->addItem(QStringLiteral("导入参考"), QStringLiteral("source_code"));
    _ui->m_referenceSourceCombo->addItem(QStringLiteral("已有 SfM 查漏"), QStringLiteral("estimated"));
    _ui->m_referenceSourceCombo->addItem(QStringLiteral("照片序列"), QStringLiteral("sequence"));
    _ui->m_referenceSourceCombo->setItemData(
        0, QStringLiteral("使用已导入相机文件、影像元数据或外方位参考生成候选匹配对。"), Qt::ToolTipRole);
    _ui->m_referenceSourceCombo->setItemData(
        1,
        QStringLiteral("使用当前 SfM 相机、稀疏点共视和场景视锥重叠查找遗漏或薄弱影像对；"
                       "不使用地球球面。重置对齐时旧位姿仅用于查漏，不直接作为新解初值。"),
        Qt::ToolTipRole);
    _ui->m_referenceSourceCombo->setItemData(
        2, QStringLiteral("按影像顺序生成邻近候选对，适合视频帧或绕目标连续拍摄。"), Qt::ToolTipRole);
    setComboByData(_ui->m_referenceSourceCombo, QStringLiteral("source_code"));

    _ui->m_maskApplyCombo->clear();
    _ui->m_maskApplyCombo->addItem(QStringLiteral("无"), QStringLiteral("none"));
    _ui->m_maskApplyCombo->addItem(QStringLiteral("关键点"), QStringLiteral("keypoints"));
    _ui->m_maskApplyCombo->addItem(QStringLiteral("连接点"), QStringLiteral("tiepoints"));
    setComboByData(_ui->m_maskApplyCombo, QStringLiteral("keypoints"));

    // 这里集中设置 GUI 的推荐初值；打开已有项目时，applySettings() 会用已保存配置覆盖它们。
    _ui->m_genericPreselectionCheck->setChecked(true);
    _ui->m_referencePreselectionCheck->setChecked(false);
    _ui->m_referenceSourceCombo->setEnabled(true);
    _ui->m_resetAlignmentCheck->setChecked(true);
    _ui->m_saveAfterEachStepCheck->setChecked(false);
    _ui->m_keypointLimitSpin->setRange(0, 1000000);
    _ui->m_keypointLimitSpin->setValue(40000);
    _ui->m_tiepointLimitSpin->setRange(0, 1000000);
    _ui->m_tiepointLimitSpin->setValue(kReferenceTiePointLimit);
    _ui->m_tiepointLimitSpin->setToolTip(
        QStringLiteral("每幅影像进入多视重建的连接点上限；参考默认值为 4000。修改后会显示缓存的实际配额，"
                       "并明确说明是直接复用还是只重建连接点网络。"));
    _ui->m_excludeFixedTiePointsCheck->setChecked(true);
    _ui->m_guidedImageMatchingCheck->setChecked(false);
    _ui->m_adaptiveCameraModelCheck->setChecked(true);
    _ui->m_adaptiveCameraModelCheck->setToolTip(
        QStringLiteral("控制正式 BA 是否联合优化共享焦距。关闭后仍会在没有完整相机先验时执行焦距初始化搜索，"
                       "并将最佳焦距固定用于重建。"));
    _ui->m_reuseExistingMatchesCheck->setChecked(true);
    _ui->m_reuseExistingMatchesCheck->setToolTip(
        QStringLiteral("这是缓存策略，不表示当前工程一定已有匹配：存在兼容缓存时只重新执行 SfM/BA；"
                       "没有缓存或缓存不完整时会自动生成缺失匹配。取消勾选会强制删除并重建全部匹配缓存。"));
    _ui->m_lockInputCameraPosesCheck->setChecked(false);
    _ui->m_lockInputCameraPosesCheck->setToolTip(
        QStringLiteral("使用项目中已导入的相机内外参，并在空三和 BA 中保持外参不变。"
                       "适用于 Middlebury、COLMAP 或测量系统提供的真实标定位姿。"));
    xjw::gui::dialogs::configureWorkflowComboBox(_ui->m_qualityCombo);
    xjw::gui::dialogs::configureWorkflowComboBox(_ui->m_referenceSourceCombo);
    xjw::gui::dialogs::configureWorkflowInputWidget(_ui->m_keypointLimitSpin);
    xjw::gui::dialogs::configureWorkflowInputWidget(_ui->m_tiepointLimitSpin);
    xjw::gui::dialogs::configureWorkflowComboBox(_ui->m_maskApplyCombo);
    for (QCheckBox* check_box : {_ui->m_genericPreselectionCheck,
                                 _ui->m_referencePreselectionCheck,
                                 _ui->m_resetAlignmentCheck,
                                 _ui->m_saveAfterEachStepCheck,
                                 _ui->m_excludeFixedTiePointsCheck,
                                 _ui->m_guidedImageMatchingCheck,
                                 _ui->m_adaptiveCameraModelCheck,
                                 _ui->m_reuseExistingMatchesCheck,
                                 _ui->m_lockInputCameraPosesCheck})
    {
        xjw::gui::dialogs::configureWorkflowCheckBox(check_box);
    }
    xjw::gui::dialogs::configureWorkflowButtonBox(_ui->m_buttonBox, tr("开始"));
    setReferencePreselectionAvailable(false, 0, 0);
    setAdvancedExpanded(false);

    // 参数发生变化时发送完整配置，而不是只发送单个控件值，便于上层统一持久化和校验。
    connect(_ui->m_buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(_ui->m_buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(
        _ui->m_advancedToggle, &QToolButton::toggled, this, [this](bool expanded) { setAdvancedExpanded(expanded); });

    connect(_ui->m_referencePreselectionCheck,
            &QCheckBox::toggled,
            this,
            [this](bool enabled)
            {
                Q_UNUSED(enabled)
                emitSettingsChanged();
            });

    connect(_ui->m_qualityCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &AerialTriangulationDialog::emitSettingsChanged);
    connect(_ui->m_referenceSourceCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            [this]()
            {
                if (!_applyingSettings && !_ui->m_referencePreselectionCheck->isChecked())
                {
                    _ui->m_referencePreselectionCheck->setChecked(true);
                    return;
                }
                emitSettingsChanged();
            });
    connect(_ui->m_maskApplyCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &AerialTriangulationDialog::emitSettingsChanged);

    const auto connectCheckBox = [this](QCheckBox* checkBox)
    { connect(checkBox, &QCheckBox::toggled, this, &AerialTriangulationDialog::emitSettingsChanged); };
    connectCheckBox(_ui->m_genericPreselectionCheck);
    connect(_ui->m_resetAlignmentCheck,
            &QCheckBox::toggled,
            this,
            [this](bool checked)
            {
                if (checked && _ui->m_lockInputCameraPosesCheck->isChecked())
                {
                    _ui->m_lockInputCameraPosesCheck->setChecked(false);
                }
                emitSettingsChanged();
            });
    connectCheckBox(_ui->m_saveAfterEachStepCheck);
    connectCheckBox(_ui->m_excludeFixedTiePointsCheck);
    connectCheckBox(_ui->m_guidedImageMatchingCheck);
    connectCheckBox(_ui->m_adaptiveCameraModelCheck);
    connectCheckBox(_ui->m_reuseExistingMatchesCheck);

    // “重新对齐”与“锁定输入相机位姿”语义互斥，界面始终只允许其中一个生效。
    connect(_ui->m_lockInputCameraPosesCheck,
            &QCheckBox::toggled,
            this,
            [this](bool checked)
            {
                if (checked && _ui->m_resetAlignmentCheck->isChecked())
                {
                    _ui->m_resetAlignmentCheck->setChecked(false);
                }
                emitSettingsChanged();
            });

    connect(_ui->m_keypointLimitSpin,
            QOverload<int>::of(&QSpinBox::valueChanged),
            this,
            &AerialTriangulationDialog::emitSettingsChanged);
    connect(_ui->m_tiepointLimitSpin,
            QOverload<int>::of(&QSpinBox::valueChanged),
            this,
            [this]()
            {
                updateStatusText();
                emitSettingsChanged();
            });
    connect(_ui->m_reuseExistingMatchesCheck, &QCheckBox::toggled, this, &AerialTriangulationDialog::updateStatusText);
    connect(_ui->m_guidedImageMatchingCheck, &QCheckBox::toggled, this, &AerialTriangulationDialog::updateStatusText);
}

void AerialTriangulationDialog::setAdvancedExpanded(bool expanded)
{
    _ui->m_advancedToggle->setChecked(expanded);
    _ui->m_advancedToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    _ui->m_advancedContent->setVisible(expanded);
    setMinimumHeight(expanded ? 620 : 360);
    if (layout())
    {
        layout()->invalidate();
    }
    adjustSize();
}

void AerialTriangulationDialog::setImageCount(int count)
{
    _imageCount = qMax(0, count);
    updateStatusText();
}

void AerialTriangulationDialog::setCachedTiePointLimit(bool cacheExists, int limit)
{
    _tiePointCacheExists = cacheExists;
    _cachedTiePointLimit = cacheExists ? limit : -1;
    updateStatusText();
}

void AerialTriangulationDialog::updateStatusText()
{
    if (!_ui || !_ui->m_statusLabel || !_ui->m_tiepointLimitSpin)
    {
        return;
    }

    const int requestedLimit = _ui->m_tiepointLimitSpin->value();
    const QString requested = QString::number(requestedLimit);
    QString cached;
    QString action;
    if (!_tiePointCacheExists)
    {
        cached = QStringLiteral("无");
        action = QStringLiteral("重新生成连接点网络");
    }
    else if (!_ui->m_reuseExistingMatchesCheck->isChecked())
    {
        cached = _cachedTiePointLimit >= 0 ? QString::number(_cachedTiePointLimit) : QStringLiteral("未知（旧格式）");
        action = QStringLiteral("清除旧匹配和连接点，按 %1 全量重建").arg(requested);
    }
    else if (_cachedTiePointLimit < 0)
    {
        cached = QStringLiteral("未知（旧格式）");
        action = QStringLiteral("使用现有连接点缓存；无法核对其历史上限");
    }
    else if (_cachedTiePointLimit != requestedLimit)
    {
        cached = QString::number(_cachedTiePointLimit);
        action = _ui->m_guidedImageMatchingCheck->isChecked()
                     ? QStringLiteral("复用兼容缓存、执行引导匹配并按 %1 重新生成连接点网络").arg(requested)
                     : QStringLiteral("复用已有特征/匹配，按 %1 重新生成连接点网络").arg(requested);
    }
    else if (_ui->m_guidedImageMatchingCheck->isChecked())
    {
        cached = QString::number(_cachedTiePointLimit);
        action = QStringLiteral("复用已有特征/匹配，并刷新连接点网络");
    }
    else
    {
        cached = QString::number(_cachedTiePointLimit);
        action = QStringLiteral("直接使用上限为 %1 的缓存连接点网络").arg(cached);
    }

    _ui->m_statusLabel->setText(QStringLiteral("当前项目影像：%1 张\n连接点上限：当前设置 %2；缓存实际 %3；执行：%4。")
                                    .arg(_imageCount)
                                    .arg(requested, cached, action));
}

void AerialTriangulationDialog::setReferencePreselectionAvailable(bool available, int cameraCount, int imageCount)
{
    if (!_ui || !_ui->m_referencePreselectionCheck || !_ui->m_referenceSourceCombo)
    {
        return;
    }

    // available 只表示相机文件是否完整；PlaMatch 的 Source/Estimated 在坐标
    // 集合为空时按参考实现回退索引邻域，Sequential 也无需相机文件。
    _ui->m_referencePreselectionCheck->setEnabled(true);
    _ui->m_referencePreselectionCheck->setToolTip(
        available ? QStringLiteral("使用已导入相机外方位/相机文件生成候选匹配对。")
                  : QStringLiteral("相机参考不完整；PlaMatch 将使用已有坐标，若没有坐标则按影像索引邻域回退。"));
    _ui->m_referenceSourceCombo->setEnabled(true);
    _ui->m_referenceSourceCombo->setToolTip(
        available ? QStringLiteral("选择参考预选的候选对来源。")
                  : QStringLiteral("相机参考不完整：当前相机 %1/%2。Source/Estimated 将按参考实现安全回退。")
                        .arg(qMax(0, cameraCount))
                        .arg(qMax(0, imageCount)));
}

void AerialTriangulationDialog::applySettings(const QJsonObject& settings)
{
    _applyingSettings = true;

    // 批量恢复时屏蔽各控件信号，避免上层收到尚未恢复完成的中间配置。
    const QSignalBlocker blockQuality(_ui->m_qualityCombo);
    const QSignalBlocker blockGeneric(_ui->m_genericPreselectionCheck);
    const QSignalBlocker blockReference(_ui->m_referencePreselectionCheck);
    const QSignalBlocker blockReferenceSource(_ui->m_referenceSourceCombo);
    const QSignalBlocker blockReset(_ui->m_resetAlignmentCheck);
    const QSignalBlocker blockSave(_ui->m_saveAfterEachStepCheck);
    const QSignalBlocker blockKeypoint(_ui->m_keypointLimitSpin);
    const QSignalBlocker blockTiepoint(_ui->m_tiepointLimitSpin);
    const QSignalBlocker blockMask(_ui->m_maskApplyCombo);
    const QSignalBlocker blockExclude(_ui->m_excludeFixedTiePointsCheck);
    const QSignalBlocker blockGuided(_ui->m_guidedImageMatchingCheck);
    const QSignalBlocker blockAdaptive(_ui->m_adaptiveCameraModelCheck);
    const QSignalBlocker blockReuseMatches(_ui->m_reuseExistingMatchesCheck);
    const QSignalBlocker blockLockPoses(_ui->m_lockInputCameraPosesCheck);

    setComboByData(_ui->m_qualityCombo, settings.value(QStringLiteral("quality")).toString(QStringLiteral("high")));
    _ui->m_genericPreselectionCheck->setChecked(settings.value(QStringLiteral("generic_preselection")).toBool(true));
    _ui->m_referencePreselectionCheck->setChecked(
        settings.value(QStringLiteral("reference_preselection")).toBool(false));
    setComboByData(
        _ui->m_referenceSourceCombo,
        settings.value(QStringLiteral("reference_preselection_source")).toString(QStringLiteral("source_code")));
    _ui->m_resetAlignmentCheck->setChecked(settings.value(QStringLiteral("reset_current_alignment")).toBool(true));
    _ui->m_saveAfterEachStepCheck->setChecked(
        settings.value(QStringLiteral("save_project_after_each_step")).toBool(false));
    _ui->m_keypointLimitSpin->setValue(settings.value(QStringLiteral("keypoint_limit")).toInt(40000));
    int tiePointLimit = settings.value(QStringLiteral("tiepoint_limit")).toInt(kReferenceTiePointLimit);
    const int defaultVersion = settings.value(QStringLiteral("tiepoint_limit_default_version")).toInt(1);
    if (settings.contains(QStringLiteral("tiepoint_limit")) && defaultVersion < kTiePointDefaultVersion &&
        tiePointLimit == 8000)
    {
        // 8000 曾是 PlaScan 的默认值，并非“对齐照片”的参考默认值。旧项目若没有
        // 新版本标记则一次性迁移到 4000；其它用户显式数值原样保留。
        tiePointLimit = kReferenceTiePointLimit;
    }
    _ui->m_tiepointLimitSpin->setValue(tiePointLimit);
    setComboByData(_ui->m_maskApplyCombo,
                   settings.value(QStringLiteral("mask_apply_mode")).toString(QStringLiteral("keypoints")));
    _ui->m_excludeFixedTiePointsCheck->setChecked(
        settings.value(QStringLiteral("exclude_fixed_tie_points")).toBool(true));
    _ui->m_guidedImageMatchingCheck->setChecked(settings.value(QStringLiteral("guided_image_matching")).toBool(false));
    _ui->m_adaptiveCameraModelCheck->setChecked(
        settings.value(QStringLiteral("adaptive_camera_model_fitting")).toBool(true));
    _ui->m_reuseExistingMatchesCheck->setChecked(settings.value(QStringLiteral("reuse_existing_matches")).toBool(true));
    _ui->m_lockInputCameraPosesCheck->setChecked(
        settings.value(QStringLiteral("lock_input_camera_poses")).toBool(false));

    // 输入配置同时启用两个互斥选项时，以“锁定输入位姿”为更强约束进行归一化。
    if (_ui->m_lockInputCameraPosesCheck->isChecked())
    {
        _ui->m_resetAlignmentCheck->setChecked(false);
    }

    _ui->m_referenceSourceCombo->setEnabled(true);
    _applyingSettings = false;
    updateStatusText();
}

QJsonObject AerialTriangulationDialog::collectSettings() const
{
    // 字段名属于项目持久化与工作流之间的接口，不能随界面对象名一起变化。
    QJsonObject settings;
    settings[QStringLiteral("workflow_kind")] = QStringLiteral("aerial_triangulation_dialog_only");
    settings[QStringLiteral("quality")] = comboDataOr(_ui->m_qualityCombo, QStringLiteral("high"));
    settings[QStringLiteral("generic_preselection")] = _ui->m_genericPreselectionCheck->isChecked();
    settings[QStringLiteral("reference_preselection")] = _ui->m_referencePreselectionCheck->isChecked();
    settings[QStringLiteral("reference_preselection_source")] =
        comboDataOr(_ui->m_referenceSourceCombo, QStringLiteral("source_code"));
    settings[QStringLiteral("reset_current_alignment")] =
        _ui->m_lockInputCameraPosesCheck->isChecked() ? false : _ui->m_resetAlignmentCheck->isChecked();
    settings[QStringLiteral("save_project_after_each_step")] = _ui->m_saveAfterEachStepCheck->isChecked();
    settings[QStringLiteral("keypoint_limit")] = _ui->m_keypointLimitSpin->value();
    settings[QStringLiteral("tiepoint_limit")] = _ui->m_tiepointLimitSpin->value();
    settings[QStringLiteral("tiepoint_limit_default_version")] = kTiePointDefaultVersion;
    settings[QStringLiteral("mask_apply_mode")] = comboDataOr(_ui->m_maskApplyCombo, QStringLiteral("keypoints"));
    settings[QStringLiteral("exclude_fixed_tie_points")] = _ui->m_excludeFixedTiePointsCheck->isChecked();
    settings[QStringLiteral("guided_image_matching")] = _ui->m_guidedImageMatchingCheck->isChecked();
    settings[QStringLiteral("adaptive_camera_model_fitting")] = _ui->m_adaptiveCameraModelCheck->isChecked();
    settings[QStringLiteral("reuse_existing_matches")] = _ui->m_reuseExistingMatchesCheck->isChecked();
    settings[QStringLiteral("lock_input_camera_poses")] = _ui->m_lockInputCameraPosesCheck->isChecked();
    return settings;
}

void AerialTriangulationDialog::emitSettingsChanged()
{
    // applySettings() 会一次修改多个控件，完成前不应暴露不一致的配置快照。
    if (_applyingSettings)
    {
        return;
    }
    emit settingsChanged(collectSettings());
}
