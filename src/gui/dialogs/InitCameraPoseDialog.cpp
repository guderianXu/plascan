#include "InitCameraPoseDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QStackedWidget>
#include <QLabel>
#include <QDialogButtonBox>
#include <QFileInfo>

InitCameraPoseDialog::InitCameraPoseDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("初始化相机位姿"));
    setMinimumSize(720, 560);

    auto *root = new QVBoxLayout(this);
    root->setSpacing(10);

    // ── 模式选择 ──
    {
        auto *box = new QGroupBox(tr("相机信息来源"));
        auto *fl = new QFormLayout(box);
        m_modeCombo = new QComboBox;
        m_modeCombo->addItems({
            tr("无相机文件 (从 EXIF 推断)"),
            tr("仅有内参矩阵"),
            tr("完整相机文件 (.tsai/.yaml/.xml)")
        });
        m_modeCombo->setToolTip(
            tr("选择当前可用的相机信息类型，系统将据此选择初始化策略"));
        fl->addRow(tr("模式:"), m_modeCombo);
        root->addWidget(box);
    }

    m_statusLabel = new QLabel(this);
    m_statusLabel->setWordWrap(true);
    m_statusLabel->setStyleSheet(QStringLiteral(
        "QLabel { background: #eef6ff; border: 1px solid #c9def5; border-radius: 6px; padding: 8px; }"));
    root->addWidget(m_statusLabel);

    // ── 应用范围（模式 0/1 使用） ──
    {
        m_applyBox = new QGroupBox(tr("写入范围"), this);
        auto *vl = new QVBoxLayout(m_applyBox);
        m_applyForm = new QFormLayout;

        m_applyScopeCombo = new QComboBox(this);
        m_applyScopeCombo->addItems({
            tr("回写全部参与影像"),
            tr("仅回写目标影像")
        });
        m_applyForm->addRow(tr("回写范围:"), m_applyScopeCombo);

        m_applyTargetImageCombo = new QComboBox(this);
        m_applyTargetImageCombo->setToolTip(tr("选择最终需要回写初始化位姿的目标影像。求解时仍会联合项目影像进行相对定向。"));
        m_applyForm->addRow(tr("目标影像:"), m_applyTargetImageCombo);

        vl->addLayout(m_applyForm);

        m_overwriteExistingCheck = new QCheckBox(tr("覆盖已有相机参数"), this);
        m_overwriteExistingCheck->setChecked(false);
        m_overwriteExistingCheck->setToolTip(tr("关闭时会跳过已经存在 camera 字段的影像，避免误覆盖已有导入结果。"));
        vl->addWidget(m_overwriteExistingCheck);

        m_applyHintLabel = new QLabel(this);
        m_applyHintLabel->setWordWrap(true);
        vl->addWidget(m_applyHintLabel);

        auto *solveForm = new QFormLayout;
        m_qualityCombo = new QComboBox(this);
        m_qualityCombo->addItem(tr("快速"), 0);
        m_qualityCombo->addItem(tr("标准"), 1);
        m_qualityCombo->addItem(tr("高质量"), 2);
        m_qualityCombo->addItem(tr("最高质量"), 3);
        m_qualityCombo->setCurrentIndex(1);
        m_qualityCombo->setToolTip(tr("影响特征匹配和增量 SFM 的过滤阈值与运行时间。"));
        solveForm->addRow(tr("求解质量:"), m_qualityCombo);

        m_threadsSpin = new QSpinBox(this);
        m_threadsSpin->setRange(1, 128);
        m_threadsSpin->setValue(8);
        m_threadsSpin->setToolTip(tr("后台相对定向 / 匹配 / SFM 线程数。"));
        solveForm->addRow(tr("线程数:"), m_threadsSpin);
        vl->addLayout(solveForm);

        root->addWidget(m_applyBox);
    }

    m_modeStack = new QStackedWidget;

    // ── 页面 0: 无相机文件 ──
    {
        auto *page = new QWidget;
        auto *vl = new QVBoxLayout(page);
        auto *intro = new QLabel(
            tr("适用于没有现成相机文件的情况。程序会优先尝试从影像元数据读取焦距；"
               "若读取不到，则按“默认焦距 + 传感器宽度”换算像素焦距，并写入项目。\n"
                    "随后会调用相对定向 / 增量 SFM 估计真实外参。"),
            page);
        intro->setWordWrap(true);
        vl->addWidget(intro);

        auto *fl = new QFormLayout;

        m_exifAutoCheck = new QCheckBox(tr("自动从 EXIF 提取焦距"));
        m_exifAutoCheck->setChecked(true);
        m_exifAutoCheck->setToolTip(
            tr("自动读取图像 EXIF 中的 FocalLength 和 FocalLengthIn35mmFilm 字段"));
        fl->addRow(m_exifAutoCheck);

        m_defaultFocalSpin = new QDoubleSpinBox;
        m_defaultFocalSpin->setRange(1.0, 10000.0);
        m_defaultFocalSpin->setDecimals(1);
        m_defaultFocalSpin->setValue(50.0);
        m_defaultFocalSpin->setSuffix(tr(" mm"));
        m_defaultFocalSpin->setToolTip(
            tr("当 EXIF 不可用时使用的默认焦距。★ 推荐根据实际镜头填写"));
        fl->addRow(tr("默认焦距:"), m_defaultFocalSpin);

        m_sensorWidthSpin = new QDoubleSpinBox;
        m_sensorWidthSpin->setRange(1.0, 100.0);
        m_sensorWidthSpin->setDecimals(2);
        m_sensorWidthSpin->setValue(23.5);
        m_sensorWidthSpin->setSuffix(tr(" mm"));
        m_sensorWidthSpin->setToolTip(
            tr("传感器宽度，用于将焦距从 mm 转换为像素。APS-C 约 23.5 mm，全幅约 36 mm"));
        fl->addRow(tr("传感器宽度:"), m_sensorWidthSpin);

        vl->addLayout(fl);
        vl->addStretch(1);

        m_modeStack->addWidget(page);
    }

    // ── 页面 1: 仅有内参 ──
    {
        auto *page = new QWidget;
        auto *vl = new QVBoxLayout(page);
        auto *intro = new QLabel(
            tr("适用于你已知内参而没有标准相机文件的情况。程序会将内参和默认外参"
                    "作为初值输入相对定向 / 增量 SFM，并回写恢复出的真实位姿。"),
            page);
        intro->setWordWrap(true);
        vl->addWidget(intro);

        m_intrinsicsForm = new QFormLayout;

        m_fxSpin = new QDoubleSpinBox;
        m_fxSpin->setRange(1.0, 100000.0);
        m_fxSpin->setDecimals(2);
        m_fxSpin->setValue(3000.0);
        m_fxSpin->setToolTip(tr("x 方向焦距 (像素)"));
        m_intrinsicsForm->addRow(tr("fx:"), m_fxSpin);

        m_fySpin = new QDoubleSpinBox;
        m_fySpin->setRange(1.0, 100000.0);
        m_fySpin->setDecimals(2);
        m_fySpin->setValue(3000.0);
        m_fySpin->setToolTip(tr("y 方向焦距 (像素)"));
        m_intrinsicsForm->addRow(tr("fy:"), m_fySpin);

        m_cxSpin = new QDoubleSpinBox;
        m_cxSpin->setRange(-1.0, 50000.0);
        m_cxSpin->setDecimals(2);
        m_cxSpin->setValue(-1.0);
        m_cxSpin->setSpecialValueText(tr("自动取图像中心"));
        m_cxSpin->setToolTip(tr("主点 x 坐标；设为自动时，将按每张影像宽度的一半计算。"));
        m_intrinsicsForm->addRow(tr("cx:"), m_cxSpin);

        m_cySpin = new QDoubleSpinBox;
        m_cySpin->setRange(-1.0, 50000.0);
        m_cySpin->setDecimals(2);
        m_cySpin->setValue(-1.0);
        m_cySpin->setSpecialValueText(tr("自动取图像中心"));
        m_cySpin->setToolTip(tr("主点 y 坐标；设为自动时，将按每张影像高度的一半计算。"));
        m_intrinsicsForm->addRow(tr("cy:"), m_cySpin);

        m_distModelCombo = new QComboBox;
        m_distModelCombo->addItems({
            tr("无畸变"),
            tr("径向 (k1, k2)"),
            tr("Brown (k1, k2, p1, p2)")
        });
        m_distModelCombo->setCurrentIndex(2);
        m_distModelCombo->setToolTip(tr("当前项目写回格式支持无畸变、径向和 Brown 模型。"));
        m_intrinsicsForm->addRow(tr("畸变模型:"), m_distModelCombo);

        m_k1Spin = new QDoubleSpinBox;
        m_k1Spin->setRange(-10.0, 10.0);
        m_k1Spin->setDecimals(6);
        m_k1Spin->setValue(0.0);
        m_intrinsicsForm->addRow(tr("k1:"), m_k1Spin);

        m_k2Spin = new QDoubleSpinBox;
        m_k2Spin->setRange(-10.0, 10.0);
        m_k2Spin->setDecimals(6);
        m_k2Spin->setValue(0.0);
        m_intrinsicsForm->addRow(tr("k2:"), m_k2Spin);

        m_p1Spin = new QDoubleSpinBox;
        m_p1Spin->setRange(-10.0, 10.0);
        m_p1Spin->setDecimals(6);
        m_p1Spin->setValue(0.0);
        m_intrinsicsForm->addRow(tr("p1:"), m_p1Spin);

        m_p2Spin = new QDoubleSpinBox;
        m_p2Spin->setRange(-10.0, 10.0);
        m_p2Spin->setDecimals(6);
        m_p2Spin->setValue(0.0);
        m_intrinsicsForm->addRow(tr("p2:"), m_p2Spin);

        vl->addLayout(m_intrinsicsForm);
        vl->addStretch(1);

        m_modeStack->addWidget(page);
    }

    // ── 页面 2: 完整相机文件 ──
    {
        auto *page = new QWidget;
        auto *vl = new QVBoxLayout(page);

        auto *intro = new QLabel(
            tr("复用现有相机导入流程。可为单张影像精确选择相机文件，"
               "也可直接选择目录按文件名自动匹配。"));
        intro->setWordWrap(true);
        vl->addWidget(intro);

        m_cameraImportForm = new QFormLayout;

        m_cameraImportModeCombo = new QComboBox;
        m_cameraImportModeCombo->addItems({
            tr("为单张影像选择对应相机文件"),
            tr("选择目录并按文件名自动匹配")
        });
        m_cameraImportModeCombo->setToolTip(
            tr("单张模式适合精确指定；目录模式适合已有一批同名相机文件时自动批量导入。"));
        m_cameraImportForm->addRow(tr("导入方式:"), m_cameraImportModeCombo);

        m_targetImageCombo = new QComboBox;
        m_targetImageCombo->setToolTip(tr("单张导入时，选择需要绑定相机文件的影像。"));
        m_cameraImportForm->addRow(tr("目标影像:"), m_targetImageCombo);

        m_cameraFormatCombo = new QComboBox;
        m_cameraFormatCombo->addItems({tr("Tsai (.tsai)"), tr("自动检测")});
        m_cameraFormatCombo->setCurrentIndex(1);
        m_cameraFormatCombo->setToolTip(tr("当前导入流程复用已有相机文件导入实现。"));
        m_cameraImportForm->addRow(tr("文件格式:"), m_cameraFormatCombo);

        vl->addLayout(m_cameraImportForm);

        m_cameraImportHintLabel = new QLabel;
        m_cameraImportHintLabel->setWordWrap(true);
        vl->addWidget(m_cameraImportHintLabel);
        vl->addStretch(1);

        m_modeStack->addWidget(page);
    }

    root->addWidget(m_modeStack);

    auto *btnBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("写入相机初值"));
    root->addWidget(btnBox);

    // 连接
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onModeChanged);
    connect(m_applyScopeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onInitTargetModeChanged);
    connect(m_cameraImportModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onCameraImportModeChanged);
    connect(m_distModelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onDistortionModelChanged);

    auto changed = [this]()
    {
        emitSettingsNow();
    };
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_applyScopeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_applyTargetImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_overwriteExistingCheck, &QCheckBox::toggled, this, changed);
    connect(m_qualityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_cameraImportModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_targetImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_cameraFormatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_exifAutoCheck, &QCheckBox::toggled, this, changed);
    connect(m_defaultFocalSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_sensorWidthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_fxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_fySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_cxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_cySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_distModelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_k1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_k2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_p1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_p2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);

    connect(btnBox, &QDialogButtonBox::accepted, this, &InitCameraPoseDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    onModeChanged(0);
    onInitTargetModeChanged(0);
    onCameraImportModeChanged(0);
    onDistortionModelChanged(m_distModelCombo->currentIndex());
}

void InitCameraPoseDialog::onModeChanged(int index)
{
    m_modeStack->setCurrentIndex(index);
    if (m_applyBox)
    {
        m_applyBox->setVisible(index != 2);
    }
    updateStatusText();
    updateTargetUi();
}

void InitCameraPoseDialog::onInitTargetModeChanged(int index)
{
    Q_UNUSED(index);
    updateTargetUi();
}

void InitCameraPoseDialog::onDistortionModelChanged(int index)
{
    const bool showK = (index >= 1);
    const bool showP = (index >= 2);

    auto toggleField = [this](QWidget *field, bool visible)
    {
        if (!field || !m_intrinsicsForm)
        {
            return;
        }
        field->setVisible(visible);
        if (QWidget *label = m_intrinsicsForm->labelForField(field))
        {
            label->setVisible(visible);
        }
    };

    toggleField(m_k1Spin, showK);
    toggleField(m_k2Spin, showK);
    toggleField(m_p1Spin, showP);
    toggleField(m_p2Spin, showP);
}

void InitCameraPoseDialog::onCameraImportModeChanged(int index)
{
    const bool exactImport = (index == 0);
    if (m_targetImageCombo)
    {
        m_targetImageCombo->setVisible(exactImport);
    }
    if (m_cameraImportForm && m_targetImageCombo)
    {
        if (QWidget *label = m_cameraImportForm->labelForField(m_targetImageCombo))
        {
            label->setVisible(exactImport);
        }
    }

    if (m_cameraImportHintLabel)
    {
        if (exactImport)
        {
            m_cameraImportHintLabel->setText(
                tr("点击[初始化位姿]后，将为所选影像打开现有的单文件导入流程。"
                   "适合逐张检查和精确绑定相机文件。"));
        }
        else
        {
            m_cameraImportHintLabel->setText(
                tr("点击[初始化位姿]后，将打开现有的目录批量导入流程。"
                   "程序会按文件名自动匹配相机文件与项目影像。"));
        }
    }

    updateStatusText();
}

void InitCameraPoseDialog::setAvailableImages(const QStringList &imagePaths)
{
    auto refillCombo = [&imagePaths](QComboBox *combo)
    {
        if (!combo)
        {
            return;
        }
        const QString currentPath = combo->currentData().toString();
        combo->blockSignals(true);
        combo->clear();

        for (const QString &path : imagePaths)
        {
            const QFileInfo fi(path);
            const QString label = fi.fileName().isEmpty() ? path : fi.fileName();
            combo->addItem(label, path);
        }

        if (!currentPath.isEmpty())
        {
            const int idx = combo->findData(currentPath);
            if (idx >= 0)
            {
                combo->setCurrentIndex(idx);
            }
        }

        combo->setEnabled(combo->count() > 0);
        if (combo->count() == 0)
        {
            combo->addItem(tr("当前项目无可选影像"), QString());
            combo->setEnabled(false);
        }
        combo->blockSignals(false);
    };

    refillCombo(m_applyTargetImageCombo);
    refillCombo(m_targetImageCombo);
    updateTargetUi();
}

QJsonObject InitCameraPoseDialog::collectSettings() const
{
    QJsonObject o;
    o["mode"] = m_modeCombo->currentIndex();
    o["applyScope"] = m_applyScopeCombo->currentIndex();
    o["applyTargetImagePath"] = m_applyTargetImageCombo->currentData().toString();
    o["applyTargetImageName"] = m_applyTargetImageCombo->currentText();
    o["overwriteExisting"] = m_overwriteExistingCheck->isChecked();
    o["quality"] = m_qualityCombo->currentData().toInt();
    o["threads"] = m_threadsSpin->value();
    // 模式 0
    o["exifAuto"]     = m_exifAutoCheck->isChecked();
    o["defaultFocal"]  = m_defaultFocalSpin->value();
    o["sensorWidth"]   = m_sensorWidthSpin->value();
    // 模式 1
    o["fx"] = m_fxSpin->value();
    o["fy"] = m_fySpin->value();
    o["cx"] = m_cxSpin->value();
    o["cy"] = m_cySpin->value();
    o["distortionModel"] = m_distModelCombo->currentText();
    o["k1"] = m_k1Spin->value();
    o["k2"] = m_k2Spin->value();
    o["p1"] = m_p1Spin->value();
    o["p2"] = m_p2Spin->value();
    // 模式 2
    o["cameraImportMode"] = m_cameraImportModeCombo->currentIndex();
    o["targetImagePath"]  = m_targetImageCombo->currentData().toString();
    o["targetImageName"]  = m_targetImageCombo->currentText();
    o["cameraFormat"] = m_cameraFormatCombo->currentText();
    return o;
}

void InitCameraPoseDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("mode"))
    {
        m_modeCombo->setCurrentIndex(s["mode"].toInt());
    }
    if (s.contains("applyScope"))
    {
        m_applyScopeCombo->setCurrentIndex(s["applyScope"].toInt());
    }
    if (s.contains("applyTargetImagePath"))
    {
        int i = m_applyTargetImageCombo->findData(s["applyTargetImagePath"].toString());
        if (i >= 0)
        {
            m_applyTargetImageCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("overwriteExisting"))
    {
        m_overwriteExistingCheck->setChecked(s["overwriteExisting"].toBool());
    }
    if (s.contains("quality"))
    {
        const int idx = m_qualityCombo->findData(s["quality"].toInt());
        if (idx >= 0)
        {
            m_qualityCombo->setCurrentIndex(idx);
        }
    }
    if (s.contains("threads")) m_threadsSpin->setValue(s["threads"].toInt());
    if (s.contains("exifAuto"))      m_exifAutoCheck->setChecked(s["exifAuto"].toBool());
    if (s.contains("defaultFocal"))  m_defaultFocalSpin->setValue(s["defaultFocal"].toDouble());
    if (s.contains("sensorWidth"))   m_sensorWidthSpin->setValue(s["sensorWidth"].toDouble());
    if (s.contains("fx")) m_fxSpin->setValue(s["fx"].toDouble());
    if (s.contains("fy")) m_fySpin->setValue(s["fy"].toDouble());
    if (s.contains("cx")) m_cxSpin->setValue(s["cx"].toDouble());
    if (s.contains("cy")) m_cySpin->setValue(s["cy"].toDouble());
    if (s.contains("distortionModel"))
    {
        int i = m_distModelCombo->findText(s["distortionModel"].toString());
        if (i >= 0) m_distModelCombo->setCurrentIndex(i);
    }
    if (s.contains("k1")) m_k1Spin->setValue(s["k1"].toDouble());
    if (s.contains("k2")) m_k2Spin->setValue(s["k2"].toDouble());
    if (s.contains("p1")) m_p1Spin->setValue(s["p1"].toDouble());
    if (s.contains("p2")) m_p2Spin->setValue(s["p2"].toDouble());
    if (s.contains("cameraImportMode"))
        m_cameraImportModeCombo->setCurrentIndex(s["cameraImportMode"].toInt());
    if (s.contains("targetImagePath"))
    {
        int i = m_targetImageCombo->findData(s["targetImagePath"].toString());
        if (i >= 0)
        {
            m_targetImageCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("cameraFormat"))
    {
        int i = m_cameraFormatCombo->findText(s["cameraFormat"].toString());
        if (i >= 0) m_cameraFormatCombo->setCurrentIndex(i);
    }
    updateTargetUi();
    updateStatusText();
}

void InitCameraPoseDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void InitCameraPoseDialog::updateStatusText()
{
    const int mode = m_modeCombo ? m_modeCombo->currentIndex() : 0;
    QString text;
    if (mode == 0)
    {
        text = tr("当前将先为缺少相机文件的影像准备内参初值（EXIF 优先，默认焦距回退），"
              "再运行相对定向 / 增量 SFM 恢复真实位姿，并将结果回写到项目。需要至少 2 张存在匹配关系的影像。");
    }
    else if (mode == 1)
    {
        text = tr("当前将使用手工内参作为求解初值，运行相对定向 / 增量 SFM 恢复外参。"
                  "这适合已有标定结果但暂无标准相机文件的情况；cx/cy 设为自动时将使用图像中心。 ");
    }
    else
    {
        const bool exactImport = (m_cameraImportModeCombo && m_cameraImportModeCombo->currentIndex() == 0);
        text = exactImport
            ? tr("当前将复用现有“单影像 → 单相机文件”导入流程；成功后写入该影像的相机参数。")
            : tr("当前将复用现有“目录批量导入”流程；程序会按文件名自动匹配相机文件与项目影像。");
    }
    if (m_statusLabel)
    {
        m_statusLabel->setText(text);
    }
}

void InitCameraPoseDialog::updateTargetUi()
{
    const bool singleImage = (m_applyScopeCombo && m_applyScopeCombo->currentIndex() == 1);
    if (m_applyTargetImageCombo)
    {
        m_applyTargetImageCombo->setVisible(singleImage);
    }
    if (m_applyForm && m_applyTargetImageCombo)
    {
        if (QWidget *label = m_applyForm->labelForField(m_applyTargetImageCombo))
        {
            label->setVisible(singleImage);
        }
    }
    if (m_applyHintLabel)
    {
        m_applyHintLabel->setText(singleImage
            ? tr("求解时仍会联合当前项目中可用影像进行相对定向，但最终只回写这张目标影像的相机位姿。")
            : tr("会对当前项目中的全部参与影像运行相对定向 / 增量 SFM，并将成功恢复的相机位姿批量回写；未勾选覆盖时会跳过已有相机参数。"));
    }
}

void InitCameraPoseDialog::onRun()
{
    emit runRequested(collectSettings());
}
