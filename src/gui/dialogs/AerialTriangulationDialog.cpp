#include "AerialTriangulationDialog.h"
#include "ui_AerialTriangulationDialog.h"

// Qt核心控件头文件
#include <QVBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QListWidget>
#include <QComboBox>
#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QDialogButtonBox>
#include <QJsonArray>
#include <QThread>
#include <QToolButton>
#include <QFrame>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QFileInfo>
#include <QSizePolicy>
#include <QListWidgetItem>
#include <QPushButton>
#include <QMenu>
#include <QDesktopServices>
#include <QUrl>
#include <QPoint>

/**
 * @brief 构造函数：初始化界面和控件
 * @param parent 父窗口指针
 */
AerialTriangulationDialog::AerialTriangulationDialog(QWidget *parent)
    : QDialog(parent)
{
    // 初始化界面布局
    initUI();
}

/**
 * @brief 初始化界面布局（核心布局逻辑，严格保留原始控件）
 */
void AerialTriangulationDialog::initUI()
{
    Ui::AerialTriangulationDialog form;
    form.setupUi(this);

    m_imageCountLabel = form.m_imageCountLabel;
    m_selectAllBtn = form.m_selectAllBtn;
    m_deselectAllBtn = form.m_deselectAllBtn;
    m_imageList = form.m_imageList;
    m_qualityCombo = form.m_qualityCombo;
    m_cameraOptCombo = form.m_cameraOptCombo;
    m_stepCompletionCheck = form.m_stepCompletionCheck;
    m_outputDirEdit = form.m_outputDirEdit;
    m_browseBtn = form.m_browseBtn;
    m_intrinsicsCheck = form.m_intrinsicsCheck;
    m_fuSpin = form.m_fuSpin;
    m_fvSpin = form.m_fvSpin;
    m_cuSpin = form.m_cuSpin;
    m_cvSpin = form.m_cvSpin;
    m_pitchSpin = form.m_pitchSpin;
    m_resetCurrentAlignCheck = form.m_resetCurrentAlignCheck;
    m_threadsCombo = form.m_threadsCombo;
    m_baRefinePoseCheck = form.m_baRefinePoseCheck;
    m_baMaxIterSpin = form.m_baMaxIterSpin;
    m_baHuberDeltaSpin = form.m_baHuberDeltaSpin;
    m_baFilterReprojSpin = form.m_baFilterReprojSpin;
    m_baDampingSpin = form.m_baDampingSpin;
    m_btnBox = form.m_btnBox;

    connect(m_imageList, &QListWidget::customContextMenuRequested, this, &AerialTriangulationDialog::onImageListContextMenu);

    // 连接全选/取消按钮
    connect(m_selectAllBtn, &QPushButton::clicked, this, &AerialTriangulationDialog::onSelectAll);
    connect(m_deselectAllBtn, &QPushButton::clicked, this, &AerialTriangulationDialog::onDeselectAll);

    connect(m_browseBtn, &QPushButton::clicked, this, &AerialTriangulationDialog::onBrowseOutputDir);

    connect(m_intrinsicsCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_fuSpin->setEnabled(on);
        m_fvSpin->setEnabled(on);
        m_cuSpin->setEnabled(on);
        m_cvSpin->setEnabled(on);
        m_pitchSpin->setEnabled(on);
    });

    form.advancedToggleButton->setStyleSheet(R"(
        QToolButton {
            border: none;
            padding: 5px;
            font-size: 14px;
        }
        QToolButton:checked {
            color: #1890FF;
        }
    )");

    // 线程数下拉框（严格保留手动选择，你的原始逻辑：1-系统最大核心数）
    int maxTh = QThread::idealThreadCount();
    if (maxTh <= 0) maxTh = 1;
    for (int i = 1; i <= maxTh; ++i) m_threadsCombo->addItem(QString::number(i));
    m_threadsCombo->setCurrentIndex(maxTh - 1); // 默认选中最大线程数

    form.advancedLine->setStyleSheet(R"(
        QFrame {
            margin: 10px 0;
            color: #E0E0E0;
        }
    )");

    // 绑定折叠按钮事件
    connect(form.advancedToggleButton, &QToolButton::toggled, this,
        [toggle = form.advancedToggleButton, content = form.advancedContentWidget](bool checked)
    {
        toggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        content->setVisible(checked);
    });

    // 默认禁用OK按钮（无影像时）
    m_btnBox->button(QDialogButtonBox::Ok)->setEnabled(false);

    // ========== 信号与槽绑定（保留所有原始绑定，新增防抖） ==========
    // 按钮点击事件（保留原始逻辑）
    connect(m_btnBox, &QDialogButtonBox::accepted, this, &AerialTriangulationDialog::onRun);
    connect(m_btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    // 影像列表复选状态变化（新增：更新数量提示+按钮状态）
    connect(m_imageList, &QListWidget::itemChanged, this, &AerialTriangulationDialog::onImageCheckStateChanged);

    // 输出目录变化时也需要刷新OK按钮状态
    connect(m_outputDirEdit, &QLineEdit::textChanged, this, &AerialTriangulationDialog::onImageCheckStateChanged);

    // 参数变化事件（防抖处理，解决信号重复发送问题）
    auto paramChangedSlot = [this]() { this->onAnyChanged(); };
    connect(m_qualityCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, paramChangedSlot);
    connect(m_cameraOptCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, paramChangedSlot);
    connect(m_stepCompletionCheck, &QCheckBox::toggled, this, paramChangedSlot);
    connect(m_resetCurrentAlignCheck, &QCheckBox::toggled, this, paramChangedSlot);
    connect(m_threadsCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, paramChangedSlot);
    connect(m_outputDirEdit, &QLineEdit::textChanged, this, paramChangedSlot);
    connect(m_intrinsicsCheck, &QCheckBox::toggled, this, paramChangedSlot);
    connect(m_fuSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, paramChangedSlot);
    connect(m_fvSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, paramChangedSlot);
    connect(m_cuSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, paramChangedSlot);
    connect(m_cvSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, paramChangedSlot);
    // BA 参数变化
    connect(m_baRefinePoseCheck, &QCheckBox::toggled, this, paramChangedSlot);
    connect(m_baMaxIterSpin, qOverload<int>(&QSpinBox::valueChanged), this, paramChangedSlot);
    connect(m_baHuberDeltaSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, paramChangedSlot);
    connect(m_baFilterReprojSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, paramChangedSlot);
    connect(m_baDampingSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, paramChangedSlot);
}

/**
 * @brief 设置可用影像列表（保留复选框模式，你的原始逻辑）
 * @param images 影像文件路径列表
 */
void AerialTriangulationDialog::setAvailableImages(const QStringList &images)
{
    m_imageList->clear();
    // 严格保留复选框模式，你的原始逻辑
    for (const QString &path : images) {
        QFileInfo fi(path);
        auto *item = new QListWidgetItem(fi.fileName(), m_imageList);
        item->setData(Qt::UserRole, path);
        item->setCheckState(Qt::Checked); // 默认勾选
    }
    // 初始更新数量提示
    onImageCheckStateChanged();
}

/**
 * @brief 设置默认输出目录（保留原始逻辑，新增默认值兜底）
 * @param outputDir 默认输出目录路径
 */
void AerialTriangulationDialog::setDefaultOutputDir(const QString &outputDir)
{
    // 保留你的原始逻辑：仅输入框为空时设置
    if (!outputDir.isEmpty() && m_outputDirEdit->text().trimmed().isEmpty()) {
        m_outputDirEdit->setText(outputDir);
    }
    // 兜底：如果仍为空，设置默认路径（避免空值）
    if (m_outputDirEdit->text().trimmed().isEmpty()) {
        m_outputDirEdit->setText(QDir::currentPath() + "/AT_Output");
    }
}

/**
 * @brief 应用保存的配置参数（严格保留原始逻辑）
 * @param settings 配置参数JSON对象
 */
void AerialTriangulationDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty()) return;

    // 保留所有原始参数恢复逻辑
    m_qualityCombo->setCurrentIndex(settings.value(QStringLiteral("quality")).toInt(m_qualityCombo->currentIndex()));
    m_cameraOptCombo->setCurrentIndex(settings.value(QStringLiteral("camera_optimization_mode")).toInt(m_cameraOptCombo->currentIndex()));
    m_resetCurrentAlignCheck->setChecked(settings.value(QStringLiteral("reset_current_align")).toBool(m_resetCurrentAlignCheck->isChecked()));
    m_stepCompletionCheck->setChecked(settings.value(QStringLiteral("save_after_step")).toBool(m_stepCompletionCheck->isChecked()));
    // 线程数恢复（保留手动选择逻辑）
    if (settings.contains(QStringLiteral("threads"))) {
        int t = settings.value(QStringLiteral("threads")).toInt(m_threadsCombo->currentText().toInt());
        if (t > 0 && t <= m_threadsCombo->count()) m_threadsCombo->setCurrentIndex(t - 1);
    }
    m_outputDirEdit->setText(settings.value(QStringLiteral("output_dir")).toString(m_outputDirEdit->text()));

    // 内方位元素
    if (settings.contains(QStringLiteral("intrinsics"))) {
        const QJsonObject intr = settings.value(QStringLiteral("intrinsics")).toObject();
        m_intrinsicsCheck->setChecked(true);
        m_fuSpin->setValue(intr.value(QStringLiteral("fu")).toDouble());
        m_fvSpin->setValue(intr.value(QStringLiteral("fv")).toDouble());
        m_cuSpin->setValue(intr.value(QStringLiteral("cu")).toDouble());
        m_cvSpin->setValue(intr.value(QStringLiteral("cv")).toDouble());
        m_pitchSpin->setValue(intr.value(QStringLiteral("pitch")).toDouble());
    }

    // 光束法平差参数恢复
    if (m_baRefinePoseCheck)
        m_baRefinePoseCheck->setChecked(settings.value(QStringLiteral("ba_refine_pose")).toBool(true));
    if (m_baMaxIterSpin)
        m_baMaxIterSpin->setValue(settings.value(QStringLiteral("ba_max_iterations")).toInt(20));
    if (m_baHuberDeltaSpin)
        m_baHuberDeltaSpin->setValue(settings.value(QStringLiteral("ba_huber_delta")).toDouble(3.0));
    if (m_baFilterReprojSpin)
        m_baFilterReprojSpin->setValue(settings.value(QStringLiteral("ba_filter_reproj")).toDouble(8.0));
    if (m_baDampingSpin)
        m_baDampingSpin->setValue(settings.value(QStringLiteral("ba_damping")).toDouble(1e-6));
}

/**
 * @brief 收集当前所有配置参数（严格保留原始逻辑）
 * @return 包含所有配置的JSON对象
 */
QJsonObject AerialTriangulationDialog::collectSettings() const
{
    QJsonObject s;
    // 收集复选框选中的影像（使用完整路径，便于后续处理）
    QJsonArray imageArr;
    for (int i = 0; i < m_imageList->count(); ++i) {
        QListWidgetItem *it = m_imageList->item(i);
        if (it && it->checkState() == Qt::Checked)
            imageArr.append(it->data(Qt::UserRole).toString());
    }
    s[QStringLiteral("images")] = imageArr;

    // 保留所有原始参数收集逻辑
    s[QStringLiteral("quality")] = m_qualityCombo->currentIndex();
    s[QStringLiteral("camera_optimization_mode")] = m_cameraOptCombo->currentIndex();
    s[QStringLiteral("reset_current_align")] = m_resetCurrentAlignCheck->isChecked();
    s[QStringLiteral("save_after_step")] = m_stepCompletionCheck->isChecked();
    s[QStringLiteral("threads")] = m_threadsCombo->currentText().toInt(); // 保留手动线程数
    s[QStringLiteral("output_dir")] = m_outputDirEdit->text().trimmed();

    // 内方位元素
    if (m_intrinsicsCheck && m_intrinsicsCheck->isChecked()) {
        QJsonObject intr;
        intr[QStringLiteral("fu")] = m_fuSpin->value();
        intr[QStringLiteral("fv")] = m_fvSpin->value();
        intr[QStringLiteral("cu")] = m_cuSpin->value();
        intr[QStringLiteral("cv")] = m_cvSpin->value();
        intr[QStringLiteral("pitch")] = m_pitchSpin->value();
        s[QStringLiteral("intrinsics")] = intr;
    }

    // 光束法平差参数
    s[QStringLiteral("ba_refine_pose")]      = m_baRefinePoseCheck  ? m_baRefinePoseCheck->isChecked()  : true;
    s[QStringLiteral("ba_max_iterations")]   = m_baMaxIterSpin      ? m_baMaxIterSpin->value()           : 20;
    s[QStringLiteral("ba_huber_delta")]      = m_baHuberDeltaSpin   ? m_baHuberDeltaSpin->value()        : 3.0;
    s[QStringLiteral("ba_filter_reproj")]    = m_baFilterReprojSpin ? m_baFilterReprojSpin->value()      : 8.0;
    s[QStringLiteral("ba_damping")]          = m_baDampingSpin      ? m_baDampingSpin->value()           : 1e-6;

    return s;
}

/**
 * @brief 任意参数变化时的统一处理函数（防抖处理，解决信号重复发送）
 * @details 仅当配置真变化时才发送settingsChanged信号
 */
void AerialTriangulationDialog::onAnyChanged()
{
    QJsonObject currentSettings = collectSettings();
    // 对比上一次的配置，仅不同时发送信号（解决重复发送问题）
    if (currentSettings != m_lastSettings) {
        m_lastSettings = currentSettings;
        emit settingsChanged(currentSettings);
    }
}

/**
 * @brief 浏览输出目录按钮点击事件（新增功能，不影响其他逻辑）
 * @details 弹出系统目录选择对话框，更新输出目录输入框
 */
void AerialTriangulationDialog::onBrowseOutputDir()
{
    // 弹出目录选择框，初始路径为当前输入框值
    QString selectedDir = QFileDialog::getExistingDirectory(
        this,
        tr("选择输出目录"),
        m_outputDirEdit->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );

    // 仅当用户选择了有效目录时更新
    if (!selectedDir.isEmpty()) {
        m_outputDirEdit->setText(selectedDir);
    }
}

/**
 * @brief 影像列表复选状态变化事件（新增功能）
 * @details 1. 更新已选影像数量提示 2. 更新OK按钮启用状态
 */
void AerialTriangulationDialog::onImageCheckStateChanged()
{
    // 1. 统计复选框选中的影像数量
    int checkedCount = 0;
    int totalCount = m_imageList->count();
    for (int i = 0; i < totalCount; ++i) {
        QListWidgetItem *item = m_imageList->item(i);
        if (item && item->checkState() == Qt::Checked) {
            checkedCount++;
        }
    }
    // 更新数量提示
    m_imageCountLabel->setText(tr("已选影像：%1/%2张").arg(checkedCount).arg(totalCount));

    // 2. 更新OK按钮状态：已选影像>0 且 输出目录非空
    bool isOkEnabled = (checkedCount > 0) && (!m_outputDirEdit->text().trimmed().isEmpty());
    m_btnBox->button(QDialogButtonBox::Ok)->setEnabled(isOkEnabled);
}

void AerialTriangulationDialog::onSelectAll()
{
    for (int i = 0; i < m_imageList->count(); ++i) {
        QListWidgetItem *it = m_imageList->item(i);
        if (it) it->setCheckState(Qt::Checked);
    }
    onImageCheckStateChanged();
}

void AerialTriangulationDialog::onDeselectAll()
{
    for (int i = 0; i < m_imageList->count(); ++i) {
        QListWidgetItem *it = m_imageList->item(i);
        if (it) it->setCheckState(Qt::Unchecked);
    }
    onImageCheckStateChanged();
}

void AerialTriangulationDialog::onImageListContextMenu(const QPoint &pos)
{
    QListWidgetItem *item = m_imageList->itemAt(pos);
    QMenu menu(this);

    QAction *selAll = menu.addAction(tr("全选"));
    QAction *deselAll = menu.addAction(tr("取消"));
    menu.addSeparator();

    QAction *openFolder = menu.addAction(tr("在文件管理器中显示"));
    QAction *openExtern = menu.addAction(tr("使用系统查看器打开"));

    openFolder->setEnabled(item != nullptr);
    openExtern->setEnabled(item != nullptr);

    QAction *a = menu.exec(m_imageList->mapToGlobal(pos));
    if (!a) return;

    if (a == selAll) {
        onSelectAll();
    } else if (a == deselAll) {
        onDeselectAll();
    } else if (a == openFolder && item) {
        QString path = item->data(Qt::UserRole).toString();
        QFileInfo fi(path);
        QString dir = fi.absolutePath();
        if (!dir.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
    } else if (a == openExtern && item) {
        QString path = item->data(Qt::UserRole).toString();
        if (!path.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    }
}

void AerialTriangulationDialog::onOpenContainingFolder()
{
    auto items = m_imageList->selectedItems();
    if (items.isEmpty()) return;
    QString path = items.first()->data(Qt::UserRole).toString();
    QFileInfo fi(path);
    QString dir = fi.absolutePath();
    if (!dir.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
}

void AerialTriangulationDialog::onOpenImageExternally()
{
    auto items = m_imageList->selectedItems();
    if (items.isEmpty()) return;
    QString path = items.first()->data(Qt::UserRole).toString();
    if (!path.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

/**
 * @brief 点击OK按钮触发的处理函数（保留原始逻辑，新增参数校验）
 * @details 新增：影像为空/输出目录创建失败时提示，避免程序异常
 */
void AerialTriangulationDialog::onRun()
{
    QJsonObject s = collectSettings();

    // 新增：校验影像数量（避免空影像）
    if (s[QStringLiteral("images")].toArray().isEmpty()) {
        QMessageBox::warning(this, tr("警告"), tr("请至少选择一张影像！"));
        return;
    }

    // 新增：校验输出目录（避免创建失败）
    QString outputDir = s[QStringLiteral("output_dir")].toString();
    if (!outputDir.isEmpty()) 
    {
        QDir dir(outputDir);
        if (!dir.exists() && !dir.mkpath(".")) 
        {
            QMessageBox::warning(this, tr("警告"), tr("输出目录创建失败：%1").arg(outputDir));
            return;
        }
    }

    // 保留原始逻辑：触发信号+关闭对话框
    emit settingsChanged(s); // 保留你的原始触发（防抖已处理重复问题）
    emit runRequested(s);
    accept();
}
