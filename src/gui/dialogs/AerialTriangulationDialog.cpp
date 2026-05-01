#include "AerialTriangulationDialog.h"

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
    // 设置窗口基本属性：标题+原始尺寸（640x480），保留你的原始设置
    setWindowTitle(tr("空中三角测量"));
    resize(640, 480); 

    // ========== 整体布局 ==========
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10); // 优化边距，保留原始视觉
    mainLayout->setSpacing(10);                     // 控件间距优化

    // 内容容器（左右布局）
    auto *contentWidget = new QWidget(this);
    auto *contentHLayout = new QHBoxLayout(contentWidget);
    contentHLayout->setSpacing(10);

    // ========== 左侧：影像列表区域（保留复选框模式） ==========
    auto *leftGroupBox = new QGroupBox(tr("影像"), contentWidget);
    auto *leftVLayout = new QVBoxLayout(leftGroupBox);
    leftVLayout->setContentsMargins(5, 5, 5, 5);

    // 已选影像数量提示标签（新增核心功能）
    m_imageCountLabel = new QLabel(tr("已选影像：0张"), leftGroupBox);
    leftVLayout->addWidget(m_imageCountLabel);

    // 全选 / 取消 按钮
    auto *selectBtnWidget = new QWidget(leftGroupBox);
    auto *selectBtnLayout = new QHBoxLayout(selectBtnWidget);
    selectBtnLayout->setContentsMargins(0, 0, 0, 0);
    selectBtnLayout->setSpacing(6);
    m_selectAllBtn = new QPushButton(tr("全选"), selectBtnWidget);
    m_deselectAllBtn = new QPushButton(tr("取消"), selectBtnWidget);
    selectBtnLayout->addWidget(m_selectAllBtn);
    selectBtnLayout->addWidget(m_deselectAllBtn);
    leftVLayout->addWidget(selectBtnWidget);

    // 影像列表控件（保留复选框模式，你的原始逻辑）
    m_imageList = new QListWidget(leftGroupBox);
    m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection); 
    leftVLayout->addWidget(m_imageList);

    // 右键菜单支持
    m_imageList->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_imageList, &QListWidget::customContextMenuRequested, this, &AerialTriangulationDialog::onImageListContextMenu);

    // 连接全选/取消按钮
    connect(m_selectAllBtn, &QPushButton::clicked, this, &AerialTriangulationDialog::onSelectAll);
    connect(m_deselectAllBtn, &QPushButton::clicked, this, &AerialTriangulationDialog::onDeselectAll);

    // 设置左侧布局权重（保留你的原始比例）
    contentHLayout->addWidget(leftGroupBox, 1);

    // ========== 右侧：参数设置区域（保留所有原始控件） ==========
    auto *rightWidget = new QWidget(contentWidget);
    auto *rightVLayout = new QVBoxLayout(rightWidget);
    rightVLayout->setContentsMargins(0, 0, 0, 0);
    rightVLayout->setSpacing(10);

    // ---------- 基础设置组（保留所有原始控件） ----------
    auto *generalGroupBox = new QGroupBox(tr("一般"), rightWidget);
    auto *generalGridLayout = new QGridLayout(generalGroupBox);
    generalGridLayout->setContentsMargins(5, 5, 5, 5);
    generalGridLayout->setSpacing(5);

    // 重建精度下拉框（保留原始4级：低/中/高/最高）
    m_qualityCombo = new QComboBox(generalGroupBox);
    m_qualityCombo->addItems({tr("低"), tr("中"), tr("高"), tr("最高")});
    m_qualityCombo->setCurrentIndex(3); // 默认选中“最高”（你的原始设置）
    generalGridLayout->addWidget(new QLabel(tr("精度"), generalGroupBox), 0, 0);
    generalGridLayout->addWidget(m_qualityCombo, 0, 1);

    // 相机优化路径下拉框（SFM/光束法平差/自动）
    m_cameraOptCombo = new QComboBox(generalGroupBox);
    m_cameraOptCombo->addItems({tr("SFM"), tr("光束法平差"), tr("自动")});
    m_cameraOptCombo->setCurrentIndex(2); // 默认选中"自动"
    generalGridLayout->addWidget(new QLabel(tr("相机优化"), generalGroupBox), 1, 0);
    generalGridLayout->addWidget(m_cameraOptCombo, 1, 1);

    // 通用预选复选框（已移除）

    // 步骤完成后保存复选框（保留你的原始控件）
    m_stepCompletionCheck = new QCheckBox(tr("在每个步骤完成后保存项目"), generalGroupBox);
    m_stepCompletionCheck->setChecked(false);
    generalGridLayout->addWidget(m_stepCompletionCheck, 3, 0, 1, 2);

    // 输出目录：输入框 + 浏览按钮（新增核心功能，不修改其他逻辑）
    m_outputDirEdit = new QLineEdit(generalGroupBox);
    m_browseBtn = new QPushButton(tr("浏览"), generalGroupBox);
    // 绑定浏览按钮点击事件
    connect(m_browseBtn, &QPushButton::clicked, this, &AerialTriangulationDialog::onBrowseOutputDir);
    generalGridLayout->addWidget(new QLabel(tr("输出目录"), generalGroupBox), 4, 0);
    generalGridLayout->addWidget(m_outputDirEdit, 4, 1);
    generalGridLayout->addWidget(m_browseBtn, 4, 2); // 新增浏览按钮，不影响其他布局

    rightVLayout->addWidget(generalGroupBox);

    // ---------- 相机内方位元素组 ----------
    auto *intrinsicsGroupBox = new QGroupBox(tr("相机内方位元素"), rightWidget);
    auto *intrinsicsLayout = new QGridLayout(intrinsicsGroupBox);
    intrinsicsLayout->setContentsMargins(5, 5, 5, 5);
    intrinsicsLayout->setSpacing(5);

    m_intrinsicsCheck = new QCheckBox(tr("手动输入内参"), intrinsicsGroupBox);
    m_intrinsicsCheck->setChecked(false);
    intrinsicsLayout->addWidget(m_intrinsicsCheck, 0, 0, 1, 4);

    auto makeSpin = [&](const QString &label, const QString &suffix,
                        int row, int decimals, double maxVal, double step) -> QDoubleSpinBox* {
        auto *spin = new QDoubleSpinBox(intrinsicsGroupBox);
        spin->setRange(0.0, maxVal);
        spin->setDecimals(decimals);
        spin->setSingleStep(step);
        spin->setValue(0.0);
        spin->setEnabled(false);
        spin->setSuffix(suffix);
        intrinsicsLayout->addWidget(new QLabel(label, intrinsicsGroupBox), row, 0);
        intrinsicsLayout->addWidget(spin, row, 1);
        return spin;
    };

    // fu/fv: 焦距，单位 mm（配合 pitch 转像素），保留 9 位小数
    m_fuSpin = makeSpin(tr("fu"), QStringLiteral(" mm"), 1, 9, 99999.0, 0.001);
    m_fvSpin = makeSpin(tr("fv"), QStringLiteral(" mm"), 2, 9, 99999.0, 0.001);
    // cu/cv: 主点偏移，单位 mm，保留 9 位小数
    m_cuSpin = makeSpin(tr("cu"), QStringLiteral(" mm"), 3, 9, 99999.0, 0.001);
    m_cvSpin = makeSpin(tr("cv"), QStringLiteral(" mm"), 4, 9, 99999.0, 0.001);
    // pitch: 像元大小，单位 mm，保留 6 位小数
    m_pitchSpin = makeSpin(tr("pitch"), QStringLiteral(" mm"), 5, 6, 1.0, 0.0001);

    connect(m_intrinsicsCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_fuSpin->setEnabled(on);
        m_fvSpin->setEnabled(on);
        m_cuSpin->setEnabled(on);
        m_cvSpin->setEnabled(on);
        m_pitchSpin->setEnabled(on);
    });

    rightVLayout->addWidget(intrinsicsGroupBox);

    // ---------- 高级选项组（折叠）（保留线程数手动选择） ----------
    auto *advancedWrapper = new QWidget(rightWidget);
    auto *advWrapperLayout = new QVBoxLayout(advancedWrapper);
    advWrapperLayout->setContentsMargins(0, 0, 0, 0);
    advWrapperLayout->setSpacing(5);

    // 折叠按钮
    auto *advToggleBtn = new QToolButton(advancedWrapper);
    advToggleBtn->setText(tr("高级选项"));
    advToggleBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    advToggleBtn->setCheckable(true);
    advToggleBtn->setChecked(false);
    advToggleBtn->setArrowType(Qt::RightArrow);
    // 优化折叠按钮样式，不修改核心逻辑
    advToggleBtn->setStyleSheet(R"(
        QToolButton {
            border: none;
            padding: 5px;
            font-size: 14px;
        }
        QToolButton:checked {
            color: #1890FF;
        }
    )");

    // 高级选项内容容器（默认隐藏）
    auto *advContentWidget = new QWidget(advancedWrapper);
    advContentWidget->setVisible(false);
    auto *advGridLayout = new QGridLayout(advContentWidget);
    advGridLayout->setContentsMargins(5, 5, 5, 5);
    advGridLayout->setSpacing(5);

    // 重置当前对齐复选框（保留）
    m_resetCurrentAlignCheck = new QCheckBox(tr("重置当前对齐"), advContentWidget);
    m_resetCurrentAlignCheck->setChecked(true);
    advGridLayout->addWidget(m_resetCurrentAlignCheck, 0, 0);

    // 线程数下拉框（严格保留手动选择，你的原始逻辑：1-系统最大核心数）
    m_threadsCombo = new QComboBox(advContentWidget);
    int maxTh = QThread::idealThreadCount();
    if (maxTh <= 0) maxTh = 1;
    for (int i = 1; i <= maxTh; ++i) m_threadsCombo->addItem(QString::number(i));
    m_threadsCombo->setCurrentIndex(maxTh - 1); // 默认选中最大线程数
    advGridLayout->addWidget(new QLabel(tr("线程数"), advContentWidget), 0, 1);
    advGridLayout->addWidget(m_threadsCombo, 0, 2);

    // ── 光束法平差参数 ──────────────────────────────────────────────────────
    auto *baSepLine = new QFrame(advContentWidget);
    baSepLine->setFrameShape(QFrame::HLine);
    baSepLine->setFrameShadow(QFrame::Sunken);
    advGridLayout->addWidget(new QLabel(QStringLiteral("<b>光束法平差</b>"), advContentWidget), 1, 0, 1, 3);
    advGridLayout->addWidget(baSepLine, 2, 0, 1, 3);

    // 是否优化相机位姿
    m_baRefinePoseCheck = new QCheckBox(tr("优化相机位姿（6-DOF）"), advContentWidget);
    m_baRefinePoseCheck->setChecked(true);
    m_baRefinePoseCheck->setToolTip(tr("勾选后 BA 将同时优化相机外参；仅优化三维点时取消勾选"));
    advGridLayout->addWidget(m_baRefinePoseCheck, 3, 0, 1, 3);

    // 最大迭代次数
    m_baMaxIterSpin = new QSpinBox(advContentWidget);
    m_baMaxIterSpin->setRange(2, 200);
    m_baMaxIterSpin->setValue(20);
    m_baMaxIterSpin->setToolTip(tr("光束法平差外层交替迭代轮数（点+相机各一次为一轮）"));
    advGridLayout->addWidget(new QLabel(tr("最大迭代次数"), advContentWidget), 4, 0);
    advGridLayout->addWidget(m_baMaxIterSpin, 4, 1);

    // Huber 阈值
    m_baHuberDeltaSpin = new QDoubleSpinBox(advContentWidget);
    m_baHuberDeltaSpin->setRange(0.5, 20.0);
    m_baHuberDeltaSpin->setDecimals(1);
    m_baHuberDeltaSpin->setSingleStep(0.5);
    m_baHuberDeltaSpin->setValue(2.0);
    m_baHuberDeltaSpin->setSuffix(tr(" px"));
    m_baHuberDeltaSpin->setToolTip(tr("Huber 损失阈值：残差超过此值的观测被降权以抑制粗差影响\n\n"
                                      "较小值粗差抑制能力更强但可能影响正常观测收敛\n"
                                      "建议值: 1.5-3.0 px（航空摄影推荐 2.0 px）"));
    advGridLayout->addWidget(new QLabel(tr("鲁棒核阈值"), advContentWidget), 5, 0);
    advGridLayout->addWidget(m_baHuberDeltaSpin, 5, 1);

    // 离群点过滤阈值
    m_baFilterReprojSpin = new QDoubleSpinBox(advContentWidget);
    m_baFilterReprojSpin->setRange(1.0, 50.0);
    m_baFilterReprojSpin->setDecimals(1);
    m_baFilterReprojSpin->setSingleStep(1.0);
    m_baFilterReprojSpin->setValue(4.0);
    m_baFilterReprojSpin->setSuffix(tr(" px"));
    m_baFilterReprojSpin->setToolTip(tr("BA 后重投影误差超过此值的三维点将被过滤删除\n\n"
                                        "较小值可剔除更多粗差但可能误删正确点\n"
                                        "建议值: 3.0-6.0 px（航空摄影推荐 4.0 px）"));
    advGridLayout->addWidget(new QLabel(tr("过滤阈值"), advContentWidget), 6, 0);
    advGridLayout->addWidget(m_baFilterReprojSpin, 6, 1);

    // LM 阻尼因子
    m_baDampingSpin = new QDoubleSpinBox(advContentWidget);
    m_baDampingSpin->setRange(1e-10, 1e-1);
    m_baDampingSpin->setDecimals(8);
    m_baDampingSpin->setSingleStep(1e-6);
    m_baDampingSpin->setValue(1e-6);
    m_baDampingSpin->setToolTip(tr("Levenberg-Marquardt 阻尼因子，较大值更稳定但收敛慢"));
    advGridLayout->addWidget(new QLabel(tr("LM 阻尼"), advContentWidget), 7, 0);
    advGridLayout->addWidget(m_baDampingSpin, 7, 1);

    // 分隔线（优化视觉，不影响逻辑）
    auto *advLine = new QFrame(advancedWrapper);
    advLine->setFrameShape(QFrame::HLine);
    advLine->setFrameShadow(QFrame::Sunken);
    advLine->setStyleSheet(R"(
        QFrame {
            margin: 10px 0;
            color: #E0E0E0;
        }
    )");

    // 绑定折叠按钮事件
    connect(advToggleBtn, &QToolButton::toggled, this, [advToggleBtn, advContentWidget](bool checked) {
        advToggleBtn->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        advContentWidget->setVisible(checked);
    });

    // 组装高级选项布局
    advWrapperLayout->addWidget(advToggleBtn);
    advWrapperLayout->addWidget(advContentWidget);
    advWrapperLayout->addWidget(advLine);

    rightVLayout->addWidget(advancedWrapper);
    rightVLayout->addStretch(); // 拉伸填充空白，优化布局

    // 设置右侧布局权重（保留你的原始比例）
    contentHLayout->addWidget(rightWidget, 0);

    // ========== 底部：按钮区域 ==========
    m_btnBox = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok, this);
    // 默认禁用OK按钮（无影像时）
    m_btnBox->button(QDialogButtonBox::Ok)->setEnabled(false);

    // ========== 组装整体布局 ==========
    mainLayout->addWidget(contentWidget);
    mainLayout->addWidget(m_btnBox);

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
