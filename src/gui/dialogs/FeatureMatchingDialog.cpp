#include "FeatureMatchingDialog.h"
#include "ui_FeatureMatchingDialog.h"
#include "MatchViewerDialog.h"

#include <QAbstractItemView>
#include <QListWidget>
#include <QPushButton>
#include <QJsonObject>
#include <QJsonArray>
#include <QLineEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QTextEdit>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QLabel>
#include <QSplitter>
#include <QStackedWidget>
#include <QToolButton>
#include <QMessageBox>
#include <QFileInfo>
#include <QRegularExpression>

#include "AlgorithmCompat.h"

FeatureMatchingDialog::FeatureMatchingDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    setupConnections();
}

FeatureMatchingDialog::~FeatureMatchingDialog() = default;

void FeatureMatchingDialog::setupUi() 
{
    setWindowTitle(tr("特征匹配"));
    resize(1000, 700);

    {
        Ui::FeatureMatchingDialog ui;
        ui.setupUi(this);

        m_imageInputWidget = ui.m_imageInputWidget;
        m_selectAllBtn = ui.m_selectAllBtn;
        m_deselectAllBtn = ui.m_deselectAllBtn;
        m_imageList = ui.m_imageList;
        m_pairPreview = ui.m_pairPreview;
        m_lisFileLine = ui.m_lisFileLine;
        m_addLisBtn = ui.m_addLisBtn;
        m_clearLisBtn = ui.m_clearLisBtn;
        m_generatePairsBtn = ui.m_generatePairsBtn;
        m_outputLine = ui.m_outputLine;
        m_browseOutBtn = ui.m_browseOutBtn;
        m_paramStack = ui.m_paramStack;
        m_advancedGroup = ui.m_advancedGroup;
        m_systemGroup = ui.m_systemGroup;
        m_debugGroup = ui.m_debugGroup;
        m_resetBtn = ui.m_resetBtn;
        m_viewMatchesBtn = ui.m_viewMatchesBtn;
        m_runBtn = ui.m_runBtn;
        m_cancelBtn = ui.m_cancelBtn;

        if (ui.topSplit)
        {
            ui.topSplit->setStretchFactor(0, 3);
            ui.topSplit->setStretchFactor(1, 2);
        }

        auto *right = ui.rightWidget;
        m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);

        auto *commonForm = ui.commonForm;
        m_matchAlgorithmCombo = new QComboBox(right);
        m_matchAlgorithmCombo->addItem(tr("SuperGlue"), "superglue");
        m_matchAlgorithmCombo->addItem(tr("LightGlue"), "lightglue");
        m_matchAlgorithmCombo->addItem(tr("LoFTR"), "loftr");
        m_matchAlgorithmCombo->addItem(tr("RoMa"), "roma");
        m_matchAlgorithmCombo->addItem(tr("BF-Hamming (ORB)"), "orb_bf_hamming");
        m_matchAlgorithmCombo->addItem(tr("BF-L2 (SIFT)"), "sift_bf_l2");
        m_matchAlgorithmCombo->addItem(tr("FLANN (SIFT)"), "sift_flann");
        m_matchAlgorithmCombo->setToolTip(tr("选择特征匹配算法\n"
                                             "SuperGlue/LightGlue: 需要预提取特征\n"
                                             "LoFTR/RoMa: 直接处理原始图像"));
        commonForm->addRow(tr("匹配算法:"), m_matchAlgorithmCombo);

        m_featureSuffixLabel = new QLabel(tr("特征类型:"), right);
        m_featureSuffixCombo = new QComboBox(right);
        m_featureSuffixCombo->setToolTip(tr("选择用于匹配的特征文件类型\n"
                                            "根据所选算法自动过滤可用后缀"));
        commonForm->addRow(m_featureSuffixLabel, m_featureSuffixCombo);

        m_maxKeypointsSpin = new QSpinBox(right);
        m_maxKeypointsSpin->setRange(-1, 100000);
        m_maxKeypointsSpin->setValue(-1);
        m_maxKeypointsSpin->setSpecialValueText(tr("不限制"));
        m_maxKeypointsSpin->setToolTip(tr("最大关键点数量限制，-1 表示不限制"));
        commonForm->addRow(tr("最大关键点数:"), m_maxKeypointsSpin);

        m_outlierMethodCombo = new QComboBox(right);
        m_outlierMethodCombo->addItem(tr("不剔除"), "none");
        m_outlierMethodCombo->addItem(tr("Fundamental RANSAC"), "fundamental");
        m_outlierMethodCombo->addItem(tr("Fundamental USAC_MAGSAC （推荐）"), "fundamental_usac_magsac");
        m_outlierMethodCombo->addItem(tr("Homography RANSAC"), "homography");
        m_outlierMethodCombo->addItem(tr("Affine RANSAC"), "affine");
        m_outlierMethodCombo->setCurrentIndex(2);
        m_outlierMethodCombo->setToolTip(tr("匹配后粗差剔除算法（推荐 USAC_MAGSAC）\n\n"
                                            "★ Fundamental USAC_MAGSAC: 自适应阈值，粗差剔除能力远超传统 RANSAC（强烈推荐）\n"
                                            "Fundamental RANSAC: 传统基础矩阵 RANSAC\n"
                                            "Homography RANSAC: 仅适用于纯平面/纯旋转场景\n"
                                            "Affine RANSAC: 仅适用于远距离平行投影场景\n"
                                            "不剔除: 仅调试时使用"));
        commonForm->addRow(tr("粗差剔除:"), m_outlierMethodCombo);

        {
            auto *sgPage = new QWidget(m_paramStack);
            auto *sgForm = new QFormLayout(sgPage);
            sgForm->setContentsMargins(0, 4, 0, 4);

            m_modelTypeCombo = new QComboBox(sgPage);
            m_modelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
            m_modelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");
            m_modelTypeCombo->setToolTip(tr("SuperGlue 预训练模型类型\nOutdoor: 适用于外景、航空摄影\nIndoor: 适用于室内、近景摄影"));
            sgForm->addRow(tr("模型类型:"), m_modelTypeCombo);

            m_matchThresholdSpin = new QDoubleSpinBox(sgPage);
            m_matchThresholdSpin->setRange(0.0, 1.0);
            m_matchThresholdSpin->setDecimals(3);
            m_matchThresholdSpin->setSingleStep(0.01);
            m_matchThresholdSpin->setValue(0.15);
            m_matchThresholdSpin->setToolTip(tr("SuperGlue 匹配置信度阈值 [0.0-1.0]\n推荐 0.15（航空）/ 0.10（近景）"));
            sgForm->addRow(tr("匹配阈值:"), m_matchThresholdSpin);

            m_sinkhornIterSpin = new QSpinBox(sgPage);
            m_sinkhornIterSpin->setRange(1, 1000);
            m_sinkhornIterSpin->setValue(150);
            m_sinkhornIterSpin->setToolTip(tr("Sinkhorn 正则化迭代次数，推荐 20-200"));
            sgForm->addRow(tr("Sinkhorn 迭代:"), m_sinkhornIterSpin);

            m_batchSizeSpin = new QSpinBox(sgPage);
            m_batchSizeSpin->setRange(1, 64);
            m_batchSizeSpin->setValue(1);
            m_batchSizeSpin->setToolTip(tr("推理批次大小，GPU 环境下可增大以提高吞吐量"));
            sgForm->addRow(tr("批处理大小:"), m_batchSizeSpin);

            m_inputWidthSpin = new QSpinBox(sgPage);
            m_inputWidthSpin->setRange(-1, 20000);
            m_inputWidthSpin->setValue(-1);
            m_inputWidthSpin->setSpecialValueText(tr("自动"));
            sgForm->addRow(tr("输入宽度:"), m_inputWidthSpin);

            m_inputHeightSpin = new QSpinBox(sgPage);
            m_inputHeightSpin->setRange(-1, 20000);
            m_inputHeightSpin->setValue(-1);
            m_inputHeightSpin->setSpecialValueText(tr("自动"));
            sgForm->addRow(tr("输入高度:"), m_inputHeightSpin);

            m_paramStack->addWidget(sgPage);
        }

        {
            auto *lgPage = new QWidget(m_paramStack);
            auto *lgForm = new QFormLayout(lgPage);
            lgForm->setContentsMargins(0, 4, 0, 4);

            m_lgMatchThresholdSpin = new QDoubleSpinBox(lgPage);
            m_lgMatchThresholdSpin->setRange(0.0, 1.0);
            m_lgMatchThresholdSpin->setDecimals(3);
            m_lgMatchThresholdSpin->setSingleStep(0.01);
            m_lgMatchThresholdSpin->setValue(0.15);
            m_lgMatchThresholdSpin->setToolTip(tr("LightGlue 匹配置信度阈值 [0.0-1.0]\n推荐 0.15（航空）/ 0.10（近景）"));
            lgForm->addRow(tr("匹配阈值:"), m_lgMatchThresholdSpin);

            m_lgBatchSizeSpin = new QSpinBox(lgPage);
            m_lgBatchSizeSpin->setRange(1, 64);
            m_lgBatchSizeSpin->setValue(1);
            m_lgBatchSizeSpin->setToolTip(tr("推理批次大小，GPU 环境下可增大以提高吞吐量"));
            lgForm->addRow(tr("批处理大小:"), m_lgBatchSizeSpin);

            m_lgInputWidthSpin = new QSpinBox(lgPage);
            m_lgInputWidthSpin->setRange(-1, 20000);
            m_lgInputWidthSpin->setValue(-1);
            m_lgInputWidthSpin->setSpecialValueText(tr("自动"));
            lgForm->addRow(tr("输入宽度:"), m_lgInputWidthSpin);

            m_lgInputHeightSpin = new QSpinBox(lgPage);
            m_lgInputHeightSpin->setRange(-1, 20000);
            m_lgInputHeightSpin->setValue(-1);
            m_lgInputHeightSpin->setSpecialValueText(tr("自动"));
            lgForm->addRow(tr("输入高度:"), m_lgInputHeightSpin);

            m_paramStack->addWidget(lgPage);
        }

        {
            auto *tradPage = new QWidget(m_paramStack);
            auto *tradLayout = new QVBoxLayout(tradPage);
            tradLayout->setContentsMargins(0, 4, 0, 4);
            auto *tradHint = new QLabel(tr("传统算法（BF / FLANN）无额外参数。\n"
                                           "匹配质量由粗差剔除算法和 RANSAC 参数控制。"), tradPage);
            tradHint->setWordWrap(true);
            tradLayout->addWidget(tradHint);
            tradLayout->addStretch();
            m_paramStack->addWidget(tradPage);
        }

        {
            auto *loftrPage = new QWidget(m_paramStack);
            auto *loftrLayout = new QVBoxLayout(loftrPage);
            loftrLayout->setContentsMargins(0, 4, 0, 4);

            auto *loftrWarning = new QLabel(tr("⚠ LoFTR 直接处理原始图像，不使用已有特征提取结果"), loftrPage);
            loftrWarning->setStyleSheet("color: #ff9800; font-weight: bold;");
            loftrWarning->setWordWrap(true);
            loftrLayout->addWidget(loftrWarning);

            auto *loftrForm = new QFormLayout();
            loftrLayout->addLayout(loftrForm);

            m_loftrModelTypeCombo = new QComboBox(loftrPage);
            m_loftrModelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
            m_loftrModelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");
            loftrForm->addRow(tr("模型类型:"), m_loftrModelTypeCombo);

            m_loftrMatchThresholdSpin = new QDoubleSpinBox(loftrPage);
            m_loftrMatchThresholdSpin->setRange(0.0, 1.0);
            m_loftrMatchThresholdSpin->setDecimals(3);
            m_loftrMatchThresholdSpin->setSingleStep(0.01);
            m_loftrMatchThresholdSpin->setValue(0.2);
            m_loftrMatchThresholdSpin->setToolTip(tr("LoFTR 匹配置信度阈值"));
            loftrForm->addRow(tr("匹配阈值:"), m_loftrMatchThresholdSpin);

            loftrLayout->addStretch();
            m_paramStack->addWidget(loftrPage);
        }

        {
            auto *romaPage = new QWidget(m_paramStack);
            auto *romaLayout = new QVBoxLayout(romaPage);
            romaLayout->setContentsMargins(0, 4, 0, 4);

            auto *romaWarning = new QLabel(tr("⚠ RoMa 直接处理原始图像，不使用已有特征提取结果\n"
                                              "适合大旋转角度的小行星影像"), romaPage);
            romaWarning->setStyleSheet("color: #ff9800; font-weight: bold;");
            romaWarning->setWordWrap(true);
            romaLayout->addWidget(romaWarning);

            auto *romaForm = new QFormLayout();
            romaLayout->addLayout(romaForm);

            m_romaModelTypeCombo = new QComboBox(romaPage);
            m_romaModelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
            m_romaModelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");
            romaForm->addRow(tr("模型类型:"), m_romaModelTypeCombo);

            m_romaMatchThresholdSpin = new QDoubleSpinBox(romaPage);
            m_romaMatchThresholdSpin->setRange(0.0, 1.0);
            m_romaMatchThresholdSpin->setDecimals(3);
            m_romaMatchThresholdSpin->setSingleStep(0.01);
            m_romaMatchThresholdSpin->setValue(0.05);
            m_romaMatchThresholdSpin->setToolTip(tr("RoMa 匹配置信度阈值"));
            romaForm->addRow(tr("匹配阈值:"), m_romaMatchThresholdSpin);

            m_romaMaxKeypointsSpin = new QSpinBox(romaPage);
            m_romaMaxKeypointsSpin->setRange(100, 50000);
            m_romaMaxKeypointsSpin->setValue(10000);
            m_romaMaxKeypointsSpin->setToolTip(tr("RoMa 最大关键点数"));
            romaForm->addRow(tr("最大关键点数:"), m_romaMaxKeypointsSpin);

            romaLayout->addStretch();
            m_paramStack->addWidget(romaPage);
        }

        connect(ui.advancedToggle, &QToolButton::toggled, this, [this, advancedToggle = ui.advancedToggle](bool checked)
        {
            m_advancedGroup->setVisible(checked);
            advancedToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        });

        auto *advancedForm = ui.advancedForm;
        m_outlierReprojSpin = new QDoubleSpinBox(right);
        m_outlierReprojSpin->setRange(0.1, 50.0);
        m_outlierReprojSpin->setDecimals(2);
        m_outlierReprojSpin->setSingleStep(0.1);
        m_outlierReprojSpin->setValue(1.5);
        m_outlierReprojSpin->setToolTip(tr("RANSAC/USAC 重投影误差阈值（像素）\n"
                                           "推荐: 1.0-1.5（高分辨率）/ 1.5-3.0（中等）/ 3.0-6.0（低分辨率）"));
        advancedForm->addRow(tr("RANSAC 阈值(px):"), m_outlierReprojSpin);

        m_outlierConfidenceSpin = new QDoubleSpinBox(right);
        m_outlierConfidenceSpin->setRange(0.50, 0.9999);
        m_outlierConfidenceSpin->setDecimals(4);
        m_outlierConfidenceSpin->setSingleStep(0.0010);
        m_outlierConfidenceSpin->setValue(0.9999);
        m_outlierConfidenceSpin->setToolTip(tr("RANSAC/USAC 置信度 [0.5-0.9999]，推荐 0.9999"));
        advancedForm->addRow(tr("RANSAC 置信度:"), m_outlierConfidenceSpin);

        m_outlierMaxItersSpin = new QSpinBox(right);
        m_outlierMaxItersSpin->setRange(100, 100000);
        m_outlierMaxItersSpin->setValue(10000);
        m_outlierMaxItersSpin->setToolTip(tr("RANSAC/USAC 最大迭代次数，推荐 10000"));
        advancedForm->addRow(tr("RANSAC 最大迭代:"), m_outlierMaxItersSpin);

        m_outlierMinInliersSpin = new QSpinBox(right);
        m_outlierMinInliersSpin->setRange(1, 100000);
        m_outlierMinInliersSpin->setValue(25);
        m_outlierMinInliersSpin->setToolTip(tr("粗差剔除后最少内点数，低于此值则丢弃该匹配对\n推荐: ≥25（航空）/ ≥15（近景）"));
        advancedForm->addRow(tr("最少内点数:"), m_outlierMinInliersSpin);

        connect(ui.systemToggle, &QToolButton::toggled, this, [this, systemToggle = ui.systemToggle](bool checked)
        {
            m_systemGroup->setVisible(checked);
            systemToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        });

        auto *systemForm = ui.systemForm;
        m_deviceCombo = new QComboBox(right);
        m_deviceCombo->addItems({tr("CUDA（如可用）"), tr("CPU")});
        m_deviceCombo->setToolTip(tr("计算设备选择\n"
                                     "CUDA需要GPU支持"));
        systemForm->addRow(tr("计算设备:"), m_deviceCombo);

        m_numThreadsSpin = new QSpinBox(right);
        m_numThreadsSpin->setRange(-1, 64);
        m_numThreadsSpin->setValue(-1);
        m_numThreadsSpin->setSpecialValueText(tr("自动"));
        m_numThreadsSpin->setToolTip(tr("CPU线程数（仅CPU模式有效）\n"
                                        "-1 表示自动检测"));
        systemForm->addRow(tr("CPU线程数:"), m_numThreadsSpin);

        m_cudaParallelSpin = new QSpinBox(right);
        m_cudaParallelSpin->setRange(1, 8);
        m_cudaParallelSpin->setValue(1);
        m_cudaParallelSpin->setToolTip(tr("CUDA模式下同时处理的影像对数\n"
                                          "每对独立持有一个模型实例，每实例约占显存 200-400 MB\n"
                                          "请确认 GPU 显存充裕再增大此参数"));
        systemForm->addRow(tr("CUDA并行对数:"), m_cudaParallelSpin);

        connect(ui.debugToggle, &QToolButton::toggled, this, [this, debugToggle = ui.debugToggle](bool checked)
        {
            m_debugGroup->setVisible(checked);
            debugToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        });

        auto *debugForm = ui.debugForm;
        m_saveCsvChk = new QCheckBox(tr("保存匹配结果为CSV"), right);
        m_saveCsvChk->setToolTip(tr("将匹配关系导出为CSV格式（便于调试）"));
        debugForm->addRow(m_saveCsvChk);

        m_saveVisChk = new QCheckBox(tr("保存匹配可视化图像"), right);
        m_saveVisChk->setToolTip(tr("生成带匹配连线的可视化图像"));
        debugForm->addRow(m_saveVisChk);

        m_verboseChk = new QCheckBox(tr("详细日志输出"), right);
        m_verboseChk->setToolTip(tr("打印详细的匹配过程信息"));
        debugForm->addRow(m_verboseChk);

        return;
    }

    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // 上部：特征列表 + 匹配对预览 + 参数面板
    QSplitter* topSplit = new QSplitter(this);

    // ==================== 左侧：特征文件与匹配对 ====================
    QWidget* left = new QWidget(this);
    QVBoxLayout* leftLayout = new QVBoxLayout(left);

    // ── 影像输入区（所有算法共用）──
    m_imageInputWidget = new QWidget(left);
    QVBoxLayout* imageInputLayout = new QVBoxLayout(m_imageInputWidget);
    imageInputLayout->setContentsMargins(0, 0, 0, 0);

    QHBoxLayout* imageHeaderRow = new QHBoxLayout();
    imageHeaderRow->addWidget(new QLabel(tr("选择影像:"), m_imageInputWidget));
    imageHeaderRow->addStretch();
    m_selectAllBtn = new QPushButton(tr("全选"), m_imageInputWidget);
    m_selectAllBtn->setFixedWidth(60);
    m_selectAllBtn->setToolTip(tr("选中所有影像"));
    m_deselectAllBtn = new QPushButton(tr("清除"), m_imageInputWidget);
    m_deselectAllBtn->setFixedWidth(60);
    m_deselectAllBtn->setToolTip(tr("取消选中所有影像"));
    imageHeaderRow->addWidget(m_selectAllBtn);
    imageHeaderRow->addWidget(m_deselectAllBtn);
    imageInputLayout->addLayout(imageHeaderRow);

    m_imageList = new QListWidget(m_imageInputWidget);
    m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_imageList->setToolTip(tr("所有算法共用的影像列表\n选中后可生成匹配对"));
    imageInputLayout->addWidget(m_imageList, 1);
    leftLayout->addWidget(m_imageInputWidget, 1);

    // lis文件输入
    QHBoxLayout* lisRow = new QHBoxLayout();
    m_lisFileLine = new QLineEdit(left);
    m_lisFileLine->setPlaceholderText(tr("可选：lis文件路径（格式：1 2\\n2 3）"));
    m_lisFileLine->setToolTip(tr("lis文件定义匹配对关系\n"
                                 "每行格式：image1_id image2_id\n"
                                 "不指定则自动两两匹配"));
    m_addLisBtn = new QPushButton(tr("浏览..."), left);
    m_clearLisBtn = new QPushButton(tr("清空"), left);
    lisRow->addWidget(new QLabel(tr("lis文件:"), left));
    lisRow->addWidget(m_lisFileLine);
    lisRow->addWidget(m_addLisBtn);
    lisRow->addWidget(m_clearLisBtn);
    leftLayout->addLayout(lisRow);

    // 生成匹配对按钮
    m_generatePairsBtn = new QPushButton(tr("生成匹配对"), left);
    m_generatePairsBtn->setToolTip(tr("根据选中的影像和lis文件生成匹配对列表"));
    leftLayout->addWidget(m_generatePairsBtn);

    // 匹配对预览
    leftLayout->addWidget(new QLabel(tr("匹配对预览:"), left));
    m_pairPreview = new QTextEdit(left);
    m_pairPreview->setReadOnly(true);
    m_pairPreview->setMaximumHeight(150);
    m_pairPreview->setToolTip(tr("将要执行的匹配对列表"));
    leftLayout->addWidget(m_pairPreview);

    // 输出设置
    QHBoxLayout* outRow = new QHBoxLayout();
    m_outputLine = new QLineEdit(left);
    m_outputLine->setPlaceholderText(tr("保存到项目 assets/matches"));
    m_outputLine->setToolTip(tr("留空时：默认保存到项目 assets/matches 目录"));
    m_browseOutBtn = new QPushButton(tr("浏览..."), left);
    outRow->addWidget(new QLabel(tr("输出目录："), left));
    outRow->addWidget(m_outputLine);
    outRow->addWidget(m_browseOutBtn);
    leftLayout->addLayout(outRow);

    topSplit->addWidget(left);

    // ==================== 右侧：参数面板 ====================
    QWidget* right = new QWidget(this);
    QVBoxLayout* rightLayout = new QVBoxLayout(right);

    // ── 始终显示的基础参数 ──────────────────────────────────────
    QFormLayout* commonForm = new QFormLayout();
    rightLayout->addLayout(commonForm);

    m_matchAlgorithmCombo = new QComboBox(right);
    m_matchAlgorithmCombo->addItem(tr("SuperGlue"), "superglue");
    m_matchAlgorithmCombo->addItem(tr("LightGlue"), "lightglue");
    m_matchAlgorithmCombo->addItem(tr("LoFTR"), "loftr");
    m_matchAlgorithmCombo->addItem(tr("RoMa"), "roma");
    m_matchAlgorithmCombo->addItem(tr("BF-Hamming (ORB)"), "orb_bf_hamming");
    m_matchAlgorithmCombo->addItem(tr("BF-L2 (SIFT)"), "sift_bf_l2");
    m_matchAlgorithmCombo->addItem(tr("FLANN (SIFT)"), "sift_flann");
    m_matchAlgorithmCombo->setToolTip(tr("选择特征匹配算法\n"
                                         "SuperGlue/LightGlue: 需要预提取特征\n"
                                         "LoFTR/RoMa: 直接处理原始图像"));
    commonForm->addRow(tr("匹配算法:"), m_matchAlgorithmCombo);

    m_featureSuffixLabel = new QLabel(tr("特征类型:"), right);
    m_featureSuffixCombo = new QComboBox(right);
    m_featureSuffixCombo->setToolTip(tr("选择用于匹配的特征文件类型\n"
                                         "根据所选算法自动过滤可用后缀"));
    commonForm->addRow(m_featureSuffixLabel, m_featureSuffixCombo);

    m_maxKeypointsSpin = new QSpinBox(right);
    m_maxKeypointsSpin->setRange(-1, 100000);
    m_maxKeypointsSpin->setValue(-1);
    m_maxKeypointsSpin->setSpecialValueText(tr("不限制"));
    m_maxKeypointsSpin->setToolTip(tr("最大关键点数量限制，-1 表示不限制"));
    commonForm->addRow(tr("最大关键点数:"), m_maxKeypointsSpin);

    m_outlierMethodCombo = new QComboBox(right);
    m_outlierMethodCombo->addItem(tr("不剔除"), "none");
    m_outlierMethodCombo->addItem(tr("Fundamental RANSAC"), "fundamental");
    m_outlierMethodCombo->addItem(tr("Fundamental USAC_MAGSAC （推荐）"), "fundamental_usac_magsac");
    m_outlierMethodCombo->addItem(tr("Homography RANSAC"), "homography");
    m_outlierMethodCombo->addItem(tr("Affine RANSAC"), "affine");
    m_outlierMethodCombo->setCurrentIndex(2);
    m_outlierMethodCombo->setToolTip(tr("匹配后粗差剔除算法（推荐 USAC_MAGSAC）\n\n"
                                        "★ Fundamental USAC_MAGSAC: 自适应阈值，粗差剔除能力远超传统 RANSAC（强烈推荐）\n"
                                        "Fundamental RANSAC: 传统基础矩阵 RANSAC\n"
                                        "Homography RANSAC: 仅适用于纯平面/纯旋转场景\n"
                                        "Affine RANSAC: 仅适用于远距离平行投影场景\n"
                                        "不剔除: 仅调试时使用"));
    commonForm->addRow(tr("粗差剔除:"), m_outlierMethodCombo);

    // ── 算法专属参数（QStackedWidget）──────────────────────────
    m_paramStack = new QStackedWidget(right);
    rightLayout->addWidget(m_paramStack);

    // ── Page 0: SuperGlue ──────────────────────────────────────
    {
        QWidget* sgPage = new QWidget();
        QFormLayout* sgForm = new QFormLayout(sgPage);
        sgForm->setContentsMargins(0, 4, 0, 4);

        m_modelTypeCombo = new QComboBox(sgPage);
        m_modelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
        m_modelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");
        m_modelTypeCombo->setToolTip(tr("SuperGlue 预训练模型类型\nOutdoor: 适用于外景、航空摄影\nIndoor: 适用于室内、近景摄影"));
        sgForm->addRow(tr("模型类型:"), m_modelTypeCombo);

        m_matchThresholdSpin = new QDoubleSpinBox(sgPage);
        m_matchThresholdSpin->setRange(0.0, 1.0);
        m_matchThresholdSpin->setDecimals(3);
        m_matchThresholdSpin->setSingleStep(0.01);
        m_matchThresholdSpin->setValue(0.15);
        m_matchThresholdSpin->setToolTip(tr("SuperGlue 匹配置信度阈值 [0.0-1.0]\n推荐 0.15（航空）/ 0.10（近景）"));
        sgForm->addRow(tr("匹配阈值:"), m_matchThresholdSpin);

        m_sinkhornIterSpin = new QSpinBox(sgPage);
        m_sinkhornIterSpin->setRange(1, 1000);
        m_sinkhornIterSpin->setValue(150);
        m_sinkhornIterSpin->setToolTip(tr("Sinkhorn 正则化迭代次数，推荐 20-200"));
        sgForm->addRow(tr("Sinkhorn 迭代:"), m_sinkhornIterSpin);

        m_batchSizeSpin = new QSpinBox(sgPage);
        m_batchSizeSpin->setRange(1, 64);
        m_batchSizeSpin->setValue(1);
        m_batchSizeSpin->setToolTip(tr("推理批次大小，GPU 环境下可增大以提高吞吐量"));
        sgForm->addRow(tr("批处理大小:"), m_batchSizeSpin);

        m_inputWidthSpin = new QSpinBox(sgPage);
        m_inputWidthSpin->setRange(-1, 20000);
        m_inputWidthSpin->setValue(-1);
        m_inputWidthSpin->setSpecialValueText(tr("自动"));
        sgForm->addRow(tr("输入宽度:"), m_inputWidthSpin);

        m_inputHeightSpin = new QSpinBox(sgPage);
        m_inputHeightSpin->setRange(-1, 20000);
        m_inputHeightSpin->setValue(-1);
        m_inputHeightSpin->setSpecialValueText(tr("自动"));
        sgForm->addRow(tr("输入高度:"), m_inputHeightSpin);

        m_paramStack->addWidget(sgPage);  // index 0
    }

    // ── Page 1: LightGlue ─────────────────────────────────────
    {
        QWidget* lgPage = new QWidget();
        QFormLayout* lgForm = new QFormLayout(lgPage);
        lgForm->setContentsMargins(0, 4, 0, 4);

        m_lgMatchThresholdSpin = new QDoubleSpinBox(lgPage);
        m_lgMatchThresholdSpin->setRange(0.0, 1.0);
        m_lgMatchThresholdSpin->setDecimals(3);
        m_lgMatchThresholdSpin->setSingleStep(0.01);
        m_lgMatchThresholdSpin->setValue(0.15);
        m_lgMatchThresholdSpin->setToolTip(tr("LightGlue 匹配置信度阈值 [0.0-1.0]\n推荐 0.15（航空）/ 0.10（近景）"));
        lgForm->addRow(tr("匹配阈值:"), m_lgMatchThresholdSpin);

        m_lgBatchSizeSpin = new QSpinBox(lgPage);
        m_lgBatchSizeSpin->setRange(1, 64);
        m_lgBatchSizeSpin->setValue(1);
        m_lgBatchSizeSpin->setToolTip(tr("推理批次大小，GPU 环境下可增大以提高吞吐量"));
        lgForm->addRow(tr("批处理大小:"), m_lgBatchSizeSpin);

        m_lgInputWidthSpin = new QSpinBox(lgPage);
        m_lgInputWidthSpin->setRange(-1, 20000);
        m_lgInputWidthSpin->setValue(-1);
        m_lgInputWidthSpin->setSpecialValueText(tr("自动"));
        lgForm->addRow(tr("输入宽度:"), m_lgInputWidthSpin);

        m_lgInputHeightSpin = new QSpinBox(lgPage);
        m_lgInputHeightSpin->setRange(-1, 20000);
        m_lgInputHeightSpin->setValue(-1);
        m_lgInputHeightSpin->setSpecialValueText(tr("自动"));
        lgForm->addRow(tr("输入高度:"), m_lgInputHeightSpin);

        m_paramStack->addWidget(lgPage);  // index 1
    }

    // ── Page 2: 传统算法（BF / FLANN）────────────────────────
    {
        QWidget* tradPage = new QWidget();
        QVBoxLayout* tradLayout = new QVBoxLayout(tradPage);
        tradLayout->setContentsMargins(0, 4, 0, 4);
        QLabel* tradHint = new QLabel(tr("传统算法（BF / FLANN）无额外参数。\n"
                                         "匹配质量由粗差剔除算法和 RANSAC 参数控制。"), tradPage);
        tradHint->setWordWrap(true);
        tradLayout->addWidget(tradHint);
        tradLayout->addStretch();
        m_paramStack->addWidget(tradPage);  // index 2
    }

    // ── Page 3: LoFTR ─────────────────────────────────────────
    {
        QWidget* loftrPage = new QWidget();
        QVBoxLayout* loftrLayout = new QVBoxLayout(loftrPage);
        loftrLayout->setContentsMargins(0, 4, 0, 4);

        QLabel* loftrWarning = new QLabel(tr("⚠ LoFTR 直接处理原始图像，不使用已有特征提取结果"), loftrPage);
        loftrWarning->setStyleSheet("color: #ff9800; font-weight: bold;");
        loftrWarning->setWordWrap(true);
        loftrLayout->addWidget(loftrWarning);

        QFormLayout* loftrForm = new QFormLayout();
        loftrLayout->addLayout(loftrForm);

        m_loftrModelTypeCombo = new QComboBox(loftrPage);
        m_loftrModelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
        m_loftrModelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");
        loftrForm->addRow(tr("模型类型:"), m_loftrModelTypeCombo);

        m_loftrMatchThresholdSpin = new QDoubleSpinBox(loftrPage);
        m_loftrMatchThresholdSpin->setRange(0.0, 1.0);
        m_loftrMatchThresholdSpin->setDecimals(3);
        m_loftrMatchThresholdSpin->setSingleStep(0.01);
        m_loftrMatchThresholdSpin->setValue(0.2);
        m_loftrMatchThresholdSpin->setToolTip(tr("LoFTR 匹配置信度阈值"));
        loftrForm->addRow(tr("匹配阈值:"), m_loftrMatchThresholdSpin);

        loftrLayout->addStretch();
        m_paramStack->addWidget(loftrPage);  // index 3
    }

    // ── Page 4: RoMa ──────────────────────────────────────────
    {
        QWidget* romaPage = new QWidget();
        QVBoxLayout* romaLayout = new QVBoxLayout(romaPage);
        romaLayout->setContentsMargins(0, 4, 0, 4);

        QLabel* romaWarning = new QLabel(tr("⚠ RoMa 直接处理原始图像，不使用已有特征提取结果\n"
                                            "适合大旋转角度的小行星影像"), romaPage);
        romaWarning->setStyleSheet("color: #ff9800; font-weight: bold;");
        romaWarning->setWordWrap(true);
        romaLayout->addWidget(romaWarning);

        QFormLayout* romaForm = new QFormLayout();
        romaLayout->addLayout(romaForm);

        m_romaModelTypeCombo = new QComboBox(romaPage);
        m_romaModelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
        m_romaModelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");
        romaForm->addRow(tr("模型类型:"), m_romaModelTypeCombo);

        m_romaMatchThresholdSpin = new QDoubleSpinBox(romaPage);
        m_romaMatchThresholdSpin->setRange(0.0, 1.0);
        m_romaMatchThresholdSpin->setDecimals(3);
        m_romaMatchThresholdSpin->setSingleStep(0.01);
        m_romaMatchThresholdSpin->setValue(0.05);
        m_romaMatchThresholdSpin->setToolTip(tr("RoMa 匹配置信度阈值"));
        romaForm->addRow(tr("匹配阈值:"), m_romaMatchThresholdSpin);

        m_romaMaxKeypointsSpin = new QSpinBox(romaPage);
        m_romaMaxKeypointsSpin->setRange(100, 50000);
        m_romaMaxKeypointsSpin->setValue(10000);
        m_romaMaxKeypointsSpin->setToolTip(tr("RoMa 最大关键点数"));
        romaForm->addRow(tr("最大关键点数:"), m_romaMaxKeypointsSpin);

        romaLayout->addStretch();
        m_paramStack->addWidget(romaPage);  // index 4
    }

    // ── 高级参数（折叠）：RANSAC 参数，所有算法共用 ──────────────
    auto *advancedToggle = new QToolButton(right);
    advancedToggle->setText(tr("高级参数（RANSAC）"));
    advancedToggle->setCheckable(true);
    advancedToggle->setChecked(false);
    advancedToggle->setArrowType(Qt::RightArrow);
    advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    rightLayout->addWidget(advancedToggle);

    m_advancedGroup = new QGroupBox(right);
    m_advancedGroup->setFlat(true);
    m_advancedGroup->setTitle(QString());
    m_advancedGroup->setVisible(false);
    QFormLayout* advancedForm = new QFormLayout(m_advancedGroup);
    rightLayout->addWidget(m_advancedGroup);

    connect(advancedToggle, &QToolButton::toggled, this, [this, advancedToggle](bool checked)
    {
        m_advancedGroup->setVisible(checked);
        advancedToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    m_outlierReprojSpin = new QDoubleSpinBox(right);
    m_outlierReprojSpin->setRange(0.1, 50.0);
    m_outlierReprojSpin->setDecimals(2);
    m_outlierReprojSpin->setSingleStep(0.1);
    m_outlierReprojSpin->setValue(1.5);
    m_outlierReprojSpin->setToolTip(tr("RANSAC/USAC 重投影误差阈值（像素）\n"
                                       "推荐: 1.0-1.5（高分辨率）/ 1.5-3.0（中等）/ 3.0-6.0（低分辨率）"));
    advancedForm->addRow(tr("RANSAC 阈值(px):"), m_outlierReprojSpin);

    m_outlierConfidenceSpin = new QDoubleSpinBox(right);
    m_outlierConfidenceSpin->setRange(0.50, 0.9999);
    m_outlierConfidenceSpin->setDecimals(4);
    m_outlierConfidenceSpin->setSingleStep(0.0010);
    m_outlierConfidenceSpin->setValue(0.9999);
    m_outlierConfidenceSpin->setToolTip(tr("RANSAC/USAC 置信度 [0.5-0.9999]，推荐 0.9999"));
    advancedForm->addRow(tr("RANSAC 置信度:"), m_outlierConfidenceSpin);

    m_outlierMaxItersSpin = new QSpinBox(right);
    m_outlierMaxItersSpin->setRange(100, 100000);
    m_outlierMaxItersSpin->setValue(10000);
    m_outlierMaxItersSpin->setToolTip(tr("RANSAC/USAC 最大迭代次数，推荐 10000"));
    advancedForm->addRow(tr("RANSAC 最大迭代:"), m_outlierMaxItersSpin);

    m_outlierMinInliersSpin = new QSpinBox(right);
    m_outlierMinInliersSpin->setRange(1, 100000);
    m_outlierMinInliersSpin->setValue(25);
    m_outlierMinInliersSpin->setToolTip(tr("粗差剔除后最少内点数，低于此值则丢弃该匹配对\n推荐: ≥25（航空）/ ≥15（近景）"));
    advancedForm->addRow(tr("最少内点数:"), m_outlierMinInliersSpin);

    // --------------- 系统参数（折叠） ---------------
    auto *systemToggle = new QToolButton(right);
    systemToggle->setText(tr("系统参数"));
    systemToggle->setCheckable(true);
    systemToggle->setChecked(false);
    systemToggle->setArrowType(Qt::RightArrow);
    systemToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    rightLayout->addWidget(systemToggle);

    m_systemGroup = new QGroupBox(right);
    m_systemGroup->setFlat(true);
    m_systemGroup->setTitle(QString());
    m_systemGroup->setVisible(false);
    QFormLayout* systemForm = new QFormLayout(m_systemGroup);
    rightLayout->addWidget(m_systemGroup);

    connect(systemToggle, &QToolButton::toggled, this, [this, systemToggle](bool checked)
    {
        m_systemGroup->setVisible(checked);
        systemToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    // 设备选择
    m_deviceCombo = new QComboBox(right);
    m_deviceCombo->addItems({tr("CUDA（如可用）"), tr("CPU")});
    m_deviceCombo->setToolTip(tr("计算设备选择\n"
                                 "CUDA需要GPU支持"));
    systemForm->addRow(tr("计算设备:"), m_deviceCombo);

    // CPU线程数
    m_numThreadsSpin = new QSpinBox(right);
    m_numThreadsSpin->setRange(-1, 64);
    m_numThreadsSpin->setValue(-1);
    m_numThreadsSpin->setSpecialValueText(tr("自动"));
    m_numThreadsSpin->setToolTip(tr("CPU线程数（仅CPU模式有效）\n"
                                    "-1 表示自动检测"));
    systemForm->addRow(tr("CPU线程数:"), m_numThreadsSpin);

    // CUDA 并行对数
    m_cudaParallelSpin = new QSpinBox(right);
    m_cudaParallelSpin->setRange(1, 8);
    m_cudaParallelSpin->setValue(1);
    m_cudaParallelSpin->setToolTip(tr("CUDA模式下同时处理的影像对数\n"
                                       "每对独立持有一个模型实例，每实例约占显存 200-400 MB\n"
                                       "请确认 GPU 显存充裕再增大此参数"));
    systemForm->addRow(tr("CUDA并行对数:"), m_cudaParallelSpin);

    // --------------- 调试参数（折叠） ---------------
    auto *debugToggle = new QToolButton(right);
    debugToggle->setText(tr("调试参数"));
    debugToggle->setCheckable(true);
    debugToggle->setChecked(false);
    debugToggle->setArrowType(Qt::RightArrow);
    debugToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    rightLayout->addWidget(debugToggle);

    m_debugGroup = new QGroupBox(right);
    m_debugGroup->setFlat(true);
    m_debugGroup->setTitle(QString());
    m_debugGroup->setVisible(false);
    QFormLayout* debugForm = new QFormLayout(m_debugGroup);
    rightLayout->addWidget(m_debugGroup);

    connect(debugToggle, &QToolButton::toggled, this, [this, debugToggle](bool checked)
    {
        m_debugGroup->setVisible(checked);
        debugToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    // 保存CSV
    m_saveCsvChk = new QCheckBox(tr("保存匹配结果为CSV"), right);
    m_saveCsvChk->setToolTip(tr("将匹配关系导出为CSV格式（便于调试）"));
    debugForm->addRow(m_saveCsvChk);

    // 保存可视化
    m_saveVisChk = new QCheckBox(tr("保存匹配可视化图像"), right);
    m_saveVisChk->setToolTip(tr("生成带匹配连线的可视化图像"));
    debugForm->addRow(m_saveVisChk);

    // 详细日志
    m_verboseChk = new QCheckBox(tr("详细日志输出"), right);
    m_verboseChk->setToolTip(tr("打印详细的匹配过程信息"));
    debugForm->addRow(m_verboseChk);

    rightLayout->addStretch();
    
    topSplit->addWidget(right);
    topSplit->setStretchFactor(0, 3);
    topSplit->setStretchFactor(1, 2);
    
    mainLayout->addWidget(topSplit);

    // ==================== 底部按钮 ====================
    QHBoxLayout* btnRow = new QHBoxLayout();
    btnRow->addStretch();
    
    m_resetBtn = new QPushButton(tr("重置默认值"), this);
    m_viewMatchesBtn = new QPushButton(tr("查看匹配"), this);
    m_runBtn = new QPushButton(tr("运行匹配"), this);
    m_runBtn->setDefault(true);
    m_cancelBtn = new QPushButton(tr("取消"), this);
    
    btnRow->addWidget(m_resetBtn);
    btnRow->addWidget(m_viewMatchesBtn);
    btnRow->addWidget(m_runBtn);
    btnRow->addWidget(m_cancelBtn);
    
    mainLayout->addLayout(btnRow);
}

void FeatureMatchingDialog::setupConnections() 
{
    // 全选/清除按钮
    connect(m_selectAllBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onSelectAll);
    connect(m_deselectAllBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onDeselectAll);
    // 文件操作
    connect(m_addLisBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onAddLisFile);
    connect(m_clearLisBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onClearLis);
    connect(m_generatePairsBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onGeneratePairs);
    connect(m_browseOutBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onBrowseOutput);
    
    // 底部按钮
    connect(m_runBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onRun);
    connect(m_viewMatchesBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onViewMatches);
    connect(m_cancelBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onCancel);
    connect(m_resetBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onResetDefaults);
    
    // 算法切换：更新参数控件启用状态 + 持久化
    connect(m_matchAlgorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int){ onAlgorithmOrFeatureChanged(); });
    connect(m_matchAlgorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 特征类型切换
    connect(m_featureSuffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int){ updatePreview(); });
    connect(m_featureSuffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 初始状态
    onAlgorithmOrFeatureChanged();
    connect(m_modelTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_matchThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_maxKeypointsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierReprojSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierConfidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierMaxItersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierMinInliersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_inputWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_inputHeightSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_cudaParallelSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_sinkhornIterSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_batchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // LightGlue 面板控件
    connect(m_lgMatchThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_lgBatchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_lgInputWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_lgInputHeightSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 影像列表 checkbox 变化时保存
    connect(m_imageList, &QListWidget::itemChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 输出路径、lis 文件改变时保存
    connect(m_outputLine, &QLineEdit::textChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_lisFileLine, &QLineEdit::textChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
}

void FeatureMatchingDialog::onSelectAll()
{
    for (int i = 0; i < m_imageList->count(); ++i) {
        m_imageList->item(i)->setCheckState(Qt::Checked);
    }
    emitSettingsNow();
}

void FeatureMatchingDialog::onDeselectAll()
{
    for (int i = 0; i < m_imageList->count(); ++i) {
        m_imageList->item(i)->setCheckState(Qt::Unchecked);
    }
    emitSettingsNow();
}

void FeatureMatchingDialog::onAddLisFile()
{
    QString path = QFileDialog::getOpenFileName(this, tr("选择lis文件"),
                                                QString(),
                                                tr("lis文件 (*.lis *.txt);;所有文件 (*.*)"));
    if (!path.isEmpty()) {
        m_lisFileLine->setText(path);
        updatePreview();
    }
}

void FeatureMatchingDialog::onClearLis()
{
    m_lisFileLine->clear();
    updatePreview();
}

void FeatureMatchingDialog::onGeneratePairs()
{
    QString lisPath = m_lisFileLine->text().trimmed();

    if (!lisPath.isEmpty()) {
        m_currentPairs = parseLisFile(lisPath);
    } else {
        m_currentPairs = generateAllPairs();
    }

    updatePreview();
    emitSettingsNow();
}

void FeatureMatchingDialog::onBrowseOutput()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("选择输出目录"));
    if (!dir.isEmpty()) {
        m_outputLine->setText(dir);
    }
}

void FeatureMatchingDialog::onRun()
{
    if (m_currentPairs.isEmpty()) {
        QMessageBox::warning(this, tr("警告"), 
                           tr("请先生成匹配对列表"));
        return;
    }
    
    QJsonObject config = collectSettings();
    emit runRequested(config, m_currentPairs);
    accept();
}

void FeatureMatchingDialog::onCancel()
{
    reject();
}

void FeatureMatchingDialog::onResetDefaults()
{
    m_matchAlgorithmCombo->setCurrentIndex(0);  // superglue
    m_modelTypeCombo->setCurrentIndex(0);  // outdoor
    m_outlierMethodCombo->setCurrentIndex(2);  // Fundamental USAC_MAGSAC（推荐默认，最优粗差剔除）
    m_matchThresholdSpin->setValue(0.15);
    m_maxKeypointsSpin->setValue(-1);
    m_sinkhornIterSpin->setValue(150);
    m_batchSizeSpin->setValue(1);
    m_outlierReprojSpin->setValue(1.5);
    m_outlierConfidenceSpin->setValue(0.9999);
    m_outlierMaxItersSpin->setValue(10000);
    m_outlierMinInliersSpin->setValue(25);
    m_inputWidthSpin->setValue(-1);
    m_inputHeightSpin->setValue(-1);
    m_lgMatchThresholdSpin->setValue(0.15);
    m_lgBatchSizeSpin->setValue(1);
    m_lgInputWidthSpin->setValue(-1);
    m_lgInputHeightSpin->setValue(-1);
    m_deviceCombo->setCurrentIndex(0);
    m_numThreadsSpin->setValue(-1);
    m_cudaParallelSpin->setValue(1);
    m_saveCsvChk->setChecked(false);
    m_saveVisChk->setChecked(false);
    m_verboseChk->setChecked(false);

    emitSettingsNow();
}

void FeatureMatchingDialog::onAlgorithmChanged(int)
{
    onAlgorithmOrFeatureChanged();
}

void FeatureMatchingDialog::setAvailableFeatureSuffixes(const QStringList &suffixes)
{
    m_featureSuffixCombo->blockSignals(true);
    m_featureSuffixCombo->clear();
    if (suffixes.size() > 1) {
        m_featureSuffixCombo->addItem(tr("所有特征类型"), QStringLiteral("__all__"));
    }
    for (const auto &s : suffixes)
        m_featureSuffixCombo->addItem(s, s);
    m_featureSuffixCombo->blockSignals(false);

    bool visible = !suffixes.isEmpty();
    m_featureSuffixLabel->setVisible(visible);
    m_featureSuffixCombo->setVisible(visible);
}

QString FeatureMatchingDialog::selectedFeatureSuffix() const
{
    QVariant data = m_featureSuffixCombo->currentData();
    return data.isValid() ? data.toString() : m_featureSuffixCombo->currentText();
}

void FeatureMatchingDialog::onAlgorithmOrFeatureChanged()
{
    const QString algo = m_matchAlgorithmCombo->currentData().toString();
    const bool isE2E = xjw::feature_match::isEndToEndAlgorithm(algo);

    // 算法参数面板切换
    if (algo == "superglue")
        m_paramStack->setCurrentIndex(0);
    else if (algo == "lightglue")
        m_paramStack->setCurrentIndex(1);
    else if (algo == "loftr")
        m_paramStack->setCurrentIndex(3);
    else if (algo == "roma")
        m_paramStack->setCurrentIndex(4);
    else
        m_paramStack->setCurrentIndex(2);

    // 更新特征后缀选择器
    if (!isE2E) {
        setAvailableFeatureSuffixes(
            xjw::feature_match::compatibleFeatureSuffixes(algo));
    } else {
        setAvailableFeatureSuffixes({});
    }

    updatePreview();
}

void FeatureMatchingDialog::onViewMatches()
{
    emit viewMatchesRequested();
}

void FeatureMatchingDialog::applySettings(const QJsonObject &settings)
{
    bool block = blockSignals(true);

    const QString matchAlgorithm = settings.value("match_algorithm").toString("superglue");
    const int matchAlgorithmIndex = m_matchAlgorithmCombo->findData(matchAlgorithm);
    if (matchAlgorithmIndex >= 0)
        m_matchAlgorithmCombo->setCurrentIndex(matchAlgorithmIndex);

    const QString outlierMethod = settings.value("outlier_method").toString("fundamental_usac_magsac");
    const int outlierIdx = m_outlierMethodCombo->findData(outlierMethod);
    if (outlierIdx >= 0) m_outlierMethodCombo->setCurrentIndex(outlierIdx);

    m_maxKeypointsSpin->setValue(settings.value("max_keypoints").toInt(-1));

    // SuperGlue 面板
    const QString modelType = settings.value("model_type").toString("outdoor");
    const int modelIdx = m_modelTypeCombo->findData(modelType);
    if (modelIdx >= 0) m_modelTypeCombo->setCurrentIndex(modelIdx);
    m_matchThresholdSpin->setValue(settings.value("match_threshold").toDouble(0.15));
    m_sinkhornIterSpin->setValue(settings.value("sinkhorn_iterations").toInt(150));
    m_batchSizeSpin->setValue(settings.value("batch_size").toInt(1));
    m_inputWidthSpin->setValue(settings.value("input_width").toInt(-1));
    m_inputHeightSpin->setValue(settings.value("input_height").toInt(-1));

    // LightGlue 面板（独立 key，避免与 SuperGlue 冲突）
    m_lgMatchThresholdSpin->setValue(settings.value("lg_match_threshold").toDouble(
        settings.value("match_threshold").toDouble(0.15)));  // 兼容旧格式
    m_lgBatchSizeSpin->setValue(settings.value("lg_batch_size").toInt(
        settings.value("batch_size").toInt(1)));
    m_lgInputWidthSpin->setValue(settings.value("lg_input_width").toInt(
        settings.value("input_width").toInt(-1)));
    m_lgInputHeightSpin->setValue(settings.value("lg_input_height").toInt(
        settings.value("input_height").toInt(-1)));

    // LoFTR 面板
    if (m_loftrModelTypeCombo) {
        int idx = m_loftrModelTypeCombo->findData(settings.value("loftr_model_type").toString("outdoor"));
        m_loftrModelTypeCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
    if (m_loftrMatchThresholdSpin)
        m_loftrMatchThresholdSpin->setValue(settings.value("loftr_match_threshold").toDouble(0.2));

    // RoMa 面板
    if (m_romaModelTypeCombo) {
        int idx = m_romaModelTypeCombo->findData(settings.value("roma_model_type").toString("outdoor"));
        m_romaModelTypeCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
    if (m_romaMatchThresholdSpin)
        m_romaMatchThresholdSpin->setValue(settings.value("roma_match_threshold").toDouble(0.05));
    if (m_romaMaxKeypointsSpin)
        m_romaMaxKeypointsSpin->setValue(settings.value("roma_max_keypoints").toInt(10000));

    // 高级参数
    m_outlierReprojSpin->setValue(settings.value("outlier_reproj_threshold").toDouble(1.5));
    m_outlierConfidenceSpin->setValue(settings.value("outlier_confidence").toDouble(0.9999));
    m_outlierMaxItersSpin->setValue(settings.value("outlier_max_iters").toInt(10000));
    m_outlierMinInliersSpin->setValue(settings.value("outlier_min_inliers").toInt(25));

    // 系统参数
    m_deviceCombo->setCurrentIndex(settings.value("use_cuda").toBool(true) ? 0 : 1);
    m_numThreadsSpin->setValue(settings.value("num_threads").toInt(-1));
    m_cudaParallelSpin->setValue(settings.value("cuda_parallel_pairs").toInt(1));

    // 调试参数
    m_saveCsvChk->setChecked(settings.value("save_csv").toBool(false));
    m_saveVisChk->setChecked(settings.value("save_visualization").toBool(false));
    m_verboseChk->setChecked(settings.value("verbose").toBool(false));

    m_outputLine->setText(settings.value("output_dir").toString());
    m_lisFileLine->setText(settings.value("lis_file").toString());

    // 恢复已生成的匹配对
    const QJsonArray savedPairs = settings.value(QStringLiteral("generated_pairs")).toArray();
    if (!savedPairs.isEmpty()) {
        m_currentPairs.clear();
        m_currentPairs.reserve(savedPairs.size());
        for (const QJsonValue &v : savedPairs)
            m_currentPairs.append(v.toString());
        updatePreview();
    }

    blockSignals(block);
    onAlgorithmOrFeatureChanged();

    // 恢复特征后缀选择（必须在 onAlgorithmOrFeatureChanged 之后，因为该函数会填充后缀列表）
    const QString featureSuffix = settings.value("feature_suffix").toString();
    if (!featureSuffix.isEmpty() && m_featureSuffixCombo->count() > 0) {
        int idx = m_featureSuffixCombo->findText(featureSuffix);
        if (idx >= 0)
            m_featureSuffixCombo->setCurrentIndex(idx);
    }
}

void FeatureMatchingDialog::setProjectImages(const QStringList &imagePaths)
{
    m_imageList->clear();
    for (const QString &path : imagePaths) {
        QFileInfo fi(path);
        QListWidgetItem *item = new QListWidgetItem(fi.fileName());
        item->setData(Qt::UserRole, path);
        item->setCheckState(Qt::Checked);
        m_imageList->addItem(item);
    }
}

void FeatureMatchingDialog::updatePreview()
{
    QString preview;
    preview += tr("匹配对数量: %1\n\n").arg(m_currentPairs.size());
    
    int maxShow = 20;
    for (int i = 0; i < qMin(m_currentPairs.size(), maxShow); ++i) {
        preview += m_currentPairs[i] + "\n";
    }
    
    if (m_currentPairs.size() > maxShow) {
        preview += tr("... （还有 %1 对）\n").arg(m_currentPairs.size() - maxShow);
    }
    
    m_pairPreview->setPlainText(preview);
}

void FeatureMatchingDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

QJsonObject FeatureMatchingDialog::collectSettings() const
{
    QJsonObject obj;

    // 基础参数
    const QString algo = m_matchAlgorithmCombo->currentData().toString();
    obj["match_algorithm"] = algo;
    obj["feature_suffix"] = selectedFeatureSuffix();
    obj["outlier_method"] = m_outlierMethodCombo->currentData().toString();
    obj["max_keypoints"] = m_maxKeypointsSpin->value();

    // 算法专属参数
    if (algo == "superglue") {
        obj["model_type"] = m_modelTypeCombo->currentData().toString();
        obj["match_threshold"] = m_matchThresholdSpin->value();
        obj["sinkhorn_iterations"] = m_sinkhornIterSpin->value();
        obj["batch_size"] = m_batchSizeSpin->value();
        obj["input_width"] = m_inputWidthSpin->value();
        obj["input_height"] = m_inputHeightSpin->value();
    } else if (algo == "lightglue") {
        obj["model_type"] = "outdoor";
        obj["lg_match_threshold"] = m_lgMatchThresholdSpin->value();
        obj["lg_batch_size"] = m_lgBatchSizeSpin->value();
        obj["lg_input_width"] = m_lgInputWidthSpin->value();
        obj["lg_input_height"] = m_lgInputHeightSpin->value();
    } else if (algo == "loftr") {
        obj["loftr_model_type"] = m_loftrModelTypeCombo->currentData().toString();
        obj["loftr_match_threshold"] = m_loftrMatchThresholdSpin->value();
    } else if (algo == "roma") {
        obj["roma_model_type"] = m_romaModelTypeCombo->currentData().toString();
        obj["roma_match_threshold"] = m_romaMatchThresholdSpin->value();
        obj["roma_max_keypoints"] = m_romaMaxKeypointsSpin->value();
    } else {
        // 传统算法无这些参数，填充默认值
        obj["model_type"] = "outdoor";
        obj["match_threshold"] = 0.15;
        obj["sinkhorn_iterations"] = 0;
        obj["batch_size"] = 1;
        obj["input_width"] = -1;
        obj["input_height"] = -1;
    }

    // 高级参数（RANSAC）
    obj["outlier_reproj_threshold"] = m_outlierReprojSpin->value();
    obj["outlier_confidence"] = m_outlierConfidenceSpin->value();
    obj["outlier_max_iters"] = m_outlierMaxItersSpin->value();
    obj["outlier_min_inliers"] = m_outlierMinInliersSpin->value();

    // 系统参数
    obj["use_cuda"] = (m_deviceCombo->currentIndex() == 0);
    obj["num_threads"] = m_numThreadsSpin->value();
    obj["cuda_parallel_pairs"] = m_cudaParallelSpin->value();

    // 调试参数
    obj["save_csv"] = m_saveCsvChk->isChecked();
    obj["save_visualization"] = m_saveVisChk->isChecked();
    obj["verbose"] = m_verboseChk->isChecked();

    // 输出设置
    obj["output_dir"] = m_outputLine->text();
    obj["lis_file"] = m_lisFileLine->text();

    // 保存已生成的匹配对列表
    QJsonArray pairsArr;
    for (const QString &p : m_currentPairs)
        pairsArr.append(p);
    obj[QStringLiteral("generated_pairs")] = pairsArr;

    return obj;
}

QStringList FeatureMatchingDialog::parseLisFile(const QString &lisPath) const
{
    QStringList pairs;
    QFile file(lisPath);
    
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(const_cast<FeatureMatchingDialog*>(this), tr("错误"),
                           tr("无法打开lis文件: %1").arg(lisPath));
        return pairs;
    }
    
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#")) continue;
        
        QStringList parts = line.split(QRegularExpression(QStringLiteral("\\s+")), Qt::SkipEmptyParts);
        if (parts.size() >= 2) {
            QString img1 = QFileInfo(parts[0]).fileName();
            QString img2 = QFileInfo(parts[1]).fileName();
            pairs.append(QString("%1__%2").arg(img1, img2));
        }
    }
    
    file.close();
    return pairs;
}

QStringList FeatureMatchingDialog::generateAllPairs() const
{
    QStringList pairs;
    QStringList selected;

    // 收集选中的影像
    for (int i = 0; i < m_imageList->count(); ++i) {
        QListWidgetItem *item = m_imageList->item(i);
        if (item->checkState() == Qt::Checked) {
            QString baseName = QFileInfo(item->data(Qt::UserRole).toString()).completeBaseName();
            selected.append(baseName);
        }
    }

    // 生成两两匹配对
    for (int i = 0; i < selected.size(); ++i) {
        for (int j = i + 1; j < selected.size(); ++j) {
            pairs.append(QString("%1__%2").arg(selected[i], selected[j]));
        }
    }

    return pairs;
}
