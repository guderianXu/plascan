// =============================================================================
// 文件: OverlapAnalysisDialog.cpp
// 说明: OverlapAnalysisDialog 的实现。
//       界面构建、影像加载、DEM 加载、调用 OverlapAnalyzer 进行分析，
//       以及将结果填入表格。
// =============================================================================
#include "OverlapAnalysisDialog.h"

#include "ProjectManager.h"
#include "ProjectSupportUtils.h"
#include "OverlapAnalyzer.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QGridLayout>
#include <QListWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QLabel>
#include <QFileDialog>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonObject>
#include <QJsonArray>
#include <QMessageBox>

// 构造函数：初始化界面布局、控件，并从项目中加载影像列表
OverlapAnalysisDialog::OverlapAnalysisDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , m_projectManager(projectManager)
{
    setWindowTitle(tr("影像重叠度获取"));
    resize(900, 660);

    auto *layout = new QVBoxLayout(this);

    auto *inputBox = new QGroupBox(tr("输入参数"), this);
    auto *grid = new QGridLayout(inputBox);

    m_imageList = new QListWidget(inputBox);
    m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    m_demPathEdit = new QLineEdit(inputBox);
    auto *browseBtn = new QPushButton(tr("浏览"), inputBox);

    m_useFixedZCheck = new QCheckBox(tr("使用假定高程 Z"), inputBox);
    m_fixedZSpin = new QDoubleSpinBox(inputBox);
    m_fixedZSpin->setRange(-100000.0, 100000.0);
    m_fixedZSpin->setDecimals(3);
    m_fixedZSpin->setValue(0.0);

    m_neighborSpin = new QDoubleSpinBox(inputBox);
    m_neighborSpin->setRange(0.1, 20.0);
    m_neighborSpin->setDecimals(2);
    m_neighborSpin->setValue(1.5);

    auto *runBtn = new QPushButton(tr("计算重叠度"), inputBox);

    grid->addWidget(new QLabel(tr("影像列表"), inputBox), 0, 0);
    grid->addWidget(m_imageList, 1, 0, 4, 2);
    grid->addWidget(new QLabel(tr("DEM (XYZ)"), inputBox), 0, 2);
    grid->addWidget(m_demPathEdit, 0, 3);
    grid->addWidget(browseBtn, 0, 4);
    grid->addWidget(m_useFixedZCheck, 1, 2, 1, 2);
    grid->addWidget(new QLabel(tr("固定 Z"), inputBox), 2, 2);
    grid->addWidget(m_fixedZSpin, 2, 3);
    grid->addWidget(new QLabel(tr("邻域系数"), inputBox), 3, 2);
    grid->addWidget(m_neighborSpin, 3, 3);
    grid->addWidget(runBtn, 4, 4);

    m_summaryLabel = new QLabel(this);
    m_resultTable = new QTableWidget(this);
    m_resultTable->setColumnCount(5);
    m_resultTable->setHorizontalHeaderLabels(
        {tr("影像A"), tr("影像B"), tr("中心距(m)"), tr("重叠评分"), tr("是否重叠")});
    m_resultTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_resultTable->setEditTriggers(QAbstractItemView::NoEditTriggers);

    layout->addWidget(inputBox);
    layout->addWidget(m_summaryLabel);
    layout->addWidget(m_resultTable, 1);

    connect(browseBtn, &QPushButton::clicked, this, &OverlapAnalysisDialog::browseDemPath);
    connect(runBtn, &QPushButton::clicked, this, &OverlapAnalysisDialog::runAnalysis);

    loadProjectImages();
}

// loadProjectImages: 从项目元数据中读取影像列表并填充界面复选框列表
// 每项以文件名作为显示文本，完整路径存储在 UserRole 中，默认全部勾选
void OverlapAnalysisDialog::loadProjectImages()
{
    m_imageList->clear();
    if (!m_projectManager)
    {
        return;
    }

    const QJsonArray images = xjw::gui::project::projectImageEntries(m_projectManager->currentMeta());
    for (const QJsonValue &v : images)
    {
        const QJsonObject obj = v.toObject();
        const QString path = obj.value(QStringLiteral("path")).toString();
        if (path.isEmpty())
        {
            continue;
        }

        // 列表项显示文件名，实际路径存于 UserRole，默认勾选
        auto *item = new QListWidgetItem(QFileInfo(path).fileName(), m_imageList);
        item->setData(Qt::UserRole, path);
        item->setCheckState(Qt::Checked);
    }
}

// browseDemPath: 弹出文件选择对话框，让用户选择 DEM（XYZ 格式）文件路径
void OverlapAnalysisDialog::browseDemPath()
{
    const QString path = QFileDialog::getOpenFileName(this, tr("选择 DEM XYZ"), QString(), tr("XYZ (*.xyz *.txt *.csv);;All files (*.*)"));
    if (!path.isEmpty())
    {
        m_demPathEdit->setText(path);
    }
}

// runAnalysis: 执行影像重叠度分析的主逻辑
// 流程：
//   1. 从项目元数据获取影像列表
//   2. 遍历界面上勾选的影像，解析 Camera 参数
//   3. （可选）加载 DEM XYZ 文件或使用固定高程值
//   4. 调用 OverlapAnalyzer::analyze 计算两两影像重叠评分
//   5. 将结果填入 m_resultTable 表格
void OverlapAnalysisDialog::runAnalysis()
{
    if (!m_projectManager)
    {
        return;
    }

    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::gui::project::projectImageMetaByPath(m_projectManager->currentMeta());

    std::vector<xjw::OverlapImageInput> inputs;
    QStringList inputNames;
    for (int i = 0; i < m_imageList->count(); ++i)
    {
        QListWidgetItem *it = m_imageList->item(i);
        if (!it || it->checkState() != Qt::Checked)
        {
            continue;
        }
        const QString path = it->data(Qt::UserRole).toString();
        if (!imageMetaByPath.contains(path))
        {
            continue;
        }

        xjw::Camera cam;
        if (!xjw::gui::project::imageCameraFromEntry(imageMetaByPath.value(path), &cam))
        {
            continue;
        }

        xjw::OverlapImageInput one;
        one.imagePath = path.toStdString();
        one.camera = cam;

        QImageReader reader(path);
        const QSize sz = reader.size();
        one.width = sz.width();
        one.height = sz.height();

        inputs.push_back(one);
        inputNames.push_back(path);
    }

    if (inputs.size() < 2)
    {
        QMessageBox::warning(this, tr("提示"), tr("至少勾选两张带相机参数的影像"));
        return;
    }

    xjw::DemSurface dem;
    const bool useFixedZ = m_useFixedZCheck->isChecked();
    if (!useFixedZ)
    {
        const QString demPath = m_demPathEdit->text().trimmed();
        if (demPath.isEmpty())
        {
            QMessageBox::warning(this, tr("提示"), tr("请提供 DEM，或勾选固定高程模式"));
            return;
        }
        std::string demErr;
        if (!dem.loadFromXYZ(demPath.toStdString(), &demErr))
        {
            QMessageBox::warning(this, tr("提示"), QString::fromStdString(demErr));
            return;
        }
    }

    xjw::OverlapAnalysisResult result;
    std::string err;
    if (!xjw::OverlapAnalyzer::analyze(inputs,
                                       useFixedZ ? nullptr : &dem,
                                       useFixedZ,
                                       m_fixedZSpin->value(),
                                       m_neighborSpin->value(),
                                       &result,
                                       &err))
    {
        QMessageBox::warning(this, tr("提示"), QString::fromStdString(err));
        return;
    }

    // 更新摘要标签（当前为 result.detail 全文）
    m_summaryLabel->setText(QString::fromStdString(result.detail));
    // 按匹配对数量设置表格行数，逐行填充分析结果
    m_resultTable->setRowCount(static_cast<int>(result.pairs.size()));
    for (int i = 0; i < static_cast<int>(result.pairs.size()); ++i)
    {
        const auto &p = result.pairs[static_cast<size_t>(i)];
        // 获取两张影像的文件名（仅用于显示）
        const QString nameA = QFileInfo(
            QString::fromStdString(inputs[static_cast<size_t>(p.indexA)].imagePath)).fileName();
        const QString nameB = QFileInfo(
            QString::fromStdString(inputs[static_cast<size_t>(p.indexB)].imagePath)).fileName();
        m_resultTable->setItem(i, 0, new QTableWidgetItem(nameA));
        m_resultTable->setItem(i, 1, new QTableWidgetItem(nameB));
        m_resultTable->setItem(i, 2, new QTableWidgetItem(QString::number(p.centerDistance, 'f', 3)));
        m_resultTable->setItem(i, 3, new QTableWidgetItem(QString::number(p.overlapScore, 'f', 4)));
        // overlapScore > 0 表示存在实际重叠区域
        m_resultTable->setItem(i, 4, new QTableWidgetItem(p.overlapScore > 0.0 ? tr("是") : tr("否")));
    }
}
