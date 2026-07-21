// =============================================================================
// 文件: OverlapAnalysisDialog.cpp
// 说明: OverlapAnalysisDialog 的实现。
//       界面构建、影像加载、DEM 加载、调用 OverlapAnalyzer 进行分析，
//       以及将结果填入表格。
// =============================================================================
#include "OverlapAnalysisDialog.h"

#include "ProjectManager.h"
#include "project/ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "OverlapAnalyzer.h"
#include "ui_OverlapAnalysisDialog.h"
#include "io/PathIO.h"

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
    , _projectManager(projectManager)
{
    Ui::OverlapAnalysisDialog ui;
    ui.setupUi(this);

    _imageList = ui.m_imageList;
    _demPathEdit = ui.m_demPathEdit;
    _useFixedZCheck = ui.m_useFixedZCheck;
    _fixedZSpin = ui.m_fixedZSpin;
    _neighborSpin = ui.m_neighborSpin;
    _summaryLabel = ui.m_summaryLabel;
    _resultTable = ui.m_resultTable;

    _resultTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    connect(ui.browseBtn, &QPushButton::clicked, this, &OverlapAnalysisDialog::browseDemPath);
    connect(ui.runBtn, &QPushButton::clicked, this, &OverlapAnalysisDialog::runAnalysis);

    loadProjectImages();
}

// loadProjectImages: 从项目元数据中读取影像列表并填充界面复选框列表
// 每项以文件名作为显示文本，完整路径存储在 UserRole 中，默认全部勾选
void OverlapAnalysisDialog::loadProjectImages()
{
    _imageList->clear();
    if (!_projectManager)
    {
        return;
    }

    const QJsonArray images = xjw::common::project::projectImageEntries(_projectManager->currentMeta());
    for (const QJsonValue &v : images)
    {
        const QJsonObject obj = v.toObject();
        const QString path = obj.value(QStringLiteral("path")).toString();
        if (path.isEmpty())
        {
            continue;
        }

        // 列表项显示文件名，实际路径存于 UserRole，默认勾选
        auto *item = new QListWidgetItem(QFileInfo(path).fileName(), _imageList);
        item->setData(Qt::UserRole, path);
        item->setCheckState(Qt::Checked);
    }
}

// browseDemPath: 弹出文件选择对话框，让用户选择 DEM（XYZ 格式）文件路径
void OverlapAnalysisDialog::browseDemPath()
{
    const QString path = QFileDialog::getOpenFileName(this,
                                                      tr("选择 DEM XYZ"),
                                                      QString(),
                                                      tr("XYZ (*.xyz *.txt *.csv);;All files (*.*)"));
    if (!path.isEmpty())
    {
        _demPathEdit->setText(path);
    }
}

// runAnalysis: 执行影像重叠度分析的主逻辑
// 流程：
//   1. 从项目元数据获取影像列表
//   2. 遍历界面上勾选的影像，解析 Camera 参数
//   3. （可选）加载 DEM XYZ 文件或使用固定高程值
//   4. 调用 OverlapAnalyzer::analyze 计算两两影像重叠评分
//   5. 将结果填入 _resultTable 表格
void OverlapAnalysisDialog::runAnalysis()
{
    if (!_projectManager)
    {
        return;
    }

    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::common::project::projectImageMetaByPath(_projectManager->currentMeta());

    std::vector<xjw::OverlapImageInput> inputs;
    QStringList inputNames;
    for (int i = 0; i < _imageList->count(); ++i)
    {
        QListWidgetItem *it = _imageList->item(i);
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
        if (!xjw::common::project::imageCameraFromEntry(imageMetaByPath.value(path), &cam))
        {
            continue;
        }

        xjw::OverlapImageInput one;
        one.imagePath = xjw::common::io::toUtf8Path(path);
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
    const bool useFixedZ = _useFixedZCheck->isChecked();
    if (!useFixedZ)
    {
        const QString demPath = _demPathEdit->text().trimmed();
        if (demPath.isEmpty())
        {
            QMessageBox::warning(this, tr("提示"), tr("请提供 DEM，或勾选固定高程模式"));
            return;
        }
        std::string demErr;
        if (!dem.loadFromXYZ(xjw::common::io::toUtf8Path(demPath), &demErr))
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
                                       _fixedZSpin->value(),
                                       _neighborSpin->value(),
                                       &result,
                                       &err))
    {
        QMessageBox::warning(this, tr("提示"), QString::fromStdString(err));
        return;
    }

    // 更新摘要标签（当前为 result.detail 全文）
    _summaryLabel->setText(QString::fromStdString(result.detail));
    // 按匹配对数量设置表格行数，逐行填充分析结果
    _resultTable->setRowCount(static_cast<int>(result.pairs.size()));
    for (int i = 0; i < static_cast<int>(result.pairs.size()); ++i)
    {
        const auto &p = result.pairs[static_cast<size_t>(i)];
        // 获取两张影像的文件名（仅用于显示）
        const QString nameA = QFileInfo(
            QString::fromStdString(inputs[static_cast<size_t>(p.indexA)].imagePath)).fileName();
        const QString nameB = QFileInfo(
            QString::fromStdString(inputs[static_cast<size_t>(p.indexB)].imagePath)).fileName();
        _resultTable->setItem(i, 0, new QTableWidgetItem(nameA));
        _resultTable->setItem(i, 1, new QTableWidgetItem(nameB));
        _resultTable->setItem(i, 2, new QTableWidgetItem(QString::number(p.centerDistance, 'f', 3)));
        _resultTable->setItem(i, 3, new QTableWidgetItem(QString::number(p.overlapScore, 'f', 4)));
        // overlapScore > 0 表示存在实际重叠区域
        _resultTable->setItem(i, 4, new QTableWidgetItem(p.overlapScore > 0.0 ? tr("是") : tr("否")));
    }
}
