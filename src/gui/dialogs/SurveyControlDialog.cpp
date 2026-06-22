#include "SurveyControlDialog.h"

#include <QDialogButtonBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QVBoxLayout>

#include <cmath>

namespace
{

QString numberText(double value)
{
    if (!std::isfinite(value))
    {
        return QString();
    }
    return QString::number(value, 'f', 4);
}

QString jsonNumberText(const QJsonObject &object, const QString &key)
{
    const QJsonValue value = object.value(key);
    if (!value.isDouble())
    {
        return QString();
    }
    return numberText(value.toDouble());
}

QString residualText(const QJsonObject &object)
{
    const QJsonObject residual = object.value(QStringLiteral("residual")).toObject();
    if (residual.contains(QStringLiteral("total_m")))
    {
        return jsonNumberText(residual, QStringLiteral("total_m"));
    }
    if (object.contains(QStringLiteral("residual_m")))
    {
        return jsonNumberText(object, QStringLiteral("residual_m"));
    }
    return QString();
}

void setReadOnlyItem(QTableWidget *table, int row, int column, const QString &text)
{
    auto *item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    table->setItem(row, column, item);
}

} // namespace

SurveyControlDialog::SurveyControlDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    refreshTables();
}

void SurveyControlDialog::setSurveyControlMetadata(const QJsonObject &metadata)
{
    m_metadata = metadata;
    refreshTables();
}

void SurveyControlDialog::setStatusMessage(const QString &message)
{
    if (m_statusLabel)
    {
        m_statusLabel->setText(message);
    }
}

void SurveyControlDialog::setupUi()
{
    setWindowTitle(tr("测绘控制"));
    resize(980, 620);

    auto *root = new QVBoxLayout(this);

    auto *summaryBox = new QGroupBox(tr("项目控制数据"), this);
    auto *summaryLayout = new QGridLayout(summaryBox);
    m_summaryLabel = new QLabel(summaryBox);
    m_summaryLabel->setObjectName(QStringLiteral("surveyControlSummaryLabel"));
    m_sourceLabel = new QLabel(summaryBox);
    m_sourceLabel->setObjectName(QStringLiteral("surveyControlSourceLabel"));
    m_sourceLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_statusLabel = new QLabel(summaryBox);
    m_statusLabel->setObjectName(QStringLiteral("surveyControlStatusLabel"));
    m_statusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_importCsvButton = new QPushButton(tr("导入 CSV..."), summaryBox);
    m_importCsvButton->setObjectName(QStringLiteral("surveyControlImportCsvButton"));
    m_importCsvButton->setToolTip(tr("导入包含控制点、检查点和比例尺的 CSV 文件"));
    connect(m_importCsvButton, &QPushButton::clicked, this, &SurveyControlDialog::importCsvRequested);

    summaryLayout->addWidget(m_summaryLabel, 0, 0);
    summaryLayout->addWidget(m_importCsvButton, 0, 1, Qt::AlignRight);
    summaryLayout->addWidget(m_sourceLabel, 1, 0, 1, 2);
    summaryLayout->addWidget(m_statusLabel, 2, 0, 1, 2);
    summaryLayout->setColumnStretch(0, 1);
    root->addWidget(summaryBox);

    auto *tabs = new QTabWidget(this);
    tabs->setObjectName(QStringLiteral("surveyControlTabs"));

    m_controlPointTable = new QTableWidget(tabs);
    m_controlPointTable->setObjectName(QStringLiteral("surveyControlPointTable"));
    m_checkPointTable = new QTableWidget(tabs);
    m_checkPointTable->setObjectName(QStringLiteral("surveyCheckPointTable"));
    m_scaleBarTable = new QTableWidget(tabs);
    m_scaleBarTable->setObjectName(QStringLiteral("surveyScaleBarTable"));

    tabs->addTab(m_controlPointTable, tr("控制点"));
    tabs->addTab(m_checkPointTable, tr("检查点"));
    tabs->addTab(m_scaleBarTable, tr("比例尺"));
    root->addWidget(tabs, 1);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    root->addWidget(buttons);
}

void SurveyControlDialog::refreshTables()
{
    const QJsonArray controlPoints = m_metadata.value(QStringLiteral("control_points")).toArray();
    const QJsonArray checkPoints = m_metadata.value(QStringLiteral("check_points")).toArray();
    const QJsonArray scaleBars = m_metadata.value(QStringLiteral("scale_bars")).toArray();

    if (m_summaryLabel)
    {
        m_summaryLabel->setText(tr("控制点 %1，检查点 %2，比例尺 %3")
                                    .arg(controlPoints.size())
                                    .arg(checkPoints.size())
                                    .arg(scaleBars.size()));
    }
    if (m_sourceLabel)
    {
        const QString sourcePath = m_metadata.value(QStringLiteral("source_path")).toString();
        m_sourceLabel->setText(sourcePath.isEmpty()
                                   ? tr("来源：未导入")
                                   : tr("来源：%1").arg(sourcePath));
    }
    if (m_statusLabel && m_statusLabel->text().isEmpty())
    {
        m_statusLabel->setText(tr("控制点参与平差，检查点用于验证；比例尺用于距离约束。"));
    }

    populatePointTable(m_controlPointTable, controlPoints, false);
    populatePointTable(m_checkPointTable, checkPoints, true);
    populateScaleBarTable(m_scaleBarTable, scaleBars);
}

void SurveyControlDialog::populatePointTable(QTableWidget *table, const QJsonArray &points, bool includeResidual)
{
    if (!table)
    {
        return;
    }

    QStringList headers = {tr("ID"), tr("X/E"), tr("Y/N"), tr("Z/H"), tr("Sigma(m)"), tr("启用")};
    if (includeResidual)
    {
        headers << tr("残差(m)");
    }

    table->clear();
    table->setColumnCount(headers.size());
    table->setHorizontalHeaderLabels(headers);
    table->setRowCount(points.size());
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);

    for (int row = 0; row < points.size(); ++row)
    {
        const QJsonObject point = points.at(row).toObject();
        int column = 0;
        setReadOnlyItem(table, row, column++, point.value(QStringLiteral("id")).toString());
        setReadOnlyItem(table, row, column++, jsonNumberText(point, QStringLiteral("x")));
        setReadOnlyItem(table, row, column++, jsonNumberText(point, QStringLiteral("y")));
        setReadOnlyItem(table, row, column++, jsonNumberText(point, QStringLiteral("z")));
        setReadOnlyItem(table, row, column++, jsonNumberText(point, QStringLiteral("sigma_m")));
        setReadOnlyItem(table, row, column++, point.value(QStringLiteral("enabled")).toBool(true) ? tr("是") : tr("否"));
        if (includeResidual)
        {
            setReadOnlyItem(table, row, column++, residualText(point));
        }
    }

    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->verticalHeader()->setVisible(false);
}

void SurveyControlDialog::populateScaleBarTable(QTableWidget *table, const QJsonArray &scaleBars)
{
    if (!table)
    {
        return;
    }

    const QStringList headers = {
        tr("ID"), tr("起点"), tr("终点"), tr("测量长度(m)"), tr("估计长度(m)"), tr("残差(m)"), tr("Sigma(m)"), tr("启用")
    };

    table->clear();
    table->setColumnCount(headers.size());
    table->setHorizontalHeaderLabels(headers);
    table->setRowCount(scaleBars.size());
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);

    for (int row = 0; row < scaleBars.size(); ++row)
    {
        const QJsonObject scaleBar = scaleBars.at(row).toObject();
        int column = 0;
        setReadOnlyItem(table, row, column++, scaleBar.value(QStringLiteral("id")).toString());
        setReadOnlyItem(table, row, column++, scaleBar.value(QStringLiteral("from_id")).toString());
        setReadOnlyItem(table, row, column++, scaleBar.value(QStringLiteral("to_id")).toString());
        setReadOnlyItem(table, row, column++, jsonNumberText(scaleBar, QStringLiteral("measured_m")));
        setReadOnlyItem(table, row, column++, jsonNumberText(scaleBar, QStringLiteral("estimated_m")));
        setReadOnlyItem(table, row, column++, residualText(scaleBar));
        setReadOnlyItem(table, row, column++, jsonNumberText(scaleBar, QStringLiteral("sigma_m")));
        setReadOnlyItem(table, row, column++, scaleBar.value(QStringLiteral("enabled")).toBool(true) ? tr("是") : tr("否"));
    }

    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->verticalHeader()->setVisible(false);
}
