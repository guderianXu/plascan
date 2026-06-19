#include "ProjectDashboardWidget.h"

#include "ProjectDashboardSummary.h"

#include <QFileInfo>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <cmath>

namespace {

QString stateDisplayName(xjw::gui::project::ProjectDashboardStepState state)
{
    using xjw::gui::project::ProjectDashboardStepState;
    switch (state)
    {
    case ProjectDashboardStepState::Missing:
        return QStringLiteral("未开始");
    case ProjectDashboardStepState::Ready:
        return QStringLiteral("可执行");
    case ProjectDashboardStepState::Complete:
        return QStringLiteral("完成");
    case ProjectDashboardStepState::Warning:
        return QStringLiteral("注意");
    }
    return QStringLiteral("未开始");
}

QString reportTypeDisplayName(QString type)
{
    type = type.trimmed().toLower();
    if (type == QStringLiteral("reconstruction_quality"))
    {
        return QStringLiteral("重建质量");
    }
    if (type == QStringLiteral("reference_quality"))
    {
        return QStringLiteral("参考数据质检");
    }
    if (type == QStringLiteral("reference_terrain_prior_preflight"))
    {
        return QStringLiteral("平差前置检查");
    }
    return type.isEmpty() ? QStringLiteral("报告") : type;
}

QString referenceTypeDisplayName(QString type)
{
    type = type.trimmed().toLower();
    if (type == QStringLiteral("lidar"))
    {
        return QStringLiteral("LiDAR");
    }
    if (type == QStringLiteral("point_cloud"))
    {
        return QStringLiteral("点云");
    }
    if (type == QStringLiteral("dem"))
    {
        return QStringLiteral("DEM");
    }
    return type.isEmpty() ? QStringLiteral("参考") : type;
}

QString referenceRoleDisplayName(QString role)
{
    role = role.trimmed().toLower();
    if (role == QStringLiteral("ba_prior")
        || role == QStringLiteral("bundle_adjustment")
        || role == QStringLiteral("reference_prior"))
    {
        return QStringLiteral("BA约束");
    }
    if (role.isEmpty()
        || role == QStringLiteral("validation")
        || role == QStringLiteral("quality_check"))
    {
        return QStringLiteral("精度检查");
    }
    if (role == QStringLiteral("alignment") || role == QStringLiteral("registration"))
    {
        return QStringLiteral("配准");
    }
    return role;
}

QString referenceDatasetPath(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("path")).toString();
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("file_path")).toString();
    }
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("dem_path")).toString();
    }
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("lidar_path")).toString();
    }
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("cloud_path")).toString();
    }
    return path;
}

bool isVisibleTaskSnapshot(const QJsonObject &record)
{
    return record.value(QStringLiteral("active")).toBool(false)
        || record.value(QStringLiteral("cancelling")).toBool(false);
}

QString taskProgressText(const QJsonObject &record)
{
    const int value = record.value(QStringLiteral("progress_value")).toInt(-1);
    const int maximum = record.value(QStringLiteral("progress_maximum")).toInt(-1);
    if (value >= 0 && maximum > 0)
    {
        return QStringLiteral("%1/%2").arg(value).arg(maximum);
    }
    if (value >= 0)
    {
        return QString::number(value);
    }
    return QString();
}

bool hasNumber(const QJsonObject &record, const QString &key)
{
    const QJsonValue value = record.value(key);
    return value.isDouble() && std::isfinite(value.toDouble());
}

QString firstMetricText(const QJsonObject &record, const QStringList &keys, const QString &label)
{
    for (const QString &key : keys)
    {
        if (hasNumber(record, key))
        {
            return QStringLiteral("%1 %2 m").arg(label, QString::number(record.value(key).toDouble(), 'f', 3));
        }
    }
    return {};
}

QTableWidgetItem *makeReadOnlyItem(const QString &text)
{
    auto *item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    return item;
}

void appendAlertRow(QTableWidget *table,
                    int *row,
                    const QString &level,
                    const QString &source,
                    const QString &detail)
{
    if (!table || !row || detail.trimmed().isEmpty())
    {
        return;
    }

    table->insertRow(*row);
    table->setItem(*row, 0, makeReadOnlyItem(level));
    table->setItem(*row, 1, makeReadOnlyItem(source));
    table->setItem(*row, 2, makeReadOnlyItem(detail));
    ++(*row);
}

QString metricValueText(const QJsonValue &value, bool asPercent = false)
{
    if (value.isUndefined() || value.isNull())
    {
        return QString();
    }
    if (asPercent)
    {
        return QStringLiteral("%1%").arg(value.toDouble() * 100.0, 0, 'f', 1);
    }
    if (value.isDouble())
    {
        const double number = value.toDouble();
        const double rounded = std::round(number);
        if (std::abs(number - rounded) < 1e-9)
        {
            return QString::number(static_cast<qint64>(rounded));
        }
        return QString::number(number, 'f', 3);
    }
    if (value.isBool())
    {
        return value.toBool() ? QStringLiteral("是") : QStringLiteral("否");
    }
    return value.toString();
}

void appendMetricRow(QTableWidget *table, int *row, const QString &name, const QString &value)
{
    if (!table || !row || value.isEmpty())
    {
        return;
    }
    table->insertRow(*row);
    table->setItem(*row, 0, makeReadOnlyItem(name));
    table->setItem(*row, 1, makeReadOnlyItem(value));
    ++(*row);
}

void configureReadOnlyTable(QTableWidget *table)
{
    if (!table)
    {
        return;
    }
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->verticalHeader()->setVisible(false);
    table->setAlternatingRowColors(true);
    table->setWordWrap(false);
    table->horizontalHeader()->setStretchLastSection(true);
}

} // namespace

ProjectDashboardWidget::ProjectDashboardWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUi();
    clear();
}

ProjectDashboardWidget::~ProjectDashboardWidget() = default;

void ProjectDashboardWidget::setupUi()
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(8);

    m_summaryLabel = new QLabel(this);
    m_summaryLabel->setObjectName(QStringLiteral("dashboardSummaryLabel"));
    m_summaryLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_summaryLabel->setWordWrap(true);
    layout->addWidget(m_summaryLabel);

    m_referenceLabel = new QLabel(this);
    m_referenceLabel->setObjectName(QStringLiteral("dashboardReferenceLabel"));
    m_referenceLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_referenceLabel->setWordWrap(true);
    layout->addWidget(m_referenceLabel);

    m_taskLabel = new QLabel(this);
    m_taskLabel->setObjectName(QStringLiteral("dashboardTaskLabel"));
    m_taskLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_taskLabel->setWordWrap(true);
    layout->addWidget(m_taskLabel);

    m_taskTable = new QTableWidget(this);
    m_taskTable->setObjectName(QStringLiteral("dashboardTaskTable"));
    m_taskTable->setColumnCount(3);
    m_taskTable->setHorizontalHeaderLabels({tr("任务"), tr("状态"), tr("进度")});
    configureReadOnlyTable(m_taskTable);
    m_taskTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_taskTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    m_taskTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    layout->addWidget(m_taskTable, 1);

    m_referenceTable = new QTableWidget(this);
    m_referenceTable->setObjectName(QStringLiteral("dashboardReferenceTable"));
    m_referenceTable->setColumnCount(3);
    m_referenceTable->setHorizontalHeaderLabels({tr("类型"), tr("用途"), tr("路径")});
    configureReadOnlyTable(m_referenceTable);
    m_referenceTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_referenceTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_referenceTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    layout->addWidget(m_referenceTable, 1);

    m_workflowTable = new QTableWidget(this);
    m_workflowTable->setObjectName(QStringLiteral("dashboardWorkflowTable"));
    m_workflowTable->setColumnCount(3);
    m_workflowTable->setHorizontalHeaderLabels({tr("状态"), tr("阶段"), tr("说明")});
    configureReadOnlyTable(m_workflowTable);
    m_workflowTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_workflowTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_workflowTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    layout->addWidget(m_workflowTable, 2);

    m_qualityTable = new QTableWidget(this);
    m_qualityTable->setObjectName(QStringLiteral("dashboardQualityTable"));
    m_qualityTable->setColumnCount(2);
    m_qualityTable->setHorizontalHeaderLabels({tr("指标"), tr("数值")});
    configureReadOnlyTable(m_qualityTable);
    m_qualityTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_qualityTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    layout->addWidget(m_qualityTable, 1);

    m_qualityAlertTable = new QTableWidget(this);
    m_qualityAlertTable->setObjectName(QStringLiteral("dashboardQualityAlertTable"));
    m_qualityAlertTable->setColumnCount(3);
    m_qualityAlertTable->setHorizontalHeaderLabels({tr("级别"), tr("来源"), tr("说明")});
    configureReadOnlyTable(m_qualityAlertTable);
    m_qualityAlertTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_qualityAlertTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_qualityAlertTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    layout->addWidget(m_qualityAlertTable, 1);

    m_reportTable = new QTableWidget(this);
    m_reportTable->setObjectName(QStringLiteral("dashboardReportTable"));
    m_reportTable->setColumnCount(2);
    m_reportTable->setHorizontalHeaderLabels({tr("报告"), tr("路径")});
    configureReadOnlyTable(m_reportTable);
    m_reportTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_reportTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    layout->addWidget(m_reportTable, 1);
}

void ProjectDashboardWidget::loadFromJson(const QJsonObject &meta)
{
    updateTables(meta);
}

void ProjectDashboardWidget::setTaskSnapshots(const QJsonArray &tasks)
{
    m_taskSnapshots = tasks;
    updateTaskTable();
}

void ProjectDashboardWidget::clear()
{
    m_taskSnapshots = QJsonArray();
    updateTables(QJsonObject());
    updateTaskTable();
}

void ProjectDashboardWidget::updateTaskTable()
{
    QVector<QJsonObject> visibleTasks;
    for (const QJsonValue &value : m_taskSnapshots)
    {
        if (!value.isObject())
        {
            continue;
        }
        const QJsonObject record = value.toObject();
        if (isVisibleTaskSnapshot(record))
        {
            visibleTasks.append(record);
        }
    }

    if (m_taskLabel)
    {
        m_taskLabel->setText(visibleTasks.isEmpty()
            ? tr("主任务空闲，概览页仅只读展示已有项目数据。")
            : tr("当前运行任务 %1；概览页只读旁听状态，不会触发处理流程。").arg(visibleTasks.size()));
    }

    if (!m_taskTable)
    {
        return;
    }

    m_taskTable->setRowCount(visibleTasks.size());
    for (int row = 0; row < visibleTasks.size(); ++row)
    {
        const QJsonObject record = visibleTasks.at(row);
        QString statusText = record.value(QStringLiteral("status_text")).toString();
        if (record.value(QStringLiteral("cancelling")).toBool(false)
            && !statusText.contains(QStringLiteral("取消")))
        {
            statusText = QStringLiteral("正在取消：%1").arg(statusText);
        }
        m_taskTable->setItem(row,
                             0,
                             makeReadOnlyItem(record.value(QStringLiteral("name")).toString()));
        m_taskTable->setItem(row,
                             1,
                             makeReadOnlyItem(statusText));
        m_taskTable->setItem(row,
                             2,
                             makeReadOnlyItem(taskProgressText(record)));
    }
}

void ProjectDashboardWidget::updateTables(const QJsonObject &meta)
{
    const auto summary = xjw::gui::project::buildProjectDashboardSummary(meta);

    if (m_summaryLabel)
    {
        m_summaryLabel->setText(tr("影像 %1  相机 %2/%3  特征 %4  匹配 %5  点云 %6")
                                    .arg(summary.imageCount)
                                    .arg(summary.cameraCount)
                                    .arg(summary.imageCount)
                                    .arg(summary.featureResultCount)
                                    .arg(summary.matchResultCount)
                                    .arg(summary.denseCloudResultCount));
    }

    if (m_referenceLabel)
    {
        m_referenceLabel->setText(tr("LiDAR %1  参考点云 %2  参考DEM %3  BA约束 %4  质量报告 %5")
                                      .arg(summary.lidarReferenceCount)
                                      .arg(summary.pointCloudReferenceCount)
                                      .arg(summary.demReferenceCount)
                                      .arg(summary.baPriorReferenceCount)
                                      .arg(summary.qualityReportCount));
    }

    if (m_workflowTable)
    {
        m_workflowTable->setRowCount(summary.workflowSteps.size());
        for (int row = 0; row < summary.workflowSteps.size(); ++row)
        {
            const auto &step = summary.workflowSteps.at(row);
            m_workflowTable->setItem(row, 0, makeReadOnlyItem(stateDisplayName(step.state)));
            m_workflowTable->setItem(row, 1, makeReadOnlyItem(step.title));
            m_workflowTable->setItem(row, 2, makeReadOnlyItem(step.detail));
        }
    }

    if (m_referenceTable)
    {
        m_referenceTable->setRowCount(summary.referenceDatasets.size());
        for (int row = 0; row < summary.referenceDatasets.size(); ++row)
        {
            const QJsonObject reference = summary.referenceDatasets.at(row).toObject();
            const QString path = referenceDatasetPath(reference);
            m_referenceTable->setItem(row,
                                      0,
                                      makeReadOnlyItem(referenceTypeDisplayName(
                                          reference.value(QStringLiteral("type")).toString())));
            m_referenceTable->setItem(row,
                                      1,
                                      makeReadOnlyItem(referenceRoleDisplayName(
                                          reference.value(QStringLiteral("role")).toString())));
            m_referenceTable->setItem(row,
                                      2,
                                      makeReadOnlyItem(QFileInfo(path).fileName().isEmpty()
                                                           ? path
                                                           : QFileInfo(path).fileName()));
        }
    }

    if (m_qualityTable)
    {
        m_qualityTable->setRowCount(0);
        int row = 0;
        for (const QJsonValue &value : summary.qualityReports)
        {
            if (!value.isObject())
            {
                continue;
            }
            const QJsonObject report = value.toObject();
            const QString type = report.value(QStringLiteral("type")).toString();
            if (type == QStringLiteral("reconstruction_quality"))
            {
                const QString registeredImageCount =
                    metricValueText(report.value(QStringLiteral("registered_image_count")));
                const QString totalImageCount =
                    metricValueText(report.value(QStringLiteral("total_image_count")));
                QString imageRegistrationText;
                if (!registeredImageCount.isEmpty() || !totalImageCount.isEmpty())
                {
                    imageRegistrationText = QStringLiteral("%1/%2").arg(registeredImageCount, totalImageCount);
                }
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("注册影像"),
                                imageRegistrationText);
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("稀疏点"),
                                metricValueText(report.value(QStringLiteral("sparse_point_count"))));
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("稠密点"),
                                metricValueText(report.value(QStringLiteral("dense_point_count"))));
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("MVS覆盖"),
                                metricValueText(report.value(QStringLiteral("mvs_valid_coverage")), true));
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("DEM覆盖"),
                                metricValueText(report.value(QStringLiteral("dem_coverage")), true));
            }
            else if (type == QStringLiteral("reference_quality"))
            {
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("参考数据状态"),
                                metricValueText(report.value(QStringLiteral("status"))));
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("参考数据可对比"),
                                metricValueText(report.value(QStringLiteral("comparison_available"))));
            }
            else if (type == QStringLiteral("reference_terrain_prior_preflight"))
            {
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("参考地形平差就绪"),
                                metricValueText(report.value(QStringLiteral("ready"))));
                appendMetricRow(m_qualityTable,
                                &row,
                                tr("BA约束参考数"),
                                metricValueText(report.value(QStringLiteral("ba_prior_reference_count"))));
            }
        }
    }

    if (m_qualityAlertTable)
    {
        m_qualityAlertTable->setRowCount(0);
        int row = 0;
        for (const QJsonValue &value : summary.qualityReports)
        {
            if (!value.isObject())
            {
                continue;
            }
            const QJsonObject report = value.toObject();
            const QString type = report.value(QStringLiteral("type")).toString();
            if (type == QStringLiteral("reconstruction_quality"))
            {
                const int registered = report.value(QStringLiteral("registered_image_count")).toInt(-1);
                const int total = report.value(QStringLiteral("total_image_count")).toInt(-1);
                if (registered >= 0 && total > 0 && registered < total)
                {
                    appendAlertRow(m_qualityAlertTable,
                                   &row,
                                   registered * 10 < total * 8 ? tr("注意") : tr("信息"),
                                   tr("重建质量"),
                                   tr("注册影像 %1/%2").arg(registered).arg(total));
                }
                const double mvsCoverage = report.value(QStringLiteral("mvs_valid_coverage")).toDouble(-1.0);
                if (mvsCoverage >= 0.0 && mvsCoverage < 0.6)
                {
                    appendAlertRow(m_qualityAlertTable,
                                   &row,
                                   tr("注意"),
                                   tr("重建质量"),
                                   tr("MVS覆盖 %1").arg(metricValueText(report.value(QStringLiteral("mvs_valid_coverage")),
                                                                      true)));
                }
                const double demCoverage = report.value(QStringLiteral("dem_coverage")).toDouble(-1.0);
                if (demCoverage >= 0.0 && demCoverage < 0.6)
                {
                    appendAlertRow(m_qualityAlertTable,
                                   &row,
                                   tr("注意"),
                                   tr("重建质量"),
                                   tr("DEM覆盖 %1").arg(metricValueText(report.value(QStringLiteral("dem_coverage")),
                                                                      true)));
                }
            }
            else if (type == QStringLiteral("reference_quality"))
            {
                const QString status = report.value(QStringLiteral("status")).toString();
                const bool comparisonAvailable =
                    report.value(QStringLiteral("comparison_available")).toBool(false);
                if (!comparisonAvailable || (!status.isEmpty() && status != QStringLiteral("ready")))
                {
                    appendAlertRow(m_qualityAlertTable,
                                   &row,
                                   tr("阻塞"),
                                   tr("参考数据质检"),
                                   tr("status %1，可对比 %2")
                                       .arg(status.isEmpty() ? QStringLiteral("unknown") : status,
                                            comparisonAvailable ? tr("是") : tr("否")));
                }

                QStringList errorMetrics;
                const QString rmse = firstMetricText(report,
                                                     {QStringLiteral("rmse_m"),
                                                      QStringLiteral("cloud_rmse_m"),
                                                      QStringLiteral("point_cloud_rmse_m"),
                                                      QStringLiteral("dem_rmse_m")},
                                                     QStringLiteral("RMSE"));
                if (!rmse.isEmpty())
                {
                    errorMetrics.append(rmse);
                }
                const QString p95 = firstMetricText(report,
                                                    {QStringLiteral("p95_distance_m"),
                                                     QStringLiteral("p95_error_m"),
                                                     QStringLiteral("cloud_p95_m"),
                                                     QStringLiteral("dem_p95_m")},
                                                    QStringLiteral("P95"));
                if (!p95.isEmpty())
                {
                    errorMetrics.append(p95);
                }
                if (!errorMetrics.isEmpty())
                {
                    appendAlertRow(m_qualityAlertTable,
                                   &row,
                                   tr("信息"),
                                   tr("参考误差"),
                                   errorMetrics.join(QStringLiteral("，")));
                }
            }
            else if (type == QStringLiteral("reference_terrain_prior_preflight"))
            {
                const bool ready = report.value(QStringLiteral("ready")).toBool(false);
                const QString status = report.value(QStringLiteral("status")).toString();
                if (!ready)
                {
                    appendAlertRow(m_qualityAlertTable,
                                   &row,
                                   tr("阻塞"),
                                   tr("参考地形平差"),
                                   tr("status %1，BA约束参考数 %2")
                                       .arg(status.isEmpty() ? QStringLiteral("unknown") : status)
                                       .arg(report.value(QStringLiteral("ba_prior_reference_count")).toInt(0)));
                }
            }
        }

        if (row == 0)
        {
            appendAlertRow(m_qualityAlertTable,
                           &row,
                           tr("信息"),
                           tr("质量中心"),
                           tr("暂无阻塞项或可显示的参考误差指标。"));
        }
    }

    if (m_reportTable)
    {
        m_reportTable->setRowCount(summary.qualityReports.size());
        for (int row = 0; row < summary.qualityReports.size(); ++row)
        {
            const QJsonObject report = summary.qualityReports.at(row).toObject();
            QString path = report.value(QStringLiteral("path")).toString();
            if (path.isEmpty())
            {
                path = report.value(QStringLiteral("json_path")).toString();
            }
            m_reportTable->setItem(row,
                                   0,
                                   makeReadOnlyItem(reportTypeDisplayName(
                                       report.value(QStringLiteral("type")).toString())));
            m_reportTable->setItem(row,
                                   1,
                                   makeReadOnlyItem(QFileInfo(path).fileName().isEmpty() ? path
                                                                                         : QFileInfo(path).fileName()));
        }
    }
}
