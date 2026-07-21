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

void updateTableVisibility(QTableWidget *table)
{
    if (table)
    {
        table->setVisible(table->rowCount() > 0);
    }
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

    _summaryLabel = new QLabel(this);
    _summaryLabel->setObjectName(QStringLiteral("dashboardSummaryLabel"));
    _summaryLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _summaryLabel->setWordWrap(true);
    layout->addWidget(_summaryLabel);

    _referenceLabel = new QLabel(this);
    _referenceLabel->setObjectName(QStringLiteral("dashboardReferenceLabel"));
    _referenceLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _referenceLabel->setWordWrap(true);
    layout->addWidget(_referenceLabel);

    _taskLabel = new QLabel(this);
    _taskLabel->setObjectName(QStringLiteral("dashboardTaskLabel"));
    _taskLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _taskLabel->setWordWrap(true);
    layout->addWidget(_taskLabel);

    _taskTable = new QTableWidget(this);
    _taskTable->setObjectName(QStringLiteral("dashboardTaskTable"));
    _taskTable->setColumnCount(3);
    _taskTable->setHorizontalHeaderLabels({tr("任务"), tr("状态"), tr("进度")});
    configureReadOnlyTable(_taskTable);
    _taskTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _taskTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    _taskTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    layout->addWidget(_taskTable, 1);

    _referenceTable = new QTableWidget(this);
    _referenceTable->setObjectName(QStringLiteral("dashboardReferenceTable"));
    _referenceTable->setColumnCount(3);
    _referenceTable->setHorizontalHeaderLabels({tr("类型"), tr("用途"), tr("路径")});
    configureReadOnlyTable(_referenceTable);
    _referenceTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _referenceTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _referenceTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    layout->addWidget(_referenceTable, 1);

    _workflowTable = new QTableWidget(this);
    _workflowTable->setObjectName(QStringLiteral("dashboardWorkflowTable"));
    _workflowTable->setColumnCount(3);
    _workflowTable->setHorizontalHeaderLabels({tr("状态"), tr("阶段"), tr("说明")});
    configureReadOnlyTable(_workflowTable);
    _workflowTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _workflowTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _workflowTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    layout->addWidget(_workflowTable, 2);

    _qualityTable = new QTableWidget(this);
    _qualityTable->setObjectName(QStringLiteral("dashboardQualityTable"));
    _qualityTable->setColumnCount(2);
    _qualityTable->setHorizontalHeaderLabels({tr("指标"), tr("数值")});
    configureReadOnlyTable(_qualityTable);
    _qualityTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _qualityTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    layout->addWidget(_qualityTable, 1);

    _qualityAlertTable = new QTableWidget(this);
    _qualityAlertTable->setObjectName(QStringLiteral("dashboardQualityAlertTable"));
    _qualityAlertTable->setColumnCount(3);
    _qualityAlertTable->setHorizontalHeaderLabels({tr("级别"), tr("来源"), tr("说明")});
    configureReadOnlyTable(_qualityAlertTable);
    _qualityAlertTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _qualityAlertTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _qualityAlertTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    layout->addWidget(_qualityAlertTable, 1);

    _reportTable = new QTableWidget(this);
    _reportTable->setObjectName(QStringLiteral("dashboardReportTable"));
    _reportTable->setColumnCount(2);
    _reportTable->setHorizontalHeaderLabels({tr("报告"), tr("路径")});
    configureReadOnlyTable(_reportTable);
    _reportTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _reportTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    layout->addWidget(_reportTable, 1);
}

void ProjectDashboardWidget::loadFromJson(const QJsonObject &meta)
{
    updateTables(meta);
}

void ProjectDashboardWidget::setTaskSnapshots(const QJsonArray &tasks)
{
    _taskSnapshots = tasks;
    updateTaskTable();
}

void ProjectDashboardWidget::clear()
{
    _taskSnapshots = QJsonArray();
    updateTables(QJsonObject());
    updateTaskTable();
}

void ProjectDashboardWidget::updateTaskTable()
{
    QVector<QJsonObject> visibleTasks;
    for (const QJsonValue &value : _taskSnapshots)
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

    if (_taskLabel)
    {
        _taskLabel->setText(visibleTasks.isEmpty()
            ? tr("主任务空闲，概览页仅只读展示已有项目数据。")
            : tr("当前运行任务 %1；概览页只读旁听状态，不会触发处理流程。").arg(visibleTasks.size()));
    }

    if (!_taskTable)
    {
        return;
    }

    _taskTable->setRowCount(visibleTasks.size());
    for (int row = 0; row < visibleTasks.size(); ++row)
    {
        const QJsonObject record = visibleTasks.at(row);
        QString statusText = record.value(QStringLiteral("status_text")).toString();
        if (record.value(QStringLiteral("cancelling")).toBool(false)
            && !statusText.contains(QStringLiteral("取消")))
        {
            statusText = QStringLiteral("正在取消：%1").arg(statusText);
        }
        _taskTable->setItem(row,
                             0,
                             makeReadOnlyItem(record.value(QStringLiteral("name")).toString()));
        _taskTable->setItem(row,
                             1,
                             makeReadOnlyItem(statusText));
        _taskTable->setItem(row,
                             2,
                             makeReadOnlyItem(taskProgressText(record)));
    }
    updateTableVisibility(_taskTable);
}

void ProjectDashboardWidget::updateTables(const QJsonObject &meta)
{
    const auto summary = xjw::gui::project::buildProjectDashboardSummary(meta);

    if (_summaryLabel)
    {
        _summaryLabel->setText(tr("影像 %1  相机 %2/%3  特征 %4  匹配 %5  点云 %6")
                                    .arg(summary.imageCount)
                                    .arg(summary.cameraCount)
                                    .arg(summary.imageCount)
                                    .arg(summary.featureResultCount)
                                    .arg(summary.matchResultCount)
                                    .arg(summary.denseCloudResultCount));
    }

    if (_referenceLabel)
    {
        _referenceLabel->setText(tr("LiDAR %1  参考点云 %2  参考DEM %3  BA约束 %4  质量报告 %5")
                                      .arg(summary.lidarReferenceCount)
                                      .arg(summary.pointCloudReferenceCount)
                                      .arg(summary.demReferenceCount)
                                      .arg(summary.baPriorReferenceCount)
                                      .arg(summary.qualityReportCount));
    }

    if (_workflowTable)
    {
        _workflowTable->setRowCount(summary.workflowSteps.size());
        for (int row = 0; row < summary.workflowSteps.size(); ++row)
        {
            const auto &step = summary.workflowSteps.at(row);
            _workflowTable->setItem(row, 0, makeReadOnlyItem(stateDisplayName(step.state)));
            _workflowTable->setItem(row, 1, makeReadOnlyItem(step.title));
            _workflowTable->setItem(row, 2, makeReadOnlyItem(step.detail));
        }
        updateTableVisibility(_workflowTable);
    }

    if (_referenceTable)
    {
        _referenceTable->setRowCount(summary.referenceDatasets.size());
        for (int row = 0; row < summary.referenceDatasets.size(); ++row)
        {
            const QJsonObject reference = summary.referenceDatasets.at(row).toObject();
            const QString path = referenceDatasetPath(reference);
            _referenceTable->setItem(row,
                                      0,
                                      makeReadOnlyItem(referenceTypeDisplayName(
                                          reference.value(QStringLiteral("type")).toString())));
            _referenceTable->setItem(row,
                                      1,
                                      makeReadOnlyItem(referenceRoleDisplayName(
                                          reference.value(QStringLiteral("role")).toString())));
            _referenceTable->setItem(row,
                                      2,
                                      makeReadOnlyItem(QFileInfo(path).fileName().isEmpty()
                                                           ? path
                                                           : QFileInfo(path).fileName()));
        }
        updateTableVisibility(_referenceTable);
    }

    if (_qualityTable)
    {
        _qualityTable->setRowCount(0);
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
                appendMetricRow(_qualityTable,
                                &row,
                                tr("注册影像"),
                                imageRegistrationText);
                appendMetricRow(_qualityTable,
                                &row,
                                tr("稀疏点"),
                                metricValueText(report.value(QStringLiteral("sparse_point_count"))));
                appendMetricRow(_qualityTable,
                                &row,
                                tr("稠密点"),
                                metricValueText(report.value(QStringLiteral("dense_point_count"))));
                const QJsonValue mvsCoverage = report.value(QStringLiteral("mvs_valid_coverage"));
                appendMetricRow(_qualityTable,
                                &row,
                                tr("MVS覆盖"),
                                mvsCoverage.isDouble() ? metricValueText(mvsCoverage, true) : tr("—"));
                appendMetricRow(_qualityTable,
                                &row,
                                tr("DEM覆盖"),
                                metricValueText(report.value(QStringLiteral("dem_coverage")), true));
            }
            else if (type == QStringLiteral("reference_quality"))
            {
                appendMetricRow(_qualityTable,
                                &row,
                                tr("参考数据状态"),
                                metricValueText(report.value(QStringLiteral("status"))));
                appendMetricRow(_qualityTable,
                                &row,
                                tr("参考数据可对比"),
                                metricValueText(report.value(QStringLiteral("comparison_available"))));
            }
            else if (type == QStringLiteral("reference_terrain_prior_preflight"))
            {
                appendMetricRow(_qualityTable,
                                &row,
                                tr("参考地形平差就绪"),
                                metricValueText(report.value(QStringLiteral("ready"))));
                appendMetricRow(_qualityTable,
                                &row,
                                tr("BA约束参考数"),
                                metricValueText(report.value(QStringLiteral("ba_prior_reference_count"))));
            }
        }
        updateTableVisibility(_qualityTable);
    }

    if (_qualityAlertTable)
    {
        _qualityAlertTable->setRowCount(0);
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
                    appendAlertRow(_qualityAlertTable,
                                   &row,
                                   registered * 10 < total * 8 ? tr("注意") : tr("信息"),
                                   tr("重建质量"),
                                   tr("注册影像 %1/%2").arg(registered).arg(total));
                }
                const QJsonValue mvsCoverage = report.value(QStringLiteral("mvs_valid_coverage"));
                if (mvsCoverage.isDouble() && mvsCoverage.toDouble() < 0.6)
                {
                    appendAlertRow(_qualityAlertTable,
                                   &row,
                                   tr("注意"),
                                   tr("重建质量"),
                                   tr("MVS覆盖 %1").arg(metricValueText(mvsCoverage, true)));
                }
                const double demCoverage = report.value(QStringLiteral("dem_coverage")).toDouble(-1.0);
                if (demCoverage >= 0.0 && demCoverage < 0.6)
                {
                    appendAlertRow(_qualityAlertTable,
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
                    appendAlertRow(_qualityAlertTable,
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
                    appendAlertRow(_qualityAlertTable,
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
                    appendAlertRow(_qualityAlertTable,
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
            appendAlertRow(_qualityAlertTable,
                           &row,
                           tr("信息"),
                           tr("质量中心"),
                           tr("暂无阻塞项或可显示的参考误差指标。"));
        }
        updateTableVisibility(_qualityAlertTable);
    }

    if (_reportTable)
    {
        _reportTable->setRowCount(summary.qualityReports.size());
        for (int row = 0; row < summary.qualityReports.size(); ++row)
        {
            const QJsonObject report = summary.qualityReports.at(row).toObject();
            QString path = report.value(QStringLiteral("path")).toString();
            if (path.isEmpty())
            {
                path = report.value(QStringLiteral("json_path")).toString();
            }
            _reportTable->setItem(row,
                                   0,
                                   makeReadOnlyItem(reportTypeDisplayName(
                                       report.value(QStringLiteral("type")).toString())));
            _reportTable->setItem(row,
                                   1,
                                   makeReadOnlyItem(QFileInfo(path).fileName().isEmpty() ? path
                                                                                         : QFileInfo(path).fileName()));
        }
        updateTableVisibility(_reportTable);
    }
}
