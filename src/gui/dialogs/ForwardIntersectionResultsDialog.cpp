#include "ForwardIntersectionResultsDialog.h"

#include "ProjectManager.h"
#include "ui_ForwardIntersectionResultsDialog.h"

#include <QComboBox>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonObject>
#include <QPushButton>
#include <QTableWidget>

#include <QSet>

#include <algorithm>

namespace {

void setDetailItem(QTableWidget *table, int row, int column, const QString &text)
{
    table->setItem(row, column, new QTableWidgetItem(text));
}

QString formatJsonNumber(const QJsonObject &object, const QString &key)
{
    return QString::number(object.value(key).toDouble(), 'f', 6);
}

QString formatArrayNumber(const QJsonArray &values, int index)
{
    const double value = values.size() > index ? values.at(index).toDouble() : 0.0;
    return QString::number(value, 'f', 6);
}

}

ForwardIntersectionResultsDialog::ForwardIntersectionResultsDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , _projectManager(projectManager)
{
    setWindowTitle(tr("前方交汇结果查看"));
    resize(980, 700);
    setupUi();
    loadResults();
}

ForwardIntersectionResultsDialog::~ForwardIntersectionResultsDialog() = default;

void ForwardIntersectionResultsDialog::setupUi()
{
    Ui::ForwardIntersectionResultsDialog ui;
    ui.setupUi(this);

    _pairCombo = ui.m_pairCombo;
    _table = ui.m_table;
    _detailTable = ui.m_detailTable;

    _table->horizontalHeader()->setStretchLastSection(true);

    _detailTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    _detailTable->horizontalHeader()->setStretchLastSection(true);

    connect(ui.refreshBtn, &QPushButton::clicked, this, &ForwardIntersectionResultsDialog::loadResults);
    connect(_pairCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &ForwardIntersectionResultsDialog::onPairChanged);
    connect(_table, &QTableWidget::cellClicked, this, &ForwardIntersectionResultsDialog::onRowChanged);
}

void ForwardIntersectionResultsDialog::loadResults()
{
    _pairCombo->clear();
    _table->setRowCount(0);
    _detailTable->setRowCount(0);

    if (!_projectManager)
    {
        return;
    }
    _allResults = _projectManager->intersectionResults();

    QSet<QString> pairSet;
    for (const QJsonValue &v : _allResults)
    {
        if (!v.isObject())
        {
            continue;
        }
        pairSet.insert(makePairKey(v.toObject()));
    }

    QStringList pairs = pairSet.values();
    std::sort(pairs.begin(), pairs.end());
    for (const QString &k : pairs)
    {
        _pairCombo->addItem(k, k);
    }

    if (_pairCombo->count() > 0)
    {
        onPairChanged();
    }
}

void ForwardIntersectionResultsDialog::onPairChanged()
{
    const QString key = _pairCombo->currentData().toString();
    fillTableForPair(key);
}

void ForwardIntersectionResultsDialog::onRowChanged(int row, int column)
{
    Q_UNUSED(column);
    if (row < 0 || !_table->item(row, 0))
    {
        return;
    }
    QJsonObject obj = _table->item(row, 0)->data(Qt::UserRole).toJsonObject();
    if (obj.isEmpty())
    {
        return;
    }
    fillDetailTable(obj);
}

void ForwardIntersectionResultsDialog::fillTableForPair(const QString &pairKey)
{
    _table->setRowCount(0);
    QList<QJsonObject> filtered;
    for (const QJsonValue &v : _allResults)
    {
        if (!v.isObject())
        {
            continue;
        }
        QJsonObject obj = v.toObject();
        if (makePairKey(obj) == pairKey)
        {
            filtered.append(obj);
        }
    }

    std::sort(filtered.begin(), filtered.end(), [](const QJsonObject &a, const QJsonObject &b)
    {
        return a.value(QStringLiteral("created_at")).toString() > b.value(QStringLiteral("created_at")).toString();
    });

    _table->setRowCount(filtered.size());
    for (int i = 0; i < filtered.size(); ++i)
    {
        const QJsonObject obj = filtered[i];
        const QJsonObject summary = obj.value(QStringLiteral("summary")).toObject();
        const int totalPoints = summary.value(QStringLiteral("total_points")).toInt();
        const int validPoints = summary.value(QStringLiteral("valid_points")).toInt();
        const double validRatio = summary.value(QStringLiteral("valid_ratio")).toDouble();
        const double meanRms = summary.value(QStringLiteral("mean_rms")).toDouble();

        auto *timeItem = new QTableWidgetItem(obj.value(QStringLiteral("created_at")).toString());
        timeItem->setData(Qt::UserRole, obj);
        _table->setItem(i, 0, timeItem);
        _table->setItem(i, 1, new QTableWidgetItem(obj.value(QStringLiteral("pick_mode")).toString()));
        _table->setItem(i, 2, new QTableWidgetItem(QString::number(totalPoints)));
        _table->setItem(i, 3, new QTableWidgetItem(QString::number(validPoints)));
        _table->setItem(i, 4, new QTableWidgetItem(QString::number(validRatio, 'f', 4)));
        _table->setItem(i, 5, new QTableWidgetItem(QString::number(meanRms, 'f', 6)));
    }

    if (!filtered.isEmpty())
    {
        _table->selectRow(0);
        onRowChanged(0, 0);
    }
}

void ForwardIntersectionResultsDialog::fillDetailTable(const QJsonObject &batchResult)
{
    _detailTable->setRowCount(0);

    QJsonArray points = batchResult.value(QStringLiteral("points")).toArray();
    if (points.isEmpty() && batchResult.contains(QStringLiteral("metrics")))
    {
        // 兼容旧格式：单点结果
        QJsonObject one;
        one[QStringLiteral("index")] = 0;
        one[QStringLiteral("point0_uv")] = batchResult.value(QStringLiteral("point0_uv")).toArray();
        one[QStringLiteral("point1_uv")] = batchResult.value(QStringLiteral("point1_uv")).toArray();
        one[QStringLiteral("metrics")] = batchResult.value(QStringLiteral("metrics")).toObject();
        points.append(one);
    }

    _detailTable->setRowCount(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        const QJsonObject one = points.at(i).toObject();
        const QJsonArray point0_uv = one.value(QStringLiteral("point0_uv")).toArray();
        const QJsonArray point1_uv = one.value(QStringLiteral("point1_uv")).toArray();
        const QJsonObject metrics = one.value(QStringLiteral("metrics")).toObject();

        setDetailItem(_detailTable, i, 0, QString::number(one.value(QStringLiteral("index")).toInt(i) + 1));
        setDetailItem(_detailTable, i, 1, formatArrayNumber(point0_uv, 0));
        setDetailItem(_detailTable, i, 2, formatArrayNumber(point0_uv, 1));
        setDetailItem(_detailTable, i, 3, formatArrayNumber(point1_uv, 0));
        setDetailItem(_detailTable, i, 4, formatArrayNumber(point1_uv, 1));
        setDetailItem(_detailTable, i, 5, metrics.value(QStringLiteral("valid")).toBool() ? tr("是") : tr("否"));
        setDetailItem(_detailTable, i, 6, formatJsonNumber(metrics, QStringLiteral("X")));
        setDetailItem(_detailTable, i, 7, formatJsonNumber(metrics, QStringLiteral("Y")));
        setDetailItem(_detailTable, i, 8, formatJsonNumber(metrics, QStringLiteral("Z")));
        setDetailItem(_detailTable, i, 9, formatJsonNumber(metrics, QStringLiteral("angle_deg")));
        setDetailItem(_detailTable, i, 10, formatJsonNumber(metrics, QStringLiteral("ray_miss_distance")));
        setDetailItem(_detailTable, i, 11, formatJsonNumber(metrics, QStringLiteral("reproj_error_cam1")));
        setDetailItem(_detailTable, i, 12, formatJsonNumber(metrics, QStringLiteral("reproj_error_cam2")));
        setDetailItem(_detailTable, i, 13, formatJsonNumber(metrics, QStringLiteral("reproj_error_rms")));
    }
}

QString ForwardIntersectionResultsDialog::makePairKey(const QJsonObject &result) const
{
    const QString a = result.value(QStringLiteral("image0_name")).toString();
    const QString b = result.value(QStringLiteral("image1_name")).toString();
    if (a <= b)
    {
        return QStringLiteral("%1 ↔ %2").arg(a, b);
    }
    return QStringLiteral("%1 ↔ %2").arg(b, a);
}
