#include "ForwardIntersectionResultsDialog.h"

#include "ProjectManager.h"
#include "ui_ForwardIntersectionResultsDialog.h"

#include <QComboBox>
#include <QHeaderView>
#include <QJsonObject>
#include <QPushButton>
#include <QTableWidget>

#include <QSet>

#include <algorithm>

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
        const QJsonArray p0 = one.value(QStringLiteral("point0_uv")).toArray();
        const QJsonArray p1 = one.value(QStringLiteral("point1_uv")).toArray();
        const QJsonObject m = one.value(QStringLiteral("metrics")).toObject();

        _detailTable->setItem(i, 0, new QTableWidgetItem(QString::number(one.value(QStringLiteral("index")).toInt(i) + 1)));
        _detailTable->setItem(i, 1, new QTableWidgetItem(QString::number(p0.size() > 0 ? p0.at(0).toDouble() : 0.0, 'f', 6)));
        _detailTable->setItem(i, 2, new QTableWidgetItem(QString::number(p0.size() > 1 ? p0.at(1).toDouble() : 0.0, 'f', 6)));
        _detailTable->setItem(i, 3, new QTableWidgetItem(QString::number(p1.size() > 0 ? p1.at(0).toDouble() : 0.0, 'f', 6)));
        _detailTable->setItem(i, 4, new QTableWidgetItem(QString::number(p1.size() > 1 ? p1.at(1).toDouble() : 0.0, 'f', 6)));
        _detailTable->setItem(i, 5, new QTableWidgetItem(m.value(QStringLiteral("valid")).toBool() ? tr("是") : tr("否")));
        _detailTable->setItem(i, 6, new QTableWidgetItem(QString::number(m.value(QStringLiteral("X")).toDouble(), 'f', 6)));
        _detailTable->setItem(i, 7, new QTableWidgetItem(QString::number(m.value(QStringLiteral("Y")).toDouble(), 'f', 6)));
        _detailTable->setItem(i, 8, new QTableWidgetItem(QString::number(m.value(QStringLiteral("Z")).toDouble(), 'f', 6)));
        _detailTable->setItem(i, 9, new QTableWidgetItem(QString::number(m.value(QStringLiteral("angle_deg")).toDouble(), 'f', 6)));
        _detailTable->setItem(i, 10, new QTableWidgetItem(QString::number(m.value(QStringLiteral("ray_miss_distance")).toDouble(), 'f', 6)));
        _detailTable->setItem(i, 11, new QTableWidgetItem(QString::number(m.value(QStringLiteral("reproj_error_cam1")).toDouble(), 'f', 6)));
        _detailTable->setItem(i, 12, new QTableWidgetItem(QString::number(m.value(QStringLiteral("reproj_error_cam2")).toDouble(), 'f', 6)));
        _detailTable->setItem(i, 13, new QTableWidgetItem(QString::number(m.value(QStringLiteral("reproj_error_rms")).toDouble(), 'f', 6)));
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
