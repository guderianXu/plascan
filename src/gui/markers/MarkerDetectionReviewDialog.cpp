#include "MarkerDetectionReviewDialog.h"

#include "MarkerWorkspaceController.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QHeaderView>
#include <QImage>
#include <QLabel>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QSplitter>
#include <QTableWidget>
#include <QVBoxLayout>

#include <algorithm>

namespace xjw::gui::markers
{
namespace
{

QString targetName(const control_points::DetectionReviewEntry &entry)
{
    const QString family = control_points::markerTargetFamilyName(
        entry.observation.detection.family);
    if (entry.observation.detection.targetId < 0) return family;
    return QStringLiteral("%1 / %2").arg(family).arg(entry.observation.detection.targetId);
}

const control_points::DetectionReviewEntry *findEntry(
    const control_points::DetectionReviewQueue &queue,
    const QString &entryId)
{
    const auto found = std::find_if(
        queue.entries.cbegin(), queue.entries.cend(),
        [&entryId](const control_points::DetectionReviewEntry &entry)
    {
        return entry.id == entryId;
    });
    return found == queue.entries.cend() ? nullptr : &*found;
}

} // namespace

MarkerDetectionReviewDialog::MarkerDetectionReviewDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("复核标靶检测"));
    resize(980, 620);

    auto *root = new QVBoxLayout(this);
    auto *splitter = new QSplitter(Qt::Horizontal, this);
    root->addWidget(splitter, 1);

    _table = new QTableWidget(splitter);
    _table->setObjectName(QStringLiteral("markerDetectionReviewTable"));
    _table->setColumnCount(6);
    _table->setHorizontalHeaderLabels({
        QStringLiteral("影像"),
        QStringLiteral("标靶"),
        QStringLiteral("中心 X"),
        QStringLiteral("中心 Y"),
        QStringLiteral("置信度"),
        QStringLiteral("待复核原因"),
    });
    _table->setSelectionBehavior(QAbstractItemView::SelectRows);
    _table->setSelectionMode(QAbstractItemView::SingleSelection);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->verticalHeader()->setVisible(false);
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    splitter->addWidget(_table);

    auto *right = new QWidget(splitter);
    auto *right_layout = new QVBoxLayout(right);
    _preview = new QLabel(right);
    _preview->setObjectName(QStringLiteral("markerDetectionReviewPreview"));
    _preview->setMinimumSize(380, 300);
    _preview->setAlignment(Qt::AlignCenter);
    _preview->setFrameShape(QFrame::StyledPanel);
    right_layout->addWidget(_preview, 1);

    _details = new QLabel(right);
    _details->setWordWrap(true);
    _details->setTextInteractionFlags(Qt::TextSelectableByMouse);
    right_layout->addWidget(_details);

    _assignment = new QComboBox(right);
    _assignment->setObjectName(QStringLiteral("markerDetectionAssignmentCombo"));
    right_layout->addWidget(_assignment);

    auto *review_buttons = new QDialogButtonBox(right);
    _accept = review_buttons->addButton(QStringLiteral("接受"), QDialogButtonBox::AcceptRole);
    _accept->setObjectName(QStringLiteral("acceptMarkerDetectionButton"));
    _discard = review_buttons->addButton(QStringLiteral("丢弃"), QDialogButtonBox::DestructiveRole);
    _discard->setObjectName(QStringLiteral("discardMarkerDetectionButton"));
    right_layout->addWidget(review_buttons);
    splitter->addWidget(right);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);

    auto *close_buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    root->addWidget(close_buttons);
    connect(close_buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(_table, &QTableWidget::itemSelectionChanged,
            this, &MarkerDetectionReviewDialog::updateSelection);
    connect(_accept, &QPushButton::clicked,
            this, &MarkerDetectionReviewDialog::acceptSelected);
    connect(_discard, &QPushButton::clicked,
            this, &MarkerDetectionReviewDialog::discardSelected);
    updateSelection();
}

bool MarkerDetectionReviewDialog::setController(MarkerWorkspaceController *controller)
{
    if (_controller) disconnect(_controller, nullptr, this, nullptr);
    _controller = controller;
    if (!_controller)
    {
        refresh();
        return false;
    }
    connect(_controller, &MarkerWorkspaceController::detectionReviewChanged,
            this, &MarkerDetectionReviewDialog::refresh);
    connect(_controller, &MarkerWorkspaceController::markerSetChanged,
            this, &MarkerDetectionReviewDialog::refreshAssignmentOptions);
    refresh();
    return true;
}

void MarkerDetectionReviewDialog::refresh()
{
    const QString previous = selectedEntryId();
    _table->setRowCount(0);
    if (!_controller)
    {
        refreshAssignmentOptions();
        updateSelection();
        return;
    }

    const auto &entries = _controller->detectionReviewQueue().entries;
    for (const control_points::DetectionReviewEntry &entry : entries)
    {
        const int row = _table->rowCount();
        _table->insertRow(row);
        auto *image_item = new QTableWidgetItem(
            QFileInfo(entry.observation.imagePathSnapshot).fileName());
        image_item->setData(Qt::UserRole, entry.id);
        _table->setItem(row, 0, image_item);
        _table->setItem(row, 1, new QTableWidgetItem(targetName(entry)));
        _table->setItem(row, 2, new QTableWidgetItem(
            QString::number(entry.observation.detection.center.x(), 'f', 2)));
        _table->setItem(row, 3, new QTableWidgetItem(
            QString::number(entry.observation.detection.center.y(), 'f', 2)));
        _table->setItem(row, 4, new QTableWidgetItem(
            QString::number(entry.observation.detection.confidence, 'f', 3)));
        _table->setItem(row, 5, new QTableWidgetItem(entry.message));
        if (entry.id == previous) _table->selectRow(row);
    }
    if (_table->rowCount() > 0 && _table->selectedItems().isEmpty())
    {
        _table->selectRow(0);
    }
    refreshAssignmentOptions();
    updateSelection();
}

void MarkerDetectionReviewDialog::updateSelection()
{
    const bool selected = !selectedEntryId().isEmpty();
    _accept->setEnabled(selected && _controller);
    _discard->setEnabled(selected && _controller);
    _assignment->setEnabled(selected && _controller);
    updatePreview();
}

void MarkerDetectionReviewDialog::acceptSelected()
{
    if (!_controller) return;
    const QString entry_id = selectedEntryId();
    if (entry_id.isEmpty()) return;
    const QString marker_id = _assignment->currentData().toString();
    QString error;
    if (!_controller->acceptDetectionReview(entry_id, marker_id, &error))
    {
        showOperationError(error);
        return;
    }
    refresh();
}

void MarkerDetectionReviewDialog::discardSelected()
{
    if (!_controller) return;
    const QString entry_id = selectedEntryId();
    if (entry_id.isEmpty()) return;
    QString error;
    if (!_controller->discardDetectionReview(entry_id, &error))
    {
        showOperationError(error);
        return;
    }
    refresh();
}

QString MarkerDetectionReviewDialog::selectedEntryId() const
{
    if (!_table || _table->currentRow() < 0) return {};
    const QTableWidgetItem *item = _table->item(_table->currentRow(), 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

void MarkerDetectionReviewDialog::refreshAssignmentOptions()
{
    const QString previous = _assignment->currentData().toString();
    _assignment->clear();
    _assignment->addItem(QStringLiteral("创建新标记"), QString());
    if (_controller)
    {
        for (const control_points::Marker &marker : _controller->markerSet().markers())
        {
            _assignment->addItem(marker.label, marker.id);
        }
    }
    const int previous_index = _assignment->findData(previous);
    if (previous_index >= 0) _assignment->setCurrentIndex(previous_index);
}

void MarkerDetectionReviewDialog::updatePreview()
{
    _preview->clear();
    _details->clear();
    if (!_controller) return;
    const auto *entry = findEntry(
        _controller->detectionReviewQueue(), selectedEntryId());
    if (!entry) return;

    QImage image(entry->observation.imagePathSnapshot);
    if (image.isNull())
    {
        _preview->setText(QStringLiteral("无法读取影像"));
    }
    else
    {
        QPainter painter(&image);
        painter.setRenderHint(QPainter::Antialiasing);
        QPen pen(QColor(0, 220, 120));
        pen.setWidthF(std::max(2.0, image.width() / 800.0));
        painter.setPen(pen);
        const QPointF center = entry->observation.detection.center;
        const qreal radius = std::max(8.0, entry->observation.detection.sizePx * 0.5);
        painter.drawEllipse(center, radius, radius);
        painter.drawLine(center + QPointF(-radius * 1.5, 0.0),
                         center + QPointF(radius * 1.5, 0.0));
        painter.drawLine(center + QPointF(0.0, -radius * 1.5),
                         center + QPointF(0.0, radius * 1.5));
        painter.end();
        _preview->setPixmap(QPixmap::fromImage(image).scaled(
            _preview->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    _details->setText(QStringLiteral("%1\n%2\n像素坐标: (%3, %4)")
                          .arg(entry->reason,
                               entry->message,
                               QString::number(entry->observation.detection.center.x(), 'f', 3),
                               QString::number(entry->observation.detection.center.y(), 'f', 3)));
}

void MarkerDetectionReviewDialog::showOperationError(const QString &message)
{
    QMessageBox::warning(this, QStringLiteral("复核标靶检测"), message);
}

} // namespace xjw::gui::markers
