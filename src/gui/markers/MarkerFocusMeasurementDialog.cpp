#include "MarkerFocusMeasurementDialog.h"

#include "DualImageViewer.h"
#include "MarkerWorkspaceController.h"
#include "ProjectData.h"
#include "project/ProjectIO.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "geometry/MarkerProjectionPredictor.h"
#include "Camera.h"
#include "io/PathIO.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QImageReader>
#include <QPushButton>
#include <QVBoxLayout>

#include <algorithm>

#include <opencv2/imgcodecs.hpp>

namespace xjw::gui::markers
{

namespace
{

bool samePath(const QString &left, const QString &right)
{
#ifdef Q_OS_WIN
    return QDir::cleanPath(QFileInfo(left).absoluteFilePath()).compare(
               QDir::cleanPath(QFileInfo(right).absoluteFilePath()), Qt::CaseInsensitive) == 0;
#else
    return QDir::cleanPath(QFileInfo(left).absoluteFilePath())
        == QDir::cleanPath(QFileInfo(right).absoluteFilePath());
#endif
}

const control_points::MarkerProjection *projectionForPath(
    const control_points::Marker &marker,
    const QString &path)
{
    const auto it = std::find_if(marker.projections.cbegin(), marker.projections.cend(),
                                 [&path](const auto &projection)
    {
        return samePath(projection.imagePathSnapshot, path);
    });
    return it == marker.projections.cend() ? nullptr : &*it;
}

} // namespace

MarkerFocusMeasurementDialog::MarkerFocusMeasurementDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
}

void MarkerFocusMeasurementDialog::setupUi()
{
    setWindowTitle(QStringLiteral("聚焦标记量测"));
    resize(1100, 720);
    auto *layout = new QVBoxLayout(this);
    auto *candidate_row = new QHBoxLayout;
    candidate_row->addWidget(new QLabel(QStringLiteral("候选影像"), this));
    _candidateCombo = new QComboBox(this);
    _candidateCombo->setObjectName(QStringLiteral("markerCandidateImageCombo"));
    candidate_row->addWidget(_candidateCombo, 1);
    layout->addLayout(candidate_row);

    _viewer = new DualImageViewer(this);
    layout->addWidget(_viewer, 1);
    _statusLabel = new QLabel(QStringLiteral("在右侧影像中右键放置候选位置"), this);
    _statusLabel->setWordWrap(true);
    layout->addWidget(_statusLabel);

    auto *buttons = new QHBoxLayout;
    _confirmButton = new QPushButton(QStringLiteral("确认投影"), this);
    _confirmButton->setObjectName(QStringLiteral("confirmMarkerProjectionButton"));
    _blockButton = new QPushButton(QStringLiteral("屏蔽"), this);
    _blockButton->setObjectName(QStringLiteral("blockMarkerProjectionButton"));
    _disableButton = new QPushButton(QStringLiteral("禁用"), this);
    _disableButton->setObjectName(QStringLiteral("disableMarkerProjectionButton"));
    _skipButton = new QPushButton(QStringLiteral("跳过"), this);
    _skipButton->setObjectName(QStringLiteral("skipMarkerCandidateButton"));
    auto *close_button = new QPushButton(QStringLiteral("关闭"), this);
    buttons->addWidget(_confirmButton);
    buttons->addWidget(_blockButton);
    buttons->addWidget(_disableButton);
    buttons->addWidget(_skipButton);
    buttons->addStretch(1);
    buttons->addWidget(close_button);
    layout->addLayout(buttons);

    connect(_candidateCombo, &QComboBox::currentIndexChanged,
            this, [this](int) { refreshMeasurement(); });
    connect(_viewer, &DualImageViewer::markerCandidatePicked,
            this, &MarkerFocusMeasurementDialog::setCandidatePixel);
    connect(_confirmButton, &QPushButton::clicked,
            this, &MarkerFocusMeasurementDialog::confirmCandidate);
    connect(_blockButton, &QPushButton::clicked,
            this, [this]() { setCandidateState(true); });
    connect(_disableButton, &QPushButton::clicked,
            this, [this]() { setCandidateState(false); });
    connect(_skipButton, &QPushButton::clicked, this, [this]()
    {
        if (_candidateCombo->count() > 0)
        {
            _candidateCombo->setCurrentIndex(
                (_candidateCombo->currentIndex() + 1) % _candidateCombo->count());
        }
    });
    connect(close_button, &QPushButton::clicked, this, &QDialog::reject);
}

bool MarkerFocusMeasurementDialog::setContext(MarkerWorkspaceController *controller,
                                              ProjectData *projectData,
                                              const QString &markerId,
                                              const QString &anchorImagePath,
                                              const QString &candidateImagePath)
{
    if (!controller || !projectData || markerId.isEmpty() || anchorImagePath.isEmpty())
    {
        return false;
    }
    try
    {
        const auto &marker = controller->markerSet().marker(markerId);
        if (!projectionForPath(marker, anchorImagePath)) return false;
    }
    catch (const std::exception &)
    {
        return false;
    }

    _controller = controller;
    _projectData = projectData;
    _markerId = markerId;
    _anchorImagePath = anchorImagePath;
    generateGeometryPredictions();
    rebuildCandidates(candidateImagePath);
    refreshMeasurement();
    return _candidateCombo->count() > 0;
}

void MarkerFocusMeasurementDialog::generateGeometryPredictions()
{
    if (!_controller || !_projectData) return;

    QVector<control_points::MarkerCamera> cameras;
    const QJsonArray images = _projectData->coreFilesMeta().value(QStringLiteral("images")).toArray();
    cameras.reserve(images.size());
    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        xjw::Camera camera_model;
        if (!xjw::common::project::imageCameraFromEntry(image, &camera_model)
            || !camera_model.isValid()) continue;
        const xjw::Camera positive = camera_model.normalizedForPositiveDepth();
        if (!positive.isValid()) continue;

        control_points::MarkerCamera camera;
        camera.imageId = image.value(QStringLiteral("image_uuid")).toString();
        camera.imagePath = image.value(QStringLiteral("path")).toString();
        camera.intrinsics = cv::Matx33d(positive.focalX(), 0.0, positive.principalX(),
                                        0.0, positive.focalY(), positive.principalY(),
                                        0.0, 0.0, 1.0);
        const std::array<double, 9> rotation = positive.worldToCameraRotation();
        const std::array<double, 3> translation = positive.worldToCameraTranslation();
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                camera.rotation(row, column) = rotation[row * 3 + column];
            }
            camera.translation[row] = translation[row];
        }
        const QJsonObject camera_json = image.value(QStringLiteral("camera")).toObject();
        QSize image_size(camera_json.value(QStringLiteral("image_width")).toInt(),
                         camera_json.value(QStringLiteral("image_height")).toInt());
        if (!image_size.isValid()) image_size = QImageReader(camera.imagePath).size();
        camera.imageSize = image_size;

        const QString mask_path = xjw::common::project::ProjectIO::findMaskForImage(
            _projectData->currentProjectPath(), camera.imagePath);
        if (!mask_path.isEmpty())
        {
            const cv::Mat mask = xjw::common::io::readImage(mask_path, cv::IMREAD_GRAYSCALE);
            if (!mask.empty())
            {
                camera.acceptsPixel = [mask](const QPointF &pixel)
                {
                    const int column = qRound(pixel.x());
                    const int row = qRound(pixel.y());
                    return row >= 0 && row < mask.rows && column >= 0 && column < mask.cols
                        && mask.at<uchar>(row, column) == 0;
                };
            }
        }
        cameras.push_back(std::move(camera));
    }

    if (cameras.size() < 2) return;
    const auto prediction = control_points::MarkerProjectionPredictor::predict(
        _controller->markerSet().marker(_markerId), cameras);
    if (!prediction.triangulation.success || prediction.predictions.isEmpty()) return;
    QString error;
    if (!_controller->applyPredictedProjections(_markerId, prediction.predictions, &error))
    {
        _statusLabel->setText(error);
    }
}

void MarkerFocusMeasurementDialog::rebuildCandidates(const QString &requestedCandidate)
{
    _candidateCombo->clear();
    if (!_projectData || !_controller) return;
    const auto &marker = _controller->markerSet().marker(_markerId);
    QVector<QPair<int, QString>> candidates;
    const QJsonArray images = _projectData->coreFilesMeta().value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : images)
    {
        const QString path = value.toObject().value(QStringLiteral("path")).toString();
        if (path.isEmpty() || samePath(path, _anchorImagePath)) continue;
        const auto *projection = projectionForPath(marker, path);
        const int rank = projection ? static_cast<int>(projection->state) : 100;
        candidates.push_back({rank, path});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto &left, const auto &right)
    {
        return left.first == right.first
            ? left.second.compare(right.second, Qt::CaseInsensitive) < 0
            : left.first < right.first;
    });
    for (const auto &candidate : candidates)
    {
        _candidateCombo->addItem(QFileInfo(candidate.second).fileName(), candidate.second);
    }
    for (int index = 0; index < _candidateCombo->count(); ++index)
    {
        if (samePath(_candidateCombo->itemData(index).toString(), requestedCandidate))
        {
            _candidateCombo->setCurrentIndex(index);
            break;
        }
    }
}

void MarkerFocusMeasurementDialog::refreshMeasurement()
{
    _candidatePixel.reset();
    if (!_controller || _markerId.isEmpty() || currentCandidatePath().isEmpty()) return;
    const auto &marker = _controller->markerSet().marker(_markerId);
    const auto *anchor = projectionForPath(marker, _anchorImagePath);
    if (!anchor) return;
    const auto *candidate = projectionForPath(marker, currentCandidatePath());
    if (candidate && candidate->state != control_points::ProjectionState::Disabled)
    {
        _candidatePixel = candidate->xy;
    }
    _viewer->setMarkerMeasurement(_anchorImagePath,
                                  currentCandidatePath(),
                                  anchor->xy,
                                  _candidatePixel);
    _confirmButton->setEnabled(_candidatePixel.has_value());
    _blockButton->setEnabled(_candidatePixel.has_value());
    _disableButton->setEnabled(_candidatePixel.has_value());
    _statusLabel->setText(_candidatePixel.has_value()
                              ? QStringLiteral("候选位置：(%1, %2)，可确认或调整")
                                    .arg(_candidatePixel->x(), 0, 'f', 2)
                                    .arg(_candidatePixel->y(), 0, 'f', 2)
                              : QStringLiteral("在右侧影像中右键放置候选位置"));
}

void MarkerFocusMeasurementDialog::setCandidatePixel(const QPointF &pixel)
{
    _candidatePixel = pixel;
    if (!_controller) return;
    const auto &marker = _controller->markerSet().marker(_markerId);
    const auto *anchor = projectionForPath(marker, _anchorImagePath);
    if (anchor)
    {
        _viewer->setMarkerMeasurement(_anchorImagePath,
                                      currentCandidatePath(),
                                      anchor->xy,
                                      _candidatePixel);
    }
    _confirmButton->setEnabled(true);
    _blockButton->setEnabled(true);
    _disableButton->setEnabled(true);
    _statusLabel->setText(QStringLiteral("候选位置：(%1, %2)")
                              .arg(pixel.x(), 0, 'f', 2)
                              .arg(pixel.y(), 0, 'f', 2));
}

void MarkerFocusMeasurementDialog::confirmCandidate()
{
    if (!_controller || !_candidatePixel.has_value()) return;
    QString error;
    if (_controller->executePhotoCommand(MarkerPhotoCommand::PlaceExistingMarker,
                                         currentCandidatePath(),
                                         _candidatePixel.value(),
                                         _markerId,
                                         &error))
    {
        _statusLabel->setText(QStringLiteral("投影已人工确认"));
    }
    else
    {
        _statusLabel->setText(error);
    }
}

void MarkerFocusMeasurementDialog::setCandidateState(bool block)
{
    if (!_controller || !_candidatePixel.has_value()) return;
    QString error;
    const auto &marker = _controller->markerSet().marker(_markerId);
    if (!projectionForPath(marker, currentCandidatePath())
        && !_controller->executePhotoCommand(MarkerPhotoCommand::PlaceExistingMarker,
                                             currentCandidatePath(),
                                             _candidatePixel.value(),
                                             _markerId,
                                             &error))
    {
        _statusLabel->setText(error);
        return;
    }
    const MarkerPhotoCommand command = block
        ? MarkerPhotoCommand::BlockProjection
        : MarkerPhotoCommand::DisableProjection;
    if (!_controller->executePhotoCommand(command,
                                          currentCandidatePath(),
                                          {},
                                          _markerId,
                                          &error))
    {
        _statusLabel->setText(error);
        return;
    }
    _statusLabel->setText(block ? QStringLiteral("候选投影已屏蔽")
                                : QStringLiteral("候选投影已禁用"));
}

QString MarkerFocusMeasurementDialog::currentCandidatePath() const
{
    return _candidateCombo->currentData().toString();
}

} // namespace xjw::gui::markers
