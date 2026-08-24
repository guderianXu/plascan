#include "MaskEditor.h"

#include "InteractiveMaskAlgorithms.h"

#include <QGraphicsPathItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace
{

    constexpr int MaximumUndoStates = 20;

    QImage normalizedMask(const QImage& source, const QSize& size)
    {
        if (size.isEmpty())
        {
            return {};
        }
        if (source.isNull())
        {
            QImage empty(size, QImage::Format_Grayscale8);
            empty.fill(0);
            return empty;
        }
        QImage mask = source.convertToFormat(QImage::Format_Grayscale8);
        if (mask.size() != size)
        {
            mask = mask.scaled(size, Qt::IgnoreAspectRatio, Qt::FastTransformation);
        }
        return mask;
    }

} // namespace

MaskEditor::MaskEditor(QGraphicsScene* scene, QObject* parent)
    : QObject(parent), _scene(scene), _settings(loadMaskEditorSettings())
{
}

void MaskEditor::setImage(const QImage& image, const QRectF& sceneBounds)
{
    resetSelection();
    // The core selection algorithms consume OpenCV's conventional BGR byte order.
    _image = image.convertToFormat(QImage::Format_BGR888);
    _sceneBounds = sceneBounds;
    _mask = normalizedMask({}, _image.size());
    _undoStack.clear();
    _redoStack.clear();
    _revision = 0;
    if (_image.isNull() || _sceneBounds.isEmpty())
    {
        if (_overlayItem)
        {
            _overlayItem->setVisible(false);
        }
        return;
    }
    renderOverlay();
}

void MaskEditor::setMask(const QImage& mask)
{
    _mask = normalizedMask(mask, _image.size());
    _undoStack.clear();
    _redoStack.clear();
    renderOverlay();
}

void MaskEditor::setTool(Tool tool)
{
    if (_tool == tool)
    {
        return;
    }
    resetSelection();
    _tool = tool;
}

void MaskEditor::setSettings(const MaskEditorSettings& settings)
{
    _settings = settings;
    renderOverlay();
    renderPreview();
}

void MaskEditor::setOverlayVisible(bool visible)
{
    _overlayVisible = visible;
    if (_overlayItem)
    {
        _overlayItem->setVisible(visible);
    }
}

bool MaskEditor::mousePress(const QPointF& scenePosition, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (button != Qt::LeftButton || _tool == Tool::None || !containsScenePoint(scenePosition))
    {
        return false;
    }

    const QPoint point = imagePoint(scenePosition);
    if (_tool == Tool::MagicWand)
    {
        applySelection(
            xjw::mask::magicWandSelection(imageMat(), cv::Point(point.x(), point.y()), _settings.colorTolerance),
            QStringLiteral("interactive_magic_wand"),
            modifiers);
        return true;
    }
    if (_tool == Tool::Scissors)
    {
        if (_scissorsAnchors.empty())
        {
            _scissorsAnchors.push_back(point);
            _scissorsPath.push_back(point);
        }
        else
        {
            const QPoint previous = _scissorsAnchors.back();
            const std::vector<cv::Point> segment = xjw::mask::edgeSnappedPath(imageMat(),
                                                                              cv::Point(previous.x(), previous.y()),
                                                                              cv::Point(point.x(), point.y()),
                                                                              _settings.scissorsSearchRadius);
            for (size_t index = 1; index < segment.size(); ++index)
            {
                _scissorsPath.emplace_back(segment[index].x, segment[index].y);
            }
            _scissorsAnchors.push_back(point);
        }
        setSelectionActive(true);
        renderPreview();
        return true;
    }

    _dragStart = point;
    _dragCurrent = point;
    _dragging = true;
    _stroke = {point};
    setSelectionActive(true);
    renderPreview();
    return true;
}

bool MaskEditor::mouseMove(const QPointF& scenePosition, Qt::MouseButtons buttons)
{
    if (!_dragging || !(buttons & Qt::LeftButton) || !containsScenePoint(scenePosition))
    {
        return false;
    }
    _dragCurrent = imagePoint(scenePosition);
    if (_tool == Tool::SmartPaint)
    {
        const QPoint previous = _stroke.back();
        const int minimumStep = std::max(1, _settings.brushRadius / 4);
        if ((previous - _dragCurrent).manhattanLength() >= minimumStep)
        {
            _stroke.push_back(_dragCurrent);
        }
    }
    renderPreview();
    return true;
}

bool MaskEditor::mouseRelease(const QPointF& scenePosition, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (button == Qt::LeftButton && _tool == Tool::Scissors && _selectionActive)
    {
        return true;
    }
    if (button != Qt::LeftButton || !_dragging)
    {
        return false;
    }
    if (containsScenePoint(scenePosition))
    {
        _dragCurrent = imagePoint(scenePosition);
        if (_tool == Tool::SmartPaint && (_stroke.empty() || _stroke.back() != _dragCurrent))
        {
            _stroke.push_back(_dragCurrent);
        }
    }

    if (_tool == Tool::Rectangle)
    {
        const QRect normalized = QRect(_dragStart, _dragCurrent).normalized().adjusted(0, 0, 1, 1);
        applySelection(xjw::mask::rectangleSelection(
                           cv::Size(_image.width(), _image.height()),
                           cv::Rect(normalized.x(), normalized.y(), normalized.width(), normalized.height())),
                       QStringLiteral("interactive_rectangle"),
                       modifiers);
    }
    else if (_tool == Tool::SmartPaint)
    {
        std::vector<cv::Point> stroke;
        stroke.reserve(_stroke.size());
        for (const QPoint& point : _stroke)
        {
            stroke.emplace_back(point.x(), point.y());
        }
        applySelection(
            xjw::mask::smartBrushSelection(imageMat(), stroke, _settings.brushRadius, _settings.colorTolerance),
            QStringLiteral("interactive_smart_paint"),
            modifiers);
    }
    resetSelection();
    return true;
}

bool MaskEditor::mouseDoubleClick(const QPointF& scenePosition, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (button != Qt::LeftButton || _tool != Tool::Scissors || !containsScenePoint(scenePosition))
    {
        return false;
    }
    finishScissors(modifiers);
    return true;
}

void MaskEditor::resetSelection()
{
    _dragging = false;
    _stroke.clear();
    _scissorsPath.clear();
    _scissorsAnchors.clear();
    if (_previewItem && _scene)
    {
        _scene->removeItem(_previewItem);
        delete _previewItem;
        _previewItem = nullptr;
    }
    setSelectionActive(false);
}

void MaskEditor::undo()
{
    if (_undoStack.isEmpty())
    {
        return;
    }
    _redoStack.push_back(_mask);
    _mask = _undoStack.takeLast();
    ++_revision;
    renderOverlay();
    emit maskChanged(_mask, QStringLiteral("interactive_undo"), _revision);
}

void MaskEditor::redo()
{
    if (_redoStack.isEmpty())
    {
        return;
    }
    _undoStack.push_back(_mask);
    _mask = _redoStack.takeLast();
    ++_revision;
    renderOverlay();
    emit maskChanged(_mask, QStringLiteral("interactive_redo"), _revision);
}

QPoint MaskEditor::imagePoint(const QPointF& scenePosition) const
{
    const double xScale = _image.width() / std::max(1.0, _sceneBounds.width());
    const double yScale = _image.height() / std::max(1.0, _sceneBounds.height());
    return QPoint(
        std::clamp(qRound((scenePosition.x() - _sceneBounds.left()) * xScale), 0, std::max(0, _image.width() - 1)),
        std::clamp(qRound((scenePosition.y() - _sceneBounds.top()) * yScale), 0, std::max(0, _image.height() - 1)));
}

QPointF MaskEditor::scenePoint(const QPoint& imagePosition) const
{
    return QPointF(_sceneBounds.left() + imagePosition.x() * _sceneBounds.width() / std::max(1, _image.width()),
                   _sceneBounds.top() + imagePosition.y() * _sceneBounds.height() / std::max(1, _image.height()));
}

bool MaskEditor::containsScenePoint(const QPointF& scenePosition) const
{
    return !_image.isNull() && _sceneBounds.contains(scenePosition);
}

void MaskEditor::applySelection(const cv::Mat& selection, const QString& method, Qt::KeyboardModifiers modifiers)
{
    if (selection.empty())
    {
        return;
    }
    pushUndoState();
    cv::Mat mask = maskMat();
    const bool exclude = _settings.excludePixels != modifiers.testFlag(Qt::AltModifier);
    xjw::mask::applySelectionToMask(&mask, selection, exclude);
    ++_revision;
    renderOverlay();
    emit maskChanged(_mask, method, _revision);
}

void MaskEditor::finishScissors(Qt::KeyboardModifiers modifiers)
{
    if (_scissorsAnchors.size() < 3 || _scissorsPath.size() < 3)
    {
        resetSelection();
        return;
    }
    const QPoint last = _scissorsAnchors.back();
    const QPoint first = _scissorsAnchors.front();
    const std::vector<cv::Point> closing = xjw::mask::edgeSnappedPath(
        imageMat(), cv::Point(last.x(), last.y()), cv::Point(first.x(), first.y()), _settings.scissorsSearchRadius);
    for (size_t index = 1; index < closing.size(); ++index)
    {
        _scissorsPath.emplace_back(closing[index].x, closing[index].y);
    }

    std::vector<cv::Point> polygon;
    polygon.reserve(_scissorsPath.size());
    for (const QPoint& point : _scissorsPath)
    {
        polygon.emplace_back(point.x(), point.y());
    }
    cv::Mat selection = cv::Mat::zeros(_image.height(), _image.width(), CV_8UC1);
    cv::fillPoly(selection, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(255));
    applySelection(selection, QStringLiteral("interactive_scissors"), modifiers);
    resetSelection();
}

void MaskEditor::renderOverlay()
{
    if (!_scene || _mask.isNull() || _sceneBounds.isEmpty())
    {
        return;
    }
    QImage overlay(_mask.size(), QImage::Format_ARGB32);
    const int maximumAlpha = qRound(255.0 * std::clamp(_settings.overlayOpacity, 0, 100) / 100.0);
    for (int y = 0; y < _mask.height(); ++y)
    {
        const unsigned char* maskLine = _mask.constScanLine(y);
        QRgb* overlayLine = reinterpret_cast<QRgb*>(overlay.scanLine(y));
        for (int x = 0; x < _mask.width(); ++x)
        {
            const int alpha = maximumAlpha * maskLine[x] / 255;
            overlayLine[x] = qRgba(235, 52, 62, alpha);
        }
    }

    if (!_overlayItem)
    {
        _overlayItem = _scene->addPixmap(QPixmap::fromImage(overlay));
        _overlayItem->setZValue(35.0);
    }
    else
    {
        _overlayItem->setPixmap(QPixmap::fromImage(overlay));
    }
    _overlayItem->setPos(_sceneBounds.topLeft());
    _overlayItem->setTransform(QTransform::fromScale(_sceneBounds.width() / std::max(1, overlay.width()),
                                                     _sceneBounds.height() / std::max(1, overlay.height())));
    _overlayItem->setVisible(_overlayVisible);
}

void MaskEditor::renderPreview()
{
    if (!_scene)
    {
        return;
    }
    if (!_previewItem)
    {
        _previewItem = _scene->addPath(QPainterPath());
        _previewItem->setZValue(60.0);
    }

    QPainterPath path;
    QPen pen(QColor(255, 235, 80, 245), 2.0, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin);
    pen.setCosmetic(true);
    if (_tool == Tool::Rectangle && _dragging)
    {
        path.addRect(QRectF(scenePoint(_dragStart), scenePoint(_dragCurrent)).normalized());
    }
    else if (_tool == Tool::SmartPaint && _dragging && !_stroke.empty())
    {
        path.moveTo(scenePoint(_stroke.front()));
        for (const QPoint& point : _stroke)
        {
            path.lineTo(scenePoint(point));
        }
        const double scale = _sceneBounds.width() / std::max(1, _image.width());
        pen.setWidthF(std::max(2.0, _settings.brushRadius * 2.0 * scale));
        pen.setStyle(Qt::SolidLine);
        pen.setColor(QColor(255, 90, 90, 150));
    }
    else if (_tool == Tool::Scissors && !_scissorsPath.empty())
    {
        path.moveTo(scenePoint(_scissorsPath.front()));
        for (const QPoint& point : _scissorsPath)
        {
            path.lineTo(scenePoint(point));
        }
    }
    _previewItem->setPen(pen);
    _previewItem->setPath(path);
}

void MaskEditor::setSelectionActive(bool active)
{
    if (_selectionActive == active)
    {
        return;
    }
    _selectionActive = active;
    emit selectionActiveChanged(active);
}

void MaskEditor::pushUndoState()
{
    _undoStack.push_back(_mask);
    while (_undoStack.size() > MaximumUndoStates)
    {
        _undoStack.removeFirst();
    }
    _redoStack.clear();
    _mask.detach();
}

cv::Mat MaskEditor::imageMat() const
{
    if (_image.isNull())
    {
        return {};
    }
    return cv::Mat(_image.height(),
                   _image.width(),
                   CV_8UC3,
                   const_cast<unsigned char*>(_image.constBits()),
                   static_cast<size_t>(_image.bytesPerLine()));
}

cv::Mat MaskEditor::maskMat()
{
    if (_mask.isNull())
    {
        return {};
    }
    _mask.detach();
    return cv::Mat(_mask.height(), _mask.width(), CV_8UC1, _mask.bits(), static_cast<size_t>(_mask.bytesPerLine()));
}
