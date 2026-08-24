#pragma once

#include "image/MaskEditorSettingsDialog.h"

#include <QImage>
#include <QList>
#include <QObject>
#include <QPointF>
#include <QRectF>
#include <Qt>

#include <vector>

class QGraphicsPathItem;
class QGraphicsPixmapItem;
class QGraphicsScene;
namespace cv
{
    class Mat;
}

class MaskEditor final : public QObject
{
    Q_OBJECT

public:
    enum class Tool
    {
        None,
        Rectangle,
        Scissors,
        SmartPaint,
        MagicWand
    };
    Q_ENUM(Tool)

    explicit MaskEditor(QGraphicsScene* scene, QObject* parent = nullptr);

    void setImage(const QImage& image, const QRectF& sceneBounds);
    void setMask(const QImage& mask);
    void setTool(Tool tool);
    Tool tool() const noexcept
    {
        return _tool;
    }
    void setSettings(const MaskEditorSettings& settings);
    void setOverlayVisible(bool visible);
    quint64 revision() const noexcept
    {
        return _revision;
    }
    bool selectionActive() const noexcept
    {
        return _selectionActive;
    }
    bool canUndo() const noexcept
    {
        return !_undoStack.isEmpty();
    }
    bool canRedo() const noexcept
    {
        return !_redoStack.isEmpty();
    }

    bool mousePress(const QPointF& scenePosition, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    bool mouseMove(const QPointF& scenePosition, Qt::MouseButtons buttons);
    bool mouseRelease(const QPointF& scenePosition, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    bool mouseDoubleClick(const QPointF& scenePosition, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);

public slots:
    void resetSelection();
    void undo();
    void redo();

signals:
    void maskChanged(const QImage& mask, const QString& method, quint64 revision);
    void selectionActiveChanged(bool active);

private:
    QPoint imagePoint(const QPointF& scenePosition) const;
    QPointF scenePoint(const QPoint& imagePosition) const;
    bool containsScenePoint(const QPointF& scenePosition) const;
    void applySelection(const cv::Mat& selection, const QString& method, Qt::KeyboardModifiers modifiers);
    void finishScissors(Qt::KeyboardModifiers modifiers);
    void renderOverlay();
    void renderPreview();
    void setSelectionActive(bool active);
    void pushUndoState();
    cv::Mat imageMat() const;
    cv::Mat maskMat();

    QGraphicsScene* _scene{};
    QGraphicsPixmapItem* _overlayItem{};
    QGraphicsPathItem* _previewItem{};
    QImage _image;
    QImage _mask;
    QRectF _sceneBounds;
    MaskEditorSettings _settings;
    Tool _tool{Tool::None};
    QPoint _dragStart;
    QPoint _dragCurrent;
    std::vector<QPoint> _stroke;
    std::vector<QPoint> _scissorsPath;
    std::vector<QPoint> _scissorsAnchors;
    QList<QImage> _undoStack;
    QList<QImage> _redoStack;
    quint64 _revision{};
    bool _dragging{};
    bool _selectionActive{};
    bool _overlayVisible{true};
};
