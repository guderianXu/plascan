// =============================================================================
// 文件: DisparityHeatmapOverlay.h
// 功能: 视差图热力图叠加层（用于密集匹配可视化）
// =============================================================================
#pragma once

#include <QImage>
#include <QPixmap>
#include <QWidget>
#include <opencv2/core.hpp>

class DisparityHeatmapOverlay : public QWidget
{
    Q_OBJECT
public:
    explicit DisparityHeatmapOverlay(QWidget *parent = nullptr);

    bool loadDisparity(const QString &filepath);
    bool loadDisparity(const cv::Mat &disparity);

    void setOpacity(float opacity);
    void setDisparityRange(float min, float max);
    void setAutoRange(bool enabled);
    void setColormap(int cvColormap);
    void setShowInvalid(bool show);

    float opacity() const { return _opacity; }
    QImage heatmapImage() const { return _heatmapImage; }

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    void rebuildHeatmap();

    cv::Mat _disparity;
    QImage  _heatmapImage;
    QPixmap _heatmap;
    float   _opacity     = 0.6f;
    float   _dispMin     = 0.0f;
    float   _dispMax     = 256.0f;
    bool    _autoRange   = true;
    int     _colormap    = 2; // COLORMAP_JET
    bool    _showInvalid = false;
};
