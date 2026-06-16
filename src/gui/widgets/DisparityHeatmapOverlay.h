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

    float opacity() const { return m_opacity; }
    QImage heatmapImage() const { return m_heatmapImage; }

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    void rebuildHeatmap();

    cv::Mat m_disparity;
    QImage  m_heatmapImage;
    QPixmap m_heatmap;
    float   m_opacity     = 0.6f;
    float   m_dispMin     = 0.0f;
    float   m_dispMax     = 256.0f;
    bool    m_autoRange   = true;
    int     m_colormap    = 2; // COLORMAP_JET
    bool    m_showInvalid = false;
};
