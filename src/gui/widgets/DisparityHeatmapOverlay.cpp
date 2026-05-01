// =============================================================================
// 文件: DisparityHeatmapOverlay.cpp
// 功能: 视差热力图叠加实现
// =============================================================================
#include "DisparityHeatmapOverlay.h"
#include <QPainter>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

DisparityHeatmapOverlay::DisparityHeatmapOverlay(QWidget *parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_TransparentForMouseEvents);
}

bool DisparityHeatmapOverlay::loadDisparity(const QString &filepath)
{
    cv::Mat disp = cv::imread(filepath.toStdString(), cv::IMREAD_UNCHANGED);
    if (disp.empty()) return false;
    return loadDisparity(disp);
}

bool DisparityHeatmapOverlay::loadDisparity(const cv::Mat &disparity)
{
    if (disparity.empty()) return false;
    m_disparity = disparity.clone();
    rebuildHeatmap();
    return true;
}

void DisparityHeatmapOverlay::setOpacity(float opacity)
{
    m_opacity = std::max(0.0f, std::min(1.0f, opacity));
    update();
}

void DisparityHeatmapOverlay::setDisparityRange(float min, float max)
{
    m_dispMin = min;
    m_dispMax = max;
    m_autoRange = false;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setAutoRange(bool enabled)
{
    m_autoRange = enabled;
    if (enabled) rebuildHeatmap();
}

void DisparityHeatmapOverlay::setColormap(int cvColormap)
{
    m_colormap = cvColormap;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setShowInvalid(bool show)
{
    m_showInvalid = show;
    update();
}

void DisparityHeatmapOverlay::rebuildHeatmap()
{
    if (m_disparity.empty()) return;

    float dMin = m_dispMin, dMax = m_dispMax;
    if (m_autoRange)
    {
        double minVal, maxVal;
        cv::Mat mask = (m_disparity > 0);
        cv::minMaxLoc(m_disparity, &minVal, &maxVal, nullptr, nullptr, mask);
        dMin = static_cast<float>(minVal);
        dMax = static_cast<float>(maxVal);
        if (dMax <= dMin) dMax = dMin + 1.0f;
    }

    cv::Mat dispF;
    m_disparity.convertTo(dispF, CV_32FC1);
    cv::Mat clamped = cv::max(dMin, cv::min(dMax, dispF));
    clamped = (clamped - dMin) / (dMax - dMin) * 255.0;
    cv::Mat normalized;
    clamped.convertTo(normalized, CV_8UC1);

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, m_colormap);

    cv::Mat rgb;
    cv::cvtColor(colored, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    m_heatmap = QPixmap::fromImage(qimg.copy());
    update();
}

void DisparityHeatmapOverlay::paintEvent(QPaintEvent *)
{
    if (m_heatmap.isNull()) return;
    QPainter painter(this);
    painter.setOpacity(m_opacity);
    painter.drawPixmap(rect(), m_heatmap.scaled(
        size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
