// =============================================================================
// 文件: DisparityHeatmapOverlay.cpp
// 功能: 视差热力图叠加实现
// =============================================================================
#include "DisparityHeatmapOverlay.h"
#include <QPainter>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

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
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::rebuildHeatmap()
{
    if (m_disparity.empty())
    {
        m_heatmapImage = QImage();
        m_heatmap = QPixmap();
        update();
        return;
    }

    cv::Mat dispF;
    m_disparity.convertTo(dispF, CV_32FC1);
    cv::Mat validMask = dispF > 0;

    float dMin = m_dispMin, dMax = m_dispMax;
    if (m_autoRange)
    {
        double minVal = 0.0;
        double maxVal = 1.0;
        if (cv::countNonZero(validMask) > 0)
        {
            cv::minMaxLoc(dispF, &minVal, &maxVal, nullptr, nullptr, validMask);
        }
        dMin = static_cast<float>(minVal);
        dMax = static_cast<float>(maxVal);
        if (dMax <= dMin) dMax = dMin + 1.0f;
    }

    cv::Mat clamped = cv::max(dMin, cv::min(dMax, dispF));
    clamped = (clamped - dMin) / (dMax - dMin) * 255.0;
    cv::Mat normalized;
    clamped.convertTo(normalized, CV_8UC1);

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, m_colormap);

    cv::Mat rgb;
    cv::cvtColor(colored, rgb, cv::COLOR_BGR2RGB);

    m_heatmapImage = QImage(rgb.cols, rgb.rows, QImage::Format_RGBA8888);
    for (int row = 0; row < rgb.rows; ++row)
    {
        const uchar *rgbRow = rgb.ptr<uchar>(row);
        const uchar *validRow = validMask.ptr<uchar>(row);
        uchar *outRow = m_heatmapImage.scanLine(row);
        std::vector<uchar> alphaRow(static_cast<size_t>(rgb.cols), 0);
        for (int col = 0; col < rgb.cols; ++col)
        {
            uchar *pixel = outRow + col * 4;
            if (validRow[col])
            {
                pixel[0] = rgbRow[col * 3 + 0];
                pixel[1] = rgbRow[col * 3 + 1];
                pixel[2] = rgbRow[col * 3 + 2];
            }
            else
            {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 0;
            }

            if (m_showInvalid)
            {
                alphaRow[col] = 255;
            }
            else
            {
                alphaRow[col] = validRow[col] ? 255 : 0;
            }
            pixel[3] = alphaRow[col];
        }
    }

    m_heatmap = QPixmap::fromImage(m_heatmapImage);
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
