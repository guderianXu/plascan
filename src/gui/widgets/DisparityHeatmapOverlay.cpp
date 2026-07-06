// =============================================================================
// 文件: DisparityHeatmapOverlay.cpp
// 功能: 视差热力图叠加实现
// =============================================================================
#include "DisparityHeatmapOverlay.h"
#include "io/PathIO.h"
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
    cv::Mat disp = xjw::common::io::readImage(filepath, cv::IMREAD_UNCHANGED);
    if (disp.empty()) return false;
    return loadDisparity(disp);
}

bool DisparityHeatmapOverlay::loadDisparity(const cv::Mat &disparity)
{
    if (disparity.empty()) return false;
    _disparity = disparity.clone();
    rebuildHeatmap();
    return true;
}

void DisparityHeatmapOverlay::setOpacity(float opacity)
{
    _opacity = std::max(0.0f, std::min(1.0f, opacity));
    update();
}

void DisparityHeatmapOverlay::setDisparityRange(float min, float max)
{
    _dispMin = min;
    _dispMax = max;
    _autoRange = false;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setAutoRange(bool enabled)
{
    _autoRange = enabled;
    if (enabled) rebuildHeatmap();
}

void DisparityHeatmapOverlay::setColormap(int cvColormap)
{
    _colormap = cvColormap;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setShowInvalid(bool show)
{
    _showInvalid = show;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::rebuildHeatmap()
{
    if (_disparity.empty())
    {
        _heatmapImage = QImage();
        _heatmap = QPixmap();
        update();
        return;
    }

    cv::Mat dispF;
    _disparity.convertTo(dispF, CV_32FC1);
    cv::Mat validMask = dispF > 0;

    float dMin = _dispMin, dMax = _dispMax;
    if (_autoRange)
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
    cv::applyColorMap(normalized, colored, _colormap);

    cv::Mat rgb;
    cv::cvtColor(colored, rgb, cv::COLOR_BGR2RGB);

    _heatmapImage = QImage(rgb.cols, rgb.rows, QImage::Format_RGBA8888);
    for (int row = 0; row < rgb.rows; ++row)
    {
        const uchar *rgbRow = rgb.ptr<uchar>(row);
        const uchar *validRow = validMask.ptr<uchar>(row);
        uchar *outRow = _heatmapImage.scanLine(row);
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

            if (_showInvalid)
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

    _heatmap = QPixmap::fromImage(_heatmapImage);
    update();
}

void DisparityHeatmapOverlay::paintEvent(QPaintEvent *)
{
    if (_heatmap.isNull()) return;
    QPainter painter(this);
    painter.setOpacity(_opacity);
    painter.drawPixmap(rect(), _heatmap.scaled(
        size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
