#include "GlobalTerrainReportRenderer.h"

#include "io/ImageIO.h"

#include <QDir>
#include <QCoreApplication>
#include <QFileInfo>
#include <QFont>
#include <QFontDatabase>
#include <QImage>
#include <QGuiApplication>
#include <QPainter>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace xjw
{
namespace
{

struct RenderedRaster
{
    cv::Mat image;
    double minimum = 0.0;
    double maximum = 1.0;
    int colorMap = cv::COLORMAP_TURBO;
};

std::pair<double, double> robustRange(const cv::Mat &values, const cv::Mat &mask)
{
    std::vector<float> samples;
    const int stride = std::max(1, static_cast<int>(std::sqrt(
        static_cast<double>(values.total()) / 200000.0)));
    for (int row = 0; row < values.rows; row += stride)
    {
        for (int col = 0; col < values.cols; col += stride)
        {
            if (mask.at<uchar>(row, col) != 0)
            {
                const float value = values.at<float>(row, col);
                if (std::isfinite(value))
                {
                    samples.push_back(value);
                }
            }
        }
    }
    if (samples.empty())
    {
        return {0.0, 1.0};
    }
    std::sort(samples.begin(), samples.end());
    const auto low = static_cast<std::size_t>(0.02 * (samples.size() - 1));
    const auto high = static_cast<std::size_t>(0.98 * (samples.size() - 1));
    double minimum = samples[low];
    double maximum = samples[high];
    if (maximum - minimum < 1e-9)
    {
        maximum = minimum + 1.0;
    }
    return {minimum, maximum};
}

cv::Mat northUp(const cv::Mat &bgr, const cv::Mat &mask)
{
    cv::Mat image;
    cv::flip(bgr, image, 0);
    cv::Mat north_mask;
    cv::flip(mask, north_mask, 0);
    image.setTo(cv::Scalar(35, 38, 42), north_mask == 0);
    return image;
}

RenderedRaster colorize(const cv::Mat &values,
                        const cv::Mat &mask,
                        int colorMap,
                        double fixedMinimum = std::numeric_limits<double>::quiet_NaN(),
                        double fixedMaximum = std::numeric_limits<double>::quiet_NaN())
{
    auto [minimum, maximum] = robustRange(values, mask);
    if (std::isfinite(fixedMinimum)) minimum = fixedMinimum;
    if (std::isfinite(fixedMaximum)) maximum = fixedMaximum;

    cv::Mat normalized(values.size(), CV_8U, cv::Scalar(0));
    const double scale = 255.0 / std::max(1e-12, maximum - minimum);
    for (int row = 0; row < values.rows; ++row)
    {
        for (int col = 0; col < values.cols; ++col)
        {
            if (mask.at<uchar>(row, col) != 0)
            {
                normalized.at<uchar>(row, col) = static_cast<uchar>(std::clamp(
                    std::lround((values.at<float>(row, col) - minimum) * scale), 0L, 255L));
            }
        }
    }
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, colorMap);
    return {northUp(colored, mask), minimum, maximum, colorMap};
}

RenderedRaster hillshade(const DemGridData &dem, double referenceRadiusM)
{
    cv::Mat shade(dem.height, dem.width, CV_32F, cv::Scalar(0.0f));
    cv::Mat shade_mask(dem.height, dem.width, CV_8U, cv::Scalar(0));
    const cv::Vec3d light(-0.45, 0.55, 0.70);
    const double longitude_step_rad = std::abs(dem.stepX) * CV_PI / 180.0;
    const double latitude_step_rad = std::abs(dem.stepY) * CV_PI / 180.0;
    const double north_south_span =
        2.0 * referenceRadiusM * latitude_step_rad;
    for (int row = 1; row + 1 < dem.height; ++row)
    {
        for (int col = 0; col < dem.width; ++col)
        {
            const int left = (col + dem.width - 1) % dem.width;
            const int right = (col + 1) % dem.width;
            if (dem.validMask.at<uchar>(row, col) == 0
                || dem.validMask.at<uchar>(row, left) == 0
                || dem.validMask.at<uchar>(row, right) == 0
                || dem.validMask.at<uchar>(row - 1, col) == 0
                || dem.validMask.at<uchar>(row + 1, col) == 0)
            {
                continue;
            }
            const double latitude_rad = (dem.minY + row * dem.stepY) * CV_PI / 180.0;
            const double east_west_span = 2.0 * referenceRadiusM
                * std::max(1.0e-6, std::abs(std::cos(latitude_rad)))
                * longitude_step_rad;
            if (east_west_span <= 0.0 || north_south_span <= 0.0)
            {
                continue;
            }
            const double dz_dx = (dem.elevation.at<float>(row, right)
                - dem.elevation.at<float>(row, left)) / east_west_span;
            const double dz_dy = (dem.elevation.at<float>(row + 1, col)
                - dem.elevation.at<float>(row - 1, col)) / north_south_span;
            const cv::Vec3d normal(-dz_dx, -dz_dy, 1.0);
            shade.at<float>(row, col) = static_cast<float>(std::clamp(
                normal.dot(light) / (cv::norm(normal) * cv::norm(light)), 0.0, 1.0));
            shade_mask.at<uchar>(row, col) = 255;
        }
    }
    return colorize(shade, shade_mask, cv::COLORMAP_BONE, 0.0, 1.0);
}

void centeredText(cv::Mat &canvas,
                  const std::string &text,
                  int centerX,
                  int baselineY,
                  double scale,
                  int thickness = 1)
{
    int baseline = 0;
    const cv::Size size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    cv::putText(canvas, text, cv::Point(centerX - size.width / 2, baselineY),
                cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(25, 25, 25),
                thickness, cv::LINE_AA);
}

double normalizedLongitude(double longitude)
{
    const double wrapped = std::fmod(longitude, 360.0);
    return wrapped < 0.0 ? wrapped + 360.0 : wrapped;
}

std::string longitudeLabel(double longitude)
{
    const double normalized = normalizedLongitude(longitude);
    if (std::abs(normalized - std::round(normalized)) < 1.0e-6)
    {
        return std::to_string(static_cast<int>(std::lround(normalized)));
    }
    return cv::format("%.1f", normalized);
}

QString reportTitleFontFamily()
{
    const QStringList candidates{
        QStringLiteral("Microsoft YaHei UI"),
        QStringLiteral("Microsoft YaHei"),
        QStringLiteral("Noto Sans CJK SC"),
        QStringLiteral("Noto Sans SC"),
        QStringLiteral("WenQuanYi Micro Hei"),
        QStringLiteral("SimHei")};
    const QStringList available_families = QFontDatabase::families();
    for (const QString &candidate : candidates)
    {
        for (const QString &available : available_families)
        {
            if (candidate.compare(available, Qt::CaseInsensitive) == 0)
            {
                return available;
            }
        }
    }

    QStringList font_files;
#if defined(Q_OS_WIN)
    QString windows_directory = qEnvironmentVariable("WINDIR");
    if (windows_directory.isEmpty())
    {
        windows_directory = QStringLiteral("C:/Windows");
    }
    const QString fonts_directory = QDir(windows_directory).filePath(
        QStringLiteral("Fonts"));
    font_files = {
        QDir(fonts_directory).filePath(QStringLiteral("msyh.ttc")),
        QDir(fonts_directory).filePath(QStringLiteral("NotoSansSC-VF.ttf")),
        QDir(fonts_directory).filePath(QStringLiteral("simhei.ttf"))};
#elif defined(Q_OS_MACOS)
    font_files = {QStringLiteral("/System/Library/Fonts/PingFang.ttc")};
#else
    font_files = {
        QStringLiteral("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        QStringLiteral("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")};
#endif
    for (const QString &font_file : font_files)
    {
        if (!QFileInfo::exists(font_file))
        {
            continue;
        }
        const int font_id = QFontDatabase::addApplicationFont(font_file);
        const QStringList loaded_families = QFontDatabase::applicationFontFamilies(font_id);
        if (!loaded_families.isEmpty())
        {
            return loaded_families.first();
        }
    }
    return {};
}

void drawAxes(cv::Mat &canvas,
              const cv::Rect &imageRect,
              double centralMeridianDeg)
{
    cv::rectangle(canvas, imageRect, cv::Scalar(35, 35, 35), 1, cv::LINE_AA);
    for (int tick = 0; tick <= 6; ++tick)
    {
        const int x = imageRect.x + tick * imageRect.width / 6;
        cv::line(canvas, cv::Point(x, imageRect.y + imageRect.height),
                 cv::Point(x, imageRect.y + imageRect.height + 5), cv::Scalar(30, 30, 30));
        centeredText(canvas, longitudeLabel(centralMeridianDeg + tick * 60.0), x,
                     imageRect.y + imageRect.height + 23, 0.40);

        const int y = imageRect.y + imageRect.height - tick * imageRect.height / 6;
        cv::line(canvas, cv::Point(imageRect.x - 5, y), cv::Point(imageRect.x, y),
                 cv::Scalar(30, 30, 30));
        cv::putText(canvas, std::to_string(-90 + tick * 30),
                    cv::Point(imageRect.x - 48, y + 5), cv::FONT_HERSHEY_SIMPLEX,
                    0.38, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    }
    centeredText(canvas, "body-fixed positive-east longitude (deg)",
                 imageRect.x + imageRect.width / 2,
                 imageRect.y + imageRect.height + 43, 0.45);
    cv::putText(canvas, "latitude", cv::Point(imageRect.x - 52, imageRect.y - 7),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
}

void drawLegend(cv::Mat &canvas,
                const cv::Rect &imageRect,
                const RenderedRaster &raster,
                const std::string &unit)
{
    cv::Mat ramp(256, 1, CV_8U);
    for (int row = 0; row < 256; ++row)
    {
        ramp.at<uchar>(row, 0) = static_cast<uchar>(255 - row);
    }
    cv::Mat colored_ramp;
    cv::applyColorMap(ramp, colored_ramp, raster.colorMap);
    const cv::Rect legend(imageRect.x + imageRect.width + 14, imageRect.y, 16, imageRect.height);
    cv::Mat resized;
    cv::resize(colored_ramp, resized, legend.size(), 0.0, 0.0, cv::INTER_NEAREST);
    resized.copyTo(canvas(legend));
    cv::rectangle(canvas, legend, cv::Scalar(30, 30, 30));
    cv::putText(canvas, cv::format("%.1f", raster.maximum),
                cv::Point(legend.x + 22, legend.y + 8), cv::FONT_HERSHEY_SIMPLEX,
                0.38, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    cv::putText(canvas, cv::format("%.1f", raster.minimum),
                cv::Point(legend.x + 22, legend.y + legend.height), cv::FONT_HERSHEY_SIMPLEX,
                0.38, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    cv::putText(canvas, unit, cv::Point(legend.x + 22, legend.y + legend.height / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
}

void drawPanel(cv::Mat &canvas,
               const cv::Rect &panel,
               const std::string &title,
               const RenderedRaster &raster,
               const std::string &legendUnit,
               bool withLegend,
               double centralMeridianDeg)
{
    centeredText(canvas, title, panel.x + panel.width / 2, panel.y + 24, 0.60, 1);
    const int right_margin = withLegend ? 95 : 24;
    const cv::Rect image_rect(panel.x + 58, panel.y + 42,
                              panel.width - 58 - right_margin, panel.height - 94);
    cv::Mat scaled;
    cv::resize(raster.image, scaled, image_rect.size(), 0.0, 0.0, cv::INTER_AREA);
    scaled.copyTo(canvas(image_rect));
    drawAxes(canvas, image_rect, centralMeridianDeg);
    if (withLegend)
    {
        drawLegend(canvas, image_rect, raster, legendUnit);
    }
}

} // namespace

bool GlobalTerrainReportRenderer::writePreview(const SmallBodyGlobalProducts &products,
                                                const SmallBodyGlobalOptions &options,
                                                const QString &outputPath,
                                                QString *errorMessage)
{
    if (!products.radialDem.isValid() || !products.elevationDem.isValid()
        || products.domBgr.empty() || products.reliability.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("全球地形报告输入不完整。");
        }
        return false;
    }

    const RenderedRaster dom{northUp(products.domBgr, products.validMask), 0.0, 1.0};
    const RenderedRaster elevation = colorize(
        products.elevationDem.elevation, products.validMask, cv::COLORMAP_TURBO);
    const RenderedRaster reliability = colorize(
        products.reliability, products.validMask, cv::COLORMAP_VIRIDIS, 0.0, 1.0);
    const RenderedRaster shaded = hillshade(
        products.elevationDem, products.referenceRadiusM);

    cv::Mat canvas(1180, 2000, CV_8UC3, cv::Scalar(255, 255, 255));
    drawPanel(canvas, cv::Rect(20, 55, 950, 510), "Native surface-colour DOM",
              dom, std::string(), false, options.centralMeridianDeg);
    drawPanel(canvas, cv::Rect(1010, 55, 950, 510),
              cv::format("Radial elevation relative to %.3f m", products.referenceRadiusM),
              elevation, "m", true, options.centralMeridianDeg);
    drawPanel(canvas, cv::Rect(20, 575, 950, 510),
              "Radial geometry reliability proxy", reliability, "0-1", true,
              options.centralMeridianDeg);
    drawPanel(canvas, cv::Rect(1010, 575, 950, 510),
              "Elevation hillshade (no external reference DEM)", shaded,
              std::string(), false, options.centralMeridianDeg);
    centeredText(canvas,
                  cv::format("planetocentric latitude | body-fixed positive-east longitude | "
                             "CM %.3f deg | solid-angle coverage %.2f%% | PlaScan C++ native",
                             options.centralMeridianDeg,
                             products.solidAngleWeightedCoverageRatio * 100.0),
                 canvas.cols / 2, 1148, 0.48);

    if (qobject_cast<QGuiApplication *>(QCoreApplication::instance()))
    {
        const QString title_font_family = reportTitleFontFamily();
        if (!title_font_family.isEmpty())
        {
            QImage canvas_image(
                canvas.data, canvas.cols, canvas.rows, static_cast<qsizetype>(canvas.step),
                QImage::Format_BGR888);
            QPainter painter(&canvas_image);
            painter.setRenderHint(QPainter::Antialiasing);
            QFont title_font(title_font_family);
            title_font.setPixelSize(28);
            title_font.setBold(true);
            painter.setFont(title_font);
            painter.setPen(QColor(25, 25, 25));
            painter.drawText(QRect(20, 2, canvas.cols - 40, 45), Qt::AlignCenter,
                             QStringLiteral("%1 全球体固连地形产品").arg(options.targetName));
            painter.end();
        }
        else
        {
            centeredText(canvas, "Global body-fixed terrain products",
                         canvas.cols / 2, 35, 0.90, 2);
        }
    }
    else
    {
        centeredText(canvas, "Global body-fixed terrain products",
                     canvas.cols / 2, 35, 0.90, 2);
    }

    QDir().mkpath(QFileInfo(outputPath).absolutePath());
    if (!xjw::common::io::writeImage(outputPath, canvas))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写出全球地形报告预览：%1").arg(outputPath);
        }
        return false;
    }
    return true;
}

} // namespace xjw
