#include "MarkerImageAdapter.h"

#include <stdexcept>

namespace xjw::app::markers
{
    QVector<control_points::MarkerDetection> detectMarkers(const control_points::MarkerDetector& detector,
                                                           const QImage& image,
                                                           const QImage& mask,
                                                           const control_points::MarkerDetectionOptions& options)
    {
        if (image.isNull() || (options.cancelRequested && options.cancelRequested->load(std::memory_order_relaxed)))
        {
            return {};
        }
        if (!mask.isNull() && mask.size() != image.size())
        {
            throw std::invalid_argument("Marker mask size must match the source image");
        }
        const QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
        const cv::Mat grayscale(gray.height(),
                                gray.width(),
                                CV_8UC1,
                                const_cast<uchar*>(gray.constBits()),
                                static_cast<std::size_t>(gray.bytesPerLine()));
        cv::Mat exclusions;
        if (mask.format() == QImage::Format_Grayscale8)
        {
            exclusions = cv::Mat(mask.height(),
                                 mask.width(),
                                 CV_8UC1,
                                 const_cast<uchar*>(mask.constBits()),
                                 static_cast<std::size_t>(mask.bytesPerLine()));
        }
        else if (!mask.isNull())
        {
            exclusions.create(mask.height(), mask.width(), CV_8UC1);
            for (int y = 0; y < mask.height(); ++y)
            {
                auto* row = exclusions.ptr<uchar>(y);
                for (int x = 0; x < mask.width(); ++x)
                {
                    row[x] = static_cast<uchar>(qGray(mask.pixel(x, y)));
                }
            }
        }
        return detector.detect(grayscale, exclusions, options);
    }
} // namespace xjw::app::markers
