#include "RpcDomGenerator.h"

#include "RpcGeospatialSupport.h"

#include "DemDomIO.h"
#include "RpcCameraIO.h"
#include "io/ImageIO.h"
#include "io/PathIO.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace xjw
{
    namespace
    {

        struct RpcImage
        {
            QString path;
            RpcCameraModel camera;
            cv::Mat bgr;
        };

        void reportProgress(const RpcDomGenerator::ProgressCallback& callback, const QString& message, int percent)
        {
            if (callback)
            {
                callback(message, percent);
            }
        }

        bool cancellationRequested(const std::atomic_bool* cancelFlag, QString* errorMessage)
        {
            if (!cancelFlag || !cancelFlag->load(std::memory_order_relaxed))
            {
                return false;
            }
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("RPC DOM 生成已取消");
            }
            return true;
        }

        bool loadRpcImage(const QString& path, RpcImage* image, QString* errorMessage)
        {
            if (!image)
            {
                return false;
            }
            std::string cameraError;
            if (!loadRpcCameraFromRaster(common::io::toUtf8Path(path), &image->camera, &cameraError))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("读取 DOM 输入影像 RPC 失败 (%1): %2")
                                        .arg(path, QString::fromUtf8(cameraError.c_str()));
                }
                return false;
            }
            QString imageError;
            image->bgr = common::io::readImage(path, cv::IMREAD_COLOR, &imageError);
            if (image->bgr.empty() || image->bgr.type() != CV_8UC3)
            {
                if (errorMessage)
                {
                    *errorMessage = imageError.isEmpty()
                                        ? QStringLiteral("DOM 输入影像无法读取为 8 位彩色图: %1").arg(path)
                                        : imageError;
                }
                return false;
            }
            image->path = QFileInfo(path).absoluteFilePath();
            return true;
        }

        bool bilinearSample(const cv::Mat& image, double sample, double line, cv::Vec3d* value)
        {
            if (!value || sample < 0.0 || line < 0.0 || sample >= static_cast<double>(image.cols - 1) ||
                line >= static_cast<double>(image.rows - 1))
            {
                return false;
            }
            const int x0 = static_cast<int>(std::floor(sample));
            const int y0 = static_cast<int>(std::floor(line));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const double dx = sample - x0;
            const double dy = line - y0;
            const cv::Vec3b& p00 = image.at<cv::Vec3b>(y0, x0);
            const cv::Vec3b& p10 = image.at<cv::Vec3b>(y0, x1);
            const cv::Vec3b& p01 = image.at<cv::Vec3b>(y1, x0);
            const cv::Vec3b& p11 = image.at<cv::Vec3b>(y1, x1);
            for (int channel = 0; channel < 3; ++channel)
            {
                (*value)[channel] = (1.0 - dx) * (1.0 - dy) * p00[channel] + dx * (1.0 - dy) * p10[channel] +
                                    (1.0 - dx) * dy * p01[channel] + dx * dy * p11[channel];
            }
            return true;
        }

        bool writeJson(const QString& path, const QJsonObject& object, QString* errorMessage)
        {
            QSaveFile file(path);
            if (!file.open(QIODevice::WriteOnly) ||
                file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) < 0 || !file.commit())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("写入 RPC DOM 报告失败: %1").arg(path);
                }
                return false;
            }
            return true;
        }

    } // namespace

    bool RpcDomGenerator::generate(const QStringList& imagePaths,
                                   const QString& demPath,
                                   const QString& outputPath,
                                   const RpcDomOptions& options,
                                   QJsonObject* result,
                                   QString* errorMessage,
                                   const ProgressCallback& progress,
                                   const std::atomic_bool* cancelFlag)
    {
        if (result)
        {
            *result = {};
        }
        if (imagePaths.isEmpty() || !QFileInfo::exists(demPath) || outputPath.trimmed().isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("RPC DOM 需要至少一张 RPC 影像、有效 DEM 和输出路径");
            }
            return false;
        }

        if (cancellationRequested(cancelFlag, errorMessage))
        {
            return false;
        }
        reportProgress(progress, QStringLiteral("读取 DEM 和 RPC 影像"), 5);
        DemGridData dem;
        if (!DemDomIO::readDemRaster(demPath, &dem, errorMessage))
        {
            return false;
        }
        if (dem.projection.projectionWkt.trimmed().isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("DEM 没有投影 WKT，无法执行 RPC 正射投影");
            }
            return false;
        }

        std::vector<RpcImage> images;
        images.reserve(static_cast<std::size_t>(imagePaths.size()));
        for (const QString& path : imagePaths)
        {
            if (cancellationRequested(cancelFlag, errorMessage))
            {
                return false;
            }
            RpcImage image;
            if (!loadRpcImage(path, &image, errorMessage))
            {
                return false;
            }
            images.push_back(std::move(image));
        }

        cv::Mat dom(dem.height, dem.width, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat validMask(dem.height, dem.width, CV_8UC1, cv::Scalar(0));
        std::vector<qint64> contributionCounts(images.size(), 0);
        qint64 validPixels = 0;

        reportProgress(progress, QStringLiteral("将 DEM 网格投影到原始影像"), 12);
        int lastProgressPercent = 12;
        for (int row = 0; row < dem.height; ++row)
        {
            if (cancellationRequested(cancelFlag, errorMessage))
            {
                return false;
            }
            std::vector<RpcCameraModel::GeodeticCoordinate> geodetic;
            if (!stereo_dem::projectedRowToGeodetic(dem, row, &geodetic, errorMessage))
            {
                return false;
            }
            for (int col = 0; col < dem.width; ++col)
            {
                if (dem.validMask.at<uchar>(row, col) == 0)
                {
                    continue;
                }
                cv::Vec3d accumulated(0.0, 0.0, 0.0);
                int contributors = 0;
                for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex)
                {
                    CameraImageCoordinate pixel;
                    if (!images[imageIndex].camera.groundToImageGeodetic(geodetic[static_cast<std::size_t>(col)],
                                                                         &pixel))
                    {
                        continue;
                    }
                    cv::Vec3d sampled;
                    if (!bilinearSample(images[imageIndex].bgr, pixel.sample, pixel.line, &sampled))
                    {
                        continue;
                    }
                    accumulated += sampled;
                    ++contributors;
                    ++contributionCounts[imageIndex];
                    if (!options.blendAllImages)
                    {
                        break;
                    }
                }
                if (contributors == 0)
                {
                    continue;
                }
                dom.at<cv::Vec3b>(row, col) = cv::Vec3b(cv::saturate_cast<uchar>(accumulated[0] / contributors),
                                                        cv::saturate_cast<uchar>(accumulated[1] / contributors),
                                                        cv::saturate_cast<uchar>(accumulated[2] / contributors));
                validMask.at<uchar>(row, col) = 255;
                ++validPixels;
            }
            const int percent = 12 + static_cast<int>(76.0 * (row + 1) / std::max(1, dem.height));
            if (percent >= lastProgressPercent + 5 || row + 1 == dem.height)
            {
                reportProgress(progress, QStringLiteral("RPC 正射采样"), percent);
                lastProgressPercent = percent;
            }
        }

        if (validPixels == 0)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("DEM 与 RPC 影像没有有效正射覆盖");
            }
            return false;
        }

        if (cancellationRequested(cancelFlag, errorMessage))
        {
            return false;
        }
        reportProgress(progress, QStringLiteral("写出带地理参考的 DOM"), 92);
        if (!DemDomIO::writeDomGeoTiff(dom, validMask, dem, outputPath, errorMessage))
        {
            return false;
        }
        QString previewPath;
        if (options.writePreview)
        {
            previewPath = QDir(QFileInfo(outputPath).absolutePath()).filePath(QStringLiteral("dom_preview.png"));
            if (!common::io::writeImage(previewPath, dom))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("写出 DOM 预览失败: %1").arg(previewPath);
                }
                return false;
            }
        }

        QJsonArray inputs;
        QJsonArray contributions;
        for (std::size_t index = 0; index < images.size(); ++index)
        {
            inputs.append(images[index].path);
            contributions.append(contributionCounts[index]);
        }
        QJsonObject output{{QStringLiteral("schema"), QStringLiteral("plascan.rpc_dom.v1")},
                           {QStringLiteral("created_at"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
                           {QStringLiteral("dem_path"), QFileInfo(demPath).absoluteFilePath()},
                           {QStringLiteral("dom_path"), QFileInfo(outputPath).absoluteFilePath()},
                           {QStringLiteral("preview_path"), previewPath},
                           {QStringLiteral("images"), inputs},
                           {QStringLiteral("image_contribution_pixels"), contributions},
                           {QStringLiteral("width"), dem.width},
                           {QStringLiteral("height"), dem.height},
                           {QStringLiteral("pixel_size_x"), dem.stepX},
                           {QStringLiteral("pixel_size_y"), dem.stepY},
                           {QStringLiteral("valid_pixels"), validPixels},
                           {QStringLiteral("coverage_fraction"),
                            static_cast<double>(validPixels) / static_cast<double>(dem.width * dem.height)},
                           {QStringLiteral("coordinate_system"), dem.projection.coordinateSystem}};
        const QString reportPath =
            QDir(QFileInfo(outputPath).absolutePath()).filePath(QStringLiteral("rpc_dom_report.json"));
        output.insert(QStringLiteral("report_path"), QFileInfo(reportPath).absoluteFilePath());
        if (!writeJson(reportPath, output, errorMessage))
        {
            return false;
        }
        if (result)
        {
            *result = output;
        }
        reportProgress(progress, QStringLiteral("RPC DOM 生成完成"), 100);
        return true;
    }

} // namespace xjw
