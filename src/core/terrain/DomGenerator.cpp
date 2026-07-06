#include "DomGenerator.h"
#include "io/PathIO.h"

#include <QtGlobal>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace xjw
{

namespace
{

cv::Size resolveOutputSize(const DemGridData &demGrid, const DomGenerationOptions &options)
{
    if (!demGrid.isValid())
    {
        return cv::Size();
    }

    if (options.outputResolution > 0.0 && options.resizeToDem)
    {
        const double scaleX = demGrid.stepX / options.outputResolution;
        const double scaleY = demGrid.stepY / options.outputResolution;
        return cv::Size(std::max(1, static_cast<int>(std::round(demGrid.width * scaleX))),
                        std::max(1, static_cast<int>(std::round(demGrid.height * scaleY))));
    }

    return cv::Size(demGrid.width, demGrid.height);
}

double computeSharpnessWeight(const cv::Mat &image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F, 3);

    cv::Scalar meanValue;
    cv::Scalar stdDev;
    cv::meanStdDev(laplacian, meanValue, stdDev);
    const double variance = stdDev[0] * stdDev[0];
    return std::max(variance, 1e-6);
}

double computeLumaMean(const cv::Mat &image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return cv::mean(gray)[0];
}

int selectResizeInterpolation(const cv::Size &source, const cv::Size &target)
{
    if (target.width > source.width || target.height > source.height)
    {
        return cv::INTER_CUBIC;
    }

    return cv::INTER_AREA;
}

bool hasDemCoverage(const DemGridData &demGrid)
{
    const cv::Mat &mask = demGrid.hasCoverageMask() ? demGrid.coverageMask : demGrid.validMask;
    return !mask.empty() && cv::countNonZero(mask) > 0;
}

} // namespace

bool DomGenerator::generateFromImages(const DemGridData &demGrid,
                                      const QStringList &images,
                                      const DomGenerationOptions &options,
                                      cv::Mat *domImage,
                                      QString *errorMsg)
{
    if (!domImage)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DOM 输出图像对象为空");
        }
        return false;
    }

    if (!demGrid.isValid())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 栅格无效，无法生成 DOM");
        }
        return false;
    }

    if (!hasDemCoverage(demGrid))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 覆盖为空，无法生成 DOM");
        }
        return false;
    }

    if (images.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("没有输入影像，无法生成 DOM");
        }
        return false;
    }

    const cv::Size outputSize = resolveOutputSize(demGrid, options);
    if (outputSize.width <= 0 || outputSize.height <= 0)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DOM 输出尺寸无效");
        }
        return false;
    }

    cv::Mat accumulator(outputSize, CV_32FC3, cv::Scalar(0, 0, 0));
    double totalWeight = 0.0;
    bool targetLumaInitialized = false;
    double targetLuma = 0.0;

    for (const QString &imagePath : images)
    {
        cv::Mat image = xjw::common::io::readImage(imagePath, cv::IMREAD_COLOR);
        if (image.empty())
        {
            continue;
        }

        if (image.size() != outputSize)
        {
            cv::resize(image,
                       image,
                       outputSize,
                       0.0,
                       0.0,
                       selectResizeInterpolation(image.size(), outputSize));
        }

        if (options.enableExposureCompensation)
        {
            const double luma = computeLumaMean(image);
            if (!targetLumaInitialized)
            {
                targetLuma = luma;
                targetLumaInitialized = true;
            }

            const double gain = qBound(0.7, targetLuma / std::max(1e-6, luma), 1.3);
            image.convertTo(image, CV_8UC3, gain, 0.0);
        }

        double blendWeight = 1.0;
        if (options.enableSharpnessWeighting)
        {
            blendWeight = computeSharpnessWeight(image);
        }
        blendWeight = std::max(options.minBlendWeight, blendWeight);

        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32FC3);
        accumulator += floatImage * static_cast<float>(blendWeight);
        totalWeight += blendWeight;
    }

    if (totalWeight <= 0.0)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("输入影像读取失败，无法生成 DOM");
        }
        return false;
    }

    accumulator *= (1.0 / totalWeight);
    accumulator.convertTo(*domImage, CV_8UC3);

    cv::Mat resizedMask;
    cv::resize(demGrid.validMask, resizedMask, outputSize, 0.0, 0.0, cv::INTER_NEAREST);
    if (!resizedMask.empty())
    {
        domImage->setTo(cv::Scalar(0, 0, 0), resizedMask == 0);
    }

    return true;
}

// =============================================================================
// DOM from textured mesh
// =============================================================================

namespace
{

// OBJ UV 约定：v=0 在底部，v=1 在顶部；OpenCV 行=0 在顶部
// 因此 tex_row = (1 - v) * (H - 1)
cv::Vec3b sampleTextureBilinear(const cv::Mat &texture, float u, float v)
{
    const float clampU = std::max(0.0f, std::min(1.0f, u));
    const float clampV = std::max(0.0f, std::min(1.0f, v));

    const float tx = clampU * static_cast<float>(texture.cols - 1);
    const float ty = (1.0f - clampV) * static_cast<float>(texture.rows - 1);

    const int x0 = std::max(0, std::min(texture.cols - 1, static_cast<int>(tx)));
    const int y0 = std::max(0, std::min(texture.rows - 1, static_cast<int>(ty)));
    const int x1 = std::min(texture.cols - 1, x0 + 1);
    const int y1 = std::min(texture.rows - 1, y0 + 1);

    const float fx = tx - static_cast<float>(x0);
    const float fy = ty - static_cast<float>(y0);

    const cv::Vec3f c00 = texture.at<cv::Vec3b>(y0, x0);
    const cv::Vec3f c10 = texture.at<cv::Vec3b>(y0, x1);
    const cv::Vec3f c01 = texture.at<cv::Vec3b>(y1, x0);
    const cv::Vec3f c11 = texture.at<cv::Vec3b>(y1, x1);

    const cv::Vec3f interp =
        (1.0f - fx) * (1.0f - fy) * c00 +
        fx           * (1.0f - fy) * c10 +
        (1.0f - fx) * fy           * c01 +
        fx           * fy           * c11;

    return cv::Vec3b(
        static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, interp[0]))),
        static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, interp[1]))),
        static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, interp[2]))));
}

// 计算点 p 在三角形 (p0,p1,p2) 中的重心坐标 (w0,w1,w2)
// 返回 false 表示退化三角形
bool barycentricCoords(double px, double py,
                       double x0, double y0,
                       double x1, double y1,
                       double x2, double y2,
                       double &w0, double &w1, double &w2)
{
    const double denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (std::abs(denom) < 1e-12)
    {
        return false;
    }

    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom;
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom;
    w2 = 1.0 - w0 - w1;
    return true;
}

void rasterizeTriangle(float v0x, float v0y,
                       float v1x, float v1y,
                       float v2x, float v2y,
                       float uv0u, float uv0v,
                       float uv1u, float uv1v,
                       float uv2u, float uv2v,
                       const DemGridData &demGrid,
                       const cv::Mat &texture,
                       cv::Mat &domImage)
{
    // 投影到栅格坐标（XY 平面，忽略 Z）
    const double gx0 = (static_cast<double>(v0x) - demGrid.minX) / demGrid.stepX;
    const double gy0 = (static_cast<double>(v0y) - demGrid.minY) / demGrid.stepY;
    const double gx1 = (static_cast<double>(v1x) - demGrid.minX) / demGrid.stepX;
    const double gy1 = (static_cast<double>(v1y) - demGrid.minY) / demGrid.stepY;
    const double gx2 = (static_cast<double>(v2x) - demGrid.minX) / demGrid.stepX;
    const double gy2 = (static_cast<double>(v2y) - demGrid.minY) / demGrid.stepY;

    // 计算包围盒（整数栅格范围）
    const int minCol = std::max(0, static_cast<int>(std::floor(std::min({gx0, gx1, gx2}))));
    const int maxCol = std::min(demGrid.width - 1,
                                static_cast<int>(std::ceil(std::max({gx0, gx1, gx2}))));
    const int minRow = std::max(0, static_cast<int>(std::floor(std::min({gy0, gy1, gy2}))));
    const int maxRow = std::min(demGrid.height - 1,
                                static_cast<int>(std::ceil(std::max({gy0, gy1, gy2}))));

    if (minCol > maxCol || minRow > maxRow)
    {
        return;
    }

    constexpr double kInsideEpsilon = 1e-6;

    for (int row = minRow; row <= maxRow; ++row)
    {
        for (int col = minCol; col <= maxCol; ++col)
        {
            const double px = static_cast<double>(col) + 0.5;
            const double py = static_cast<double>(row) + 0.5;

            double w0 = 0.0;
            double w1 = 0.0;
            double w2 = 0.0;
            if (!barycentricCoords(px, py, gx0, gy0, gx1, gy1, gx2, gy2, w0, w1, w2))
            {
                continue;
            }

            if (w0 < -kInsideEpsilon || w1 < -kInsideEpsilon || w2 < -kInsideEpsilon)
            {
                continue;
            }

            // 插值 UV
            const double interpolated_u =
                static_cast<double>(w0) * uv0u + static_cast<double>(w1) * uv1u + static_cast<double>(w2) * uv2u;
            const double interpolated_v =
                static_cast<double>(w0) * uv0v + static_cast<double>(w1) * uv1v + static_cast<double>(w2) * uv2v;
            const float u = static_cast<float>(interpolated_u);
            const float v = static_cast<float>(interpolated_v);

            domImage.at<cv::Vec3b>(row, col) = sampleTextureBilinear(texture, u, v);
        }
    }
}

} // namespace

bool DomGenerator::generateFromTexturedMesh(const TerrainMeshInput &input,
                                             const DemGridData &demGrid,
                                             cv::Mat *domImage,
                                             QString *errorMsg)
{
    if (!domImage)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DOM 输出图像对象为空");
        }
        return false;
    }

    if (!demGrid.isValid())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 栅格无效，无法生成 DOM");
        }
        return false;
    }

    if (!hasDemCoverage(demGrid))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("DEM 覆盖为空，无法生成 DOM");
        }
        return false;
    }

    if (input.texture.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("纹理图像为空，无法生成 DOM");
        }
        return false;
    }

    if (!input.mesh.hasFaces())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("网格无三角面，无法生成 DOM");
        }
        return false;
    }

    // 若 domImage 为空则初始化（多瓦片拼接时调用方传入已分配的图像）
    if (domImage->empty())
    {
        *domImage = cv::Mat(demGrid.height, demGrid.width, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    // 确保纹理为 BGR 3 通道
    cv::Mat texture = input.texture;
    if (texture.channels() == 1)
    {
        cv::cvtColor(texture, texture, cv::COLOR_GRAY2BGR);
    }
    else if (texture.channels() == 4)
    {
        cv::cvtColor(texture, texture, cv::COLOR_BGRA2BGR);
    }

    // 需要逐顶点 UV 数组
    if (!input.mesh.hasTextureCoords())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("网格缺少 UV 坐标，无法生成 DOM");
        }
        return false;
    }

    const auto &texCoords = *input.mesh.textureCoords();
    const auto &faces = *input.mesh.faces();
    const size_t nPts = input.mesh.size();
    const size_t nTex = static_cast<size_t>(texCoords.rows());
    const plamatrix::Index nFaces = faces.rows();

    for (plamatrix::Index fi = 0; fi < nFaces; ++fi)
    {
        const int i0 = faces(fi, 0);
        const int i1 = faces(fi, 1);
        const int i2 = faces(fi, 2);

        if (i0 < 0 || i1 < 0 || i2 < 0)
            continue;
        const size_t u0 = static_cast<size_t>(i0), u1 = static_cast<size_t>(i1), u2 = static_cast<size_t>(i2);
        if (u0 >= nPts || u1 >= nPts || u2 >= nPts)
            continue;
        if (u0 >= nTex || u1 >= nTex || u2 >= nTex)
            continue;

        auto v0 = input.mesh[u0], v1 = input.mesh[u1], v2 = input.mesh[u2];
        rasterizeTriangle(
            v0.x(), v0.y(), v1.x(), v1.y(), v2.x(), v2.y(),
            texCoords(static_cast<plamatrix::Index>(u0), 0), texCoords(static_cast<plamatrix::Index>(u0), 1),
            texCoords(static_cast<plamatrix::Index>(u1), 0), texCoords(static_cast<plamatrix::Index>(u1), 1),
            texCoords(static_cast<plamatrix::Index>(u2), 0), texCoords(static_cast<plamatrix::Index>(u2), 1),
            demGrid, texture, *domImage);
    }

    return true;
}

} // namespace xjw
