// ============================================================
// 文件：FramePinholeCamera.cpp
// 功能：实现 FramePinholeCamera 类的全部方法，包括：
//   - 从 ASP Tsai 格式文件加载相机参数（loadFromFile）
//   - Tsai 畸变模型（applyTsaiDistortion）
//   - 世界坐标系到相机坐标系的转换（worldToCameraFromCameraToWorldPose）
//   - 六自由度外参增量更新（applyDeltaPose）
//   - 常规/有符号三维点投影（projectWorldPoint/projectWorldPointSigned）
//   - 像素的 Brown-Conrady 数值反畸变（undistortPixel）
//   - 将相机参数保存回文件（saveToFile）
//
// 单位边界：运行态焦距和主点统一使用 pixel；Tsai 文件通过 pitch
// 在 mm 与 pixel 间转换。相机中心不参与 pitch 换算，保持世界坐标单位。
// ============================================================

#include "FramePinholeCamera.h"
#include "io/PathIO.h"
#include "string_utils/StringParsing.h"
#include "string_utils/StringTransform.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <cctype>

namespace xjw
{
using common::string_utils::asciiLowerCopy;
using common::string_utils::extractDoublesFromText;
using common::string_utils::trimAsciiWhitespace;

void FramePinholeCamera::setPose(const std::array<double, 9> &R, const std::array<double, 3> &C)
{
    _pose.cameraToWorldRotation = R;
    _pose.cameraCenter = C;
    _isLoaded = true;
}

void FramePinholeCamera::setCameraCenter(const std::array<double, 3> &cameraCenter)
{
    _pose.cameraCenter = cameraCenter;
}

void FramePinholeCamera::setIntrinsics(double fu, double fv, double cu, double cv)
{
    _intrinsics.focalX = fu;
    _intrinsics.focalY = fv;
    _intrinsics.principalX = cu;
    _intrinsics.principalY = cv;
    _isLoaded = true;
}

void FramePinholeCamera::setIntrinsicsMillimeters(double fuMm,
                                      double fvMm,
                                      double cuMm,
                                      double cvMm,
                                      double pitchMmPerPixel)
{
    _intrinsics.pixelPitch = (pitchMmPerPixel > 0.0) ? pitchMmPerPixel : 1.0;
    setIntrinsics(fuMm / _intrinsics.pixelPitch,
                  fvMm / _intrinsics.pixelPitch,
                  cuMm / _intrinsics.pixelPitch,
                  cvMm / _intrinsics.pixelPitch);
}

void FramePinholeCamera::setPixelPitch(double pixelPitch)
{
    if (pixelPitch > 0.0)
    {
        _intrinsics.pixelPitch = pixelPitch;
    }
}

void FramePinholeCamera::setAxisDirections(int uDir, int vDir)
{
    _intrinsics.uAxisSign = (uDir < 0 ? -1 : 1);
    _intrinsics.vAxisSign = (vDir < 0 ? -1 : 1);
    _isLoaded = true;
}

void FramePinholeCamera::setDepthAxisFlipped(bool depthAxisFlipped)
{
    _pose.depthAxisFlipped = depthAxisFlipped;
    _isLoaded = true;
}

void FramePinholeCamera::setDistortion(double k1, double k2, double k3, double p1, double p2)
{
    _distortion.radialK1 = k1;
    _distortion.radialK2 = k2;
    _distortion.radialK3 = k3;
    _distortion.tangentialP1 = p1;
    _distortion.tangentialP2 = p2;
    _isLoaded = true;
}

void FramePinholeCamera::setDistortion(const Distortion &distortion)
{
    _distortion = distortion;
    _isLoaded = true;
}

std::array<double, 9> FramePinholeCamera::worldToCameraRotation() const
{
    const auto &rotation = _pose.cameraToWorldRotation;
    return std::array<double, 9>
    {
        {
        rotation[0], rotation[3], rotation[6],
        rotation[1], rotation[4], rotation[7],
        rotation[2], rotation[5], rotation[8]
    }};
}

std::array<double, 3> FramePinholeCamera::worldToCameraTranslation() const
{
    const auto rotation = worldToCameraRotation();
    const auto &camera_center = _pose.cameraCenter;
    std::array<double, 3> translation{{0.0, 0.0, 0.0}};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            translation[row] -= rotation[row * 3 + col] * camera_center[col];
        }
    }
    return translation;
}

FramePinholeCamera FramePinholeCamera::scaledIntrinsics(double scaleX, double scaleY) const
{
    // 这是图像重采样后的像素坐标缩放，不改变物理 pixel pitch 或相机外参。
    FramePinholeCamera scaledCamera = *this;
    scaledCamera._intrinsics.focalX *= scaleX;
    scaledCamera._intrinsics.focalY *= scaleY;
    scaledCamera._intrinsics.principalX *= scaleX;
    scaledCamera._intrinsics.principalY *= scaleY;

    if (scaledCamera._imageSize.has_value())
    {
        const double scaled_samples = static_cast<double>(scaledCamera._imageSize->samples) * scaleX;
        const double scaled_lines = static_cast<double>(scaledCamera._imageSize->lines) * scaleY;
        const double maximum_dimension = static_cast<double>(std::numeric_limits<int>::max());
        if (!std::isfinite(scaled_samples) || !std::isfinite(scaled_lines)
            || scaled_samples < 1.0 || scaled_lines < 1.0
            || scaled_samples > maximum_dimension || scaled_lines > maximum_dimension)
        {
            scaledCamera._imageSize.reset();
        }
        else
        {
            scaledCamera._imageSize = CameraImageSize{
                static_cast<int>(std::lround(scaled_samples)),
                static_cast<int>(std::lround(scaled_lines))};
        }
    }
    return scaledCamera;
}

FramePinholeCamera FramePinholeCamera::normalizedForPositiveDepth() const
{
    FramePinholeCamera normalized = *this;
    if (!_isLoaded)
    {
        return normalized;
    }

    const int u_sign = _intrinsics.uAxisSign < 0 ? -1 : 1;
    const int v_sign = _intrinsics.vAxisSign < 0 ? -1 : 1;
    const double z_sign = _pose.depthAxisFlipped ? -1.0 : 1.0;
    const double axis_signs[3] = {
        z_sign * static_cast<double>(u_sign),
        z_sign * static_cast<double>(v_sign),
        z_sign
    };

    const std::array<double, 9> rotation_world_to_camera = worldToCameraRotation();
    std::array<double, 9> normalized_world_to_camera{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            normalized_world_to_camera[row * 3 + column] =
                axis_signs[row] * rotation_world_to_camera[row * 3 + column];
        }
    }

    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            normalized._pose.cameraToWorldRotation[row * 3 + column] =
                normalized_world_to_camera[column * 3 + row];
        }
    }

    normalized._intrinsics.focalX = std::fabs(_intrinsics.focalX);
    normalized._intrinsics.focalY = std::fabs(_intrinsics.focalY);
    normalized._intrinsics.uAxisSign = 1;
    normalized._intrinsics.vAxisSign = 1;
    normalized._distortion.tangentialP1 =
        static_cast<double>(v_sign) * _distortion.tangentialP1;
    normalized._distortion.tangentialP2 =
        static_cast<double>(u_sign) * _distortion.tangentialP2;
    normalized._pose.depthAxisFlipped = false;
    return normalized;
}

/**
 * @brief 从 ASP `.tsai` 格式文本文件加载相机全部参数。
 *
 * 文件格式说明（每行一个键值对，大小写不敏感）：
 *   fu / fv：焦距（mm，最终除以 pitch 转像素）
 *   cu / cv：主点坐标（mm）
 *   c：相机中心三维坐标（世界系，m），3 个浮点数
 *   r：camera-to-world 旋转矩阵，行优先，9 个浮点数
 *   k1/k2/k3：径向畸变系数；p1/p2：切向畸变系数
 *   pitch：像元大小（mm/pixel），用于单位换算
 *   u_direction / v_direction：坐标轴方向（正或负）
 *
 * @param path  `.tsai` 文件路径
 * @return 成功加载返回 true，文件不存在或读取失败返回 false
 */
bool FramePinholeCamera::loadFromFile(const std::string &path)
{
    std::ifstream ifs = xjw::common::io::openInputFile(path);
    if (!ifs)
    {
        return false;
    }

    // 每次解析都恢复三个参数结构的确定默认值，避免字段沿用上次加载结果。
    // depthAxisFlipped 保留既有行为：只有文件显式提供 w_direction 时才更新。
    const bool preserved_depth_axis_flipped = _pose.depthAxisFlipped;
    _intrinsics = Intrinsics{};
    _distortion = Distortion{};
    _pose = Pose{};
    _pose.depthAxisFlipped = preserved_depth_axis_flipped;
    _isLoaded = false;

    // 六个标志对应能够构成针孔相机的最小必选集合；畸变、pitch 和方向均有默认值。
    bool hasFu = false;
    bool hasFv = false;
    bool hasCu = false;
    bool hasCv = false;
    bool hasC = false;
    bool hasR = false;

    auto startsWithKey = [](const std::string &textLower, const std::string &keyLower)
    {
        if (textLower.rfind(keyLower, 0) != 0)
        {
            return false;
        }
        if (textLower.size() == keyLower.size())
        {
            return true;
        }
        // 分隔符检查防止短键互相吞并，例如 `c` 不能匹配 `cu`，`r` 不能匹配其它单词。
        const char next = textLower[keyLower.size()];
        return std::isspace(static_cast<unsigned char>(next)) || next == '=' || next == ':';
    };

    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty())
        {
            continue;
        }
        const std::string s = trimAsciiWhitespace(line);
        if (s.empty())
        {
            continue;
        }
        const std::string sl = asciiLowerCopy(s);
        // find keyword (case-insensitive)
        if (startsWithKey(sl, "fu"))
        {
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && !v.empty())
            {
                _intrinsics.focalX = v[0];
                hasFu = true;
            }
        }
        else if (startsWithKey(sl, "fv"))
        {
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && !v.empty())
            {
                _intrinsics.focalY = v[0];
                hasFv = true;
            }
        }
        else if (startsWithKey(sl, "cu"))
        {
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && !v.empty())
            {
                _intrinsics.principalX = v[0];
                hasCu = true;
            }
        }
        else if (startsWithKey(sl, "cv"))
        {
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && !v.empty())
            {
                _intrinsics.principalY = v[0];
                hasCv = true;
            }
        }
        else if (startsWithKey(sl, "c"))
        {
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && v.size() >= 3)
            {
                _pose.cameraCenter[0] = v[0];
                _pose.cameraCenter[1] = v[1];
                _pose.cameraCenter[2] = v[2];
                hasC = true;
            }
        }
        else if (startsWithKey(sl, "r"))
        {
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && v.size() >= 9)
            {
                for (int i = 0; i < 9; ++i)
                {
                    _pose.cameraToWorldRotation[i] = v[i];
                }
                hasR = true;
            }
        }
        else if (startsWithKey(sl, "k1"))
        {
            std::vector<double> v;
            // 跳过键名后再解析，避免正则把 "k1" 中的 "1" 当作系数值。
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && extractDoublesFromText(s.substr(pos + 1), v) && !v.empty())
            {
                _distortion.radialK1 = v[0];
            }
        }
        else if (startsWithKey(sl, "k2"))
        {
            std::vector<double> v;
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && extractDoublesFromText(s.substr(pos + 1), v) && !v.empty())
            {
                _distortion.radialK2 = v[0];
            }
        }
        else if (startsWithKey(sl, "k3"))
        {
            std::vector<double> v;
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && extractDoublesFromText(s.substr(pos + 1), v) && !v.empty())
            {
                _distortion.radialK3 = v[0];
            }
        }
        else if (startsWithKey(sl, "p1"))
        {
            std::vector<double> v;
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && extractDoublesFromText(s.substr(pos + 1), v) && !v.empty())
            {
                _distortion.tangentialP1 = v[0];
            }
        }
        else if (startsWithKey(sl, "p2"))
        {
            std::vector<double> v;
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && extractDoublesFromText(s.substr(pos + 1), v) && !v.empty())
            {
                _distortion.tangentialP2 = v[0];
            }
        }
        else if (startsWithKey(sl, "pitch"))
        {
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && !v.empty())
            {
                _intrinsics.pixelPitch = v[0];
            }
        }
        else if (startsWithKey(sl, "u_direction"))
        {
            // 兼容两种格式：
            //   向量格式 "u_direction = 1 0 0"（ASP 标准），取 x 分量符号
            //   标量格式 "u_direction = 1"（ba.tsai 简化），直接取符号
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && !v.empty())
            {
                double dominant;
                if (v.size() >= 3)
                {
                    dominant = v[0]; // 向量：u 轴主分量（x）
                    if (dominant == 0.0)
                    {
                        dominant = v[1]; // fallback y
                    }
                }
                else
                {
                    dominant = v[0]; // 标量：直接表示方向符号
                }
                _intrinsics.uAxisSign = (dominant < 0.0 ? -1 : 1);
            }
        }
        else if (startsWithKey(sl, "v_direction"))
        {
            // 兼容两种格式：
            //   向量格式 "v_direction = 0 1 0"（ASP 标准），取 y 分量符号
            //   标量格式 "v_direction = 1"（ba.tsai 简化），直接取符号
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && !v.empty())
            {
                double dominant;
                if (v.size() >= 3)
                {
                    dominant = v[1]; // 向量：v 轴主分量（y）
                    if (dominant == 0.0)
                    {
                        dominant = v[0]; // fallback x
                    }
                }
                else
                {
                    dominant = v[0]; // 标量：直接表示方向符号
                }
                _intrinsics.vAxisSign = (dominant < 0.0 ? -1 : 1);
            }
        }
        else if (startsWithKey(sl, "w_direction"))
        {
            // w_direction 指相机视线方向（光轴）在相机体坐标系中的向量。
            // 标准 "0 0 1"：场景在 Z_cam>0 侧，depthFlippedZ=false（常见情况）。
            // "0 0 -1"：场景在 Z_cam<0 侧，depthFlippedZ=true（需翻转深度符号）。
            // 标量 "-1" 亦合法，直接表示 z 分量符号。
            std::vector<double> v;
            if (extractDoublesFromText(s, v) && !v.empty())
            {
                double wz = (v.size() >= 3) ? v[2] : v[0];
                _pose.depthAxisFlipped = (wz < 0.0);
            }
        }
        // ignore other lines (VERSION_, PINHOLE, TSAI, directions)
    }

    if (!(hasFu && hasFv && hasCu && hasCv && hasC && hasR))
    {
        // 缺失任何必选字段都不能建立确定的投影关系；保持 `_isLoaded=false`。
        return false;
    }

    // pitch/focal 决定透视除法后的像素比例，非正数会使模型不可用，因此属于硬失败。
    if (_intrinsics.pixelPitch <= 0.0)
    {
        fprintf(stderr, "[FramePinholeCamera] loadFromFile 失败：pitch=%.6g 必须 > 0\n", _intrinsics.pixelPitch);
        return false;
    }
    if (_intrinsics.focalX <= 0.0 || _intrinsics.focalY <= 0.0)
    {
        fprintf(stderr,
                "[FramePinholeCamera] loadFromFile 失败：fu=%.6g fv=%.6g 必须 > 0\n",
                _intrinsics.focalX,
                _intrinsics.focalY);
        return false;
    }

    // 旋转矩阵行列式应接近 ±1。这里仅告警以兼容包含轻微数值漂移或反射轴约定的历史数据。
    {
        const auto &R = _pose.cameraToWorldRotation;
        double det = R[0]*(R[4]*R[8]-R[5]*R[7])
                   - R[1]*(R[3]*R[8]-R[5]*R[6])
                   + R[2]*(R[3]*R[7]-R[4]*R[6]);
        if (std::fabs(std::fabs(det) - 1.0) > 0.01)
        {
            fprintf(stderr, "[FramePinholeCamera] loadFromFile 警告：旋转矩阵行列式=%.6f，期望 ±1，相机数据可能损坏\n", det);
        }
    }

    // 单位转换集中在解析成功之后执行，避免前面校验混用 mm 和 pixel。
    // 将 fu/fv/cu/cv 从 mm 换算到像素，供内部投影与几何计算使用。
    _intrinsics.focalX = _intrinsics.focalX / _intrinsics.pixelPitch;
    _intrinsics.focalY = _intrinsics.focalY / _intrinsics.pixelPitch;
    _intrinsics.principalX = _intrinsics.principalX / _intrinsics.pixelPitch;
    _intrinsics.principalY = _intrinsics.principalY / _intrinsics.pixelPitch;

    _isLoaded = true;
    return true;
}

/**
 * @brief 对归一化图像坐标应用 Tsai（Brown-Conrady）畸变模型。
 *
 * 数学公式：
 *   r² = x² + y²
 *   径向因子：radial = 1 + k1·r² + k2·r⁴ + k3·r⁶
 *   切向畸变（Plumb Bob 模型）：
 *     xd = x·radial + 2·p1·x·y + p2·(r² + 2·x²)
 *     yd = y·radial + p1·(r² + 2·y²) + 2·p2·x·y
 *
 * @param x   归一化坐标 x（相机坐标系下 Xc/Zc）
 * @param y   归一化坐标 y（相机坐标系下 Yc/Zc）
 * @param xd  输出：畸变后归一化 x
 * @param yd  输出：畸变后归一化 y
 */
void FramePinholeCamera::applyTsaiDistortion(double x, double y, double &xd, double &yd) const
{
    const Distortion &distortion = _distortion;
    const double r2 = x * x + y * y;
    const double radial = 1.0 + distortion.radialK1 * r2
                        + distortion.radialK2 * r2 * r2
                        + distortion.radialK3 * r2 * r2 * r2;
    // 径向畸变 + 切向畸变（分别对 x 和 y 方向）
    xd = x * radial + 2.0 * distortion.tangentialP1 * x * y
       + distortion.tangentialP2 * (r2 + 2.0 * x * x);
    yd = y * radial + distortion.tangentialP1 * (r2 + 2.0 * y * y)
       + 2.0 * distortion.tangentialP2 * x * y;
}

/**
 * @brief 将世界坐标系中的点变换到相机坐标系（基于 camera-to-world 的 R^T）。
 *
 * ASP `.tsai` 文件中存储的 R 是 camera-to-world 旋转（R_cw），
 * 即相机坐标轴在世界坐标系中的方向。
 * 世界点 Xw 到相机点 Xc 的变换公式为：
 *   Xc = R_cw^T * (Xw - C)
 * 由于 R 为行优先存储，R^T 对应的乘法等价于按列访问 R。
 *
 * @param world  输入，世界坐标 [Xw, Yw, Zw]
 * @param cam    输出，相机坐标 [Xc, Yc, Zc]
 */
void FramePinholeCamera::worldToCameraFromCameraToWorldPose(const double world[3], double cameraPoint[3]) const
{
    const auto &camera_center = _pose.cameraCenter;
    const auto &rotation = _pose.cameraToWorldRotation;
    // 先计算偏移量 Xw - C
    const double x = world[0] - camera_center[0];
    const double y = world[1] - camera_center[1];
    const double z = world[2] - camera_center[2];
    // 用 R^T 左乘（即 R 按列读取），得到相机坐标
    cameraPoint[0] = rotation[0] * x + rotation[3] * y + rotation[6] * z;
    cameraPoint[1] = rotation[1] * x + rotation[4] * y + rotation[7] * z;
    cameraPoint[2] = rotation[2] * x + rotation[5] * y + rotation[8] * z;
}

void FramePinholeCamera::worldToCamera(const double world[3], double cameraPoint[3]) const
{
    worldToCameraFromCameraToWorldPose(world, cameraPoint);
}

/**
 * @brief 将六自由度增量 delta = [rx, ry, rz, tx, ty, tz] 应用到外参。
 *
 * 旋转更新策略（SO(3) 李代数指数映射 / Rodrigues 公式）：
 *   1. 将 ω = [rx, ry, rz] 视为旋转向量（轴角表示）
 *   2. 通过 Rodrigues 公式计算增量旋转矩阵 dR = exp(ω)
 *   3. 更新 R：R_new = dR * R_old（camera-to-world 矩阵）
 *
 * 该方法避免了欧拉角在大姿态变化或万向节锁时的数值不稳定。
 *
 * 平移更新：直接将 (tx, ty, tz) 加到相机中心 C。
 *
 * @param delta  6 元素数组 [rx, ry, rz, tx, ty, tz]，旋转单位为弧度
 */
void FramePinholeCamera::applyDeltaPose(const double delta[6])
{
    auto &rotation = _pose.cameraToWorldRotation;
    auto &camera_center = _pose.cameraCenter;
    // 旋转向量 ω = (rx, ry, rz)
    double wx = delta[0], wy = delta[1], wz = delta[2];
    double tx = delta[3], ty = delta[4], tz = delta[5];

    // Rodrigues 公式: dR = I + sin(θ)/θ * [ω]_× + (1 - cos(θ))/θ² * [ω]_×²
    double theta2 = wx * wx + wy * wy + wz * wz;
    double dR[9];

    if (theta2 < 1e-20)
    {
        // θ 极小时直接使用一阶近似，避免 sin(theta)/theta 和
        // (1-cos(theta))/theta² 的消减误差与除零风险。
        dR[0] = 1.0;   dR[1] = -wz;   dR[2] = wy;
        dR[3] = wz;    dR[4] = 1.0;   dR[5] = -wx;
        dR[6] = -wy;   dR[7] = wx;    dR[8] = 1.0;
    }
    else
    {
        double theta = std::sqrt(theta2);
        double sinc = std::sin(theta) / theta;
        double cosc = (1.0 - std::cos(theta)) / theta2;

        // [ω]_× 的平方项元素
        double wxwy = wx * wy, wxwz = wx * wz, wywz = wy * wz;

        dR[0] = 1.0 - cosc * (wy * wy + wz * wz);
        dR[1] = cosc * wxwy - sinc * wz;
        dR[2] = cosc * wxwz + sinc * wy;
        dR[3] = cosc * wxwy + sinc * wz;
        dR[4] = 1.0 - cosc * (wx * wx + wz * wz);
        dR[5] = cosc * wywz - sinc * wx;
        dR[6] = cosc * wxwz - sinc * wy;
        dR[7] = cosc * wywz + sinc * wx;
        dR[8] = 1.0 - cosc * (wx * wx + wy * wy);
    }

    // R_new = dR * R_old（两者均为 camera-to-world，行优先）
    double Rnew[9];
    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            double s = 0;
            for (int k = 0; k < 3; k++)
            {
                s += dR[r * 3 + k] * rotation[k * 3 + c];
            }
            Rnew[r * 3 + c] = s;
        }
    }
    for (int i = 0; i < 9; i++)
    {
        rotation[i] = Rnew[i];
    }
    // 更新相机中心：直接累加平移增量
    camera_center[0] += tx;
    camera_center[1] += ty;
    camera_center[2] += tz;
}

/**
 * @brief 将三维世界坐标点投影到图像像素坐标。
 *
 * 完整投影流程：
 *   1. 世界坐标 → 相机坐标：Xc = R^T * (Xw - C)
 *   2. 深度检查：物理前向深度必须 > 1e-9
 *      - 常规相机：forwardDepth = Zc
 *      - flipped-depth 相机：forwardDepth = -Zc
 *   3. 透视除法：x = Xc/Zc，y = Yc/Zc（归一化坐标）
 *   4. 畸变：(xd, yd) = Tsai_distortion(x, y)
 *   5. 像素化：u = u_dir * fu * xd + cu
 *              v = v_dir * fv * yd + cv
 *
 * @param world  三维世界坐标 [X, Y, Z]
 * @param uv     输出像素坐标 [u, v]
 * @return 相机已加载且点在前方时返回 true，否则返回 false
 */
bool FramePinholeCamera::projectWorldPoint(const double world[3], double pixel[2]) const
{
    if (!_isLoaded) return false;
    const Intrinsics &intrinsics = _intrinsics;
    double cameraPoint[3];
    // 按 ASP 约定，R 为 camera-to-world，故用 R^T 将世界点转换到相机坐标系
    worldToCameraFromCameraToWorldPose(world, cameraPoint);
    // 物理前向深度需要与 w_direction 语义一致。
    const double forwardDepth = _pose.depthAxisFlipped ? -cameraPoint[2] : cameraPoint[2];
    // 1e-9 同时排除物理后方点和过近点，避免后续透视除法数值爆炸。
    if (!(forwardDepth > 1e-9)) return false;
    // 透视除法仍使用带符号的 Z_cam，保证 flipped-depth 情况下像素坐标不镜像。
    double x = cameraPoint[0] / cameraPoint[2];
    double y = cameraPoint[1] / cameraPoint[2];
    // 应用 Tsai 畸变模型
    double xd, yd; applyTsaiDistortion(x, y, xd, yd);
    // 根据文件中读取的坐标轴方向符号，计算最终像素坐标
    pixel[0] = intrinsics.uAxisSign * (intrinsics.focalX * xd) + intrinsics.principalX;
    pixel[1] = intrinsics.vAxisSign * (intrinsics.focalY * yd) + intrinsics.principalY;
    return true;
}

bool FramePinholeCamera::projectWorldPointWithDepth(const double world[3],
                                        double pixel[2],
                                        double &positiveDepth) const
{
    if (!_isLoaded || world == nullptr || pixel == nullptr)
    {
        return false;
    }

    double camera_point[3];
    worldToCameraFromCameraToWorldPose(world, camera_point);
    positiveDepth = _pose.depthAxisFlipped ? -camera_point[2] : camera_point[2];
    if (!(positiveDepth > 1e-9) || !std::isfinite(positiveDepth))
    {
        return false;
    }

    // `camera_point` 已经在上面计算过。旧实现再次调用
    // projectWorldPoint()，会为每个像素重复一次完整的世界到相机变换；
    // 多视深度一致性会调用数十亿次该路径，因此直接完成透视投影。
    const double camera_z = camera_point[2];
    if (std::fabs(camera_z) < 1.0e-9)
    {
        return false;
    }
    const double normalized_x = camera_point[0] / camera_z;
    const double normalized_y = camera_point[1] / camera_z;
    double distorted_x = 0.0;
    double distorted_y = 0.0;
    applyTsaiDistortion(
        normalized_x, normalized_y, distorted_x, distorted_y);
    pixel[0] = static_cast<double>(_intrinsics.uAxisSign) *
                   (_intrinsics.focalX * distorted_x) +
               _intrinsics.principalX;
    pixel[1] = static_cast<double>(_intrinsics.vAxisSign) *
                   (_intrinsics.focalY * distorted_y) +
               _intrinsics.principalY;
    return true;
}

double FramePinholeCamera::positiveDepth(const double world[3]) const
{
    if (!_isLoaded || world == nullptr)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double camera_point[3] = {0.0, 0.0, 0.0};
    worldToCameraFromCameraToWorldPose(world, camera_point);
    const double depth = _pose.depthAxisFlipped ? -camera_point[2] : camera_point[2];
    return std::isfinite(depth) ? depth : std::numeric_limits<double>::quiet_NaN();
}

bool FramePinholeCamera::isPointInFront(const double world[3], double minimumDepth) const
{
    if (!std::isfinite(minimumDepth))
    {
        return false;
    }
    return positiveDepth(world) > std::max(0.0, minimumDepth);
}

/**
 * @brief 有符号投影：允许 Z_cam < 0，直接用 cam[2]（带符号）做透视除法。
 *
 * 核心公理：u = u_dir * fu * (X_cam / Z_cam) + cu
 *   该公式在 Z_cam < 0 时同样成立，两个负号在相对定向数据中自洽，
 *   **给出正确的物理像素坐标**。
 *   反之，若改用 |Z_cam|（projectAnyDir 旧实现），则得到镜像像素（错误）。
 */
bool FramePinholeCamera::projectWorldPointSigned(const double world[3], double pixel[2]) const
{
    if (!_isLoaded) return false;
    const Intrinsics &intrinsics = _intrinsics;
    double cameraPoint[3];
    worldToCameraFromCameraToWorldPose(world, cameraPoint);
    // Signed 版本不检查前后方，只排除无法稳定做透视除法的近零深度。
    if (std::fabs(cameraPoint[2]) < 1e-9) return false;
    // 有符号（signed）透视除法：保留 Z_cam 自然符号
    const double x = cameraPoint[0] / cameraPoint[2];
    const double y = cameraPoint[1] / cameraPoint[2];
    double xd, yd; applyTsaiDistortion(x, y, xd, yd);
    pixel[0] = intrinsics.uAxisSign * (intrinsics.focalX * xd) + intrinsics.principalX;
    pixel[1] = intrinsics.vAxisSign * (intrinsics.focalY * yd) + intrinsics.principalY;
    return true;
}

/**
 * @brief 像素坐标 → 归一化相机坐标（精确反畸变，Newton 迭代）。
 *
 * 正向关系（applyTsaiDistortion）：
 *   (x, y) → (xd, yd) → pixel
 *
 * 反向：给定 pixel，求 (x, y)。
 * 初始估计用无畸变公式：x0 = u_dir*(u-cu)/fu，然后 Newton 迭代修正畸变。
 *
 * 每步残差：r = f(x_current) - target，其中 f(x) = u_dir*fu*distort(x)+cu
 * Jacobian（对 x 数值差分 1e-7 精度足够）。
 */
bool FramePinholeCamera::undistortPixel(const double pixel[2], double norm[2],
                             int maxIter, double tol) const
{
    if (!_isLoaded) return false;
    const Intrinsics &intrinsics = _intrinsics;

    // 初始估计（忽略畸变）
    double x = static_cast<double>(intrinsics.uAxisSign)
             * (pixel[0] - intrinsics.principalX) / intrinsics.focalX;
    double y = static_cast<double>(intrinsics.vAxisSign)
             * (pixel[1] - intrinsics.principalY) / intrinsics.focalY;

    // MVS 统一使用去畸变后的针孔相机。零畸变时 Newton 迭代的首轮
    // 必然得到零残差，直接返回同一解析结果可避免每个像素的函数调用、
    // 多项式计算和收敛判断。
    const Distortion &distortion = _distortion;
    if (distortion.radialK1 == 0.0 && distortion.radialK2 == 0.0 &&
        distortion.radialK3 == 0.0 && distortion.tangentialP1 == 0.0 &&
        distortion.tangentialP2 == 0.0)
    {
        norm[0] = x;
        norm[1] = y;
        return true;
    }

    for (int iter = 0; iter < maxIter; ++iter)
    {
        double xd, yd;
        applyTsaiDistortion(x, y, xd, yd);

        double u_current = static_cast<double>(intrinsics.uAxisSign)
                         * (intrinsics.focalX * xd) + intrinsics.principalX;
        double v_current = static_cast<double>(intrinsics.vAxisSign)
                         * (intrinsics.focalY * yd) + intrinsics.principalY;

        double ru = u_current - pixel[0];
        double rv = v_current - pixel[1];

        if (std::fabs(ru) < tol && std::fabs(rv) < tol)
            break;

        // 使用前向差分估计 2×2 Jacobian；1e-7 在归一化坐标尺度上兼顾截断误差与浮点消减。
        const double h = 1e-7;
        double xdh, ydh;
        applyTsaiDistortion(x + h, y, xdh, ydh);
        double J00 = static_cast<double>(intrinsics.uAxisSign) * intrinsics.focalX * (xdh - xd) / h;
        double J10 = static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY * (ydh - yd) / h;

        applyTsaiDistortion(x, y + h, xdh, ydh);
        double J01 = static_cast<double>(intrinsics.uAxisSign) * intrinsics.focalX * (xdh - xd) / h;
        double J11 = static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY * (ydh - yd) / h;

        // Newton step: delta = -J^{-1} * r
        double det = J00 * J11 - J01 * J10;
        // 局部映射不可逆时停止更新，保留最后一个有限估计，不把病态步长放大到输出。
        if (std::fabs(det) < 1e-15) break;

        x -= ( J11 * ru - J01 * rv) / det;
        y -= (-J10 * ru + J00 * rv) / det;
    }

    // 接口只用 false 表示相机未初始化；达到迭代上限或奇异退出时仍返回当前最佳估计。
    norm[0] = x;
    norm[1] = y;
    return true;
}

bool FramePinholeCamera::unprojectPixel(const double pixel[2], double positiveDepth, double world[3]) const
{
    if (!_isLoaded || pixel == nullptr || world == nullptr
        || !(positiveDepth > 0.0) || !std::isfinite(positiveDepth)
        || !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]))
    {
        return false;
    }

    double normalized[2] = {0.0, 0.0};
    if (!undistortPixel(pixel, normalized))
    {
        return false;
    }

    const double camera_z = _pose.depthAxisFlipped ? -positiveDepth : positiveDepth;
    const double camera_point[3] = {
        normalized[0] * camera_z,
        normalized[1] * camera_z,
        camera_z
    };
    const std::array<double, 9> &rotation = _pose.cameraToWorldRotation;
    const std::array<double, 3> &center = _pose.cameraCenter;
    for (int row = 0; row < 3; ++row)
    {
        world[row] = center[row]
                   + rotation[row * 3] * camera_point[0]
                   + rotation[row * 3 + 1] * camera_point[1]
                   + rotation[row * 3 + 2] * camera_point[2];
    }
    return std::isfinite(world[0]) && std::isfinite(world[1]) && std::isfinite(world[2]);
}

/**
 * @brief 将当前相机参数保存到 Tsai 风格文本文件。
 *
 * 保存时将内存中的像素单位焦距/主点乘回 pitch，恢复为 mm，
 * 保证与 ASP 工具读取格式完全兼容。
 * 使用 double 最大精度（max_digits10）确保数值不损失。
 *
 * @param path  目标文件路径
 * @return 成功写入返回 true，文件无法创建时返回 false
 */
bool FramePinholeCamera::saveToFile(const std::string &path) const
{
    std::ofstream ofs = xjw::common::io::openOutputFile(path);
    if (!ofs) return false;
    const Intrinsics &intrinsics = _intrinsics;
    const Distortion &distortion = _distortion;
    const Pose &pose = _pose;
    // 使用 double 类型能表示的最大有效位数，避免精度损失
    ofs << std::setprecision(std::numeric_limits<double>::max_digits10);
    // 乘回 pitch，将像素单位的内参转换为 mm 后写入文件。
    ofs << "fu = " << (intrinsics.focalX * intrinsics.pixelPitch) << "\n";
    ofs << "fv = " << (intrinsics.focalY * intrinsics.pixelPitch) << "\n";
    ofs << "cu = " << (intrinsics.principalX * intrinsics.pixelPitch) << "\n";
    ofs << "cv = " << (intrinsics.principalY * intrinsics.pixelPitch) << "\n";
    ofs << "c = " << pose.cameraCenter[0] << " " << pose.cameraCenter[1]
        << " " << pose.cameraCenter[2] << "\n";
    // 旋转矩阵以空格分隔，行优先，共 9 个值
    ofs << "r = ";
    for (int i = 0; i < 9; i++)
    {
        ofs << pose.cameraToWorldRotation[i];
        if (i < 8)
        {
            ofs << " ";
        }
    }
    ofs << "\n";
    ofs << "k1 = " << distortion.radialK1 << "\n";
    ofs << "k2 = " << distortion.radialK2 << "\n";
    ofs << "k3 = " << distortion.radialK3 << "\n";
    ofs << "p1 = " << distortion.tangentialP1 << "\n";
    ofs << "p2 = " << distortion.tangentialP2 << "\n";
    ofs << "pitch = " << intrinsics.pixelPitch << "\n";
    ofs << "u_direction = " << intrinsics.uAxisSign << "\n";
    ofs << "v_direction = " << intrinsics.vAxisSign << "\n";
    ofs << "w_direction = " << (pose.depthAxisFlipped ? -1 : 1) << "\n";
    return true;
}

} // namespace xjw
