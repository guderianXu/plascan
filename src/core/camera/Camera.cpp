// ============================================================
// 文件：Camera.cpp
// 功能：实现 Camera 类的全部方法，包括：
//   - 从 ASP Tsai 格式文件加载相机参数（loadFromFile）
//   - Tsai 畸变模型（applyTsaiDistortion）
//   - 世界坐标系到相机坐标系的转换（worldToCamera_W_from_camToWorld）
//   - 六自由度外参增量更新（applyDeltaPose）
//   - 三维点投影到图像平面（project）
//   - 将相机参数保存回文件（saveToFile）
// ============================================================

#include "Camera.h"
#include "PositiveDepthCameraModel.h"
#include "io/PathIO.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <regex>
#include <cctype>

namespace xjw
{
/**
 * @brief 从一行文本中解析所有浮点数，忽略非数值 token（包括 "="）。
 * @param line  待解析的文本行，例如 "R = 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"
 * @param out   输出：解析出的所有 double 值
 * @return 至少解析到一个数值时返回 true
 */
static bool parseDoublesFromLine(const std::string &line, std::vector<double> &out)
{
    out.clear();
    static const std::regex kNumberPattern(
        R"([+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
    );

    auto begin = std::sregex_iterator(line.begin(), line.end(), kNumberPattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it)
    {
        try
        {
            out.push_back(std::stod(it->str()));
        }
        catch (...)
        {
        }
    }
    return !out.empty();
}

Camera::Intrinsics Camera::intrinsics() const
{
    Intrinsics intrinsicsValue;
    intrinsicsValue.focalX = _focalX;
    intrinsicsValue.focalY = _focalY;
    intrinsicsValue.principalX = _principalX;
    intrinsicsValue.principalY = _principalY;
    intrinsicsValue.pixelPitch = _pixelPitch;
    intrinsicsValue.uAxisSign = _uAxisSign;
    intrinsicsValue.vAxisSign = _vAxisSign;
    return intrinsicsValue;
}

Camera::Distortion Camera::distortion() const
{
    Distortion distortionValue;
    distortionValue.radialK1 = _radialK1;
    distortionValue.radialK2 = _radialK2;
    distortionValue.radialK3 = _radialK3;
    distortionValue.tangentialP1 = _tangentialP1;
    distortionValue.tangentialP2 = _tangentialP2;
    return distortionValue;
}

Camera::Pose Camera::pose() const
{
    Pose poseValue;
    poseValue.cameraToWorldRotation = _cameraToWorldRotation;
    poseValue.cameraCenter = _cameraCenter;
    poseValue.depthAxisFlipped = _depthAxisFlipped;
    return poseValue;
}

std::array<double, 9> Camera::worldToCameraRotation() const
{
    return std::array<double, 9>{{
        _cameraToWorldRotation[0], _cameraToWorldRotation[3], _cameraToWorldRotation[6],
        _cameraToWorldRotation[1], _cameraToWorldRotation[4], _cameraToWorldRotation[7],
        _cameraToWorldRotation[2], _cameraToWorldRotation[5], _cameraToWorldRotation[8]
    }};
}

std::array<double, 3> Camera::worldToCameraTranslation() const
{
    const auto rotation = worldToCameraRotation();
    std::array<double, 3> translation{{0.0, 0.0, 0.0}};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            translation[row] -= rotation[row * 3 + col] * _cameraCenter[col];
        }
    }
    return translation;
}

Camera::PositiveDepthModel Camera::toPositiveDepthModel() const
{
    return PositiveDepthModel(*this);
}

Camera Camera::scaledIntrinsics(double scaleX, double scaleY) const
{
    Camera scaledCamera = *this;
    scaledCamera._focalX *= scaleX;
    scaledCamera._focalY *= scaleY;
    scaledCamera._principalX *= scaleX;
    scaledCamera._principalY *= scaleY;
    return scaledCamera;
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
bool Camera::loadFromFile(const std::string &path)
{
    std::ifstream ifs = xjw::common::io::openInputFile(path);
    if (!ifs)
    {
        return false;
    }

    _focalX = 0.0;
    _focalY = 0.0;
    _principalX = 0.0;
    _principalY = 0.0;
    _cameraCenter = {{0.0, 0.0, 0.0}};
    _cameraToWorldRotation = {{1.0, 0.0, 0.0,
           0.0, 1.0, 0.0,
           0.0, 0.0, 1.0}};
    _radialK1 = 0.0;
    _radialK2 = 0.0;
    _radialK3 = 0.0;
    _tangentialP1 = 0.0;
    _tangentialP2 = 0.0;
    _pixelPitch = 1.0;
    _uAxisSign = 1;
    _vAxisSign = 1;
    _isLoaded = false;

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
        // 去除行首空白字符
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
        {
            continue;
        }
        auto s = line.substr(start);
        // 构造小写副本，用于大小写不敏感的关键字匹配
        std::string sl = s;
        for (char &ch : sl)
        {
            ch = (char)std::tolower((unsigned char)ch);
        }
        // find keyword (case-insensitive)
        if (startsWithKey(sl, "fu"))
        {
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && !v.empty())
            {
                _focalX = v[0];
                hasFu = true;
            }
        }
        else if (startsWithKey(sl, "fv"))
        {
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && !v.empty())
            {
                _focalY = v[0];
                hasFv = true;
            }
        }
        else if (startsWithKey(sl, "cu"))
        {
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && !v.empty())
            {
                _principalX = v[0];
                hasCu = true;
            }
        }
        else if (startsWithKey(sl, "cv"))
        {
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && !v.empty())
            {
                _principalY = v[0];
                hasCv = true;
            }
        }
        else if (startsWithKey(sl, "c"))
        {
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && v.size() >= 3)
            {
                _cameraCenter[0] = v[0];
                _cameraCenter[1] = v[1];
                _cameraCenter[2] = v[2];
                hasC = true;
            }
        }
        else if (startsWithKey(sl, "r"))
        {
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && v.size() >= 9)
            {
                for (int i = 0; i < 9; ++i)
                {
                    _cameraToWorldRotation[i] = v[i];
                }
                hasR = true;
            }
        }
        else if (startsWithKey(sl, "k1"))
        {
            std::vector<double> v;
            // 跳过 "k1" 键名后再解析，避免把 "k1" 中的 "1" 当作数字
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && parseDoublesFromLine(s.substr(pos + 1), v) && !v.empty())
            {
                _radialK1 = v[0];
            }
        }
        else if (startsWithKey(sl, "k2"))
        {
            std::vector<double> v;
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && parseDoublesFromLine(s.substr(pos + 1), v) && !v.empty())
            {
                _radialK2 = v[0];
            }
        }
        else if (startsWithKey(sl, "k3"))
        {
            std::vector<double> v;
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && parseDoublesFromLine(s.substr(pos + 1), v) && !v.empty())
            {
                _radialK3 = v[0];
            }
        }
        else if (startsWithKey(sl, "p1"))
        {
            std::vector<double> v;
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && parseDoublesFromLine(s.substr(pos + 1), v) && !v.empty())
            {
                _tangentialP1 = v[0];
            }
        }
        else if (startsWithKey(sl, "p2"))
        {
            std::vector<double> v;
            auto pos = s.find_first_of("=:");
            if (pos != std::string::npos && parseDoublesFromLine(s.substr(pos + 1), v) && !v.empty())
            {
                _tangentialP2 = v[0];
            }
        }
        else if (startsWithKey(sl, "pitch"))
        {
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && !v.empty())
            {
                _pixelPitch = v[0];
            }
        }
        else if (startsWithKey(sl, "u_direction"))
        {
            // 兼容两种格式：
            //   向量格式 "u_direction = 1 0 0"（ASP 标准），取 x 分量符号
            //   标量格式 "u_direction = 1"（ba.tsai 简化），直接取符号
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && !v.empty())
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
                _uAxisSign = (dominant < 0.0 ? -1 : 1);
            }
        }
        else if (startsWithKey(sl, "v_direction"))
        {
            // 兼容两种格式：
            //   向量格式 "v_direction = 0 1 0"（ASP 标准），取 y 分量符号
            //   标量格式 "v_direction = 1"（ba.tsai 简化），直接取符号
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && !v.empty())
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
                _vAxisSign = (dominant < 0.0 ? -1 : 1);
            }
        }
        else if (startsWithKey(sl, "w_direction")) 
        {
            // w_direction 指相机视线方向（光轴）在相机体坐标系中的向量。
            // 标准 "0 0 1"：场景在 Z_cam>0 侧，depthFlippedZ=false（常见情况）。
            // "0 0 -1"：场景在 Z_cam<0 侧，depthFlippedZ=true（需翻转深度符号）。
            // 标量 "-1" 亦合法，直接表示 z 分量符号。
            std::vector<double> v;
            if (parseDoublesFromLine(s, v) && !v.empty()) 
            {
                double wz = (v.size() >= 3) ? v[2] : v[0];
                _depthAxisFlipped = (wz < 0.0);
            }
        }
        // ignore other lines (VERSION_, PINHOLE, TSAI, directions)
    }

    if (!(hasFu && hasFv && hasCu && hasCv && hasC && hasR))
    {
        return false;
    }

    // 基本参数合法性校验
    if (_pixelPitch <= 0.0)
    {
        fprintf(stderr, "[Camera] loadFromFile 失败：pitch=%.6g 必须 > 0\n", _pixelPitch);
        return false;
    }
    if (_focalX <= 0.0 || _focalY <= 0.0)
    {
        fprintf(stderr, "[Camera] loadFromFile 失败：fu=%.6g fv=%.6g 必须 > 0\n", _focalX, _focalY);
        return false;
    }

    // 旋转矩阵正交性校验（行列式应 ≈ ±1，每行范数应 ≈ 1）
    {
        const auto &R = _cameraToWorldRotation;
        double det = R[0]*(R[4]*R[8]-R[5]*R[7])
                   - R[1]*(R[3]*R[8]-R[5]*R[6])
                   + R[2]*(R[3]*R[7]-R[4]*R[6]);
        if (std::fabs(std::fabs(det) - 1.0) > 0.01)
        {
            fprintf(stderr, "[Camera] loadFromFile 警告：旋转矩阵行列式=%.6f，期望 ±1，相机数据可能损坏\n", det);
        }
    }

    // 将 fu/fv/cu/cv 从 mm 换算到像素，供内部投影与几何计算使用。
    _focalX = _focalX / _pixelPitch;
    _focalY = _focalY / _pixelPitch;
    _principalX = _principalX / _pixelPitch;
    _principalY = _principalY / _pixelPitch;

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
void Camera::applyTsaiDistortion(double x, double y, double &xd, double &yd) const
{
    double r2 = x * x + y * y;                              // 归一化半径平方
    double radial = 1.0 + _radialK1 * r2 + _radialK2 * r2 * r2 + _radialK3 * r2 * r2 * r2; // 径向畸变因子
    // 径向畸变 + 切向畸变（分别对 x 和 y 方向）
    xd = x * radial + 2.0 * _tangentialP1 * x * y + _tangentialP2 * (r2 + 2.0 * x * x);
    yd = y * radial + _tangentialP1 * (r2 + 2.0 * y * y) + 2.0 * _tangentialP2 * x * y;
}

/**
 * @brief 将世界坐标系中的点变换到相机坐标系（基于 camera-to-world 的 R^T）。
 *
 * ASP `.tsai` 文件中存储的 R 是 camera-to-world 旋转（R_cw），
 * 即相机坐标轴在世界坐标系中的方向。
 * 世界点 Xw 到相机点 Xc 的变换公式为：
 *   Xc = R_cw^T * (Xw - C)
 * 由于 R 为行优先存储，R^T 对应的乘法等价于按列访问 R（即 _cameraToWorldRotation[0],_cameraToWorldRotation[3],_cameraToWorldRotation[6] 为第一列）。
 *
 * @param world  输入，世界坐标 [Xw, Yw, Zw]
 * @param cam    输出，相机坐标 [Xc, Yc, Zc]
 */
void Camera::worldToCameraFromCameraToWorldPose(const double world[3], double cameraPoint[3]) const
{
    // 先计算偏移量 Xw - C
    double x = world[0] - _cameraCenter[0];
    double y = world[1] - _cameraCenter[1];
    double z = world[2] - _cameraCenter[2];
    // 用 R^T 左乘（即 R 按列读取），得到相机坐标
    cameraPoint[0] = _cameraToWorldRotation[0] * x + _cameraToWorldRotation[3] * y + _cameraToWorldRotation[6] * z;
    cameraPoint[1] = _cameraToWorldRotation[1] * x + _cameraToWorldRotation[4] * y + _cameraToWorldRotation[7] * z;
    cameraPoint[2] = _cameraToWorldRotation[2] * x + _cameraToWorldRotation[5] * y + _cameraToWorldRotation[8] * z;
}

void Camera::worldToCamera(const double world[3], double cameraPoint[3]) const
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
void Camera::applyDeltaPose(const double delta[6])
{
    // 旋转向量 ω = (rx, ry, rz)
    double wx = delta[0], wy = delta[1], wz = delta[2];
    double tx = delta[3], ty = delta[4], tz = delta[5];

    // Rodrigues 公式: dR = I + sin(θ)/θ * [ω]_× + (1 - cos(θ))/θ² * [ω]_×²
    double theta2 = wx * wx + wy * wy + wz * wz;
    double dR[9];

    if (theta2 < 1e-20)
    {
        // θ ≈ 0: dR ≈ I + [ω]_×（一阶近似）
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
                s += dR[r * 3 + k] * _cameraToWorldRotation[k * 3 + c];
            }
            Rnew[r * 3 + c] = s;
        }
    }
    for (int i = 0; i < 9; i++)
    {
        _cameraToWorldRotation[i] = Rnew[i];
    }
    // 更新相机中心：直接累加平移增量
    _cameraCenter[0] += tx; _cameraCenter[1] += ty; _cameraCenter[2] += tz;
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
bool Camera::projectWorldPoint(const double world[3], double pixel[2]) const
{
    if (!_isLoaded) return false;
    double cameraPoint[3];
    // 按 ASP 约定，R 为 camera-to-world，故用 R^T 将世界点转换到相机坐标系
    worldToCameraFromCameraToWorldPose(world, cameraPoint);
    // 物理前向深度需要与 w_direction 语义一致。
    const double forwardDepth = _depthAxisFlipped ? -cameraPoint[2] : cameraPoint[2];
    if (!(forwardDepth > 1e-9)) return false;
    // 透视除法仍使用带符号的 Z_cam，保证 flipped-depth 情况下像素坐标不镜像。
    double x = cameraPoint[0] / cameraPoint[2];
    double y = cameraPoint[1] / cameraPoint[2];
    // 应用 Tsai 畸变模型
    double xd, yd; applyTsaiDistortion(x, y, xd, yd);
    // 根据文件中读取的坐标轴方向符号，计算最终像素坐标
    pixel[0] = _uAxisSign * (_focalX * xd) + _principalX;
    pixel[1] = _vAxisSign * (_focalY * yd) + _principalY;
    return true;
}

/**
 * @brief 有符号投影：允许 Z_cam < 0，直接用 cam[2]（带符号）做透视除法。
 *
 * 核心公理：u = u_dir * fu * (X_cam / Z_cam) + cu
 *   该公式在 Z_cam < 0 时同样成立，两个负号在相对定向数据中自洽，
 *   **给出正确的物理像素坐标**。
 *   反之，若改用 |Z_cam|（projectAnyDir 旧实现），则得到镜像像素（错误）。
 */
bool Camera::projectWorldPointSigned(const double world[3], double pixel[2]) const
{
    if (!_isLoaded) return false;
    double cameraPoint[3];
    worldToCameraFromCameraToWorldPose(world, cameraPoint);
    if (std::fabs(cameraPoint[2]) < 1e-9) return false;  // 仅排除极接近零的情况
    // 有符号（signed）透视除法：保留 Z_cam 自然符号
    const double x = cameraPoint[0] / cameraPoint[2];
    const double y = cameraPoint[1] / cameraPoint[2];
    double xd, yd; applyTsaiDistortion(x, y, xd, yd);
    pixel[0] = _uAxisSign * (_focalX * xd) + _principalX;
    pixel[1] = _vAxisSign * (_focalY * yd) + _principalY;
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
bool Camera::undistortPixel(const double pixel[2], double norm[2],
                             int maxIter, double tol) const
{
    if (!_isLoaded) return false;

    // 初始估计（忽略畸变）
    double x = static_cast<double>(_uAxisSign) * (pixel[0] - _principalX) / _focalX;
    double y = static_cast<double>(_vAxisSign) * (pixel[1] - _principalY) / _focalY;

    for (int iter = 0; iter < maxIter; ++iter)
    {
        double xd, yd;
        applyTsaiDistortion(x, y, xd, yd);

        double u_current = static_cast<double>(_uAxisSign) * (_focalX * xd) + _principalX;
        double v_current = static_cast<double>(_vAxisSign) * (_focalY * yd) + _principalY;

        double ru = u_current - pixel[0];
        double rv = v_current - pixel[1];

        if (std::fabs(ru) < tol && std::fabs(rv) < tol)
            break;

        // 数值 Jacobian（dx = dy = 1e-7）
        const double h = 1e-7;
        double xdh, ydh;
        applyTsaiDistortion(x + h, y, xdh, ydh);
        double J00 = static_cast<double>(_uAxisSign) * _focalX * (xdh - xd) / h;
        double J10 = static_cast<double>(_vAxisSign) * _focalY * (ydh - yd) / h;

        applyTsaiDistortion(x, y + h, xdh, ydh);
        double J01 = static_cast<double>(_uAxisSign) * _focalX * (xdh - xd) / h;
        double J11 = static_cast<double>(_vAxisSign) * _focalY * (ydh - yd) / h;

        // Newton step: delta = -J^{-1} * r
        double det = J00 * J11 - J01 * J10;
        if (std::fabs(det) < 1e-15) break;

        x -= ( J11 * ru - J01 * rv) / det;
        y -= (-J10 * ru + J00 * rv) / det;
    }

    norm[0] = x;
    norm[1] = y;
    return true;
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
bool Camera::saveToFile(const std::string &path) const
{
    std::ofstream ofs = xjw::common::io::openOutputFile(path);
    if (!ofs) return false;
    // 使用 double 类型能表示的最大有效位数，避免精度损失
    ofs << std::setprecision(std::numeric_limits<double>::max_digits10);
    // 乘回 pitch，将像素单位的内参转换为 mm 后写入文件。
    ofs << "fu = " << (_focalX * _pixelPitch) << "\n";
    ofs << "fv = " << (_focalY * _pixelPitch) << "\n";
    ofs << "cu = " << (_principalX * _pixelPitch) << "\n";
    ofs << "cv = " << (_principalY * _pixelPitch) << "\n";
    ofs << "c = " << _cameraCenter[0] << " " << _cameraCenter[1] << " " << _cameraCenter[2] << "\n";
    // 旋转矩阵以空格分隔，行优先，共 9 个值
    ofs << "r = ";
    for (int i = 0; i < 9; i++)
    {
        ofs << _cameraToWorldRotation[i];
        if (i < 8)
        {
            ofs << " ";
        }
    }
    ofs << "\n";
    ofs << "k1 = " << _radialK1 << "\n";
    ofs << "k2 = " << _radialK2 << "\n";
    ofs << "k3 = " << _radialK3 << "\n";
    ofs << "p1 = " << _tangentialP1 << "\n";
    ofs << "p2 = " << _tangentialP2 << "\n";
    ofs << "pitch = " << _pixelPitch << "\n";
    ofs << "u_direction = " << _uAxisSign << "\n";
    ofs << "v_direction = " << _vAxisSign << "\n";
    ofs << "w_direction = " << (_depthAxisFlipped ? -1 : 1) << "\n";
    return true;
}

} // namespace xjw
