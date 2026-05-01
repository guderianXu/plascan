#pragma once

// ============================================================
// 文件：Camera.h
// 功能：定义单个针孔相机模型（Tsai/ASP 格式），包含：
//   - 内参（焦距 fu/fv，主点 cu/cv，像元大小 pitch）
//   - 外参（旋转矩阵 R，相机中心 C）
//   - 径向 + 切向畸变系数（k1/k2/k3, p1/p2）
//   - 像平面坐标轴方向（u_dir, v_dir）
//
// 约定（ASP .tsai 格式）：
//   * R 矩阵存储的是"相机坐标系→世界坐标系"（camera-to-world）的旋转。
//   * 世界点转换到相机坐标系时需用 R^T 乘以偏移向量。
//   * 文件中的 fu/fv/cu/cv 采用 mm，读取后会结合 pitch(mm/pixel) 转成像素用于计算。
//   * 相机中心 C 在文件和内存中统一按米（m）表示。
// ============================================================

#include <string>
#include <array>
#include <vector>

#include "PositiveDepthCameraModel.h"

namespace xjw
{

/**
 * @brief 单个针孔相机模型。
 *
 * 支持从 ASP `.tsai` 格式文本文件加载相机参数，并提供将三维世界
 * 坐标投影为图像像素坐标的功能。同时暴露内外参的存取接口，方便
 * 光束法平差（BundleAdjust）在优化过程中直接修改相机状态。
 */
class Camera
{
public:
    /**
    * @brief 相机运行态内参描述。
    *
    * 所有焦距与主点坐标均为像素单位；像元大小单独存放在 `pixelPitch` 中。
    */
    struct Intrinsics
    {
        double focalX = 0.0;
        double focalY = 0.0;
        double principalX = 0.0;
        double principalY = 0.0;
        double pixelPitch = 1.0;
        int uAxisSign = 1;
        int vAxisSign = 1;
    };

    /**
     * @brief Tsai/Brown-Conrady 畸变参数。
     */
    struct Distortion
    {
        double radialK1 = 0.0;
        double radialK2 = 0.0;
        double radialK3 = 0.0;
        double tangentialP1 = 0.0;
        double tangentialP2 = 0.0;
    };

    /**
     * @brief 相机位姿描述。
     *
    * `cameraToWorldRotation` 采用 camera-to-world 约定，
    * `cameraCenter` 为世界坐标系中的相机光心，单位为米（m）。
     */
    struct Pose
    {
        std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0,
                                                     0.0, 1.0, 0.0,
                                                     0.0, 0.0, 1.0}};
        std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
        bool depthAxisFlipped = false;
    };

    using PositiveDepthModel = PositiveDepthCameraModel;

    /// 默认构造函数，相机处于“未加载”状态（`isValid()` 返回 false）
    Camera() = default;

    /// 返回相机是否有效。
    bool isValid() const { return _loaded; }

    /// 返回显式内参描述。
    Intrinsics intrinsics() const;

    /// 返回显式畸变描述。
    Distortion distortion() const;

    /// 返回显式位姿描述。
    Pose pose() const;

    /// 返回 camera-to-world 旋转矩阵。
    std::array<double, 9> cameraToWorldRotation() const { return _R; }

    /// 返回 world-to-camera 旋转矩阵。
    std::array<double, 9> worldToCameraRotation() const;

    /// 返回世界坐标系中的相机中心。
    std::array<double, 3> cameraCenter() const { return _C; }

    /// 返回 world-to-camera 平移向量 `t = -R_wc * C`。
    std::array<double, 3> worldToCameraTranslation() const;

    /// 返回运行态 x 方向焦距（像素）。
    double focalX() const { return _fu; }
    /// 返回运行态 y 方向焦距（像素）。
    double focalY() const { return _fv; }
    /// 返回运行态主点 u 坐标（像素）。
    double principalX() const { return _cu; }
    /// 返回运行态主点 v 坐标（像素）。
    double principalY() const { return _cv; }
    /// 返回像元大小（mm/pixel）。
    double pixelPitch() const { return _pitch; }
    /// 返回文件/项目元数据语义下的 x 方向焦距（mm）。
    double focalXMillimeters() const { return _fu * _pitch; }
    /// 返回文件/项目元数据语义下的 y 方向焦距（mm）。
    double focalYMillimeters() const { return _fv * _pitch; }
    /// 返回文件/项目元数据语义下的主点 u 坐标（mm）。
    double principalXMillimeters() const { return _cu * _pitch; }
    /// 返回文件/项目元数据语义下的主点 v 坐标（mm）。
    double principalYMillimeters() const { return _cv * _pitch; }
    int uAxisSign() const { return _u_dir; }
    int vAxisSign() const { return _v_dir; }
    bool depthAxisFlipped() const { return _depth_flipped_z; }

    /// 生成适用于 MVS 的正深度相机模型。
    PositiveDepthModel toPositiveDepthModel() const;

    /// 按比例缩放内参，位姿与畸变保持不变。
    Camera scaledIntrinsics(double scaleX, double scaleY) const;

    /**
     * @brief 从 ASP `.tsai` 格式文本文件加载所有相机参数。
     * @param path  文件路径
    * @return 成功返回 true，文件无法打开则返回 false
    * @note 文件中的 fu/fv/cu/cv 使用 mm，读取完成后会除以 pitch(mm/pixel)
    *       转换为像素单位供内部计算；C 保持为米（m）。
     */
    bool loadFromFile(const std::string &path);

    /**
     * @brief 将三维世界坐标投影到图像像素坐标。
     * @param world  输入，三维世界坐标 [X, Y, Z]
     * @param uv     输出，图像像素坐标 [u, v]
     * @return 投影成功（相机已加载且点在相机前方）返回 true
     * @note 投影流程：
     *   1. 用 R^T 将世界点变换到相机坐标系
     *   2. 透视除法，得到归一化坐标 (x, y)
     *   3. 应用 Tsai 畸变（径向 + 切向）
     *   4. 乘以焦距、加主点，同时考虑 u_dir/v_dir 的符号
     */
    bool projectWorldPoint(const double world[3], double pixel[2]) const;

    /**
     * @brief 有符号投影：允许 Z_cam < 0，使用 cam[2] 本身（带符号）做透视除法。
     * @details 对于相对定向产生的「场景在相机-Z方向」数据，Z_cam = -|depth| < 0，
     *   但投影公式 u = u_dir * fu * (X_cam / Z_cam) + cu 在 Z_cam < 0 时仍然
     *   能给出正确的物理像素坐标（u_dir=1 时给出「镜像」视觉但实为物理正确）。
     *   相比之下，|Z_cam| 版本给出的是镜像像素（反而错误）。
     * @return 相机已加载且 |cam[2]| > 1e-9 时返回 true。
     */
    bool projectWorldPointSigned(const double world[3], double pixel[2]) const;

    /**
     * @brief 将世界坐标变换到相机坐标系（公开版，供外部直接使用）。
     * @param world  输入，世界坐标 [X, Y, Z]
     * @param cam    输出，相机坐标 [Xc, Yc, Zc]（Zc 可为负）
     */
    void worldToCamera(const double world[3], double cameraPoint[3]) const;

    // ---------- 设置器（供测试和光束法平差使用） ----------

    /**
     * @brief 同时设置外参旋转矩阵和相机中心。
     * @param R  camera-to-world 旋转矩阵（行优先，9元素）
     * @param C  相机中心在世界坐标系中的坐标（3元素）
     */
    void setPose(const std::array<double, 9> &R, const std::array<double, 3> &C)
    {
        _R = R;
        _C = C;
        _loaded = true;
    }

    /// 仅更新相机中心坐标（不修改旋转矩阵）
    void setCameraCenter(const std::array<double, 3> &cameraCenter)
    {
        _C = cameraCenter;
    }

    /**
     * @brief 设置内部计算使用的相机内参（焦距 + 主点），单位为像素。
     */
    void setIntrinsics(double fu, double fv, double cu, double cv)
    {
        _fu = fu;
        _fv = fv;
        _cu = cu;
        _cv = cv;
        _loaded = true;
    }

    /**
     * @brief 设置文件/项目元数据语义下的相机内参（mm）。
     * @param fuMm   x 方向焦距（mm）
     * @param fvMm   y 方向焦距（mm）
     * @param cuMm   主点 u 坐标（mm）
     * @param cvMm   主点 v 坐标（mm）
     * @param pitchMmPerPixel 像元大小（mm/pixel）
     */
    void setIntrinsicsMillimeters(double fuMm,
                                  double fvMm,
                                  double cuMm,
                                  double cvMm,
                                  double pitchMmPerPixel)
    {
        _pitch = (pitchMmPerPixel > 0.0) ? pitchMmPerPixel : 1.0;
        setIntrinsics(fuMm / _pitch,
                      fvMm / _pitch,
                      cuMm / _pitch,
                      cvMm / _pitch);
    }

    /// 设置像元大小（mm/pixel）。
    void setPixelPitch(double pixelPitch)
    {
        if (pixelPitch > 0.0)
        {
            _pitch = pixelPitch;
        }
    }

    /**
     * @brief 设置像平面坐标轴方向符号。
     * @param uDir  u 轴方向，正数表示 +1，负数表示 -1
     * @param vDir  v 轴方向，正数表示 +1，负数表示 -1
     */
    void setAxisDirections(int uDir, int vDir)
    {
        _u_dir = (uDir < 0 ? -1 : 1);
        _v_dir = (vDir < 0 ? -1 : 1);
        _loaded = true;
    }

    /// 设置光轴正方向是否翻转到负 Z。
    void setDepthAxisFlipped(bool depthAxisFlipped)
    {
        _depth_flipped_z = depthAxisFlipped;
        _loaded = true;
    }

    /**
     * @brief 设置 Tsai 畸变系数。
     * @param k1,k2,k3  径向畸变系数（偶次幂）
     * @param p1,p2     切向畸变系数（Plumb Bob 模型）
     */
    void setDistortion(double k1, double k2, double k3, double p1, double p2)
    {
        _k1 = k1;
        _k2 = k2;
        _k3 = k3;
        _p1 = p1;
        _p2 = p2;
        _loaded = true;
    }

    void setDistortion(const Distortion &distortion)
    {
        setDistortion(distortion.radialK1,
                      distortion.radialK2,
                      distortion.radialK3,
                      distortion.tangentialP1,
                      distortion.tangentialP2);
    }
    /**
     * @brief 将六自由度增量应用到相机外参。
     * @param delta  增量数组 [rx, ry, rz, tx, ty, tz]
     *               - (rx, ry, rz)：小角度轴角表示的旋转增量，依次构造
     *                 Rx、Ry、Rz，然后 dR = Rz * Ry * Rx
     *               - (tx, ty, tz)：平移增量，直接加到相机中心 C
     * @note 更新规则：R_new = dR * R_old（均为 camera-to-world 矩阵）；
     *       C_new = C_old + t
     */
    void applyDeltaPose(const double delta[6]);

    /**
     * @brief 像素坐标 → 归一化相机坐标（精确反畸变，Newton 迭代法）。
     *
     * 正向投影：归一化坐标 (x,y) → 畸变后 (xd,yd) → 像素 (u,v)。
     * 反畸变：给定像素 (u,v)，恢复原始归一化坐标 (x,y)（Brown-Conrady 模型的数值反演）。
     *
     * @param pixel  输入像素坐标 [u, v]
     * @param norm   输出归一化相机坐标 [x, y]（即 Xc/Zc, Yc/Zc）
     * @param maxIter Newton 迭代最大次数（默认 20，通常 5 次内收敛）
     * @param tol    收敛阈值（归一化坐标单位，默认 1e-8）
     * @return 相机已加载且迭代收敛时返回 true
     */
    bool undistortPixel(const double pixel[2], double norm[2],
                        int maxIter = 20, double tol = 1e-8) const;

    /**
     * @brief 将当前相机参数保存为 Tsai 风格文本文件。
     * @param path  目标文件路径
     * @return 成功写入返回 true
     * @note 保存时将 fu/fv/cu/cv 乘回 pitch，按 mm 写回文件；C 按米写回。
     */
    bool saveToFile(const std::string &path) const;

private:
    // ---------- 内参 ----------
    double _fu=0.0; ///< x 方向焦距（内部计算单位：像素）
    double _fv=0.0; ///< y 方向焦距（内部计算单位：像素）
    double _cu=0.0; ///< 主点 u 坐标（内部计算单位：像素）
    double _cv=0.0; ///< 主点 v 坐标（内部计算单位：像素）

    // ---------- 外参 ----------
    std::array<double,3> _C{{0,0,0}};          ///< 相机中心在世界坐标系中的位置，单位为米（m）
    std::array<double,9> _R{{1,0,0,0,1,0,0,0,1}}; ///< camera-to-world 旋转矩阵，行优先存储（3×3）

    // ---------- 畸变系数（Tsai/Brown-Conrady 模型） ----------
    double _k1=0; ///< 径向畸变一阶系数（r^2 项）
    double _k2=0; ///< 径向畸变二阶系数（r^4 项）
    double _k3=0; ///< 径向畸变三阶系数（r^6 项）
    double _p1=0; ///< 切向畸变系数 1
    double _p2=0; ///< 切向畸变系数 2

    double _pitch=1.0; ///< 像元大小（mm/pixel），用于 mm 与像素之间的换算
    int _u_dir = 1;    ///< u 轴方向符号（+1 或 -1）
    int _v_dir = 1;    ///< v 轴方向符号（+1 或 -1）
    bool _depth_flipped_z = false; ///< w_direction z 分量 < 0 时为 true（场景在 Z_cam<0 侧）
    bool _loaded = false; ///< 是否已成功加载（或手动初始化）

    /**
     * @brief 利用 R^T 将世界坐标系下的点转换到相机坐标系。
     * @note ASP `.tsai` 文件中的 R 为 camera-to-world，故世界到相机需要转置：
     *       Xc = R^T * (Xw - C)
     */
    void worldToCameraFromCameraToWorldPose(const double world[3], double cameraPoint[3]) const;

    /**
     * @brief 对归一化坐标 (x, y) 应用 Tsai 径向+切向畸变，输出畸变后坐标 (xd, yd)。
     * @param x   归一化 x 坐标（cam[0]/cam[2]）
     * @param y   归一化 y 坐标（cam[1]/cam[2]）
     * @param xd  输出：畸变后归一化 x
     * @param yd  输出：畸变后归一化 y
     */
    void applyTsaiDistortion(double x, double y, double &xd, double &yd) const;
};
} // namespace xjw
