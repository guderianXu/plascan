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
        double focalX = 0.0; ///< 水平方向焦距 fx，单位为像素（pixel）。
        double focalY = 0.0; ///< 垂直方向焦距 fy，单位为像素（pixel）。
        double principalX = 0.0; ///< 主点横坐标 cx，单位为像素（pixel）。
        double principalY = 0.0; ///< 主点纵坐标 cy，单位为像素（pixel）。
        double pixelPitch = 1.0; ///< 像元尺寸，单位为 mm/pixel，用于 Tsai 文件和运行态单位换算。
        int uAxisSign = 1; ///< 像素 u 轴相对相机 x 轴的方向，只允许 +1 或 -1。
        int vAxisSign = 1; ///< 像素 v 轴相对相机 y 轴的方向，只允许 +1 或 -1。
    };

    /**
     * @brief Tsai/Brown-Conrady 畸变参数。
     *
     * 系数作用于透视除法后的无量纲归一化坐标；默认全零表示不施加镜头畸变。
     */
    struct Distortion
    {
        double radialK1 = 0.0; ///< 一阶径向畸变系数，对应 r^2 项。
        double radialK2 = 0.0; ///< 二阶径向畸变系数，对应 r^4 项。
        double radialK3 = 0.0; ///< 三阶径向畸变系数，对应 r^6 项。
        double tangentialP1 = 0.0; ///< 第一切向畸变系数 p1。
        double tangentialP2 = 0.0; ///< 第二切向畸变系数 p2。
    };

    /**
     * @brief 相机位姿描述。
     *
     * `cameraToWorldRotation` 采用 camera-to-world 约定，
     * `cameraCenter` 为世界坐标系中的相机光心，单位为米（m）。
     */
    struct Pose
    {
        /// 行优先 3×3 旋转矩阵 R_cw，把相机坐标轴旋转到世界坐标系；默认单位阵。
        std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0,
                                                     0.0, 1.0, 0.0,
                                                     0.0, 0.0, 1.0}};
        std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}}; ///< 世界坐标系中的相机光心 C，单位为米。
        bool depthAxisFlipped = false; ///< true 表示物理前方位于 Z_cam < 0 一侧，否则位于 Z_cam > 0。
    };

    /// 默认构造函数，相机处于“未加载”状态（`isValid()` 返回 false）
    Camera() = default;

    /// 返回相机是否已由文件或显式 setter 初始化；不代表旋转矩阵等参数已经过完整物理校验。
    bool isValid() const { return _isLoaded; }

    /// 返回显式内参描述。
    Intrinsics intrinsics() const { return _intrinsics; }

    /// 返回显式畸变描述。
    Distortion distortion() const { return _distortion; }

    /// 返回显式位姿描述。
    Pose pose() const { return _pose; }

    /// 返回 camera-to-world 旋转矩阵。
    std::array<double, 9> cameraToWorldRotation() const { return _pose.cameraToWorldRotation; }

    /// 返回 world-to-camera 旋转矩阵。
    std::array<double, 9> worldToCameraRotation() const;

    /// 返回世界坐标系中的相机中心。
    std::array<double, 3> cameraCenter() const { return _pose.cameraCenter; }

    /// 返回 world-to-camera 平移向量 `t = -R_wc * C`。
    std::array<double, 3> worldToCameraTranslation() const;

    /// 返回运行态 x 方向焦距（像素）。
    double focalX() const { return _intrinsics.focalX; }
    /// 返回运行态 y 方向焦距（像素）。
    double focalY() const { return _intrinsics.focalY; }
    /// 返回运行态主点 u 坐标（像素）。
    double principalX() const { return _intrinsics.principalX; }
    /// 返回运行态主点 v 坐标（像素）。
    double principalY() const { return _intrinsics.principalY; }
    /// 返回像元大小（mm/pixel）。
    double pixelPitch() const { return _intrinsics.pixelPitch; }
    /// 返回文件/项目元数据语义下的 x 方向焦距（mm）。
    double focalXMillimeters() const { return _intrinsics.focalX * _intrinsics.pixelPitch; }
    /// 返回文件/项目元数据语义下的 y 方向焦距（mm）。
    double focalYMillimeters() const { return _intrinsics.focalY * _intrinsics.pixelPitch; }
    /// 返回文件/项目元数据语义下的主点 u 坐标（mm）。
    double principalXMillimeters() const { return _intrinsics.principalX * _intrinsics.pixelPitch; }
    /// 返回文件/项目元数据语义下的主点 v 坐标（mm）。
    double principalYMillimeters() const { return _intrinsics.principalY * _intrinsics.pixelPitch; }
    /// 返回像素 u 轴相对相机 x 轴的方向符号（仅为 +1 或 -1）。
    int uAxisSign() const { return _intrinsics.uAxisSign; }
    /// 返回像素 v 轴相对相机 y 轴的方向符号（仅为 +1 或 -1）。
    int vAxisSign() const { return _intrinsics.vAxisSign; }
    /// 返回是否使用 `Z_cam < 0` 作为物理前向；该标志来自 Tsai `w_direction`。
    bool depthAxisFlipped() const { return _pose.depthAxisFlipped; }

    /**
     * @brief 返回投影等价、物理前方统一为 `Z_cam > 0` 的 Camera。
     * @details 将 Tsai 的 u/v/w 轴方向折叠进位姿，并同步变换切向畸变系数；
     *          返回值仍然保留完整 Camera 语义，不创建第二套相机模型。
     */
    Camera normalizedForPositiveDepth() const;

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
     * @brief 将世界点投影到像素并返回物理前向正深度。
     * @param world 世界坐标 `[X, Y, Z]`。
     * @param pixel 输出带 Brown-Conrady 畸变的像素坐标 `[u, v]`。
     * @param positiveDepth 输出沿物理前向光轴的正深度。
     * @return 相机有效且点位于物理前方时返回 true。
     */
    bool projectWorldPointWithDepth(const double world[3],
                                    double pixel[2],
                                    double &positiveDepth) const;

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
    void setPose(const std::array<double, 9> &R, const std::array<double, 3> &C);

    /// 仅更新相机中心坐标（不修改旋转矩阵）
    void setCameraCenter(const std::array<double, 3> &cameraCenter);

    /**
     * @brief 设置内部计算使用的相机内参（焦距 + 主点），单位为像素。
     */
    void setIntrinsics(double fu, double fv, double cu, double cv);

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
                                  double pitchMmPerPixel);

    /// 设置像元大小（mm/pixel）。
    void setPixelPitch(double pixelPitch);

    /**
     * @brief 设置像平面坐标轴方向符号。
     * @param uDir  u 轴方向，正数表示 +1，负数表示 -1
     * @param vDir  v 轴方向，正数表示 +1，负数表示 -1
     */
    void setAxisDirections(int uDir, int vDir);

    /// 设置光轴正方向是否翻转到负 Z。
    void setDepthAxisFlipped(bool depthAxisFlipped);

    /**
     * @brief 设置 Tsai 畸变系数。
     * @param k1,k2,k3  径向畸变系数（偶次幂）
     * @param p1,p2     切向畸变系数（Plumb Bob 模型）
     */
    void setDistortion(double k1, double k2, double k3, double p1, double p2);

    /// 使用结构化参数整体更新 Brown-Conrady 畸变系数。
    void setDistortion(const Distortion &distortion);

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
     * @return 相机已加载时返回 true；奇异 Jacobian 或达到迭代上限时返回当前最佳估计
     * @note 当前接口不单独报告“未达到 tol”，调用方如需严格收敛判定，应将返回值投影回像素检查残差。
     */
    bool undistortPixel(const double pixel[2], double norm[2],
                        int maxIter = 20, double tol = 1e-8) const;

    /**
     * @brief 用带畸变像素和物理前向正深度恢复世界点。
     * @param pixel 输入原相机像素坐标 `[u, v]`。
     * @param positiveDepth 沿物理前向光轴的深度，必须为正数。
     * @param world 输出世界坐标 `[X, Y, Z]`。
     * @return 相机有效、输入有限且深度为正时返回 true。
     */
    bool unprojectPixel(const double pixel[2], double positiveDepth, double world[3]) const;

    /**
     * @brief 将当前相机参数保存为 Tsai 风格文本文件。
     * @param path  目标文件路径
     * @return 成功写入返回 true
     * @note 保存时将 fu/fv/cu/cv 乘回 pitch，按 mm 写回文件；C 按米写回。
     */
    bool saveToFile(const std::string &path) const;

private:
    Intrinsics _intrinsics; ///< 唯一内参状态，运行态焦距和主点使用像素单位。
    Distortion _distortion; ///< 唯一 Brown-Conrady 畸变状态。
    Pose _pose; ///< 唯一位姿和深度轴方向状态。
    bool _isLoaded = false; ///< 是否已成功加载（或手动初始化）

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
