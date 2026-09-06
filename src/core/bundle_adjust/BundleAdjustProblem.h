#pragma once

#include <array>
#include <string>
#include <vector>

namespace xjw
{

    /**
     * @brief 单条观测：某个三维点在某台相机中对应的像点坐标。
     *
     * 每条观测将一个三维轨迹点与某台相机中的匹配点关联起来，
     * 是重投影误差计算的最小单元。
     */
    struct BAObservation
    {
        int cameraIndex = -1; ///< 观测所属相机的索引（对应 cameras 向量下标）
        double u = 0.0;       ///< 相机像平面上的 u（列）坐标（像素）
        double v = 0.0;       ///< 相机像平面上的 v（行）坐标（像素）
        double weight = 1.0;  ///< 观测置信权重，通常来自特征尺度、匹配分数或 track confidence
        /// 特征检测尺度（像素）。参考 BA 使用 `weight / measurementScale^2`
        /// 对像方残差白化；保持独立字段，避免把描述子置信度与定位精度混为一谈。
        double measurementScale = 1.0;
    };

    /**
     * @brief LiDAR 局部平面约束：约束 BA 三维点靠近激光点云中的局部平面。
     */
    struct BALaserPlaneConstraint
    {
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        std::array<double, 3> normal{{0.0, 0.0, 1.0}};
        double weight = 1.0;
        double initialSignedDistance = 0.0;
        int sourceFrameIndex = -1;
    };

    enum class BALaserPointMode
    {
        Unspecified, ///< 未解析/未指定，输入验证会拒绝，防止缺先验被误当成 Fixed
        Fixed,       ///< 落点固定在 initialPoint，不创建可变点自由度
        Constrained, ///< 落点可变，并使用完整 3x3 先验平方根信息矩阵
        Free,        ///< 落点可变，必须由至少两台相机的真实 measured 像点约束
    };

    /**
     * @brief 行星激光测高 shot 的独立测距约束。
     *
     * shot 落点不是影像匹配 track，也不需要伪造像点。它使用世界/行星固连坐标系中的
     * initialPoint 作为独立辅助参数块，并把 cameraIndex 对应帧相机的激光发射点与该
     * 落点之间的距离约束到 observedRangeMeters。
     *
     * Fixed、Constrained、Free 三态与 ISIS 激光控制点语义对齐。Constrained 模式由
     * pointPriorSqrtInformation（row-major）对白化后的 `W * (point - pointPrior)`
     * 约束；该 3x3 矩阵可完整表达 ISIS aprioriMatrix 转换得到的非对角相关性。
     * measuredImageObservations 只能放真实测量像点，ProjectedVirtual 不得传入。
     * 当前 frame-camera MVP 仅使用 cameraIndex 的单帧刚性位姿；时间字段只保留用于
     * 报告和未来轨迹扩展，不执行 line-scan/SPICE 轨迹插值。
     */
    struct BALaserRangeConstraint
    {
        int cameraIndex = -1;
        std::array<double, 3> initialPoint{{0.0, 0.0, 0.0}};
        double observedRangeMeters = 0.0;
        double sigmaRangeMeters = 1.0;
        double weight = 1.0;
        std::array<double, 3> leverArmCameraMeters{{0.0, 0.0, 0.0}};
        BALaserPointMode pointMode = BALaserPointMode::Unspecified;
        std::array<double, 3> pointPrior{{0.0, 0.0, 0.0}};
        std::array<double, 9> pointPriorSqrtInformation{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        std::vector<BAObservation> measuredImageObservations;
        std::string shotId;
        double ephemerisTimeSeconds = 0.0;
        int sourceIndex = -1;
    };

    /**
     * @brief 测绘控制点软约束：约束 BA 三维点靠近已知物方控制点坐标。
     */
    struct BAControlPointConstraint
    {
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        double sigmaMeters = 1.0;
        double weight = 1.0;
        int sourceIndex = -1;
    };

    /**
     * @brief 比例尺/标尺软约束：约束两条 BA track 之间的物方距离。
     */
    struct BAScaleBarConstraint
    {
        int trackIndexA = -1;
        int trackIndexB = -1;
        double measuredDistanceMeters = 0.0;
        double sigmaMeters = 1.0;
        double weight = 1.0;
        int sourceIndex = -1;
    };

    /**
     * @brief 相机位姿软先验：用于已知外参不完全可靠时约束 BA 不发生无意义漂移。
     */
    struct BACameraPosePrior
    {
        bool enabled = false;
        std::array<double, 9> cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> cameraCenter{{0.0, 0.0, 0.0}};
        double positionSigmaMeters = 1.0;
        double rotationSigmaDegrees = 2.0;
    };

    /**
     * @brief 相机中心参考层软约束：只限制光心沿参考法向的低频漂移。
     *
     * `referenceSignedDistances` 非空时，每台相机保留进入 BA 前相对参考面的原始
     * 法向偏移，因此不会把真实弧形或起伏轨迹强制压平。空数组保留旧的零距离平面
     * 约束语义，供明确知道相机应共面的调用方使用。
     */
    struct BACameraPlaneConstraint
    {
        bool enabled = false;
        std::array<double, 3> point{{0.0, 0.0, 0.0}};
        std::array<double, 3> normal{{0.0, 0.0, 1.0}};
        std::vector<double> referenceSignedDistances;
        double sigmaMeters = 1.0;
        double weight = 1.0;
    };

    /**
     * @brief 轨迹：一个三维点及其在多幅图像中的观测集合。
     *
     * 轨迹表示多相机共观的同一个物方点，是光束法平差的核心数据单元。
     * 每条轨迹至少需要 2 个观测才能参与优化。
     */
    struct BATrack
    {
        std::array<double, 3> initialPoint{{0.0, 0.0, 0.0}};       ///< 三维点的初始坐标（优化起始值）
        std::vector<BAObservation> observations;                   ///< 所有相机中对该点的观测列表
        std::vector<BALaserPlaneConstraint> laserPlaneConstraints; ///< 可选 LiDAR 点到面软约束
        std::vector<BAControlPointConstraint> controlPointConstraints; ///< 可选 GCP/控制点软约束
    };

} // namespace xjw
