#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <string>
#include <vector>

// Forward declaration - 避免强制包含 SuperPoint.h
struct FeatureOutput;

namespace xjw::feature_extractors
{

/**
 * @brief 通用特征提取结果接口。
 *
 * 统一封装传统算法（ORB / SIFT）与深度学习算法（SuperPoint 等）的特征提取输出，
 * 便于下游匹配器（SuperGlue、LightGlue、传统匹配器）统一消费，避免各处重复适配。
 *
 * 存储约定：
 *  - keypoints  : 像素坐标，顺序与 descriptors 行对应。
 *  - scores     : 各关键点响应得分；传统算法（cv::KeyPoint::response）会自动填充，
 *                 若无则为 0.0。
 *  - descriptors: [N, D] float32 CV_32F 矩阵；binary 描述子（CV_8U）在构建时自动
 *                 bit-expand 转换为 float，以保持统一格式。
 *  - sourceAlgorithm: 小写规范名，如 "superpoint" / "orb" / "sift"。
 */
struct FeatureData
{
    std::vector<cv::KeyPoint> keypoints; ///< 关键点列表（像素坐标）
    std::vector<float> scores;           ///< 各关键点响应得分
    cv::Mat descriptors;                 ///< 描述子矩阵 [N, D]，CV_32F
    std::string sourceAlgorithm;         ///< 来源算法名（小写），如 "superpoint"/"orb"/"sift"
    int imageWidth = 0;                  ///< 原图宽（像素），0 表示未知
    int imageHeight = 0;                 ///< 原图高（像素），0 表示未知

    FeatureData() = default;

    bool empty() const { return keypoints.empty(); }
    int  size()  const { return static_cast<int>(keypoints.size()); }

    /// @brief 描述子维度 (0 表示无描述子)
    int  descriptorDim() const { return descriptors.empty() ? 0 : descriptors.cols; }

    /// @brief 是否为 binary 描述子 (需 Hamming 距离匹配)
    bool isBinary() const { return sourceAlgorithm == "orb"; }

    /// @brief 推荐的匹配器算法名
    std::string recommendedMatcher() const
    {
        if (sourceAlgorithm == "superpoint") return "superglue";
        if (sourceAlgorithm == "disk")       return "lightglue";
        if (sourceAlgorithm == "aliked")     return "lightglue";
        if (sourceAlgorithm == "sift")       return "sift_bf_l2";
        if (sourceAlgorithm == "orb")        return "orb_bf_hamming";
        if (sourceAlgorithm == "akaze")      return "orb_bf_hamming";
        return "sift_bf_l2";
    }

    // -----------------------------------------------------------------------
    // 格式转换
    // -----------------------------------------------------------------------

    /**
     * @brief 转换为 torch::Tensor [N, D]，float32，CPU。
     *        描述子为空时返回空张量。
     */
    torch::Tensor toTensorND() const;

    /**
     * @brief 转换为 SuperGlue 所需的 torch::Tensor [1, D, N]，float32，CPU。
     *        描述子为空时返回空张量。
     */
    torch::Tensor toTensorSuperGlue() const;

    /**
     * @brief 将描述子转换为传统匹配算法所需的 cv::Mat 格式。
     *
     * - ORB / Hamming 匹配 → CV_8U bit-packed（每行 D/8 字节）
     * - SIFT / FLANN / L2 匹配 → CV_32F，直接返回 descriptors
     *
     * @param matchAlgorithm 小写规范算法名，如 "orb"/"sift"/"flann"。
     */
    cv::Mat toCvDescriptors(const std::string &matchAlgorithm) const;

    /**
     * @brief 从传统特征描述子 cv::Mat 构建 torch::Tensor [N, targetDim]，
     *        用于写入 FeatureOutput::descriptors。
     *
     * - ORB (CV_8U): 逐位展开为 {-1.0, +1.0}，L2 归一化，不足 targetDim 部分补零。
     * - 其他 (CV_32F 等): 转为 float，截断/补零到 targetDim，L2 归一化。
     *
     * @param descriptors            cv::Mat 描述子（CV_8U 或 CV_32F）
     * @param targetDim              目标维度，如 256
     * @param sourceFeatureAlgorithm 来源算法名（小写），如 "orb"/"sift"
     */
    static torch::Tensor cvDescriptorsToTensor(const cv::Mat &descriptors,
                                               int targetDim,
                                               const std::string &sourceFeatureAlgorithm);

    // -----------------------------------------------------------------------
    // 工厂方法
    // -----------------------------------------------------------------------

    /**
     * @brief 从 FeatureOutput 构建。
     *
     * FeatureOutput::descriptors 为 [N, D] CPU float32 torch::Tensor，
     * 会被转换为 cv::Mat [N, D] CV_32F 存储。
     *
     * @param out       SuperPoint 推理结果。
     * @param algoName  算法名，默认为 "superpoint"。
     */
    static FeatureData fromFeatureOutput(const FeatureOutput &out,
                                            const std::string &algoName = "superpoint");

    /**
     * @brief 从 OpenCV 关键点与描述子矩阵构建。
     *
     * @param keypoints  cv::KeyPoint 列表；scores 从 cv::KeyPoint::response 提取。
     * @param descriptors CV_8U（binary）或 CV_32F 描述子矩阵，行数须与 keypoints 一致。
     *                   CV_8U 描述子会 bit-expand 转换为 CV_32F（每位展开为 0.0/1.0）。
     * @param algoName   算法名，如 "orb"/"sift"。
     */
    static FeatureData fromOpenCV(const std::vector<cv::KeyPoint> &keypoints,
                                  const cv::Mat &descriptors,
                                  const std::string &algoName = "");

    /// @brief 从 DISK/Aliked torch::Tensor 输出构建 (keypoints[N,2], descriptors[N,D], scores[N])
    static FeatureData fromTensorOutput(const std::vector<cv::KeyPoint> &kpts,
                                        const std::vector<float> &scores,
                                        const torch::Tensor &descriptors,
                                        const std::string &algoName);
};

} // namespace xjw::feature_extractors
