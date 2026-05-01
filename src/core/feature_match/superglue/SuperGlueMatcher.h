// =============================================================================
// 文件: SuperGlueMatcher.h
// 模块: SuperGlue 包角 (Graph Neural Network 特征匹配)
// 说明:
//   准载并运行基于注意力机制的 SuperGlue 匹配网络。
//   SuperGlue 将两张图像的关键点+评分+描述子作为输入，
//   通过 GNN 实现跨图像注意力传递，最终利用 Sinkhorn 算法
//   求解最优点图匹配（含垂堀点 dustbin 处理无匹配点）。
//
//   依赖: LibTorch (TorchScript)、OpenCV
//   主要流程:
//     1. 加载模型、上设备
//     2. 构建输入字典 (keypoints0/1, scores0/1, descriptors0/1, image0/1 形状)
//     3. 执行前向推理
//     4. 输出 matches0/1 及 matching_scores0/1
//   限制: descriptors 形状应为 [1, D, N]（与 SuperPoint 输出 [N, D] 不同，调用前需转置）
// =============================================================================
#ifndef SUPERGLUE_MATCHER_H
#define SUPERGLUE_MATCHER_H

#include "../match.h"

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <mutex>

namespace superglue 
{

using MatchResult = xjw::feature_match::MatchResult;

/**
 * @brief 关键点数据结构
 */
struct KeypointData 
{
    std::vector<cv::KeyPoint> keypoints;  // 关键点位置
    std::vector<float> scores;             // 关键点得分
    torch::Tensor descriptors;             // 描述符 [1, D, N]
    
    KeypointData() = default;
    KeypointData(const std::vector<cv::KeyPoint>& kpts, 
                 const std::vector<float>& sc,
                 const torch::Tensor& desc)
        : keypoints(kpts), scores(sc), descriptors(desc) {}
};

/**
 * @brief SuperGlue 配置参数
 * 分为四类：基础参数、高级参数、系统参数、调试参数
 */
struct SuperGlueConfig 
{
    // ========== 基础参数 ==========
    std::string model_path;              // 模型文件路径
    float match_threshold;               // 匹配置信度阈值 [0.0, 1.0]，默认0.2
    
    // ========== 高级参数 ==========
    int batch_size;                      // 批处理大小，默认1
    int sinkhorn_iterations;             // Sinkhorn迭代次数，默认100
    int max_keypoints;                   // 最大关键点数量限制，-1表示无限制
    
    // ========== 系统参数 ==========
    bool use_cuda;                       // 是否使用CUDA加速，默认true
    int cuda_device_id;                  // CUDA设备ID，默认0
    int num_threads;                     // CPU线程数，默认-1（自动）
    
    // ========== 调试参数 ==========
    bool enable_csv_output;              // 是否输出匹配关系到CSV文件
    std::string csv_output_path;         // CSV输出路径，默认"matches.csv"
    bool enable_visualization;           // 是否输出匹配可视化图像
    std::string visualization_output_path; // 可视化图像输出路径，默认"matches.jpg"
    bool verbose;                        // 是否打印详细日志
    
    // 默认构造函数
    SuperGlueConfig()
        : model_path("")
        , match_threshold(0.2f)
        , batch_size(1)
        , sinkhorn_iterations(100)
        , max_keypoints(-1)
        , use_cuda(true)
        , cuda_device_id(0)
        , num_threads(-1)
        , enable_csv_output(false)
        , csv_output_path("matches.csv")
        , enable_visualization(false)
        , visualization_output_path("matches.jpg")
        , verbose(false)
    {}
    
    // 带模型路径的构造函数
    explicit SuperGlueConfig(const std::string& path)
        : SuperGlueConfig()
    {
        model_path = path;
    }
};

/**
 * @brief SuperGlue匹配器类
 * 用于加载和运行SuperGlue模型进行特征匹配
 * 支持批处理、CUDA加速、调试输出等功能
 */
class SuperGlueMatcher : public xjw::feature_match::IFeatureMatcher 
{
public:
    /**
     * @brief 构造函数（使用配置对象）
     * @param config SuperGlue配置参数
     */
    explicit SuperGlueMatcher(const SuperGlueConfig& config);
    
    /**
     * @brief 构造函数（简化版，仅指定模型路径）
     * @param model_path SuperGlue TorchScript模型路径
     * @param use_cuda 是否使用CUDA加速
     */
    explicit SuperGlueMatcher(const std::string& model_path, 
                             bool use_cuda = true);
    
    /**
     * @brief 析构函数
     */
    ~SuperGlueMatcher();

    /**
     * @brief IFeatureMatcher 统一接口：基于 FeatureData 执行匹配。
     */
    MatchResult match(const xjw::feature_extractors::FeatureData &feat0,
                     const xjw::feature_extractors::FeatureData &feat1) override;

    std::string algorithmName() const override { return "superglue"; }
    
    /**
     * @brief 执行特征匹配
     * @param keypoints0 图像0的关键点数据
     * @param keypoints1 图像1的关键点数据
     * @param image0_shape 图像0的形状 [1, 1, H, W]
     * @param image1_shape 图像1的形状 [1, 1, H, W]
     * @return 匹配结果
     */
    MatchResult match(const KeypointData& keypoints0,
                     const KeypointData& keypoints1,
                     const std::vector<int>& image0_shape,
                     const std::vector<int>& image1_shape);
    
    /**
     * @brief 执行特征匹配（从OpenCV图像）
     * @param image0 输入图像0
     * @param image1 输入图像1
     * @param keypoints0 图像0的关键点数据
     * @param keypoints1 图像1的关键点数据
     * @param image_pair_name 图像对名称（用于调试输出）
     * @return 匹配结果
     */
    MatchResult match(const cv::Mat& image0,
                     const cv::Mat& image1,
                     const KeypointData& keypoints0,
                     const KeypointData& keypoints1,
                     const std::string& image_pair_name = "");
    
    /**
     * @brief 批量特征匹配
     * @param images0 第一组图像列表
     * @param images1 第二组图像列表
     * @param keypoints0_batch 第一组关键点数据列表
     * @param keypoints1_batch 第二组关键点数据列表
     * @return 匹配结果列表
     */
    std::vector<MatchResult> matchBatch(
        const std::vector<cv::Mat>& images0,
        const std::vector<cv::Mat>& images1,
        const std::vector<KeypointData>& keypoints0_batch,
        const std::vector<KeypointData>& keypoints1_batch);
    
    // ========== 参数设置接口 ==========
    
    /**
     * @brief 设置匹配阈值
     */
    void setMatchThreshold(float threshold) 
    { 
        config_.match_threshold = threshold; 
    }
    
    /**
     * @brief 设置批处理大小
     */
    void setBatchSize(int batch_size) 
    { 
        config_.batch_size = batch_size; 
    }
    
    /**
     * @brief 设置是否启用CSV输出
     */
    void setEnableCSVOutput(bool enable, const std::string& path = "") 
    {
        config_.enable_csv_output = enable;
        if (!path.empty()) config_.csv_output_path = path;
        if (enable && !csv_file_.is_open()) 
        {
            openCSVFile();
        }
    }
    
    /**
     * @brief 设置是否启用可视化输出
     */
    void setEnableVisualization(bool enable, const std::string& path = "") 
    {
        config_.enable_visualization = enable;
        if (!path.empty()) config_.visualization_output_path = path;
    }
    
    /**
     * @brief 设置详细日志输出
     */
    void setVerbose(bool verbose) 
    { 
        config_.verbose = verbose; 
    }
    
    /**
     * @brief 获取当前配置
     */
    const SuperGlueConfig& getConfig() const
    { 
        return config_; 
    }
    
    /**
     * @brief 更新配置（部分参数不能运行时修改，如模型路径、CUDA设置）
     */
    void updateConfig(const SuperGlueConfig& config);
    
    // ========== 状态查询接口 ==========
    
    /**
     * @brief 获取匹配阈值
     */
    float getMatchThreshold() const 
    { 
        return config_.match_threshold; 
    }
    
    /**
     * @brief 检查模型是否已加载
     */
    bool isLoaded() const 
    { 
        return model_loaded_; 
    }
    
    /**
     * @brief 获取设备信息
     */
    torch::Device getDevice() const 
    { 
        return device_; 
    }
    
    /**
     * @brief 获取已处理的匹配对数量
     */
    int getProcessedCount() const 
    { 
        return processed_count_; 
    }

private:
    /**
     * @brief 初始化模型
     */
    void initModel();
    
    /**
     * @brief 准备输入数据字典
     */
    torch::Dict<std::string, torch::Tensor> prepareInput(
        const KeypointData& keypoints0,
        const KeypointData& keypoints1,
        const std::vector<int>& image0_shape,
        const std::vector<int>& image1_shape);
    
    /**
     * @brief 从关键点向量转换为tensor
     */
    torch::Tensor keypointsToTensor(const std::vector<cv::KeyPoint>& keypoints);
    
    /**
     * @brief 从得分向量转换为tensor
     */
    torch::Tensor scoresToTensor(const std::vector<float>& scores);
    
    /**
     * @brief 解析模型输出
     */
    MatchResult parseOutput(const torch::Dict<std::string, torch::Tensor>& output);
    
    /**
     * @brief 创建虚拟图像张量（用于形状信息）
     */
    torch::Tensor createDummyImage(const std::vector<int>& shape);
    
    /**
     * @brief 打开CSV文件并写入表头
     */
    void openCSVFile();
    
    /**
     * @brief 关闭CSV文件
     */
    void closeCSVFile();
    
    /**
     * @brief 输出匹配关系到CSV文件
     */
    void saveMatchesToCSV(const MatchResult& result,
                         const std::string& image_pair_name);
    
    /**
     * @brief 保存匹配可视化图像
     */
    void saveMatchVisualization(const cv::Mat& image0,
                               const cv::Mat& image1,
                               const KeypointData& keypoints0,
                               const KeypointData& keypoints1,
                               const MatchResult& result,
                               const std::string& output_name = "");
    
    /**
     * @brief 限制关键点数量
     */
    KeypointData limitKeypoints(const KeypointData& kpts, int max_num);
    
    /**
     * @brief 日志输出
     */
    void log(const std::string& message) const;

    SuperGlueConfig config_;              // 配置参数
    torch::jit::script::Module module_;  // TorchScript模型
    torch::Device device_;                // 计算设备
    bool model_loaded_;                   // 模型加载状态
    int processed_count_;                 // 已处理匹配对计数
    std::ofstream csv_file_;              // CSV文件流
    mutable std::mutex csv_mutex_;        // CSV文件操作互斥锁
};

/**
 * @brief 辅助函数：可视化匹配结果
 * @param image0 输入图像0
 * @param image1 输入图像1
 * @param keypoints0 图像0的关键点
 * @param keypoints1 图像1的关键点
 * @param result 匹配结果
 * @param output_path 输出图像路径
 */
void visualizeMatches(const cv::Mat& image0,
                     const cv::Mat& image1,
                     const std::vector<cv::KeyPoint>& keypoints0,
                     const std::vector<cv::KeyPoint>& keypoints1,
                     const MatchResult& result,
                     const std::string& output_path);

} // namespace superglue

#endif // SUPERGLUE_MATCHER_H
