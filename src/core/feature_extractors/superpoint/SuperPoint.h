// =============================================================================
// 文件: SuperPoint.h
// 模块: 特征点提取 (SuperPoint)
// 说明:
//   封装基于 TorchScript 导出的 SuperPoint 深度学习模型的 C++ 推理接口。
//   SuperPoint 是一种自监督训练的特征点检测与描述算法，能同时输出：
//     1. 关键点 (keypoints)：像素坐标 + 响应值 (score)
//     2. 密集描述子 (dense descriptors)：对每个关键点采样得到 256 维向量
//
//   本模块支持：
//     - 单张图像推理 (detect)
//     - 批量推理 (detectBatch)，相同尺寸图像可合并为 batch 以提高 GPU 利用率
//     - 可选的灰度过滤、边界剔除、非极大值抑制 (NMS)、子像素定位
//     - 邻域黑边检测（过滤靠近无效区域的关键点）
//     - 调试输出（CSV 关键点文件、叠加可视化图像）
//
//   依赖: LibTorch (torch/script.h), OpenCV
//   线程安全: detect/detectBatch 在不同线程调用同一实例时需外部加锁（模型推理非线程安全）。
// =============================================================================
#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 配置结构：按功能分组 (基础参数 / 高级参数 / 系统参数 / 调试参数)
//
// 说明：此结构体用于在 C++ 层控制 SuperPoint 推理的行为。
// 摄影测量 GUI 场景下常关心的点包括：检测密度、非极大值抑制半径、描述子维度、
// 输入图像的位深处理以及是否对检测结果按灰度做过滤。下面的注释尽量详细
// 说明每个字段的语义、单位、推荐范围以及与训练时的一致性要求。
struct SuperPointConfig 
{
    // -----------------
    // 基础参数 (Basic)
    // -----------------
    // 非极大值抑制半径（像素）：在关键点局部抑制的半径，值越大抑制越强，
    // 推荐范围 1-8（图像尺度较大时可增大）。用于控制关键点的空间分布。
    // 设置为 3 可在保证关键点质量的同时获得更密集的分布，提升匹配对数量。
    int nms_radius = 3;

    // 检测阈值（score map 阈值）：阈值越低检测到的点越多。
    // 注意：阈值的具体含义依赖于模型输出范围，通常当输入被归一化到 [0,1]
    // 且模型训练时使用相同尺度，0.005 是对 SuperPoint 类模型的常见默认值。
    // 如果训练时输入尺度不同（例如 0-255），则需要调整该阈值。
    // 降低至 0.003 可检测更多弱响应角点，有助于纹理稀疏区域的匹配覆盖。
    float detection_threshold = 0.003f;

    // 最大关键点数量：用于对检测到的关键点按 score 做截断，<=0 表示不限制。
    // 在实时 GUI 中可设置较小值（例如 500-2000）以控制可视化和匹配负载。
    int max_num_keypoints = -1;

    // 在图像边界内移除指定像素数（像素）：防止靠近边界的点被误检或无法正常采样描述子。
    int remove_borders = 4;

    // 灰度过滤阈值区间（归一化后，范围 0.0 - 1.0）。例如设置为 [0.05,0.95]
    // 可以去掉极暗或极亮区域的检测。
    float grayscale_min = 0.0f;
    float grayscale_max = 1.0f;

    // -----------------
    // 高级参数 (Advanced)
    // -----------------
    // 是否对输入图像进行按位深度归一化到浮点数范围（推荐归一化到 0-1，详见实现）。
    // 当你的相机数据为 8/16 位时，建议启用并设置为与训练时一致的规范。
    bool normalize_input = true;

    // 描述子维度（通常为 256），用于分配输出张量大小与匹配逻辑。
    int descriptor_dim = 256;

    // 网络输出到输入影像的下采样步长（grid size），通常为 8。
    // 用于从 dense descriptors 中采样局部描述子。
    int grid_size = 8;

    // 批处理大小（用于 detectBatch）: 当同时处理大量影像时，按相同分辨率分批送入模型，
    // 以减少 C++ 循环开销并提高 GPU 利用率。设置为 <=0 表示禁用批处理，使用逐张调用。
    int batch_size = 8;

    // ===== 邻域判断参数（边界点过滤）=====
    // 检查特征点周围邻域是否有纯黑像素，如果有则丢弃该点
    // 邻域判断范围（像素半径）
    int neighborhood_check_radius = 3;

    // 邻域灰度阈值（0.0-1.0），邻域内有像素低于此值则认为靠近边界
    float neighborhood_threshold = 0.05f;

    // -----------------
    // 系统参数 (System)
    // -----------------
    // 运行设备（CPU 或 CUDA）。构造时会尝试将模型放到此设备上，
    // 如果设置为 CUDA 且系统不支持，会依据 allow_device_fallback 做回退处理。
    torch::Device device = torch::kCUDA;

    // 是否允许在设备不可用时回退到 CPU（默认 true）。GUI 应用通常希望容错。
    bool allow_device_fallback = true;

    // -----------------
    // 调试 (Debug)
    // -----------------

    // 是否在每次推理后将关键点写入 CSV（调试用，默认 false）
    bool save_keypoints_csv = false;

    // 是否在每次推理后将关键点叠加输出为图像（调试用，默认 false）
    bool save_overlay_image = false;

    // 推理前图像最大边长（像素），超过则等比缩放后推理，关键点坐标自动还原到原图。
    // 0 表示不限制。遥感大图建议设置为 2048 或 4096。
    int max_image_size = 0;

    // 构造函数保持默认安全值
    SuperPointConfig() = default;
};

// SuperPoint 单张图像推理结果结构体
// 包含本次推理检测到的所有关键点信息
struct SuperPointOutput 
{
    // 关键点列表（像素坐标系，(pt.x=列, pt.y=行)，已可选做子像素精化）
    std::vector<cv::KeyPoint> keypoints;
    // 每个关键点的检测响应值（与 scores map 中的 softmax 输出对应）
    std::vector<float> scores;
    // 每个关键点对应的描述子，形状 [N, 256]（CPU float32 Tensor）
    // 描述子已经过 L2 归一化，可直接用于余弦相似度匹配
    torch::Tensor descriptors;  // [N, 256]
};

// SuperPoint 特征点检测器类
// 负责加载 TorchScript 模型、管理设备（CPU/CUDA）、执行推理并后处理输出。
// 典型使用流程：
//   1. 构造：SuperPoint sp("model.pt", config);  // 加载模型并预热
//   2. 推理：auto out = sp.detect(image);          // 获得关键点+描述子
//   3. 保存：QFileBinaryIO::write(path, name, out); // 持久化
class SuperPoint 
{
public:
    // 构造/析构
    // model_path: TorchScript 模型路径（.pt），模型应接受一个单通道张量并返回
    // score map 与 dense descriptors（或与导出脚本约定的 tuple 输出）。
    // 构造时会自动：加载模型 → 移动到目标设备 → eval 模式 → CUDA 预热（若使用 GPU）
    SuperPoint(const std::string& model_path, const SuperPointConfig& config = SuperPointConfig());
    ~SuperPoint();

    // 单张图像推理接口
    // image: 支持 CV_8U / CV_16U / CV_16S / CV_32F 等常见格式。函数内部会根据
    // config_.normalize_input 做相应缩放（推荐使用 0-1 归一化）。返回值为
    // 包含关键点（像素坐标）、分数和描述子的 `SuperPointOutput`。
    SuperPointOutput detect(const cv::Mat& image);

    // 批量推理：接受多张图像并返回每张的输出。实现可用于并行推理或批处理。
    std::vector<SuperPointOutput> detectBatch(const std::vector<cv::Mat>& images);

    // 输出接口（便于 GUI 层或调试使用）
    // 将关键点写为 CSV 文件（每行: x,y,score）
    static bool saveKeypointsCSV(const SuperPointOutput& output, const std::string& path);

    // 将关键点叠加到原图并保存为图像文件（BGR 输出）
    static bool saveOverlayImage(const cv::Mat& image, const SuperPointOutput& output, const std::string& path);

    // 辅助函数
    // setDevice: 在运行时切换设备（例如从 CPU 切换到 CUDA），会尝试将模型和缓存移动到新设备。
    void setDevice(torch::Device device);
    SuperPointConfig getConfig() const { return config_; }
    // NOTE: 保存/加载由独立的二进制 I/O 类处理（例如 `QFileBinaryIO`），
    //   - 设计理由：保持推理类集中于推理逻辑，I/O 由 GUI 层或专用类控制更灵活。
    //   - 如果需要不同序列化格式（例如 NPY、PROTOBUF），在 GUI 层调用相应工具类即可。

private:
    // TorchScript 模型对象（持有模型权重与计算图）
    torch::jit::script::Module model_;
    // 推理配置（含设备、阈值、NMS半径等所有超参数）
    SuperPointConfig config_;
    // 保存最近一次预处理的灰度图（CPU，归一化 [0,1]），用于灰度过滤
    // 形状: [H, W]，float32
    torch::Tensor last_gray_cpu_;
    // 原始模型路径（用于诊断时加载 CPU 版本的模型进行对比）
    std::string model_path_;
    
    // 预处理图像，返回 CPU float32 tensor，形状 [1,1,H,W]
    // 支持 CV_8U/CV_16U/CV_16S/CV_32F 位深，按配置决定是否归一化到 [0,1]
    // set_last_gray=true 时同步更新 last_gray_cpu_（单张推理路径使用）
    // 批量推理预处理时传 false，避免多线程写共享状态
    torch::Tensor preprocessImageCPU(const cv::Mat& image, bool set_last_gray = true);
    // 兼容包装器：始终更新 last_gray_cpu_，供 detect() 单张路径调用
    torch::Tensor preprocessImage(const cv::Mat& image);
    
    // 密集 score map 的批量非极大值抑制 (NMS)
    // 输入 scores: [B,C,H,W]，在 nms_radius 像素半径内保留局部最大值
    // 采用 max_pool2d + 迭代抑制策略，抑制半径内的非最大点
    torch::Tensor batchedNMS(const torch::Tensor& scores, int nms_radius);
    
    // 后处理：从 dense score map 和 dense descriptors 中提取关键点
    // 步骤: NMS → 阈值过滤 → top-k → 灰度过滤 → 边界剔除 → 描述子采样
    // scores: [H, W]，descriptors_dense: [1,256,H/8,W/8]
    SuperPointOutput postprocess(const torch::Tensor& scores, 
                                  const torch::Tensor& descriptors_dense,
                                  int img_width, int img_height);
    
    // 从密集描述子特征图中为关键点采样描述子（双线性插值）
    // keypoints: [1,N,2]（x,y 像素坐标）
    // descriptors: [1,C,H,W]（dense feature map，通常 H=orig_H/8, W=orig_W/8）
    // grid_size: 下采样倍数（通常为 8），用于归一化坐标到 [-1,1]
    // 返回: [N, C]，L2 归一化的每点描述子
    torch::Tensor sampleDescriptors(const torch::Tensor& keypoints,
                                     const torch::Tensor& descriptors,
                                     int grid_size = 8);
};

#endif // SUPERPOINT_H
