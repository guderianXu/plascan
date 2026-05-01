// =============================================================================
// 文件: SuperPoint.cpp
// 说明:
//   SuperPoint 特征点提取器的完整实现。
//   主要函数调用关系：
//     detect(image)
//       → preprocessImage      将 cv::Mat 转换为模型输入张量
//       → model_.forward         TorchScript 前向推理
//       → postprocess(封闭)    NMS+阈值+子像素+描述子采样
//     detectBatch(images)
//       → preprocessImageCPU    并行预处理 (std::async)
//       → model_.forward(批)
//       → 分解每张结果并采样描述子
// =============================================================================
#include "SuperPoint.h"
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>
#include <iomanip>
#include <future>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <c10/cuda/CUDACachingAllocator.h>

namespace {
// 邻域判断：检查特征点周围邻域是否有纯黑像素（低于阈值），若有则认为靠近边界
static bool isNearBlackBoundary(const cv::Mat &gray_image, int x, int y, int radius, float threshold)
{
    if (gray_image.empty() || gray_image.type() != CV_32F) {
        return false;
    }
    
    const int h = gray_image.rows;
    const int w = gray_image.cols;
    
    // 遍历邻域窗口
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            
            // 边界检查
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
                continue;
            }
            
            // 检查像素值
            float intensity = gray_image.at<float>(ny, nx);
            if (intensity < threshold) {
                return true;  // 邻域内发现纯黑像素，认为靠近边界
            }
        }
    }
    
    return false;
}

// -------------------------------------------------------------------------
// applySparseNmsByRadius
// 功能: 稀疏 NMS（关键点级非极大值抑制）
//         按分数降序排列，依次检查每个点与已保留点的最小 L2 距离；
//         若距离小于 nmsRadius 则抑制当前点
// 参数:
//   keypoints, scores, indices: 庅平具有相同元素顺序的关键点信息列表
//   nmsRadius: 抑制半径（像素），<=0 时跳过
static void applySparseNmsByRadius(std::vector<cv::KeyPoint> &keypoints,
                                   std::vector<float> &scores,
                                   std::vector<int64_t> &indices,
                                   int nmsRadius)
{
    if (nmsRadius <= 0 || keypoints.empty()) return;

    std::vector<int> order(static_cast<int>(keypoints.size()));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
    });

    const float r2 = static_cast<float>(nmsRadius * nmsRadius);
    std::vector<cv::KeyPoint> keptKpts;
    std::vector<float> keptScores;
    std::vector<int64_t> keptIndices;
    keptKpts.reserve(keypoints.size());
    keptScores.reserve(scores.size());
    keptIndices.reserve(indices.size());

    for (int idx : order) {
        const cv::Point2f p = keypoints[static_cast<size_t>(idx)].pt;
        bool suppressed = false;
        for (const auto &kk : keptKpts) {
            const float dx = p.x - kk.pt.x;
            const float dy = p.y - kk.pt.y;
            if (dx * dx + dy * dy <= r2) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            keptKpts.push_back(keypoints[static_cast<size_t>(idx)]);
            keptScores.push_back(scores[static_cast<size_t>(idx)]);
            keptIndices.push_back(indices[static_cast<size_t>(idx)]);
        }
    }

    keypoints.swap(keptKpts);
    scores.swap(keptScores);
    indices.swap(keptIndices);
}

} // namespace

SuperPoint::SuperPoint(const std::string& model_path, const SuperPointConfig& config)
    : config_(config) 
{
    try 
    {
        // 保存模型路径以便诊断时使用
        model_path_ = model_path;
        // 使用 TorchScript 加载模型（.pt 文件应通过 torch.jit.script/trace 导出）
        model_ = torch::jit::load(model_path);
        
        // 尝试将模型放到指定设备，CUDA 失败时按 allow_device_fallback 回退到 CPU
        if (config_.device.is_cuda()) 
        {
            try 
            {
                // 先检查CUDA是否真的可用
                if (!torch::cuda::is_available()) 
                {
                    std::cout << "警告: CUDA不可用，使用CPU" << std::endl;
                    config_.device = torch::kCPU;
                    model_.to(torch::kCPU);
                } 
                else 
                {
                    model_.to(config_.device);
                    std::cout << "使用CUDA设备" << std::endl;
                }
            } 
            catch (const c10::Error& cuda_error) 
            {
                std::cout << "警告: CUDA初始化失败 (" << cuda_error.what_without_backtrace() 
                         << ")，回退到CPU" << std::endl;
                config_.device = torch::kCPU;
                model_.to(torch::kCPU);
            }
        } 
        else 
        {
            model_.to(config_.device);
            std::cout << "使用CPU设备" << std::endl;
        }
        
        model_.eval();  // 切换为推理模式，禁用 Dropout/BatchNorm 的训练行为
        
        // CUDA预热推理：避免第一张影像推理时的 JIT 编译开销
        if (config_.device.is_cuda()) {
            try {
                torch::NoGradGuard no_grad;
                // 创建480x640的dummy输入（典型图像尺寸）
                auto dummy_input = torch::ones({1, 1, 480, 640}, 
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(config_.device));
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(dummy_input);
                
                // 执行预热推理（触发CUDA kernel编译和内存预分配）
                auto _ = model_.forward(inputs);
                
                // 等待CUDA操作完成
                if (config_.device.is_cuda()) {
                    torch::cuda::synchronize();
                }
                
                std::cout << "CUDA预热完成" << std::endl;
            } catch (const c10::Error& warmup_error) {
                // 预热失败不应导致构造失败，只打印警告
                std::cout << "警告: CUDA预热失败 - " << warmup_error.what_without_backtrace() << std::endl;
            }
        }
        
        std::cout << "SuperPoint模型加载成功: " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        throw;
    }
}

SuperPoint::~SuperPoint() 
{
}

void SuperPoint::setDevice(torch::Device device) 
{
    config_.device = device;
    model_.to(device);
}

torch::Tensor SuperPoint::preprocessImageCPU(const cv::Mat& image, bool set_last_gray) 
{
    cv::Mat gray;
    
    // 如果输入是彩色图则转为灰度图（BGR 格式）
    if (image.channels() == 3) 
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } 
    else if (image.channels() == 1) 
    {
        gray = image.clone();
    } 
    else 
    {
        throw std::runtime_error("不支持的图像通道数");
    }

    // 根据图像位深将其归一化到 [0,1] 并转换为 float32
    // 务必保证归一化方式与训练时一致，否则检测会异常
    cv::Mat float_image;
    int depth = gray.depth();
    if (config_.normalize_input) 
    {
        switch (depth) 
        {
            case CV_8U:
                gray.convertTo(float_image, CV_32F, 1.0 / 255.0);
                break;
            case CV_16U:
                gray.convertTo(float_image, CV_32F, 1.0 / 65535.0);
                break;
            case CV_16S:
                gray.convertTo(float_image, CV_32F, 1.0 / 32767.0);
                break;
            case CV_32F:
                gray.convertTo(float_image, CV_32F);
                // 如果浮点图像值远大于1.0，按最大值归一化
                {
                    double minVal, maxVal;
                    cv::minMaxLoc(float_image, &minVal, &maxVal);
                    if (maxVal > 1.0) float_image = float_image / static_cast<float>(maxVal);
                }
                break;
            default:
                gray.convertTo(float_image, CV_32F, 1.0 / 255.0);
                break;
        }
    } 
    else 
    {
        // 仅转换为浮点，不缩放
        gray.convertTo(float_image, CV_32F);
    }


    // 创建CPU端的tensor并保存灰度图（按配置是否已归一化），用于后续的灰度过滤
    torch::Tensor cpu_tensor = torch::from_blob(
        float_image.data,
        {1, float_image.rows, float_image.cols, 1},
        torch::kFloat32
    ).clone();

    // 调整维度顺序: [1, H, W, 1] -> [1, 1, H, W]
    auto cpu_perm = cpu_tensor.permute({0, 3, 1, 2});

    // 保存为 [H, W] 的CPU灰度图（可选，避免并行写共享状态）
    if (set_last_gray) {
        last_gray_cpu_ = cpu_perm.squeeze(0).squeeze(0).to(torch::kCPU);
    }

    // 返回 CPU tensor（由调用方统一上设备并可选择 pin_memory/非阻塞拷贝）
    return cpu_perm; // shape [1,1,H,W], dtype float32, on CPU
}

// 兼容包装器：默认会更新 last_gray_cpu_
torch::Tensor SuperPoint::preprocessImage(const cv::Mat& image) {
    return preprocessImageCPU(image, true);
}

// -------------------------------------------------------------------------
// batchedNMS: 使用 max_pool2d 实现密集 score map 的 NMS
// 算法步骤:
//   1. 对 scores 做大小为 (2*r+1) 的最大池化，得到局部最大値
//   2. 利用 scores == max_pool(scores) 找到局郠最大点 max_mask
//   3. 迭代抑制 2 次：小邻域内已选点附近的点再尝试插入局郠高分点则保留
// 输入: scores [B,C,H,W]
// 输出: 抑制后的 scores [B,C,H,W]，非最大点被置零
torch::Tensor SuperPoint::batchedNMS(const torch::Tensor& scores, int nms_radius) 
{
    if (nms_radius < 0) 
    {
        return scores;
    }
    
    auto zeros = torch::zeros_like(scores);
    int kernel_size = nms_radius * 2 + 1;
    
    // 最大池化
    auto max_pool = [&](const torch::Tensor& x) 
    {
        return torch::max_pool2d(x, {kernel_size, kernel_size}, 
                                 {1, 1}, {nms_radius, nms_radius});
    };
    
    auto max_mask = scores == max_pool(scores);
    
    // 迭代抑制
    for (int i = 0; i < 2; i++) 
    {
        auto supp_mask = max_pool(max_mask.to(torch::kFloat32)) > 0;
        auto supp_scores = torch::where(supp_mask, zeros, scores);
        auto new_max_mask = supp_scores == max_pool(supp_scores);
        max_mask = max_mask | (new_max_mask & (~supp_mask));
    }
    
    return torch::where(max_mask, scores, zeros);
}

// -------------------------------------------------------------------------
// sampleDescriptors: 利用 grid_sample 对密集描述子特征图做双线性插值采样
// 算法说明:
//   dense descriptors 位于 原图 1/8 分辨率的网格上。
//   要对局郠任意坐标 (x,y) 采样，需将像素坐标归一化到 [-1,1]：
//     x_norm = ((x + 0.5) / (W * grid_size)) * 2 - 1
//     y_norm = ((y + 0.5) / (H * grid_size)) * 2 - 1
//   然后用 torch::grid_sampler 做双线性插值，最后做 L2 归一化
torch::Tensor SuperPoint::sampleDescriptors(const torch::Tensor& keypoints,
                                             const torch::Tensor& descriptors,
                                             int grid_size) 
{
    // keypoints: [N, 2] (x, y) 坐标
    // descriptors: [1, C, H, W] dense描述子特征图
    // grid_size: 下采样倍数 (通常为8)
    // 返回: [N, C] 采样后的per-keypoint描述子
    
    auto sizes = descriptors.sizes();
    int64_t b = sizes[0], c = sizes[1], h = sizes[2], w = sizes[3];

    // 统一 keypoints 为 [N, 2]，兼容调用方传入 [1,N,2] 或 [N,2]
    torch::Tensor kpts_2d = keypoints;
    if (kpts_2d.dim() == 3 && kpts_2d.size(0) == 1)
        kpts_2d = kpts_2d.squeeze(0);  // [N, 2]

    if (kpts_2d.size(0) == 0) {
        // 没有关键点，返回空张量
        return torch::empty({0, c}, descriptors.options());
    }

    auto kpts = kpts_2d.unsqueeze(0); // [1, N, 2]
    float s = static_cast<float>(grid_size);
    auto scale = torch::tensor({
        static_cast<float>(w) * s - s / 2.0f - 0.5f,
        static_cast<float>(h) * s - s / 2.0f - 0.5f
    }, keypoints.options());
    kpts = (kpts - s / 2.0f + 0.5f) / scale;
    kpts = kpts * 2 - 1;

    // 调整形状用于grid_sample: [1, 1, N, 2]
    kpts = kpts.unsqueeze(1); // [1, 1, N, 2]

    // 双线性插值采样描述子（align_corners=true，与 LightGlue 官方一致）
    // grid_sampler 参数：interpolation_mode=0(bilinear), padding_mode=1(border), align_corners=true
    auto sampled = torch::grid_sampler(descriptors, kpts, 0, 1, true);
    // sampled shape: [1, C, 1, N]
    
    // 重塑并归一化: [1, C, 1, N] -> [1, C, N] -> [C, N] -> [N, C]
    sampled = sampled.squeeze(2).squeeze(0); // [C, N]
    sampled = torch::nn::functional::normalize(sampled, 
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
    sampled = sampled.transpose(0, 1); // [N, C]
    
    return sampled;
}

// -------------------------------------------------------------------------
// postprocess: 从封闭的 score map + dense descriptors 中提取关键点
// 处理流程:
//   1. batchedNMS    在 score map 上做密集 NMS（优先聊天）
//   2. 阈值过滤  保留 score > detection_threshold 的值
//   3. top-k 截断  按 max_num_keypoints 取分数最高的前 k 点
//   4. 灰度过滤  查询 last_gray_cpu_ 起为 [grayscale_min, grayscale_max]
//   5. 边界剔除  过滤边缘 remove_borders 像素内的点
//   6. 描述子采样 sampleDescriptors(双线性插值) + L2 归一化
SuperPointOutput SuperPoint::postprocess(const torch::Tensor& scores,
                                         const torch::Tensor& descriptors_dense,
                                         int img_width, int img_height) 
{
    SuperPointOutput output;
    
    // 在CPU上处理
    auto scores_cpu = scores.to(torch::kCPU);
    auto desc_cpu = descriptors_dense.to(torch::kCPU);
    
    // 先做 NMS（使用 batchedNMS，需要 4D 输入）然后阈值筛选
    auto scores_nms = scores_cpu;
    if (config_.nms_radius > 0) 
    {
        // batchedNMS 实现期望 [B,C,H,W] 输入，传入后再 squeeze 回来
        auto tmp = scores_cpu.unsqueeze(0).unsqueeze(0); // [1,1,H,W]
        auto nms_out = batchedNMS(tmp, config_.nms_radius);
        scores_nms = nms_out.squeeze(0).squeeze(0);
    }

    // 找到所有高于阈值的点
    auto mask = scores_nms > config_.detection_threshold;
    auto indices = torch::nonzero(mask);
    
    if (indices.size(0) == 0) 
    {
        return output;  // 没有检测到关键点
    }
    
    // 提取坐标 [N, 2] (y, x)
    auto y_coords = indices.select(1, 0);
    auto x_coords = indices.select(1, 1);
    
    // 提取分数
    auto keypoint_scores = scores_cpu.index({mask});
    
    // 排序并选择top-k
    int num_keypoints = indices.size(0);
    if (config_.max_num_keypoints > 0 && num_keypoints > config_.max_num_keypoints) 
    {
        auto sorted = torch::topk(keypoint_scores, config_.max_num_keypoints);
        auto top_indices = std::get<1>(sorted);
        
        x_coords = x_coords.index_select(0, top_indices);
        y_coords = y_coords.index_select(0, top_indices);
        keypoint_scores = std::get<0>(sorted);
        num_keypoints = config_.max_num_keypoints;
    }
    
    // 如果启用了灰度过滤，先基于灰度阈值过滤候选点
    if (!last_gray_cpu_.defined()) 
    {
        // 没有灰度图，跳过过滤
    } 
    else 
    {
        // 将灰度图拉平成一维，按线性索引选取强度
        int img_w = img_width;
        auto gray_flat = last_gray_cpu_.reshape({-1}); // [H*W]
        auto linear = y_coords * img_w + x_coords; // LongTensor
        auto intensities = gray_flat.index_select(0, linear);
        // 构建阈值mask
        auto mask_int = (intensities >= config_.grayscale_min) & (intensities <= config_.grayscale_max);
        // 找到保留的索引
        auto keep_idx = torch::nonzero(mask_int).squeeze(1);
        if (keep_idx.numel() == 0) 
        {
            // 没有任何点满足灰度条件，返回空结果
            return output;
        }
        x_coords = x_coords.index_select(0, keep_idx);
        y_coords = y_coords.index_select(0, keep_idx);
        keypoint_scores = keypoint_scores.index_select(0, keep_idx);
        num_keypoints = x_coords.size(0);
    }
    // 移除图像边界附近的关键点（如果配置了 remove_borders）
    if (config_.remove_borders > 0 && x_coords.numel() > 0) 
    {
        auto cond_x = (x_coords >= config_.remove_borders) & (x_coords < (img_width - config_.remove_borders));
        auto cond_y = (y_coords >= config_.remove_borders) & (y_coords < (img_height - config_.remove_borders));
        auto keep_mask = cond_x & cond_y;
        auto keep_idx = torch::nonzero(keep_mask).squeeze(1);
        if (keep_idx.numel() == 0) return output;
        x_coords = x_coords.index_select(0, keep_idx);
        y_coords = y_coords.index_select(0, keep_idx);
        keypoint_scores = keypoint_scores.index_select(0, keep_idx);
        num_keypoints = x_coords.size(0);
    }

    // 构建关键点张量 [1, N, 2] (x, y)
    auto keypoints_tensor = torch::stack({x_coords.to(torch::kFloat32), 
                                          y_coords.to(torch::kFloat32)}, 1).unsqueeze(0);
    
    // sampleDescriptors 已返回 [N, 256]，已归一化
    auto descriptors = sampleDescriptors(keypoints_tensor, desc_cpu, config_.grid_size);
    
    // 转换为OpenCV格式
    auto x_coords_acc = x_coords.accessor<long, 1>();
    auto y_coords_acc = y_coords.accessor<long, 1>();
    auto scores_acc = keypoint_scores.accessor<float, 1>();
    
    for (int i = 0; i < num_keypoints; i++) 
    {
        // 使用子像素偏移（若 scores_cpu 为密集图可用）来修正整数坐标
        float kx = static_cast<float>(x_coords_acc[i]);
        float ky = static_cast<float>(y_coords_acc[i]);

        if (scores_cpu.defined() && scores_cpu.dim() == 2) {
            auto s_acc = scores_cpu.accessor<float,2>();
            int s_h = static_cast<int>(scores_cpu.size(0));
            int s_w = static_cast<int>(scores_cpu.size(1));
            int xi = static_cast<int>(std::round(kx));
            int yi = static_cast<int>(std::round(ky));
            if (xi > 0 && xi < (s_w - 1) && yi > 0 && yi < (s_h - 1)) {
                float f0 = s_acc[yi][xi];
                float fxm = s_acc[yi][xi - 1];
                float fxp = s_acc[yi][xi + 1];
                float denomx = fxm - 2.0f * f0 + fxp;
                float offx = 0.0f;
                if (std::fabs(denomx) > 1e-6f) offx = 0.5f * (fxm - fxp) / denomx;

                float fym = s_acc[yi - 1][xi];
                float fyp = s_acc[yi + 1][xi];
                float denomy = fym - 2.0f * f0 + fyp;
                float offy = 0.0f;
                if (std::fabs(denomy) > 1e-6f) offy = 0.5f * (fym - fyp) / denomy;

                if (std::fabs(offx) < 1.0f && std::fabs(offy) < 1.0f) {
                    kx += offx;
                    ky += offy;
                }
            }
        }

        cv::KeyPoint kp;
        kp.pt.x = kx;
        kp.pt.y = ky;
        kp.response = scores_acc[i];
        kp.size = 8.0f;  // SuperPoint的网格大小

        output.keypoints.push_back(kp);
        output.scores.push_back(scores_acc[i]);
    }
    
    output.descriptors = descriptors;
    
    return output;
}

// -------------------------------------------------------------------------
// detect: 对单张图像进行完整的 SuperPoint 推理
// 步骤:
//   1. 预处理图像（灰度化+归一化+张量化）并更新 last_gray_cpu_
//   2. 将张量移动到目标设备（CPU 或 CUDA）
//   3. 做前向推理并解析输出（支持 tuple 和 dict 两种模型格式）
//   4. 对模型咆出的局郠关键点做子像素化、灰度迧滤、边界剔除
//   5. 邻域黑边检查、稀疏 NMS、top-k 截断
//   6. 从 dense descriptors 中为每个关键点采样调用一维描述子
SuperPointOutput SuperPoint::detect(const cv::Mat& image) 
{
    torch::NoGradGuard no_grad;  // 禁用梯度计算，减少显存占用
    
    // 预处理：将 cv::Mat 转为归一化浮点 tensor [1,1,H,W]，同时更新 last_gray_cpu_
    auto input_tensor = preprocessImage(image);
    // 将 CPU tensor 移动到目标推理设备（non_blocking=false 以确保同步）
    auto input_device = input_tensor.to(config_.device, /*non_blocking=*/false);

    // 前向推理
    // 新模型wrapper返回: (keypoints [N,2], scores [N], dense_descriptors [1,256,H/8,W/8])
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_device);
    
    // 前向推理：首次 OOM 清空缓存后重试一次；两次都失败返回空结果
    torch::jit::IValue output;
    try {
        output = model_.forward(inputs);
    } catch (const c10::OutOfMemoryError &) {
        if (config_.device.is_cuda()) {
            torch::cuda::synchronize();
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
        try {
            output = model_.forward(inputs);
        } catch (const c10::Error &retry_err) {
            std::cerr << "[SuperPoint] 推理 OOM 重试失败，返回空结果: "
                      << retry_err.what_without_backtrace() << std::endl;
            return SuperPointOutput{};
        }
    } catch (const c10::Error &e) {
        std::cerr << "[SuperPoint] 推理异常，返回空结果: "
                  << e.what_without_backtrace() << std::endl;
        return SuperPointOutput{};
    }
    
    // 解析输出：新模型可能返回 tuple of tensors (3 或 4 元素)
    torch::Tensor keypoints, scores, descriptors_dense, scores_dense;

    if (output.isTuple()) 
    {
        auto output_tuple = output.toTuple();
        size_t ne = output_tuple->elements().size();
        if (ne < 3) {
            throw std::runtime_error("Expected at least 3 outputs from model");
        }

        auto e0 = output_tuple->elements()[0];
        auto e1 = output_tuple->elements()[1];
        auto e2 = output_tuple->elements()[2];

        if (e0.isTensor()) keypoints = e0.toTensor();
        else throw std::runtime_error("Expected keypoints to be tensor");

        if (e1.isTensor()) scores = e1.toTensor();
        else throw std::runtime_error("Expected scores to be tensor");

        if (e2.isTensor()) descriptors_dense = e2.toTensor();
        else throw std::runtime_error("Expected descriptors to be tensor");

        if (ne >= 4) {
            auto e3 = output_tuple->elements()[3];
            if (e3.isTensor()) scores_dense = e3.toTensor();
            // else silently ignore
        }
    } 
    else if (output.isGenericDict()) 
    {
        // 字典模式（旧模型格式，已弃用）
        auto output_dict = output.toGenericDict();
        auto keypoints_list = output_dict.at("keypoints").toList();
        auto scores_list = output_dict.at("keypoint_scores").toList();
        auto descriptors_list = output_dict.at("descriptors").toList();
        
        keypoints = keypoints_list.get(0).toTensor();
        scores = scores_list.get(0).toTensor();
        descriptors_dense = descriptors_list.get(0).toTensor();
    } 
    else 
    {
        throw std::runtime_error("Unsupported model output format");
    }
    
    // 转换为输出格式
    SuperPointOutput result;

    // 当模型返回空关键点（如 superpoint_v1_compat 导出的占位格式）时，
    // 从 dense score map 提取关键点
    if (keypoints.size(0) == 0 && scores_dense.defined())
    {
        auto scores_dense_cpu = scores_dense.to(torch::kCPU);
        if (scores_dense_cpu.dim() == 3)
            scores_dense_cpu = scores_dense_cpu.squeeze(0);
        return postprocess(scores_dense_cpu, descriptors_dense,
                           image.cols, image.rows);
    }

    if (keypoints.size(0) > 0) 
    {
        auto keypoints_cpu = keypoints.to(torch::kCPU);
        auto scores_cpu = scores.to(torch::kCPU);

        auto kpts_acc = keypoints_cpu.accessor<float, 2>();
        auto scores_acc = scores_cpu.accessor<float, 1>();

        // 如果有预处理保存的灰度图，则启用灰度过滤
        bool has_gray = last_gray_cpu_.defined();
        torch::Tensor gray_cpu;
        int gray_h = 0, gray_w = 0;
        if (has_gray) {
            gray_cpu = last_gray_cpu_.to(torch::kCPU);
            if (gray_cpu.dim() == 2) {
                gray_h = static_cast<int>(gray_cpu.size(0));
                gray_w = static_cast<int>(gray_cpu.size(1));
            } else {
                has_gray = false;
            }
        }

        std::vector<cv::KeyPoint> raw_kpts;
        std::vector<float> raw_scores;
        std::vector<int64_t> raw_indices;
        raw_kpts.reserve(static_cast<size_t>(keypoints.size(0)));
        raw_scores.reserve(static_cast<size_t>(keypoints.size(0)));
        raw_indices.reserve(static_cast<size_t>(keypoints.size(0)));

        // 如果模型返回了 dense score map，则将其转换为 CPU 以供子像素化使用
        torch::Tensor scores_dense_cpu;
        if (scores_dense.defined()) {
            scores_dense_cpu = scores_dense.to(torch::kCPU);
            if (scores_dense_cpu.dim() == 3 && scores_dense_cpu.size(0) == 1) {
                scores_dense_cpu = scores_dense_cpu.squeeze(0);
            }
        }

        for (int i = 0; i < keypoints.size(0); i++) 
        {
            if (scores_acc[i] < config_.detection_threshold) {
                continue;
            }

            float kx = kpts_acc[i][0];
            float ky = kpts_acc[i][1];

            int xi = static_cast<int>(std::round(kx));
            int yi = static_cast<int>(std::round(ky));

            if (has_gray) {
                xi = std::max(0, std::min(xi, gray_w - 1));
                yi = std::max(0, std::min(yi, gray_h - 1));
                auto gray_acc = gray_cpu.accessor<float,2>();
                float intensity = gray_acc[yi][xi];
                if (intensity < config_.grayscale_min || intensity > config_.grayscale_max) {
                    continue;
                }
            }

            if (config_.remove_borders > 0 && has_gray) {
                if (xi < config_.remove_borders || xi >= (gray_w - config_.remove_borders) ||
                    yi < config_.remove_borders || yi >= (gray_h - config_.remove_borders)) {
                    continue;
                }
            }

            // 子像素化：使用3x3邻域二次拟合（parabola），基于 dense score map
            if (scores_dense_cpu.defined()) {
                if (scores_dense_cpu.dim() == 2) {
                    auto s_acc = scores_dense_cpu.accessor<float,2>();
                    int s_h = static_cast<int>(scores_dense_cpu.size(0));
                    int s_w = static_cast<int>(scores_dense_cpu.size(1));
                    // 确保邻域完整
                    if (xi > 0 && xi < (s_w - 1) && yi > 0 && yi < (s_h - 1)) {
                        float f0 = s_acc[yi][xi];
                        float fxm = s_acc[yi][xi - 1];
                        float fxp = s_acc[yi][xi + 1];
                        float denomx = fxm - 2.0f * f0 + fxp;
                        float offx = 0.0f;
                        if (std::fabs(denomx) > 1e-6f) offx = 0.5f * (fxm - fxp) / denomx;

                        float fym = s_acc[yi - 1][xi];
                        float fyp = s_acc[yi + 1][xi];
                        float denomy = fym - 2.0f * f0 + fyp;
                        float offy = 0.0f;
                        if (std::fabs(denomy) > 1e-6f) offy = 0.5f * (fym - fyp) / denomy;

                        // 限制偏移范围，避免过大跳变
                        if (std::fabs(offx) < 1.0f && std::fabs(offy) < 1.0f) {
                            kx += offx;
                            ky += offy;
                        }
                    }
                }
            }

            cv::KeyPoint kp;
            kp.pt.x = kx;
            kp.pt.y = ky;
            kp.response = scores_acc[i];
            kp.size = 8.0f;

            raw_kpts.push_back(kp);
            raw_scores.push_back(scores_acc[i]);
            raw_indices.push_back(static_cast<int64_t>(i));
        }

        // 邻域判断：过滤靠近黑色边界的特征点
        if (config_.neighborhood_check_radius > 0 && config_.neighborhood_threshold > 0.0f && has_gray && !raw_kpts.empty()) {
            cv::Mat gray_mat(gray_h, gray_w, CV_32F, gray_cpu.contiguous().data_ptr<float>());
            
            std::vector<cv::KeyPoint> f_kpts;
            std::vector<float> f_scores;
            std::vector<int64_t> f_indices;
            f_kpts.reserve(raw_kpts.size());
            f_scores.reserve(raw_scores.size());
            f_indices.reserve(raw_indices.size());
            
            for (size_t i = 0; i < raw_kpts.size(); ++i) {
                int xi = static_cast<int>(std::round(raw_kpts[i].pt.x));
                int yi = static_cast<int>(std::round(raw_kpts[i].pt.y));
                xi = std::max(0, std::min(xi, gray_w - 1));
                yi = std::max(0, std::min(yi, gray_h - 1));
                
                // 检查邻域内是否有纯黑像素
                if (!isNearBlackBoundary(gray_mat, xi, yi, config_.neighborhood_check_radius, config_.neighborhood_threshold)) {
                    f_kpts.push_back(raw_kpts[i]);
                    f_scores.push_back(raw_scores[i]);
                    f_indices.push_back(raw_indices[i]);
                }
            }
            
            raw_kpts.swap(f_kpts);
            raw_scores.swap(f_scores);
            raw_indices.swap(f_indices);
        }

        // 稀疏关键点NMS（按分数排序后在像素半径内抑制）
        if (config_.nms_radius > 0 && !raw_kpts.empty()) {
            applySparseNmsByRadius(raw_kpts, raw_scores, raw_indices, config_.nms_radius);
        }
        
        // 应用 max_num_keypoints（在过滤后）
        if (config_.max_num_keypoints > 0 && static_cast<int>(raw_kpts.size()) > config_.max_num_keypoints) {
            std::vector<int> order(static_cast<int>(raw_kpts.size()));
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + config_.max_num_keypoints, order.end(), [&](int a, int b) {
                return raw_scores[static_cast<size_t>(a)] > raw_scores[static_cast<size_t>(b)];
            });
            order.resize(config_.max_num_keypoints);

            std::vector<cv::KeyPoint> top_kpts;
            std::vector<float> top_scores;
            std::vector<int64_t> top_indices;
            top_kpts.reserve(order.size());
            top_scores.reserve(order.size());
            top_indices.reserve(order.size());
            for (int idx : order) {
                top_kpts.push_back(raw_kpts[static_cast<size_t>(idx)]);
                top_scores.push_back(raw_scores[static_cast<size_t>(idx)]);
                top_indices.push_back(raw_indices[static_cast<size_t>(idx)]);
            }
            raw_kpts.swap(top_kpts);
            raw_scores.swap(top_scores);
            raw_indices.swap(top_indices);
        }

        result.keypoints = raw_kpts;
        result.scores = raw_scores;

        // 从dense descriptors中采样per-keypoint descriptors
        if (descriptors_dense.defined() && descriptors_dense.dim() == 4 && !raw_kpts.empty()) {
            // 构建过滤后的关键点tensor [N, 2]
            std::vector<float> kp_coords;
            kp_coords.reserve(raw_kpts.size() * 2);
            for (const auto& kp : raw_kpts) {
                kp_coords.push_back(kp.pt.x);
                kp_coords.push_back(kp.pt.y);
            }
            auto kp_tensor = torch::from_blob(kp_coords.data(), 
                                               {static_cast<int64_t>(raw_kpts.size()), 2}, 
                                               torch::kFloat32).clone();
            // 移到与descriptors_dense相同的设备上进行采样
            kp_tensor = kp_tensor.to(descriptors_dense.device());
            
            // 采样描述子，立即移到 CPU 防止显存跨图像累积
            result.descriptors = sampleDescriptors(kp_tensor, descriptors_dense, config_.grid_size)
                                      .to(torch::kCPU);
        } else {
            result.descriptors = torch::Tensor();
        }
    }
    
    // 推理结束后主动释放 CUDA 缓存，阻止碎片在多图提取时堆积
    if (config_.device.is_cuda()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    return result;
}

// -------------------------------------------------------------------------
// saveKeypointsCSV: 将关键点坐标、分数和描述子导出为 CSV 文件
// 表头格式: x,y,score[,d0,...,d{D-1}]
// 描述子派算中会输出说明数据的评断信息
// （为 0 比例较高时会打印警告，帮助排查描述子异常）
bool SuperPoint::saveKeypointsCSV(const SuperPointOutput& output, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    // 如果有描述子，则在表头添加 d0..d{D-1}
    int D = 0;
    torch::Tensor desc_cpu;
    if (output.descriptors.defined()) {
        // 强制转为 CPU + float32 并连续，避免 dtype/布局导致读取错误
        desc_cpu = output.descriptors.to(torch::kCPU);
        if (desc_cpu.dtype() != torch::kFloat32) desc_cpu = desc_cpu.to(torch::kFloat32);
        desc_cpu = desc_cpu.contiguous();
        if (desc_cpu.dim() == 2) D = static_cast<int>(desc_cpu.size(1));
    }

    ofs << "x,y,score";
    if (D > 0) {
        for (int d = 0; d < D; ++d) {
            ofs << ",d" << d;
        }
    }
    ofs << "\n";

    if (D > 0 && desc_cpu.defined() && desc_cpu.dim() == 2 && static_cast<size_t>(desc_cpu.size(0)) == output.keypoints.size()) {
        const float* data_ptr = desc_cpu.data_ptr<float>();
        const int rows = static_cast<int>(desc_cpu.size(0));
        const int cols = D;

        // 诊断：计算非零比例与均值，便于排查全零问题
        double abs_sum = 0.0;
        size_t nonzero = 0;
        size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        for (size_t i = 0; i < static_cast<size_t>(rows); ++i) {
            const float* row_ptr = data_ptr + i * static_cast<size_t>(cols);
            for (int d = 0; d < cols; ++d) {
                float v = row_ptr[d];
                abs_sum += std::abs(static_cast<double>(v));
                if (std::abs(v) > 1e-9f) ++nonzero;
            }
        }
        double mean_abs = total > 0 ? (abs_sum / static_cast<double>(total)) : 0.0;
        std::cerr << "CSV descriptors stats: rows=" << rows << " cols=" << cols << " nonzero=" << nonzero << " total=" << total << " mean_abs=" << mean_abs << std::endl;
        if (nonzero == 0) {
            std::cerr << "WARN: CSV descriptors appear all zeros for file: " << path << std::endl;
        }

        for (int i = 0; i < rows; ++i) {
            ofs << std::fixed << std::setprecision(6)
                << output.keypoints[static_cast<size_t>(i)].pt.x << ","
                << output.keypoints[static_cast<size_t>(i)].pt.y << ","
                << output.scores[static_cast<size_t>(i)];
            const float* row_ptr = data_ptr + static_cast<size_t>(i) * static_cast<size_t>(cols);
            for (int d = 0; d < cols; ++d) {
                ofs << "," << std::setprecision(6) << row_ptr[d];
            }
            ofs << "\n";
        }
    } else {
        for (size_t i = 0; i < output.keypoints.size(); ++i) {
            ofs << std::fixed << std::setprecision(6)
                << output.keypoints[i].pt.x << ","
                << output.keypoints[i].pt.y << ","
                << output.scores[i] << "\n";
        }
    }
    ofs.close();
    return true;
}

bool SuperPoint::saveOverlayImage(const cv::Mat& image, const SuperPointOutput& output, const std::string& path) 
{
    if (image.empty()) return false;
    cv::Mat vis;
    if (image.channels() == 1) cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
    else vis = image.clone();
    cv::drawKeypoints(vis, output.keypoints, vis, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return cv::imwrite(path, vis);
}

// Note: File-based saving/reading is intentionally handled outside this class.
// Use a separate Qt-based I/O helper (e.g. FeatureFileIO) in the GUI layer to write/read binary files via QFile/QDataStream.

// -------------------------------------------------------------------------
// detectBatch: 批量推理，提高 GPU 容量利用率
// 优化策略:
//   1. 按图像尺寸分组，尺寸一致的图像才能堆叠为 batch
//   2. 并行预处理（std::async）并在 CPU 上合并小 tensor
//   3. 通过 pin_memory + 非阵塞传输到 GPU，减少 PCIe 带宽干扰
//   4. 他得模型输出列表，逐张解析和采样描述子
//   5. 无法识别的输出格式则回退到逐张 detect()
std::vector<SuperPointOutput> SuperPoint::detectBatch(const std::vector<cv::Mat>& images) 
{
    std::vector<SuperPointOutput> results;
    results.resize(images.size());

    // 如果禁用批处理或 batch_size <= 1，退回到逐张调用
    if (config_.batch_size <= 1) 
    {
        for (size_t i = 0; i < images.size(); ++i) 
        {
            results[i] = detect(images[i]);
            if (config_.save_keypoints_csv) 
            {
                std::string p = "keypoints_" + std::to_string(i) + ".csv";
                saveKeypointsCSV(results[i], p);
            }
            if (config_.save_overlay_image) 
            {
                std::string p = "overlay_" + std::to_string(i) + ".png";
                saveOverlayImage(images[i], results[i], p);
            }
        }
        return results;
    }

    // 按宽高分组：只有尺寸相同的图像才能堆叠为 batch
    std::map<std::pair<int,int>, std::vector<size_t>> groups;
    for (size_t i = 0; i < images.size(); ++i) 
    {
        groups[{images[i].cols, images[i].rows}].push_back(i);
    }

    for (auto &g : groups) 
    {
        const auto& idxs = g.second;
        size_t start = 0;
        while (start < idxs.size()) 
        {
            size_t end = std::min(idxs.size(), start + static_cast<size_t>(config_.batch_size));
            // 预处理并堆叠 tensor
            std::vector<torch::Tensor> batch_tensors;
            batch_tensors.reserve(end - start);
            // 并行预处理：使用 std::async 并行调用 preprocessImageCPU（不写 last_gray）
            std::vector<std::future<torch::Tensor>> futures;
            futures.reserve(end - start);
            for (size_t k = start; k < end; ++k) 
            {
                // 将 cv::Mat 以值传递 (浅拷贝)，并在子线程中生成 CPU tensor
                futures.emplace_back(std::async(std::launch::async,
                    &SuperPoint::preprocessImageCPU, this, images[idxs[k]], false));
            }
            for (auto &fut : futures) 
            {
                batch_tensors.push_back(fut.get());
            }
            // 提取每张图像对应的灰度图（CPU, [H,W]）用于后续过滤
            std::vector<torch::Tensor> batch_grays;
            batch_grays.reserve(batch_tensors.size());
            for (auto &t : batch_tensors) {
                // t shape: [1,1,H,W]
                auto g = t.squeeze(0).squeeze(0).to(torch::kCPU);
                batch_grays.push_back(g);
            }
            // 在 CPU 上拼接小 tensor，然后通过 pin_memory + 非阵塞传到设备
            // pin_memory 使 DMA 传输更快，non_blocking=true 与后续计算交叠
            auto batch_cpu = torch::cat(batch_tensors, 0); // [B,1,H,W] on CPU
            auto batch_pinned = batch_cpu.pin_memory();
            auto batch_input = batch_pinned.to(config_.device, /*non_blocking=*/true);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(batch_input);
            // 在批量推理中禁用梯度以减少显存占用
            torch::NoGradGuard no_grad;

            torch::jit::IValue out;
            try {
                out = model_.forward(inputs);
            } catch (const c10::OutOfMemoryError &) {
                // 显存不足：清空缓存后重试；若仍失败则回退到逐张处理
                if (config_.device.is_cuda()) {
                    torch::cuda::synchronize();
                    c10::cuda::CUDACachingAllocator::emptyCache();
                }
                try {
                    out = model_.forward(inputs);
                } catch (const c10::Error &retry_err) {
                    std::cerr << "[SuperPoint] 批量推理 OOM 重试失败，回退逐张处理: "
                              << retry_err.what_without_backtrace() << std::endl;
                    for (size_t k = start; k < end; ++k) {
                        results[idxs[k]] = detect(images[idxs[k]]);
                    }
                    continue;
                }
            } catch (const c10::Error &e) {
                std::cerr << "[SuperPoint] 批量推理异常，回退逐张处理: "
                          << e.what_without_backtrace() << std::endl;
                for (size_t k = start; k < end; ++k) {
                    results[idxs[k]] = detect(images[idxs[k]]);
                }
                continue;
            }

            if (out.isTuple())
            {
                auto tup = out.toTuple();
                auto e0 = tup->elements()[0];
                auto e1 = tup->elements()[1];
                auto e2 = tup->elements()[2];
                if (e0.isList()) 
                {
                    auto list0 = e0.toList();
                    auto list1 = e1.toList();
                    auto list2 = e2.toList();
                    for (size_t k = start; k < end; ++k) 
                    {
                        size_t out_idx = k - start;
                        auto kp = list0.get(out_idx).toTensor();
                        auto sc = list1.get(out_idx).toTensor();
                        auto desc = list2.get(out_idx).toTensor();
                        // 诊断：输出描述子形状/范围
                        if (desc.defined()) {
                            try {
                                auto tmp = desc.to(torch::kCPU);
                                std::string info = "model batch desc shape:";
                                for (auto s : tmp.sizes()) { info += " "; info += std::to_string(s); }
                                std::cerr << info << std::endl;
                                // 计算少量统计量（若为2D或3D）
                                if (tmp.numel() > 0 && tmp.dim() <= 4) {
                                    auto tfloat = tmp.to(torch::kFloat32).contiguous();
                                    const float* dp = tfloat.data_ptr<float>();
                                    int64_t ne = tfloat.numel();
                                    double ssum = 0.0; int64_t nz = 0; float mn = 0.0f, mx = 0.0f;
                                    int64_t lim = std::min<int64_t>(ne, 1000);
                                    for (int64_t ii = 0; ii < lim; ++ii) {
                                        float v = dp[ii]; ssum += v; if (v != 0.0f) ++nz; if (ii==0) { mn = mx = v; } else { if (v < mn) mn = v; if (v > mx) mx = v; }
                                    }
                                    double total_sum = 0.0;
                                    try { total_sum = tfloat.sum().item<double>(); } catch(...) {}
                                    std::cerr << "model desc sample stats: ne=" << ne << " sample_nz=" << nz << " mean_sample=" << (lim? ssum / lim : 0.0) << " min_sample=" << mn << " max_sample=" << mx << " sum=" << total_sum << std::endl;
                                    int show = static_cast<int>(std::min<int64_t>(ne, 20));
                                    std::cerr << "model desc first_values:";
                                    for (int i=0;i<show;i++) std::cerr << " " << dp[i];
                                    std::cerr << std::endl;
                                }
                            } catch(...) { std::cerr << "warn: failed to inspect model desc" << std::endl; }
                        }
                        SuperPointOutput sop;
                        if (kp.numel() > 0) 
                        {
                            auto kp_cpu = kp.to(torch::kCPU);
                            auto sc_cpu = sc.to(torch::kCPU);
                                auto kp_acc = kp_cpu.accessor<float,2>();
                                auto sc_acc = sc_cpu.accessor<float,1>();
                                // 对应的灰度图（如果可用）
                                bool has_gray = false;
                                torch::Tensor gray = torch::Tensor();
                                if (out_idx < batch_grays.size()) {
                                    gray = batch_grays[out_idx];
                                    if (gray.defined() && gray.dim() == 2) has_gray = true;
                                }
                                for (int ii = 0; ii < kp_cpu.size(0); ++ii) 
                                {
                                    float kx = kp_acc[ii][0];
                                    float ky = kp_acc[ii][1];
                                    if (has_gray) {
                                        int gray_h = static_cast<int>(gray.size(0));
                                        int gray_w = static_cast<int>(gray.size(1));
                                        int xi = static_cast<int>(std::round(kx));
                                        int yi = static_cast<int>(std::round(ky));
                                        xi = std::max(0, std::min(xi, gray_w - 1));
                                        yi = std::max(0, std::min(yi, gray_h - 1));
                                        auto gray_acc = gray.accessor<float,2>();
                                        float intensity = gray_acc[yi][xi];
                                        if (intensity < config_.grayscale_min || intensity > config_.grayscale_max) {
                                            continue;
                                        }
                                    }
                                    cv::KeyPoint kpt;
                                    kpt.pt.x = kx;
                                    kpt.pt.y = ky;
                                    kpt.response = sc_acc[ii];
                                    kpt.size = 8.0f;
                                    sop.keypoints.push_back(kpt);
                                    sop.scores.push_back(sc_acc[ii]);
                                }
                            // 处理 descriptors：模型在不同导出方式下可能返回
                            //  - 每点描述子 [N, D]
                            //  - 密集描述子 [C, H, W] 或 [H, W, C]
                            // 若为密集描述子，则对关键点做 grid-sample 采样以得到每点描述子
                            if (desc.defined()) {
                                auto desc_t = desc;
                                // 将 desc 迁移到 CPU 并转换为 float
                                desc_t = desc_t.to(torch::kCPU);
                                if (desc_t.dtype() != torch::kFloat32) desc_t = desc_t.to(torch::kFloat32);

                                const int num_kp = static_cast<int>(sop.keypoints.size());
                                // 情形1: 已是按点描述子 [N, D]
                                if (desc_t.dim() == 2 && desc_t.size(0) == num_kp) {
                                    sop.descriptors = desc_t.contiguous();
                                }
                                // 情形2: 密集描述子 [C, H, W]
                                else if (desc_t.dim() == 3) {
                                    torch::Tensor desc_chw = desc_t;
                                    // 如果形状为 [H, W, C]，则转为 [C, H, W]
                                    if (desc_chw.size(2) == config_.descriptor_dim && desc_chw.size(0) != config_.descriptor_dim) {
                                        desc_chw = desc_chw.permute({2, 0, 1});
                                    }

                                    if (desc_chw.size(0) == config_.descriptor_dim) {
                                        // 构建 keypoints tensor [1, N, 2]
                                        std::vector<float> kps;
                                        kps.reserve(static_cast<size_t>(num_kp) * 2);
                                        for (const auto &kp : sop.keypoints) {
                                            kps.push_back(static_cast<float>(kp.pt.x));
                                            kps.push_back(static_cast<float>(kp.pt.y));
                                        }
                                        if (!kps.empty()) {
                                            auto kpt_tensor = torch::from_blob(kps.data(), {num_kp, 2}, torch::kFloat32).clone();
                                            kpt_tensor = kpt_tensor.unsqueeze(0); // [1,N,2]
                                            // sampleDescriptors 期望 descriptors 为 [B,C,H,W]
                                            auto desc_bchw = desc_chw.unsqueeze(0);
                                            auto sampled = sampleDescriptors(kpt_tensor, desc_bchw, config_.grid_size);
                                            sampled = sampled.squeeze(0).transpose(0, 1); // [N, D]
                                            sop.descriptors = sampled.contiguous();
                                        } else {
                                            sop.descriptors = torch::Tensor();
                                        }
                                    } else {
                                        // 无法识别的描述子形状，保空
                                        sop.descriptors = torch::Tensor();
                                    }
                                } else {
                                    sop.descriptors = torch::Tensor();
                                }
                            } else {
                                sop.descriptors = torch::Tensor();
                            }
                        }
                        size_t global_idx = idxs[k];
                        results[global_idx] = sop;
                        if (config_.save_keypoints_csv) 
                        {
                            std::string p = "keypoints_" + std::to_string(global_idx) + ".csv";
                            saveKeypointsCSV(results[global_idx], p);
                        }
                        if (config_.save_overlay_image) 
                        {
                            std::string p = "overlay_" + std::to_string(global_idx) + ".png";
                            saveOverlayImage(images[global_idx], results[global_idx], p);
                        }
                    }
                } 
                else 
                {
                    // 无法识别的批量输出格式，回退到逐张处理
                    for (size_t k = start; k < end; ++k) 
                    {
                        results[idxs[k]] = detect(images[idxs[k]]);
                        if (config_.save_keypoints_csv) 
                        {
                            std::string p = "keypoints_" + std::to_string(idxs[k]) + ".csv";
                            saveKeypointsCSV(results[idxs[k]], p);
                        }
                        if (config_.save_overlay_image) 
                        {
                            std::string p = "overlay_" + std::to_string(idxs[k]) + ".png";
                            saveOverlayImage(images[idxs[k]], results[idxs[k]], p);
                        }
                    }
                }
            } 
            else 
            {
                // 非 tuple 返回，退回逐张处理
                for (size_t k = start; k < end; ++k) 
                {
                    results[idxs[k]] = detect(images[idxs[k]]);
                    if (config_.save_keypoints_csv) 
                    {
                        std::string p = "keypoints_" + std::to_string(idxs[k]) + ".csv";
                        saveKeypointsCSV(results[idxs[k]], p);
                    }
                    if (config_.save_overlay_image) 
                    {
                        std::string p = "overlay_" + std::to_string(idxs[k]) + ".png";
                        saveOverlayImage(images[idxs[k]], results[idxs[k]], p);
                    }
                }
            }

            start = end;
        }
    }

    return results;
}
