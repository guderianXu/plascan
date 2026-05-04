#include "SuperPoint.h"
#include "SuperPointUtils.h"
#include <iostream>
#include <cmath>
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

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
FeatureOutput SuperPoint::postprocess(const torch::Tensor& scores,
                                         const torch::Tensor& descriptors_dense,
                                         int img_width, int img_height) 
{
    FeatureOutput output;
    
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
