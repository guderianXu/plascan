#include "SuperPoint.h"
#include "SuperPointUtils.h"
#include <iostream>
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

// -------------------------------------------------------------------------
// detect: 对单张图像进行完整的 SuperPoint 推理
// 步骤:
//   1. 预处理图像（灰度化+归一化+张量化）并更新 last_gray_cpu_
//   2. 将张量移动到目标设备（CPU 或 CUDA）
//   3. 做前向推理并解析输出（支持 tuple 和 dict 两种模型格式）
//   4. 对模型咆出的局郠关键点做子像素化、灰度迧滤、边界剔除
//   5. 邻域黑边检查、稀疏 NMS、top-k 截断
//   6. 从 dense descriptors 中为每个关键点采样调用一维描述子
FeatureOutput SuperPoint::detect(const cv::Mat& image) 
{
    torch::NoGradGuard no_grad;  // 禁用梯度计算，减少显存占用
    
    // 预处理：将 cv::Mat 转为归一化浮点 tensor [1,1,H,W]，同时更新 last_gray_cpu_
    auto input_tensor = preprocessImage(image);
    // 将 CPU tensor 移动到目标推理设备（non_blocking=false 以确保同步）
    auto input_device = input_tensor.to(config_.device, /*non_blocking=*/false);

    // 前向推理
    // 新模型wrapper返回: (keypoints [N,2], scores [N], dense_descriptors [1,256,H/8,W/8])
    auto orig_wh = torch::tensor({static_cast<float>(image.cols), static_cast<float>(image.rows)},
        torch::TensorOptions().dtype(torch::kFloat32).device(config_.device));
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_device);
    inputs.push_back(orig_wh);
    
    // 前向推理：首次 OOM 清空缓存后重试一次；两次都失败返回空结果
    torch::jit::IValue output;
    try {
        output = model_.forward(inputs);
    } catch (const c10::OutOfMemoryError &) {
        if (config_.device.is_cuda()) {
            torch::cuda::synchronize();
    #ifdef USE_CUDA
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
        }
        try {
            output = model_.forward(inputs);
        } catch (const c10::Error &retry_err) {
            std::cerr << "[SuperPoint] 推理 OOM 重试失败，返回空结果: "
                      << retry_err.what_without_backtrace() << std::endl;
            return FeatureOutput{};
        }
    } catch (const c10::Error &e) {
        std::cerr << "[SuperPoint] 推理异常，返回空结果: "
                  << e.what_without_backtrace() << std::endl;
        return FeatureOutput{};
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

        // 真实模型返回带 batch 维 [1,N,...], squeeze 掉 batch dim
        if (keypoints.dim() == 3) keypoints = keypoints.squeeze(0);
        if (scores.dim() == 2) scores = scores.squeeze(0);
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
    FeatureOutput result;
    result.imageWidth = image.cols;
    result.imageHeight = image.rows;

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
#ifdef USE_CUDA
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    }

    return result;
}

// -------------------------------------------------------------------------
// saveKeypointsCSV: 将关键点坐标、分数和描述子导出为 CSV 文件
// 表头格式: x,y,score[,d0,...,d{D-1}]
// 描述子派算中会输出说明数据的评断信息
// （为 0 比例较高时会打印警告，帮助排查描述子异常）
