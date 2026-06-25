#include "SuperPoint.h"
#include "SuperPointUtils.h"
#include <iostream>
#include <future>
#include <algorithm>
#include <cmath>
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

// -------------------------------------------------------------------------
// detectBatch: 批量推理，提高 GPU 容量利用率
// 优化策略:
//   1. 按图像尺寸分组，尺寸一致的图像才能堆叠为 batch
//   2. 并行预处理（std::async）并在 CPU 上合并小 tensor
//   3. 通过 pin_memory + 非阵塞传输到 GPU，减少 PCIe 带宽干扰
//   4. 他得模型输出列表，逐张解析和采样描述子
//   5. 无法识别的输出格式则回退到逐张 detect()
std::vector<FeatureOutput> SuperPoint::detectBatch(const std::vector<cv::Mat>& images) 
{
    std::vector<FeatureOutput> results;
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
            // 在 CPU 上拼接小 tensor。只有 CUDA 路径才使用 pinned memory；
            // CPU-only LibTorch 没有 accelerator，调用 pin_memory() 会直接抛异常。
            auto batch_cpu = torch::cat(batch_tensors, 0); // [B,1,H,W] on CPU
            torch::Tensor batch_input;
            if (config_.device.is_cuda())
            {
                auto batch_pinned = batch_cpu.pin_memory();
                batch_input = batch_pinned.to(config_.device, /*non_blocking=*/true);
            }
            else
            {
                batch_input = batch_cpu.to(config_.device);
            }

            // 批量 orig_wh: [B,2] 每行 [W,H]
            int bw = g.first.first, bh = g.first.second;
            auto wh_tpl = torch::tensor({static_cast<float>(bw), static_cast<float>(bh)},
                torch::TensorOptions().dtype(torch::kFloat32).device(config_.device));
            auto batch_wh = wh_tpl.unsqueeze(0).repeat({static_cast<long>(end - start), 1});
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(batch_input);
            inputs.push_back(batch_wh);
            // 在批量推理中禁用梯度以减少显存占用
            torch::NoGradGuard no_grad;

            torch::jit::IValue out;
            try {
                out = model_.forward(inputs);
            } catch (const c10::OutOfMemoryError &) {
                // 显存不足：清空缓存后重试；若仍失败则回退到逐张处理
                if (config_.device.is_cuda()) {
                    torch::cuda::synchronize();
            #ifdef USE_CUDA
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
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
                                for (auto s : tmp.sizes())
                                {
                                    info += " ";
                                    info += std::to_string(s);
                                }
                                std::cerr << info << std::endl;
                                // 计算少量统计量（若为2D或3D）
                                if (tmp.numel() > 0 && tmp.dim() <= 4)
                                {
                                    auto tfloat = tmp.to(torch::kFloat32).contiguous();
                                    const float* dp = tfloat.data_ptr<float>();
                                    int64_t ne = tfloat.numel();
                                    double ssum = 0.0;
                                    int64_t nz = 0;
                                    float mn = 0.0f;
                                    float mx = 0.0f;
                                    int64_t lim = std::min<int64_t>(ne, 1000);
                                    for (int64_t ii = 0; ii < lim; ++ii)
                                    {
                                        const float v = dp[ii];
                                        ssum += v;
                                        if (v != 0.0f)
                                        {
                                            ++nz;
                                        }
                                        if (ii == 0)
                                        {
                                            mn = mx = v;
                                        }
                                        else
                                        {
                                            if (v < mn)
                                            {
                                                mn = v;
                                            }
                                            if (v > mx)
                                            {
                                                mx = v;
                                            }
                                        }
                                    }
                                    double total_sum = 0.0;
                                    try
                                    {
                                        total_sum = tfloat.sum().item<double>();
                                    }
                                    catch (...)
                                    {
                                    }
                                    std::cerr << "model desc sample stats: ne=" << ne
                                              << " sample_nz=" << nz
                                              << " mean_sample=" << (lim ? ssum / lim : 0.0)
                                              << " min_sample=" << mn
                                              << " max_sample=" << mx
                                              << " sum=" << total_sum << std::endl;
                                    int show = static_cast<int>(std::min<int64_t>(ne, 20));
                                    std::cerr << "model desc first_values:";
                                    for (int i = 0; i < show; i++)
                                    {
                                        std::cerr << " " << dp[i];
                                    }
                                    std::cerr << std::endl;
                                }
                            }
                            catch (...)
                            {
                                std::cerr << "warn: failed to inspect model desc" << std::endl;
                            }
                        }
                        FeatureOutput sop;
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
                                if (desc_t.dtype() != torch::kFloat32)
                                {
                                    desc_t = desc_t.to(torch::kFloat32);
                                }

                                const int num_kp = static_cast<int>(sop.keypoints.size());
                                // 情形1: 已是按点描述子 [N, D]
                                if (desc_t.dim() == 2 && desc_t.size(0) == num_kp)
                                {
                                    sop.descriptors = desc_t.contiguous();
                                }
                                // 情形2: 密集描述子 [C, H, W]
                                else if (desc_t.dim() == 3)
                                {
                                    torch::Tensor desc_chw = desc_t;
                                    // 如果形状为 [H, W, C]，则转为 [C, H, W]
                                    if (desc_chw.size(2) == config_.descriptor_dim &&
                                        desc_chw.size(0) != config_.descriptor_dim)
                                    {
                                        desc_chw = desc_chw.permute({2, 0, 1});
                                    }

                                    if (desc_chw.size(0) == config_.descriptor_dim)
                                    {
                                        // 构建 keypoints tensor [1, N, 2]
                                        std::vector<float> kps;
                                        kps.reserve(static_cast<size_t>(num_kp) * 2);
                                        for (const auto &kp : sop.keypoints)
                                        {
                                            kps.push_back(static_cast<float>(kp.pt.x));
                                            kps.push_back(static_cast<float>(kp.pt.y));
                                        }
                                        if (!kps.empty())
                                        {
                                            auto kpt_tensor =
                                                torch::from_blob(kps.data(), {num_kp, 2}, torch::kFloat32).clone();
                                            kpt_tensor = kpt_tensor.unsqueeze(0); // [1,N,2]
                                            // sampleDescriptors 期望 descriptors 为 [B,C,H,W]
                                            auto desc_bchw = desc_chw.unsqueeze(0);
                                            auto sampled = sampleDescriptors(kpt_tensor, desc_bchw, config_.grid_size);
                                            sampled = sampled.squeeze(0).transpose(0, 1); // [N, D]
                                            sop.descriptors = sampled.contiguous();
                                        }
                                        else
                                        {
                                            sop.descriptors = torch::Tensor();
                                        }
                                    }
                                    else
                                    {
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
