#include "SuperPoint.h"
#include "SuperPointUtils.h"
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>
#include <iomanip>
#include <future>
#include <algorithm>
#include <cmath>
#include <numeric>
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

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
                
                auto dummy_wh = torch::tensor({640.0f, 480.0f},
                    torch::TensorOptions().dtype(torch::kFloat32).device(config_.device));
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(dummy_input);
                inputs.push_back(dummy_wh);

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

// Note: File-based saving/reading is intentionally handled outside this class.
// Use a separate Qt-based I/O helper (e.g. FeatureFileIO) in the GUI layer to write/read binary files via QFile/QDataStream.

// 表头格式: x,y,score[,d0,...,d{D-1}]
// 描述子派算中会输出说明数据的评断信息
// （为 0 比例较高时会打印警告，帮助排查描述子异常）
bool SuperPoint::saveKeypointsCSV(const FeatureOutput& output, const std::string& path) {
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

bool SuperPoint::saveOverlayImage(const cv::Mat& image, const FeatureOutput& output, const std::string& path) 
{
    if (image.empty()) return false;
    cv::Mat vis;
    if (image.channels() == 1) cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
    else vis = image.clone();
    cv::drawKeypoints(vis, output.keypoints, vis, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return cv::imwrite(path, vis);
}
