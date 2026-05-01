// =============================================================================
// 文件: SuperGlueMatcher.cpp
// 说明:
//   SuperGlue GNN 匹配器引擎的实现。
//   输入: 两张图像的关键点坐标、评分、描述子 (dict 格式)
//   输出: 双向匹配索引 matches0/matches1 和置信度分数
//
//   注意事项:
//     - SuperPoint 输出的 descriptors 形状为 [N, D]，但 SuperGlue 要求 [1, D, N]
//       因此在构建输入字典前需要转置调用方
//     - 模型输出为字典（matches0/1, matching_scores0/1）
// =============================================================================
#include "SuperGlueMatcher.h"
#include "../../feature_extractors/FeatureData.h"

#include "log/Logger.h"
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <iomanip>
#include <ctime>
#include <fstream>

namespace superglue {

SuperGlueMatcher::SuperGlueMatcher(const SuperGlueConfig& config)
    : config_(config)
    , device_(torch::kCPU)
    , model_loaded_(false)
    , processed_count_(0) {
    
    initModel();
}

SuperGlueMatcher::SuperGlueMatcher(const std::string& model_path, bool use_cuda)
    : SuperGlueMatcher(SuperGlueConfig(model_path)) {
    config_.use_cuda = use_cuda;
    initModel();
}

SuperGlueMatcher::~SuperGlueMatcher() {
    closeCSVFile();
}

MatchResult SuperGlueMatcher::match(const xjw::feature_extractors::FeatureData &feat0,
                                    const xjw::feature_extractors::FeatureData &feat1)
{
    if (!model_loaded_) {
        throw std::runtime_error("模型未加载");
    }

    if (feat0.empty() || feat1.empty()) {
        log("警告: FeatureData 为空");
        return MatchResult();
    }

    auto buildScores = [](const xjw::feature_extractors::FeatureData &feat) {
        std::vector<float> scores = feat.scores;
        if (scores.size() != feat.keypoints.size()) {
            scores.assign(feat.keypoints.size(), 1.0f);
            for (size_t i = 0; i < feat.keypoints.size(); ++i) {
                scores[i] = feat.keypoints[i].response;
            }
        }
        return scores;
    };

    auto inferShape = [](const xjw::feature_extractors::FeatureData &feat) {
        int width = feat.imageWidth;
        int height = feat.imageHeight;
        if (width <= 0 || height <= 0) {
            float maxX = 1.0f;
            float maxY = 1.0f;
            for (const auto &kp : feat.keypoints) {
                maxX = std::max(maxX, kp.pt.x);
                maxY = std::max(maxY, kp.pt.y);
            }
            width = std::max(1, static_cast<int>(maxX) + 1);
            height = std::max(1, static_cast<int>(maxY) + 1);
        }
        return std::vector<int>{1, 1, height, width};
    };

    torch::Tensor desc0 = feat0.toTensorSuperGlue();
    torch::Tensor desc1 = feat1.toTensorSuperGlue();
    if (!desc0.defined() || !desc1.defined() || desc0.numel() == 0 || desc1.numel() == 0) {
        log("警告: 描述子为空");
        return MatchResult();
    }

    KeypointData kpts0(feat0.keypoints, buildScores(feat0), desc0);
    KeypointData kpts1(feat1.keypoints, buildScores(feat1), desc1);
    return match(kpts0, kpts1, inferShape(feat0), inferShape(feat1));
}

// initModel: 加载 TorchScript 模型并设置运行设备
void SuperGlueMatcher::initModel() {
    try {
        // 优先尝试 CUDA，失败或不可用时回退到 CPU
        if (config_.use_cuda && torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, config_.cuda_device_id);
            log("使用 CUDA 设备 " + std::to_string(config_.cuda_device_id));
        } else {
            device_ = torch::Device(torch::kCPU);
            log("使用 CPU 设备");
            
            // 在 CPU 模式下按配置设置推理线程数
            if (config_.num_threads > 0) {
                torch::set_num_threads(config_.num_threads);
                log("设置 CPU 线程数: " + std::to_string(config_.num_threads));
            }
        }
        
        // 加载模型
        if (config_.model_path.empty()) {
            throw std::runtime_error("模型路径为空");
        }
        
        log("加载模型: " + config_.model_path);
        module_ = torch::jit::load(config_.model_path, device_);
        module_.eval();
        
        model_loaded_ = true;
        log("模型加载成功");
        
        // 如果启用CSV输出，打开文件
        if (config_.enable_csv_output) {
            openCSVFile();
        }
        
    } catch (const c10::Error& e) {
        LOG_ERROR("加载模型失败: %s", e.what());
        throw std::runtime_error("无法加载 SuperGlue 模型");
    }
}

// keypointsToTensor: 将 cv::KeyPoint 列表转为 [1, N, 2] float32 Tensor 并上指定设备
// 格式: [batch=1, num_kps, {x, y}]
torch::Tensor SuperGlueMatcher::keypointsToTensor(const std::vector<cv::KeyPoint>& keypoints) {
    int num_kpts = keypoints.size();
    torch::Tensor kpts_tensor = torch::zeros({1, num_kpts, 2}, torch::kFloat32);
    
    auto accessor = kpts_tensor.accessor<float, 3>();
    for (int i = 0; i < num_kpts; ++i) {
        accessor[0][i][0] = keypoints[i].pt.x;
        accessor[0][i][1] = keypoints[i].pt.y;
    }
    
    return kpts_tensor.to(device_);
}

// scoresToTensor: 将关键点分数列表转为 [1, N] float32 Tensor 并上指定设备
torch::Tensor SuperGlueMatcher::scoresToTensor(const std::vector<float>& scores) {
    int num_scores = scores.size();
    torch::Tensor scores_tensor = torch::zeros({1, num_scores}, torch::kFloat32);
    
    auto accessor = scores_tensor.accessor<float, 2>();
    for (int i = 0; i < num_scores; ++i) {
        accessor[0][i] = scores[i];
    }
    
    return scores_tensor.to(device_);
}

torch::Tensor SuperGlueMatcher::createDummyImage(const std::vector<int>& shape) {
    if (shape.size() != 4) {
        throw std::runtime_error("Image shape must be [1, 1, H, W]");
    }
    return torch::zeros({shape[0], shape[1], shape[2], shape[3]}, 
                       torch::TensorOptions().dtype(torch::kFloat32).device(device_));
}

// prepareInput: 构建 SuperGlue 所需的输入字典
// SuperGlue TorchScript 接受一个 GenericDict，键名固定为:
//   keypoints0/1, scores0/1, descriptors0/1, image0/1
// image0/1 仅用于提供 H、W 形状信息，值为全零虚拟张量
// descriptors0/1 形状: [1, D, N]（注意不是 [N, D]，调用前需转置）
torch::Dict<std::string, torch::Tensor> SuperGlueMatcher::prepareInput(
    const KeypointData& keypoints0,
    const KeypointData& keypoints1,
    const std::vector<int>& image0_shape,
    const std::vector<int>& image1_shape) {
    
    torch::Dict<std::string, torch::Tensor> input_dict;
    
    // 转换关键点和得分
    torch::Tensor kpts0 = keypointsToTensor(keypoints0.keypoints);
    torch::Tensor kpts1 = keypointsToTensor(keypoints1.keypoints);
    torch::Tensor scores0 = scoresToTensor(keypoints0.scores);
    torch::Tensor scores1 = scoresToTensor(keypoints1.scores);
    
    // 描述符已经是tensor，只需要转移到正确的设备
    torch::Tensor desc0 = keypoints0.descriptors.to(device_);
    torch::Tensor desc1 = keypoints1.descriptors.to(device_);
    
    // 创建虚拟图像（仅用于提供形状信息）
    torch::Tensor img0 = createDummyImage(image0_shape);
    torch::Tensor img1 = createDummyImage(image1_shape);
    
    // 填充输入字典
    input_dict.insert("keypoints0", kpts0);
    input_dict.insert("keypoints1", kpts1);
    input_dict.insert("scores0", scores0);
    input_dict.insert("scores1", scores1);
    input_dict.insert("descriptors0", desc0);
    input_dict.insert("descriptors1", desc1);
    input_dict.insert("image0", img0);
    input_dict.insert("image1", img1);
    
    return input_dict;
}

// parseOutput: 解析 SuperGlue 的 GenericDict 输出为 MatchResult
// matches0[i]: 图像0中第 i 个关键点匹配到图像1中的索引（-1 表示未匹配，dustbin）
// matching_scores0[i]: 对应的置信度分数 [0,1]
// 同时构建 cv::DMatch 列表（仅含有效匹配且分数 > match_threshold 的对）
MatchResult SuperGlueMatcher::parseOutput(const torch::Dict<std::string, torch::Tensor>& output) {
    MatchResult result;

    // 检查必需的键是否存在
    if (!output.contains("matches0") || !output.contains("matches1") ||
        !output.contains("matching_scores0") || !output.contains("matching_scores1")) {
        LOG_ERROR("SuperGlue 模型输出缺少必需的键");
        return result;
    }

    // 将所有输出张量迁移到 CPU 再访问
    torch::Tensor matches0 = output.at("matches0").cpu();
    torch::Tensor matches1 = output.at("matches1").cpu();
    torch::Tensor mscores0 = output.at("matching_scores0").cpu();
    torch::Tensor mscores1 = output.at("matching_scores1").cpu();
    
    // 使用 accessor 高效访问 Tensor 数据（避免逐元素 .item() 调用）
    auto matches0_acc = matches0.accessor<int64_t, 2>();
    auto matches1_acc = matches1.accessor<int64_t, 2>();
    auto mscores0_acc = mscores0.accessor<float, 2>();
    auto mscores1_acc = mscores1.accessor<float, 2>();
    
    int num_kpts0 = matches0.size(1);
    int num_kpts1 = matches1.size(1);
    
    result.matches0.resize(num_kpts0);
    result.matches1.resize(num_kpts1);
    result.matchingScores0.resize(num_kpts0);
    result.matchingScores1.resize(num_kpts1);

    result.sourceAlgorithm = "superglue";
    result.numMatches = 0;
    
    for (int i = 0; i < num_kpts0; ++i) {
        result.matches0[i] = static_cast<int>(matches0_acc[0][i]);
        result.matchingScores0[i] = mscores0_acc[0][i];
        
        // 只有 match_idx >= 0 且分数超过阈值才认定为有效匹配
        if (result.matches0[i] >= 0 && result.matchingScores0[i] > config_.match_threshold) {
            // DMatch.distance 转换: 分数 [0,1] -> 距离 [1,0]（分数越高距离越小）
            cv::DMatch dmatch;
            dmatch.queryIdx = i;
            dmatch.trainIdx = result.matches0[i];
            dmatch.distance = 1.0f - result.matchingScores0[i];
            result.cvMatches.push_back(dmatch);
            result.numMatches++;
        }
    }
    
    for (int i = 0; i < num_kpts1; ++i) {
        result.matches1[i] = static_cast<int>(matches1_acc[0][i]);
        result.matchingScores1[i] = mscores1_acc[0][i];
    }
    
    return result;
}

// limitKeypoints: 若关键点超过 max_num 则按分数降序保留前 max_num 个
// 同时截取对应的 descriptors 子集（descriptors 形状 [1, D, N]）
KeypointData SuperGlueMatcher::limitKeypoints(const KeypointData& kpts, int max_num) {
    if (max_num <= 0 || static_cast<int>(kpts.keypoints.size()) <= max_num) {
        return kpts;
    }
    
    // 按得分排序，保留前max_num个
    std::vector<std::pair<float, int>> score_indices;
    for (size_t i = 0; i < kpts.scores.size(); ++i) {
        score_indices.push_back({kpts.scores[i], i});
    }
    
    std::sort(score_indices.begin(), score_indices.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    KeypointData limited_kpts;
    limited_kpts.keypoints.reserve(max_num);
    limited_kpts.scores.reserve(max_num);
    
    std::vector<int> selected_indices;
    for (int i = 0; i < max_num; ++i) {
        int idx = score_indices[i].second;
        selected_indices.push_back(idx);
        limited_kpts.keypoints.push_back(kpts.keypoints[idx]);
        limited_kpts.scores.push_back(kpts.scores[idx]);
    }
    
    // 提取对应的描述符
    // descriptors shape: [1, D, N]
    int desc_dim = kpts.descriptors.size(1);
    limited_kpts.descriptors = torch::zeros({1, desc_dim, max_num}, kpts.descriptors.options());
    
    for (int i = 0; i < max_num; ++i) {
        int idx = selected_indices[i];
        limited_kpts.descriptors.index({0, torch::indexing::Slice(), i}) = 
            kpts.descriptors.index({0, torch::indexing::Slice(), idx});
    }
    
    return limited_kpts;
}

// match (核心接口): 执行 SuperGlue 一次双向匹配推理
// 步骤:
//   1. 可选地限制关键点上限（按分数截断）
//   2. 构建模型输入字典
//   3. torch::NoGradGuard 下执行前向推理
//   4. 解析并返回 MatchResult
MatchResult SuperGlueMatcher::match(
    const KeypointData& keypoints0,
    const KeypointData& keypoints1,
    const std::vector<int>& image0_shape,
    const std::vector<int>& image1_shape) {
    
    if (!model_loaded_) {
        throw std::runtime_error("模型未加载");
    }
    
    if (keypoints0.keypoints.empty() || keypoints1.keypoints.empty()) {
        log("警告: 图像中没有关键点");
        return MatchResult();
    }
    
    // 限制关键点数量
    KeypointData kpts0 = (config_.max_keypoints > 0) ? 
        limitKeypoints(keypoints0, config_.max_keypoints) : keypoints0;
    KeypointData kpts1 = (config_.max_keypoints > 0) ? 
        limitKeypoints(keypoints1, config_.max_keypoints) : keypoints1;
    
    torch::NoGradGuard no_grad;  // 禁用梯度计算
    
    // 准备输入
    auto input_dict = prepareInput(kpts0, kpts1, image0_shape, image1_shape);
    
    // 执行推理
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_dict);
    
    auto output = module_.forward(inputs).toGenericDict();
    
    // 将 GenericDict 的引用类型转换为有类型的 Dict<string, Tensor>
    // （torch::jit forward 返回的 GenericDict 需要手动转换以便后续访问）
    torch::Dict<std::string, torch::Tensor> output_dict;
    for (const auto& item : output) {
        output_dict.insert(item.key().toStringRef(), item.value().toTensor());
    }
    
    // 解析输出
    MatchResult result = parseOutput(output_dict);
    
    log("匹配完成: " + std::to_string(result.numMatches) + " 对");
    
    processed_count_++;
    
    return result;
}

// match (从 OpenCV 图像入口): 先提取形状再调用核心接口，并可选做调试输出
MatchResult SuperGlueMatcher::match(
    const cv::Mat& image0,
    const cv::Mat& image1,
    const KeypointData& keypoints0,
    const KeypointData& keypoints1,
    const std::string& image_pair_name) {
    
    // 从 cv::Mat 获取图像形状 [batch=1, channel=1, H, W]
    std::vector<int> shape0 = {1, 1, image0.rows, image0.cols};
    std::vector<int> shape1 = {1, 1, image1.rows, image1.cols};
    
    MatchResult result = match(keypoints0, keypoints1, shape0, shape1);
    
    // 调试: 若启用则将匹配关系写 CSV / 保存可视化图像
    if (config_.enable_csv_output) {
        saveMatchesToCSV(result, image_pair_name);
    }
    
    if (config_.enable_visualization) {
        saveMatchVisualization(image0, image1, keypoints0, keypoints1, result, image_pair_name);
    }
    
    return result;
}

std::vector<MatchResult> SuperGlueMatcher::matchBatch(
    const std::vector<cv::Mat>& images0,
    const std::vector<cv::Mat>& images1,
    const std::vector<KeypointData>& keypoints0_batch,
    const std::vector<KeypointData>& keypoints1_batch) {
    
    if (images0.size() != images1.size() || 
        images0.size() != keypoints0_batch.size() ||
        images0.size() != keypoints1_batch.size()) {
        throw std::runtime_error("批处理输入大小不匹配");
    }
    
    std::vector<MatchResult> results;
    results.reserve(images0.size());
    
    // TODO: 实现真正的批处理推理
    // 目前是逐对处理
    for (size_t i = 0; i < images0.size(); ++i) {
        std::string pair_name = "batch_" + std::to_string(i);
        results.push_back(match(images0[i], images1[i], 
                               keypoints0_batch[i], keypoints1_batch[i], 
                               pair_name));
    }
    
    return results;
}

void SuperGlueMatcher::updateConfig(const SuperGlueConfig& config) {
    // 只更新运行时可修改的参数（不重新加载模型）
    config_.match_threshold = config.match_threshold;
    config_.batch_size = config.batch_size;
    config_.max_keypoints = config.max_keypoints;
    config_.enable_csv_output = config.enable_csv_output;
    config_.csv_output_path = config.csv_output_path;
    config_.enable_visualization = config.enable_visualization;
    config_.visualization_output_path = config.visualization_output_path;
    config_.verbose = config.verbose;
    
    // 根据新配置决定是否打开/关闭 CSV 文件句柄
    if (config_.enable_csv_output && !csv_file_.is_open()) {
        openCSVFile();
    } else if (!config_.enable_csv_output && csv_file_.is_open()) {
        closeCSVFile();
    }
}

void SuperGlueMatcher::openCSVFile() {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    if (csv_file_.is_open()) {
        csv_file_.close(); // 先关闭旧句柄
    }

    csv_file_.open(config_.csv_output_path, std::ios::out | std::ios::trunc);

    if (!csv_file_.is_open()) {
        LOG_WARN("无法打开CSV文件: %s", config_.csv_output_path.c_str());
        return;
    }

    // 写入表头行
    csv_file_ << "image_pair,keypoint0_idx,keypoint1_idx,matching_score,distance" << std::endl;

    log("CSV文件已打开: " + config_.csv_output_path);
}

void SuperGlueMatcher::closeCSVFile() {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    if (csv_file_.is_open()) {
        csv_file_.close();
        log("CSV文件已关闭");
    }
}

// saveMatchesToCSV: 将单次匹配结果追加到已打开的 CSV 文件
// 分数与距离互补: distance = 1 - matching_score
void SuperGlueMatcher::saveMatchesToCSV(const MatchResult& result,
                                        const std::string& image_pair_name) {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    if (!csv_file_.is_open()) {
        return;
    }

    std::string pair_name = image_pair_name.empty() ?
        "pair_" + std::to_string(processed_count_) : image_pair_name;

    for (const auto& match : result.cvMatches) {
        csv_file_ << pair_name << ","
                 << match.queryIdx << ","
                 << match.trainIdx << ","
                 << (1.0f - match.distance) << ","
                 << match.distance << std::endl;
    }

    csv_file_.flush();
}

void SuperGlueMatcher::saveMatchVisualization(
    const cv::Mat& image0,
    const cv::Mat& image1,
    const KeypointData& keypoints0,
    const KeypointData& keypoints1,
    const MatchResult& result,
    const std::string& output_name) {
    
    std::string output_path;
    if (output_name.empty()) {
        // 使用默认路径和计数器
        std::filesystem::path base_path(config_.visualization_output_path);
        std::string stem = base_path.stem().string();
        std::string ext = base_path.extension().string();
        std::filesystem::path parent = base_path.parent_path();

        output_path = (parent / (stem + "_" + std::to_string(processed_count_) + ext)).string();
    } else {
        output_path = output_name;
    }

    // 如果没有文件扩展名，追加默认扩展（优先使用 config_.visualization_output_path 的扩展）
    if (output_path.find_last_of('.') == std::string::npos) {
        std::string default_ext = ".jpg";
        size_t dpos = config_.visualization_output_path.find_last_of('.');
        if (dpos != std::string::npos) {
            default_ext = config_.visualization_output_path.substr(dpos);
        }
        output_path += default_ext;
    }
    
    visualizeMatches(image0, image1, keypoints0.keypoints, keypoints1.keypoints,
                    result, output_path);
}

void SuperGlueMatcher::log(const std::string& message) const {
    if (config_.verbose) {
        // 获取时间戳
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        
        std::cout << "[SuperGlue " 
                 << std::put_time(&tm, "%H:%M:%S") 
                 << "] " << message << std::endl;
    }
}

// visualizeMatches: 使用 cv::drawMatches 生成并保存匹配可视化图像
// 若 cv::imwrite 失败则回退为 PPM 格式（不依赖外部编解码器）
void visualizeMatches(const cv::Mat& image0,
                     const cv::Mat& image1,
                     const std::vector<cv::KeyPoint>& keypoints0,
                     const std::vector<cv::KeyPoint>& keypoints1,
                     const MatchResult& result,
                     const std::string& output_path) {
    
    cv::Mat output_image;
    cv::drawMatches(image0, keypoints0, image1, keypoints1, 
                   result.cvMatches, output_image,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // 添加文字信息
    std::string text = "Matches: " + std::to_string(result.numMatches);
    cv::putText(output_image, text, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    try {
        if (!cv::imwrite(output_path, output_image)) {
            throw std::runtime_error("cv::imwrite returned false");
        }
    } catch (const cv::Exception& e) {
        LOG_WARN("cv::imwrite failed: %s, falling back to PPM writer for '%s'", e.what(), output_path.c_str());
        std::string ppm_path = output_path;
        size_t dot = ppm_path.find_last_of('.');
        if (dot != std::string::npos) ppm_path = ppm_path.substr(0, dot) + ".ppm";

        cv::Mat rgb;
        if (output_image.type() == CV_8UC3) {
            cv::cvtColor(output_image, rgb, cv::COLOR_BGR2RGB);
        } else if (output_image.type() == CV_8UC1) {
            cv::cvtColor(output_image, rgb, cv::COLOR_GRAY2RGB);
        } else {
            output_image.convertTo(rgb, CV_8U, 255.0);
            if (rgb.channels() == 1) cv::cvtColor(rgb, rgb, cv::COLOR_GRAY2RGB);
            else if (rgb.channels() == 4) cv::cvtColor(rgb, rgb, cv::COLOR_BGRA2RGB);
            else if (rgb.channels() == 3) cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        }

        std::ofstream ofs(ppm_path, std::ios::binary);
        if (!ofs.is_open()) {
            LOG_ERROR("无法打开回退 PPM 文件写入: %s", ppm_path.c_str());
        } else {
            ofs << "P6\n" << rgb.cols << " " << rgb.rows << "\n255\n";
            for (int r = 0; r < rgb.rows; ++r) {
                const unsigned char* ptr = rgb.ptr<unsigned char>(r);
                ofs.write(reinterpret_cast<const char*>(ptr), rgb.cols * 3);
            }
            ofs.close();
            LOG_INFO("已回退写入 PPM: %s", ppm_path.c_str());
        }
    } catch (const std::exception& e) {
        LOG_ERROR("imwrite fallback error: %s", e.what());
    }
}

} // namespace superglue
