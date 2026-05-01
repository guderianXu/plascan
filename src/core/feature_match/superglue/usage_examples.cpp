// usage_examples.cpp
// 简化示例：使用指定图像与CSV关键点文件运行 SuperGlueMatcher

#include "SuperGlueMatcher.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace superglue;

// 读取 CSV 文件，格式: x,y,score (带表头)
static KeypointData readKeypointsFromCSV(const std::string& csv_path) {
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("无法打开 CSV 文件: " + csv_path);
    }

    std::string line;
    // 读取表头
    if (!std::getline(ifs, line)) {
        throw std::runtime_error("CSV 文件为空: " + csv_path);
    }

    std::vector<cv::KeyPoint> keypoints;
    std::vector<float> scores;
    std::vector<std::vector<float>> descriptors_list; // 每个关键点的描述符

    const int D = 256; // 期望描述符维度

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ',')) parts.push_back(token);
        if (parts.size() < 3) continue;
        float x = std::stof(parts[0]);
        float y = std::stof(parts[1]);
        float sc = std::stof(parts[2]);
        cv::KeyPoint kp(cv::Point2f(x, y), 1.0f, -1, sc);
        keypoints.push_back(kp);
        scores.push_back(sc);

        // 解析描述符（如果存在）
        std::vector<float> desc;
        if (parts.size() >= 3 + D) {
            desc.reserve(D);
            for (int d = 0; d < D; ++d) {
                desc.push_back(std::stof(parts[3 + d]));
            }
        } else {
            // 若 CSV 中没有描述符，则填充零向量
            desc.assign(D, 0.0f);
        }
        descriptors_list.push_back(std::move(desc));
    }

    int N = static_cast<int>(keypoints.size());
    torch::Tensor descriptors;
    if (N == 0) {
        descriptors = torch::zeros({1, D, 0}, torch::kFloat32);
    } else {
        // 创建 [1, D, N] 并按列填充: 数据内存布局为 C-contiguous
        descriptors = torch::zeros({1, D, N}, torch::kFloat32);
        float* ptr = descriptors.data_ptr<float>();
        // 索引计算: ptr[(0*D + d)*N + i] == ptr[d*N + i]
        for (int i = 0; i < N; ++i) {
            const std::vector<float>& desc = descriptors_list[i];
            for (int d = 0; d < D; ++d) {
                ptr[d * N + i] = desc[d];
            }
        }
    }

    return KeypointData(keypoints, scores, descriptors);
}

int main(int argc, char** argv) {
    try {
        // 指定路径（按用户要求）
        std::string base = "/home/guderian/code/superglue/SuperGluePretrainedNetwork-master/superglue_cpp/test";
        std::string img0_path = base + "/1.png";
        std::string img1_path = base + "/3.png";
        std::string csv0 = base + "/1.csv";
        std::string csv1 = base + "/3.csv";

        // 读取图像
        cv::Mat img0 = cv::imread(img0_path);
        cv::Mat img1 = cv::imread(img1_path);
        if (img0.empty() || img1.empty()) {
            std::cerr << "无法加载输入图像。请检查路径:\n" << img0_path << "\n" << img1_path << std::endl;
            return 1;
        }

        // 读取关键点CSV
        std::cout << "读取关键点: " << csv0 << " , " << csv1 << std::endl;
        KeypointData kpts0 = readKeypointsFromCSV(csv0);
        KeypointData kpts1 = readKeypointsFromCSV(csv1);

        std::cout << "关键点数量: " << kpts0.keypoints.size() << " vs " << kpts1.keypoints.size() << std::endl;

        // 配置 SuperGlue
        SuperGlueConfig config;
        // 请根据实际导出的模型文件修改路径
        config.model_path = "superglue_outdoor_cuda.pt";
        config.use_cuda = true; // 如需使用GPU，设置为true并使用CUDA模型
        config.enable_csv_output = true;
        config.csv_output_path = "matches_test.csv";
        config.enable_visualization = true;
        config.visualization_output_path = "matches_test.jpg";
        config.verbose = true;

        SuperGlueMatcher matcher(config);
        if (!matcher.isLoaded()) {
            std::cerr << "模型未加载，请检查 model_path: " << config.model_path << std::endl;
            return 1;
        }

        // 获取图像形状
        std::vector<int> shape0 = {1, 1, img0.rows, img0.cols};
        std::vector<int> shape1 = {1, 1, img1.rows, img1.cols};

        // 执行匹配
        std::cout << "开始匹配..." << std::endl;
        MatchResult result = matcher.match(img0, img1, kpts0, kpts1, "test_pair");

        std::cout << "匹配完成。找到 " << result.numMatches << " 对匹配" << std::endl;
        std::cout << "CSV 输出: " << config.csv_output_path << std::endl;
        std::cout << "可视化输出: " << config.visualization_output_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
