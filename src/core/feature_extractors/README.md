# 特征提取模块 (feature_extractors)

从影像中检测关键点并计算描述子。所有提取器输出统一的 `SuperPointOutput` 结构，可复用 `.sp` 文件格式和下游匹配流程。

## 目录结构

```
feature_extractors/
├── FeatureData.h/cpp          # 通用特征数据容器 (从 SuperPointOutput 转换)
├── CMakeLists.txt             # 注册所有提取器子模块
│
├── superpoint/                # SuperPoint 深度学习提取器
│   ├── SuperPoint.h/cpp       # TorchScript 模型加载 + 推理
│   ├── QFileBinaryIO.h/cpp    # .sp 二进制格式 I/O
│   └── tests/
│
├── tradition/                 # 传统算法 (OpenCV)
│   ├── TraditionalFeatureExtractor.h/cpp  # SIFT/ORB/AKAZE 统一接口
│   └── tests/
│
├── disk/                      # DISK 深度学习提取器 (新增)
│   ├── DiskExtractor.h/cpp    # LibTorch wrapper (kornia DISK)
│   ├── DiskConfig/DiskOutput  # 配置与输出结构
│   └── tests/test_disk.cpp
│
├── aliked/                    # ALIKED 深度学习提取器 (新增)
│   ├── AlikedExtractor.h/cpp  # LibTorch wrapper (lightglue ALIKED)
│   ├── AlikedConfig/AlikedOutput
│   └── tests/test_aliked.cpp
│
├── testdata/                  # 测试影像
│   ├── 1.tif                  # 4608×3456 纳卫星PAN
│   └── 2.tif                  # 4608×3456 纳卫星PAN
│
└── README.md                  # 本文档
```

## 提取器对比

| 提取器 | 模型 | 描述子维度 | 设备 | 特点 |
|--------|------|:---:|------|------|
| SuperPoint | superpoint_extractor_{cuda,cpu}.torchscript | 256 | GPU/CPU | 最通用, 与SuperGlue/LightGlue配套 |
| SIFT | OpenCV内置 | 128 | CPU | 尺度不变, 卫星影像鲁棒 |
| ORB | OpenCV内置 | 32 | CPU | 极快, binary描述子 |
| AKAZE | OpenCV内置 | 61 | CPU | 非线性扩散, 边缘保持 |
| DISK | disk_extractor_{cuda,cpu}_8192.torchscript | 128 | GPU/CPU | 深度学习, 与LightGlue disk配套 |
| ALIKED | aliked_extractor_{cuda,cpu}_480.torchscript | 128 | GPU/CPU | 深度学习, 与LightGlue aliked配套 |

## 性能基准 (testdata/1.tif, 4608×3456)

| 提取器 | 关键点数 | 耗时 | 设备 |
|--------|---------|------|------|
| SuperPoint (max-dim=2800) | 924 | 1.9s | GPU |
| SIFT 4k | 428 | 1.3s | CPU |
| ORB 4k | 351 | 0.0s | CPU |
| AKAZE | 188 | 0.9s | CPU |
| DISK (8192 keypoints) | up to 8192 | 视图像尺寸而定 | GPU/CPU |
| ALIKED (480px) | ~150 | ~0.3s | CPU |

## 使用方式

### C++ CLI

```bash
# SuperPoint
feature_extract_cli -a superpoint -m superpoint_extractor_cpu.torchscript -i img.tif -o out.sp --cuda --max-dim 2800

# SIFT/ORB/AKAZE (无需模型)
feature_extract_cli -a sift  -i img.tif -o out.sp -n 4096
feature_extract_cli -a orb   -i img.tif -o out.sp -n 5000
feature_extract_cli -a akaze -i img.tif -o out.sp
```

### C++ 库调用

```cpp
// SuperPoint
SuperPointConfig cfg;
cfg.max_num_keypoints = 4096;
SuperPoint sp("superpoint_extractor_cpu.torchscript", cfg);
auto output = sp.detect(image);

// DISK
xjw::feature_extractors::DiskConfig dcfg;
dcfg.modelPath = "disk_extractor_cuda_8192.torchscript";
xjw::feature_extractors::DiskExtractor disk(dcfg);
auto dout = disk.extract(grayImage);

// ALIKED
xjw::feature_extractors::AlikedConfig acfg;
acfg.modelPath = "aliked_extractor_cpu_480.torchscript";
xjw::feature_extractors::AlikedExtractor aliked(acfg);
auto aout = aliked.extract(grayImage);

// 传统
auto output = TraditionalFeatureExtractor::detect(image, config, "sift");
```

## 统一输出格式

所有提取器输出 `SuperPointOutput` (或可转换为此格式):
```cpp
struct SuperPointOutput {
    std::vector<cv::KeyPoint> keypoints;  // 像素坐标 [0,W]×[0,H]
    std::vector<float>        scores;     // 响应值
    torch::Tensor             descriptors; // [N, D] float32, L2归一化
};
```

通过 `QFileBinaryIO::write()` 保存为 `.sp` 二进制文件，`QFileBinaryIO::read()` 加载。

## 新增提取器

1. 在对应子目录创建 `XxxExtractor.h/cpp` (参考 `DiskExtractor`)
2. 创建 `CMakeLists.txt`
3. 在父 `CMakeLists.txt` 中添加 `add_subdirectory()`
4. 写测试 `tests/test_xxx.cpp`
5. 更新本文档
