# Metashape 对齐照片逆向分析

本目录集中保存 Agisoft Metashape 2.3.1 “对齐照片（Align Photos）”工作流的算法、参数和证据说明。分析基于用户本机合法离线样本的静态取证、Ghidra 12.1.3 反编译，以及 Agisoft 2.3.1 官方手册/API；没有执行、注入、修改或绕过 Metashape。

## 快速结论

- “对齐照片”由 `MatchPhotos` 与 `AlignCameras` 两个阶段组成：前者生成关键点、像对匹配和 tie-point tracks，后者执行初始像对选择、增量相机注册、三角化和 bundle adjustment。
- 特征前端的静态证据指向自有 Gaussian/DoG/LoG 多尺度检测、方向估计和 MLDB 类二进制描述子；不能仅凭 `MLDB` 名称把整套实现称为标准 AKAZE。
- 匹配性能来自连续候选削减：低精度通用预选、参考预选、KD/RKDTree 近似近邻或 GPU 二进制比较、匹配过滤，再由少量高置信点引导第二轮匹配。
- 相机求解是增量 SfM：`evaluateInitialPair` 建立并前瞻评估双相机种子，随后循环 resection、双视/三视三角化、异常轨迹重三角化和反复 BA。
- 可见参数只覆盖部分控制面；`ntrees`、`neighbour_checks`、`binary_features`、`guided_matching_neighbors`、`hierarchical_threshold` 等隐藏键能确认存在，但精确默认值、阈值和数学形式仍未知。

## 文档导航

- [clean-room 兼容实现正式报告](./2026-08-27_clean-room-alignment-implementation-report.md)：本次代码落地、
  Evidence→Finding→Path、验证结果和剩余等价边界。
- [完整分析报告](./metashape-align-photos-analysis.md)：内部算法、主调用链、可信度边界和 Evidence→Finding→Path。
- [参数参考](./parameter-reference.md)：GUI/API 参数、2.3.1 默认值、内部作用阶段、参数耦合和隐藏键。
- [clean-room 复刻状态](./implementation-status.md)：PlaScan 已实现的对齐链路、本次新增二进制兼容通道、
  使用方式、验证结果以及仍无法声称逐结果相同的边界。
- [证据与复现](./evidence-reproduction.md)：样本身份、函数地址、Ghidra 产物哈希和复现命令。
- [可编辑流程图](./workflow.mmd)：报告内 Mermaid 图的独立源文件。
- [复刻实现流程图](./implementation-flow.mmd)：PlaScan 的算法分支、几何验证和增量 SfM 数据流。
- [公开资料索引](./references/public-sources.md)：本次检索并固化的官方资料和原始论文。

历史上较早、包含 PlaScan 对照建议的报告仍保留在 [2026-08-26_reverse-metashape-alignment-report.md](../2026-08-26_reverse-metashape-alignment-report.md)。本目录版本更聚焦 Metashape 本身，并补充了 2.3.1 API 默认值与隐藏参数分层。

## 结论边界

本次可以确认算法族、阶段边界、配置键、缓存对象和若干关键控制流；不能由静态证据确认描述子精确位数、匹配 ratio/距离阈值、首轮几何模型、PnP 最小解、BA 鲁棒核或线性求解器。文档中分别使用“确认”“高置信推断”和“未知”标注，避免把行业常见实现当成 Metashape 事实。
