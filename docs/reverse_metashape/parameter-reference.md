# Metashape 2.3.1 对齐照片参数参考

本文把 2.3.1 GUI、Python/Java API 和二进制内部键分开列出。这里的“默认值”优先指 2.3.1 Python API 函数签名；GUI 对话框可能保存上次值或采用不同推荐预设，不能混为一谈。

## 公开参数与内部作用

| GUI / API 参数 | 2.3.1 API 默认 | 作用阶段 | 内部语义与影响 |
|---|---:|---|---|
| Accuracy / `downscale` | `1`（High） | 特征检测 | `0/1/2/4/8` 对应 Highest/High/Medium/Low/Lowest。High 使用原图；Medium、Low、Lowest 的边长分别缩小 2/4/8 倍，像素量约缩小 4/16/64 倍；Highest 对边长放大 2 倍。它改变检测尺度、定位精度、内存和计算量，不是 BA 精度开关。 |
| `generic_preselection` | `True` | 像对预选 | 先以更低精度匹配图片，召回可能重叠的像对，再进入正式特征匹配；主要降低全部 `N(N-1)/2` 像对的成本。 |
| `reference_preselection` | `True` | 像对预选 | 用相机先验进一步裁剪候选。与 generic preselection 可同时启用，并非互斥算法。 |
| `reference_preselection_mode` | `Source` | 像对预选 | `Source` 使用测得位置/姿态和斜视拍摄距离；`Estimated` 使用已有对齐外方位；`Sequential` 按照片序列建立候选，并包含首尾比较。 |
| Reset current alignment / `reset_matches`, `reset_alignment` | 均为 `False` | 缓存与重算 | GUI 的“重置当前对齐”会丢弃关键点、匹配、tie points 和现有对齐；API 将前端匹配重置与后端相机重置拆成两个布尔参数。 |
| Save project after each step | GUI 选项 | 调度/恢复 | 在已完成子任务后保存工程，用于崩溃或断电后的续算；它改变持久化与恢复粒度，不应改变几何结果。 |
| `keypoint_limit` | `40000` | 特征检测/选择 | 每张图进入匹配阶段的关键点上限；`0` 表示不施加该上限，可能引入更多弱点并显著增加内存与匹配成本。 |
| `keypoint_limit_per_mpx` | `1000` | Guided matching | 每百万像素关键点预算。官方定义的总预算为该值乘影像 Mpx；guided 模式仅对小部分点做充分匹配，再用其几何关系引导剩余点。 |
| `tiepoint_limit` | `4000` | 轨迹保留/网络稀疏化 | 每张图保留的匹配点上限；`0` 表示不做 tie-point 上限过滤。2.3 GUI 手册截图/推荐值为 `10000`，与 API 默认 `4000` 不同。它是网络预算，不等同于关键点数。 |
| Apply masks to key points / `filter_mask` | `False` | 特征检测 | 在被掩膜像素中不检测关键点，适合每张图都具有可靠前景掩膜的场景。 |
| Apply masks to tie points / `mask_tiepoints` | `True` | 轨迹过滤 | 若某一图中的对应区域被掩膜，会排除相关多图 tie-point track；这是轨迹级传播，语义强于只过滤本图检测。 |
| Exclude stationary tie points / `filter_stationary_points` | `True` | 匹配/轨迹过滤 | 排除在多张不同影像中像素位置几乎不变的对应，可抑制固定相机转台背景、传感器污点或镜头伪影；对于同一位置多次拍摄的有效内容需要谨慎。 |
| Guided image matching / `guided_matching` | `False` | 几何验证后的第二轮匹配 | 小规模高置信匹配先建立几何约束，再限制剩余点的候选集合。可提高高分辨率、植被、球面相机或扫描航片上的有效点数，但会增加关键点存储和第二轮调度。 |
| Adaptive camera model fitting / `adaptive_fitting` | `False` | AlignCameras / BA | 依据参数可靠性自动决定释放哪些畸变参数；强几何可释放更多参数，弱航带几何则冻结不可靠高阶项。关闭时，手册说明固定精化 `f, cx, cy, k1, k2, k3, p1, p2`。 |
| `keep_keypoints` | `False` | 缓存/增量对齐 | 将关键点保存在工程内，允许后来新增照片复用已有关键点；代价是工程体积增大。 |
| `pairs`, `cameras` | 未指定 | 任务范围 | 可直接指定待匹配像对或相机集合，覆盖/收紧自动候选范围，适合重试、增量任务和外部调度。 |
| `subdivide_task` | `True` | 并行/网络调度 | 启用细粒度任务拆分。几何语义理论上不变，但影响并行度、失败恢复和调度开销。 |
| `workitem_size_cameras` | `20` | 特征任务分块 | 每个相机 workitem 的相机数。 |
| `workitem_size_pairs` | `80` | 匹配任务分块 | 每个像对 workitem 的像对数。 |
| `max_workgroup_size` | `100` | 网络/设备调度 | 单个工作组上限；它是工程调度预算，不是匹配接受阈值。 |
| AlignCameras `min_image` | `2` | 三角化/对齐 | 点的最少影像投影数，默认要求至少双视观测。 |
| AlignCameras `subdivide_task` | `True` | 后端调度 | 对相机对齐阶段继续细分工作项。 |

`downscale_3d`、`keypoint_limit_3d`、`keypoint_limit_depth_maps`、`match_laser_scans` 和 `match_depth_maps` 属于激光扫描/深度图联合对齐扩展，不是普通照片对齐主链；2.3.1 API 默认分别为 `1`、`100000`、`10000`、`False`、`False`。

## 参数之间最重要的耦合

1. `downscale` 决定检测输入尺度，`keypoint_limit` 决定每图候选上限，`tiepoint_limit` 决定最终参与网络的匹配点上限；三者分别控制不同阶段，不能用同一个“点越多越好”逻辑解释。
2. 启用 `guided_matching` 后，`keypoint_limit_per_mpx` 变得关键。内部不是对所有高密度点做全量匹配，而是以小样本充分匹配建立几何，再为其余点缩小候选。
3. `generic_preselection` 解决“没有可靠先验时哪些图可能重叠”，`reference_preselection` 利用外部或已有位姿；同时启用相当于视觉召回与先验裁剪的组合。
4. `filter_mask` 在检测前删区域，`mask_tiepoints` 在形成跨图轨迹后删对应，后者可能让一张图的掩膜影响其它图上的同一 3D 候选。
5. `adaptive_fitting` 只改变 BA 中内参/畸变参数的释放策略，不改变前端使用哪种特征或描述子。
6. `reset_matches`、`reset_alignment`、`keep_keypoints` 和保存子任务共同定义缓存生命周期；对大型工程而言，它们也是算法可重入性的一部分。

## 二进制确认但未公开稳定契约的键

| 内部键 | 高置信语义 | 当前证据边界 |
|---|---|---|
| `MatchPhotos/ntrees` | randomized KD-tree forest 的树数/索引预算 | 与 `feat::KDTree`、`feat::RKDTree` RTTI 共现；未恢复 2.3.1 默认值和所有 descriptor mode 是否共用。 |
| `MatchPhotos/neighbour_checks` | ANN 查询检查预算，值越大通常召回越高、耗时越大 | 算法族为高置信推断；未恢复精确停止条件。 |
| `generic_preselection_limit` | 通用预选候选像对/邻居上限 | 键存在；精确计数单位未知。 |
| `generic_preselection_level`、`generic_preselection_reduce` | 通用预选的低精度层级与进一步缩减策略 | 与官方“先以较低精度匹配”一致；具体金字塔层和削减公式未知。 |
| `reference_preselection_neighbors` | 参考预选每图候选邻居预算 | 键存在；空间/姿态评分公式未知。 |
| `filter_matches`、`filter_weak_points` | 描述子或几何阶段的弱匹配/弱点过滤开关 | 不能据此声明使用 Lowe ratio、USAC 或 MAGSAC。 |
| `binary_features` | 切换/启用二进制特征通道 | 与 `gpu::MLDB`、`mldb_compare_values_device` 交叉印证。 |
| `guided_matching_neighbors` | guided second pass 的候选邻居数/`max_candidates` 预算 | `FUN_141cf1cd0` 会拒绝 `max_candidates == 0` 并构建受限 query/train 候选；默认值未知。 |
| `descriptor_type`、`descriptor_version` | 工程元数据中记录描述子类型和版本 | 可用于缓存兼容与重算判断；不公开具体编码布局。 |
| `AlignCameras/hierarchical_threshold` | 分层相机对齐/模型选择阈值 | 仅确认配置键和读取路径；数学定义与单位未知，置信度低。 |
| `MatchPhotos/exclude_corners`、`AlignCameras/exclude_corners` | 角点/边缘区域排除策略 | 键存在且进入 GUI 参数装配函数；判据与默认值未知。 |
| `duration`、`ram_used` | 任务耗时和峰值内存元数据 | 属于可观测性，不是几何参数。 |

这些内部键应视为版本私有实现细节，不适合在兼容实现中直接暴露为稳定 API。若要继续确认默认值或阈值，需要在授权小数据集上做参数 I/O 差分或针对构造函数进行更深的类型恢复；本次静态分析没有猜填。

## 场景化理解

- 大规模航测、参考位置可靠：High、generic + Source reference preselection 构成粗到细候选图；是否启用 guided matching取决于植被/弱纹理和可接受的内存成本。
- 转台或固定相机背景：stationary filter 与 tie-point mask 能抑制静态背景，但需确认真实主体是否也在相同像素位置长期停留。
- 弱航带几何：adaptive fitting 的意义是限制不可观测畸变参数发散，不是自动保证更高精度。
- 增量补图：必须保留 keypoints，并避免重置现有 matches/alignment，才能真正复用缓存。

参数来源详见 [公开资料索引](./references/public-sources.md)；内部键证据详见 [证据与复现](./evidence-reproduction.md)。
