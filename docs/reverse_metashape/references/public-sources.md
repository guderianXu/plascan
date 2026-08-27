# 公开资料与知识固化

本页记录 2026-08-27 为本次分析实际核对的公开资料。产品行为优先使用 Agisoft 官方手册/API；论文只说明算法思想谱系，不证明 Agisoft 逐行采用某篇论文或某个开源实现。

## Agisoft 官方资料

| 来源 | 本次使用的信息 |
|---|---|
| [Metashape Professional 2.3 User Manual](https://www.agisoft.com/pdf/metashape-pro_2_3_en.pdf) | Align Photos GUI 参数；Accuracy 的缩放语义；Generic/Reference/Sequential 预选；mask、stationary filter、guided matching、adaptive fitting；重置与增量对齐。 |
| [Metashape Python API 2.3.1](https://www.agisoft.com/pdf/metashape_python_api_2_3_1.pdf) | `matchPhotos(...)` 与 `alignCameras(...)` 的 2.3.1 参数名、API 默认值和任务分块参数。 |
| [MatchPhotos Java API](https://download.agisoft.com/metashape-java-api/latest/com/agisoft/metashape/tasks/MatchPhotos.html) | MatchPhotos 任务对象的参数类型、默认值和 camera/pair workitem 契约。 |
| [AlignCameras Java API](https://download.agisoft.com/metashape-java-api/latest/com/agisoft/metashape/tasks/AlignCameras.html) | `adaptive_fitting`、`min_image`、`reset_alignment` 和 task subdivision 契约。 |

### 2.3.1 默认值快照

Python API 2.3.1 的 `matchPhotos` 签名给出：`downscale=1`、generic/reference preselection 均为 `True`、Source reference mode、`filter_mask=False`、`mask_tiepoints=True`、stationary filter `True`、`keypoint_limit=40000`、`keypoint_limit_per_mpx=1000`、`tiepoint_limit=4000`、`keep_keypoints=False`、guided matching `False`、reset matches `False`、subdivide task `True`、camera/pair workitem 为 `20/80`、max workgroup `100`。

`alignCameras` 签名给出：`min_image=2`、`adaptive_fitting=False`、`reset_alignment=False`、`subdivide_task=True`。GUI 手册同时展示/推荐 tie point limit `10000`，因此报告明确区分“API 默认 4000”和“GUI/手册推荐 10000”。

## 原始算法资料

| 资料 | 与本次静态证据的关系 |
|---|---|
| D. Lowe, [Distinctive Image Features from Scale-Invariant Keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), IJCV 2004 | Gaussian/DoG 尺度空间、极值定位和方向归一化的公开经典谱系。 |
| P. Alcantarilla et al., [Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces](https://www.bmva.org/bmvc/2013/Papers/paper0013/index.html), BMVC 2013 | M-LDB 二进制描述子的公开来源；标准 AKAZE 的非线性扩散检测与本样本 Gaussian/DoG/LoG 证据不同。 |
| M. Muja and D. Lowe, [FLANN](https://www.cs.ubc.ca/research/flann/) | randomized KD forests 与 `ntrees`/search checks 参数族的公开背景。 |
| J. Schönberger and J.-M. Frahm, [Structure-from-Motion Revisited](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html), CVPR 2016 | 初始像对、增量注册、三角化、过滤和 BA 的现代公开范式；仅用于解释算法族。 |

## 可信度使用规则

- 官方手册/API + 二进制配置键/RTTI：可确认阶段和参数语义。
- 配置键/RTTI + kernel 名 + Ghidra 控制流：可形成高置信算法族推断。
- 只有论文相似：不能提升为产品实现事实。
- 缺少运行时 I/O 或明确数学操作：阈值、损失函数、最小解与 solver 保持未知。
