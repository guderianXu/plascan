# 证据、函数地址与复现说明

## 范围

- Case：`work/metashape-align-photos-20260827/`
- Scope：[scope.md](local-evidence.md)
- 网络模式：目标分析为 offline；公开网络仅用于 Agisoft 文档和论文核对
- 未执行或修改目标样本；Ghidra 只复用同 SHA-256 的本地分析工程并重新导出证据

## 样本与工具

| 项目 | 值 |
|---|---|
| 样本 | `D:\metashape2.3.1\metashape.exe` |
| SHA-256 | `457BC052641A52C938CBED51C98E927A1C9120A7694F7CCF0E2F1E942B5F3E50` |
| Ghidra | 12.1.3，`build/tmp/metashape-reverse/tools/ghidra_12.1.3_PUBLIC/` |
| JDK | 21.0.12.1，`build/tmp/metashape-reverse/tools/jdk-21.0.12.1+1/` |
| PE 工具 | MSVC 14.44 `dumpbin.exe` |
| Python | 仓库 `.venv` |

Windows `VersionInfo` 对该可执行文件返回通用 `1.0.0.1`，不能作为产品版本依据；本报告的 2.3.1 版本来自安装包/产品环境与 2.3.1 API 契约，样本身份始终以 SHA-256 固定。

## 固定产物

| 产物 | SHA-256 | 内容 |
|---|---|---|
| `evidence/binary/alignment_strings.tsv` | `DE3BC99131D77E6C3129CC90AC0546B631589E6B914154A1A622B4F066284BA5` | 按 PE 节提取的 alignment/matching/SfM/CUDA 字符串 |
| `evidence/binary/alignment_strings_summary.json` | `C70F1FB09938CF44C77B4742A0D4E69274952E9CCDBC3ABF1968E02265AF2FDD` | 3461 个命中字符串与节统计 |
| `evidence/ghidra/alignment-xrefs.tsv` | `637EF8666B04C4A903B255C3AA04AA37F31BD7B13552083F6FE15AAF7378E82F` | 36 个关键字符串到函数的 xref |
| `evidence/ghidra/alignment-direct-functions.md` | `CFE41B2FCD9C3A2326206F7D04AA047004276BFC7FFFB42248A2B2FBC7D5D681` | 26 个直接相关函数的反编译正文 |
| `evidence/ghidra/alignment-function-neighborhood.tsv` | `4DF9C6ED7BE9AA78A89D148286CF39CEB3E3CEC1F07054980B1170BC903A9323` | 直接函数的一层 caller/callee 邻域 |

前三类可点击产物位于 [evidence 目录](local-evidence.md)。Ghidra 邻域表的行顺序可能受无序集合遍历影响，语义核对以 xref、函数地址和反编译主产物为准。

## 关键地址

| 地址 | Ghidra 名称 | 作用锚点 |
|---|---|---|
| `0x140176690` | `FUN_140176690` | `main/match_*` 到 `MatchPhotos/*` / `AlignCameras/*` 的配置迁移与内部隐藏键 |
| `0x1414b10e0` | `FUN_1414b10e0` | Align Photos GUI 参数装配；读取 keypoint/tiepoint/guided/adaptive 等 QVariant |
| `0x141cf1cd0` | `FUN_141cf1cd0` | guided candidate 限制、并行/GPU helper 与匹配输出 |
| `0x1423ef5d0` | `FUN_1423ef5d0` | `log_locate_points_device` wrapper/xref |
| `0x142404720` | `FUN_142404720` | orientation histogram/device wrapper/xref |
| `0x142405110` | `FUN_142405110` | MLDB extract wrapper/xref |
| `0x1420a2de0` | `FUN_1420a2de0` | collapsed candidate RANSAC 重三角化 |
| `0x1420b2370` | `FUN_1420b2370` | triplet triangulation |
| `0x1431e1080` | `FUN_1431e1080` | `Block::align` 双相机 seed 与 resection 循环 |
| `0x1431ed310` | `FUN_1431ed310` | `evaluateInitialPair` 与有限前瞻 |
| `0x143252840` | `FUN_143252840` | `bundle_adjust:` 主锚点 |

## 复现命令

以下命令从仓库根目录 `E:\code\plascan` 执行。

```powershell
$sample = 'D:\metashape2.3.1\metashape.exe'
$dumpbin = 'C:\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\dumpbin.exe'

Get-FileHash -Algorithm SHA256 -LiteralPath $sample
Get-AuthenticodeSignature -LiteralPath $sample |
    Select-Object Status, StatusMessage, SignerCertificate
& $dumpbin /headers /imports $sample
```

重新提取 PE/CUDA 字符串：

```powershell
& '.\.venv\Scripts\python.exe' `
    'build\tmp\metashape-alignment-analysis\scripts\extract_alignment_strings.py' `
    'D:\metashape2.3.1\metashape.exe' `
    'work\metashape-align-photos-20260827\evidence\binary'
```

重新从现有 Ghidra 工程导出关键函数：

```powershell
$env:JAVA_HOME = 'E:\code\plascan\build\tmp\metashape-reverse\tools\jdk-21.0.12.1+1'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"

& 'E:\code\plascan\build\tmp\metashape-reverse\tools\ghidra_12.1.3_PUBLIC\support\analyzeHeadless.bat' `
    'E:\code\plascan\build\tmp\metashape-reverse\ghidra-project' `
    'MetashapeStatic' `
    -process 'metashape.exe' `
    -noanalysis `
    -scriptPath 'E:\code\plascan\build\tmp\metashape-alignment-analysis\ghidra-scripts' `
    -postScript 'ExportAlignmentAnchors.java' `
    'E:\code\plascan\work\metashape-align-photos-20260827\evidence\ghidra'
```

核对关键调用锚点：

```powershell
rg -n -C 18 `
    'Block::align: before resection|evaluateInitialPair|Triangulating triplets|bundle_adjust:' `
    'work\metashape-align-photos-20260827\evidence\ghidra\alignment-direct-functions.md'

rg -n `
    'ntrees|neighbour_checks|binary_features|guided_matching|adaptive_fitting|hierarchical_threshold' `
    'work\metashape-align-photos-20260827\evidence\binary\alignment_strings.tsv'
```

## 失败与恢复记录

第一次 Ghidra 复核因当前 shell 未设置 `java`/`JAVA_HOME` 而在启动器阶段退出，未进入样本处理。随后从既有成功分析工具目录定位 JDK 21.0.12.1，显式设置 `JAVA_HOME` 后重跑成功，日志显示 `Alignment anchor hits: 36`、`Direct functions: 26` 且无 stderr。该问题属于工具运行时环境，不影响样本或 Ghidra 工程完整性。

## 剩余风险

- 反编译函数缺少完整原始类型和符号，局部变量名与部分控制流是 Ghidra 恢复结果。
- 同一字符串键可被配置迁移、GUI 装配和实际算法多处引用；报告只在有数据流/多点互证时提升结论。
- 没有动态参数差分，私有键默认值、阈值单位和运行时分支保持未知。
- 没有用同一影像集做 Metashape 与其它实现的性能/精度黑盒比较，因此不提供百分比排名。
