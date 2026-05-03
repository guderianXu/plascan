# CLAUDE.md
# 通用
- 优先选择编辑而非重写整个文件
- 除非文件被编辑过，否则不要重复阅读已读过的文件
- 输出追求简洁，但推理过程必须详尽

# 代码规范
- 一个文件不超过 400 行，超了就拆
- 嵌套不超过4层
- 花括号使用 Allman 风格（左花括号独占一行），不用 K&R 风格

# Git 同步
- **每次 commit 后必须 push 到 GitHub**：`git push origin main`（含 tags: `git push origin <tag>`）
- 仓库地址：`https://github.com/guderianXu/plascan`
- **Git 作者配置**：必须使用 GitHub 关联邮箱，否则提交不计入贡献统计
  - `git config user.email "guderian_xu@henu.edu.cn"`
  - `git config user.name "guderianXu"`
  - 注意：`guderian@plascan.local` 是不关联 GitHub 的本地邮箱，不要使用
  - 如已用错误邮箱提交，需要用 `git filter-branch --env-filter` 重写历史

# 编译验证
- **每次修改代码后必须在 build 目录编译验证**：
  ```bash
  cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)
  ```
- 编译失败不能提交；编译警告需评估

# 开发流程 (Feature Branch + TDD)

**分支策略**：
- `main` 分支始终稳定可构建，不直接在 main 上开发
- 每个功能/修复在独立 feature 分支上开发：`feat/<描述>` 或 `fix/<描述>`
- 分支上完成 TDD 循环（红→绿→重构）且测试全部通过后，合并回 main

**TDD 铁律**：
1. 先写测试 → 运行确认失败（红）
2. 写最小实现 → 运行确认通过（绿）
3. 重构优化 → 保持测试通过
4. 每次提交前跑相关测试，禁止提交破坏测试的代码

**分支操作规范**：
```bash
git checkout -b feat/<功能名>   # 从 main 创建功能分支
# ... TDD 开发 ...
git checkout main && git merge feat/<功能名>   # 测试通过后合并
git push origin main
```

# 项目结构
- 项目整体架构文档在 `docs/PROJECT_ARCHITECTURE.md`，包含完整目录树、模块职责、菜单结构、数据流和已知技术债务
- 密集匹配模块文档在 `src/core/dense_match/README.md`
- 特征提取模块文档在 `src/core/feature_extractors/README.md`
- **每次修改代码前后必须对照上述文档**：新增/删除/移动文件后同步更新文档中对应的目录树和模块说明，若发现文档与实际不符立即修正
