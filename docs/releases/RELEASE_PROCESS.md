# PlaScan Release Process

本流程用于记录 PlaScan 的分支、tag 和 GitHub Release 更新方式。

## 分支

- `main`：稳定可运行分支，所有对外可见版本必须从这里打 tag。
- `develop`：可选集成分支，用于中期汇总多个功能。
- `feature/<name>`：功能开发分支。
- `fix/<name>`：缺陷修复分支。
- `release/vX.Y.Z`：可选发版准备分支，用于最终文档、版本号和打包验证。

小规模迭代允许直接使用 `main` 加 `feature/*` / `fix/*`，但发布前必须把变更合入 `main`。

## Tag 规则

- 使用语义化版本：`vX.Y.Z`。
- 快速迭代和未稳定版本使用预发布后缀：`vX.Y.Z-alpha.N`、`vX.Y.Z-beta.N`、`vX.Y.Z-rc.N`。
- tag 必须从 `main` 创建，并优先使用 annotated tag。

示例：

```bash
git switch main
git pull --ff-only origin main
git tag -a v1.1.0-alpha.2 -m "PlaScan v1.1.0-alpha.2"
git push origin main
git push origin v1.1.0-alpha.2
```

## Release 文档

每次打 tag 前同步维护：

- `CHANGELOG.md`
- `docs/releases/vX.Y.Z*.md`

版本文档至少包含：

- 新增
- 优化
- 修复
- 验证
- 已知问题

验证项必须写具体命令和结果，不能只写“测试通过”。

## GitHub Release

GitHub Release 说明从对应 `docs/releases/vX.Y.Z*.md` 摘要生成，并保留：

- 版本号和日期
- 主要变更
- 验证命令
- 已知问题
- 下载或打包说明
