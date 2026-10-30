# 本地模型资产目录

本目录只在服务器运行时使用。公开仓库不包含模型权重、第三方模型源码、模型缓存或 Hugging Face 下载缓存。

运行模型类 A 工具前，需要在本地准备：

- `tools/weights/`：各模型权重和对应的 `provenance.json`；
- `tools/src/`：固定 commit 的第三方推理源码；
- `tools/cache/`：运行时模块缓存；
- `.venv-models/`：模型 worker 环境。

模型来源、版本、许可、SHA256 和预处理约定记录在 `doc/项目现状.md`、`doc/项目介绍.md` 和 `tool_verification/README.md`。主环境依赖见项目根目录 `pyproject.toml`，模型环境依赖锁定见 `tools/requirements-models.lock`。

请先确认各模型和数据集的使用许可，再下载到这些本地目录；不要把权重、医学影像或访问凭据提交到 GitHub。
