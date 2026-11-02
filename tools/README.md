# 本地模型资产目录

本目录只在服务器运行时使用。公开仓库不包含模型权重、第三方模型源码、模型缓存或 Hugging Face 下载缓存。

运行模型类 A 工具前，需要在本地准备：

- `tools/weights/`：各模型权重和对应的 `provenance.json`；
- `tools/src/`：固定 commit 的第三方推理源码；
- `tools/cache/`：运行时模块缓存；
- `.venv-models/`：模型 worker 环境。
- `.venv-vqa/`：VQA 终端模型环境，单独安装兼容 Qwen3-VL 的 Transformers；不修改 `.venv-models/`。

模型来源、版本、许可、SHA256 和预处理约定记录在 `doc/项目现状.md`、`doc/项目介绍.md` 和 `tool_verification/README.md`。主环境依赖见项目根目录 `pyproject.toml`，A 工具模型环境依赖锁定见 `tools/requirements-models.lock`，VQA 模型环境依赖锁定见 `tools/requirements-vqa.lock`。

创建隔离的 VQA 环境：

```bash
.venv-models/bin/python -m venv --system-site-packages .venv-vqa
.venv-vqa/bin/python -m pip install -r tools/requirements-vqa.lock
```

VQA 环境继承服务器已有 PyTorch/CUDA 基础，只覆盖 Transformers 和 tokenizer 版本。模型权重保存在 `/ssd/visual_reasoning/zym/models/` 或经批准的模型目录，不放进 Git 仓库。

请先确认各模型和数据集的使用许可，再下载到这些本地目录；不要把权重、医学影像或访问凭据提交到 GitHub。
