# MedVisEvi：胸片视觉证据工具

MedVisEvi 是一个面向胸片的可调用、可复算、可追溯视觉证据工具包。项目把模型输出和确定性几何/图像计算统一为结构化证据，并保留输入图像、坐标、单位、模型身份、参数、来源和拒答原因。

当前研究顺序是：先实现并验证 A1–A26 基础工具，再把这些工具组合成 Q/P/T/S/G 五类、26 项业务功能；E1–E10 暂缓。当前已经完成大部分工程实现验证，但尚未完成独立医学准确性评估。

## 文档

- [项目介绍](doc/项目介绍.md)：目标、系统结构、接口约定、环境和调用示例
- [项目现状](doc/项目现状.md)：A 工具矩阵、模型来源、数据限制、修复记录和当前状态
- [数据集与工具评估说明](doc/数据集与工具评估说明.md)：数据集、测试层级、指标和 A1–A26 逐工具评估方法
- [未来计划](doc/未来计划.md)：阶段验收门、组合功能协议和后续任务
- [组合功能与证据评估框架](doc/组合功能与证据评估框架.md)：26 项功能、证据图、EvidenceCard 和评估框架
- [功能集成实现说明](doc/功能集成实现说明.md)：FunctionSpec、依赖路由、三层证据图和统一执行入口
- [A 工具实现验证方案](tool_verification/README.md)：V1–V5 验证方法、冻结样本、容差和结果摘要

## 目录

```text
MedVisEvi/
├── src/visevi/             # 核心库：会话、注册表、A 工具、后端和 worker
├── tests/                  # pytest 软件测试
├── scripts/                # 数据/模型准备、逐步核验和多来源冒烟脚本
├── tool_verification/      # 独立 oracle、验证套件、冻结清单和报告
├── doc/                    # 项目设计、现状、计划和评估说明
├── tools/                  # 依赖锁和本地资产目录说明；权重/第三方源码不入库
├── pyproject.toml          # Python 包和主环境依赖
└── .gitignore              # 数据、权重、环境、缓存和运行产物排除规则
```

原始数据集、模型权重、第三方源码、虚拟环境和实验运行目录没有提交到 GitHub。它们需要按照文档中的来源、版本和 SHA256 记录，在获得相应使用许可后单独准备。

## 环境和测试

主环境使用 Python 3.12。安装主程序和测试依赖：

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[eval]'
```

运行软件测试：

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
```

模型类工具需要单独的模型环境、模型权重、固定版本的第三方源码和本地数据。模型环境依赖锁定在 [`tools/requirements-models.lock`](tools/requirements-models.lock)；模型资产准备和许可限制见项目文档。不要把权重或数据复制进 Git 仓库。

在数据和模型资产准备完成后，可以运行：

```bash
.venv/bin/python scripts/verify_a_tools_stepwise.py \
  --out trials/<new-run> --device cpu --model-device cuda:4

.venv/bin/python scripts/run_a_tools_multisource.py \
  --out trials/<new-run> --per-source 3 --model-device cuda:5
```

`--out` 必须指向新的运行目录。脚本会保存运行结果和中间产物；这些运行目录被 `.gitignore` 排除。

## 当前状态

- 27 个正式接口覆盖 A1–A26，其中 A12 有自动 CTR 和给定边界 CTR 两个接口。
- V1 接口契约和 V2 确定性计算验证已通过；PSPNet 官方流程差分已在修复预处理后通过。
- A5、A26、CXAS 的官方流程差分通过；A8 严格 mask、CheXOne 严格 token 和部分变换一致性检查仍有未决项。
- 当前结果是工程实现核验，不是独立医学准确率结论。
- 26 项组合功能均已有版本化 `FunctionSpec`；Q4、P9、G1 已接入统一 `run_function_episode()`，其余 23 项会返回明确的 `blocked`，尚未实现功能声明程序。
- 功能层新增依赖图、闭集规则路由和 Observation/Relation/Claim 三层证据节点；这属于组合框架的软件实现，不代表功能医学准确性已经评估。
