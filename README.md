# MedVisEvi：胸片视觉证据工具

MedVisEvi 是一个面向胸片的可调用、可复算、可追溯视觉证据工具包。项目把模型输出和确定性几何/图像计算统一为结构化证据，并保留输入图像、坐标、单位、模型身份、参数、来源和拒答原因。

当前工作并行推进 A1–A26 基础工具的实现验证与 Q/P/T/S/G 五类、26 项业务功能的组合开发；正式功能评估依赖冻结协议和合适参照。E1–E10 暂缓。

## 文档

- [项目介绍](doc/项目介绍.md)：目标、系统结构、接口约定、环境和调用示例
- [项目现状](doc/项目现状.md)：A 工具矩阵、模型来源、数据限制、修复记录和当前状态
- [A 工具详细说明与评估矩阵](doc/A工具详细说明与评估矩阵.md)：A1–A26 每项工具的后端、输出形态、输出内容和评估状态
- [数据集与工具评估说明](doc/数据集与工具评估说明.md)：数据集、测试层级、指标和 A1–A26 逐工具评估方法
- [MIMIC 数据调研与分批评估计划](doc/MIMIC数据调研与分批评估计划.md)：26 项功能的数据适配矩阵、标注边界与小批量下载方案
- [MedVisEvi 方法框架](doc/MedVisEvi方法框架.md)：项目的方法分层、证据信息流、功能执行和评估层级
- [未来计划](doc/未来计划.md)：阶段验收门、组合功能协议和后续任务
- [组合功能与证据评估框架](doc/组合功能与证据评估框架.md)：26 项功能、证据图、EvidenceCard 和评估框架
- [VQA 端到端 Pipeline 计划](doc/VQA端到端Pipeline计划.md)：从图片和问题到功能路由、工具执行、证据卡、医学模型和答案核验的分阶段计划
- [VQA 图像、问题与证据联合输入方案](doc/VQA图像问题与证据联合输入方案.md)：一手文献调研、EvidenceCard 组织、冲突处理、消融实验和当前实现边界
- [五类 Gate 网络调研与实现](doc/五类Gate网络调研与实现.md)：Q/P/T/S/G 五个独立 Gate、线性/MLP 结构、类别选择策略与可选 VQA 接入；训练暂缓
- [功能集成实现说明](doc/功能集成实现说明.md)：FunctionSpec、依赖路由、三层证据图和统一执行入口
- [Q/P 开发集流程记录](doc/Q_P开发集流程记录.md)：Q1–Q4、P1–P12 的 400 张开发批次覆盖、拒答统计和后续评估缺口
- [T4 胸腔引流管调研与实现方案](doc/T4胸腔引流管调研与实现方案.md)：T4 reference adapter 的边界和自动后端路线
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
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider
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
- 26 项组合功能均已有版本化 `FunctionSpec`；全部 26 项已接入统一 `run_function_episode()`。T4 当前是显式 caller-reference adapter：缺少带来源的引流管 reference 时结构化拒答，尚没有自动胸腔引流管模型。
- 代表性功能已通过人工可算输入和假模型输出的软件集成测试，用于验证依赖、证据关系、拒答和溯源；医学准确性仍按独立协议评估。
- 功能层已实现依赖图、闭集规则路由和 Observation/Relation/Claim 三层证据节点。
- Q/P/T/S/G 五个独立类别 Gate 已有线性/MLP 网络、独立 Sigmoid 阈值策略和可选 VQA 适配；问题编码器、权重训练与校准暂缓。未训练时返回 `untrained`，默认仍用规则路由。
- VQA Python API 已将 26 项单功能接入统一路由、FunctionSpec、A 工具、EvidenceCard、consumer 和 verifier。Q4 用专用闭集答案协议，其他功能只接受与 EvidenceCard 完全一致且 Claim 绑定的结构化输出。
- S1–S3 支持 prior/current 双影像；G3、P8、P10、T4 等外部 ROI/reference 输入通过带图像来源的 `EvidenceSession` 提供。自然语言槽位抽取、多功能联合回答、通用冲突仲裁和真实 VLM 评估尚未完成。
- 单样本通用 VQA 命令为 `scripts/run_vqa.py`；`--inputs-json` 和 `--slots-json` 提供功能参数，`--prior-image` 提供配对检查影像。`evidence-echo` 只用于软件链路 smoke test，不是模型或医学准确性评估。
- 最新软件测试结果与复现记录统一见[项目现状](doc/项目现状.md)，软件通过不表示路由或医学准确性通过。
