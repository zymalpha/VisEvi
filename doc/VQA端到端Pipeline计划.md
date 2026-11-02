# MedVisEvi：胸片 VQA 端到端 Pipeline 计划

> 更新日期：2026-11-02。本文是 VQA 线路的执行计划，建立在当前 26 项 `FunctionSpec`、依赖图、`run_function_episode()` 和 Observation/Relation/Claim 证据图之上。本文中的“跑通”只表示软件链路可执行和结果可追溯，不表示医学准确性已经通过。

> **全功能接入进展（2026-11-02）**：`run_vqa_episode()` 已允许 26 个已注册功能进入同一执行入口。Q4 保留 `q4-response-2` 专用 verifier；其他功能使用 `function-response-1`，要求模型原样返回 EvidenceCard 的完整 `function_output` 并引用绑定该输出的 Claim。`VQARequest` 新增 `inputs`、`prior_image_ref`；S1–S3 可运行双图流程，G3/P8/P10/T4 等可通过 Python API 传入预置 `EvidenceSession`。该通用 verifier 只检查结构、逐值一致和 Claim 绑定，不验证上游视觉/医学准确性。问题槽位抽取、多个功能联合回答、真实模型推理和独立医学评估仍未完成。

> 联合输入的最新决策见[图像、问题与工具证据联合输入方案](VQA图像问题与证据联合输入方案.md)：主设置采用问题+原图+相关证据卡，evidence_only 与独立 image_only baseline 分开消融。此处较早的执行细节以该方案的输入边界和当前代码状态为准。

## 1. 目标和边界

系统输入一张胸片和一个自然语言问题，输出一个有证据依据、可引用、可拒答的回答：

```text
image + question
        |
        v
问题规范化 -> 功能门控 -> FunctionSpec 校验 -> A 工具组合
                                      |
                                      v
                         Observation / Relation / Claim
                                      |
                                      v
                            EvidenceCard 确定性渲染
                                      |
     原图 + 问题 + EvidenceCard -------+
                                      v
                         本地多模态模型生成回答
                                      |
                                      v
                    引用/数值/适用条件核验 -> answer or abstain
```

VQA adapter 当前支持五类中的单一功能调用；Q4 首条纵向切片的评估方案仍保留。S1–S3 通过 prior/current 两张输入影像接入；设备 reference、外部 ROI/mask 则由调用方提供有来源且绑定图像的输入。外部知识检索和 E 类工具不在本轮范围内。

## 2. 当前起点

已经具备的基础：

| 模块 | 当前状态 | 对 VQA 的作用 |
|---|---|---|
| A 工具 | A1–A26 有正式接口；部分模型类工具仍有独立准确性评估未决项 | 产生结构化视觉观察 |
| `FunctionSpec` | Q1–Q4、P1–P12、T1–T4、S1–S3、G1–G3 共 26 份 | 规定工具依赖、输入槽位、输出、适用条件和拒答 |
| 依赖图/规则路由 | 已有闭集 `RuleRouter`，能按功能 ID/关键词路由 | 可作为门控的确定性基线 |
| `run_function_episode()` | 已能按规格执行 26 项功能并保存 episode/session | 统一工具调用入口 |
| 证据图 | Observation、Relation、Claim 和来源边已实现 | 保存工具输出、关系和结论的溯源 |
| EvidenceCard | `vqa/evidence_card.py` 为任一功能保留完整嵌套 JSON、上游节点、步骤和 provenance；成对图像记录两侧哈希 | 把功能 episode 确定性地交给回答模型 |
| VQA 适配层 | `vqa/schema.py`、`gate.py`、`runner.py` 已接入 26 项单功能；`slots` 与功能执行 `inputs` 分开校验，可传 pair image 和已有 session | 规则路由、FunctionSpec 执行和结构化拒答 |
| 功能门控 | Q4/P9 规则别名和冲突拒答已接入；当前仍是规则基线，不是已训练的 MoE | 唯一匹配才调用功能 |
| LLM consumer/verifier | MedGemma 1.5 4B、Qwen3-VL、Lingshu 有本地 Transformers consumer。Q4 使用 `q4-response-2`；其他功能使用 `function-response-1`，逐值校验 EvidenceCard 输出与 Claim 绑定。Q4 evaluator 另有独立 image-only baseline | 三种互不混池的 Q4 输入对照；其他功能的受约束结构化转述 |

VQA runner 不会从自然语言猜测所有操作参数。调用方应把 FunctionSpec 概念槽位放入 `slots`，把功能程序需要的精确参数/reference 放入 `inputs`。例如 G1/P3/S 类的 `finding` 必须明确提供；S 类还要提供 `pair_metadata` 和 prior 图；P8/P10、G3、T2–T4 需要 mask 或有来源的 reference。缺失字段由 VQA 层提前拒答，输入结构正确但 A 工具或证据不满足条件时由统一功能 runner 拒答。预置 session 只支持 Python API，不提供序列化 mask 的 CLI 通道。

历史记录（2026-11-01）：13 项 Q4 合成测试和 20 张 MIMIC 开发图 A24 + evidence-echo smoke 通过。该 smoke 只证明当时的集成链路可运行；开发元数据不是独立医学金标准。2026-11-02 更新 Q4 verifier、审计记录并接入 26 项 VQA API；当前全量 CPU 软件回归 529 项通过，不代表医学准确性。模型文件检查结果：MedGemma 4B 两个权重分片及配置通过发布哈希核对，但未加载推理；Qwen3-VL 和 Lingshu 各缺三个权重分片，Lingshu 另缺 `tokenizer.json`。下载目前已暂停；三者均未完成真实模型推理。

当前工作区已有约 400 张 MIMIC-CXR-JPG 开发图像，约 0.61 GB；它们是开发样本，不是 VQA 锁定测试集。工作区还已有 VQA-RAD 和 VinDr-CXR-VQA 的公开元数据，但 VQA-RAD 是混合模态且存在跨 split 图像重复，VinDr 元数据没有图像，因此二者只能用于数据读取和接口 smoke test。

## 3. 数据路线

### 3.1 首选数据

首个受控 VQA 评估集使用 PhysioNet 的 [Visual Question Answering evaluation dataset for MIMIC CXR](https://www.physionet.org/content/vqa-evaluation-mimic-cxr/1.0.0/)：

| split | 病例 | 问答 | 用途 |
|---|---:|---:|---|
| validation | 23 | 111 | 开发、路由和提示词调试 |
| test | 48 | 226 | 冻结后的小规模答案评估 |

其中多数问题是是/否题，覆盖投照、疾病/设备存在性、位置、大小和严重程度。参考答案由有经验的放射科医师根据胸片和报告整理。文件和对应图像需要注册用户并签署该项目 DUA；图像通过 `dicom_id`/`study_id` 与 MIMIC-CXR-JPG 对齐。下载前先核查 48 个 test 图像是否已经在现有 400 张开发图像内。

### 3.2 扩展数据

流程稳定后使用 [MIMIC-Ext-MIMIC-CXR-VQA](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/) 做规模化覆盖：约 377,391 条 QA，测试部分 500 张图、13,793 个样本，含 705 条 size 类问题。训练/验证主要是 silver 标签，测试使用 gold 标签；CTR 与 MTR 问题必须分开筛选。该数据适合覆盖压力和路由统计，不能单独替代人工 CTR 连续测量。

### 3.3 数据使用规则

1. 受限 MIMIC 图像和标注只在授权服务器本地处理，不发送到在线 API。
2. 原始图像、患者/检查 ID、VQA 标注和逐例输出不提交 GitHub。
3. 所有下载保存版本、来源、SHA256、授权范围和匹配失败记录。
4. 训练来源重叠只影响独立性声明，不阻止当前的流程开发；正式医学结论需要患者级划分和训练来源审计。
5. VQA 参考答案评估的是最终回答；P9 的连续 CTR 准确性仍需另建同一测量口径的人工心界和胸廓内缘标注。

## 4. 首条垂直切片：Q4 投照识别

### 4.1 为什么先做 Q4

Q4 只依赖 `classify_view`（A24），链路短，能先验证问题解析、功能选择、FunctionSpec 绑定、工具执行、证据渲染、LLM 回答和引用核验。MIMIC 元数据还提供 PA/AP 视角字段，可用作工具和 VQA 答案的开发对照。

Q4 的问题示例：

```text
问题：这张胸片是 PA 还是 AP 投照？
路由：Q4
工具：classify_view
回答约束：只能输出 PA、AP、lateral 或 unknown，并引用 view 证据。
```

Q4 的 `unknown` 是合法结果；低置信度或冲突时必须拒答，不能猜测 PA/AP。

### 4.2 Q4 数据流

1. `VQARequest` 接收 `image_ref`、`question`、数据集样本 ID 和运行配置。
2. 输入检查图像存在、格式可读、来源允许本地处理。
3. 规则 Gate 产生 Q4 候选、路由理由和是否 abstain；当前没有校准置信度。
4. 将路由与 Q4 `FunctionSpec` 校验，拒绝未实现功能和未声明槽位。
5. 调用 `run_function_episode(function_id="Q4", image_ref=..., query=...)`。
6. 从证据图中取 view observation、view relation 和 view claim。
7. 用新的 EvidenceCard renderer 输出稳定的 JSON 和文本视图，保留嵌套字段、来源、版本、适用条件和拒答原因。
8. 主实验将 EvidenceCard、原问题、原图和严格回答格式交给本地多模态 consumer；evidence_only 作为独立消融组。所有模式分别标记，不能合并指标。
9. Verifier 检查回答是否属于允许集合、是否引用存在的 claim、是否与 claim 方向一致。
10. 输出最终回答、证据引用、路由结果、工具状态和完整 trace。

### 4.3 Q4 验收条件

- 同一输入和同一版本得到相同的 route/spec hash、工具结果和 EvidenceCard。
- 明确的 PA/AP 问题只路由到 Q4；模糊问题返回 `ambiguous` 或要求澄清。
- 工具失败、图像缺失、未知投照和证据缺失均产生结构化 abstain。
- LLM 不能输出证据卡没有的投照类别；Verifier 能抓住错误类别、无效引用和额外诊断内容。
- validation 集完成逐例 JSON 记录；test 集在协议冻结前不得用于提示词、阈值或路由调参。

## 5. 第二条垂直切片：P9 CTR

P9 已按第二条垂直切片接入 VQA runner，用来验证多工具组合。以下记录原定执行链和仍需评估的条件：

```text
Q：心胸比是多少？心影是否增大？
 -> P9
 -> A24 投照 + A25 质量 + A2 心影 + A12 CTR（胸廓内缘由 CXAS 肋骨分割）
 -> CTR、心宽、胸廓内宽、投照和质量条件
 -> 条件化 Claim
 -> EvidenceCard
 -> LLM 联合查看原图与证据；测量值只引用有效工具 Claim
```

P9 必须把“测量值”和“临床解释”分开：

- PA/AP 或质量条件不满足时，可以返回几何测量不可用/条件不足。
- 没有人工同口径 CTR 数值时，只评估流程、答案格式和证据一致性。
- 不能把 MIMIC-Ext 的程序生成 CTR/MTR 答案当作人工连续 CTR 金标准。
- P9 正式评估需要人工心界、胸廓内缘端点和投照适用条件；主指标为 CTR MAE、偏差和一致性界限。

P9 验收重点是共享上游工具不重复运行、心肺掩膜坐标一致、FOV 截断时拒答、数值单位完整、EvidenceCard 不丢失嵌套测量字段，以及模型不能把“有 CTR 数值”改写成未经协议支持的心脏病诊断。

## 6. 应用层 MoE Gate 设计

这里的 MoE 是功能选择层的 mixture-of-experts，不要求底层医学语言模型本身采用 MoE 架构。第一版按可审计优先设计为两阶段：

### 6.1 阶段 A：规则基线

当前闭集 `RuleRouter` 已通过 `RuleFunctionGate` 接入 VQA，提供 Q4 投照和 P9 CTR 的规则别名、候选冲突拒答和 FunctionSpec 目录校验。当前路由不输出未校准分数：

```json
{
  "question": "这张胸片是 PA 还是 AP？",
  "intent": "projection_classification",
  "function_candidates": [
    {"function_id": "Q4", "reason": "curated_alias", "score": null}
  ],
  "selected_function_ids": ["Q4"],
  "reason": "one curated function alias matched",
  "matched_by": "alias",
  "gate_version": "rules-1.0",
  "abstain": false
}
```

规则基线必须支持显式功能 ID、关键词、同义词、问题类型和冲突检测。没有唯一候选时保留 `ambiguous`，不强行调用多个功能。

### 6.2 类别 Gate：已实现接口，训练待定

五类 Gate 网络和可选 VQA 适配已实现，训练与校准暂缓，详见[五类 Gate 网络调研与实现](五类Gate网络调研与实现.md)：

- Q、P、T、S、G 各有一个参数独立的二层 MLP，默认隐藏宽度 128；保留单层线性模式作对照。
- 共享问题向量输入，五个头分别输出一个 logit，独立 Sigmoid 与阈值选择类别；允许多选或全不选，不把 26 项功能作为平坦分类输出。问题编码器尚未选型，当前通过接口注入。
- 类别选择后，类内功能仍由规则和 FunctionSpec 目录约束；网络不输出槽位或医学答案。`CategoryFunctionGate` 可注入现有 runner；单功能 VQA 遇到多个类别明确拒绝执行并保留原始选择。
- 未配置已训练权重时返回 `untrained`，不以随机分数执行工具。默认规则 Gate 不变，部署结果保留五类分数、阈值、版本和拒绝原因，未校准分数不称为置信度。
- 标注路由监督或最终回答 loss 监督尚未选择，本次不实现训练；部署的硬选择和外部工具链也尚不可端到端反向传播。训练、校准与锁定测试集协议随训练方案另定，测试参考答案不得进入路由输入。
- 后续应分别评估五类多标签路由的 macro/micro-F1、集合精确匹配、覆盖率、拒绝率，再评估类内功能选择，避免用一个 26 类 top-1 指标混合两层错误。

Gate 不直接生成医学答案，也不改变 A 工具输出。它只决定“调用哪些已经登记的功能”。所有 gate 输出必须进入 episode trace，便于定位路由错误、工具错误和 LLM 消费错误。

## 7. FunctionSpec 和工具执行层

VQA adapter 不重新实现工具逻辑。当前 26 项均能进入 `run_function_episode()`；没有通用的自然语言槽位抽取器。`slots` 按选中 FunctionSpec 校验，`inputs` 按该功能引用字段和运行时输入合同校验。未声明槽位、缺少必需输入或输入类型不符时，在 A 工具执行前结构化拒答：

| 输入字段 | 作用 |
|---|---|
| `function_id` | 由规则 Gate 从显式编号或登记关键词唯一选出 |
| `image_ref` | 本地图像路径；也可传入与其绑定的现有 session |
| `slots` | JSON 对象；字段必须出现在对应 FunctionSpec 的语义槽列表 |
| `inputs` | JSON 对象；功能执行参数/reference，按选中规格和运行时合同校验 |
| `prior_image_ref` | S1–S3 的历史影像；也可在 Python API 传入 prior session |
| `query` | 原始问题，写入 Demand 和 trace |

执行层继续使用当前 `run_function_episode()` 的新输出目录约束、缓存绑定、工具 provenance、拒答和 session/episode 保存逻辑。VQA 层不能绕过 registry 直接调用模型，也不能把缓存目录中的旧 mask 当作未经绑定的人工输入。G3/P8/P10/T4 等使用 caller mask 的任务，必须通过已绑定当前影像、带来源 Observation 和 mask metadata 的 `EvidenceSession` 进入 API。

## 8. 证据组织和 EvidenceCard

### 8.1 三层证据

- **Observation**：工具直接产生的事实，例如 `view=PA`、`heart_width_px=...`、候选框坐标或 mask 来源。
- **Relation**：确定性计算或空间关系，例如 `heart_width / thoracic_width = CTR`、候选区域位于右下肺区。
- **Claim**：功能层允许对外表达的结论，例如“投照判断为 PA”或“CTR 可报告，但临床增大解释条件不足”。

每个节点必须保存 `source_ids`、工具/模型版本、图像哈希、坐标系、单位、适用条件和置信度语义。多个工具给出冲突时，EvidenceCard 同时呈现支持、冲突和拒答状态；不能用平均值隐藏冲突。

### 8.2 EvidenceCard 最小结构

```json
{
  "card_version": "1.0",
  "function_id": "Q4",
  "question": "这张胸片是 PA 还是 AP？",
  "status": "answerable",
  "claims": [
    {
      "claim_id": "claim-...",
      "statement": "投照判断为 PA",
      "value": "PA",
      "support_observations": ["obs-..."],
      "conditions": ["frontal_projection_only"]
    }
  ],
  "measurements": [],
  "conflicts": [],
  "refusal": null,
  "provenance": {"image_sha256": "...", "spec_hash": "..."}
}
```

renderer 必须保留嵌套字典和列表，输出 JSON 与供模型阅读的文本视图；文本视图只能由 JSON 确定性渲染，不能由 LLM 重新总结。

## 9. 医学 LLM 消费层

当前并列比较三种本地候选，不预先把任何一个称为最优：

| 模型 | 定位 | 本地模型目录 | 备注 |
|---|---|---|---|
| Qwen3-VL 8B Instruct | 通用多模态基线 | `/ssd/visual_reasoning/zym/models/qwen3-vl-8b-instruct` | Transformers 4.57.1 支持；Qwen3-VL 不专门面向医学 |
| Lingshu 7B | 医学多模态模型 | `/ssd/visual_reasoning/zym/models/lingshu-7b` | 基于 Qwen2.5-VL，MIT；直接胸片和证据消费都需要实测 |
| MedGemma 1.5 4B IT | 医学多模态模型 | `/ssd/visual_reasoning/models/medgemma-1.5-4b-it` | 官方实际型号为 4B；训练源包含 MIMIC-CXR，MIMIC 图像结果要披露重叠风险 |

三个 consumer 的正式 VQA 路径支持两种分开的输入条件：`image_and_evidence` 同时传入原图与证据卡，是主设置；`evidence_only` 只发送证据卡，是消融组。evaluator 另提供独立 `image_only` Q4 baseline，只向 consumer 发送问题和原图，不运行工具链、不发送 EvidenceCard，并直接按闭集标签评分；它不经过 Claim verifier。三种条件不得合并指标。Q4 verifier 接受版本化 `q4-response-2`：闭集 `answer`、Claim ID 列表、枚举限制代码以及 `visual_assessment`（`consistent/conflict/uncertain/not_assessed`）。自由文本不能藏入额外医学断言；但 verifier 只能检查结构、引用与模型声明，不能独立确认模型是否诚实地识别图像冲突。逐例审计记录保存原始 consumer 输出、解析结果、verification、最终响应、EvidenceCard/prompt 哈希、模型与输入配置；每个样本使用独立私有输出目录。evaluator 分别报告原始输出匹配率及患者聚类 bootstrap 区间，不导出患者 ID。真实 VLM 使用独立 `.venv-vqa`，固定依赖见 `tools/requirements-vqa.lock`；权重只从本地路径加载，BF16 推理限制在 GPU 4–7。MedGemma 默认路径由 `VISEVI_MEDGEMMA_MODEL` 或当前服务器本地目录配置。

消费提示词固定为：

1. 联合输入模式同时查看原图和 EvidenceCard；Claim 是带来源的预测证据，不是图像真值。
2. 结论引用已有 Claim；不得把模型自行观察伪装成已核验的工具事实，也不得补充病史或诊断。
3. 先判断 `status`；`refused`/`insufficient` 必须明确说明无法判断。
4. 每个事实后引用 claim ID；数值必须保留单位和精度。
5. 输出结构固定为 `schema_version`、`answer`、`citations`、`limitations`、`visual_assessment`；`limitations` 只允许注册代码，不接受自由文本。

LLM 只负责语言化和有限的回答组织，不负责重新选择未授权工具，不负责修改工具数值，也不负责把低置信度观察升级为确定诊断。

## 10. Verifier 和评估指标

每条样本保存五类结果：

1. **路由结果**：功能是否选对、是否合理拒答。
2. **工具结果**：每一步成功、拒答、异常、耗时和 provenance。
3. **证据结果**：图是否有效、关系是否可重算、claim 是否有来源。
4. **消费结果**：回答、引用、数值和格式。
5. **参考对照**：VQA gold answer、元数据标签或人工测量，注明粒度和来源。

Q4 首轮指标：

| 层 | 指标 |
|---|---|
| Gate | route accuracy、macro-F1、ambiguous/abstain rate |
| 工具 | PA/AP/unknown macro-F1、coverage、拒答率 |
| EvidenceCard | claim support rate、完整性、重放一致性 |
| LLM | normalized exact match、允许同义答案准确率、引用有效率、恰当拒答率 |
| 系统 | end-to-end answer accuracy、conditional accuracy、失败率、延迟 |

P9 增加 CTR MAE、偏差、Bland–Altman 界限、ICC 和临床解释覆盖率。所有指标都保留拒答和失败分母，并按 PA/AP、质量状态和数据来源分层。

## 11. 分阶段实施计划

| 阶段 | 工作内容 | 交付物 | 通过条件 | 当前状态 |
|---|---|---|---|---|
| VQA-0 | 锁定数据版本、授权、图像匹配和问题 schema | 已有 400 张开发图 manifest；PhysioNet VQA 访问/匹配待完成 | 标注与图像按 ID 一对一匹配；不含未授权文件 | 部分完成 |
| VQA-1 | 实现 VQARequest、规则 adapter 和路由结果 schema | `vqa/schema.py`、`gate.py`；Q4/P9 规则基线完成 | 未知、歧义和未声明槽位可拒答 | 完成基线 |
| VQA-2 | 接入 `FunctionSpec` 校验和 `run_function_episode()` | 26 项单功能统一 runner；S1–S3 支持配对影像 | route、工具、Claim 和拒答进入 episode | 全功能 Python API 已接入；自然语言输入抽取待做 |
| VQA-3 | 实现 EvidenceCard renderer 和 verifier | Q4 使用 `q4-response-2`；其余功能使用 `function-response-1`；EvidenceCard 保存完整输出及 Claim provenance | 精确输出、有效 Claim、路径脱敏和逐例审计可验证 | 工程实现已接入；当前全量 CPU 回归 529 passed |
| VQA-4 | 测试本地 LLM consumer/verifier | 三种 consumer 均有本地适配器；权重状态与结构哈希记录在上文 | 完成三模型生成和 verifier 实测 | 适配器存在；Qwen/Lingshu 权重不全，三模型真实推理尚未完成 |
| VQA-5 | 跑 Q4 development；冻结 prompt、阈值和统计脚本 | 20 张 A24 + evidence-echo smoke 是 2026-11-01 历史结果；evaluator 支持 image-only/evidence-only/image+evidence 三条件、逐例审计与患者聚类区间 | 开发集对照可复现；不把 smoke 当准确率；检查完整拒答分母 | 软件入口和回归完成；需完整模型权重后再运行真实 VLM 对照 |
| VQA-6 | 跑 Q4 锁定 test 并完成回答评估 | 待 PhysioNet VQA 标注访问及图像匹配 | 逐例结果、指标、覆盖率、拒答和限制齐全 | 未开始 |
| VQA-7 | 完成 P9 VQA 与多工具证据对照 | P9 已进入 VQA API；合成 Harness 证明工具、卡片和 Claim verifier 闭环 | FOV、投照、质量和单位检查通过；数值可重算；再跑真实图与模型对照 | 软件链路已接入；真实 VLM/独立参照评估未开始 |
| VQA-8 | 获取人工同口径 CTR 标注并做独立评估 | 未开始 | 报告 MAE/偏差/一致性，独立性和训练重叠清楚 | 未开始 |
| VQA-9 | 评估全功能 Gate 和其余 Q/P/T/S/G 功能 | 26 项 API 接入完成；无通用槽位抽取器、Gate 权重、功能级 VQA 数据与独立参照 | 路由错误与工具错误分开统计，覆盖和拒答稳定 | 接口已接入；系统评估未开始 |

## 12. 当前模块与运行入口

当前实现位于以下模块：

```text
src/visevi/vqa/
  schema.py          # VQARequest、RouteDecision、VQAResponse
  gate.py            # FunctionSpec 闭集规则路由和别名
  category_gate.py   # 五类类别路由契约与独立阈值
  gate_network.py    # 五个独立线性/MLP Gate；训练暂缓
  category_function_gate.py # 类别 Gate 到现有单功能 VQA 的可选适配
  evidence_card.py   # 保留嵌套结构的确定性 JSON/text renderer
  consumer.py        # CallableConsumer 与三种本地 Transformers VLM consumer
  verifier.py        # Q4 专用核验与其他功能的结构/Claim 绑定核验
  runner.py          # 26 项单功能 VQA episode -> answer/abstain 统一入口
scripts/
  run_vqa.py         # 任一功能的通用单样本入口
  run_vqa_q4.py      # 单样本 Q4 运行；echo 模式仅作流程 smoke
  evaluate_vqa_q4.py # 私有开发清单的批量 Q4/A24 smoke 与聚合指标
tests/
  test_category_gate.py
  test_vqa_pipeline.py
  test_vqa_evaluator.py
```

Q4 仍有专用 runner/evaluator；通用单样本入口 `scripts/run_vqa.py` 可选择 26 项中的任一功能，Q4/P9 的批量 evaluator 不会自动扩展为全功能评估。使用 `--function-id`、`--inputs-json`、`--slots-json` 和必要的 `--prior-image` 提供接口输入；`evidence-echo` 只验证 EvidenceCard 到 verifier 的软件链路，不调用语言模型，也不评估医学准确性：

```bash
PYTHONPATH=src .venv-vqa/bin/python scripts/run_vqa_q4.py \
  --image /path/to/local-chest-xray.jpg \
  --out-dir /path/to/new-private-run \
  --consumer qwen3-vl --input-mode image_and_evidence --device cuda:4
```

通用单例运行示例（P9）：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_vqa.py \
  --image /path/to/local-chest-xray.jpg \
  --question "Report the measured cardiothoracic ratio" \
  --function-id P9 --out-dir /path/to/new-private-run \
  --consumer evidence-echo
```

批量评估同一份 development manifest 的示例：

```bash
PYTHONPATH=src .venv-vqa/bin/python scripts/evaluate_vqa_q4.py \
  --output /path/to/new-private-eval --limit 20 \
  --consumer lingshu --input-mode evidence_only --device cuda:4
```

不要先训练 Gate，也不要下载整个 MIMIC；按授权先完成 VQA 标注小集，再冻结评估清单。

## 13. 风险和处理

| 风险 | 处理 |
|---|---|
| PhysioNet 403 或 DUA 未完成 | 保留 manifest 和下载脚本；使用本地开发图做接口 smoke，答案评估标为 pending |
| VQA 标注与现有图像不重合 | 按白名单补下载匹配图像，不递归下载全库 |
| VQA 问题没有对应的 A 功能 | Gate 返回 unknown/ambiguous；明确提供功能输入或新增 FunctionSpec 后再接入 |
| 模型训练与 MIMIC 重叠 | 标为 development/same-source；不宣称外部泛化 |
| EvidenceCard 丢字段 | 用结构化 JSON schema 和嵌套字段回归测试阻断发布 |
| LLM 幻觉或引用错误 | 强制枚举/数值/claim ID verifier；错误时返回受限回答或拒答 |
| P9 没有人工 CTR | 只报告流程和 VQA 对齐，不报告连续测量准确性 |
| GPU 资源冲突 | 推理只使用 GPU 4–7，先用 4B 单卡调通，再评估 27B 多卡 |

## 14. 完成定义

Q4 线路完成必须同时满足：

- 一条命令从本地图像和问题生成最终回答；
- route、FunctionSpec、工具、证据图、EvidenceCard、LLM 输出和 verifier 结果均可追溯；
- 明确错误、缺失、未知和冲突都能结构化拒答；
- 结果可在新目录重放，不覆盖旧运行；
- 开发、验证和测试集用途分开；
- 报告把工具错误、路由错误、证据错误和 LLM 消费错误分开；
- 没有把流程通过写成医学准确性结论。

全功能 VQA 软件接口已接入。后续工作转为 Q4/P9 真实模型对照、功能所需数据与独立参照、功能级评估协议，以及缺失的自然语言槽位抽取和多功能请求策略。
