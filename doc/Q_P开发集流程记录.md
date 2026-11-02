# Q/P 开发集流程与集成记录

> 版本：v1.0
>
> 工作根目录：`/ssd/visual_reasoning/zym/MedVisEvi`
>
> 范围：记录 Q1-Q4、P1-P12 的 runner 集成和 400 张开发批次流程结果；正式准确性评估仍待单独冻结协议和参照。

## 1. 范围和完成定义

本阶段只做 Q/P 两类功能。A 工具是输入证据来源，Q/P 程序负责组合和解释条件；不在本阶段实现 E 类消费模型，也不把组合输出包装成未经验证的临床诊断。

每个功能完成必须同时满足以下条件：

1. 对应 `FunctionSpec` 能通过 schema、依赖图和注册表校验。
2. 统一 runner 能按依赖顺序执行所有 required step，并记录 optional step 缺失。
3. 每个输出都是 Observation、Relation/Measurement 或 EvidenceClaim，保留工具、图像、mask/点/框和上游节点的 provenance。
4. 输入缺失、输出格式错误、图像或 mask 不一致、视野截断和不适用投照会产生结构化拒答，不能生成伪造的临床结论。
5. 同一输入、同一工具版本和同一参数重复运行时，结构化输出和证据图稳定一致。
6. 至少有一个正常路径、一个缺失依赖路径、一个错误绑定或越界路径测试；测试不只 monkeypatch 组合函数，还要经过真实 registry 和 runner。

当前组合程序的输出是原始证据，`clinical_interpretation` 保持 `not_assessed`。在独立人工参照、适用条件和阈值冻结前，不输出“确诊”“排除”或校准概率。

### 实施与运行记录（更新至 2026-11-02）

- 已将 Q1-Q4、P1-P12 的组合程序注册到统一 runner；A registry 的 27 个接口覆盖全部 Q/P required step。
- Q1 在真实 MIMIC 图像上端到端通过：`classify_view`、`locate_lung_landmarks`、`assess_image_quality` 均成功，生成原始吸气证据。
- Q3 在真实 MIMIC 图像上端到端通过：生成整图灰度、清晰度和心后对比度证据。
- Q4 在真实 MIMIC 图像上端到端通过：真实模型返回 PA 投照证据，原始选项概率保持未校准。
- Q2 在真实 MIMIC 图像上按协议拒答：肩部 mask 触碰模型内部视野边界；这是预期的截断保护，不计作工具崩溃或医学阴性。
- 真实 Q1 运行曾暴露 runner 对 NumPy landmark 缓存直接 JSON 序列化的缺陷，已改为稳定数组指纹并加入回归测试。
- Q/P 组合测试已改为使用 runner 的正式程序注册表；此前测试通过 monkeypatch 注入程序的情况已清除。
- 已修复 optional step 格式错误导致整项功能中止、P10 optional 属性未解析导致主测量丢失，以及组合程序失败留下半成品 Relation 的问题。
- 全量真实 registry PA 单图冒烟已覆盖 16 项：10 项生成 raw evidence，6 项按协议拒答；无未处理异常。拒答原因为 Q2/P11 肩部视野截断、P2/P3/P12 未生成候选 ROI mask、P9 质量/FOV 条件不满足。P8/P10 使用带 `smoke_fixture` 来源的合成矩形 ROI，只验证接口与证据绑定，不代表解剖 ROI 标注。
- P1、P4、P5、P6、P7 的真实 registry 流程成功；P4 的 optional detector 未提供可用属性时仍保留主结果。P10 在 optional A18 拒答时保留宽度和灰度结果。
- 400 张开发集聚合已完成：Q4 400/400 成功；Q1 386 成功、14 拒答；Q2 212 成功、188 拒答；Q3 399 成功、1 拒答；P1 209/191，P2 147/253，P3 222/178，P4 310/90，P5 329/71，P6 316/84，P7 378/22，P9 39/361，P11 167/233，P12 70/330。P8 0/400、P10 0/400 是因为清单没有肺门/主动脉结 ROI 输入，不是静默失败。四个 shard 无 runner exception，投照统计为 PA 275、AP 125。聚合报告：`tool_verification/runs/20261101/pq_smoke_summary.json`。

上述统计用于开发集接口覆盖和拒答分析。P1/P2/P3/P5/P12 的拒答较多来自没有候选 ROI；Q2/P4/P6/P7/P9/P11 的拒答包含视野截断或质量门控。

## 2. 当前基线

规格目录已经包含 Q1-Q4、P1-P12。`runner.py` 已注册全部 16 项组合程序；Q1-Q3、P1、P2、P4、P5、P7、P8、P10、P11、P12 的程序来自 `q_programs.py`、`p_programs_a.py` 和 `p_programs_b.py`。当前 27 个 A 接口覆盖所有 Q/P required step。

程序注册、接口接线和合成端到端契约已完成；400 张开发批次聚合也已完成。后续工作是扩大真实样本覆盖，并细分 `ok`、`refused`、`blocked`、optional 缺失和失败原因。该批次仍属于流程核对，不构成医学准确性评估。

## 3. 集成要求与执行记录

### 阶段 0：接口和依赖（已完成）

交付物：

- 校验每份 Q/P `FunctionSpec` 的 step 名称、`depends_on`、required/optional、参数占位符和输入类型。
- 给每个组合程序建立明确的 `step_id -> Observation` 输入映射，禁止按列表顺序猜测工具输出。
- 在 runner 内集中维护程序注册表；未注册程序不能被误判为“工具已完成”。
- 为每个功能定义输出关系名、证据节点类型和 raw-only 限制。

验收：

```bash
.venv/bin/python -m pytest -q tests/test_function_specs.py tests/test_function_planning.py
```

应能加载完整 26 项规格，并在任何依赖循环、未知 step、未知工具或不合法输入时在执行前失败。

### 阶段 1：Q 类程序（已接入）

先接入已有 `q_programs.py` 的 Q1-Q3，再保留现有 Q4。Q 类是元证据，主要用于限制或折扣 P 类解释，不直接对病理命题投票。

| 功能 | required steps | optional steps | 组合输出 | 接入验收 |
|---|---|---|---|---|
| Q1 吸气程度 | `classify_view`, `locate_lung_landmarks`, `assess_image_quality` | 无 | 肺底点/轮廓、膈顶、肋骨原始计数、适用条件 | frontal view、点和轮廓均在原图坐标；没有临床等级 |
| Q2 体位旋转 | `classify_view`, `segment_shoulder_bones`, `assess_image_quality` | `measure_region_extent` | 锁骨-脊柱几何、rotation index、适用条件 | 四个肩部 mask 与当前 observation 绑定；截断或未定义中线拒答 |
| Q3 曝光/对比度 | `summarize_roi_intensity`, `measure_roi_sharpness`, `assess_image_quality` | 无 | 全图灰度统计、清晰度、心后对比度、适用条件 | 统计量绑定当前完整图像；不把灰度当物理曝光或临床等级 |
| Q4 PA/AP | 现有程序和 `classify_view` | 无 | PA/AP/lateral/unknown、原始回答、状态 | 保持已通过的观测解析、unknown 和拒答测试 |

Q 类共同规则：lateral/unknown 不得被当作 frontal；任何临床 grade 在没有冻结映射时都返回 `not_assessed`；Q 类中的质量值作为条件节点进入后续 P 类，但不能重复计为病理支持证据。

### 阶段 2：P 类测量和病例证据（已接入）

按依赖复杂度分三批接入：

**P1/P2/P4/P5：** 先接入已有的外观、候选区域、质量和肺部几何程序。

| 功能 | required evidence | 主要输出 | 必须拒答的情况 |
|---|---|---|---|
| P1 实变/渗出 | consolidation 候选、ROI texture/intensity、consolidation 与 air-bronchogram 属性 | 病例证据、候选框、外观和属性 | 候选 mask/框不是当前 detector 的输出；属性缺失或答案未解析 |
| P2 间质改变 | ILD 候选、texture、reticular-opacity 属性 | 限定表型的原始证据 | 不把 broad ILD 自动等价为 reticular opacity |
| P4 肺气肿/过度充气 | A5 肺底轮廓、Q1 原始计数、双肺 intensity、肺 mask | 几何量、外观证据、适用条件 | 肺 mask 为空/截断；吸气上下文缺失；不输出疾病等级 |
| P5 胸腔积液 | 肺底 landmark、effusion 候选、intensity/sharpness | 肋膈角和候选区域原始证据 | 角点缺失、候选与 ROI mask provenance 不一致 |

**P7/P8/P10：** 再接入解剖区域测量。调用方提供的 ROI 必须有图像 hash、坐标系和 observation owner，不能把任意同名 mask 当成真值。

**P11/P12：** 最后接入骨折和皮下气肿的候选/外观/空间关系程序，重点测试多来源冲突和肺外区域关系。

| 功能 | required evidence | 主要输出 | 评估边界 |
|---|---|---|---|
| P7 纵隔宽度 | 纵隔 mask、extent/距离 | 边界、宽度和距离 | 几何宽度不自动等于临床纵隔宽度 |
| P8 肺门改变 | 肺 mask、双侧肺门 ROI、intensity/texture、双侧分布 | 双侧 ROI 与分布原始证据 | ROI 来源和专家分级尚未冻结 |
| P9 CTR | 已接入：心影、CTR、投照、质量 | CTR、条件和可用性 | 缺胸廓内缘时拒答；AP 不进入 PA 主协议 |
| P10 主动脉结 | 主动脉结 ROI、extent、intensity | 宽度、灰度、钙化候选 | 不把候选变成突出或钙化诊断 |
| P11 骨折 | 肩部骨结构、fracture 候选、sharpness | 候选框和局部外观 | 候选框不是骨折真值或断端分割 |
| P12 皮下气肿 | 肺 mask、文本/属性候选、强度、空间关系 | 病例、区域和肺外关系证据 | 缺区域参照时只返回未定的病例证据 |

### 阶段 3：证据图、缓存和拒答（已实现）

所有 Q/P 程序接入后，集中处理跨功能行为：

- Observation 只能由 A 工具产生；组合程序产生 Relation/Measurement 和 Claim。
- 组合程序不得重复执行已经由同一 session、同一图像和同一参数生成的工具；缓存命中必须记录 `reused` 和原始节点 ID。
- 共享上游（例如同一心影 mask）在证据卷宗中只计一个来源，不能因多个测量重复计数。
- Q 质量节点影响适用性和情境折扣，不直接作为 P 类疾病支持票。
- 冲突保留为 `contradicts` 关系；冲突或关键 evidence 缺失时返回 `indeterminate`/`refused`，不能平均或静默丢弃。
- 所有空间结果保留原图坐标、图像 hash、mask hash、模型版本和视野截断信息。

验收：同一 session 先运行 P9 再运行 P1/P4 时，上游工具只能调用一次；删除一个必需节点后，相关功能只生成结构化拒答，不能留下孤立 Claim。

### 阶段 4：400 张开发批次（已完成）

当前批次只用于流程验证，不作为独立医学准确率测试：

- 400 张 MIMIC-CXR JPG，294 名患者，275 PA、125 AP；
- 200 张有人工心肺 mask；
- 200 张来自 Chest ImaGenome gold-development 子集；
- 所有图片和人工 mask 已通过完整性校验。

执行口径：

1. 先按 manifest 逐例建立 `EvidenceSession`，只记录聚合计数和失败原因。
2. 对每个 Q/P 功能分别记录 `ok`、`refused`、`blocked`、`invalid_output`、`reused`。
3. 检查 PA/AP 路由、非方图视野截断、mask 绑定、坐标回映、缺少工具和不完整 evidence 的拒答。
4. 把每个功能的逐例结果写到不覆盖的 trial 目录；报告只上传聚合统计，不上传原始医学影像。

当前数据能支持的内容：Q4 路由、A1/A2 中间结果、P9 心影和几何流程、Chest ImaGenome 解剖框/关系的接口试跑，以及所有功能的运行稳定性和拒答测试。

当前数据不能单独支持：Q1-Q3 专家质量等级、病灶框/分割准确率、设备尖端位置、随访变化方向、严格 CTR（缺胸廓内缘端点）和完整 P10/P12 临床准确率。

### 阶段 5：功能准确性评估准备（待推进）

集成通过后才建立功能评估表，评估分三层：

1. **程序正确性**：合成 mask/点/框、手算值、非法输入、变换一致性和证据图核验。
2. **开发数据可运行性**：400 张数据上的覆盖率、拒答率、冲突率、运行失败类型和证据完整率。
3. **功能准确性**：针对每个功能补充独立人工参照；病例标签、框、mask、关键点和连续测量分别使用 AUROC/AUPRC、FROC/AP、Dice/IoU/HD95、像素误差、MAE/ICC/Bland-Altman 等指标。

开发数据中的训练重叠可以接受，但报告中必须明确“流程验证”，不能称为独立泛化结果。正式评估要患者级划分，冻结协议后再锁定测试集。

## 4. 后续工作

Q/P 软件集成已完成，后续工作转为补充数据条件并准备独立评估：

1. 对 P8/P10 取得带图像 hash、坐标系、来源和 owner 的真实肺门/主动脉结 ROI，再重新跑流程覆盖。
2. 对 Q2、P9、P11 继续按 FOV 和质量条件分层分析；保留截断图作为拒答用例。
3. 对 P1/P2/P3/P5/P12 单独统计无候选拒答，不把它们转换成疾病阴性。
4. 在每个程序的输入、输出 schema 与证据图契约冻结后，补人工参照和正式功能评估协议。

## 5. 当前状态

| 项目 | 状态 |
|---|---|
| Q/P 16 项 FunctionSpec 与组合程序 | 已注册统一 runner；合成集成测试通过 |
| 400 张开发批次 | 流程覆盖和拒答原因已汇总，见 `tool_verification/runs/20261101/pq_smoke_summary.json` |
| P8/P10 的真实 ROI 输入 | 当前批次缺少肺门/主动脉结 ROI，需补充绑定图像与来源的输入 |
| 26 项独立评估协议 | `draft_not_frozen`；独立参照和患者级测试设计待完成 |

400 张结果用于流程覆盖和拒答分析，不作为独立医学准确性结论。缺少适配参照的功能暂不评分。
