# A 工具实现验证报告（续跑）

> **2026-10-31 续跑修订**：新增 A5 Montgomery 138 例语义/轮廓核对、A12/CXAS CTR 交叉、跨工具一致性、变异自检，以及 PSPNet 修复后的工程核验。下方原始表保留此前结果；本节的结果优先。

## 当前结论（2026-10-31）

### 已完成

- 变异自检 6/6 捕获：`tool_verification/runs/20261030/mutations_retry4/summary.json`。
- A5 Montgomery：138/138 成功；右/左轮廓 Dice 中位数 0.9751/0.9730，达到 0.90 的实现核对门槛；六项语义检查 136/138 全部满足，未满足的两个病例分别为 `MCUCXR_0080_0` 和 `MCUCXR_0275_1` 的 apex 几何并列/局部次序变化。产物：`tool_verification/runs/20261030/v5_a5_montgomery_retry1/`。
- 跨工具一致性描述性诊断已完成：PSPNet/CXAS 心影 Dice 中位数 0.9358；右/左肺 0.8376/0.8951；HybridGNet/PSPNet 右/左肺 0.9056/0.8177；A26 隆突与 CXAS 气管分叉质心距离中位数 0.0478 图宽。产物：`tool_verification/runs/20261030/v5_cross_tool_retry2/`。
- PSPNet 修复后工程核验：27 接口、29 次调用、70/70 检查通过，2 次是预期拒答。多来源 12 图 312 次调用中 309 成功，3 次是 Montgomery 左右肺重叠的设计拒答。软件测试 149 passed。

### 未通过或需保留的结果

- A12 与 CXAS 原作者 CTR 交叉未达到预设 Spearman ρ≥0.8：39/40 成功，ρ=0.5806，MAE=0.0436，最大绝对差 0.1970。两者分母定义不同（本项目 CXAS 肋骨内缘 vs 作者实现肺野最宽行），因此这是“交叉核对未通过/需逐例解释”，不是医学准确性判定。产物：`tool_verification/runs/20261030/v5_ctr_crosscheck_retry4/a12_ctr_summary.json`。
- V3 A8 严格 512 帧 mask 仍有阈值像素差异；病例分数一致。
- V3 CheXOne 选项原始概率一致，但 240 个回答只有 191 个逐 token 一致；严格 V3 未通过。
- V4 A5/A8/A12/A6 变换检查未通过；PSPNet 补边/缩放和 A26 隆突通过。最坏值分别为 67.46 px、IoU 0.348、CTR 差 0.135、3 网格格。
- 以上 V3/V4 结果首先表示流程或变换一致性问题，不能写成医学准确性结论；当前仍没有 A 工具的独立医学准确性结论。

> 运行日期：2026-10-30/31。本文只报告实现验证，不报告独立医学准确性。
>
> 所有新产物均写入 `tool_verification/runs/20261030/` 的新目录；旧的中断结果保留。

## 总体结论

- V1：已通过，27 个接口 162 项检查，0 失败。
- V2：已通过，16 组随机独立参照和 8 组性质测试通过。
- V3 PSPNet：已通过，官方流程对照 14 通道 × 40 张，IoU 全部 1.0。
- V3 其余模型：A5、A26、CXAS 通过；A8 病例分数通过但严格 512 mask 有 1 个边界像素不一致；CheXOne 选项原始概率通过，但 240 个回答只有 221 个 token 序列完全一致，严格逐 token 标准未通过。
- V4：整体未通过。PSPNet 补边/缩放和 A26 隆突通过；A5、A8、A12、A6 的冻结蜕变检查未通过。结果见 `runs/20261030/v4_retry1/v4_summary.json`。
- V5：完成 Montgomery A1、JSRT A1/A2、NIH A24 的实现核对；未完成 A5 语义节点、A12/CXAS CTR 交叉实现和跨工具一致性。
- 变异自检：6/6 个局部变异探针被独立断言捕获，产物为 `runs/20261030/mutations_retry3/summary.json`；这是验证器自检，不等于完整 V1–V5 注入式测试。
- 当前全套软件测试：149 passed（`pytest -q -p no:cacheprovider`）。

## 接口结论表

| 接口 | V1 | V2 | V3 | V4 | V5 | 变异自检 | 当前结论 |
|---|---|---|---|---|---|---|---|
| `segment_lungs` A1 | 通过 | 不适用 | 通过（PSPNet 14 通道对照） | 补边/缩放通过；翻转侧别信息记录 | Montgomery 右中位 Dice 0.904、左 0.841；JSRT 右 0.915、左 0.797 | 6/6 局部探针通过 | 工程流程通过；V5 有侧别/数据口径待解释项 |
| `segment_heart` A2 | 通过 | 不适用 | 通过（PSPNet） | 缩放通过 | JSRT 心影 Dice 中位数 0.936（第三方镜像，仅实现核对） | 未做 | 实现验证通过，V5 参照为候选镜像 |
| `segment_mediastinum` A3 | 通过 | 不适用 | 通过（PSPNet） | 未单独列入 V4 | 未做 | 未做 | 实现验证通过，V5 未覆盖 |
| `segment_shoulder_bones` A4 | 通过 | 不适用 | 通过（PSPNet） | 未单独列入 V4 | JSRT 锁骨未汇总 | 未做 | 工程流程通过，V5 未完成 |
| `locate_lung_landmarks` A5 | 通过 | 不适用 | 通过，节点最大偏差 0 px/1024 | 未通过，补边/缩放最大偏差 67.46 px | 未完成 Montgomery 节点语义与轮廓 Dice | 未做 | 官方流程对齐；变换一致性未通过 |
| `detect_abnormalities` A6 | 通过 | 不适用 | CheXOne 严格逐 token 未通过；概率通过 | 未通过，缩放框最大偏差 3 网格格 | 未做 | 实现缺陷未证实；需解释生成/框坐标差异 |
| `ground_phrase` A7 | 通过 | 不适用 | 同 A6 | 未单独判定 | 未做 | 未做 | V3 共享 CheXOne 未通过 |
| `segment_pneumothorax` A8 | 通过 | 不适用 | 病例分数通过；1 个 512 阈值像素不一致 | 未通过，存在样本 IoU 0.348 | ChestX-Det 仅塌陷记录未完成 | 未做 | 严格流程验证未通过，需分析 fp16/变换行为 |
| `measure_region_extent` A9 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `measure_region_area` A10 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `measure_region_distance` A11 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `measure_ctr` A12 | 通过 | 通过 | CXAS 标签对照通过 | 未通过，变换 CTR 最大差 0.135 | CXAS 作者 CTR 交叉实现未完成 | 未做 | 确定性规则通过；模型变换核对未通过 |
| `compute_ctr_from_borders` A12 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `compare_lung_geometry` A13 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `describe_region_shape` A14 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `summarize_roi_intensity` A15 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `extract_roi_texture` A16 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `measure_roi_sharpness` A17 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `classify_roi_attributes` A18 | 通过 | 不适用 | CheXOne 严格逐 token 未通过；概率通过 | 未单独判定 | 未做 | 未做 | V3 共享 CheXOne 未通过 |
| `measure_lung_symmetry` A19 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `assign_lung_zones` A20 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `relate_lesion_to_anatomy` A21 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `relate_multiple_regions` A22 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `summarize_bilateral_distribution` A23 | 通过 | 通过 | 不适用 | 不适用 | 未做 | 未做 | 实现验证通过 |
| `classify_view` A24 | 通过 | 不适用 | CheXOne 严格逐 token 未通过；概率通过 | 缩放类别保持 | NIH 292 例准确率 0.976（训练来源内，仅实现核对） | 未做 | V3 严格逐 token 未通过，类别级核对通过 |
| `assess_image_quality` A25 | 通过 | 通过 | CXAS 标签流程对照通过 | 未通过/未设独立通过项；旋转量变化记录 | 未做 | 未做 | CXAS 流程通过，蜕变结果需解释 |
| `localize_devices` A26 | 通过 | 不适用 | 存在分数、分类、尖端、隆突和 fold 全部通过 | 隆突通过 | 未做 | 未做 | 实现验证通过 |

## 关键产物

| 内容 | 路径 |
|---|---|
| V3 主接口缓存和官方参照 | `tool_verification/runs/20261030/v3_models_retry1/` |
| V3 汇总差分 | `v3_models_retry1/v3_models_summary_retry1.json` |
| CheXOne 修复后差分 | `tool_verification/runs/20261030/v3_chexone_retry2/` |
| V4 汇总 | `tool_verification/runs/20261030/v4_retry1/v4_summary.json` |
| Montgomery V5 | `tool_verification/runs/20261030/v5_montgomery_retry1/` |
| JSRT V5（修正图像侧映射） | `tool_verification/runs/20261030/v5_jsrt_retry2/` |
| NIH ViewPosition V5 | `tool_verification/runs/20261030/v5_nih_retry1/` |

V5 数值只是模型训练/作者相关数据上的实现核对，不能写成独立医学准确性结论。V4 的失败项也不能直接等同于模型医学准确性差，它们首先表示对输入变换的稳定性未满足本项目冻结标准。
