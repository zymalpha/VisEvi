# 接手 Prompt：A 工具实现验证（继续执行）

把下面代码块完整交给接手的 AI（如 Codex）。

```text
你接手一个胸片视觉证据工具项目的“A 工具实现验证”工作。目标：证明 A1–A26 的 27 个接口实现正确——代码按定义计算，模型类工具忠实复现作者官方推理流程，坐标、侧别、单位、拒答语义正确。不是医学准确性评估。

工作根目录：/ssd/visual_reasoning/zym/MedVisEvi
验证工作目录：/ssd/visual_reasoning/zym/MedVisEvi/tool_verification/

先读（按顺序）：
1. tool_verification/README.md —— 方案（V1–V5 五级验证 + 变异自检）、逐工具设计、通过标准；第 8 节是当前进度、已修复缺陷和未决事项。方案、容差、区间都已经用户确认，不要改动；确需改动先停下来问用户。
2. tool_verification/cases/tolerances.json 与 cases/*.json —— 冻结的容差和样本清单，不得修改。
3. doc/项目介绍.md（坐标与像素约定、环境、目录）、doc/项目现状.md（A1–A26 矩阵、模型来源与许可、训练重叠）。
4. 代码：src/visevi/（被验证对象；backends/ 与 workers/ 是隔离模型推理框架，perception/ 是各工具），tool_verification/oracles/（参照实现，不得 import visevi），tool_verification/suites/（验证脚本）。

环境与硬性规则：
- 三个环境：.venv（主程序、确定性工具、PSPNet、测试）；.venv-models（模型 worker 专用，禁止安装或升级任何包——其锁文件 tools/requirements-models.lock 进入所有模型缓存绑定，改动会使缓存全部失效）；.venv-oracle（参照脚本专用，可在此安装参照所需依赖）。
- GPU 只用 4–7 号。
- 不重新下载模型权重；tools/weights/*/provenance.json 记录了逐文件 SHA256。不访问需要账号或授权的数据（Kaggle、PhysioNet、Stanford AIMI），不要索要账号。
- PyTorch 检查点一律用 torch.load(..., weights_only=True)（CXAS 仅额外放行 argparse.Namespace），不要用 weights_only=False。
- 每次运行写入 tool_verification/runs/<日期>/<步骤>/ 新目录，只追加不覆盖；失败或中断的运行目录加后缀标明（如 _INCOMPLETE_aborted、_FAILED），不得删除。
- 参照实现（oracles/）不得 import visevi，避免用同一份代码证明自己。
- 发现被验证代码有缺陷时：先写能复现的最小用例，再改 src/，补 tests/ 单测，跑 .venv/bin/python -m pytest -q -p no:cacheprovider 全绿，然后从受影响的验证步骤重跑；把缺陷、修复和证据追加到 README 第 8 节。
- 报告中区分“本轮实测”与“引用记录”；V5 区间只用于发现实现错误，不得写成准确性结论。

当前进度（详见 README 第 8 节）：
- 已完成并通过：冻结样本与容差；V1（27 接口 162 项）；V2 随机比对（16 组 × 1000）；V2 性质测试（8 组 × 300）；V3 PSPNet（改为官方预处理后 14 通道 × 40 张 IoU 全为 1.0）。
- 已修复缺陷：PSPNet 预处理/裁剪偏移与官方不一致（PREPROCESSING_VERSION=2）；A14 最小外接矩形并列不唯一；A25 旋转指数越界；CheXOne 选项概率取自惩罚后分数（改为原始 logits，并输出 token_ids）。
- 未完成：V3 其余模型（A5、A8、A26、CXAS、CheXOne）、V4、V5、跨工具一致性、变异自检、汇总报告。

请按以下顺序继续：

步骤 1：V3 模型类工具差分
- 运行 .venv/bin/python tool_verification/suites/v3_models.py --out tool_verification/runs/<日期>/v3_models --device cuda:4
- 这个套件和 oracles/{hybridgnet,pneumothorax,lines_tubes,cxas,chexone}_official.py 已写好，但从未完整跑通过，预计需要先调试（参照脚本在 .venv-oracle 运行，日志在 official/log_*.txt）。调试只允许修正脚本错误，不得放宽 tolerances.json。
- 已知需要关注的点：
  - CheXOne：作者发布的 generation_config 为 do_sample=True、temperature=1e-6、repetition_penalty=1.05，我们的 worker 强制贪心；要求生成 token 序列逐一相同，选项字母原始 logits 概率差 ≤ 1e-3。
  - A5：作者用 cv2.imread(GRAYSCALE) 解码，我们用 PIL convert('L')；要求 1024 帧节点最大偏差 ≤ 0.5 px；若只在调色板（JSRT）或 RGBA 图上超限，先单独做“解码一致性”比较（cv2 与 PIL 灰度逐像素差），再决定是否改 worker 的解码以对齐作者，并记录。
  - A8：我们保存的概率图是 fp16，概率比较只能到 fp16 精度；判定以病例分数 |差| ≤ 1e-4 与 512 帧阈值 mask 完全一致为准。
  - A26：存在分数与分类 |差| ≤ 1e-4，ETT 尖端与隆突坐标、fold_ids 完全一致。
  - CXAS：159 个标签在原图尺寸上 IoU ≥ 0.999（预期完全一致）。

步骤 2：V4 变换一致性
- 运行 tool_verification/suites/v4_metamorphic.py（已写好、未运行）：10 张 1024 方图，做上下补黑边（PSPNet 必须完全一致，IoU = 1）、2 倍放大、左右翻转。
- 先检查脚本逻辑与 README 第 2 节、tolerances.json 的 V4 项是否一致。PSPNet 翻转后 Right_Lung 与镜像 Left_Lung 的 IoU 只作信息记录（说明侧别来自位置还是解剖）。

步骤 3：V5 预期表现核对（新写 suites/v5_expected.py）
- Montgomery 138 例：
  - A1 PSPNet 左右肺 Dice，映射为 Montgomery leftMask ↔ Right_Lung、rightMask ↔ Left_Lung（leftMask 在图像左侧）；只在 PSPNet 中心裁剪视野内计分，视野外 GT 丢失比例单列；中位数 ≥ 0.85。
  - A5 把 HybridGNet 左右肺节点围成多边形，与人工 mask 比 Dice，中位数 ≥ 0.90；同时在全部 138 例上复核语义节点表（0/21/29/44/65/71）是否为对应几何极值，报告一致比例。
- JSRT SCR 镜像 mask（1024²，原图 2048²）：先确定并核对坐标映射（叠加若干例确认），再算 A1 肺 Dice 中位数 ≥ 0.85、A2 心 ≥ 0.75，锁骨只记录。
- NIH 292 例 ViewPosition：A24 准确率 ≥ 0.85；JSRT、Montgomery 全为 PA，记录误判率。
- CheXOne 框网格尺度：在 ≥ 50 个 ChestX-Det 参照框上确认 0–100 网格（已有 6 例记录：tool_verification/runs/_review_records/20261030_a_tools/chexone_box_scale.json）。
- A12：在 core40 上与 CXAS 作者自带的 CTR 提取器（tools/src/cxas/.../cxas/extraction/cardiothoracic_ratio.py）比较，口径不同不要求相等，要求 Spearman ≥ 0.8，并逐例解释大偏差。
- 跨工具一致性（只报告分布，极端不一致逐例查看）：PSPNet 心与 CXAS heart、PSPNet 肺与 CXAS lung 与 HybridGNet 肺轮廓、A26 隆突与 CXAS tracheal bifurcation。
- V5 未达区间时，先排查实现错误（归一化、通道、坐标、侧别），不得调整区间。

步骤 4：变异自检（新建 mutations/）
- 对被验证代码逐个注入典型错误（在临时副本或用 monkeypatch，不得留在 src/）：交换 x/y、mask 偏移一像素、左右通道互换、阈值 0.5 改 0.4、CheXOne 网格 100 改 1000、PSPNet 回到 v1 预处理、Feret 改回像素中心、A20 分区边界偏移。
- 每个注入都必须使至少一项 V1–V5 检查失败；仍全绿的说明该部分验证不足，需补用例。产出注入—失败检查对照表。

步骤 5：重跑工程核验并汇总
- 用新代码重跑：.venv/bin/python scripts/verify_a_tools_stepwise.py --out trials/a_tools_verification_<日期> --device cpu --model-device cuda:4，以及 .venv/bin/python scripts/run_a_tools_multisource.py --out trials/a_tools_multisource_<日期> --per-source 3 --model-device cuda:5。旧结果是 PSPNet 预处理修改前的，已过时。
- 在 tool_verification/reports/ 写每个接口一行的结论表：接口 | V1 | V2 | V3 | V4 | V5 | 变异自检 | 结论；结论只能是“实现验证通过”“实现缺陷（已修复/未修复）”“不适用/缺参照（写原因）”。
- 更新 tool_verification/README.md 第 8 节、doc/项目现状.md（矩阵状态、证据、变更记录），不要新建其他文档。

最终回复给用户（中文）：每个接口的结论、发现并修复的缺陷（附复现与证据路径）、仍未解决或无法验证的项、所有偏离方案之处及原因。
```
