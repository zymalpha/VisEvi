# A 类工具详细说明与评估矩阵

本文按项目正式能力编号说明 A1–A26 的用途、实现后端、输出形态、输出字段和目前的评估状态。工具编号和实现以 `src/visevi/eval/runner.py:build_a_registry()`、`src/visevi/perception/` 为准；历史兼容接口不计入本表。

## 如何理解接口数量和评估状态

项目有 **27 个正式 A 接口，对应 26 类能力**，因为 A12 同时有两个不同用途的接口：`measure_ctr` 用 CXAS 结构预测自动定位胸廓内缘并计算 CTR；`compute_ctr_from_borders` 只计算给定边界的比值，不负责胸廓内缘定位。旧版 `build_registry()` 中的 8 个兼容接口不属于正式 A 工具清单。

本表中的输出形态指调用方得到的机器可读内容。多数接口把主要结果作为 `EvidenceUnit`（Observation）写入 `EvidenceSession`；分割或候选定位工具还会将 mask 保存在会话运行时缓存，供其他工具组合使用。表中“图像”表示这种可复用的空间产物，不表示接口一定直接返回一张可视化 PNG。

评估状态严格区分三类工作：

- **工程验证**：检查接口契约、拒答、坐标/单位、缓存绑定、确定性计算和边界行为是否符合实现约定。
- **模型流程核验**：把本项目模型接口与作者流程对照，或在已有数据上作流程/塌陷检查；它不自动构成独立医学准确性结果。
- **独立医学准确性评估**：使用与相关模型训练来源隔离、具有适当人工参照的数据，对患者级锁定测试集进行评估。当前**尚无任何 A 工具完成此类评估**。

## A1–A26 工具矩阵

| 编号 | 正式接口 | 功能 | 后端 / 算法 | 输出形态 | 输出内容 | 评估状况 |
|---|---|---|---|---|---|---|
| A1 | `segment_lungs` | 左右肺分割 | TorchXRayVision ChestX-Det PSPNet；两个肺通道 | 左右二值 mask + Observation 数值 | `Right_Lung`、`Left_Lung` mask；像素面积、最大行宽、bbox、坐标系和视野截断元数据 | V1 接口契约通过；V3 PSPNet 官方流程 14 通道×40 图的 IoU 均为 1.0。Montgomery/JSRT 有实现核对 Dice，但涉及训练来源、轮廓口径或第三方 mask 限制；无独立医学准确性评估。 |
| A2 | `segment_heart` | 心脏/心影分割 | 同 A1 PSPNet | 二值 mask + Observation 数值 | `Heart` mask、面积、bbox、视野信息，供 CTR 等下游计算 | V1、V3 PSPNet 官方流程通过；JSRT 第三方候选心影 mask Dice 中位数约 0.936，仅作实现核对，不是独立测试；无独立医学准确性评估。 |
| A3 | `segment_mediastinum` | 纵隔分割 | 同 A1 PSPNet | 二值 mask + Observation 数值 | `Mediastinum` mask 及其面积/bbox 等基础区域信息 | V1、V3 PSPNet 官方流程通过；无独立纵隔人工参照或医学准确性评估。 |
| A4 | `segment_shoulder_bones` | 锁骨与肩胛骨分割 | 同 A1 PSPNet | 四张二值 mask + Observation 数值 | 左右锁骨及左右肩胛骨 mask、区域边界和视野元数据 | V1、V3 PSPNet 官方流程通过；JSRT 候选锁骨 mask 仅用于核对记录，侧别/来源尚不足以作正式 GT；无独立医学准确性评估。 |
| A5 | `locate_lung_landmarks` | 肺尖、肋膈角/心膈角关键点及肺底轮廓 | CheXmask-U HybridGNet `v1_skip` | 关键点坐标、折线、完整节点数组；保存于 Observation/会话 landmarks | 六个语义点、右/左肺底折线、120 个肺/心轮廓节点；原图像素坐标、节点索引及可选采样离散度 | V3 与作者流程节点差分通过；V4 变换一致性未通过。Montgomery 轮廓和语义检查是作者相关数据上的核对；HybridGNet 训练数据包含 JSRT/Montgomery，不能据此宣称独立准确性。 |
| A6 | `detect_abnormalities` | 固定词表异常候选定位 | CheXOne（Qwen2.5-VL-3B）grounding | 文本解析结果 + 候选框坐标 + 框并集 mask + Observation | 每个 finding 的 detected/not_detected/unparsed、原始文本、0–100 网格框及原图像素框；候选 mask 是框近似，不是分割 | 接口/解析有测试；V3 选项概率差约 2.6e-5，但严格 token 一致性未通过；V4 缩放/框一致性未通过。ChestX-Det 等来源存在训练重叠，不能作独立准确性结果。 |
| A7 | `ground_phrase` | 自由短语对应区域定位 | CheXOne phrase grounding | 短语文本、候选框、框并集 mask、Observation | 输入短语、解析状态、一个或多个原图像素框、原始回答及 Phrase_Region 框近似 mask | 解析和坐标回映有工程测试；CheXOne 严格 token 对照未通过。公开样例只够格式开发，缺训练来源独立的文本-区域参照；无独立医学准确性评估。 |
| A8 | `segment_pneumothorax` | 气胸分割 | `ianpan/pneumothorax-cxr` EfficientNetV2-S U-Net | 病例原始分数 + 概率图文件 + 二值候选 mask + Observation | case raw score、阈值、最大像素分数、面积、bbox、原图帧映射；分数未校准 | V3 病例分数差分通过，但严格 mask 有阈值像素差异；V4 变换一致性未通过。模型训练来源未确认，尚无可确认独立的气胸 mask 评估。 |
| A9 | `measure_region_extent` | 区域范围/直径 | 确定性像素几何；像素单元边界约定 | 结构化数值 Observation | bbox 宽高、最大行宽及位置、最大 Feret 直径；可按调用方提供的行列 spacing 计算物理值 | V1、V2 工程验证通过：手算/独立几何 oracle、随机样本和变换性质核对。它本身不预测病灶；医学测量误差取决于输入 mask，无单独医学准确率。 |
| A10 | `measure_region_area` | 区域面积 | 确定性前景像素计数 | 结构化数值 Observation | 像素面积；可选按提供的 spacing 计算面积及记录单位来源 | V1、V2 确定性计算验证通过；属于几何算子，不是独立影像预测。没有单独医学准确率；需评估上游 mask 对面积的误差传播。 |
| A11 | `measure_region_distance` | 两区域间距 | 确定性像素单元几何 | 结构化数值 Observation | 两区域像素单元 gap distance、最近前景像素中心距离；可选按 spacing 换算 | V1、V2 通过；与 A22 的像素单元间隙口径已统一。医学意义依赖区域定义和上游 mask/点，不单独做模型准确率评估。 |
| A12 | `measure_ctr` | CTR 比值（自动结构定位） | CXAS `UNet_ResNet50_default` 预测心脏相关结构/肋骨，再由确定性规则找胸廓内缘 | CTR 数值及边界来源、位置和限制 Observation；CXAS mask 作为缓存产物 | 默认以 A2 `Heart` mask 的横向范围作心宽；以 CXAS 肋骨内缘最宽行作胸廓宽，输出 CTR、分子分母和算法口径；不判断 PA/AP | 算术与确定性规则通过相应工程检查；V4 CTR 变换一致性未通过。与 CXAS 作者 CTR 的相关性核对未达预设 ρ≥0.8，且口径不同待解释；无人工端点独立医学评估。 |
| A12 | `compute_ctr_from_borders` | CTR 比值（给定边界计算） | `geometry.ctr_from_borders` 确定性比例计算 | 结构化数值 Observation | 使用调用者给定的心脏左右边界和胸廓左右内缘计算宽度与 CTR；默认心界也可取 `Heart` mask 外缘；明确标注胸廓内缘未定位、实现为 partial | 70/100 等合成边界算例和范围拒答属工程核验；它核对的是比例公式，不验证边界来源或医学准确性。无独立医学准确性评估。 |
| A13 | `compare_lung_geometry` | 左右肺面积/宽度比较 | 确定性 mask 统计和左右差值计算 | 数值 Observation | 左右肺面积、最大行宽及差异/比值等几何比较量 | V1、V2 工程验证通过，含左右交换与不适用拒答检查；结果受 A1 mask 质量影响，无独立医学准确性评估。 |
| A14 | `describe_region_shape` | 区域形状描述 | 确定性 Shapely/像素单元几何和连通域分析 | 数值与枚举 Observation | 周长、圆度、solidity、凸包相关量、定向长宽比和组件信息 | V1、V2 工程验证通过；已固定并列最小外接矩形的选择规则。若将其用于病灶形态判断，仍需独立病灶轮廓/标签，当前无医学准确性评估。 |
| A15 | `summarize_roi_intensity` | ROI 灰度统计 | NumPy 灰度统计 | 数值 Observation | mask 或全图范围内的均值、标准差/方差、分位数、像素数及解码信息 | V1、V2 工程验证通过，覆盖原生高位深统计和边界输入。灰度未等同于曝光剂量或疾病证据；无独立医学准确性评估。 |
| A16 | `extract_roi_texture` | ROI 纹理 | NumPy 稀疏 GLCM：固定 bin width 16、四个对称方向 | 数值/特征向量 Observation | 六项二维 GLCM 特征及方向汇总；要求指定 ROI mask | V1、V2 与独立实现/scikit-image 参照通过。纹理特征不是诊断准确率；要验证临床表型须先定义标签并用独立参照。 |
| A17 | `measure_roi_sharpness` | ROI 清晰度/模糊代理 | OpenCV Sobel Tenengrad 确定性计算 | 数值 Observation | ROI 或全图的梯度能量/清晰度分数及强度缩放信息 | V1、V2 工程验证通过，含卷积参照和模糊响应检查；没有人工质量等级时不应称为图像质量准确率。 |
| A18 | `classify_roi_attributes` | ROI 异常属性判别 | CheXOne 全图 + ROI bbox 放大裁剪的多模态问答 | 分类文本/枚举 + 原始选项分数 Observation | 对冻结属性输出 present/absent/uncertain/unparsed、原回答、选项原始概率、ROI 名称与裁剪坐标 | 接口、裁剪坐标和解析有工程测试；CheXOne 严格 token 对照未通过。ROI 外可见像素是已知限制；缺独立 ROI 标签，无独立医学准确性评估。 |
| A19 | `measure_lung_symmetry` | 左右肺镜像相似性 | 确定性镜像 Dice | 数值 Observation | 给定或由肺联合 bbox 推得中线下的镜像 Dice、左右差异和中线来源 | V1、V2 工程验证通过，含镜像/平移/越界性质测试。衡量的是肺 mask 几何对称性，不是疾病标签；无独立医学准确性评估。 |
| A20 | `assign_lung_zones` | 肺野上/中/下区分配 | 确定性肺高度三等分和区域 mask 重叠统计 | 分区数值/空间关系 Observation | 每侧区域所属上/中/下区、交叠像素/比例、肺外范围等 | V1、V2 工程验证通过，边界和不适用情形有核对。这里是二维操作分区，不是解剖肺叶；没有单独医学准确性结论。 |
| A21 | `relate_lesion_to_anatomy` | 病灶与解剖区域关系 | 确定性二维集合/几何关系 | 结构化空间关系 Observation | 重叠、包含、交并比例、距离及图像轴相对方向；输入区域可为候选框并集 | V1、V2 工程验证通过，与像素/几何参照核对。二维关系不表示三维遮挡；关系准确性依赖独立标注及上游 mask，无独立医学评估。 |
| A22 | `relate_multiple_regions` | 多区域关系 | Shapely/像素单元拓扑和距离 | 成对关系列表 Observation | 多个命名区域之间的相交、接触、包含、分离及像素单元间隙等 | V1、V2 工程验证通过，与 A11 间隙口径一致。不能把投影接触解释成三维临床关系；无独立医学准确性评估。 |
| A23 | `summarize_bilateral_distribution` | 双侧异常分布汇总 | 连通组件分析 + 左右肺交叠聚合 | 分布结果 Observation | 候选区域在右肺/左肺/双侧/肺外的像素与组件汇总 | V1、V2 工程验证通过，含翻转和跨侧样例；需独立病灶侧别参照才能评估实际分布准确性，目前没有。 |
| A24 | `classify_view` | PA/AP/侧位分类 | CheXOne 多选问答 | 分类枚举、原始文本/选项分数 Observation | PA、AP、lateral 或 unknown/unparsed；记录 prompt 版本和未校准选项分数 | NIH 292 例 ViewPosition 核对准确率 0.976，但 NIH 与 CheXOne 训练来源重叠，不能当独立准确率；V3 严格 token 一致性未通过。 |
| A25 | `assess_image_quality` | 胸片技术质量原始量 | CXAS 结构分割 + 确定性规则 | 多个结构化 Observation 数值/状态 | 锁骨-脊柱旋转不对称量、右膈顶附近肋骨计数、心后脊柱灰度对比代理、视野边缘接触；不产生 grade | 确定性规则的 V1/V2 工程核验通过，CXAS 流程差分有核验；质量分级均为未判定，缺人工技术质量等级，未完成医学准确性评估。 |
| A26 | `localize_devices` | ETT/NGT/CVC/Swan-Ganz 设备定位 | `ianpan/cxr-lines-and-tubes` `max` profile 多模型集成 | 设备存在原始分数 + 设备线 mask + 尖端/隆突坐标 + 分类原始分数 | 四类设备的阈值筛查状态、线区域、末端坐标、carina 点、设备位置类别分数和模型来源信息；当前不覆盖胸腔引流管/起搏器 | V3 作者流程差分和 V4 部分变换测试通过。训练包含 NIH/RANZCR 等来源；无独立设备线/端点参照的医学准确性评估。 |

## 评估状态总览与使用限制

1. 27 个正式接口均纳入 V1 接口契约检查；此前报告为 162 项检查、0 失败。
2. A9–A23 等确定性算法通过手算例、独立 oracle、随机样本或性质测试，证明的是实现和口径行为，不是上游模型预测准确。
3. 模型流程核验并非全部通过：A5/A6/A8/A12/A18/A24 等存在已记录的变换一致性、严格 token、mask 差异或交叉核对未决项；细节以 [`数据集与工具评估说明.md`](数据集与工具评估说明.md) 和 [`tool_verification/README.md`](../tool_verification/README.md) 为准。
4. Montgomery、JSRT、NIH、ChestX-Det 上的部分结果存在模型训练来源重叠、非官方标签或定义差异，只能作为工程核验或描述性结果，不能写成独立医学准确率。
5. 正式医学评估仍需逐工具冻结目标口径、独立人工参照、患者级数据划分、拒答/覆盖统计和置信区间。数据集标签必须与具体指标相匹配：例如病例级疾病标签不能代替人工轮廓或测量端点。

具体样本、训练来源、容差及逐阶段报告请查阅 [`数据集与工具评估说明.md`](数据集与工具评估说明.md)、[`项目现状.md`](项目现状.md) 和 [`未来计划.md`](未来计划.md)。
