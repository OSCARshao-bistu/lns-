# MAPF-LNS Baseline Reproduction Results

## 实现边界

本实现严格组合 Li et al. 2021 baseline 的 PP、三种 destroy、PP repair 与纯
贪婪接受，不包含 Adaptive-LNS、LNS2 或 ML-LNS。Repair 采用 `rng.shuffle`
随机优先顺序，对应 Li et al. 2021 Section 4。

## 运行环境

- Python 3.12.10，Windows 11
- 12th Gen Intel Core i7-12700H，14 核、20 逻辑处理器
- 单个 benchmark 内为单线程顺序 LNS；独立 benchmark 曾并行启动
- 全部使用 `seed=0`，每组配置 300 秒时间上限

## 已废弃的场景

上一阶段的 empty/random/corridor 矩阵已废弃：空地图和稀疏随机障碍下，固定
顺序 repair 反复生成同成本路径；高障碍与窄走廊则在初始 PP 阶段失败。这些
场景无法区分 repair 固定点与 PP 初始解质量，因此本报告改用 dense、双缺口
bottleneck 和四象限 hub。旧数据没有混入下表。

## 新结果表

`PP_FAILED` 表示初始 PP 失败；没有重采样、重试或求解器兜底。其
`runtime(s)` 是失败前 PP 耗时，成功组则是完整 `run_lns` 耗时。

| 场景 | size | agents | gamma | iterations | initial_soc | best_soc | improvement | improvement_ratio | accept_rate | runtime(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 8 | 20 | 0.3 | 500 | PP_FAILED | — | — | — | — | 0.004 |
| dense | 8 | 20 | 0.5 | 500 | PP_FAILED | — | — | — | — | 0.004 |
| dense | 16 | 80 | 0.3 | 1000 | PP_FAILED | — | — | — | — | 4.288 |
| bottleneck | 16 | 12 | 0.3 | 1000 | 185 | 175 | 10 | 5.41% | 0.856 | 3.824 |
| bottleneck | 16 | 12 | 0.5 | 1000 | 185 | 175 | 10 | 5.41% | 0.667 | 6.019 |
| bottleneck | 24 | 20 | 0.3 | 1000 | 513 | 465 | 48 | 9.36% | 0.779 | 25.122 |
| hub | 16 | 12 | 0.3 | 1000 | PP_FAILED | — | — | — | — | 0.362 |
| hub | 16 | 12 | 0.5 | 1000 | PP_FAILED | — | — | — | — | 0.370 |
| hub | 24 | 20 | 0.3 | 1000 | 642 | 592 | 50 | 7.79% | 0.456 | 37.725 |

## 逐场景分析

### dense 8×8 / 20 agents / gamma 0.3

31.25% agent 密度下，固定 ID 顺序的初始 PP 失败，LNS 未启动。失败发生在
destroy 和随机 repair 之前，因此不能用该组评价 LNS 改进率。

### dense 8×8 / 20 agents / gamma 0.5

该组与上一组共享同一实例，故初始 PP 同样失败；gamma 尚未参与执行。把
gamma 从 0.3 改为 0.5 不可能修复初始求解器的不完备性。

### dense 16×16 / 80 agents / gamma 0.3

31.25% 密度的大实例也在初始 PP 阶段失败。耗时高于 8×8 组，但仍未进入
任何 LNS 迭代，说明随机端点的高密度空图对当前 PP 过紧。

### bottleneck 16×16 / 12 agents / gamma 0.3

SOC 从 185 降到 175，改进 5.41%。接受率 0.856，不再是修复前所有候选
等成本、全部接受的固定点行为。

### bottleneck 16×16 / 12 agents / gamma 0.5

更大邻域得到相同最优记录 175，改进 5.41%。接受率降到 0.667，说明随机
重排更多 agent 会产生更多较差候选，但贪婪准则正确拒绝了它们。

### bottleneck 24×24 / 20 agents / gamma 0.3

SOC 从 513 降到 465，改进 48（9.36%），是 bottleneck 中改进最大的一组。
接受率 0.779，表明双缺口竞争为随机优先顺序提供了有效腾挪空间。

### hub 16×16 / 12 agents / gamma 0.3

四个单格门上的双向交通令初始 PP 失败，因此没有 LNS 数据。该失败属于 PP
不完备性，不是 repair failure，也未触发实例重采样。

### hub 16×16 / 12 agents / gamma 0.5

与上一组使用同一实例，所以仍在初始 PP 阶段失败；gamma 不会被使用。不能
据此比较 0.3 和 0.5 的 repair 效果。

### hub 24×24 / 20 agents / gamma 0.3

该 hub 实例成功进入 LNS，SOC 从 642 降至 592，改进 7.79%。接受率 0.456，
显示大量随机顺序候选被拒绝，同时保留了严格改进路径。

## 修复前后对比

- 修复前（第三阶段）：所有能运行的场景 accept_rate=1.0、improvement=0，
  因为 repair 按固定 ID 顺序导致确定性固定点。
- 修复后（本阶段）：所有成功进入 LNS 的 4 组都出现显著改进，范围为
  5.41%～9.36%；accept_rate 为 0.456～0.856，证明候选已具有真实差异且
  纯贪婪接受正在发挥作用。

## 诊断

- 结论：**repair 随机化与拥挤场景成功复现了 LNS 的改进能力。**四个可运行
  场景全部达到至少 5% 改进，严格优于修复前的零改进固定点。
- 形式判据未达到 5/9：只有 4 组达到 5%，因为其余 5 组均在初始 PP 阶段
  失败，而非 improvement ratio 低。不得把这些无 LNS 数据的失败组计为改进组。
- dense 三组零改进的具体原因是随机端点加 31.25% 密度使固定顺序 PP 无法
  构造初始解；改变 gamma 对初始 PP 没有影响。
- hub 16×16 两组零改进的具体原因是较小地图中的四个单格门形成过紧的双向
  交通，初始 PP 无解；24×24 提供更大缓冲区域后，PP 成功且 LNS 改进 7.79%。

因此，本轮不存在“repair 随机化后仍未改进”的 anomaly；实际 anomaly 是
预期“大概率成功”的 dense 和小型 hub 在 `seed=0` 下暴露了 PP 初始解能力
边界。按约束保留原始失败结果，不重试、不换 seed、不换求解器。

## 行为对齐清单

- [x] 初始解由 PP 生成，PP 失败原样抛出
- [x] Destroy 三算子严格对应论文 baseline
- [x] Repair 用 PP + 随机优先顺序（对应论文 Section 4 原文）
- [x] 接受准则纯贪婪
- [x] 启发式选择均匀随机
- [x] 无自适应权重、无学习、无 bandit、无 delay destroy
- [x] 无 CBS / ECBS / LaCAM / SAT 兜底
