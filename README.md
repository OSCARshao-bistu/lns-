# MAPF-LNS Baseline Reproduction

严格复现 Li et al. 2021 (IJCAI) "Anytime Multi-Agent Path Finding via Large
Neighborhood Search" 的 **基线 MAPF-LNS**（非 Adaptive LNS）。

## 论文引用

```bibtex
@inproceedings{li2021mapf,
  title     = {Anytime Multi-Agent Path Finding via Large Neighborhood Search},
  author    = {Li, Jiaoyang and Chen, Zhe and Harabor, Daniel and Stuckey, Peter J. and Koenig, Sven},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence (IJCAI)},
  pages     = {4127--4135},
  year      = {2021}
}
```

## 复现了什么

- ✅ Algorithm 1 主循环（逐行对应，`lns.py` 顶部注释标注行号）
- ✅ 三种 Destroy 算子：Random、Agent-based、Neighborhood-based
- ✅ PP Repair，**随机优先顺序**（论文 Section 4）
- ✅ 纯贪婪接受准则
- ✅ 初始解由 PP 生成，PP 失败原样抛出（论文明确承认 PP 不完备）

## 不包含什么（变体声明）

- ❌ Adaptive LNS（权重自适应 / UCB / bandit）
- ❌ LNS2（处理不可行初始解）
- ❌ ML-LNS / 学习增强 Destroy
- ❌ delay-based / collision-based destroy
- ❌ 任何 CBS / ECBS / LaCAM / SAT 兜底

## 快速开始

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python benchmark_run.py --scenario bottleneck --size 24 --agents 20 --gamma 0.3
```

## 复现结果摘要

| 场景 | 可运行组数 | 改进幅度 | 备注 |
|---|---:|---:|---|
| bottleneck | 3/3 | 5.41%~9.36% | 全部达标 |
| hub 24×24 | 1/1 | 7.79% | 达标 |
| dense / hub 16×16 | 0/5 | PP_FAILED | PP 不完备性，非 LNS 问题 |

完整 9 组数据、逐场景分析与行为对齐清单见 [RESULTS.md](RESULTS.md)。

## 目录结构

```text
src/mapf_lns/            核心实现
  models.py                数据结构
  conflicts.py             冲突检测
  prioritized_planning.py  PP + 时空 A*
  destroy.py               三算子
  repair.py                随机顺序 PP repair
  lns.py                   Algorithm 1
benchmark.py               场景生成器
benchmark_run.py           实验入口
tests/                     26 个单测
RESULTS.md                 完整实验报告
REPRODUCE.md               复现步骤
LICENSE                    MIT 许可证
```

## 技术细节

### MAPF 基础语义

- 网格采用 4-connected 移动，并允许 wait。
- 冲突包括标准顶点冲突和边交换冲突。
- 智能体到达目标后永久占据目标位置。
- 目标函数为 Sum of Costs（SOC），即所有路径到达时间步之和。

### 初始解与 Repair

初始解由固定 agent ID 顺序的 Prioritized Planning 生成，每个 agent 使用时空
A*，已规划路径作为动态预约。Repair 固定邻域外路径，对邻域内 agent 使用
同一个 seeded RNG 随机打乱优先顺序，再逐个执行同一时空 A*。PP 不完备，
任何失败都会如实抛出 `PlanningFailure`。

### Destroy 与接受准则

- Random：无放回均匀采样 agent。
- Agent-based：随机种子，在当前冲突图上随机游走。
- Neighborhood-based：随机种子，按静态地图 hop 距离由近到远扩展。

每轮从三种算子中均匀随机选择一种。候选 SOC 不高于当前解时才接受；最优解
仅在 SOC 严格降低时更新。所有随机性均由 `random.Random(seed)` 控制。

### 安装与测试

项目仅依赖 Python 标准库，要求 Python 3.10 或更高版本：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

完整复现实验命令见 [REPRODUCE.md](REPRODUCE.md)。
