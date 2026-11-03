# MAPF-LNS Baseline Reproduction

基于 Li et al. 2021 (IJCAI) "Anytime Multi-Agent Path Finding via Large
Neighborhood Search" 的 **非自适应 MAPF-LNS 教学实现**。Agent-based 已改为
延误驱动的受限随机游走；其他算子及主循环的本地实现边界见下文，不能将整个
项目视为作者实现的逐项等价复现。

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
- ❌ 额外的自适应延误算子；Agent-based 自身会使用延误选择种子
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
更多扩展实验数据见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。
扩展实验（28 组 × 3 seeds）见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。
实验对比图（gamma 曲线、规模曲线、场景柱状图、收敛曲线）见 [figures/README.md](figures/README.md)。

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
- Agent-based：在禁忌列表外选择延误最大的智能体，执行可能缩短当前路径的
  受限随机游走，收集试探移动遇到的阻挡智能体。这里不构建当前解的冲突图。
- Neighborhood-based：保留本项目的简化版本，随机种子，按静态地图 hop
  距离由近到远扩展；尚未改为作者实现的交叉点选择法。

#### Agent-based 的执行细节

延误为当前路径步数减去忽略其他智能体时的静态最短距离。静态距离通过从目标
反向 BFS 计算，考虑地图障碍，按实例和目标缓存。种子选择使用持久禁忌列表，
同延误时优先较小 ID；禁忌列表需要重置时本轮可以返回空集合。

每次试探移动必须满足 `t + 1 + h(next) < 当前路径步数`，其中 `t` 是绝对
时间步。试探只遵守地图可通行性，允许遇到其他路径并收集其 ID；它不修改真实
路径，也不作为候选解提交。首次从种子起点试探，不足时最多再试探 10 次，
从已选智能体中随机选择后续游走对象，并从其原路径的随机时刻重新出发。

`ceil(gamma*N)` 是 Agent-based 的目标数量：不会随机补齐，因此结果可能
不足目标。收集到不足两个智能体时返回空集合，主循环跳过本轮；这也意味着
目标数量为 1 时通常不能生成有效 Agent-based 邻域。当目标数量覆盖全部
智能体时，直接返回全部 ID，包括单智能体实例。

参考作者实现的 `generateNeighborByRandomWalk`、`findMostDelayedAgent` 和
`randomWalk`：
[LNS.cpp](https://github.com/Jiaoyang-Li/MAPF-LNS/blob/master/src/LNS.cpp)。
本地适配保留 `gamma` 接口、Python seeded RNG，并将收集数量封顶为目标 k；
阻挡查询同时覆盖顶点、边交换及永久终点占用。作者 C++ 的阻挡收集函数尚将
永久终点占用列为 TODO，本项目按已有 MAPF 语义处理该情况。

主循环在空邻域时记录 `failure_reason="neighborhood_unavailable"`，在 Repair
失败时记录 `failure_reason="repair_failed"`；两者均不更新 current 或 best。
统计 Repair 失败次数时应使用此字段区分，不能把所有 `soc_candidate=None`
都算作 Repair 失败。

旧实验数字来自修改前的选择逻辑；本次未重新运行这些实验，不能将旧数字当作
新 Agent-based 实现的验证结果。当前独立回归测试见
`tests/test_agent_based_destroy.py`；已删除的旧测试和实验文件不在本次恢复范围内。

每轮从三种算子中均匀随机选择一种。候选 SOC 不高于当前解时才接受；最优解
仅在 SOC 严格降低时更新。所有随机性均由 `random.Random(seed)` 控制。

### 安装与测试

项目仅依赖 Python 标准库，要求 Python 3.10 或更高版本：

```bash
python -m pip install -e .
pip install -e ".[viz]"
python -m unittest discover -s tests -v
```

完整复现实验命令见 [REPRODUCE.md](REPRODUCE.md)。

## Moving AI Lab 对比实验

本项目实现了 CA*、LRA* 和以 CA* 解为起点的 LNS，并在 Moving AI Lab
风格的 32×32、20% 随机障碍地图上进行了横向规模对比与纵向收敛实验。

![Scalability](figures/fig1_movingai_scalability.png)

![LNS Anytime Optimization](figures/fig2_movingai_convergence.png)

随着智能体数量增加，LNS 相比 CA* 和 LRA* 的 SOC 优势逐渐明显。图 2 展示了
LNS 的 Anytime 优势：CA* 产生的初始解（SOC 688）是一条静态基线，LNS 在
50 轮后已将其降至 669，并在 100 轮时优化到 666，比 CA* 降低了约 3.2%。
