# How to Reproduce

## 环境

- Python >= 3.10
- 无第三方依赖（仅标准库）

## 安装

```bash
python -m pip install -e .
```

## 单元测试

```bash
python -m unittest discover -s tests -v
# 期望：Ran 26 tests ... OK
```

## 复现 RESULTS.md 的 9 组矩阵

按 [RESULTS.md](RESULTS.md) 中“新结果表”的顺序逐条运行：

```bash
python benchmark_run.py --scenario dense      --size 8  --agents 20 --gamma 0.3 --iterations 500  --seed 0 --time-limit 300
python benchmark_run.py --scenario dense      --size 8  --agents 20 --gamma 0.5 --iterations 500  --seed 0 --time-limit 300
python benchmark_run.py --scenario dense      --size 16 --agents 80 --gamma 0.3 --iterations 1000 --seed 0 --time-limit 300
python benchmark_run.py --scenario bottleneck --size 16 --agents 12 --gamma 0.3 --iterations 1000 --seed 0 --time-limit 300
python benchmark_run.py --scenario bottleneck --size 16 --agents 12 --gamma 0.5 --iterations 1000 --seed 0 --time-limit 300
python benchmark_run.py --scenario bottleneck --size 24 --agents 20 --gamma 0.3 --iterations 1000 --seed 0 --time-limit 300
python benchmark_run.py --scenario hub        --size 16 --agents 12 --gamma 0.3 --iterations 1000 --seed 0 --time-limit 300
python benchmark_run.py --scenario hub        --size 16 --agents 12 --gamma 0.5 --iterations 1000 --seed 0 --time-limit 300
python benchmark_run.py --scenario hub        --size 24 --agents 20 --gamma 0.3 --iterations 1000 --seed 0 --time-limit 300
```

所有组的 seed 都是 0，应精确复现 RESULTS.md 中的数字（runtime 除外，受硬件
负载影响）。每条命令最多运行 300 秒。

## 已知限制

- PP 在 dense 8×8 / 20-agent、hub 16×16 / 12-agent 等场景会失败，这是 PP
  不完备性的已知现象（非本实现缺陷），与论文一致。
- 本实现未加入 PP 失败后的兜底求解器。
