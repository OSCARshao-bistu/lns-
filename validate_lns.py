from mapf_lns.models import MAPFInstance, Agent
from mapf_lns.lns import run_lns, LNSConfig
from mapf_lns.conflicts import find_all_conflicts

starts = [(0, 0), (7, 0), (0, 7), (7, 7), (0, 3), (7, 3), (3, 0), (3, 7)]
goals = [(7, 7), (0, 7), (7, 0), (0, 0), (7, 3), (0, 3), (3, 7), (3, 0)]
inst = MAPFInstance(8, 8, set(), [Agent(s, g) for s, g in zip(starts, goals)])

config = LNSConfig(max_iterations=300, gamma=0.3, seed=42)
best, logs = run_lns(inst, config)

# 1) 无冲突
assert find_all_conflicts(best) == [], "final best must be conflict-free"

# 2) best SOC 单调不增
socs = [l.soc_best_after for l in logs]
assert all(socs[i] >= socs[i + 1] for i in range(len(socs) - 1)), \
    "best SOC should be non-increasing"

# 3) accepted 轮的 candidate SOC 不超过轮前 current
for log in logs:
    if log.accepted:
        assert log.soc_candidate is not None
        assert log.soc_candidate <= log.soc_current_before

initial = logs[0].soc_current_before
final = best.compute_soc()
accepted = sum(l.accepted for l in logs)
print(f"initial SOC={initial}, final SOC={final}, improvement={initial - final}")
print(f"accepted {accepted}/{len(logs)} iterations")
print("PASS")