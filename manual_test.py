from mapf_lns.models import MAPFInstance, Agent, Solution
from mapf_lns.prioritized_planning import prioritized_planning, PlanningFailure
from mapf_lns.conflicts import find_all_conflicts

print("=== 场景 A：3×3 单智能体，SOC 应等于 4 ===")
inst = MAPFInstance(3, 3, set(), [Agent((0, 0), (2, 2))])
sol = prioritized_planning(inst)
print("path:", sol[0], "SOC:", sol.compute_soc())
assert sol.compute_soc() == 4, "空地图最短路应为 4 步"

print("\n=== 场景 B：5×5 交叉，PP 应成功或抛 PlanningFailure（都是合法的）===")
inst = MAPFInstance(5, 5, set(),
                    [Agent((0, 0), (4, 4)), Agent((4, 0), (0, 4))])
try:
    sol = prioritized_planning(inst, agent_order=[0, 1])
    print("PP 成功，SOC:", sol.compute_soc())
    print("冲突数:", len(find_all_conflicts(sol)))
    assert len(find_all_conflicts(sol)) == 0, "PP 解必须无冲突"
    assert sol.compute_soc() >= 8 + 8, "SOC 不可能小于两个 agent 的曼哈顿下界和"
except PlanningFailure as e:
    print("PP 失败（合法，PP 不完备）:", e)

print("\n=== 场景 C：1×2 走廊对穿，PP 必须抛 PlanningFailure ===")
inst = MAPFInstance(1, 2, set(),
                    [Agent((0, 0), (0, 1)), Agent((0, 1), (0, 0))])
try:
    sol = prioritized_planning(inst)
    print("!!!!! 不该成功，PP 有 bug !!!!!")
except PlanningFailure:
    print("正确：无解场景抛了 PlanningFailure")

print("\n=== 场景 D：手工顶点冲突检测 ===")
sol = Solution()
sol[0] = [(0, 0), (1, 0), (2, 0)]
sol[1] = [(2, 1), (1, 0), (0, 0)]
cs = find_all_conflicts(sol)
print("conflicts:", cs)
assert any(c.type == "vertex" and c.timestep == 1 for c in cs), "应识别 t=1 的顶点冲突"

print("\n=== 场景 E：手工边冲突检测 ===")
sol = Solution()
sol[0] = [(0, 0), (1, 0)]
sol[1] = [(1, 0), (0, 0)]
cs = find_all_conflicts(sol)
print("conflicts:", cs)
assert any(c.type == "edge" and c.timestep == 1 for c in cs), "应识别 t=1 的边冲突"

print("\n=== 场景 F：SOC 计入等待 ===")
sol = Solution()
sol[0] = [(0, 0), (1, 0), (1, 0), (1, 0)]  # 到 (1,0) 后等了 2 步
print("SOC:", sol.compute_soc())
assert sol.compute_soc() == 3, "等待必须计入 SOC"

print("\n=== 场景 G：5×5 密集 4 agent，观察 PP 成功率 ===")
starts = [(0, 0), (4, 0), (0, 4), (4, 4)]
goals = [(4, 4), (0, 4), (4, 0), (0, 0)]
inst = MAPFInstance(5, 5, set(), [Agent(s, g) for s, g in zip(starts, goals)])
try:
    sol = prioritized_planning(inst)
    print("PP 成功，SOC:", sol.compute_soc(), "冲突:", len(find_all_conflicts(sol)))
except PlanningFailure as e:
    print("PP 失败（密集场景可能失败，合法）:", e)

print("\n全部断言通过。")