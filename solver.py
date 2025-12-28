import pulp
import math

# ---------- INPUT ----------
STOCK = 6500  # mm

pieces = {
    1200: 12,
    1280: 12
}
# ---------------------------

# expand pieces into flat list
cuts = []
for length, count in pieces.items():
    cuts.extend([length] * count)

n = len(cuts)
bars = math.ceil(sum(cuts) / STOCK) + 1

model = pulp.LpProblem("CuttingStockMM", pulp.LpMinimize)

# x[i][j] = cut i assigned to bar j
x = pulp.LpVariable.dicts("x", (range(n), range(bars)), cat="Binary")
# y[j] = bar j is used
y = pulp.LpVariable.dicts("y", range(bars), cat="Binary")

# objective: minimize number of bars
model += pulp.lpSum(y[j] for j in range(bars))

# each cut must be assigned once
for i in range(n):
    model += pulp.lpSum(x[i][j] for j in range(bars)) == 1

# bar capacity constraints
for j in range(bars):
    model += pulp.lpSum(cuts[i] * x[i][j] for i in range(n)) <= STOCK * y[j]

model.solve(pulp.PULP_CBC_CMD(msg=False))

# ---------- OUTPUT ----------
bar_plans = []

for j in range(bars):
    if y[j].value() == 1:
        assigned = [cuts[i] for i in range(n) if x[i][j].value() == 1]
        used = sum(assigned)
        bar_plans.append({
            "cuts": assigned,
            "used_mm": used,
            "waste_mm": STOCK - used
        })

print(f"Bars needed: {len(bar_plans)}")
for i, bar in enumerate(bar_plans, 1):
    print(f"Bar {i}: cuts={bar['cuts']} used={bar['used_mm']} waste={bar['waste_mm']}")
