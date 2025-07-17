from functools import reduce

from pulp import LpMaximize, LpStatus, LpVariable, getSolver, pulp, value


def get_areas_by_sizes(lengths):
    volumn = reduce(lambda x, y: x * y, lengths)
    areas = [volumn // size for size in lengths]
    return areas


def make_ilp(n, hyperplanes, lengths, exclude_null_space_of):
    areas = get_areas_by_sizes(lengths)

    model = pulp.LpProblem("linear_programming", LpMaximize)
    solver = getSolver("PULP_CBC_CMD")

    x = [LpVariable(f"x{i}", cat="Integer") for i in range(n)]
    y = [LpVariable(f"y{i}", cat="Integer") for i in range(n)]

    u = [LpVariable(f"u{i}", cat="Integer") for i in range(n)]
    v = [LpVariable(f"v{i}", cat="Integer") for i in range(n)]

    a = [LpVariable(f"a{i}", cat="Integer") for i in range(n)]
    b = [LpVariable(f"b{i}", cat="Integer") for i in range(n)]
    c = [LpVariable(f"c{i}", cat="Integer") for i in range(n)]
    d = [LpVariable(f"d{i}", cat="Integer") for i in range(n)]

    row = len(exclude_null_space_of)
    z = [LpVariable(f"z{i}", cat="Integer") for i in range(row)]
    p = [LpVariable(f"p{i}", cat="Integer") for i in range(row)]
    q = [LpVariable(f"q{i}", cat="Integer") for i in range(row)]

    # declare objective
    model += -sum([areas[i] * y[i] for i in range(n)])

    # abs constraints
    # y = |x|
    M = max(lengths) + 1
    for i in range(n):
        model += x[i] <= y[i]
        model += -x[i] <= y[i]
        model += y[i] <= x[i] + (1 - u[i]) * M
        model += y[i] <= -x[i] + u[i] * M

    for i in range(n):
        model += u[i] >= 0
        model += u[i] <= 1

    # hyperplane constraints
    # hyperplane * x = 0
    for hyperplane in hyperplanes:
        model += sum([hyperplane[j] * x[j] for j in range(n)]) == 0

    # constraints for a[:]
    # a = 0 iff y = 0
    # a = 1 iff y > 0
    for i in range(n):
        model += y[i] <= a[i] * M
        model += a[i] <= y[i]

    for i in range(n):
        model += a[i] >= 0
        model += a[i] <= 1

    model += sum(a) >= 1

    # Constraints for b[:]
    # b[0] = a[0]
    # b[i] = b[i-1] + a[i]
    model += b[0] == a[0]
    for i in range(1, n):
        model += b[i] == b[i - 1] + a[i]

    # Constraints for c[:]
    # c = |b-1|
    M2 = n + 1
    for i in range(n):
        model += (b[i] - 1) <= c[i]
        model += -(b[i] - 1) <= c[i]
        model += c[i] <= (b[i] - 1) + (1 - v[i]) * M2
        model += c[i] <= -(b[i] - 1) + v[i] * M2

        model += v[i] >= 0
        model += v[i] <= 1

    # Constraints for d[:]
    # M2 = n + 1
    for i in range(n):
        model += c[i] <= (1 - d[i]) * M2
        model += (1 - d[i]) <= c[i]

    # # Constraints between d and x
    for i in range(n):
        model += x[i] >= (d[i] - 1) * M

    # Constraint for linear independent of previous solutions
    row = len(exclude_null_space_of)
    if row > 0:
        col = len(exclude_null_space_of[0])
        for r in range(row):
            model += sum(exclude_null_space_of[r][c] * x[c] for c in range(col)) == z[r]
            mdoel += z[r] <= (1 - p[r]) * M - 1
            mdoel += -z[r] <= (1 - q[r]) * M - 1

        for i in range(row):
            model += p[row] >= 0
            model += p[row] <= 1
            model += q[row] >= 0
            model += q[row] <= 1

        model += sum(p) + sum(q) >= 1

    return model, solver, {"x": x, "y": y, "a": a, "b": b, "c": c, "d": d}


n = 2
hyperplanes = [[1, 1]]
lengths = [8, 3]
model, solver, decision_vars = make_ilp(n, hyperplanes, lengths)
print(model)

# solve
results = model.solve(solver=solver)

# print results
if LpStatus[results] == "Optimal":
    print("The solution is optimal.")
    print(f"Objective value: z* = {value(model.objective)}")
    print(f"Solution: ")
    x = decision_vars["x"]
    y = decision_vars["y"]
    a = decision_vars["a"]
    b = decision_vars["b"]
    c = decision_vars["c"]
    d = decision_vars["d"]
    print(f"  x = {[value(xi) for xi in x]}")
    print(f"  y = {[value(yi) for yi in y]}")
    print(f"  a = {[value(ai) for ai in a]}")
    print(f"  b = {[value(bi) for bi in b]}")
    print(f"  c = {[value(ci) for ci in c]}")
    print(f"  d = {[value(di) for di in d]}")
elif LpStatus[results] == "Infeasible":
    print("Problem is infeasible - no solution exists")
else:
    print(f"Solver status: {LpStatus[results]}")
