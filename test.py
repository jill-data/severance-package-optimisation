from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, lpSum

# Problem 1
model = LpProblem(name='test', sense=LpMaximize)

# Num of packages of each cargo type
xc1f = LpVariable(name='xc1f', lowBound=0, cat='Integer')
xc2f = LpVariable(name='xc2f', lowBound=0, cat='Integer')
xc3f = LpVariable(name='xc3f', lowBound=0, cat='Integer')
xc4f = LpVariable(name='xc4f', lowBound=0, cat='Integer')
xc1c = LpVariable(name='xc1c', lowBound=0, cat='Integer')
xc2c = LpVariable(name='xc2c', lowBound=0, cat='Integer')
xc3c = LpVariable(name='xc3c', lowBound=0, cat='Integer')
xc4c = LpVariable(name='xc4c', lowBound=0, cat='Integer')
xc1b = LpVariable(name='xc1b', lowBound=0, cat='Integer')
xc2b = LpVariable(name='xc2b', lowBound=0, cat='Integer')
xc3b = LpVariable(name='xc3b', lowBound=0, cat='Integer')
xc4b = LpVariable(name='xc4b', lowBound=0, cat='Integer')

objective = (
    2 * xc1f * 320 + 1.6 * xc2f * 400 + 2.5 * xc3f * 360 + 1.3 * xc4f * 290 +
    2 * xc1c * 320 + 1.6 * xc2c * 400 + 2.5 * xc3c * 360 + 1.3 * xc4c * 290 +
    2 * xc1b * 320 + 1.6 * xc2b * 400 + 2.5 * xc3b * 360 + 1.3 * xc4b * 290
)

weight_constraints = [
    (2 * xc1f + 1.6 * xc2f + 2.5 * xc3f + 1.3 * xc4f <= 12, 'front_weight'),
    (2 * xc1c + 1.6 * xc2c + 2.5 * xc3c + 1.3 * xc4c <= 18, 'center_weight'),
    (2 * xc1b + 1.6 * xc2b + 2.5 * xc3b + 1.3 * xc4b <= 10, 'back_weight'),
]

volume_constraints = [
    (1000 * xc1f + 1150 * xc2f + 1400 * xc3f + 780 * xc4f <= 7000, 'front_capacity'),
    (1000 * xc1c + 1150 * xc2c + 1400 * xc3c + 780 * xc4c <= 9000, 'center_capacity'),
    (1000 * xc1b + 1150 * xc2b + 1400 * xc3b + 780 * xc4b <= 5000, 'back_capacity'),
]

model += objective
for constraint in weight_constraints:
    model += constraint

for constraint in volume_constraints:
    model += constraint

model

LpStatus[model.status]

status = model.solve()

model.objective.value()

for var in model.variables():
    print(var.name, var.value())


def get_constraint_value(constraint):
    constraint_value = 0
    for variable, coefficient in constraint.items():
        constraint_value += variable.value() * coefficient
    return constraint_value


for name, constraint in model.constraints.items():
    print(f"{name}: {get_constraint_value(constraint)}")

# Problem 2
model = LpProblem(name='vc', sense=LpMaximize)

projects = [LpVariable(name=f'project{i + 1}', cat='Binary') for i in range(6)]
initial_costs = [1.3, 0.8, 0.6, 1.8, 1.2, 2.4]
profit_rates = [10, 20, 20, 10, 10, 10]
risks = [6, 4, 6, 5, 5, 4]

objective = lpSum(
    [project * intial_cost * profit_rate]
    for project, intial_cost, profit_rate
    in zip(projects, initial_costs, profit_rates)
)

cost_constraint = (
    lpSum(
        [project * intial_cost]
        for project, intial_cost
        in zip(projects, initial_costs)
    ) <= 4,
    'cost_constraint',
)

risk_constraint = (
    lpSum(
        [project * (risk - 5)]
        for project, risk
        in zip(projects, risks)
    ) <= 0,
    'risk_constraint',
)

model += objective
model += cost_constraint
model += risk_constraint

model

status = model.solve()
LpStatus[model.status]

model.objective.value()

for var in model.variables():
    print(var.name, var.value())

for name, constraint in model.constraints.items():
    print(f"{name}: {get_constraint_value(constraint)}")

# Risk constraint: Need to + 5 * number of projects, then average
num_projects = sum([var.value() for var in model.variables()])
avg_failure_risk = (get_constraint_value(model.constraints['risk_constraint']) + num_projects * 5) / num_projects
