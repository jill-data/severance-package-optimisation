import pandas as pd
from pulp import LpMinimize, LpProblem, LpStatus, LpVariable, lpSum

data_raw = pd.read_csv('./data/employee_attrition_lyon_probs.csv')

data = data_raw.copy()

# region Severance package
data['eligible_for_1/4_monthly_income'] = data['YearsAtCompany'].where(data['YearsAtCompany'] < 10, 10)
data['eligible_for_1/3_monthly_income'] = data['YearsAtCompany'] - data['eligible_for_1/4_monthly_income']

# We need to use the monthly income instead of monthly rate because according to Maurice's email, the RCC
# needs to be based on the monthly salary, which is defined as 1/12 of the total amount OBTAINED over the
# past year --> Needs to be the actual income.
data['severance_package'] =\
    data['MonthlyIncome'] / 4 * data['eligible_for_1/4_monthly_income'] +\
    data['MonthlyIncome'] / 3 * data['eligible_for_1/3_monthly_income'] +\
    data['MonthlyIncome'] * 2

# endregion

# region Divide employees into groups
category_name = 'MonthlyIncome'
n_buckets = 20
data[f'{category_name}_category'] = pd.qcut(data[category_name], n_buckets)
data[f'{category_name}_category_code'] = data[f'{category_name}_category'].cat.codes
# data.groupby(f'{category_name}_category_code').size()

# Make sure there's no NA
assert not data.query(f'{category_name}_category_code < 0').shape[0]

data['severance_group'] = data[f'{category_name}_category_code']

# Make sure there's enough employee in each category
min_employees = 10
assert (data.groupby('severance_group').size() > min_employees).all()

# endregion

# region metrics for constraints

# region Constraint 1: Min number to offer RCC: 40.
# This is based on the probability so no need to do anything here
# endregion

# region Constraint 2: Yearly saving from salary is > 3million
# we calculate the yearly rate from monthly rate. We can't use monthly income because
# that's what the employee receives, but we are actually interested in what the company
# has to pay out
data['YearlyRate'] = data['MonthlyRate'] * 12
# endregion

# region Constraint 3: Leaving ppl are spread evenly across job roles.
data['JobRole'] = 'role_' + data['JobRole']
role_list = data['JobRole'].unique().tolist()
role_counts = data.groupby('JobRole').size()

# Transform to wide format
data = pd.concat(
    [
        data.drop(['JobRole'], axis=1),
        data[['JobRole', 'prob_take_rcc']].pivot(columns='JobRole', values='prob_take_rcc').fillna(0),
    ],
    axis=1,
)
# endregion

# region Constraint 4: >= 80% of the employees have to stay in each department
data['Department'] = 'dept_' + data['Department']
dept_list = data['Department'].unique().tolist()
dept_counts = data.groupby('Department').size()

# Transform to wide format
data = pd.concat(
    [
        data.drop(['Department'], axis=1),
        data[['Department', 'prob_take_rcc']].pivot(columns='Department', values='prob_take_rcc').fillna(0),
    ],
    axis=1,
)
# endregion

# endregion

# region Compute the statistics for each employee category
data['employee_count'] = 1
severance_group_stats = data.groupby('severance_group', as_index=False).agg({
    'prob_take_rcc': 'sum',  # constraint 1
    'YearlyRate': 'sum',  # constraint 2
    **{role: 'sum' for role in role_list},  # constraint 3
    **{dept: 'sum' for dept in dept_list},  # constraint 4
    'severance_package': 'sum',  # objective
    'employee_count': 'sum',  # total number of employees in each group
})

severance_group_stats.to_csv('./output/severance_groups.csv', index=False)
# endregion

# region Optimise

# region Setup
groups = severance_group_stats.copy()
model = LpProblem(name='test', sense=LpMinimize)

group_chosen = [LpVariable(name=f'severance_group{i}', cat='Binary') for i in groups['severance_group']]

severance_pays = groups['severance_package'].tolist()
total_employees_before = groups['employee_count'].sum()
# endregion

# region objective
objective = lpSum(
    [group * severance_package]
    for group, severance_package
    in zip(group_chosen, groups['severance_package'])
)

model += objective
# endregion

# region Constraint 1: Min number to offer RCC: 40.
employees_to_leave = lpSum([
    group * employee_count
    for group, employee_count
    in zip(group_chosen, groups['prob_take_rcc'])
])

employee_constraint = employees_to_leave >= 40

model += (employee_constraint, 'employee_to_offer_rcc')
# endregion

# region Constraint 2: Yearly saving from salary is >3 million
salary_constraint = lpSum([
    group * group_yearly_salary
    for group, group_yearly_salary
    in zip(group_chosen, groups['YearlyRate'])
]) >= 3_000_000

model += (salary_constraint, 'yearly_saving')
# endregion

# region Constraint 3: Leaving ppl are spread evenly across job roles.
tolerance = 0.1
roles = role_counts.index

total_employees_after = total_employees_before - employees_to_leave

for role in roles:
    # Number of employees in the current role before the RCC
    count_employee_in_role_before = role_counts[role]

    # Pct of employees from the current role before the RCC
    pct_in_role_before = count_employee_in_role_before / total_employees_before

    # Number of employees remaining in the current role after the RCC
    count_employee_in_role_after = count_employee_in_role_before - lpSum([
        group * count_employees_with_role_in_group
        for group, count_employees_with_role_in_group
        in zip(group_chosen, groups[role])
    ])

    # Target number of employees remaining in the current role after the RCC
    target_count_employee_in_role_after = pct_in_role_before * total_employees_after

    # Add constraint
    model += (
        count_employee_in_role_after >= (target_count_employee_in_role_after * (1 - tolerance)),
        f'min_remaining_{role}'
    )

    model += (
        count_employee_in_role_after <= (target_count_employee_in_role_after * (1 + tolerance)),
        f'max_remaining_{role}'
    )
# endregion

# region Constraint 4: >= 80% of the employees have to stay in each department
depts = dept_counts.index

for dept in depts:
    # Number of employees in the department before
    count_employee_in_department_before = dept_counts[dept]

    total_employee_in_dept_after = count_employee_in_department_before - lpSum([
        group * count_employees_in_department_in_group
        for group, count_employees_in_department_in_group
        in zip(group_chosen, groups[dept])
    ])

    model += (
        total_employee_in_dept_after >= (count_employee_in_department_before * 0.8),
        f'remaining_{dept}'
    )
# endregion

# show model
print(model)

# Solve
status = model.solve()

# Get status
print(LpStatus[model.status])

# region gather optimisation results
severance_group_results = []
for var in model.variables():
    severance_group_results.append({
        'severance_group': int(var.name.replace('severance_group', '')),
        'rcc_offer': var.value()
    })
severance_group_results_df = pd.DataFrame(severance_group_results)

groups_results = pd.merge(
    groups,
    severance_group_results_df,
    on='severance_group'
)

rcc_offer_groups = groups_results.query('rcc_offer == 1')
# endregion

# region analyse optimisation results
# Objective: severance pay
severance_pay = rcc_offer_groups['severance_package'].sum() / 1000000
print(f'Severance pay: €{severance_pay:.2f} million')

# Constraint 1: Number of people to offer RCC
leavings = rcc_offer_groups['prob_take_rcc'].sum()
print(f'Number of employees leaving {leavings:.2f}')

assert leavings >= 40

# Constraint 2: Yearly saving
yearly_saving = rcc_offer_groups['YearlyRate'].sum() / 1000000
print(f'Yearly saving: €{yearly_saving:.2f} million')

assert yearly_saving > 3

# Constraint 3: Leaving ppl are spread evenly across job roles
total_headcount_after = total_employees_before - leavings

role_changes = pd.concat(
    [
        role_counts / total_employees_before * 100,  # before
        (role_counts - rcc_offer_groups[roles].sum()) / total_headcount_after * 100,  # after
    ],
    axis=1
)
role_changes.columns = ['pct_before', 'pct_after']
role_changes = role_changes.eval('relative_change = (pct_after - pct_before) / pct_before * 100').reset_index()
role_changes['JobRole'] = role_changes['JobRole'].str.replace('role_', '')

assert (role_changes['relative_change'].abs() <= tolerance * 100).all()

# Constraint 4: >= 80% of the employees have to stay in each department
dept_changes = pd.concat(
    [
        dept_counts,  # before
        dept_counts - rcc_offer_groups[depts].sum(),  # after
    ],
    axis=1,
)
dept_changes.columns = ['employee_count_before', 'employee_count_after']
dept_changes = dept_changes.eval('pct_remaining = employee_count_after / employee_count_before * 100').reset_index()
dept_changes['Department'] = dept_changes['Department'].str.replace('dept_', '')

assert (dept_changes['pct_remaining'].abs() >= 0.8).all()

# endregion

# analyse protected attributes
data = pd.merge(
    data,
    severance_group_results_df,
    on='severance_group'
)


def analyse_rcc_offer_rate(data, attribute_name):
    rcc_offer_by_attribute = (
        data
        .groupby(['rcc_offer', attribute_name], as_index=False).size()
        .pivot(index=attribute_name, columns='rcc_offer', values='size')
    )
    rcc_offer_by_attribute.columns = ['rcc_offered' if x == 1 else 'no_rcc_offered' for x in rcc_offer_by_attribute.columns]
    rcc_offer_by_attribute.reset_index()
    rcc_offer_by_attribute = rcc_offer_by_attribute.eval('pct_rcc_offered = rcc_offered / (rcc_offered + no_rcc_offered) * 100')

    return rcc_offer_by_attribute


# Gender
analyse_rcc_offer_rate(data, 'Gender')

# Age
data['age_group'] = pd.qcut(data['Age'], 10)
analyse_rcc_offer_rate(data, 'age_group')

# Marital Status
analyse_rcc_offer_rate(data, 'MaritalStatus')
