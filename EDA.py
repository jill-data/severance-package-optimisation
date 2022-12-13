import pandas as pd

data = pd.read_csv('./data/employee_attrition_previous_closure.csv')
lyon = pd.read_csv('./data/employee_attrition_lyon.csv')

data.isna().sum().sum()

data.dropna()

1029 - 775

lyon.melt().groupby('variable').nunique()
