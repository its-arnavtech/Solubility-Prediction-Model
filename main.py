import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#load dataset
df = pd.read_json(r"C:\LearnML\PythonML\salaries.json")

#print(df)
y = df['salary']
x = df.drop(columns=['salary','experience_level', 'salary_currency', 'employee_residence',
                      'company_location', 'employment_type', 'job_title', 'company_size'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15, random_state=100)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr)
#print(x_train.dtypes)
#print(y_train.dtypes)
#print(x_train.apply(lambda col: col.astype(str).str.contains('SE').any()))
#print('SE' in y_train.values)
#x_train = x_train.apply(pd.to_numeric, errors='coerce').fillna(0)

y_lr_test_pred = lr.predict(x_test)
y_lr_train_pred = lr.predict(x_train)
print(y_lr_train_pred)
#model's built