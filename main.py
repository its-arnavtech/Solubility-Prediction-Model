import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#load dataset
df = pd.read_json(r"C:\LearnML\PythonML\salaries.json")

print(df)