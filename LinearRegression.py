#first machine learning model. model is a variable called 'lr'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#loading dataset
df = pd.read_csv(r"https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")

#printing dataset
    #print(df)

#split to x and y
y = df['logS']
#print(y)

x = df.drop('logS', axis=1)
#print(x)

#split dataset to training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100)
#print(x_test)

#linear regression model training
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
#print(lr)

#apply model to make prediction
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
#print(y_lr_test_pred)
#print(y_lr_train_pred)
#model is built by this point

#random forest model training
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

#evaluate model performance
#print(y_train)
#print(y_lr_train_pred)
#from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
rf_results = pd.DataFrame(["Linear Regression", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ["Method", "Training MSE", "Training R2", "Testing MSE", "Testing R2"]
print(rf_results)

#print("LR MSE (Train): ", lr_train_mse)
#print("LR R2 (Train): ", lr_train_r2)
#print("LR MSE (Test): ", lr_test_mse)
#print("LR R2 (Test): ", lr_test_r2)

lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ["Method", "Training MSE", "Training R2", "Testing MSE", "Testing R2"]
#print(lr_results)

#data visualization
import matplotlib.pyplot as plt
import numpy as np #for trendline in graph

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.4)
plt.plot(y_train, p(y_train), "#E00A0A")
plt.xlabel("Actual logS")
plt.ylabel("Predicted logS")
plt.title("Training Data: Actual vs Predicted")
plt.show()