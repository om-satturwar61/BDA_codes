import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from time import sleep

df = pd.read_csv('50_Startups.csv')

X = df.iloc[:,0 : 4]
Y = df['Profit']

categorical_feature = ['State']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_feature)],
                                remainder = "passthrough")
transformed_X = transformer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(transformed_X, Y, test_size = 0.25, random_state = 2509)

lr = LinearRegression()
lr.fit(X_train, Y_train)

print("The model score obtained is ", lr.score(X_test, Y_test))

Y_pred = lr.predict(X_test)


x_rd_spend = X_test[:, 0]  
x_administration = X_test[:, 1]  
y_actual = Y_test
y_predicted = Y_pred

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_rd_spend, x_administration, y_actual, color='r', label='Actual')

ax.scatter(x_rd_spend, x_administration, y_predicted, color='g', label='Predicted')

xx, yy = np.meshgrid(np.linspace(x_rd_spend.min(), x_rd_spend.max(), 10),
                     np.linspace(x_administration.min(), x_administration.max(), 10))
zz = lr.coef_[0] * xx + lr.coef_[1] * yy + lr.intercept_

ax.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(), alpha=0.5, color='b')

ax.set_xlabel('R&D Spend')
ax.set_ylabel('Administration')
ax.set_zlabel('Profit')
ax.legend()

plt.show()
sleep(100)