import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# plt.rcParams['figure.figsize'] = (12.0, 9.0)

data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')
X  = data1.iloc[:, 0]
Y1 = data1.iloc[:, 1]
Y2 = data2.iloc[:, 1]

fig = plt.figure()

fig1 = fig.add_subplot(311)
fig1.set_xlabel(data1.columns[0])
fig1.set_ylabel(data1.columns[1])
model1 = LinearRegression()
model1.fit(X.values.reshape(-1, 1), Y1.values.reshape(-1, 1))
y1_fit = model1.predict(X.values.reshape(-1, 1))
plt.scatter(X, Y1)
r2_error1 = r2_score(Y1, y1_fit)
line1 = plt.plot(X, y1_fit, color='red', label="%.4f"%r2_error1)
plt.legend(handles=line1)

fig2 = fig.add_subplot(312)
fig2.set_xlabel(data2.columns[0])
fig2.set_ylabel(data2.columns[1])
model2 = LinearRegression()
model2.fit(X.values.reshape(-1, 1), Y2.values.reshape(-1, 1))
y2_fit = model2.predict(X.values.reshape(-1, 1))
plt.scatter(X, Y2)
r2_error2 = r2_score(Y2, y2_fit)
line2 = plt.plot(X, y2_fit, color='red', label="%.4f"%r2_error2)
plt.legend(handles=line2)

fig3 = fig.add_subplot(313)
fig3.set_xlabel(data1.columns[1])
fig3.set_ylabel(data2.columns[1])
model3 = LinearRegression()
model3.fit(Y1.values.reshape(-1, 1), Y2.values.reshape(-1, 1))
y3_fit = model3.predict(Y1.values.reshape(-1, 1))
plt.scatter(Y1, Y2)
r2_error3 = r2_score(Y2, y3_fit)
line3 = plt.plot(Y1, y3_fit, color='red', label="%.4f"%r2_error3)
plt.legend(handles=line3)

print('r squared error 1: %.2f' % r2_score(Y1, y1_fit))
print('r squared error 2: %.2f' % r2_score(Y2, y2_fit))
print('r squared error 3: %.2f' % r2_score(Y2, y3_fit))

# fig = plt.scatter(Y1, Y2)
# fig.set_xlabel(data1.columns[1])
# fig.set_ylabel(data2.columns[1])

plt.show()
