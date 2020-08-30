import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("Homeprices.csv")
# print(df.head())

plt.xlabel('Area(sq. ft.)')
plt.ylabel('Cost(USD)')
plt.scatter(df.area, df.price, color='red', marker='+')
# plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# Case 1: Area = 3300
arr = np.array(3300)
arr = arr.reshape(1, -1)

# print(reg.predict(arr))

# Case 2: Area = 5000

arr = np.array(5000)
arr = arr.reshape(1, -1)

# print(reg.predict(arr))

# Case 3: for a range of Home prices
d = pd.read_csv('PredictHP.csv')
print(d.head())
p = reg.predict(d)
print(p)

# Writing the predictions
d['prices'] = p
d.to_csv('PredHP.csv', index=False)
